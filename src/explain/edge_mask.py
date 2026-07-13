"""Differentiable per-edge mask for PaGE-Link-style explanation.

PaGE-Link stage 1 learns a soft mask m_e in [0,1] over the subgraph edges by
optimising the masked subgraph's ability to reproduce the model's link logit.

Mechanism (no fork of HGTConv): we monkey-patch each conv's ``message`` (the
same hook attention_extractor.py uses) to multiply a per-edge mask into the
post-softmax message output ``out = v_j * alpha * m_e``. Because HGTConv
aggregates messages by sum, scaling an edge's message by m_e in [0,1]
interpolates between "edge present" (1) and "edge removed" (0) while keeping
gradients flowing to the mask. The model weights stay frozen; only the mask
logits require grad.

The mask is a single flat [E_total] tensor aligned to the concatenated
bipartite edge order (edge_index_dict.keys() order), identical to how
read_attention slices alpha back per type.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Dict, List, Tuple

import torch

EdgeType = Tuple[str, str, str]


class EdgeMask(torch.nn.Module):
    """Learnable per-edge-type mask logits over one pair's subgraph.

    ``values()`` returns {edge_type: sigmoid(logit)} in [0,1]; ``flat()`` returns
    the concatenated mask in ``edge_order`` for injection into message passing.
    Only these logits require grad — the explained model is frozen.
    """

    def __init__(self, edge_counts: Dict[EdgeType, int], edge_order: List[EdgeType],
                 init: float = 1.0, device=None):
        super().__init__()
        self.edge_order = list(edge_order)
        # init>0 starts the mask near "all edges on" (sigmoid(1)=0.73), the
        # standard GNNExplainer warm start.
        self.logits = torch.nn.ParameterDict({
            "__".join(et): torch.nn.Parameter(
                torch.full((edge_counts[et],), float(init), device=device))
            for et in self.edge_order
        })

    def values(self) -> Dict[EdgeType, torch.Tensor]:
        return {et: torch.sigmoid(self.logits["__".join(et)]) for et in self.edge_order}

    def flat(self) -> torch.Tensor:
        """Concatenated sigmoid mask over ``edge_order`` -> [E_total]."""
        return torch.cat([torch.sigmoid(self.logits["__".join(et)])
                          for et in self.edge_order], dim=0)

    def regularisation(self, size_coeff: float, entropy_coeff: float) -> torch.Tensor:
        """GNNExplainer penalty: size (mean m_e, drives sparsity) + entropy
        (drives m_e to 0/1 so the mask is decisive)."""
        m = self.flat()
        size = m.mean()
        eps = 1e-6
        ent = -(m * torch.log(m + eps) + (1 - m) * torch.log(1 - m + eps)).mean()
        return size_coeff * size + entropy_coeff * ent


def _patch_conv_with_mask(conv, mask_flat_fn):
    """Wrap ``conv.message`` so the message output is scaled by the per-edge mask.

    ``mask_flat_fn`` is a 0-arg callable returning the current [E_total] mask
    (re-read each call so the same mask object tracks optimisation steps). Edge
    order matches the bipartite concatenation, so a flat mask aligns directly.
    Returns (original_message, original_forward) for restoration.
    """
    from torch_geometric.utils import softmax as _softmax

    original_message = conv.message
    original_forward = conv.forward

    def patched_message(k_j, q_i, v_j, edge_attr, index, ptr,
                        temporal_features, ef_scalar, size_i):
        if temporal_features is not None:
            k_j = k_j + temporal_features
            v_j = v_j + temporal_features
        alpha = (q_i * k_j).sum(dim=-1) * edge_attr
        if ef_scalar is not None:
            alpha = alpha * ef_scalar
        alpha = alpha / math.sqrt(q_i.size(-1))
        alpha = _softmax(alpha, index, ptr, size_i)
        out = v_j * alpha.view(-1, conv.heads, 1)              # [E, heads, d]
        m = mask_flat_fn().to(out.device)                     # [E_total]
        out = out * m.view(-1, 1, 1)                          # scale each edge's message
        return out.view(-1, conv.out_channels)

    conv.message = patched_message
    conv.forward = original_forward  # unchanged; kept for symmetric restore
    return original_message, original_forward


@contextmanager
def apply_edge_mask(model, mask: EdgeMask):
    """Patch every HGTConv in ``model`` to scale messages by ``mask.flat()``.

    Inside the context, forward passes are differentiable w.r.t. the mask logits
    (model params should already be frozen by the caller). Restores on exit.
    """
    convs, originals = [], []
    for m in model.modules():
        if m.__class__.__name__ == "HGTConv":
            orig = _patch_conv_with_mask(m, mask.flat)
            convs.append(m)
            originals.append(orig)
    try:
        yield convs
    finally:
        for m, (orig_msg, orig_fwd) in zip(convs, originals):
            m.message = orig_msg
            m.forward = orig_fwd
