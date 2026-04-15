"""LambdaRank loss (NDCG-weighted pairwise logistic)."""

from typing import Optional

import torch


def lambdarank_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sigma: float = 1.0,
    k: Optional[int] = None,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Batch-level LambdaRank loss (Burges, 2010).

    Treats the full batch as a single ranking list. For every pair (i, j)
    with l_i > l_j, contributes
        |ΔNDCG_ij| · log(1 + exp(-σ(s_i - s_j)))
    where ΔNDCG_ij is the change in NDCG if positions of i and j were swapped
    under the ranking induced by the current scores.

    Args:
        logits: [B] predicted scores.
        labels: [B] relevance labels (>= 0). Binary labels (0/1) are fine.
        sigma:  logistic slope (Burges σ).
        k:      NDCG truncation for the ΔNDCG weighting. None = full list.
        eps:    numerical floor for IDCG.

    Returns:
        Scalar loss. Returns 0 * logits.sum() if the batch has no valid pairs
        (e.g. all labels identical), so autograd still has a path.
    """
    logits = logits.view(-1)
    labels = labels.view(-1).to(logits.dtype)

    if logits.numel() < 2:
        return 0.0 * logits.sum()

    pos_mask = labels > 0
    if not pos_mask.any() or pos_mask.all():
        return 0.0 * logits.sum()

    n = logits.shape[0]
    device = logits.device

    # Ranks under current model: rank 1 = highest score.
    order = torch.argsort(logits, descending=True)
    ranks = torch.empty(n, device=device, dtype=torch.long)
    ranks[order] = torch.arange(1, n + 1, device=device)

    # Gains g_i = 2^l_i - 1; discounts d_i = 1 / log2(rank_i + 1).
    gains = torch.pow(2.0, labels) - 1.0
    discounts = 1.0 / torch.log2(ranks.to(logits.dtype) + 1.0)

    if k is not None and k < n:
        discounts = torch.where(
            ranks <= k,
            discounts,
            torch.zeros_like(discounts),
        )

    # IDCG@k for normalisation.
    ideal_labels, _ = torch.sort(labels, descending=True)
    ideal_gains = torch.pow(2.0, ideal_labels) - 1.0
    ideal_ranks = torch.arange(1, n + 1, device=device, dtype=logits.dtype)
    ideal_discounts = 1.0 / torch.log2(ideal_ranks + 1.0)
    if k is not None and k < n:
        ideal_discounts = ideal_discounts.clone()
        ideal_discounts[k:] = 0.0
    idcg = (ideal_gains * ideal_discounts).sum().clamp_min(eps)

    # |ΔNDCG_ij| = |g_i - g_j| · |d_i - d_j| / IDCG.
    gain_diff = gains.unsqueeze(1) - gains.unsqueeze(0)      # [n, n]
    disc_diff = discounts.unsqueeze(1) - discounts.unsqueeze(0)
    delta_ndcg = (gain_diff * disc_diff).abs() / idcg

    # Pairs with l_i > l_j.
    label_diff = labels.unsqueeze(1) - labels.unsqueeze(0)   # [n, n]
    pair_mask = label_diff > 0

    if not pair_mask.any():
        return 0.0 * logits.sum()

    score_diff = logits.unsqueeze(1) - logits.unsqueeze(0)   # s_i - s_j
    # log(1 + exp(-σ Δs)) — numerically stable via softplus.
    pairwise = torch.nn.functional.softplus(-sigma * score_diff)

    weighted = delta_ndcg * pairwise * pair_mask.to(logits.dtype)
    # Normalise by the number of pairs that actually carry weight (nonzero
    # ΔNDCG), so the loss scale stays comparable across batch sizes and
    # ndcg_k settings. Falls back to 1 if every ΔNDCG is zero.
    effective_pairs = (delta_ndcg > 0).to(logits.dtype).sum().clamp_min(1.0)
    return weighted.sum() / effective_pairs
