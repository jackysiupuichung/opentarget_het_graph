"""Shared runtime for the advancement explainers.

Factors the model/graph/subgraph plumbing out of ``explain_advancement.py`` so
the fidelity harness (evaluate_explanation_fidelity.py) and the PaGE-Link
explainer (pagelink_explain.py) build the SAME model, on the SAME context graph,
with the SAME temporally-constrained ``LinkNeighborLoader`` and the SAME forward
call — rather than each re-deriving it (and risking drift).

Nothing here is explainer-specific: it loads a trained checkpoint and yields
per-pair subgraphs plus a single ``predict_logit`` that mirrors exactly how
``explain_advancement.py`` invokes ``model.forward``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch_geometric.loader import LinkNeighborLoader

from src.data.temporal_loader import (
    ADV_ETYPE,
    build_context_graph,
    build_edge_time_dict,
    load_event_graph,
    split_advancement_edges,
)
from src.models.utils import build_model

EdgeType = Tuple[str, str, str]


def build_edge_feat_dict(batch, edge_feat_cols: List[int]) -> Dict[EdgeType, torch.Tensor]:
    """The exact edge-feature dict ``explain_advancement.py`` passes to forward:
    the selected feature columns for every non-advancement edge type that carries
    edge_attr. Centralised so every caller (predict, IG, mask) is consistent."""
    return {
        et: batch[et].edge_attr[:, edge_feat_cols]
        for et in batch.edge_types
        if et != ADV_ETYPE
        and hasattr(batch[et], "edge_attr")
        and batch[et].edge_attr is not None
    }


@dataclass
class ExplainRuntime:
    """Loaded model + context graph + everything needed to explain pairs.

    Use :meth:`from_config` to construct; then :meth:`pair_loader` for subgraphs
    and :meth:`predict_logit` for the (differentiable) query-edge logit.
    """

    cfg: object
    model: torch.nn.Module
    context: object
    device: torch.device
    edge_feat_cols: List[int]
    num_neighbors: List[int]
    mappings: dict
    id_maps: Dict[str, Dict[int, str]]
    # test split tensors (advancement edges)
    test_edge_index: torch.Tensor
    test_edge_labels: torch.Tensor
    test_edge_times: torch.Tensor
    latest_edge_only: bool = False
    # {node_type: {accession -> human name}} from the KG node parquets
    name_maps: Dict[str, Dict[str, str]] = field(default_factory=dict)
    _node_idx: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # ---- construction -----------------------------------------------------
    @classmethod
    def from_config(cls, config_path: str, checkpoint_path: str,
                    device: Optional[torch.device] = None) -> "ExplainRuntime":
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg = OmegaConf.load(config_path)

        data = load_event_graph(cfg.data.graph_file)
        mappings = torch.load(cfg.data.mappings_file, weights_only=False)
        id_maps = {
            nt: {int(v): k for k, v in nm.items()}
            for nt, nm in mappings["node_mapping"].items()
        }
        name_maps = cls._load_name_maps(cfg.data.graph_file)

        _, _, test_mask, _ = split_advancement_edges(data)
        edge_index = data[ADV_ETYPE].edge_index
        edge_attr = data[ADV_ETYPE].edge_attr
        edge_time = data[ADV_ETYPE].edge_time
        context = build_context_graph(data)

        model = build_model(
            model_name=cfg.model.name,
            data=context,
            hidden_dim=cfg.model.hidden_dim,
            out_dim=cfg.model.hidden_dim,
            num_heads=cfg.model.num_heads,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            use_rte=cfg.model.get("use_rte", False),
            use_edge_features=cfg.model.get("use_edge_features", False),
            edge_feat_dim=cfg.model.get("edge_feat_dim", 2),
            use_recency=cfg.model.get("use_recency", False),
            time_dim=cfg.model.get("time_dim", 0),
            latest_edge_only=cfg.model.get("latest_edge_only", False),
        ).to(device)
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
        model.eval()

        latest_edge_only = bool(cfg.model.get("latest_edge_only", False))
        # When the model collapses to the latest edge per (src,dst) inside
        # forward, the explainer must reason over the SAME post-collapse edge
        # set, or masks/attributions (built on the pre-collapse subgraph) will
        # not align with the model's messages. We collapse the batch ourselves
        # (see collapse_batch) and disable the model's internal collapse so it
        # does not run twice / on a different edge set.
        if latest_edge_only:
            model.encoder.latest_edge_only = False

        # STRICT_BEFORE=1 shifts the label time down by one year so the loader's
        # `edge_time <= edge_label_time` becomes `edge_time < transition_year`
        # (the strict future-link-prediction window). SIMULATION ONLY on a model
        # trained under <= — it previews how the explanation subgraph/paths change
        # when same-year edges are masked out, NOT a retrained model.
        import os as _os
        _shift = 1 if _os.environ.get("STRICT_BEFORE") else 0
        test_times = edge_time[test_mask] - _shift

        return cls(
            cfg=cfg, model=model, context=context, device=device,
            edge_feat_cols=list(cfg.model.get("edge_feat_cols", [0, 1])),
            num_neighbors=list(cfg.train.num_neighbors),
            mappings=mappings, id_maps=id_maps, name_maps=name_maps,
            test_edge_index=edge_index[:, test_mask],
            test_edge_labels=edge_attr[test_mask, 0],
            test_edge_times=test_times,
            latest_edge_only=latest_edge_only,
        )

    # ---- node names -------------------------------------------------------
    @staticmethod
    def _load_name_maps(graph_file: str) -> Dict[str, Dict[str, str]]:
        """{node_type: {accession -> human name}} from the KG node parquets.

        Mirrors explain_advancement._load_name_maps: target/reactome/go from
        evidences/nodes, disease from evidenceDated/diseases (fallback to
        nodes/diseases description), molecule from evidenceDated/molecule.
        Best-effort — missing parquets are skipped, names fall back to accession.
        """
        base = Path(graph_file).resolve().parent.parent
        nodes = base / "evidences" / "nodes"
        out: Dict[str, Dict[str, str]] = {}

        def _clean(x):
            return "" if x is None else str(x).strip()

        def _read(p, idcol, namecol):
            df = pd.read_parquet(p, columns=[idcol, namecol])
            return {str(k): _clean(v) for k, v in zip(df[idcol], df[namecol])
                    if _clean(v)}

        for nt, fname in (("target", "targets.parquet"),
                          ("reactome", "reactome.parquet"),
                          ("go", "go_ontology_terms.parquet")):
            p = nodes / fname
            if p.exists():
                try:
                    out[nt] = _read(p, "id", "name")
                except Exception:
                    pass
        # disease
        dmap: Dict[str, str] = {}
        ed = base / "evidenceDated" / "diseases"
        if ed.exists():
            try:
                dmap.update(_read(ed, "id", "name"))
            except Exception:
                pass
        fb = nodes / "diseases.parquet"
        if fb.exists():
            try:
                cols = pd.read_parquet(fb).columns
                col = "description" if "description" in cols else (
                    "name" if "name" in cols else None)
                if col:
                    for k, v in _read(fb, "id", col).items():
                        dmap.setdefault(k, v)
            except Exception:
                pass
        if dmap:
            out["disease"] = dmap
        mp = base / "evidenceDated" / "molecule"
        if mp.exists():
            try:
                out["molecule"] = _read(mp, "id", "name")
            except Exception:
                pass
        return out

    def node_label(self, ntype: str, idx: int) -> str:
        """'<accession> (<name>)' for a global node index, or a #idx fallback."""
        acc = self.id_maps.get(ntype, {}).get(int(idx))
        if acc is None:
            return f"{ntype}#{idx}"
        name = self.name_maps.get(ntype, {}).get(str(acc))
        return f"{acc} ({name})" if name else str(acc)

    # ---- pair selection ---------------------------------------------------
    def select_pairs_from_csv(self, pairs_csv: str) -> np.ndarray:
        """Positions (into the test edge tensor) for the (target_id, disease_id)
        rows of ``pairs_csv`` that exist in the test split. Mirrors the csv path
        in explain_advancement.main."""
        wanted = pd.read_csv(pairs_csv)[["target_id", "disease_id"]]
        t2i = self.mappings["node_mapping"]["target"]
        d2i = self.mappings["node_mapping"]["disease"]
        src = self.test_edge_index[0].cpu().numpy()
        dst = self.test_edge_index[1].cpu().numpy()
        pos_of = {(int(s), int(d)): i for i, (s, d) in enumerate(zip(src, dst))}
        out = []
        for _, r in wanted.iterrows():
            ti, di = t2i.get(r.target_id), d2i.get(r.disease_id)
            if ti is None or di is None:
                continue
            p = pos_of.get((int(ti), int(di)))
            if p is not None:
                out.append(p)
        return np.array(out, dtype=np.int64)

    # ---- latest-edge collapse (mask alignment) ----------------------------
    def collapse_batch(self, batch):
        """If the model uses ``latest_edge_only``, collapse this batch's edges
        to the latest edge per (src,dst) per type IN PLACE, so the explainer's
        edge universe matches the model's post-collapse messages. Idempotent
        and a no-op when the model does not collapse. Must be called on every
        batch before attribution / masking / prediction.
        """
        if not self.latest_edge_only:
            return batch
        from src.models.hgt import _keep_latest_edge_per_pair
        ei = {et: batch[et].edge_index for et in batch.edge_types
              if hasattr(batch[et], "edge_index")}
        et_time = {et: batch[et].edge_time for et in ei
                   if hasattr(batch[et], "edge_time") and batch[et].edge_time is not None}
        et_feat = {et: batch[et].edge_attr for et in ei
                   if hasattr(batch[et], "edge_attr") and batch[et].edge_attr is not None}
        new_ei, new_time, new_feat = _keep_latest_edge_per_pair(ei, et_time, et_feat)
        for et in ei:
            # Recompute the keep columns for this type by matching the collapsed
            # index back; _keep_latest_edge_per_pair already returns the reduced
            # tensors, so assign them straight onto the store.
            batch[et].edge_index = new_ei[et]
            if et in new_time and new_time[et] is not None:
                batch[et].edge_time = new_time[et]
            if et in new_feat and new_feat[et] is not None:
                batch[et].edge_attr = new_feat[et]
        return batch

    # ---- subgraph loader --------------------------------------------------
    def pair_loader(self, pair_idx: np.ndarray) -> LinkNeighborLoader:
        """A batch_size=1 temporally-constrained subgraph loader over the given
        test positions — identical settings to explain_advancement.py."""
        return LinkNeighborLoader(
            data=self.context,
            num_neighbors=self.num_neighbors,
            edge_label_index=(ADV_ETYPE, self.test_edge_index[:, pair_idx]),
            edge_label=self.test_edge_labels[pair_idx],
            edge_label_time=self.test_edge_times[pair_idx],
            time_attr="edge_time",
            temporal_strategy="last",
            batch_size=1,
            shuffle=False,
        )

    # ---- forward ----------------------------------------------------------
    def predict_logit(
        self,
        batch,
        edge_index_dict: Optional[Dict[EdgeType, torch.Tensor]] = None,
        edge_feat_dict: Optional[Dict[EdgeType, torch.Tensor]] = None,
        edge_time_dict: Optional[Dict[EdgeType, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Query-edge advancement logit for ``batch``.

        Defaults reproduce explain_advancement.py exactly. Callers that perturb
        the graph (fidelity masking) or inject a mask (PaGE-Link) pass their own
        ``edge_index_dict`` / ``edge_feat_dict`` while keeping everything else
        identical. NOT wrapped in no_grad so PaGE-Link can backprop to a mask;
        wrap the call site in torch.no_grad() for pure inference.
        """
        if edge_index_dict is None:
            edge_index_dict = batch.edge_index_dict
        if edge_time_dict is None:
            edge_time_dict = build_edge_time_dict(batch, ADV_ETYPE)
        if edge_feat_dict is None:
            edge_feat_dict = build_edge_feat_dict(batch, self.edge_feat_cols)
        return self.model(
            batch.x_dict, edge_index_dict,
            batch[ADV_ETYPE].edge_label_index,
            src_type="target", dst_type="disease",
            edge_time_dict=edge_time_dict,
            edge_feat_dict=edge_feat_dict,
            edge_label_time=getattr(batch[ADV_ETYPE], "edge_label_time", None),
        )

    # ---- id helpers -------------------------------------------------------
    def pair_ids(self, batch) -> Tuple[str, str]:
        """(target_id, disease_id) accessions for the query edge of ``batch``."""
        ls = int(batch[ADV_ETYPE].edge_label_index[0, 0].item())
        ld = int(batch[ADV_ETYPE].edge_label_index[1, 0].item())
        gs = int(batch["target"].n_id[ls].item())
        gd = int(batch["disease"].n_id[ld].item())
        return (self.id_maps["target"].get(gs, f"target#{gs}"),
                self.id_maps["disease"].get(gd, f"disease#{gd}"))
