#!/usr/bin/env python3
"""Post-hoc explainability for a trained HGT advancement predictor.

For a configurable subset of test (target, disease) pairs:
  1. Samples a temporally-constrained subgraph with LinkNeighborLoader.
  2. Computes IntegratedGradients attribution to edge features (score, novelty)
     against a zero-feature baseline.
  3. Captures per-edge post-softmax attention from every HGTConv layer.

Outputs:
  - per_pair_edges.parquet      edge-level IG attributions (signed)
  - per_pair_nodes.parquet      node-level IG attributions (signed + |·|),
                                summed over feature dims per node
  - case_studies/<t>_<d>.png    subgraph plot for each explained pair

Note: no population-level rollup is written. Naive mean-|IG| across pairs is
biased by per-pair logit magnitude and by sample-size differences across
edge types; build any global summary deliberately from per_pair_edges.parquet.

Usage:
    uv run python explain_advancement.py \\
        --config runs/<exp>/config.yaml \\
        --checkpoint runs/<exp>/best_model.pt \\
        --out-dir runs/<exp>/explanations \\
        --num-pairs 50 --strategy top_k --n-steps 32
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from scipy.special import expit
from torch_geometric.loader import LinkNeighborLoader

ROOT = Path(__file__).resolve().parents[2]  # repo root (explain/cli/ -> repo)
sys.path.insert(0, str(ROOT))

from src.data.temporal_loader import (
    ADV_ETYPE,
    build_context_graph,
    build_edge_time_dict as _build_edge_time_dict,
    load_event_graph,
    split_advancement_edges,
)
from src.models.utils import build_model

from src.explain.attention_extractor import capture_attention, read_attention
from src.explain.captum_edge_explainer import integrated_gradients_for_pair
from src.explain.aggregate import per_pair_edges_df, per_pair_nodes_df
from src.explain.viz import plot_pair_subgraph


_FEAT_COL_NAME = {0: "ig_score", 1: "ig_novelty"}


class _RawEdgeIndex:
    """Lazy index over parsed-edge parquets, grouping rows by
    ``(source_type, relation, target_type, sourceId, targetId)``.

    For an edge in the explainer subgraph we want to recover the underlying
    evidence row(s): either the OT evidence ``id`` (hash) or the inline
    ``literature`` PMID list, depending on which the parquet carries. The
    index groups by the natural lookup key once per file (one-time cost on
    first hit) and then answers ``get()`` calls in O(1).
    """

    def __init__(self, raw_edges_dir: str):
        self.raw_edges_dir = raw_edges_dir
        self._files_by_triple: dict[tuple, list[str]] = {}
        self._loaded: dict[str, dict] = {}
        self._index_files()

    def _index_files(self) -> None:
        """Scan files and group by (source_type, relation, target_type).

        Uses pyarrow's row-group iterator to read just one row per file
        rather than loading the whole parquet (europepmc is 9M rows; a
        naive ``read_parquet().head(1)`` OOMs on a login node).
        """
        from glob import glob
        import pyarrow.parquet as pq
        for path in sorted(glob(os.path.join(self.raw_edges_dir, "*.parquet"))):
            try:
                pf = pq.ParquetFile(path)
                batch = next(pf.iter_batches(
                    batch_size=1,
                    columns=["source_type", "relation", "target_type"],
                ))
                head = batch.to_pandas()
            except (StopIteration, Exception):
                continue
            if head.empty:
                continue
            triple = (
                str(head["source_type"].iloc[0]),
                str(head["relation"].iloc[0]),
                str(head["target_type"].iloc[0]),
            )
            self._files_by_triple.setdefault(triple, []).append(path)

    def _ensure_loaded(self, path: str) -> dict:
        """Load one parquet, sort+index by (sourceId, targetId), return cache.

        Indexing is O(n log n) once; subsequent .loc lookups are O(log n)
        without duplicating row data. For europepmc's 9M rows this is a
        few-second one-time cost and ~1.5 GB resident — vs. OOM if we
        materialise per-group DataFrames.
        """
        if path in self._loaded:
            return self._loaded[path]
        import pyarrow.parquet as pq
        cols_meta = set(pq.ParquetFile(path).schema_arrow.names)
        evidence_col = "literature" if "literature" in cols_meta else (
            "id" if "id" in cols_meta else None
        )
        load_cols = ["sourceId", "targetId", "datasourceId", "year"]
        if "score" in cols_meta:
            load_cols.append("score")     # OT per-evidence score, for presentation
        if evidence_col:
            load_cols.append(evidence_col)
        df = pd.read_parquet(path, columns=load_cols)
        # Stable string dtype on the join keys so .loc lookups don't surprise.
        df["sourceId"] = df["sourceId"].astype(str)
        df["targetId"] = df["targetId"].astype(str)
        df = df.sort_values(["sourceId", "targetId"]).set_index(
            ["sourceId", "targetId"], drop=False
        )
        cache = {
            "evidence_col": evidence_col,
            "df": df,
            "datasourceId": (
                str(df["datasourceId"].iloc[0]) if len(df) else None
            ),
        }
        self._loaded[path] = cache
        return cache

    def get(
        self,
        edge_type: tuple,
        source_id: str,
        target_id: str,
        year_max: int | None,
    ) -> pd.DataFrame:
        """Return the parsed-edge rows for one edge in the subgraph.

        ``edge_type`` is ``(source_type, relation, target_type)`` — same as
        the explainer's HeteroData keys. Multiple datasource files can share
        a triple (e.g. crispr + reactome both produce
        ``target_affected_pathway_disease``); we scan all of them.
        """
        files = self._files_by_triple.get(edge_type, [])
        if not files:
            return pd.DataFrame()
        chunks = []
        for path in files:
            cache = self._ensure_loaded(path)
            df = cache["df"]
            key = (str(source_id), str(target_id))
            try:
                hit = df.loc[[key]]
            except KeyError:
                continue
            if year_max is not None and "year" in hit.columns:
                hit = hit[hit["year"] <= year_max]
                if hit.empty:
                    continue
            cols = ["sourceId", "targetId", "datasourceId", "year"]
            if "score" in hit.columns:
                cols.append("score")
            if cache["evidence_col"]:
                cols.append(cache["evidence_col"])
            chunks.append(hit[cols].reset_index(drop=True).copy())
        if not chunks:
            return pd.DataFrame()
        return pd.concat(chunks, ignore_index=True)


def _lookup_subgraph_evidence(
    raw_index: _RawEdgeIndex,
    global_edge_index_dict,
    id_maps,
    target_id: str,
    disease_id: str,
    year_max: int | None,
) -> pd.DataFrame:
    """For one explained pair, return one row per (edge, evidence row).

    Each output row carries either ``literature`` (PMID list, for intact /
    gene_ontology) or ``id`` (OT 23.06 evidence hash, for everything else).
    Downstream tooling resolves either to canonical PMIDs / abstracts.
    """
    rows = []
    for et, ei in global_edge_index_dict.items():
        src_type, _rel, dst_type = et
        if src_type not in id_maps or dst_type not in id_maps:
            continue
        ei_np = ei.cpu().numpy()
        src_map = id_maps[src_type]
        dst_map = id_maps[dst_type]
        for i in range(ei_np.shape[1]):
            src_acc = src_map.get(int(ei_np[0, i]))
            dst_acc = dst_map.get(int(ei_np[1, i]))
            if src_acc is None or dst_acc is None:
                continue
            hits = raw_index.get(et, src_acc, dst_acc, year_max=year_max)
            if hits.empty:
                continue
            hits = hits.copy()
            hits["target_id"] = target_id
            hits["disease_id"] = disease_id
            hits["edge_type"] = "::".join(et)
            hits["src"] = int(ei_np[0, i])
            hits["dst"] = int(ei_np[1, i])
            # Normalise: one of these may be missing depending on datasource.
            if "literature" not in hits.columns:
                hits["literature"] = None
            if "id" not in hits.columns:
                hits["id"] = None
            rows.append(hits)
    if not rows:
        return pd.DataFrame(columns=[
            "target_id", "disease_id", "edge_type", "src", "dst",
            "sourceId", "targetId", "datasourceId", "year", "score",
            "literature", "id",
        ])
    return pd.concat(rows, ignore_index=True)


# Mapping from heterogeneous node type → (parquet filename in the nodes dir,
# name column). target/reactome carry a `name`. GO names live in a separate
# go_ontology_terms parquet (go.parquet itself is id-only in the 26.03 build).
# disease and molecule are resolved separately below (their name sources differ).
_NAME_PARQUET = {
    "target":   ("targets.parquet",           "name"),
    "reactome": ("reactome.parquet",          "name"),
    "go":       ("go_ontology_terms.parquet", "name"),
}


def _clean_name(text):
    """Strip + drop NaN/empty. Names are kept verbatim — no truncation."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None
    s = str(text).strip()
    return s or None


def _load_name_maps(graph_file: str, id_maps):
    """Return ``{node_type: {accession: display_name}}``. Sources (26.03 layout,
    with 23.06 ``kg_output/nodes`` as a fallback):
      - target / reactome: ``evidences/nodes/<file>.parquet`` 'name'
      - go: ``evidences/nodes/go_ontology_terms.parquet`` 'name'
      - molecule: ``evidenceDated/molecule/*.parquet`` 'name' (drug pref name)
      - disease: ``evidenceDated/diseases/*.parquet`` 'name'
    Missing files / unmatched accessions just don't add entries.
    """
    graph_dir = Path(graph_file).parent              # .../26.03/graph
    base_dir = graph_dir.parent                       # .../26.03
    # Prefer the 26.03 evidences/nodes layout; fall back to the legacy 23.06
    # kg_output/nodes path so older graphs still resolve.
    nodes_dir = base_dir / "evidences" / "nodes"
    if not nodes_dir.exists():
        nodes_dir = base_dir / "kg_output" / "nodes"
    name_maps: dict = {}

    for nt, (fname, col) in _NAME_PARQUET.items():
        path = nodes_dir / fname
        if not path.exists():
            print(f"[explain] note: no name parquet for {nt} at {path}")
            continue
        df = pd.read_parquet(path, columns=["id", col])
        df[col] = df[col].apply(_clean_name)
        name_maps[nt] = {k: v for k, v in zip(df["id"].astype(str), df[col]) if v}

    # Molecule (drug) names: evidenceDated/molecule '*.parquet' 'name'. Self-named
    # CHEMBL ids (name == id) are dropped so they don't masquerade as resolved.
    mol_dir = base_dir / "evidenceDated" / "molecule"
    if mol_dir.exists():
        try:
            df_mol = pd.read_parquet(mol_dir, columns=["id", "name"])
            df_mol["name"] = df_mol["name"].apply(_clean_name)
            name_maps["molecule"] = {
                k: v for k, v in zip(df_mol["id"].astype(str), df_mol["name"])
                if v and v != k
            }
        except Exception as e:
            print(f"[explain] warn: could not read {mol_dir}: {e}")

    # Disease names: prefer evidenceDated/diseases.name, fall back to the
    # nodes-dir diseases.parquet description.
    disease_map: dict = {}
    ev_dir = base_dir / "evidenceDated" / "diseases"
    if ev_dir.exists():
        try:
            df_ev = pd.read_parquet(ev_dir, columns=["id", "name"])
            df_ev["name"] = df_ev["name"].apply(_clean_name)
            disease_map.update(
                {k: v for k, v in zip(df_ev["id"].astype(str), df_ev["name"]) if v}
            )
        except Exception as e:
            print(f"[explain] warn: could not read {ev_dir}: {e}")
    fb_path = nodes_dir / "diseases.parquet"
    if fb_path.exists():
        _fb_cols = pd.read_parquet(fb_path).columns
        _fb_col = "description" if "description" in _fb_cols else (
            "name" if "name" in _fb_cols else None)
        if _fb_col:
            df_fb = pd.read_parquet(fb_path, columns=["id", _fb_col])
            df_fb[_fb_col] = df_fb[_fb_col].apply(_clean_name)
            for k, v in zip(df_fb["id"].astype(str), df_fb[_fb_col]):
                if v and k not in disease_map:
                    disease_map[k] = v
    if disease_map:
        name_maps["disease"] = disease_map

    return name_maps


def _select_pairs(
    model,
    context,
    edge_index,
    edge_labels,
    edge_times,
    num_neighbors,
    batch_size,
    device,
    edge_feat_cols,
    strategy: str,
    num_pairs: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return indices (into the test edge tensor) of pairs to explain."""
    n_test = edge_index.size(1)
    if num_pairs >= n_test:
        return np.arange(n_test)

    if strategy == "random":
        return rng.choice(n_test, size=num_pairs, replace=False)

    if strategy in ("top_k", "top_positives"):
        # Score every test edge, take top-num_pairs by predicted prob.
        model.eval()
        loader = LinkNeighborLoader(
            data=context,
            num_neighbors=num_neighbors,
            edge_label_index=(ADV_ETYPE, edge_index),
            edge_label=edge_labels,
            edge_label_time=edge_times,
            time_attr="edge_time",
            temporal_strategy="last",
            batch_size=batch_size,
            shuffle=False,
        )
        all_logits = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                edge_time_dict = _build_edge_time_dict(batch, ADV_ETYPE)
                out = model(
                    batch.x_dict, batch.edge_index_dict,
                    batch[ADV_ETYPE].edge_label_index,
                    src_type="target", dst_type="disease",
                    edge_time_dict=edge_time_dict,
                    edge_feat_dict={
                        et: batch[et].edge_attr[:, edge_feat_cols]
                        for et in batch.edge_types
                        if et != ADV_ETYPE
                        and hasattr(batch[et], "edge_attr")
                        and batch[et].edge_attr is not None
                    },
                    edge_label_time=getattr(batch[ADV_ETYPE], "edge_label_time", None),
                )
                all_logits.append(out.cpu())
        scores = torch.cat(all_logits).numpy()
        order = np.argsort(scores)[::-1]
        if strategy == "top_positives":
            labels_np = (edge_labels.cpu().numpy() > 0)
            order = order[labels_np[order]]
        return order[:num_pairs]

    raise ValueError(f"Unknown strategy: {strategy}")


def main(args):
    seed = int(args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    cfg = OmegaConf.load(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    case_dir = out_dir / "case_studies"
    case_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[explain] device={device}", flush=True)

    print(f"[explain] loading graph: {cfg.data.graph_file}", flush=True)
    data = load_event_graph(cfg.data.graph_file)
    mappings = torch.load(cfg.data.mappings_file, weights_only=False)
    # Build {node_type: {idx: external_id}} for *every* node type so the
    # case-study plots can label go/molecule/reactome nodes too, not just
    # target/disease.
    id_maps = {
        nt: {v: k for k, v in nm.items()}
        for nt, nm in mappings["node_mapping"].items()
    }
    inv_target = id_maps["target"]
    inv_disease = id_maps["disease"]

    # Human-readable names from the kg_output node parquets. Falls back to
    # the accession when a node has no name (or the parquet is missing).
    name_maps = _load_name_maps(cfg.data.graph_file, id_maps)

    # Optional evidence lookup: index the parsed edge parquets (under
    # evidences/edges/) by (source_type, relation, target_type). For each
    # explained subgraph edge we surface either the inline `literature`
    # column (intact, gene_ontology) or the OT evidence `id` hash (every
    # other datasource), deferring PMID resolution to downstream tooling.
    raw_index = None
    if args.raw_edges_dir:
        if os.path.isdir(args.raw_edges_dir):
            raw_index = _RawEdgeIndex(args.raw_edges_dir)
            print(f"[explain] indexed raw edges: {args.raw_edges_dir} "
                  f"({len(raw_index._files_by_triple)} edge-type triples)",
                  flush=True)
        else:
            print(f"[explain] warn: --raw-edges-dir not a directory: "
                  f"{args.raw_edges_dir}; skipping evidence lookup",
                  flush=True)

    train_mask, val_mask, test_mask, _ = split_advancement_edges(data)
    edge_index = data[ADV_ETYPE].edge_index
    edge_attr = data[ADV_ETYPE].edge_attr
    edge_time = data[ADV_ETYPE].edge_time

    context = build_context_graph(data)

    print(f"[explain] building model: {cfg.model.name}", flush=True)
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
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"[explain] loaded checkpoint: {args.checkpoint}", flush=True)

    edge_feat_cols = list(cfg.model.get("edge_feat_cols", [0, 1]))
    feat_col_names = [_FEAT_COL_NAME.get(c, f"ig_feat{c}") for c in edge_feat_cols]
    num_neighbors = list(cfg.train.num_neighbors)

    # Test set edges.
    test_edge_index = edge_index[:, test_mask]
    test_edge_labels = edge_attr[test_mask, 0]
    test_edge_times = edge_time[test_mask]

    if args.pairs_csv:
        print(f"[explain] loading pair list from {args.pairs_csv}", flush=True)
        wanted = pd.read_csv(args.pairs_csv)[["target_id", "disease_id"]]
        target_to_idx = mappings["node_mapping"]["target"]
        disease_to_idx = mappings["node_mapping"]["disease"]
        test_src = test_edge_index[0].cpu().numpy()
        test_dst = test_edge_index[1].cpu().numpy()
        test_pair_to_pos = {(int(s), int(d)): i for i, (s, d) in enumerate(zip(test_src, test_dst))}
        selected = []
        for _, row in wanted.iterrows():
            t_idx = target_to_idx.get(row.target_id)
            d_idx = disease_to_idx.get(row.disease_id)
            if t_idx is None or d_idx is None:
                print(f"[explain]   skip (unknown id): {row.target_id} / {row.disease_id}", flush=True)
                continue
            pos = test_pair_to_pos.get((int(t_idx), int(d_idx)))
            if pos is None:
                print(f"[explain]   skip (not in test split): {row.target_id} / {row.disease_id}", flush=True)
                continue
            selected.append(pos)
        pair_idx = np.array(selected, dtype=np.int64)
        print(f"[explain] selected {len(pair_idx)} / {len(wanted)} pairs from csv", flush=True)
    else:
        print(f"[explain] selecting {args.num_pairs} pairs by strategy={args.strategy}",
              flush=True)
        pair_idx = _select_pairs(
            model, context, test_edge_index, test_edge_labels, test_edge_times,
            num_neighbors, cfg.train.batch_size, device, edge_feat_cols,
            args.strategy, args.num_pairs, rng,
        )
        print(f"[explain] selected {len(pair_idx)} pairs", flush=True)

    # Per-pair loader (batch_size=1).
    selected_ei = test_edge_index[:, pair_idx]
    selected_lbl = test_edge_labels[pair_idx]
    selected_t = test_edge_times[pair_idx]

    per_pair_loader = LinkNeighborLoader(
        data=context,
        num_neighbors=num_neighbors,
        edge_label_index=(ADV_ETYPE, selected_ei),
        edge_label=selected_lbl,
        edge_label_time=selected_t,
        time_attr="edge_time",
        temporal_strategy="last",
        batch_size=1,
        shuffle=False,
    )

    all_rows = []
    all_node_rows = []
    all_ev_rows = []
    for i, batch in enumerate(per_pair_loader):
        batch = batch.to(device)
        edge_time_dict = _build_edge_time_dict(batch, ADV_ETYPE)

        # 1) Attention pass.
        with capture_attention(model) as convs:
            with torch.no_grad():
                _ = model(
                    batch.x_dict, batch.edge_index_dict,
                    batch[ADV_ETYPE].edge_label_index,
                    src_type="target", dst_type="disease",
                    edge_time_dict=edge_time_dict,
                    edge_feat_dict={
                        et: batch[et].edge_attr[:, edge_feat_cols]
                        for et in batch.edge_types
                        if et != ADV_ETYPE
                        and hasattr(batch[et], "edge_attr")
                        and batch[et].edge_attr is not None
                    },
                    edge_label_time=getattr(batch[ADV_ETYPE], "edge_label_time", None),
                )
            attention = read_attention(convs)

        # 2) IG pass.
        ig = integrated_gradients_for_pair(
            model=model,
            batch=batch,
            edge_feat_cols=edge_feat_cols,
            edge_time_dict=edge_time_dict,
            n_steps=int(args.n_steps),
        )

        # Map the queried edge_label_index[:, 0] back to entity IDs.
        # batch[ADV_ETYPE].edge_label_index lives in the batch's *local* node-id
        # space. Use the batch's n_id maps to recover global indices.
        local_src = int(batch[ADV_ETYPE].edge_label_index[0, 0].item())
        local_dst = int(batch[ADV_ETYPE].edge_label_index[1, 0].item())
        global_src = int(batch["target"].n_id[local_src].item())
        global_dst = int(batch["disease"].n_id[local_dst].item())
        target_id = inv_target.get(global_src, f"target#{global_src}")
        disease_id = inv_disease.get(global_dst, f"disease#{global_dst}")

        # Re-map edge_index entries from local to global (so node ids in the
        # parquet are stable across pairs).
        n_id_dict = {nt: batch[nt].n_id.cpu().numpy() for nt in batch.node_types}
        global_edge_index_dict = {}
        for et, ei in ig.edge_index_dict.items():
            src_type, _rel, dst_type = et
            ei_np = ei.cpu().numpy()
            ei_global = np.stack([
                n_id_dict[src_type][ei_np[0]],
                n_id_dict[dst_type][ei_np[1]],
            ], axis=0)
            global_edge_index_dict[et] = torch.from_numpy(ei_global)

        rows_df = per_pair_edges_df(
            target_id=target_id,
            disease_id=disease_id,
            edge_index_dict=global_edge_index_dict,
            edge_feat_attr=ig.edge_feat_attr,
            attention=attention,
            logit=ig.logit,
            feat_col_names=feat_col_names,
        )
        all_rows.append(rows_df)

        node_rows_df = per_pair_nodes_df(
            target_id=target_id,
            disease_id=disease_id,
            node_feat_attr=ig.node_feat_attr,
            n_id_dict=n_id_dict,
            logit=ig.logit,
        )
        all_node_rows.append(node_rows_df)

        if raw_index is not None:
            # Cutoff year: this pair's edge_label_time (the temporal frontier
            # at which we'd be predicting). Caps returned evidence to rows
            # available at prediction time — no temporal leakage.
            label_time = batch[ADV_ETYPE].edge_label_time
            year_max = int(label_time[0].item()) if label_time is not None else None
            ev_rows_df = _lookup_subgraph_evidence(
                raw_index=raw_index,
                global_edge_index_dict=global_edge_index_dict,
                id_maps=id_maps,
                target_id=target_id,
                disease_id=disease_id,
                year_max=year_max,
            )
            if not ev_rows_df.empty:
                all_ev_rows.append(ev_rows_df)

        if i < int(args.case_studies):
            plot_pair_subgraph(
                rows_df, target_id, disease_id,
                case_dir / f"{target_id}__{disease_id}.png",
                top_k=int(args.case_top_k),
                id_maps=id_maps,
                target_idx=global_src,
                disease_idx=global_dst,
                name_maps=name_maps,
                nodes_df=node_rows_df,
            )

        if (i + 1) % max(1, len(pair_idx) // 10) == 0:
            print(f"[explain]   {i+1}/{len(pair_idx)} pairs done", flush=True)

    long_df = pd.concat(all_rows, ignore_index=True)
    long_path = out_dir / "per_pair_edges.parquet"
    long_df.to_parquet(long_path, index=False)
    print(f"[explain] wrote {long_path}  ({len(long_df):,} rows)", flush=True)

    node_long_df = pd.concat(all_node_rows, ignore_index=True)
    node_path = out_dir / "per_pair_nodes.parquet"
    node_long_df.to_parquet(node_path, index=False)
    print(f"[explain] wrote {node_path}  ({len(node_long_df):,} rows)", flush=True)

    if raw_index is not None:
        ev_path = out_dir / "per_pair_evidence.parquet"
        if all_ev_rows:
            ev_df = pd.concat(all_ev_rows, ignore_index=True)
        else:
            ev_df = pd.DataFrame(columns=[
                "target_id", "disease_id", "edge_type", "src", "dst",
                "sourceId", "targetId", "datasourceId", "year",
                "literature", "id",
            ])
        # `literature` may be a list per row; store as-is. pyarrow handles
        # both list and scalar columns; mixed-dtype rows are coerced to
        # object when concatenated.
        ev_df.to_parquet(ev_path, index=False)
        n_with_lit = ev_df["literature"].notna().sum() if not ev_df.empty else 0
        n_with_id = ev_df["id"].notna().sum() if not ev_df.empty else 0
        print(f"[explain] wrote {ev_path}  ({len(ev_df):,} rows; "
              f"{n_with_lit:,} with inline PMIDs, {n_with_id:,} with OT id)",
              flush=True)
    # Note: no population-level relation_importance rollup is emitted.
    # Mean-|IG| across pairs is biased by per-pair logit magnitude and by
    # sample-size differences across edge types (see aggregate.py docstring).
    # Build a rollup deliberately downstream from this parquet if needed
    # (e.g. attribution-share normalised within pair, or IGC vs logit).
    print(f"[explain] done.", flush=True)


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to experiment config yaml.")
    p.add_argument("--checkpoint", required=True, help="Path to best_model.pt.")
    p.add_argument("--out-dir", required=True, help="Output directory.")
    p.add_argument("--num-pairs", type=int, default=50,
                   help="Number of (target, disease) pairs to explain.")
    p.add_argument("--strategy", choices=["top_k", "top_positives", "random"],
                   default="top_k",
                   help="How to pick pairs: top_k (highest model score), "
                        "top_positives (highest score among true positives), "
                        "random.")
    p.add_argument("--n-steps", type=int, default=32,
                   help="IntegratedGradients integration steps.")
    p.add_argument("--case-studies", type=int, default=10,
                   help="How many of the explained pairs also get a "
                        "case-study subgraph plot.")
    p.add_argument("--case-top-k", type=int, default=20,
                   help="Top-N most-attributed *nodes* to show in each "
                        "case-study plot. The plot then includes every "
                        "attributed edge between any two selected nodes.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pairs-csv", default=None,
                   help="CSV with columns target_id,disease_id. If set, "
                        "overrides --strategy/--num-pairs and explains "
                        "exactly these test pairs (order preserved).")
    p.add_argument("--raw-edges-dir", default=None,
                   help="Directory of parsed edge parquets "
                        "(e.g. .../23.06/evidences/edges/). When set, the "
                        "explainer writes per_pair_evidence.parquet linking "
                        "each subgraph edge to its OT evidence row(s) — "
                        "either the inline `literature` PMID list "
                        "(intact, gene_ontology) or the OT evidence `id` "
                        "hash (every other datasource).")
    return p.parse_args()


if __name__ == "__main__":
    main(_parse())
