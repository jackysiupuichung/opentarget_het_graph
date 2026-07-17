#!/usr/bin/env python3
"""Generate the 35 w3-retrain configs (7 models x 5 seeds) for the 26.03 w3
pipeline-label graph. Clones the existing ablation/encoder configs verbatim,
changing ONLY: graph_file, mappings_file, output_dir, experiment name.

Models:
  encoder baselines (use_edge_features:false): hgt, gatv2, rgcn, compgcn
  edge-feature ablation (HGT):
     score   -> edge_feat_cols [0]
     novelty -> edge_feat_cols [1]
     both    -> edge_feat_cols [0,1]   (headline; new 5-seed matched-recipe variant)

All keep the current recipe: random_val_frac 0.2, allrank_grouped, ndcg_k 100,
group_all_tas, 40 epochs, early_stopping disabled. Only paths change.
"""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CFG = REPO / "config" / "experiments"
OUT = CFG / "w3_retrain"
SEEDS = [1, 7, 42, 123, 2024]

W3_GRAPH = "/gpfs/scratch/bty414/opentarget_evidences/26.03/graph/hetero_graph_with_features_datatype_w3.pt"
W3_MAP   = "/gpfs/scratch/bty414/opentarget_evidences/26.03/progression/temporal_graph_datatype_w3_mappings.pt"
OUT_ROOT = "/gpfs/scratch/bty414/opentarget_evidences/26.03/runs/w3_retrain"

# (family, subdir, source-template-dir, name-prefix)
ENCODERS = [("hgt","hgt"), ("gatv2","gatv2"), ("rgcn","rgcn"), ("compgcn","compgcn")]


def repath(text, name, out_dir):
    text = re.sub(r"graph_file: .*", f"graph_file: {W3_GRAPH}", text)
    text = re.sub(r"mappings_file: .*", f"mappings_file: {W3_MAP}", text)
    text = re.sub(r"(experiment:\n\s*name: ).*", rf"\g<1>{name}", text)
    text = re.sub(r"output_dir: .*", f"output_dir: {out_dir}", text)
    return text


def emit(src_path, name, out_dir, dst_path, edge_cols=None):
    text = src_path.read_text()
    text = repath(text, name, out_dir)
    if edge_cols is not None:
        # set edge_feat_cols + dim for the ablation variants (both/score/novelty)
        cols_str = "[" + ", ".join(str(c) for c in edge_cols) + "]"
        text = re.sub(r"edge_feat_cols: .*", f"edge_feat_cols: {cols_str}", text)
        text = re.sub(r"edge_feat_dim: .*", f"edge_feat_dim: {len(edge_cols)}", text)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text(text)
    return dst_path


n = 0
# --- Encoder baselines: clone each encoder's own seed configs ---
for fam, sub in ENCODERS:
    for s in SEEDS:
        src = CFG / "encoder_baselines" / sub / f"enc_{fam}_s{s}.yaml"
        name = f"enc_{fam}_w3_s{s}"
        outd = f"{OUT_ROOT}/encoder_baselines/{fam}/{name}"
        dst = OUT / "encoder_baselines" / fam / f"{name}.yaml"
        emit(src, name, outd, dst); n += 1

# --- Edge-feature ablation: score / novelty / both, all from the ablation template ---
ABL = {"score": [0], "novelty": [1], "both": [0, 1]}
for variant, cols in ABL.items():
    for s in SEEDS:
        # score/novelty have their own configs; 'both' clones the score config
        src_variant = variant if variant in ("score", "novelty") else "score"
        src = CFG / "ablation_matched" / src_variant / f"abl_{src_variant}_s{s}.yaml"
        name = f"abl_{variant}_w3_s{s}"
        outd = f"{OUT_ROOT}/ablation_matched/{variant}/{name}"
        dst = OUT / "ablation_matched" / variant / f"{name}.yaml"
        emit(src, name, outd, dst, edge_cols=cols); n += 1

print(f"Generated {n} w3 configs under {OUT}")
