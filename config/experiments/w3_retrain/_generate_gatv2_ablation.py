#!/usr/bin/env python3
"""Generate the GATv2 edge-feature ablation configs (score/novelty/both x 5 seeds)
on the 26.03 w3 graph. Clones each gatv2 ENCODER-BASELINE config (keeps its tuned
hyperparams: hidden_dim 64, heads 4, layers 1, its lr/wd) but turns ON edge
features with the ablation's edge_feat_cols. Mirrors the HGT ablation exactly,
only the encoder differs.

  gatv2_score   -> edge_feat_cols [0],    edge_feat_dim 1
  gatv2_novelty -> edge_feat_cols [1],    edge_feat_dim 1
  gatv2_both    -> edge_feat_cols [0,1],  edge_feat_dim 2
"""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CFG = REPO / "config" / "experiments"
OUT = CFG / "w3_retrain" / "gatv2_ablation"
SEEDS = [1, 7, 42, 123, 2024]
OUT_ROOT = "/gpfs/scratch/bty414/opentarget_evidences/26.03/runs/w3_retrain/gatv2_ablation"
ABL = {"score": [0], "novelty": [1], "both": [0, 1]}

n = 0
for variant, cols in ABL.items():
    for s in SEEDS:
        src = CFG / "w3_retrain" / "encoder_baselines" / "gatv2" / f"enc_gatv2_w3_s{s}.yaml"
        text = src.read_text()
        name = f"gatv2_{variant}_w3_s{s}"
        outd = f"{OUT_ROOT}/{variant}/{name}"
        cols_str = "[" + ", ".join(str(c) for c in cols) + "]"
        text = re.sub(r"(experiment:\n\s*name: ).*", rf"\g<1>{name}", text)
        text = re.sub(r"output_dir: .*", f"output_dir: {outd}", text)
        text = re.sub(r"use_edge_features: .*", "use_edge_features: true", text)
        text = re.sub(r"edge_feat_cols: .*", f"edge_feat_cols: {cols_str}", text)
        text = re.sub(r"edge_feat_dim: .*", f"edge_feat_dim: {len(cols)}", text)
        dst = OUT / variant / f"{name}.yaml"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text)
        n += 1

print(f"Generated {n} gatv2-ablation configs under {OUT}")
