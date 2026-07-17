#!/usr/bin/env python3
"""Generate config files and Slurm scripts for the V4-ablation sweep.

7 ablation variants × 3 seeds = 21 jobs. Reuses each v3 variant's existing
hyperparameters (from the May-8 ablation runs) and only overrides:
  - seed
  - data.val_min_year / data.val_max_year = 2015 / 2015 (V4 narrow-adjacent)
  - data.train_cutoff_year = 2010 (made explicit)
  - train.output_dir → /gpfs/.../runs/v4_ablation/{slug}_s{seed}
  - train.early_stopping.metric = ndcg_ta_mean@30, mode = max
  - top-level seed

V3 hyperparams are reused (not re-tuned) per the May-2026 decision: the
hyperparameter→val-window interaction is expected to be small, and re-tuning
under V4 would cost ~100h of compute for marginal expected change. One
deviation from v3: batch_size is forced to 256 (vs v3's 512/1024) so the runs
fit on the 16 GB V100s in the gpushort partition without OOM.

Selection metric is val_ndcg_ta_mean@30 (max), chosen via regret analysis
on the val-window sweep — it is the most stable selection metric across
seeds while retaining the highest test-rr signal.
"""
from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT  = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "config" / "experiments" / "v4_ablation"
SCRIPT_DIR = REPO_ROOT / "scripts" / "advancement_prediction" / "v4_ablation"
V3_CONFIG_DIR = REPO_ROOT / "config" / "experiments"
RUNS_BASE = "/gpfs/scratch/bty414/opentarget_evidences/26.03/runs/v4_ablation"
# 26.03 w3 graph generation (23.06 retired for the same-year trial-edge leak).
# The v3 base configs still carry 23.06 paths; the override below overwrites them.
W3_GRAPH = "/gpfs/scratch/bty414/opentarget_evidences/26.03/graph/hetero_graph_with_features_datatype_w3.pt"
W3_MAP   = "/gpfs/scratch/bty414/opentarget_evidences/26.03/progression/temporal_graph_datatype_w3_mappings.pt"

VARIANTS = [
    # (slug, source v3 config stem, gres)
    # HGT-based variants (P1, P2, P3, B1) OOM on V100-16GB at batch_size 256;
    # pin them to A100-80GB. GNN baselines fit on any GPU.
    ("p1_eahgt_score",   "p1_eahgt_score",   "nvidia_a100_80gb_pcie:1"),
    ("p2_eahgt_novelty", "p2_eahgt_novelty", "nvidia_a100_80gb_pcie:1"),
    ("p3_eahgt_both",    "p3_eahgt_both",    "nvidia_a100_80gb_pcie:1"),
    ("b1_hgt",           "b1_hgt",           "nvidia_a100_80gb_pcie:1"),
    ("b3_gatv2",         "b3_gatv2",         "1"),
    ("b6_rgcn",          "b6_rgcn",          "1"),
    ("b7_compgcn",       "b7_compgcn",       "1"),
]
SEEDS = [0, 1, 7]

# V4 ablation overrides applied on top of each v3 base config.
def make_override(slug: str, seed: int) -> dict:
    return {
        "seed": seed,
        "experiment": {"name": f"{slug}_v4_s{seed}"},
        "data": {
            "graph_file": W3_GRAPH,
            "mappings_file": W3_MAP,
            "train_cutoff_year": 2010,
            "val_min_year": 2015,
            "val_max_year": 2015,
        },
        "train": {
            "output_dir": f"{RUNS_BASE}/{slug}_s{seed}",
            "batch_size": 256,  # override v3 to fit on 16GB V100s in gpushort
            "num_epochs": 30,   # match V4 sweep; trainer saves best ckpt by ES metric
            "early_stopping": {
                # Disabled to match the V4 sweep methodology (full trajectory,
                # post-hoc best-by-ndcg_ta_mean@30 selection). When `enabled:
                # false` the trainer still tracks `metric` for best-checkpoint
                # saving, it just never stops early — so the saved ckpt is
                # argmax val/ndcg_ta_mean@30 across all 30 epochs.
                "enabled": False,
                "patience": 5,
                "metric": "ndcg_ta_mean@30",
                "mode": "max",
            },
        },
    }

SLURM_TEMPLATE = """\
#!/bin/bash
#SBATCH -J v4_{slug}_s{seed}
#SBATCH -o %x.o%j
#SBATCH -p gpushort
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 1:0:0
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:{gres}

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
REPO_ROOT="${{SLURM_SUBMIT_DIR:-$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")}}"
cd "$REPO_ROOT"

source .venv/bin/activate
export WANDB_MODE="disabled"

python src/train_advancement_lambdarank.py \\
  --config config/experiments/v4_ablation/{slug}_s{seed}.yaml
"""

ORCHESTRATOR_TEMPLATE = """\
#!/bin/bash
# Submit the V4-ablation sweep: 7 model variants (P1-P3 + B1, B3, B6, B7) × 3 seeds = 21 jobs.
# Each run uses the V4 narrow-adjacent val window (val_min=val_max=2015) and selects the
# best epoch by val_ndcg_ta_mean@30 (max) — see val-window-sweep analysis for rationale.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$REPO_ROOT"

SWEEP_DIR="$SCRIPT_DIR/v4_ablation"

JOBS=({jobs_quoted})

for job in "${{JOBS[@]}}"; do
    echo "Submitting $job"
    sbatch "$SWEEP_DIR/$job"
done

echo "All ${{#JOBS[@]}} jobs submitted. Use 'squeue -u $USER' to monitor."
"""


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    job_filenames: list[str] = []

    for slug, source_stem, gres in VARIANTS:
        base_cfg = OmegaConf.load(V3_CONFIG_DIR / f"{source_stem}.yaml")
        # Drop the legacy `tune:` section from p3 (search_space, not used by training).
        if "tune" in base_cfg:
            del base_cfg["tune"]

        for seed in SEEDS:
            override = OmegaConf.create(make_override(slug, seed))
            merged = OmegaConf.merge(base_cfg, override)
            # Stamp a header by writing through OmegaConf so types are preserved.
            cfg_path = CONFIG_DIR / f"{slug}_s{seed}.yaml"
            header = (
                f"# Auto-generated by v4_ablation/_generate.py — do not edit by hand.\n"
                f"# V4 val window (val_min=val_max=2015); selection by val_ndcg_ta_mean@30 (max).\n"
                f"# Source v3 hyperparams: config/experiments/{source_stem}.yaml.\n"
            )
            body = OmegaConf.to_yaml(merged, sort_keys=False)
            cfg_path.write_text(header + body)

            slurm_text = SLURM_TEMPLATE.format(slug=slug, seed=seed, gres=gres)
            slurm_path = SCRIPT_DIR / f"run_{slug}_s{seed}.sh"
            slurm_path.write_text(slurm_text)
            slurm_path.chmod(0o755)
            job_filenames.append(slurm_path.name)

    jobs_quoted = " ".join(f'"{j}"' for j in job_filenames)
    orch_path = REPO_ROOT / "scripts" / "advancement_prediction" / "run_v4_ablation_sweep.sh"
    orch_path.write_text(ORCHESTRATOR_TEMPLATE.format(jobs_quoted=jobs_quoted))
    orch_path.chmod(0o755)

    print(f"Wrote {len(VARIANTS) * len(SEEDS)} configs to {CONFIG_DIR}")
    print(f"Wrote {len(VARIANTS) * len(SEEDS)} Slurm scripts to {SCRIPT_DIR}")
    print(f"Wrote orchestrator to {orch_path}")


if __name__ == "__main__":
    main()
