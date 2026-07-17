#!/usr/bin/env python3
"""EAHGT-only sweep with `lambdarank.impl: allrank_grouped`.

Same protocol as headline/ (val_year=2013, ES patience=10, rs_ta_mean@50)
but with per-TA slated lambdarank loss, to test whether the per-TA
distribution flattens (less disease-collapse).

3 seeds × 1 architecture = 3 runs. Outputs go to
/gpfs/scratch/.../runs/headline_grouped/p3_eahgt_both_s<seed>/ so the
flat-allrank baselines in runs/headline/ stay untouched.
"""
from pathlib import Path
import re

REPO = Path(__file__).resolve().parents[3]
CFG_DIR = Path(__file__).resolve().parent
V4_DIR = REPO / "config" / "experiments" / "v4_ablation"
SCRIPTS_DIR = REPO / "scripts" / "advancement_prediction" / "headline_grouped"
SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

ARCH = "p3_eahgt_both"
SRC = "p3_eahgt_both_s7.yaml"
SEEDS = [1, 7, 42]
OUTPUT_ROOT = "/gpfs/scratch/bty414/opentarget_evidences/26.03/runs/headline_grouped"
# 26.03 w3 graph generation (23.06 retired for the same-year trial-edge leak).
# The v4 base config still carries 23.06 paths, so overwrite them here.
W3_GRAPH = "/gpfs/scratch/bty414/opentarget_evidences/26.03/graph/hetero_graph_with_features_datatype_w3.pt"
W3_MAP   = "/gpfs/scratch/bty414/opentarget_evidences/26.03/progression/temporal_graph_datatype_w3_mappings.pt"


def patch(text: str, seed: int) -> str:
    text = re.sub(r"graph_file: .*", f"graph_file: {W3_GRAPH}", text)
    text = re.sub(r"mappings_file: .*", f"mappings_file: {W3_MAP}", text)
    text = re.sub(r"experiment:\n  name: .*",
                  f"experiment:\n  name: {ARCH}_grouped_s{seed}", text)
    text = re.sub(r"val_min_year: \d+", "val_min_year: 2013", text)
    text = re.sub(r"val_max_year: \d+", "val_max_year: 2013", text)
    text = re.sub(r"output_dir: .*",
                  f"output_dir: {OUTPUT_ROOT}/{ARCH}_s{seed}", text)
    text = re.sub(r"num_epochs: \d+", "num_epochs: 40", text)
    text = re.sub(r"^seed: \d+", f"seed: {seed}", text, flags=re.MULTILINE)
    # ES block: patience 10 on rs_ta_mean@50
    text = re.sub(
        r"  early_stopping:\n(    \S.*\n)+",
        ("  early_stopping:\n"
         "    enabled: true\n"
         "    patience: 10\n"
         "    metric: rs_ta_mean@50\n"
         "    mode: max\n"),
        text,
    )
    # Switch lambdarank.impl from allrank → allrank_grouped. The grouped
    # impl builds one slate per primary TA per batch (items replicated
    # across multi-TA diseases) so the listwise loss optimises ranking
    # WITHIN each TA, removing the global "dominate one hot disease"
    # shortcut that flat allrank rewards.
    text = re.sub(r"impl: allrank\b(?!_)", "impl: allrank_grouped", text)
    return text


SBATCH_TMPL = """\
#!/bin/bash
#SBATCH -J g_{arch}_s{seed}
#SBATCH -o %x.o%j
#SBATCH -p gpushort
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 1:0:0
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1

set -euo pipefail

REPO_ROOT="/data/home/bty414/opentarget_temporal_study/src/opentarget_het_graph"
cd "$REPO_ROOT"

source .venv/bin/activate
export WANDB_MODE="disabled"
export SAVE_PER_EPOCH_TOPK=100

python src/train_advancement_lambdarank.py \\
    --config config/experiments/headline_grouped/{arch}_s{seed}.yaml
"""

SWEEP_TMPL = r"""#!/bin/bash
# Submit all 3 EAHGT-allrank_grouped runs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
for f in run_*.sh; do
    sbatch "$f"
done
"""


def main():
    n = 0
    ref = (V4_DIR / SRC).read_text()
    for seed in SEEDS:
        out = patch(ref, seed)
        (CFG_DIR / f"{ARCH}_s{seed}.yaml").write_text(out)
        sb_path = SCRIPTS_DIR / f"run_{ARCH}_s{seed}.sh"
        sb_path.write_text(SBATCH_TMPL.format(arch=ARCH, seed=seed))
        sb_path.chmod(0o755)
        n += 1
    sweep = SCRIPTS_DIR / "submit_all.sh"
    sweep.write_text(SWEEP_TMPL)
    sweep.chmod(0o755)
    print(f"Wrote {n} configs + sbatch scripts.")


if __name__ == "__main__":
    main()
