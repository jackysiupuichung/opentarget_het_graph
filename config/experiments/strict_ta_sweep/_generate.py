#!/usr/bin/env python3
"""Strict-mask TA-generalisation sweep: lift TA-mean/median RS@10.

Strict `<` masking (committed default) generalises well at @50/@100 (TA-mean
beats RDG/OTS) but loses the head: TA-mean RS@10 3.26 < RDG 4.10, TA-median
1.87 worst of four. Goal: raise TA-mean/median RS@10 WITHOUT re-admitting the
same-year leak. Three levers (user-selected):

  ndcg_k    : {10, 30}       # tighter loss truncation -> focus gradient on head
  reduction : {sum, mean}    # mean balances the 13 TA slates equally (per-TA)
  capacity  : {h128, h256, h128_do2}  # more capacity / more reg for honest model

2 x 2 x 3 = 12 probe configs at seed 42. Grouped impl, strict masking (trainer
default), val-select on rs_ta_mean@10 (the head-of-TA metric we optimise).
Winners get scaled to 5 seeds separately. Base = strictmask_s42 recipe.
"""
from pathlib import Path
import re

REPO = Path(__file__).resolve().parents[3]
CFG_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = REPO / "scripts" / "advancement_prediction" / "strict_ta_sweep"
SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

# Base recipe: the committed k10/mean/h128 sweep config is self-contained and
# carries every field patch() rewrites (former lr_grouped_k100_strictmask base
# is no longer on disk). patch() is idempotent, so re-patching it is safe.
SRC = CFG_DIR / "st_k10_mean_h128.yaml"
OUTPUT_ROOT = "/gpfs/scratch/bty414/opentarget_evidences/26.03/runs/strict_ta_sweep"

NDCG_KS = [10, 30]
REDUCTIONS = ["sum", "mean"]
# capacity/reg variants: (hidden_dim, dropout, decoder_dropout, tag)
# h256 dropped: OOMs even on the 80GB A100 (needs batch/grad-accum rework).
CAPS = [
    (128, 0.1, 0.1, "h128"),
    (128, 0.2, 0.2, "h128do2"),
]


def tag(k, reduction, cap):
    return f"st_k{k}_{reduction}_{cap}"


def patch(text, k, reduction, cap_hd, cap_do, cap_dd, name):
    text = re.sub(r"experiment:\n  name: .*", f"experiment:\n  name: {name}", text)
    text = re.sub(r"ndcg_k: \d+", f"ndcg_k: {k}", text)
    # reduction: insert after ndcg_k if absent
    if re.search(r"reduction: \w+", text):
        text = re.sub(r"reduction: \w+", f"reduction: {reduction}", text)
    else:
        text = re.sub(r"(ndcg_k: \d+\n)", rf"\1    reduction: {reduction}\n", text)
    # capacity / regularisation
    text = re.sub(r"hidden_dim: \d+", f"hidden_dim: {cap_hd}", text)
    text = re.sub(r"  dropout: [\d.]+", f"  dropout: {cap_do}", text)
    text = re.sub(r"decoder_dropout: [\d.]+", f"decoder_dropout: {cap_dd}", text)
    # val-select + early-stop metric = head-of-TA
    text = re.sub(r"metric: rs_ta_median@50", "metric: rs_ta_mean@10", text)
    # eval ks: add 10 for per-epoch head tracking
    text = re.sub(r"ks: \[100\]", "ks: [10, 50, 100]", text)
    text = re.sub(r"output_dir: .*", f"output_dir: {OUTPUT_ROOT}/{name}", text)
    return text


SBATCH_TMPL = """\
#!/bin/bash
#SBATCH -J {name}
#SBATCH -o %x.o%j
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 240:0:0
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1

set -euo pipefail
REPO_ROOT="/data/home/bty414/opentarget_temporal_study/src/opentarget_het_graph"
cd "$REPO_ROOT"
source .venv/bin/activate
export WANDB_MODE="disabled"
python src/train_advancement_lambdarank.py \\
    --config config/experiments/strict_ta_sweep/{name}.yaml
"""

SUBMIT_TMPL = """#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
for f in run_*.sh; do sbatch "$f"; done
"""


def main():
    ref = SRC.read_text()
    n = 0
    for k in NDCG_KS:
        for reduction in REDUCTIONS:
            for cap_hd, cap_do, cap_dd, cap in CAPS:
                name = tag(k, reduction, cap)
                (CFG_DIR / f"{name}.yaml").write_text(
                    patch(ref, k, reduction, cap_hd, cap_do, cap_dd, name))
                sb = SCRIPTS_DIR / f"run_{name}.sh"
                sb.write_text(SBATCH_TMPL.format(name=name))
                sb.chmod(0o755)
                n += 1
    submit = SCRIPTS_DIR / "submit_all.sh"
    submit.write_text(SUBMIT_TMPL)
    submit.chmod(0o755)
    print(f"Wrote {n} configs + sbatch to {CFG_DIR} and {SCRIPTS_DIR}")


if __name__ == "__main__":
    main()
