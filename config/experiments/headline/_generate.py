#!/usr/bin/env python3
"""Generate the 15-run headline sweep:
  5 architectures × 3 seeds × val_year=2013
  Lambdarank loss, ES patience=10, metric val_rs_ta_mean@50, mode max.

HPs per architecture are inherited from config/experiments/v4_ablation/*_s7.yaml
(the canonical v4 set). We only override:
  - data.val_min_year, data.val_max_year  → 2013
  - train.early_stopping                  → enabled, patience=10, rs_ta_mean@50
  - train.num_epochs                      → 40 (loose upper bound; ES decides)
  - train.output_dir                      → runs/headline/<arch>_s<seed>
  - experiment.name, seed
"""
from pathlib import Path
import re

REPO = Path(__file__).resolve().parents[3]
CFG_DIR = Path(__file__).resolve().parent
V4_DIR = REPO / "config" / "experiments" / "v4_ablation"
SCRIPTS_DIR = REPO / "scripts" / "advancement_prediction" / "headline"
SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

ARCHS = {
    "b1_hgt": "b1_hgt_s7.yaml",
    "b3_gatv2": "b3_gatv2_s7.yaml",
    "b6_rgcn": "b6_rgcn_s7.yaml",
    "b7_compgcn": "b7_compgcn_s7.yaml",
    "p3_eahgt_both": "p3_eahgt_both_s7.yaml",
}
SEEDS = [1, 7, 42]
OUTPUT_ROOT = "/gpfs/scratch/bty414/opentarget_evidences/26.03/runs/headline"
# 26.03 w3 graph generation (the 23.06 generation had a same-year clinical-trial
# edge leak and is retired). The v4 base configs still carry 23.06 paths, so we
# overwrite graph_file/mappings_file here rather than inherit them.
W3_GRAPH = "/gpfs/scratch/bty414/opentarget_evidences/26.03/graph/hetero_graph_with_features_datatype_w3.pt"
W3_MAP   = "/gpfs/scratch/bty414/opentarget_evidences/26.03/progression/temporal_graph_datatype_w3_mappings.pt"


def patch(text: str, arch: str, seed: int) -> str:
    text = re.sub(r"graph_file: .*", f"graph_file: {W3_GRAPH}", text)
    text = re.sub(r"mappings_file: .*", f"mappings_file: {W3_MAP}", text)
    text = re.sub(r"experiment:\n  name: .*",
                  f"experiment:\n  name: {arch}_headline_s{seed}", text)
    text = re.sub(r"val_min_year: \d+", "val_min_year: 2013", text)
    text = re.sub(r"val_max_year: \d+", "val_max_year: 2013", text)
    text = re.sub(r"output_dir: .*",
                  f"output_dir: {OUTPUT_ROOT}/{arch}_s{seed}", text)
    text = re.sub(r"num_epochs: \d+", "num_epochs: 40", text)
    text = re.sub(r"^seed: \d+", f"seed: {seed}", text, flags=re.MULTILINE)
    # Replace whole early_stopping block — patience 10, metric rs_ta_mean@50,
    # enabled true.
    text = re.sub(
        r"  early_stopping:\n(    \S.*\n)+",
        ("  early_stopping:\n"
         "    enabled: true\n"
         "    patience: 10\n"
         "    metric: rs_ta_mean@50\n"
         "    mode: max\n"),
        text,
    )
    return text


SBATCH_TMPL = """\
#!/bin/bash
#SBATCH -J h_{arch}_s{seed}
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
    --config config/experiments/headline/{arch}_s{seed}.yaml
"""


SWEEP_TMPL = r"""#!/bin/bash
# Submit all 15 headline-sweep runs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
for f in run_*.sh; do
    sbatch "$f"
done
"""


def main():
    n = 0
    for arch, src in ARCHS.items():
        ref = (V4_DIR / src).read_text()
        for seed in SEEDS:
            out = patch(ref, arch, seed)
            (CFG_DIR / f"{arch}_s{seed}.yaml").write_text(out)
            sbatch = SBATCH_TMPL.format(arch=arch, seed=seed)
            sb_path = SCRIPTS_DIR / f"run_{arch}_s{seed}.sh"
            sb_path.write_text(sbatch)
            sb_path.chmod(0o755)
            n += 1
    sweep = SCRIPTS_DIR / "submit_all.sh"
    sweep.write_text(SWEEP_TMPL)
    sweep.chmod(0o755)
    print(f"Wrote {n} configs to {CFG_DIR}")
    print(f"Wrote {n} sbatch scripts + submit_all.sh to {SCRIPTS_DIR}")


if __name__ == "__main__":
    main()
