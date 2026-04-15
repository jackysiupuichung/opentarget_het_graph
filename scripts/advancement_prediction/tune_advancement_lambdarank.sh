#!/bin/bash
#SBATCH -J tune_advancement_lambdarank
#SBATCH -o %x.o%j
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 240:0:0
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu

# W&B-sweep hyperparameter tuning for LambdaRank advancement prediction.
#
# All tuning settings (n_trials, search_space, …) live in the experiment config
# under the `tune:` key. CLI flags here are optional overrides.
#
# Usage:
#   bash scripts/advancement_prediction/tune_advancement_lambdarank.sh [options]
#   sbatch scripts/advancement_prediction/tune_advancement_lambdarank.sh [options]
#
# Options:
#   --config PATH        Experiment config with a tune: section
#   --n_trials N         Override tune.n_trials for this agent
#   --sweep_id ID        Join an existing W&B sweep instead of creating one
#   --output_dir PATH    Override output directory
#   --entity NAME        W&B entity (default: logged-in user)
#   --offline            Run W&B offline (default: online)

set -euo pipefail

REPO_ROOT="/data/home/bty414/opentarget_temporal_study/src/opentarget_het_graph"
cd "$REPO_ROOT"

source .venv/bin/activate

# ── Defaults ──────────────────────────────────────────────────────────────────
CONFIG="config/experiments/advancement_lambdarank_tune.yaml"
WANDB_MODE_VAL="online"

N_TRIALS=""
SWEEP_ID=""
OUTPUT_DIR=""
ENTITY=""

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)      CONFIG="$2";     shift 2 ;;
        --n_trials)    N_TRIALS="$2";   shift 2 ;;
        --sweep_id)    SWEEP_ID="$2";   shift 2 ;;
        --output_dir)  OUTPUT_DIR="$2"; shift 2 ;;
        --entity)      ENTITY="$2";     shift 2 ;;
        --offline)     WANDB_MODE_VAL="offline"; shift ;;
        *)             echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "Config     : $CONFIG"
echo "W&B        : $WANDB_MODE_VAL"
[[ -n "$N_TRIALS"   ]] && echo "n_trials   : $N_TRIALS (override)"
[[ -n "$SWEEP_ID"   ]] && echo "sweep_id   : $SWEEP_ID"
[[ -n "$OUTPUT_DIR" ]] && echo "output_dir : $OUTPUT_DIR (override)"
[[ -n "$ENTITY"     ]] && echo "entity     : $ENTITY"
echo "--------------------------------------------"

# ── Environment ───────────────────────────────────────────────────────────────
export WANDB_MODE="$WANDB_MODE_VAL"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# ── Build command ─────────────────────────────────────────────────────────────
CMD=(
    python src/tune_advancement_lambdarank.py
    --config "$CONFIG"
)

[[ -z "$SWEEP_ID"   ]] && CMD+=(--create_sweep)
[[ -n "$SWEEP_ID"   ]] && CMD+=(--sweep_id   "$SWEEP_ID")
[[ -n "$N_TRIALS"   ]] && CMD+=(--n_trials   "$N_TRIALS")
[[ -n "$OUTPUT_DIR" ]] && CMD+=(--output_dir "$OUTPUT_DIR")
[[ -n "$ENTITY"     ]] && CMD+=(--entity     "$ENTITY")

"${CMD[@]}"
