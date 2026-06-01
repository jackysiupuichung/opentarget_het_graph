#!/bin/bash
#SBATCH -J tune_bilinear_sweep
#SBATCH -o %x.o%j
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 24:0:0
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu

# W&B Bayesian sweep over bilinear-decoder HPs for EAHGT (s42).
# Mirrors the historical tune_advancement_hgt.sh pattern: one long sbatch
# job runs the agent for many trials sequentially.
#
# Usage:
#   sbatch scripts/advancement_prediction/bilinear_sweep/tune_bilinear_sweep.sh
#   sbatch scripts/advancement_prediction/bilinear_sweep/tune_bilinear_sweep.sh --sweep_id <id>
#
# Options:
#   --sweep_id ID      Join an existing W&B sweep instead of creating one
#   --n_trials  N      Override tune.n_trials for this agent
#   --entity   NAME    W&B entity (default: logged-in user)
#   --offline          Run W&B offline (default: online)

set -euo pipefail

REPO_ROOT="/data/home/bty414/opentarget_temporal_study/src/opentarget_het_graph"
cd "$REPO_ROOT"

source .venv/bin/activate

CONFIG="config/experiments/bilinear_sweep/sweep.yaml"
WANDB_MODE_VAL="online"

N_TRIALS=""
SWEEP_ID=""
ENTITY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --n_trials)  N_TRIALS="$2"; shift 2 ;;
        --sweep_id)  SWEEP_ID="$2"; shift 2 ;;
        --entity)    ENTITY="$2";   shift 2 ;;
        --offline)   WANDB_MODE_VAL="offline"; shift ;;
        *)           echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "Config   : $CONFIG"
echo "W&B mode : $WANDB_MODE_VAL"
[[ -n "$N_TRIALS" ]] && echo "n_trials : $N_TRIALS (override)"
[[ -n "$SWEEP_ID" ]] && echo "sweep_id : $SWEEP_ID"
[[ -n "$ENTITY"   ]] && echo "entity   : $ENTITY"
echo "--------------------------------------------"

export WANDB_MODE="$WANDB_MODE_VAL"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

CMD=(
    python -m src.tune_advancement_lambdarank
    --config "$CONFIG"
)

[[ -z "$SWEEP_ID" ]] && CMD+=(--create_sweep)
[[ -n "$SWEEP_ID" ]] && CMD+=(--sweep_id "$SWEEP_ID")
[[ -n "$N_TRIALS" ]] && CMD+=(--n_trials "$N_TRIALS")
[[ -n "$ENTITY"   ]] && CMD+=(--entity   "$ENTITY")

"${CMD[@]}"
