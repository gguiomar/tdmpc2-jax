#!/usr/bin/env bash
# Short MJX DMC environment gate validation on NCC.

#SBATCH --job-name=tdmpc2-jax-mjx-gate
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=%x_%j.log

set -euo pipefail

VENV_PATH="${VENV_PATH:-$HOME/.venvs/temporalhorizon-jax}"
TASKS="${TASKS:-fish-swim finger-turn_hard cheetah-run cartpole-swingup acrobot-swingup walker-run hopper-hop}"
OUT_DIR="${OUT_DIR:-runs/results/mjx_gates}"
SEED="${SEED:-0}"
NUM_ENVS="${NUM_ENVS:-4}"
STEPS="${STEPS:-16}"
RESET_POOL_SIZE="${RESET_POOL_SIZE:-8}"
CHAOS="${CHAOS:-false}"

source "$VENV_PATH/bin/activate"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "HOST=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "TASKS=${TASKS}"
echo "OUT_DIR=${OUT_DIR}"

ARGS=(
  --tasks ${TASKS}
  --out-dir "${OUT_DIR}"
  --seed "${SEED}"
  --num-envs "${NUM_ENVS}"
  --steps "${STEPS}"
  --reset-pool-size "${RESET_POOL_SIZE}"
)

if [[ "${CHAOS}" == "true" ]]; then
  ARGS+=(--chaos)
fi

python scripts/check_mjx_dmc_gate.py "${ARGS[@]}"
