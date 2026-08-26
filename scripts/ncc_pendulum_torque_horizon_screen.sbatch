#!/usr/bin/env bash
# Fixed-horizon Pendulum torque screen. Every job restores the same validated
# 30k parent checkpoint; GPU execution is Slurm-only.

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.log

set -euo pipefail

VENV_PATH="${VENV_PATH:-$HOME/.venvs/temporalhorizon-jax}"
MODE="${MODE:?MODE must be smoke or full}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
RUN_DIR="${RUN_DIR:?RUN_DIR is required}"
PARENT_RUN_DIR="${PARENT_RUN_DIR:?PARENT_RUN_DIR is required}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
CONFIG_HASH="${CONFIG_HASH:?CONFIG_HASH is required}"
TORQUE_SCALE="${TORQUE_SCALE:?TORQUE_SCALE is required}"
FIXED_HORIZON="${FIXED_HORIZON:?FIXED_HORIZON is required}"

case "${MODE}" in
  smoke)
    MAX_STEPS=30800
    SAVE_INTERVAL_STEPS=30800
    ;;
  full)
    MAX_STEPS=46000
    # Orbax evaluates this cadence against the absolute restored global step.
    SAVE_INTERVAL_STEPS=23000
    ;;
  *)
    echo "Unsupported MODE=${MODE}" >&2
    exit 2
    ;;
esac
case "${FIXED_HORIZON}" in
  3|8) ;;
  *) echo "FIXED_HORIZON must be 3 or 8" >&2; exit 2 ;;
esac

source "${VENV_PATH}/bin/activate"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export WANDB_CONSOLE=off
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd "${SLURM_SUBMIT_DIR:-$PWD}"
SUBMIT_REPO="$(pwd)"
case "${RUN_DIR}" in /*) ;; *) RUN_DIR="${SUBMIT_REPO}/${RUN_DIR}" ;; esac
case "${PARENT_RUN_DIR}" in
  /*) ;;
  *) PARENT_RUN_DIR="${SUBMIT_REPO}/${PARENT_RUN_DIR}" ;;
esac

ACTUAL_COMMIT="$(git rev-parse HEAD)"
if [[ "${ACTUAL_COMMIT}" != "${EXPECTED_COMMIT}" ]]; then
  echo "Commit mismatch: expected ${EXPECTED_COMMIT}, got ${ACTUAL_COMMIT}" >&2
  exit 3
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Tracked scientific files are dirty in the Slurm checkout" >&2
  git status --short >&2
  exit 3
fi

JOB_SOURCE_PARENT="${SLURM_TMPDIR:-/tmp}"
JOB_SOURCE="${JOB_SOURCE_PARENT}/pendulum-torque-screen-${SLURM_JOB_ID:-$$}"
if [[ -e "${JOB_SOURCE}" ]]; then
  echo "Refusing to overwrite existing job source ${JOB_SOURCE}" >&2
  exit 3
fi
git clone --quiet --shared --no-checkout "${SUBMIT_REPO}" "${JOB_SOURCE}"
git -C "${JOB_SOURCE}" checkout --quiet --detach "${EXPECTED_COMMIT}"
cd "${JOB_SOURCE}"

if [[ -e "${RUN_DIR}" ]]; then
  echo "Refusing to overwrite existing run directory ${RUN_DIR}" >&2
  exit 3
fi
mkdir -p "${RUN_DIR}"

python scripts/analyze_pendulum_h8_torque02.py validate \
  --phase base \
  --run-dir "${PARENT_RUN_DIR}"
cp -a "${PARENT_RUN_DIR}/checkpoint" "${RUN_DIR}/checkpoint"

python scripts/analyze_pendulum_torque_horizon_screen.py write-manifest \
  --run-dir "${RUN_DIR}" \
  --run-id "${RUN_ID}" \
  --mode "${MODE}" \
  --commit "${EXPECTED_COMMIT}" \
  --config-hash "${CONFIG_HASH}" \
  --seed 7 \
  --torque-scale "${TORQUE_SCALE}" \
  --horizon "${FIXED_HORIZON}" \
  --parent-run-dir "${PARENT_RUN_DIR}"

echo "HOST=$(hostname) SLURM_JOB_ID=${SLURM_JOB_ID:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "RUN_ID=${RUN_ID} MODE=${MODE} TORQUE_SCALE=${TORQUE_SCALE} FIXED_HORIZON=${FIXED_HORIZON}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader

PY_ARGS=(
  hydra.run.dir="${RUN_DIR}"
  +run_id="${RUN_ID}"
  +controller=fixed
  +score_mode=tdmpc2_fixed
  seed=7
  max_steps="${MAX_STEPS}"
  save_interval_steps="${SAVE_INTERVAL_STEPS}"
  log_interval_steps=2000
  update_chunk_size=128
  collect_chunk_steps=100
  seed_steps_override=4000
  checkpoint_buffer=true
  artifact_capture_enabled=false
  artifact_anchor_enabled=false
  probe_timing.enabled=false
  reference_probe.enabled=false
  env.backend=mjx_dmc
  env.env_id=pendulum-swingup
  env.num_envs=8
  env.utd_ratio=1.0
  env.asynchronous=false
  env.mjx_dmc.task=pendulum-swingup
  env.mjx_dmc.action_repeat=2
  env.mjx_dmc.episode_length=500
  env.mjx_dmc.enable_domain_randomization=false
  env.mjx_dmc.enable_observation_noise=false
  env.mjx_dmc.observation_noise_scale=0.0
  env.mjx_dmc.actuator_strength_scale="${TORQUE_SCALE}"
  env.mjx_dmc.base_action_delay=0
  +env.mjx_dmc.action_delay_schedule_enabled=false
  +env.mjx_dmc.action_delay_observation_enabled=false
  env.mjx_dmc.reset_pool_size=64
  dense_rhs.enabled=false
  evaluation.enabled=true
  evaluation.interval_steps=2000
  evaluation.num_episodes=10
  evaluation.clean=true
  evaluation.prewarm=false
  tdmpc2.horizon="${FIXED_HORIZON}"
  tdmpc2.population_size=512
  tdmpc2.policy_prior_samples=24
  tdmpc2.num_elites=64
  tdmpc2.mppi_iterations=6
  tdmpc2.temperature=0.5
  wandb.enabled=false
)

python -m tdmpc2_jax.train "${PY_ARGS[@]}"
touch "${RUN_DIR}/TRAINING_COMPLETE"
python scripts/analyze_pendulum_torque_horizon_screen.py validate \
  --run-dir "${RUN_DIR}"
touch "${RUN_DIR}/RUN_VALID"
echo "PENDULUM_TORQUE_SCREEN_RUN_VALID RUN_DIR=${RUN_DIR}"
