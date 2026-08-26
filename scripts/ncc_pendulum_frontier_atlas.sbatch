#!/usr/bin/env bash
# Frozen-checkpoint Pendulum frontier atlas. GPU execution is Slurm-only.

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%A_%a.log

set -euo pipefail

VENV_PATH="${VENV_PATH:-$HOME/.venvs/temporalhorizon-jax}"
MODE="${MODE:?MODE must be smoke or full}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/pendulum_frontier_atlas}"
SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-outputs/pendulum_h8_torque02/pendh8__base_t1p0_h3__s7/attempt_1}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
ATTEMPT="${ATTEMPT:-1}"

case "${MODE}" in
  smoke)
    SHARD_INDEX=0
    NUM_SHARDS=1
    CONDITION_LIMIT=1
    REPLICAS=4
    EVAL_STEPS=32
    RUN_ID="pendatlas__smoke_nominal__s7"
    ;;
  full)
    SHARD_INDEX="${SLURM_ARRAY_TASK_ID:?full mode requires a Slurm array task}"
    NUM_SHARDS=4
    CONDITION_LIMIT=""
    REPLICAS=32
    EVAL_STEPS=500
    RUN_ID="pendatlas__broad_shard${SHARD_INDEX}of${NUM_SHARDS}__s7"
    ;;
  *)
    echo "Unsupported MODE=${MODE}" >&2
    exit 2
    ;;
esac

CONFIG_HASH="$(printf '%s|mode=%s|shard=%s/%s|limit=%s|replicas=%s|steps=%s|source=30000|horizons=2:8|planner=256:12:32:4:0.5|seed=7' \
  'pendulum-frontier-atlas-v1' "${MODE}" "${SHARD_INDEX}" "${NUM_SHARDS}" \
  "${CONDITION_LIMIT:-none}" "${REPLICAS}" "${EVAL_STEPS}" | sha256sum | awk '{print $1}')"
RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}/attempt_${ATTEMPT}"

source "${VENV_PATH}/bin/activate"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export WANDB_CONSOLE=off
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd "${SLURM_SUBMIT_DIR:-$PWD}"
SUBMIT_REPO="$(pwd)"
case "${RUN_DIR}" in /*) ;; *) RUN_DIR="${SUBMIT_REPO}/${RUN_DIR}" ;; esac
case "${SOURCE_RUN_DIR}" in
  /*) ;;
  *) SOURCE_RUN_DIR="${SUBMIT_REPO}/${SOURCE_RUN_DIR}" ;;
esac

ACTUAL_COMMIT="$(git rev-parse HEAD)"
if [[ "${ACTUAL_COMMIT}" != "${EXPECTED_COMMIT}" ]]; then
  echo "Commit mismatch: expected ${EXPECTED_COMMIT}, got ${ACTUAL_COMMIT}" >&2
  exit 3
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Tracked scientific files are dirty in the isolated Slurm checkout" >&2
  git status --short >&2
  exit 3
fi

JOB_SOURCE_PARENT="${SLURM_TMPDIR:-/tmp}"
JOB_SOURCE="${JOB_SOURCE_PARENT}/pendulum-frontier-atlas-${SLURM_JOB_ID:-$$}"
if [[ -e "${JOB_SOURCE}" ]]; then
  echo "Refusing to overwrite existing job source ${JOB_SOURCE}" >&2
  exit 3
fi
git clone --quiet --shared --no-checkout "${SUBMIT_REPO}" "${JOB_SOURCE}"
git -C "${JOB_SOURCE}" checkout --quiet --detach "${EXPECTED_COMMIT}"
cd "${JOB_SOURCE}"

python scripts/analyze_pendulum_h8_torque02.py validate \
  --phase base \
  --run-dir "${SOURCE_RUN_DIR}"

MANIFEST_ARGS=(
  write-manifest
  --run-dir "${RUN_DIR}"
  --run-id "${RUN_ID}"
  --mode "${MODE}"
  --commit "${EXPECTED_COMMIT}"
  --config-hash "${CONFIG_HASH}"
  --seed 7
  --source-run-dir "${SOURCE_RUN_DIR}"
  --shard-index "${SHARD_INDEX}"
  --num-shards "${NUM_SHARDS}"
  --replicas "${REPLICAS}"
  --eval-steps "${EVAL_STEPS}"
)
if [[ -n "${CONDITION_LIMIT}" ]]; then
  MANIFEST_ARGS+=(--condition-limit "${CONDITION_LIMIT}")
fi
python -m scripts.analyze_pendulum_frontier_atlas "${MANIFEST_ARGS[@]}"
cp -a "${SOURCE_RUN_DIR}/checkpoint" "${RUN_DIR}/checkpoint"

echo "HOST=$(hostname) SLURM_JOB_ID=${SLURM_JOB_ID:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "RUN_ID=${RUN_ID} MODE=${MODE} SHARD=${SHARD_INDEX}/${NUM_SHARDS} CONFIG_HASH=${CONFIG_HASH}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader

PY_ARGS=(
  hydra.run.dir="${RUN_DIR}"
  +run_id="${RUN_ID}"
  +controller=frontier_atlas
  +score_mode=paired_frozen_evaluation
  seed=7
  max_steps=30000
  save_interval_steps=30000
  checkpoint_buffer=true
  artifact_capture_enabled=false
  artifact_anchor_enabled=false
  probe_timing.enabled=false
  reference_probe.enabled=false
  frontier_atlas.enabled=true
  frontier_atlas.shard_index="${SHARD_INDEX}"
  frontier_atlas.num_shards="${NUM_SHARDS}"
  frontier_atlas.horizons='[2,3,4,5,6,7,8]'
  frontier_atlas.num_env_eval_replicas="${REPLICAS}"
  frontier_atlas.env_eval_steps="${EVAL_STEPS}"
  frontier_atlas.population_size=256
  frontier_atlas.policy_prior_samples=12
  frontier_atlas.num_elites=32
  frontier_atlas.mppi_iterations=4
  frontier_atlas.temperature=0.5
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
  env.mjx_dmc.actuator_strength_scale=1.0
  env.mjx_dmc.joint_damping_scale=1.0
  env.mjx_dmc.gravity_scale=1.0
  env.mjx_dmc.fixed_observation_noise_scale=null
  env.mjx_dmc.base_action_delay=0
  +env.mjx_dmc.action_delay_schedule_enabled=false
  +env.mjx_dmc.action_delay_observation_enabled=false
  env.mjx_dmc.reset_pool_size=64
  dense_rhs.enabled=true
  dense_rhs.initial_horizon=3
  dense_rhs.start_query_step=34000
  dense_rhs.query_interval_steps=4000
  dense_rhs.query_recent_transition_steps=4000
  dense_rhs.horizons='[2,3,4,5,6,7,8]'
  dense_rhs.hmax=8
  dense_rhs.horizon_buckets='[]'
  dense_rhs.roughness_probe=projected_jvp
  dense_rhs.num_roughness_probes=8
  dense_rhs.score_mode=additive
  dense_rhs.additive_return_scale=50.0
  dense_rhs.additive_return_std_scale=50.0
  dense_rhs.additive_log_roughness_scale=1.0
  dense_rhs.decision_rule=paired_lcb
  dense_rhs.confidence_z=1.6448536
  dense_rhs.switch_threshold=0.02
  dense_rhs.robust_return=mean
  dense_rhs.phase_pruning_enabled=false
  dense_rhs.phase_min_samples_to_drop=1000000000
  dense_rhs.candidate_budget.A=7
  dense_rhs.candidate_budget.B1=7
  dense_rhs.candidate_budget.B2=7
  dense_rhs.candidate_budget.B3=7
  dense_rhs.candidate_budget.B4=7
  dense_rhs.num_env_eval_replicas=32
  dense_rhs.env_eval_enabled=true
  dense_rhs.env_eval_steps=128
  dense_rhs.prewarm_horizons=false
  dense_rhs.prewarm_all_phases=false
  evaluation.enabled=false
  tdmpc2.horizon=3
  tdmpc2.population_size=512
  tdmpc2.policy_prior_samples=24
  tdmpc2.num_elites=64
  tdmpc2.mppi_iterations=6
  tdmpc2.temperature=0.5
  wandb.enabled=false
)
if [[ -n "${CONDITION_LIMIT}" ]]; then
  PY_ARGS+=(frontier_atlas.condition_limit="${CONDITION_LIMIT}")
else
  PY_ARGS+=(frontier_atlas.condition_limit=null)
fi

python -m tdmpc2_jax.train "${PY_ARGS[@]}"
touch "${RUN_DIR}/ATLAS_COMPLETE"
python -m scripts.analyze_pendulum_frontier_atlas validate \
  --run-dir "${RUN_DIR}" \
  --allow-missing-valid-marker
touch "${RUN_DIR}/RUN_VALID"
python -m scripts.analyze_pendulum_frontier_atlas validate --run-dir "${RUN_DIR}"
echo "PENDULUM_FRONTIER_ATLAS_RUN_VALID RUN_DIR=${RUN_DIR}"
