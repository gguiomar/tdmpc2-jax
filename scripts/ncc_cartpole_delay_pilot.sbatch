#!/usr/bin/env bash
# Frozen seven-profile Cartpole delay pilot. One profile per H200.

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.log

set -euo pipefail

VENV_PATH="${VENV_PATH:-$HOME/.venvs/temporalhorizon-jax}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
CONTROLLER="${CONTROLLER:?CONTROLLER is required}"
SCORE_MODE="${SCORE_MODE:?SCORE_MODE is required}"
SEED="${SEED:?SEED is required}"
INITIAL_HORIZON="${INITIAL_HORIZON:?INITIAL_HORIZON is required}"
RUN_DIR="${RUN_DIR:?RUN_DIR is required}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
CONFIG_HASH="${CONFIG_HASH:?CONFIG_HASH is required}"
AUGMENT_DELAY_OBSERVATION="${AUGMENT_DELAY_OBSERVATION:-true}"
PROBE_TIMING="${PROBE_TIMING:-false}"
REFERENCE_PROBE="${REFERENCE_PROBE:-false}"
SMOKE_MODE="${SMOKE_MODE:-false}"

MAX_STEPS=500000
SAVE_INTERVAL_STEPS=50000
EVAL_INTERVAL_STEPS=50000
EVAL_NUM_EPISODES=20
ANCHOR_STEPS_HYDRA='[0,100000,150000,250000,350000,450000,500000]'
ANCHOR_STEPS_CSV='0,100000,150000,250000,350000,450000,500000'
PROBE_ANCHOR_STEPS='[100000,250000,450000]'
REFERENCE_ANCHOR_STEPS='[100000,250000,450000]'
TIMING_WARMUP_CALLS=5
TIMING_REPETITIONS=30
REFERENCE_EVAL_STEPS=256
CONDITIONAL_REFERENCE_EVAL_STEPS=500
if [[ "${SMOKE_MODE}" == "true" ]]; then
  MAX_STEPS=24000
  SAVE_INTERVAL_STEPS=24000
  EVAL_INTERVAL_STEPS=24000
  EVAL_NUM_EPISODES=4
  ANCHOR_STEPS_HYDRA='[0,24000]'
  ANCHOR_STEPS_CSV='0,24000'
  PROBE_TIMING=true
  REFERENCE_PROBE=true
  PROBE_ANCHOR_STEPS='[20000]'
  REFERENCE_ANCHOR_STEPS='[20000]'
  TIMING_WARMUP_CALLS=1
  TIMING_REPETITIONS=2
  REFERENCE_EVAL_STEPS=32
  CONDITIONAL_REFERENCE_EVAL_STEPS=32
fi

case "${CONTROLLER}" in
  adaptive)
    DENSE_ENABLED=true
    SCRIPTED_ENABLED=false
    ;;
  scripted)
    DENSE_ENABLED=false
    SCRIPTED_ENABLED=true
    ;;
  fixed)
    DENSE_ENABLED=false
    SCRIPTED_ENABLED=false
    ;;
  *)
    echo "Unsupported CONTROLLER=${CONTROLLER}" >&2
    exit 2
    ;;
esac

source "${VENV_PATH}/bin/activate"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export WANDB_CONSOLE=off
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "${SLURM_SUBMIT_DIR:-$PWD}"
SUBMIT_REPO="$(pwd)"

case "${RUN_DIR}" in
  /*) ;;
  *) RUN_DIR="${SUBMIT_REPO}/${RUN_DIR}" ;;
esac

ACTUAL_COMMIT="$(git rev-parse HEAD)"
if [[ "${ACTUAL_COMMIT}" != "${EXPECTED_COMMIT}" ]]; then
  echo "Commit mismatch: expected ${EXPECTED_COMMIT}, got ${ACTUAL_COMMIT}" >&2
  exit 3
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Tracked scientific files are dirty in Slurm checkout" >&2
  git status --short >&2
  exit 3
fi

JOB_SOURCE_PARENT="${SLURM_TMPDIR:-/tmp}"
JOB_SOURCE="${JOB_SOURCE_PARENT}/cartpole-delay-source-${SLURM_JOB_ID:-$$}"
if [[ -e "${JOB_SOURCE}" ]]; then
  echo "Refusing to overwrite existing job source snapshot ${JOB_SOURCE}" >&2
  exit 3
fi
git clone --quiet --shared --no-checkout "${SUBMIT_REPO}" "${JOB_SOURCE}"
git -C "${JOB_SOURCE}" checkout --quiet --detach "${EXPECTED_COMMIT}"
cd "${JOB_SOURCE}"
if [[ "$(git rev-parse HEAD)" != "${EXPECTED_COMMIT}" ]]; then
  echo "Detached job source revision mismatch" >&2
  exit 3
fi

mkdir -p "${RUN_DIR}"
echo "HOST=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "RUN_ID=${RUN_ID} CONTROLLER=${CONTROLLER} SCORE_MODE=${SCORE_MODE} SEED=${SEED}"
echo "EXPECTED_COMMIT=${EXPECTED_COMMIT} CONFIG_HASH=${CONFIG_HASH}"
echo "MAX_STEPS=${MAX_STEPS} PROBE_TIMING=${PROBE_TIMING} REFERENCE_PROBE=${REFERENCE_PROBE}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader

PY_ARGS=(
  hydra.run.dir="${RUN_DIR}"
  +run_id="${RUN_ID}"
  +controller="${CONTROLLER}"
  +score_mode="${SCORE_MODE}"
  seed="${SEED}"
  max_steps="${MAX_STEPS}"
  save_interval_steps="${SAVE_INTERVAL_STEPS}"
  log_interval_steps=10000
  update_chunk_size=128
  collect_chunk_steps=100
  seed_steps_override=4000
  checkpoint_buffer=true
  artifact_capture_enabled=true
  artifact_anchor_enabled=true
  artifact_anchor_steps="${ANCHOR_STEPS_HYDRA}"
  artifact_run_id="${RUN_ID}"
  scripted_horizon.enabled="${SCRIPTED_ENABLED}"
  scripted_horizon.schedule_steps='[0,150000,350000]'
  scripted_horizon.schedule_values='[3,7,3]'
  probe_timing.enabled="${PROBE_TIMING}"
  probe_timing.anchor_steps="${PROBE_ANCHOR_STEPS}"
  probe_timing.probe_counts='[0,2,4,8,16,32,64]'
  probe_timing.warmup_calls="${TIMING_WARMUP_CALLS}"
  probe_timing.repetitions="${TIMING_REPETITIONS}"
  reference_probe.enabled="${REFERENCE_PROBE}"
  reference_probe.anchor_steps="${REFERENCE_ANCHOR_STEPS}"
  reference_probe.num_env_eval_replicas=128
  reference_probe.env_eval_steps="${REFERENCE_EVAL_STEPS}"
  reference_probe.conditional_reference_env_eval_steps="${CONDITIONAL_REFERENCE_EVAL_STEPS}"
  env.backend=mjx_dmc
  env.env_id=cartpole-swingup
  env.num_envs=8
  env.utd_ratio=1.0
  env.asynchronous=false
  env.mjx_dmc.task=cartpole-swingup
  env.mjx_dmc.action_repeat=2
  env.mjx_dmc.episode_length=500
  env.mjx_dmc.enable_domain_randomization=false
  env.mjx_dmc.enable_observation_noise=false
  env.mjx_dmc.observation_noise_scale=0.0
  env.mjx_dmc.base_action_delay=0
  +env.mjx_dmc.action_delay_schedule_enabled=true
  +env.mjx_dmc.action_delay_observation_enabled="${AUGMENT_DELAY_OBSERVATION}"
  env.mjx_dmc.reset_pool_size=64
  dense_rhs.enabled="${DENSE_ENABLED}"
  dense_rhs.initial_horizon="${INITIAL_HORIZON}"
  dense_rhs.start_query_step=20000
  dense_rhs.query_interval_steps=20000
  dense_rhs.horizons='[2,3,4,5,6,7,8]'
  dense_rhs.hmax=8
  dense_rhs.horizon_buckets='[]'
  dense_rhs.roughness_probe=projected_jvp
  dense_rhs.num_roughness_probes=64
  dense_rhs.score_mode="${SCORE_MODE}"
  dense_rhs.additive_return_scale=50.0
  dense_rhs.additive_return_std_scale=50.0
  dense_rhs.additive_log_roughness_scale=1.0
  dense_rhs.score_evidence_floor=1e-6
  dense_rhs.decision_rule=paired_lcb
  dense_rhs.confidence_z=1.6448536
  dense_rhs.switch_threshold=0.02
  dense_rhs.robust_return=mean
  dense_rhs.selection_return_power=1.0
  dense_rhs.roughness_weight=1.0
  dense_rhs.return_std_weight=1.0
  dense_rhs.learner_proxy_enabled=false
  dense_rhs.learner_proxy_weight=0.0
  dense_rhs.deployment_utility_enabled=false
  dense_rhs.local_window_radius=1
  dense_rhs.max_transition_delta=1
  dense_rhs.incumbent_switch_margin=0.0
  dense_rhs.credible_transition_enabled=false
  dense_rhs.transition_cost_scale=0.0
  dense_rhs.phase_pruning_enabled=false
  dense_rhs.phase_min_samples_to_drop=1000000000
  dense_rhs.candidate_budget.A=3
  dense_rhs.candidate_budget.B1=3
  dense_rhs.candidate_budget.B2=3
  dense_rhs.candidate_budget.B3=3
  dense_rhs.candidate_budget.B4=3
  dense_rhs.num_env_eval_replicas=32
  dense_rhs.env_eval_enabled=true
  dense_rhs.env_eval_steps=256
  dense_rhs.query_population_size=256
  dense_rhs.query_policy_prior_samples=12
  dense_rhs.query_num_elites=32
  dense_rhs.query_mppi_iterations=4
  dense_rhs.query_temperature=0.5
  dense_rhs.prewarm_horizons=true
  dense_rhs.prewarm_all_phases=false
  evaluation.enabled=true
  evaluation.interval_steps="${EVAL_INTERVAL_STEPS}"
  evaluation.num_episodes="${EVAL_NUM_EPISODES}"
  evaluation.clean=true
  evaluation.prewarm=false
  tdmpc2.horizon="${INITIAL_HORIZON}"
  tdmpc2.population_size=512
  tdmpc2.policy_prior_samples=24
  tdmpc2.num_elites=64
  tdmpc2.mppi_iterations=6
  tdmpc2.temperature=0.5
  wandb.enabled=false
)

python -m tdmpc2_jax.train "${PY_ARGS[@]}"
touch "${RUN_DIR}/TRAINING_COMPLETE"
python scripts/plot_run_summary.py --run-dir "${RUN_DIR}" || true
python scripts/render_cartpole_delay_gifs.py \
  "${RUN_DIR}" \
  --run-id "${RUN_ID}" \
  --steps "${ANCHOR_STEPS_CSV}" \
  --width 256 \
  --height 256 \
  --frame-stride 4 \
  --mujoco-gl egl
python scripts/finalize_cartpole_delay_run.py \
  --run-dir "${RUN_DIR}" \
  --run-id "${RUN_ID}" \
  --controller "${CONTROLLER}" \
  --score-mode "${SCORE_MODE}" \
  --seed "${SEED}" \
  --expected-step "${MAX_STEPS}" \
  --anchor-steps "${ANCHOR_STEPS_CSV}" \
  --expected-commit "${EXPECTED_COMMIT}" \
  --config-hash "${CONFIG_HASH}" \
  --conditional-reference-eval-steps "${CONDITIONAL_REFERENCE_EVAL_STEPS}"

VALIDATE_ARGS=(
  --run-dir "${RUN_DIR}"
  --controller "${CONTROLLER}"
  --min-steps "${MAX_STEPS}"
  --anchor-steps "${ANCHOR_STEPS_CSV}"
  --expected-run-id "${RUN_ID}"
  --expected-score-mode "${SCORE_MODE}"
  --expected-seed "${SEED}"
  --expected-commit "${EXPECTED_COMMIT}"
  --expected-config-hash "${CONFIG_HASH}"
)
if [[ "${PROBE_TIMING}" == "true" ]]; then
  VALIDATE_ARGS+=(--require-probe-timing)
fi
if [[ "${REFERENCE_PROBE}" == "true" ]]; then
  VALIDATE_ARGS+=(--require-reference-probe)
fi
python scripts/validate_cartpole_delay_run.py "${VALIDATE_ARGS[@]}"

echo "CARTPOLE_DELAY_RUN_VALID RUN_DIR=${RUN_DIR}"
