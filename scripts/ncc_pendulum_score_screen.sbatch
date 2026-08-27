#!/usr/bin/env bash
# Checkpoint-forked Pendulum score-formulation screen. GPU work is Slurm-only.

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.log

set -euo pipefail

VENV_PATH="${VENV_PATH:-$HOME/.venvs/temporalhorizon-jax}"
MODE="${MODE:?MODE must be source, smoke, or full}"
PROFILE="${PROFILE:?PROFILE must be s0, s1, s2, or s3}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
RUN_DIR="${RUN_DIR:?RUN_DIR is required}"
PARENT_RUN_DIR="${PARENT_RUN_DIR:?PARENT_RUN_DIR is required}"
PARENT_CHECKPOINT_STEP="${PARENT_CHECKPOINT_STEP:?PARENT_CHECKPOINT_STEP is required}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
CONFIG_HASH="${CONFIG_HASH:?CONFIG_HASH is required}"

case "${PROFILE}" in
  s0) SCORE_VARIANT=current_additive ;;
  s1) SCORE_VARIANT=calibrated_local_roughness ;;
  s2) SCORE_VARIANT=return_first ;;
  s3) SCORE_VARIANT=curvature_bellman ;;
  *) echo "Unsupported PROFILE=${PROFILE}" >&2; exit 2 ;;
esac

case "${MODE}" in
  source)
    [[ "${PROFILE}" == "s0" ]] || { echo "Source preparation uses s0" >&2; exit 2; }
    MAX_STEPS=34000
    QUERY_START=32000
    QUERY_INTERVAL=4000
    DELAY_ENABLED=false
    DELAY_BOUNDARIES='[]'
    DELAY_VALUES='[]'
    ARTIFACT_ENABLED=false
    ARTIFACT_ANCHORS='[]'
    ARTIFACT_STEPS_CSV=''
    REFERENCE_ENABLED=false
    REFERENCE_ANCHORS='[]'
    ;;
  smoke)
    [[ "${PROFILE}" == "s3" ]] || { echo "Smoke exercises s3" >&2; exit 2; }
    MAX_STEPS=36000
    QUERY_START=34400
    QUERY_INTERVAL=400
    DELAY_ENABLED=true
    DELAY_BOUNDARIES='[34400,34800,35200,35600]'
    DELAY_VALUES='[0,2,6,4,0]'
    ARTIFACT_ENABLED=true
    ARTIFACT_ANCHORS='[34000,34400,34800,35200,35600,36000]'
    ARTIFACT_STEPS_CSV='34000,34400,34800,35200,35600,36000'
    REFERENCE_ENABLED=false
    REFERENCE_ANCHORS='[]'
    ;;
  full)
    MAX_STEPS=54000
    QUERY_START=36000
    QUERY_INTERVAL=2000
    DELAY_ENABLED=true
    DELAY_BOUNDARIES='[38000,42000,46000,50000]'
    DELAY_VALUES='[0,2,6,4,0]'
    ARTIFACT_ENABLED=true
    ARTIFACT_ANCHORS='[34000,38000,42000,46000,50000,54000]'
    ARTIFACT_STEPS_CSV='34000,38000,42000,46000,50000,54000'
    REFERENCE_ENABLED=true
    REFERENCE_ANCHORS='[36000,40000,44000,48000,52000,54000]'
    ;;
  *) echo "Unsupported MODE=${MODE}" >&2; exit 2 ;;
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
case "${PARENT_RUN_DIR}" in /*) ;; *) PARENT_RUN_DIR="${SUBMIT_REPO}/${PARENT_RUN_DIR}" ;; esac

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
JOB_SOURCE="${JOB_SOURCE_PARENT}/pendulum-score-screen-${SLURM_JOB_ID:-$$}"
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
if [[ ! -d "${PARENT_RUN_DIR}/checkpoint/${PARENT_CHECKPOINT_STEP}" ]]; then
  echo "Parent checkpoint step missing: ${PARENT_RUN_DIR}/checkpoint/${PARENT_CHECKPOINT_STEP}" >&2
  exit 3
fi
mkdir -p "${RUN_DIR}"
cp -a "${PARENT_RUN_DIR}/checkpoint" "${RUN_DIR}/checkpoint"
python scripts/analyze_pendulum_score_screen.py write-manifest \
  --run-dir "${RUN_DIR}" \
  --run-id "${RUN_ID}" \
  --mode "${MODE}" \
  --profile "${PROFILE}" \
  --commit "${EXPECTED_COMMIT}" \
  --config-hash "${CONFIG_HASH}" \
  --parent-run-dir "${PARENT_RUN_DIR}" \
  --parent-checkpoint-step "${PARENT_CHECKPOINT_STEP}"

echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-unset} GPU=${CUDA_VISIBLE_DEVICES:-unset}"
echo "RUN_ID=${RUN_ID} MODE=${MODE} PROFILE=${PROFILE} VARIANT=${SCORE_VARIANT}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader

PY_ARGS=(
  hydra.run.dir="${RUN_DIR}"
  +run_id="${RUN_ID}"
  +controller=adaptive
  +score_mode="${PROFILE}"
  seed=7
  max_steps="${MAX_STEPS}"
  save_interval_steps="${MAX_STEPS}"
  log_interval_steps=400
  update_chunk_size=128
  collect_chunk_steps=100
  seed_steps_override=4000
  checkpoint_buffer=true
  artifact_capture_enabled="${ARTIFACT_ENABLED}"
  artifact_anchor_enabled="${ARTIFACT_ENABLED}"
  artifact_anchor_steps="${ARTIFACT_ANCHORS}"
  artifact_run_id="${RUN_ID}"
  probe_timing.enabled=false
  reference_probe.enabled="${REFERENCE_ENABLED}"
  reference_probe.anchor_steps="${REFERENCE_ANCHORS}"
  reference_probe.num_env_eval_replicas=128
  reference_probe.env_eval_steps=256
  reference_probe.conditional_reference_env_eval_steps=256
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
  env.mjx_dmc.base_action_delay=0
  env.mjx_dmc.action_delay_schedule_enabled="${DELAY_ENABLED}"
  env.mjx_dmc.action_delay_observation_enabled=false
  env.mjx_dmc.action_delay_schedule_boundaries="${DELAY_BOUNDARIES}"
  env.mjx_dmc.action_delay_schedule_values="${DELAY_VALUES}"
  env.mjx_dmc.allow_hidden_action_delay_schedule=true
  env.mjx_dmc.reset_pool_size=64
  dense_rhs.enabled=true
  dense_rhs.initial_horizon=3
  dense_rhs.start_query_step="${QUERY_START}"
  dense_rhs.query_interval_steps="${QUERY_INTERVAL}"
  dense_rhs.query_recent_transition_steps="${QUERY_INTERVAL}"
  dense_rhs.reset_query_schedule_on_restore=true
  dense_rhs.horizons='[2,3,4,5,6,7,8]'
  dense_rhs.hmax=8
  dense_rhs.horizon_buckets='[]'
  dense_rhs.roughness_probe=projected_jvp
  dense_rhs.num_roughness_probes=8
  dense_rhs.score_mode=additive
  dense_rhs.score_variant="${SCORE_VARIANT}"
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
  dense_rhs.calibrated_return_weight=10.0
  dense_rhs.calibrated_roughness_weight=1.0
  dense_rhs.horizon_switch_cost=0.05
  dense_rhs.return_first_tolerance=5.0
  dense_rhs.roughness_discount=0.99
  dense_rhs.curvature_risk_weight=1.0
  dense_rhs.bellman_risk_weight=0.1
  dense_rhs.curvature_risk_scale=0.01
  dense_rhs.learner_proxy_enabled=false
  dense_rhs.learner_proxy_weight=0.0
  dense_rhs.deployment_utility_enabled=false
  dense_rhs.local_window_radius=0
  dense_rhs.max_transition_delta=0
  dense_rhs.incumbent_switch_margin=0.0
  dense_rhs.credible_transition_enabled=false
  dense_rhs.transition_cost_scale=0.0
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
  dense_rhs.query_population_size=256
  dense_rhs.query_policy_prior_samples=12
  dense_rhs.query_num_elites=32
  dense_rhs.query_mppi_iterations=4
  dense_rhs.query_temperature=0.5
  dense_rhs.prewarm_horizons=true
  dense_rhs.prewarm_all_phases=false
  evaluation.enabled=true
  evaluation.interval_steps=400
  evaluation.num_episodes=10
  evaluation.clean=true
  evaluation.prewarm=false
  tdmpc2.horizon=3
  tdmpc2.population_size=512
  tdmpc2.policy_prior_samples=24
  tdmpc2.num_elites=64
  tdmpc2.mppi_iterations=6
  tdmpc2.temperature=0.5
  wandb.enabled=false
)

python -m tdmpc2_jax.train "${PY_ARGS[@]}"
touch "${RUN_DIR}/TRAINING_COMPLETE"
if [[ "${ARTIFACT_ENABLED}" == "true" ]]; then
  python scripts/render_cartpole_delay_gifs.py \
    "${RUN_DIR}" \
    --run-id "${RUN_ID}" \
    --steps "${ARTIFACT_STEPS_CSV}" \
    --width 240 \
    --height 240 \
    --camera 0 \
    --frame-stride 5 \
    --mujoco-gl egl
  touch "${RUN_DIR}/MEDIA_COMPLETE"
fi
python scripts/analyze_pendulum_score_screen.py validate --run-dir "${RUN_DIR}"
touch "${RUN_DIR}/RUN_VALID"
echo "PENDULUM_SCORE_SCREEN_VALID RUN_DIR=${RUN_DIR}"
