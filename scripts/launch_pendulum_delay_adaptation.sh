#!/usr/bin/env bash
# One gated launcher: adaptive GPU/render smoke, three matched full runs, reducer.

set -euo pipefail

EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
PARENT_RUN_DIR="${PARENT_RUN_DIR:-outputs/pendulum_h8_torque02/pendh8__base_t1p0_h3__s7/attempt_1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/pendulum_delay_adaptation}"
ATTEMPT="${ATTEMPT:-1}"
VENV_PATH="${VENV_PATH:-$HOME/.venvs/temporalhorizon-jax}"
PYTHON_BIN="${VENV_PATH}/bin/python"
SBATCH_SCRIPT=scripts/ncc_pendulum_delay_adaptation.sbatch
REDUCE_SCRIPT=scripts/ncc_pendulum_delay_adaptation_reduce.sbatch

if [[ "$(git rev-parse HEAD)" != "${EXPECTED_COMMIT}" ]]; then
  echo "Launcher commit mismatch" >&2
  exit 3
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Tracked files are dirty; refusing launch" >&2
  git status --short >&2
  exit 3
fi
"${PYTHON_BIN}" scripts/analyze_pendulum_h8_torque02.py validate \
  --phase base --run-dir "${PARENT_RUN_DIR}"

config_hash() {
  printf 'pendulum-delay-adaptation-v1|mode=%s|profile=%s|source=30000|final=%s|delay=%s:%s:4|readout=400|seed=7|commit=%s' \
    "$1" "$2" "$3" "$4" "$5" "${EXPECTED_COMMIT}" | sha256sum | awk '{print $1}'
}

SMOKE_ID=penddelay__smoke_adaptive__s7
SMOKE_DIR="${OUTPUT_ROOT}/${SMOKE_ID}/attempt_${ATTEMPT}"
SMOKE_HASH="$(config_hash smoke adaptive 32000 30800 31600)"
SMOKE_JOB="$(sbatch --parsable \
  --job-name=penddelay_smoke \
  --export=ALL,MODE=smoke,PROFILE=adaptive,RUN_ID="${SMOKE_ID}",RUN_DIR="${SMOKE_DIR}",PARENT_RUN_DIR="${PARENT_RUN_DIR}",EXPECTED_COMMIT="${EXPECTED_COMMIT}",CONFIG_HASH="${SMOKE_HASH}" \
  "${SBATCH_SCRIPT}")"
echo "SMOKE_JOB=${SMOKE_JOB} RUN_ID=${SMOKE_ID} CONFIG_HASH=${SMOKE_HASH}"

FULL_JOBS=()
for profile in fixed_h3 fixed_h7 adaptive; do
  case "${profile}" in
    fixed_h3) run_id=penddelay__fixed_h3__s7; job_name=penddelay_h3 ;;
    fixed_h7) run_id=penddelay__fixed_h7__s7; job_name=penddelay_h7 ;;
    adaptive) run_id=penddelay__adaptive_h2to8__s7; job_name=penddelay_adapt ;;
  esac
  run_dir="${OUTPUT_ROOT}/${run_id}/attempt_${ATTEMPT}"
  hash="$(config_hash full "${profile}" 46000 34000 42000)"
  job_id="$(sbatch --parsable \
    --dependency=afterok:"${SMOKE_JOB}" \
    --job-name="${job_name}" \
    --export=ALL,MODE=full,PROFILE="${profile}",RUN_ID="${run_id}",RUN_DIR="${run_dir}",PARENT_RUN_DIR="${PARENT_RUN_DIR}",EXPECTED_COMMIT="${EXPECTED_COMMIT}",CONFIG_HASH="${hash}" \
    "${SBATCH_SCRIPT}")"
  FULL_JOBS+=("${job_id}")
  echo "FULL_JOB=${job_id} PROFILE=${profile} RUN_ID=${run_id} CONFIG_HASH=${hash}"
done

DEPENDENCY="$(IFS=:; echo "${FULL_JOBS[*]}")"
REDUCE_JOB="$(sbatch --parsable \
  --dependency=afterok:"${DEPENDENCY}" \
  --job-name=penddelay_reduce \
  --export=ALL,OUTPUT_ROOT="${OUTPUT_ROOT}",ATTEMPT="${ATTEMPT}",EXPECTED_COMMIT="${EXPECTED_COMMIT}" \
  "${REDUCE_SCRIPT}")"
echo "REDUCE_JOB=${REDUCE_JOB} DEPENDENCY=afterok:${DEPENDENCY}"
