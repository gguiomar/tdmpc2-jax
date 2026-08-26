#!/usr/bin/env bash
# Submit one restore smoke and six dependent fixed-horizon screen jobs.

set -euo pipefail

EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
PARENT_RUN_DIR="${PARENT_RUN_DIR:-outputs/pendulum_h8_torque02/pendh8__base_t1p0_h3__s7/attempt_1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/pendulum_torque_horizon_screen}"
ATTEMPT="${ATTEMPT:-1}"
SBATCH_SCRIPT="scripts/ncc_pendulum_torque_horizon_screen.sbatch"

if [[ "$(git rev-parse HEAD)" != "${EXPECTED_COMMIT}" ]]; then
  echo "Launcher commit mismatch" >&2
  exit 3
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Tracked files are dirty; refusing screen launch" >&2
  git status --short >&2
  exit 3
fi
python scripts/analyze_pendulum_h8_torque02.py validate \
  --phase base --run-dir "${PARENT_RUN_DIR}"

config_hash() {
  printf 'pendulum-torque-screen-v1|mode=%s|torque=%s|horizon=%s|parent=30000|final=%s|seed=7' \
    "$1" "$2" "$3" "$4" | sha256sum | awk '{print $1}'
}

SMOKE_ID="pendgrid__restore_smoke_t0p6_h8__s7"
SMOKE_DIR="${OUTPUT_ROOT}/${SMOKE_ID}/attempt_${ATTEMPT}"
SMOKE_HASH="$(config_hash smoke 0.6 8 30800)"
SMOKE_JOB="$(sbatch --parsable \
  --job-name=pendgrid_smoke \
  --export=ALL,MODE=smoke,RUN_ID="${SMOKE_ID}",RUN_DIR="${SMOKE_DIR}",PARENT_RUN_DIR="${PARENT_RUN_DIR}",EXPECTED_COMMIT="${EXPECTED_COMMIT}",CONFIG_HASH="${SMOKE_HASH}",TORQUE_SCALE=0.6,FIXED_HORIZON=8 \
  "${SBATCH_SCRIPT}")"
echo "SMOKE_JOB=${SMOKE_JOB} RUN_ID=${SMOKE_ID} CONFIG_HASH=${SMOKE_HASH}"

for torque_code in 0p4 0p6 0p8; do
  case "${torque_code}" in
    0p4) torque=0.4 ;;
    0p6) torque=0.6 ;;
    0p8) torque=0.8 ;;
  esac
  for horizon in 3 8; do
    run_id="pendgrid__t${torque_code}_h${horizon}__s7"
    run_dir="${OUTPUT_ROOT}/${run_id}/attempt_${ATTEMPT}"
    hash="$(config_hash full "${torque}" "${horizon}" 46000)"
    job_id="$(sbatch --parsable \
      --dependency=afterok:"${SMOKE_JOB}" \
      --job-name="pg_t${torque_code}_h${horizon}" \
      --export=ALL,MODE=full,RUN_ID="${run_id}",RUN_DIR="${run_dir}",PARENT_RUN_DIR="${PARENT_RUN_DIR}",EXPECTED_COMMIT="${EXPECTED_COMMIT}",CONFIG_HASH="${hash}",TORQUE_SCALE="${torque}",FIXED_HORIZON="${horizon}" \
      "${SBATCH_SCRIPT}")"
    echo "FULL_JOB=${job_id} RUN_ID=${run_id} TORQUE=${torque} HORIZON=${horizon} CONFIG_HASH=${hash}"
  done
done
