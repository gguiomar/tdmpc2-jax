#!/usr/bin/env bash
# One gated launcher: nominal smoke, four-GPU atlas array, CPU aggregation.

set -euo pipefail

EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-outputs/pendulum_h8_torque02/pendh8__base_t1p0_h3__s7/attempt_1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/pendulum_frontier_atlas}"
ATTEMPT="${ATTEMPT:-1}"
VENV_PATH="${VENV_PATH:-$HOME/.venvs/temporalhorizon-jax}"
PYTHON_BIN="${VENV_PATH}/bin/python"
ATLAS_SCRIPT="scripts/ncc_pendulum_frontier_atlas.sbatch"
REDUCE_SCRIPT="scripts/ncc_pendulum_frontier_atlas_reduce.sbatch"

if [[ "$(git rev-parse HEAD)" != "${EXPECTED_COMMIT}" ]]; then
  echo "Launcher commit mismatch" >&2
  exit 3
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Tracked files are dirty; refusing atlas launch" >&2
  git status --short >&2
  exit 3
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing project Python interpreter ${PYTHON_BIN}" >&2
  exit 3
fi
"${PYTHON_BIN}" scripts/analyze_pendulum_h8_torque02.py validate \
  --phase base --run-dir "${SOURCE_RUN_DIR}"

SMOKE_JOB="$(sbatch --parsable \
  --job-name=pendatlas_smoke \
  --export=ALL,MODE=smoke,OUTPUT_ROOT="${OUTPUT_ROOT}",SOURCE_RUN_DIR="${SOURCE_RUN_DIR}",EXPECTED_COMMIT="${EXPECTED_COMMIT}",ATTEMPT="${ATTEMPT}" \
  "${ATLAS_SCRIPT}")"
echo "SMOKE_JOB=${SMOKE_JOB} RUN_ID=pendatlas__smoke_nominal__s7"

ARRAY_JOB="$(sbatch --parsable \
  --dependency=afterok:"${SMOKE_JOB}" \
  --array=0-3%4 \
  --job-name=pendatlas_broad \
  --export=ALL,MODE=full,OUTPUT_ROOT="${OUTPUT_ROOT}",SOURCE_RUN_DIR="${SOURCE_RUN_DIR}",EXPECTED_COMMIT="${EXPECTED_COMMIT}",ATTEMPT="${ATTEMPT}" \
  "${ATLAS_SCRIPT}")"
echo "ATLAS_ARRAY_JOB=${ARRAY_JOB} SHARDS=0-3 DEPENDENCY=afterok:${SMOKE_JOB}"

REDUCE_JOB="$(sbatch --parsable \
  --dependency=afterok:"${ARRAY_JOB}" \
  --job-name=pendatlas_reduce \
  --export=ALL,OUTPUT_ROOT="${OUTPUT_ROOT}",EXPECTED_COMMIT="${EXPECTED_COMMIT}",ATTEMPT="${ATTEMPT}" \
  "${REDUCE_SCRIPT}")"
echo "REDUCE_JOB=${REDUCE_JOB} DEPENDENCY=afterok:${ARRAY_JOB}"

