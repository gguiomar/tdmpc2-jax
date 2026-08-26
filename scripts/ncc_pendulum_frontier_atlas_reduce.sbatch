#!/usr/bin/env bash
# Aggregate four validated GPU atlas shards after their Slurm dependency clears.

#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=%x_%j.log

set -euo pipefail

VENV_PATH="${VENV_PATH:-$HOME/.venvs/temporalhorizon-jax}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/pendulum_frontier_atlas}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
ATTEMPT="${ATTEMPT:-1}"

source "${VENV_PATH}/bin/activate"
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR:-$PWD}"
SUBMIT_REPO="$(pwd)"
if [[ "$(git rev-parse HEAD)" != "${EXPECTED_COMMIT}" ]]; then
  echo "Reducer commit mismatch" >&2
  exit 3
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Tracked scientific files are dirty in the isolated reducer checkout" >&2
  exit 3
fi

JOB_SOURCE_PARENT="${SLURM_TMPDIR:-/tmp}"
JOB_SOURCE="${JOB_SOURCE_PARENT}/pendulum-frontier-reducer-${SLURM_JOB_ID:-$$}"
git clone --quiet --shared --no-checkout "${SUBMIT_REPO}" "${JOB_SOURCE}"
git -C "${JOB_SOURCE}" checkout --quiet --detach "${EXPECTED_COMMIT}"
cd "${JOB_SOURCE}"

case "${OUTPUT_ROOT}" in /*) ;; *) OUTPUT_ROOT="${SUBMIT_REPO}/${OUTPUT_ROOT}" ;; esac
RUN_DIRS=()
for shard in 0 1 2 3; do
  RUN_DIRS+=("${OUTPUT_ROOT}/pendatlas__broad_shard${shard}of4__s7/attempt_${ATTEMPT}")
done
python scripts/analyze_pendulum_frontier_atlas.py aggregate \
  --output-dir "${OUTPUT_ROOT}/aggregate/attempt_${ATTEMPT}" \
  "${RUN_DIRS[@]}"
echo "PENDULUM_FRONTIER_ATLAS_AGGREGATE_VALID OUTPUT_ROOT=${OUTPUT_ROOT}"

