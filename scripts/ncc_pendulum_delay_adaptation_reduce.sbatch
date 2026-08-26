#!/usr/bin/env bash

#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.log

set -euo pipefail

VENV_PATH="${VENV_PATH:-$HOME/.venvs/temporalhorizon-jax}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/pendulum_delay_adaptation}"
ATTEMPT="${ATTEMPT:-1}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"

source "${VENV_PATH}/bin/activate"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
cd "${SLURM_SUBMIT_DIR:-$PWD}"
if [[ "$(git rev-parse HEAD)" != "${EXPECTED_COMMIT}" ]]; then
  echo "Reducer commit mismatch" >&2
  exit 3
fi

python scripts/analyze_pendulum_delay_adaptation.py aggregate \
  --output-dir "${OUTPUT_ROOT}/aggregate/attempt_${ATTEMPT}" \
  "${OUTPUT_ROOT}/penddelay__fixed_h3__s7/attempt_${ATTEMPT}" \
  "${OUTPUT_ROOT}/penddelay__fixed_h7__s7/attempt_${ATTEMPT}" \
  "${OUTPUT_ROOT}/penddelay__adaptive_h2to8__s7/attempt_${ATTEMPT}"
echo "PENDULUM_DELAY_ADAPTATION_AGGREGATE_VALID"
