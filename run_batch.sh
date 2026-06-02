#!/bin/bash
#SBATCH --job-name=neuroloop_batch
#SBATCH --output=logs/batch_%j.log
#SBATCH --error=logs/batch_%j.log
#SBATCH --ntasks=1
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
# Time is set at submit time via submit_hpc.sh (see BATCH_TIME or auto estimate).
#SBATCH --time=72:00:00
#
# One batch job runs every trial in MANIFEST sequentially.
# Submit: ./submit_hpc.sh manifests/flappy_pretrain_binary.csv

set -euo pipefail

echo "Batch job ID: ${SLURM_JOB_ID:-local}"
echo "Running on node: $(hostname)"
echo "Started at: $(date)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Writable work root on compute nodes (home/repo may be read-only).
WORK_ROOT="${NEUROLOOP_WORK_ROOT:-${SCRATCH:-${SLURM_TMPDIR:-${SCRIPT_DIR}}}/neuroloop_${SLURM_JOB_ID:-local}}"
export NEUROLOOP_WORK_ROOT="${WORK_ROOT}"
mkdir -p "${WORK_ROOT}/logs" "${WORK_ROOT}/src/results/runs"

# Prefer scratch logs; fall back to repo logs/ if writable.
LOG_DIR="${NEUROLOOP_LOG_DIR:-${WORK_ROOT}/logs}"
if mkdir -p "${SCRIPT_DIR}/logs" 2>/dev/null; then
  BATCH_LOG="${SCRIPT_DIR}/logs/batch_${SLURM_JOB_ID:-local}.log"
else
  BATCH_LOG="${LOG_DIR}/batch_${SLURM_JOB_ID:-local}.log"
fi

module load miniforge/24.11.2-py312 2>/dev/null || true
# source activate <your-env>

if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
  export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
  export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
fi

MANIFEST="${MANIFEST:-${SCRIPT_DIR}/trial_manifest.csv}"
if [[ ! -f "${MANIFEST}" ]]; then
  echo "Missing MANIFEST: ${MANIFEST}" | tee -a "${BATCH_LOG}"
  exit 1
fi

N_TRIALS=$(($(wc -l < "${MANIFEST}") - 1))
if [[ "${N_TRIALS}" -lt 1 ]]; then
  echo "No trials in ${MANIFEST}" | tee -a "${BATCH_LOG}"
  exit 1
fi

echo "MANIFEST=${MANIFEST}" | tee -a "${BATCH_LOG}"
echo "N_TRIALS=${N_TRIALS}" | tee -a "${BATCH_LOG}"
echo "WORK_ROOT=${WORK_ROOT}" | tee -a "${BATCH_LOG}"

EXTRA_ARGS=()
if [[ "${SKIP_COMPLETED:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--skip-if-done)
fi

FAILURES=0
for TRIAL_ID in $(seq 1 "${N_TRIALS}"); do
  echo "" | tee -a "${BATCH_LOG}"
  echo "=== Trial ${TRIAL_ID} / ${N_TRIALS} at $(date) ===" | tee -a "${BATCH_LOG}"
  if ! python3 run_trial.py \
    --manifest "${MANIFEST}" \
    --trial-id "${TRIAL_ID}" \
    "${EXTRA_ARGS[@]}" 2>&1 | tee -a "${BATCH_LOG}"; then
    echo "Trial ${TRIAL_ID} failed (exit $?)" | tee -a "${BATCH_LOG}"
    FAILURES=$((FAILURES + 1))
    if [[ "${STOP_ON_ERROR:-0}" == "1" ]]; then
      exit 1
    fi
  fi
done

echo "Finished at: $(date)" | tee -a "${BATCH_LOG}"
echo "Failures: ${FAILURES} / ${N_TRIALS}" | tee -a "${BATCH_LOG}"
[[ "${FAILURES}" -eq 0 ]]
