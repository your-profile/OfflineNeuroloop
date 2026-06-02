#!/bin/bash
#SBATCH --job-name=neuroloop
#SBATCH --ntasks=1
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
# Time and log paths are set by submit_hpc.sh at sbatch time.
#SBATCH --time=4:00:00
#
# One array task = one trial. Submit via submit_hpc.sh (default: parallel array).
#   ./submit_hpc.sh manifests/flappy_pretrain_binary.csv

set -euo pipefail

echo "Job Array ID: ${SLURM_ARRAY_JOB_ID:-local}, Task ID: ${SLURM_ARRAY_TASK_ID:-}"
echo "Running on node: $(hostname)"
echo "Started at: $(date)"

# SLURM may execute a copy of this script under /var/spool/slurm/job<id>/.
if [[ -n "${NEUROLOOP_REPO:-}" ]]; then
  REPO_DIR="${NEUROLOOP_REPO}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_DIR="${SLURM_SUBMIT_DIR}"
else
  REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "${REPO_DIR}"
echo "REPO_DIR=${REPO_DIR}"

# Writable work root on compute nodes (project dir may be read-only on workers).
WORK_ROOT="${NEUROLOOP_WORK_ROOT:-${SCRATCH:-${SLURM_TMPDIR:-}}}"
if [[ -n "${WORK_ROOT}" ]]; then
  export NEUROLOOP_WORK_ROOT="${WORK_ROOT}/neuroloop_${SLURM_JOB_ID:-local}"
  mkdir -p "${NEUROLOOP_WORK_ROOT}/src/results/runs"
  export NEUROLOOP_RESULTS_ROOT="${NEUROLOOP_RESULTS_ROOT:-${NEUROLOOP_WORK_ROOT}}"
fi

# Logs/results under scratch when set; else try repo (login node / writable clones).
if [[ -n "${NEUROLOOP_LOG_DIR:-}" ]]; then
  mkdir -p "${NEUROLOOP_LOG_DIR}"
elif mkdir -p "${REPO_DIR}/logs" 2>/dev/null; then
  :
else
  echo "WARNING: could not create logs under repo; set NEUROLOOP_LOG_DIR or SCRATCH." >&2
fi

module load miniforge/24.11.2-py312 2>/dev/null || true
# source activate <your-env>

if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
  export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
  export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
fi

MANIFEST="${MANIFEST:-${REPO_DIR}/trial_manifest.csv}"
if [[ ! -f "${MANIFEST}" ]]; then
  echo "Missing ${MANIFEST}."
  exit 1
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "Submit with SLURM array, e.g.: ./submit_hpc.sh manifests/<shard>.csv"
  exit 1
fi

EXTRA_ARGS=()
if [[ "${SKIP_COMPLETED:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--skip-if-done)
fi

python3 "${REPO_DIR}/run_trial.py" \
  --manifest "${MANIFEST}" \
  --trial-id "${SLURM_ARRAY_TASK_ID}" \
  "${EXTRA_ARGS[@]}"

echo "Finished at: $(date)"
