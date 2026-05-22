#!/bin/bash
#SBATCH --job-name=neuroloop
#SBATCH --output=logs/neuroloop_%A_%a.log
#SBATCH --error=logs/neuroloop_%A_%a.log
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
# Submit: sbatch --array=1-$(tail -n +2 trial_manifest.csv | wc -l | tr -d ' ') run_bash.sh
# Or cap concurrency: sbatch --array=1-450%50 run_bash.sh

set -euo pipefail

echo "Job Array ID: ${SLURM_ARRAY_JOB_ID:-local}, Task ID: ${SLURM_ARRAY_TASK_ID:-}"
echo "Running on node: $(hostname)"
echo "Started at: $(date)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
mkdir -p logs src/results/runs

module load miniforge/24.11.2-py312 2>/dev/null || true
# source activate <your-env>

if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
  export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
  export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
fi

MANIFEST="${SCRIPT_DIR}/trial_manifest.csv"
if [[ ! -f "${MANIFEST}" ]]; then
  echo "Missing ${MANIFEST}. Generate with: python generate_manifest.py -s configs/sweep_hpc.yaml"
  exit 1
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "Set SLURM_ARRAY_TASK_ID or submit with sbatch --array=1-N"
  exit 1
fi

python3 run_trial.py --manifest "${MANIFEST}" --trial-id "${SLURM_ARRAY_TASK_ID}"

echo "Finished at: $(date)"
