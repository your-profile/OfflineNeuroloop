#!/bin/bash
#SBATCH --job-name=neuroloop
#SBATCH --output=logs/neuroloop_%A_%a.log
#SBATCH --error=logs/neuroloop_%A_%a.log
#SBATCH --ntasks=1
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
# Time is set at submit time (domain-specific). Default fallback:
#SBATCH --time=4:00:00
#
# One array task = one trial. Submit via submit_hpc.sh (recommended), e.g.:
#   ./submit_hpc.sh manifests/flappy__pretrain__mlp_model_noise.csv
#
# Manual:
#   export MANIFEST=manifests/flappy__pretrain__mlp_model_noise.csv
#   sbatch --time=1:30:00 --array=1-$(($(wc -l < "$MANIFEST") - 1)) run_bash.sh

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

MANIFEST="${MANIFEST:-${SCRIPT_DIR}/trial_manifest.csv}"
if [[ ! -f "${MANIFEST}" ]]; then
  echo "Missing ${MANIFEST}."
  echo "Generate shards: python generate_manifest.py -s configs/sweep_hpc.yaml --shard-by domain_integration_ablation"
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

python3 run_trial.py \
  --manifest "${MANIFEST}" \
  --trial-id "${SLURM_ARRAY_TASK_ID}" \
  "${EXTRA_ARGS[@]}"

echo "Finished at: $(date)"
