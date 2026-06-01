#!/bin/bash
# Submit SLURM array jobs: one task per trial, one manifest shard per sbatch.
#
# Usage:
#   ./submit_hpc.sh --generate-shards          # write manifests/*.csv from sweep_hpc.yaml
#   ./submit_hpc.sh --all                      # submit every shard in manifests/
#   ./submit_hpc.sh manifests/flappy__pretrain__mlp_model_noise.csv
#   ./submit_hpc.sh --dry-run --all
#
# Concurrency cap (optional): ARRAY_CAP=50 ./submit_hpc.sh --all
#
# Optional filters when generating shards:
#   FILTER_GRANULARITY=binary ./submit_hpc.sh --generate-shards
#   FILTER_DOMAIN=Lunar FILTER_INTEGRATION=pretrain ./submit_hpc.sh --generate-shards

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
mkdir -p logs manifests

SWEEP="${SWEEP:-configs/sweep_hpc.yaml}"
SHARD_BY="${SHARD_BY:-domain_integration_ablation_granularity}"
ARRAY_CAP="${ARRAY_CAP:-80}"
PARTITION="${PARTITION:-batch}"

generate_shards() {
  local -a cmd=(python3 generate_manifest.py -s "${SWEEP}" --shard-by "${SHARD_BY}")
  [[ -n "${FILTER_DOMAIN:-}" ]] && cmd+=(--filter-domain ${FILTER_DOMAIN})
  [[ -n "${FILTER_INTEGRATION:-}" ]] && cmd+=(--filter-integration ${FILTER_INTEGRATION})
  [[ -n "${FILTER_ABLATION_KEY:-}" ]] && cmd+=(--filter-ablation-key ${FILTER_ABLATION_KEY})
  [[ -n "${FILTER_CONDITION:-}" ]] && cmd+=(--filter-condition ${FILTER_CONDITION})
  [[ -n "${FILTER_TASK:-}" ]] && cmd+=(--filter-task ${FILTER_TASK})
  [[ -n "${FILTER_GRANULARITY:-}" ]] && cmd+=(--filter-granularity ${FILTER_GRANULARITY})
  "${cmd[@]}"
}

count_trials() {
  local manifest="$1"
  local n
  n=$(($(wc -l < "${manifest}") - 1))
  if [[ "${n}" -lt 1 ]]; then
    echo "No trials in ${manifest}" >&2
    return 1
  fi
  echo "${n}"
}

domain_from_manifest() {
  local base
  base="$(basename "$1" .csv)"
  echo "${base%%__*}"
}

slurm_time_for_manifest() {
  local domain
  domain="$(domain_from_manifest "$1")"
  case "${domain}" in
    flappy) echo "1:30:00" ;;  # ~40 min/trial + buffer
    lunar)  echo "3:00:00" ;;  # ~2 h/trial + buffer
    robot)  echo "8:00:00" ;;  # ~5+ h/trial + buffer
    *)      echo "4:00:00" ;;
  esac
}

submit_one() {
  local manifest="$1"
  local n_tasks time_limit array_spec job_id

  if [[ ! -f "${manifest}" ]]; then
    echo "Missing ${manifest}" >&2
    return 1
  fi

  n_tasks="$(count_trials "${manifest}")"
  time_limit="$(slurm_time_for_manifest "${manifest}")"
  array_spec="1-${n_tasks}"
  if [[ -n "${ARRAY_CAP}" ]]; then
    array_spec="${array_spec}%${ARRAY_CAP}"
  fi

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[dry-run] ${manifest}: ${n_tasks} tasks, time=${time_limit}, array=${array_spec}"
    return 0
  fi

  job_id=$(sbatch \
    --partition="${PARTITION}" \
    --time="${time_limit}" \
    --array="${array_spec}" \
    --export=ALL,MANIFEST="${SCRIPT_DIR}/${manifest}",SKIP_COMPLETED=1 \
    "${SCRIPT_DIR}/run_bash.sh" \
    | awk '{print $NF}')

  echo "Submitted ${manifest} (${n_tasks} trials, ${time_limit}) -> job ${job_id}"
}

if [[ "${1:-}" == "--generate-shards" ]]; then
  generate_shards
  exit 0
fi

if [[ "${1:-}" == "--all" ]]; then
  if [[ ! -d manifests ]] || [[ -z "$(ls -A manifests/*.csv 2>/dev/null)" ]]; then
    echo "No manifests/*.csv — run: ./submit_hpc.sh --generate-shards"
    exit 1
  fi
  for manifest in manifests/*.csv; do
    submit_one "${manifest}"
  done
  exit 0
fi

if [[ "${1:-}" == "--dry-run" && "${2:-}" == "--all" ]]; then
  DRY_RUN=1
  for manifest in manifests/*.csv; do
    submit_one "${manifest}"
  done
  exit 0
fi

if [[ -z "${1:-}" ]]; then
  echo "Pass a manifest path, --all, or --generate-shards" >&2
  exit 1
fi

submit_one "$1"
