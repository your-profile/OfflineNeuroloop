#!/bin/bash
# Submit SLURM job arrays: one batch (non-interactive) job per manifest, many trials in parallel.
#
# Typical workflow (9+ manifest CSVs, 1000+ trials each):
#   # 1) Edit configs/sweep_hpc.yaml paths.results_path -> your $SCRATCH (then regenerate manifests)
#   export NEUROLOOP_RESULTS_ROOT=$SCRATCH/OfflineNeuroloop_results   # optional override at submit
#   ./submit_hpc.sh --generate-shards
#   ./submit_hpc.sh --dry-run --all
#   ./submit_hpc.sh --all
#
# One manifest:
#   ./submit_hpc.sh manifests/flappy_pretrain_binary.csv
#
# Sequential (slow, one job): SUBMIT_MODE=batch ./submit_hpc.sh manifests/foo.csv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

SWEEP="${SWEEP:-configs/sweep_hpc.yaml}"
SHARD_BY="${SHARD_BY:-domain_integration_ablation_granularity}"
SUBMIT_MODE="${SUBMIT_MODE:-array}"
ARRAY_CAP="${ARRAY_CAP:-50}"
PARTITION="${PARTITION:-batch}"
MAX_BATCH_HOURS="${MAX_BATCH_HOURS:-72}"

# Log directory for SLURM stdout/stderr (must be writable on compute nodes).
resolve_log_dir() {
  if [[ -n "${NEUROLOOP_LOG_DIR:-}" ]]; then
    echo "${NEUROLOOP_LOG_DIR}"
  elif [[ -n "${SCRATCH:-}" ]]; then
    echo "${SCRATCH}/neuroloop_logs"
  elif mkdir -p "${SCRIPT_DIR}/logs" 2>/dev/null; then
    echo "${SCRIPT_DIR}/logs"
  else
    echo "${SCRIPT_DIR}/logs"
  fi
}

LOG_DIR="$(resolve_log_dir)"
mkdir -p "${LOG_DIR}" manifests 2>/dev/null || true

# Pass through to run_bash.sh / run_trial.py on compute nodes.
export_exports() {
  local manifest_abs="$1"
  local parts=()
  parts+=("ALL")
  parts+=("MANIFEST=${manifest_abs}")
  parts+=("SKIP_COMPLETED=${SKIP_COMPLETED:-1}")
  parts+=("NEUROLOOP_LOG_DIR=${LOG_DIR}")
  parts+=("NEUROLOOP_REPO=${SCRIPT_DIR}")
  parts+=("NEUROLOOP_CONDA_ENV=${NEUROLOOP_CONDA_ENV:-offline-neuroloop}")
  parts+=("MINIFORGE_MODULE=${MINIFORGE_MODULE:-miniforge/25.3.0}")
  [[ -n "${NEUROLOOP_RESULTS_ROOT:-}" ]] && parts+=("NEUROLOOP_RESULTS_ROOT=${NEUROLOOP_RESULTS_ROOT}")
  [[ -n "${NEUROLOOP_DATA_ROOT:-}" ]] && parts+=("NEUROLOOP_DATA_ROOT=${NEUROLOOP_DATA_ROOT}")
  [[ -n "${SCRATCH:-}" ]] && parts+=("SCRATCH=${SCRATCH}")
  (IFS=,; echo "${parts[*]}")
}

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

minutes_per_trial() {
  case "$(domain_from_manifest "$1")" in
    flappy) echo 40 ;;
    lunar)  echo 120 ;;
    robot)  echo 300 ;;
    *)      echo 60 ;;
  esac
}

slurm_time_per_trial() {
  case "$(domain_from_manifest "$1")" in
    flappy) echo "1:30:00" ;;
    lunar)  echo "3:00:00" ;;
    robot)  echo "8:00:00" ;;
    *)      echo "4:00:00" ;;
  esac
}

# Rough wall-clock if ARRAY_CAP tasks run in parallel until done.
estimate_array_wall_hours() {
  local manifest="$1"
  local n mins cap waves
  n="$(count_trials "${manifest}")"
  mins="$(minutes_per_trial "${manifest}")"
  cap="${ARRAY_CAP:-50}"
  if [[ "${cap}" -lt 1 ]]; then
    cap=1
  fi
  waves=$(( (n + cap - 1) / cap ))
  echo $(( waves * mins / 60 ))
}

estimate_batch_time() {
  local manifest="$1"
  local n mins total_min hours rem
  n="$(count_trials "${manifest}")"
  mins="$(minutes_per_trial "${manifest}")"
  total_min=$((n * mins))
  hours=$((total_min / 60))
  rem=$((total_min % 60))
  if [[ "${hours}" -gt "${MAX_BATCH_HOURS}" ]]; then
    hours="${MAX_BATCH_HOURS}"
    rem=0
  fi
  printf '%d:%02d:00' "${hours}" "${rem}"
}

submit_array() {
  local manifest="$1"
  local manifest_abs n_tasks time_limit array_spec job_id wall_h

  if [[ ! -f "${manifest}" ]]; then
    echo "Missing ${manifest}" >&2
    return 1
  fi

  manifest_abs="${SCRIPT_DIR}/${manifest}"
  [[ "${manifest}" == /* ]] && manifest_abs="${manifest}"

  n_tasks="$(count_trials "${manifest}")"
  time_limit="$(slurm_time_per_trial "${manifest}")"
  wall_h="$(estimate_array_wall_hours "${manifest}")"
  array_spec="1-${n_tasks}"
  if [[ -n "${ARRAY_CAP}" ]]; then
    array_spec="${array_spec}%${ARRAY_CAP}"
  fi

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[dry-run] array ${manifest}: ${n_tasks} trials, --array=${array_spec}, --time=${time_limit}/task, ~${wall_h}h wall"
    return 0
  fi

  job_id=$(sbatch \
    --job-name="nl_$(basename "${manifest}" .csv | tr '/' '_' | cut -c1-24)" \
    --partition="${PARTITION}" \
    --time="${time_limit}" \
    --array="${array_spec}" \
    --output="${LOG_DIR}/neuroloop_%A_%a.log" \
    --error="${LOG_DIR}/neuroloop_%A_%a.log" \
    --export="$(export_exports "${manifest_abs}")" \
    "${SCRIPT_DIR}/run_bash.sh" \
    | awk '{print $NF}')

  echo "Submitted array ${manifest}"
  echo "  job_id=${job_id}  trials=${n_tasks}  array=${array_spec}  time=${time_limit}/task  ~${wall_h}h wall"
  echo "  logs: ${LOG_DIR}/neuroloop_${job_id}_*.log"
}

submit_batch() {
  local manifest="$1"
  local manifest_abs n_tasks time_limit job_id

  if [[ ! -f "${manifest}" ]]; then
    echo "Missing ${manifest}" >&2
    return 1
  fi

  manifest_abs="${SCRIPT_DIR}/${manifest}"
  [[ "${manifest}" == /* ]] && manifest_abs="${manifest}"

  n_tasks="$(count_trials "${manifest}")"
  time_limit="${BATCH_TIME:-$(estimate_batch_time "${manifest}")}"

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[dry-run] batch ${manifest}: ${n_tasks} trials sequential, time=${time_limit}"
    return 0
  fi

  job_id=$(sbatch \
    --job-name="nl_batch_$(basename "${manifest}" .csv | cut -c1-16)" \
    --partition="${PARTITION}" \
    --time="${time_limit}" \
    --output="${LOG_DIR}/batch_%j.log" \
    --error="${LOG_DIR}/batch_%j.log" \
    --export="$(export_exports "${manifest_abs}")",STOP_ON_ERROR="${STOP_ON_ERROR:-0}" \
    "${SCRIPT_DIR}/run_batch.sh" \
    | awk '{print $NF}')

  echo "Submitted batch ${manifest} (${n_tasks} trials sequential) -> job ${job_id}"
  echo "  log: ${LOG_DIR}/batch_${job_id}.log"
}

submit_one() {
  if [[ "${SUBMIT_MODE}" == "batch" ]]; then
    submit_batch "$1"
  else
    submit_array "$1"
  fi
}

if [[ "${1:-}" == "--generate-shards" ]]; then
  generate_shards
  exit 0
fi

if [[ "${1:-}" == "--dry-run" && "${2:-}" == "--all" ]] || [[ "${1:-}" == "--all" && "${DRY_RUN:-0}" == "1" ]]; then
  DRY_RUN=1
  echo "SUBMIT_MODE=${SUBMIT_MODE}  ARRAY_CAP=${ARRAY_CAP}  LOG_DIR=${LOG_DIR}"
  [[ -n "${NEUROLOOP_RESULTS_ROOT:-}" ]] && echo "NEUROLOOP_RESULTS_ROOT=${NEUROLOOP_RESULTS_ROOT}"
  for manifest in manifests/*.csv; do
    submit_one "${manifest}"
  done
  exit 0
fi

if [[ "${1:-}" == "--all" ]]; then
  if [[ ! -d manifests ]] || [[ -z "$(ls -A manifests/*.csv 2>/dev/null)" ]]; then
    echo "No manifests/*.csv — run: ./submit_hpc.sh --generate-shards"
    exit 1
  fi
  echo "Submitting ${SUBMIT_MODE} jobs (ARRAY_CAP=${ARRAY_CAP}) for manifests/*.csv"
  echo "LOG_DIR=${LOG_DIR}"
  for manifest in manifests/*.csv; do
    submit_one "${manifest}"
  done
  exit 0
fi

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  if [[ -z "${2:-}" ]]; then
    echo "Pass a manifest path with --dry-run" >&2
    exit 1
  fi
  submit_one "$2"
  exit 0
fi

if [[ -z "${1:-}" ]]; then
  echo "Usage:" >&2
  echo "  ./submit_hpc.sh --generate-shards" >&2
  echo "  ./submit_hpc.sh --dry-run --all" >&2
  echo "  ./submit_hpc.sh --all" >&2
  echo "  ./submit_hpc.sh manifests/<shard>.csv" >&2
  echo "" >&2
  echo "Default: SUBMIT_MODE=array, ARRAY_CAP=50 (parallel trials via sbatch --array)." >&2
  echo "Set results_path in configs/sweep_hpc.yaml to \$SCRATCH before generating manifests." >&2
  exit 1
fi

submit_one "$1"
