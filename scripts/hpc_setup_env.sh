#!/bin/bash
# Load miniforge + conda env on Tufts/HPC compute nodes. Sourced by run_bash.sh / run_batch.sh.
set -euo pipefail

_hpc_load_miniforge() {
  local mod candidates=()
  if [[ -n "${MINIFORGE_MODULE:-}" ]]; then
    candidates+=("${MINIFORGE_MODULE}")
  fi
  candidates+=(miniforge/25.3.0 miniforge/24.11.2-py312)
  module purge 2>/dev/null || true
  local m
  for m in "${candidates[@]}"; do
    if module load "${m}" 2>/dev/null; then
      echo "Loaded module: ${m}"
      return 0
    fi
  done
  echo "ERROR: could not load miniforge. Run: module avail miniforge" >&2
  return 1
}

_hpc_activate_conda() {
  local env_name="$1"
  if [[ -z "${env_name}" ]]; then
    echo "ERROR: NEUROLOOP_CONDA_ENV is not set (conda env name, e.g. offline-neuroloop)." >&2
    return 1
  fi
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${env_name}"
  else
    source activate "${env_name}"
  fi
  echo "Active conda env: ${env_name} ($(which python3))"
}

_hpc_check_python_deps() {
  python3 -c "import pandas, yaml, sklearn, torch" 2>/dev/null && return 0
  echo "ERROR: Python env missing dependencies (need pandas, yaml, sklearn, torch)." >&2
  echo "  Create env from environment.yml on the cluster, then:" >&2
  echo "  export NEUROLOOP_CONDA_ENV=offline-neuroloop" >&2
  python3 -c "import pandas, yaml, sklearn, torch"
  return 1
}

_hpc_load_miniforge
_hpc_activate_conda "${NEUROLOOP_CONDA_ENV:-offline-neuroloop}"
_hpc_check_python_deps
