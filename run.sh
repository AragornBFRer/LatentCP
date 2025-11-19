#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH=${1:-"${ROOT_DIR}/experiments/configs/gmm_em.yaml"}
TRIALS=${2:-100}
PLOTS_DIR=${3:-"${ROOT_DIR}/experiments/plots"}
RESULTS_PATH=${4:-}
SKIP_PLOTS=${SKIP_PLOTS:-0}

# Avoid MKL / KMeans thread explosions on shared HPC nodes unless caller overrides.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

cmd=(python "${ROOT_DIR}/main.py" --config "${CONFIG_PATH}" --trials "${TRIALS}")
if [[ -n "${RESULTS_PATH}" ]]; then
  cmd+=(--results "${RESULTS_PATH}")
fi
"${cmd[@]}"

if [[ "${SKIP_PLOTS}" != "1" ]]; then
  plot_cmd=(python "${ROOT_DIR}/plot_results.py" --config "${CONFIG_PATH}" --out "${PLOTS_DIR}")
  if [[ -n "${RESULTS_PATH}" ]]; then
    plot_cmd+=(--results "${RESULTS_PATH}")
  fi
  "${plot_cmd[@]}"
else
  echo "Skipping plotting step because SKIP_PLOTS=${SKIP_PLOTS}"
fi
