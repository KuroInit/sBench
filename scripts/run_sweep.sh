#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/results}"
export SWEEP_CONFIG="${SWEEP_CONFIG:-${REPO_ROOT}/configs/sweep.yaml}"
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-${RESULTS_DIR}/checkpoint.yaml}"
mkdir -p "${RESULTS_DIR}"
cd "${REPO_ROOT}"
python orchestrator.py
python analyze.py "${RESULTS_DIR}"
