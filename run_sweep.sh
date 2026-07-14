#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results}"
export SWEEP_CONFIG="${SWEEP_CONFIG:-${SCRIPT_DIR}/configs/sweep.yaml}"
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-${RESULTS_DIR}/checkpoint.yaml}"
mkdir -p "${RESULTS_DIR}"
cd "${SCRIPT_DIR}"
python orchestrator.py
python analyze.py "${RESULTS_DIR}"
