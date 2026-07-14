#!/usr/bin/env bash
set -euo pipefail

cd "${SBENCH_WORKSPACE:-/workspace}"

mkdir -p "${RESULTS_DIR:-/workspace/results}"
mkdir -p "${HF_HOME:-/cache/huggingface}"

case "${1:-run}" in
  run)
    python orchestrator.py
    python analyze.py "${RESULTS_DIR:-/workspace/results}"
    ;;
  test)
    python -m pytest -q
    ;;
  shell)
    exec bash
    ;;
  *)
    exec "$@"
    ;;
esac
