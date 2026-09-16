#!/usr/bin/env bash
# Standalone-server environment for sBench (equivalent of scripts/nscc_env.sh).
#
# Source this file from scripts_server/server_sweep.sh. Everything defaults to
# locations under SBENCH_SERVER_BASE (default: ${HOME}/sbench_data); override
# any variable in the environment before sourcing.

set -euo pipefail

# Layout on this server: ~/ashwin/{sBench, hf_cache, ...}
export BASE="${SBENCH_SERVER_BASE:-/export/home/yulin/ashwin}"
export RUN_DIR="${BASE}/sbench-run"
export REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VENV_DIR="${RUN_DIR}/.venv"

# Python environment: existing conda env (named ".venv" in miniforge3).
# server_sweep.sh activates it via SBENCH_CONDA_BASE/etc/profile.d/conda.sh.
export SBENCH_CONDA_ENV="${SBENCH_CONDA_ENV:-.venv}"
export SBENCH_CONDA_BASE="${SBENCH_CONDA_BASE:-/export/home/yulin/miniforge3}"

export HF_HOME="${BASE}/hf_cache"
export HF_TOKEN="${HF_TOKEN:-REPLACE_WITH_YOUR_HF_TOKEN}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
export RESULTS_DIR="${RUN_DIR}/results"
export SWEEP_CONFIG="${SWEEP_CONFIG:-${REPO_DIR}/configs/sweep.yaml}"
export CHECKPOINT_PATH="${RESULTS_DIR}/checkpoint.yaml"
export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR="${RESULTS_DIR}/expert_records"

export SBENCH_SIF_CACHE_DIR="${BASE}/sif_cache"
export SBENCH_PREWARM_MINI_SWE="${SBENCH_PREWARM_MINI_SWE:-1}"

# Optional local dataset inputs (empty = fall back to Hugging Face datasets).
export S_MFU_AZURE_CHAT_PATH=""
export S_MFU_SHAREGPT_PATH=""
export S_MFU_MMLU_PRO_PATH=""
export S_MFU_SWEBENCH_PATH=""

export DISABLE_RADIX_CACHE="1"
export AUTO_SELECT_GPUS="0"

# GPU type: auto-detected from nvidia-smi. The raw nvidia-smi name must
# normalize through the alias table in sbench/hardware.py (e.g.
# "NVIDIA H100 80GB HBM3" -> NVIDIA-H100-HBM3-80GB). Override this in the
# environment if your card is missing from that table; unknown GPU types are
# reported as failed rows by the estimator (no fake fallback peaks).
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | sed -e 's/[[:space:]]*$//' || true)"
export SBENCH_GPU_TYPE="${SBENCH_GPU_TYPE:-${GPU_NAME}}"
export ANALYZE_GPU_TYPE="${SBENCH_GPU_TYPE}"

# Let the harness import local probe modules inside SGLang worker processes.
export PYTHONPATH="${REPO_DIR}"
