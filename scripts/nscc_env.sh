#!/usr/bin/env bash
# NSCC Aspire2A environment for sBench.
#
# Source this file from the PBS job after entering the project directory.
# Sourcing this file sets the values below exactly as written.

set -euo pipefail

export BASE="/home/users/ntu/ashwin01/scratch"
export FYP_DIR="${BASE}/fyp"
export REPO_DIR="${FYP_DIR}/sBench"
export VENV_DIR="${FYP_DIR}/.venv"

export HF_HOME="${BASE}/hf_cache"
export HF_TOKEN="REPLACE_WITH_YOUR_HF_TOKEN"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
export RESULTS_DIR="${FYP_DIR}/results"
export SWEEP_CONFIG="${REPO_DIR}/configs/sweep.yaml"
export CHECKPOINT_PATH="${RESULTS_DIR}/checkpoint.yaml"
export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR="${RESULTS_DIR}/expert_records"

export APPTAINER_CACHEDIR="${BASE}/apptainer_cache"
export APPTAINER_TMPDIR="${BASE}/apptainer_tmp"
export SINGULARITY_CACHEDIR="${APPTAINER_CACHEDIR}"
export SINGULARITY_TMPDIR="${APPTAINER_TMPDIR}"
export SBENCH_SIF_CACHE_DIR="${BASE}/sif_cache"
export SBENCH_PREWARM_MINI_SWE="1"

# Optional local dataset inputs.
export S_MFU_AZURE_CHAT_PATH="${FYP_DIR}/trace_parsed.jsonl"
export S_MFU_SHAREGPT_PATH=""
export S_MFU_MMLU_PRO_PATH=""
export S_MFU_SWEBENCH_PATH=""

export DISABLE_RADIX_CACHE="1"
export AUTO_SELECT_GPUS="0"
export SBENCH_GPU_TYPE="NVIDIA A100-SXM4-40GB"
export ANALYZE_GPU_TYPE="${SBENCH_GPU_TYPE}"

# Let the harness import local probe modules inside SGLang worker processes.
export PYTHONPATH="${REPO_DIR}"
