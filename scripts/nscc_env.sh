#!/usr/bin/env bash
# NSCC Aspire2A environment for sBench.
#
# Source this file from the PBS job after entering the project directory.
# Keep secrets such as HF_TOKEN outside this file; export them before qsub or
# add them in your shell profile if your site policy allows it.

set -euo pipefail

export BASE="${BASE:-/home/users/ntu/ashwin01/scratch}"
export FYP_DIR="${FYP_DIR:-${BASE}/fyp}"
export REPO_DIR="${REPO_DIR:-${FYP_DIR}/sBench}"
export VENV_DIR="${VENV_DIR:-${FYP_DIR}/.venv}"

export HF_HOME="${HF_HOME:-${BASE}/hf_cache}"
export HF_TOKEN="${HF_TOKEN:-}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN}}"
export RESULTS_DIR="${RESULTS_DIR:-${FYP_DIR}/results}"
export SWEEP_CONFIG="${SWEEP_CONFIG:-${REPO_DIR}/configs/sweep.yaml}"
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-${RESULTS_DIR}/checkpoint.yaml}"
export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR="${SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR:-${RESULTS_DIR}/expert_records}"

export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${BASE}/apptainer_cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${BASE}/apptainer_tmp}"
export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-${APPTAINER_CACHEDIR}}"
export SINGULARITY_TMPDIR="${SINGULARITY_TMPDIR:-${APPTAINER_TMPDIR}}"
export SBENCH_SIF_CACHE_DIR="${SBENCH_SIF_CACHE_DIR:-${BASE}/sif_cache}"
export SBENCH_PREWARM_MINI_SWE="${SBENCH_PREWARM_MINI_SWE:-1}"

# Optional local dataset inputs.
export S_MFU_AZURE_CHAT_PATH="${S_MFU_AZURE_CHAT_PATH:-${FYP_DIR}/trace_parsed.jsonl}"
export S_MFU_SHAREGPT_PATH="${S_MFU_SHAREGPT_PATH:-}"
export S_MFU_MMLU_PRO_PATH="${S_MFU_MMLU_PRO_PATH:-}"
export S_MFU_SWEBENCH_PATH="${S_MFU_SWEBENCH_PATH:-}"

export DISABLE_RADIX_CACHE="${DISABLE_RADIX_CACHE:-1}"
export AUTO_SELECT_GPUS="${AUTO_SELECT_GPUS:-0}"
export SBENCH_GPU_TYPE="${SBENCH_GPU_TYPE:-NVIDIA A100-SXM4-40GB}"
export ANALYZE_GPU_TYPE="${ANALYZE_GPU_TYPE:-${SBENCH_GPU_TYPE}}"

# Let the harness import local probe modules inside SGLang worker processes.
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
