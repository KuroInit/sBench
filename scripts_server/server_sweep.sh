#!/usr/bin/env bash
# Standalone-server sweep runner for sBench (equivalent of scripts/nscc_job.pbs).
#
# Usage:
#   bash scripts_server/server_sweep.sh                      # foreground
#   nohup bash scripts_server/server_sweep.sh > server_sweep.log 2>&1 &

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${SCRIPT_DIR}/env.sh"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Could not locate ${ENV_FILE}" >&2
  exit 1
fi
cd "${REPO_DIR}"

echo "=== Job Info ==="
echo "PWD: $(pwd)"
echo "REPO_DIR: ${REPO_DIR}"
echo "ENV_FILE: ${ENV_FILE}"
echo "HOST: $(hostname)"
echo "DATE: $(date)"
echo "================"

source "${ENV_FILE}"

# Virtualenv: created on first run; pinned requirements installed once.
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  PYTHON_BIN="$(command -v python3.12 || command -v python3)"
  echo "=== Creating virtualenv at ${VENV_DIR} with ${PYTHON_BIN} ==="
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
if [[ ! -f "${VENV_DIR}/.sbench-deps" ]]; then
  echo "=== Installing pinned requirements (one-time; large download) ==="
  python -m pip install --upgrade pip
  python -m pip install -r "${REPO_DIR}/requirements.txt"
  touch "${VENV_DIR}/.sbench-deps"
fi

# CUDA toolkit (nvcc) is needed to JIT-compile FlashInfer/Triton kernels.
if command -v nvcc >/dev/null 2>&1; then
  export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(which nvcc)")")}"
  export CUDA_PATH="${CUDA_HOME}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
  export FLASHINFER_CUDA_HOME="${CUDA_HOME}"
else
  echo "[server] WARNING: nvcc not found; kernel JIT may fail for some SGLang paths" >&2
fi

# Avoid stale compiled kernels hardcoding a different CUDA installation.
rm -rf "${HOME}/.cache/flashinfer" "${HOME}/.cache/torch_extensions"

mkdir -p "${HF_HOME}"
mkdir -p "${RESULTS_DIR}"
mkdir -p "${SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR}"
mkdir -p "${SBENCH_SIF_CACHE_DIR}"

echo "=== Harness Env ==="
echo "REPO_DIR=${REPO_DIR}"
echo "VENV_DIR=${VENV_DIR}"
echo "HF_HOME=${HF_HOME}"
echo "RESULTS_DIR=${RESULTS_DIR}"
echo "SWEEP_CONFIG=${SWEEP_CONFIG}"
echo "CHECKPOINT_PATH=${CHECKPOINT_PATH}"
echo "SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=${SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR}"
echo "SBENCH_PREWARM_MINI_SWE=${SBENCH_PREWARM_MINI_SWE}"
echo "SBENCH_GPU_TYPE=${SBENCH_GPU_TYPE:-}"
echo "AUTO_SELECT_GPUS=${AUTO_SELECT_GPUS}"
echo "DISABLE_RADIX_CACHE=${DISABLE_RADIX_CACHE}"
echo "CUDA_HOME=${CUDA_HOME:-}"
echo "dcgmi=$(command -v dcgmi || echo missing)"
echo "==================="

python --version
nvcc --version || true
nvidia-smi || true

if [[ "${HF_TOKEN}" == "REPLACE_WITH_YOUR_HF_TOKEN" ]]; then
  echo "[server] WARNING: HF_TOKEN is not set; gated datasets/models will fail" >&2
fi

if [[ -z "${SBENCH_GPU_TYPE:-}" || "${SBENCH_GPU_TYPE}" == "unknown" ]]; then
  echo "[server] WARNING: SBENCH_GPU_TYPE not detected; set it in scripts_server/env.sh" >&2
fi

if [[ "${SBENCH_PREWARM_MINI_SWE}" == "1" ]]; then
  if command -v docker >/dev/null 2>&1 || command -v singularity >/dev/null 2>&1 || command -v apptainer >/dev/null 2>&1; then
    echo "=== Prewarming mini-SWE container images ==="
    python -m sbench.mini_swe_prewarm --sweep-config "${SWEEP_CONFIG}"
    echo "=== Prewarm complete ==="
  else
    echo "[server] SBENCH_PREWARM_MINI_SWE=1 but no docker/singularity/apptainer runtime found; skipping prewarm" >&2
  fi
fi

python "${REPO_DIR}/orchestrator.py"
python "${REPO_DIR}/analyze.py" "${RESULTS_DIR}"
