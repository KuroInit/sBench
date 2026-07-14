FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    RESULTS_DIR=/workspace/results \
    SWEEP_CONFIG=/workspace/configs/sweep.yaml \
    CHECKPOINT_PATH=/workspace/results/checkpoint.yaml \
    DISABLE_RADIX_CACHE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    docker.io \
    git \
    libnuma1 \
    numactl \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/sbench-venv
ENV PATH=/opt/sbench-venv/bin:$PATH

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install \
      "sglang[all]==0.5.9" \
      transformers \
      datasets \
      pyyaml \
      matplotlib \
      numpy \
      requests \
      aiohttp \
      mini-swe-agent \
      pytest

WORKDIR /workspace
COPY . /workspace

RUN chmod +x /workspace/docker/entrypoint.sh /workspace/scripts/run_sweep.sh

ENTRYPOINT ["/workspace/docker/entrypoint.sh"]
CMD ["run"]
