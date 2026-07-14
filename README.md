# sBench

Newer-SGLang lightweight probing plus component-wise S-MFU/S-MBU estimation.

The project mirrors the old `s_mfu` sweep/analyze workflow but does not import
MoE-CAP. Runtime records are collected by `sbench_probe`, then `sbench` assembles
architecture-specific components to estimate utilization.

## Run

Edit the single sweep file:

```text
configs/sweep.yaml
```

Set the datasets under `benchmark_types`, then add the models under `models`.
The harness runs every selected model against every selected dataset and batch
size in that file.

```bash
./run_sweep.sh
```

## Docker

The Docker image pins a CUDA/SGLang/Python runtime and includes `libnuma`, which
avoids host-library issues such as missing `libnuma.so.1`.

Build and test:

```bash
docker compose build
docker compose run --rm sbench test
```

Run a sweep:

```bash
docker compose run --rm --service-ports sbench run
```

Useful environment overrides:

```bash
SBENCH_GPU_TYPE="NVIDIA GeForce RTX 3070 Ti" \
SWEEP_CONFIG=/workspace/configs/sweep.yaml \
docker compose run --rm --service-ports sbench run
```

The compose file mounts:

- the repo at `/workspace` and at `SBENCH_WORKSPACE`
- the Hugging Face cache at `/cache/huggingface`
- `/var/run/docker.sock` for mini-SWE-agent Docker mode

Important environment variables:

- `RESULTS_DIR` output root
- `SWEEP_CONFIG` sweep YAML, defaults to `configs/sweep.yaml`
- `CHECKPOINT_PATH` resume file
- `SBENCH_PROBE_RECORD_PATH` probe JSONL destination, normally set by orchestrator
- `SBENCH_PROFILING_ONLY=1` records timing but sets expert activation to 0
- `S_MFU_SHAREGPT_PATH`, `S_MFU_AZURE_CHAT_PATH`, `S_MFU_SWEBENCH_PATH` dataset overrides
- `S_MFU_MMLU_PRO_PATH` local MMLU-Pro override
- `S_MFU_*_HF_DATASET`, `S_MFU_*_HF_CONFIG`, `S_MFU_*_HF_DATA_FILES` optional Hugging Face dataset overrides
- `SBENCH_GPU_TYPE` or `ANALYZE_GPU_TYPE` hardware normalization key

Runs are strict by default:

- Ordinary request datasets require every request to succeed unless the dataset config sets `min_success_rate`.
- Probe JSONL must be parseable and contain required fields before a run is checkpointed as successful.
- Checkpoints include a signature of the model, TP size, batch size, dataset config, and probe schema.
- Unknown GPU types are reported as failed rows instead of normalized against fake fallback peaks.

## Agentic SWE-bench

For real SWE-bench agent runs, use `mini_swe_agent` instead of the prompt-only
`swe_bench` loader:

```yaml
benchmark_types:
  agentic: [mini_swe_agent]
```

Configure the environment backend in `configs/mini_swe_agent.yaml`:

```yaml
runner: mini_swe_agent
subset: lite
split: dev
workers: 1
environment_class: docker      # or singularity
```

The orchestrator starts the local SGLang server, points mini-SWE-agent at
`http://127.0.0.1:<port>/v1`, and preserves the lightweight probe records for
S-MFU/S-MBU analysis. Install mini-SWE-agent separately with the Docker or
Singularity/Apptainer setup required by your machine.

When running the harness itself in Docker, `environment_class: docker` uses the
host Docker daemon through `/var/run/docker.sock`. That is Docker-outside-of-
Docker: task containers are siblings of the sBench container, not nested inside
it. Because the host daemon interprets bind-mount paths on the host, set
`SBENCH_WORKSPACE` to the same absolute repo path on the host and in the
container. On HPC systems without Docker socket access, keep using
`environment_class: singularity`.

## Outputs

```text
results/<slug>/bs<N>/<dataset>/<model_id>/server_records_*.jsonl
results/<slug>/bs<N>/<dataset>/<model_id>/metadata_*.json
results/raw_values.csv
results/component_breakdown.csv
```
