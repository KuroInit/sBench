# sBench

Newer-SGLang lightweight probing plus component-wise S-MFU/S-MBU estimation.

The project mirrors the old `s_mfu` sweep/analyze workflow but does not import
MoE-CAP. Runtime records are collected by `sbench_probe`, then `sbench` assembles
architecture-specific components to estimate utilization.

## Layout

```text
configs/        User-editable sweep and dataset configs
docs/           Design notes and planning references
docker/         Container entrypoint scripts
scripts/        Local and NSCC run scripts
sbench/         Dataset loaders, estimator, adapters, and runners
sbench_probe/   SGLang probe and entrypoint
tests/          Unit tests
```

## Run

Edit the single sweep file:

```text
configs/sweep.yaml
```

Set the datasets under `benchmark_types`, then add the models under `models`.
The harness runs every selected model against every selected dataset and batch
size in that file.

Model architecture is loaded from the model `config.json` through
`transformers.AutoConfig` using the model id/path in `models[].id`. Do not copy
full `hf_config` blocks into `configs/sweep.yaml`; the run fails if `config.json`
cannot be loaded. Use model `architecture` overrides only for fields that are
missing, renamed, or intentionally corrected.

```bash
./scripts/run_sweep.sh
```

On NSCC, submit the PBS script from the repo root:

```bash
qsub scripts/nscc_job.pbs
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
issue_count: 1
model_class: litellm_textbased
mini_swe_configs: ["swebench.yaml", "swebench_xml"]
```

The orchestrator starts the local SGLang server, points mini-SWE-agent at
`http://127.0.0.1:<port>/v1`, and preserves the lightweight probe records for
S-MFU/S-MBU analysis. Install mini-SWE-agent separately with the Docker or
Singularity/Apptainer setup required by your machine.

mini-SWE-agent v2 uses native tool calls by default. Local SGLang-served models
can emit textual action blocks instead of native OpenAI `tool_calls`; set
`model_class: litellm_textbased` with `mini_swe_configs: ["swebench.yaml",
"swebench_xml"]` or another matching text parser config.

Batch-mode mini-SWE-agent writes `preds.json`. sBench requires one nonempty
`model_patch` submission per selected issue before accepting the run. This
validates that the agent submitted a patch, not that the patch solves the issue:
run the official SWE-Bench evaluator separately when correctness is required.
For Singularity runs, prewarming derives its image selection from the same
`issue_count`, `instance_ids`, or explicit `prewarm_slice` setting as the agent
run.

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

## Post-Run Validation

After a sweep finishes, validate estimator behavior from the saved artifacts
without rerunning SGLang or any dataset:

```bash
python validate_estimator.py results
```

The validator reads `metadata_*.json` and `server_records_*.jsonl`, recomputes
component-wise estimates, compares them with MoE-CAP-compatible estimates where
the architecture supports it, and writes:

```text
results/validation/estimator_comparison.csv
results/validation/validation_summary.json
```

For lightweight DCGM or `nvidia-smi` validation, collect a small aggregate CSV
per run/phase and pass it in:

```bash
python validate_estimator.py results --telemetry-summary telemetry_summary.csv
```

Expected telemetry columns are `slug,batch_size,dataset,phase` plus one compute
utilization column and one memory utilization column. Supported names include:

```text
gpu_util_pct, sm_util_pct, sm_active_pct, DCGM_FI_PROF_SM_ACTIVE, sm
memory_util_pct, mem_util_pct, dram_util_pct, DCGM_FI_PROF_DRAM_ACTIVE, mem
```

The output is:

```text
results/validation/telemetry_comparison.csv
```

This is a coarse validation layer. It is useful for trend and magnitude checks,
but it is not expected to equal estimator S-MFU/S-MBU exactly because it includes
runtime stalls, scheduling gaps, communication, CUDA graph behavior, and kernel
implementation details.

To compare against Nsight or other profiler measurements, provide a profiler
summary CSV:

```bash
python validate_estimator.py results --profiler-summary profiler_summary.csv
```

Expected profiler columns are `slug,batch_size,dataset,phase,profiled_smfu,profiled_smbu`.
Use `--sample-limit N` for sampled checks, especially when validating long
mini-SWE-agent runs.

Workload configs use `num_samples` for request count, except mini-SWE-agent,
where `metric_sample_steps` is the required number of usable probe records.
`prefix_cache: true` retains SGLang radix caching for agentic tool loops;
chat, reasoning, and batched-prefill configs disable it so each request is
measured independently. ShareGPT can set `max_input_tokens`; sBench applies
the model chat template and truncates oldest turns at an exact token boundary.
MMLU-Pro refuses an overlong prompt instead of truncating away the question.
