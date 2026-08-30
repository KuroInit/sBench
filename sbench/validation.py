"""Post-run validation utilities for estimator and profiler comparisons."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .adapters import resolve_adapter
from .estimator import EstimateResult, estimate_records, usable_records
from .hardware import peak_bandwidth_tb, peak_flops_tf
from .moe_cap_estimator import estimate_moe_cap_compatible, support_status
from .trace import moe_activation_note, required_forward_modes


METRICS = (
    "prefill_smfu",
    "prefill_smbu",
    "decoding_smfu",
    "decoding_smbu",
    "prefill_tokens_per_sec",
    "decoding_tokens_per_sec",
    "ttft",
    "tpot",
    "kv_size",
)


@dataclass(frozen=True)
class ValidationResult:
    summary_path: Path
    estimator_comparison_path: Path
    profiler_comparison_path: Path | None = None
    telemetry_comparison_path: Path | None = None


def run_validation(
    results_dir: Path,
    *,
    out_dir: Path,
    profiler_summary: Path | None = None,
    telemetry_summary: Path | None = None,
    sample_limit: int | None = None,
) -> ValidationResult:
    """Validate existing sBench artifacts without rerunning the benchmark."""
    out_dir.mkdir(parents=True, exist_ok=True)
    estimator_rows, run_rows = estimator_comparison_rows(results_dir, sample_limit=sample_limit)
    estimator_path = out_dir / "estimator_comparison.csv"
    _write_csv(estimator_path, estimator_rows)

    profiler_path = None
    if profiler_summary is not None:
        profiler_rows = profiler_comparison_rows(estimator_rows, profiler_summary)
        profiler_path = out_dir / "profiler_comparison.csv"
        _write_csv(profiler_path, profiler_rows)

    telemetry_path = None
    if telemetry_summary is not None:
        telemetry_rows = telemetry_comparison_rows(estimator_rows, telemetry_summary)
        telemetry_path = out_dir / "telemetry_comparison.csv"
        _write_csv(telemetry_path, telemetry_rows)

    summary_path = out_dir / "validation_summary.json"
    summary_path.write_text(json.dumps(summary(run_rows, estimator_rows, profiler_path, telemetry_path), indent=2))
    return ValidationResult(summary_path, estimator_path, profiler_path, telemetry_path)


def estimator_comparison_rows(results_dir: Path, *, sample_limit: int | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    for leaf in leaf_dirs(results_dir):
        dataset, slug, batch_size = parts(results_dir, leaf)
        base = {"dataset": dataset, "slug": slug, "batch_size": batch_size, "leaf": str(leaf)}
        meta_path = latest(leaf, "metadata_*.json")
        records_path = latest(leaf, "server_records_*.jsonl")
        if not meta_path or not records_path:
            run_rows.append({**base, "status": "skipped", "error": "missing metadata or server_records"})
            continue
        try:
            meta = json.loads(meta_path.read_text())
            records = read_jsonl(records_path)
            filtered = usable_records(records)
            limit = sample_limit if sample_limit is not None else metric_sample_limit(meta, dataset)
            if limit is not None:
                filtered = filtered[:limit]
            if not filtered:
                raise ValueError("no usable records")
            required_modes = required_forward_modes(meta.get("dataset_config", {}) or {}, dataset) if meta.get("dataset_config") else set()
            missing_modes = required_modes - {str(record.get("forward_mode")) for record in filtered}
            if missing_modes:
                raise ValueError(f"metric sample is missing required forward mode(s): {', '.join(sorted(missing_modes))}")
            adapter = adapter_from_metadata(meta, filtered)
            component = estimate_records(adapter.descriptor, filtered, components=adapter.components)
            cap_result = None
            cap_status = support_status(adapter.descriptor)
            cap_error = ""
            if cap_status.supported:
                try:
                    cap_result = estimate_moe_cap_compatible(adapter.descriptor, filtered)
                except ValueError as exc:
                    cap_error = str(exc)
            else:
                cap_error = cap_status.reason
            note = moe_activation_note(filtered) if adapter.descriptor.moe.enabled else ""
            run_rows.append({
                **base,
                "status": "success",
                "adapter": adapter.name,
                "records": len(filtered),
                "moe_cap_supported": cap_status.supported,
                "moe_cap_error": cap_error,
                "note": note,
            })
            metric_rows.extend(metric_comparison_rows(base, component, cap_result, cap_error))
        except Exception as exc:
            run_rows.append({**base, "status": "failed", "error": str(exc)})
    return metric_rows, run_rows


def adapter_from_metadata(meta: dict[str, Any], records: list[dict[str, Any]] | None = None):
    model_cfg = meta.get("model_config", {}) or {}
    model = model_cfg.get("model_name", "")
    precision = model_cfg.get("precision", "bfloat16")
    hardware = meta.get("hardware", {}) or {}
    gpu = os.environ.get("ANALYZE_GPU_TYPE") or hardware.get("gpu_type") or "unknown"
    cfg = meta.get("hf_config") or {}
    if not cfg:
        raise ValueError("metadata is missing hf_config")
    overrides = {"architecture": meta.get("architecture_overrides", {}) or {}}
    return resolve_adapter(
        cfg,
        model_name=model,
        overrides=overrides,
        precision_bytes=precision_bytes(precision),
        num_gpus=analysis_num_gpus(hardware, records or []),
        peak_bandwidth_tb=peak_bandwidth_tb(gpu),
        peak_flops_tf=peak_flops_tf(gpu, precision),
    )


def analysis_num_gpus(hardware: dict[str, Any], records: list[dict[str, Any]]) -> int:
    values = [int(record.get("num_gpus") or record.get("gpu_num") or 0) for record in records]
    record_gpus = max(values) if values else 0
    meta_gpus = int(hardware.get("num_gpus", 1) or 1)
    return max(meta_gpus, record_gpus, 1)


def metric_comparison_rows(
    base: dict[str, Any],
    component: EstimateResult,
    moe_cap: EstimateResult | None,
    moe_cap_error: str,
) -> list[dict[str, Any]]:
    rows = []
    component_values = result_values(component)
    cap_values = result_values(moe_cap) if moe_cap else {}
    for metric in METRICS:
        component_value = component_values.get(metric)
        cap_value = cap_values.get(metric)
        rows.append({
            **base,
            "metric": metric,
            "component_wise": component_value,
            "moe_cap": cap_value,
            "ratio_component_over_moe_cap": ratio(component_value, cap_value),
            "delta_pct_component_vs_moe_cap": delta_pct(component_value, cap_value),
            "moe_cap_available": moe_cap is not None,
            "note": "" if moe_cap is not None else moe_cap_error,
        })
    return rows


def profiler_comparison_rows(estimator_rows: list[dict[str, Any]], profiler_summary: Path) -> list[dict[str, Any]]:
    profiler_rows = list(csv.DictReader(profiler_summary.open()))
    profiler_index = {
        (
            row.get("slug", ""),
            str(row.get("batch_size", "")),
            row.get("dataset", ""),
            row.get("phase", ""),
        ): row
        for row in profiler_rows
    }
    out = []
    phase_metric_pairs = {
        "prefill": ("prefill_smfu", "prefill_smbu"),
        "decode": ("decoding_smfu", "decoding_smbu"),
    }
    by_run: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for row in estimator_rows:
        key = (row["slug"], str(row["batch_size"]), row["dataset"])
        by_run.setdefault(key, {})[row["metric"]] = row
    for (slug, batch_size, dataset), metrics in by_run.items():
        for phase, (smfu_metric, smbu_metric) in phase_metric_pairs.items():
            prof = profiler_index.get((slug, batch_size, dataset, phase))
            if not prof:
                continue
            smfu = metrics.get(smfu_metric, {})
            smbu = metrics.get(smbu_metric, {})
            prof_compute = as_float_or_none(prof.get("profiled_smfu") or prof.get("profiled_compute_util"))
            prof_memory = as_float_or_none(prof.get("profiled_smbu") or prof.get("profiled_memory_util"))
            est_smfu = as_float_or_none(smfu.get("component_wise"))
            est_smbu = as_float_or_none(smbu.get("component_wise"))
            out.append({
                "slug": slug,
                "batch_size": batch_size,
                "dataset": dataset,
                "phase": phase,
                "estimated_smfu": est_smfu,
                "profiled_smfu": prof_compute,
                "delta_pct_smfu": delta_pct(est_smfu, prof_compute),
                "estimated_smbu": est_smbu,
                "profiled_smbu": prof_memory,
                "delta_pct_smbu": delta_pct(est_smbu, prof_memory),
            })
    return out


def telemetry_comparison_rows(estimator_rows: list[dict[str, Any]], telemetry_summary: Path) -> list[dict[str, Any]]:
    telemetry_rows = list(csv.DictReader(telemetry_summary.open()))
    telemetry_index = {
        (
            row.get("slug", ""),
            str(row.get("batch_size", "")),
            row.get("dataset", ""),
            row.get("phase", ""),
        ): row
        for row in telemetry_rows
    }
    out = []
    phase_metric_pairs = {
        "prefill": ("prefill_smfu", "prefill_smbu"),
        "decode": ("decoding_smfu", "decoding_smbu"),
    }
    by_run: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for row in estimator_rows:
        key = (row["slug"], str(row["batch_size"]), row["dataset"])
        by_run.setdefault(key, {})[row["metric"]] = row
    for (slug, batch_size, dataset), metrics in by_run.items():
        for phase, (smfu_metric, smbu_metric) in phase_metric_pairs.items():
            observed = telemetry_index.get((slug, batch_size, dataset, phase))
            if not observed:
                continue
            smfu = metrics.get(smfu_metric, {})
            smbu = metrics.get(smbu_metric, {})
            observed_sm = observed_util_pct(
                observed,
                (
                    "observed_smfu",
                    "gpu_util_pct",
                    "sm_util_pct",
                    "sm_active_pct",
                    "DCGM_FI_PROF_SM_ACTIVE",
                    "utilization.gpu [%]",
                    "sm",
                ),
            )
            observed_memory = observed_util_pct(
                observed,
                (
                    "observed_smbu",
                    "memory_util_pct",
                    "mem_util_pct",
                    "dram_util_pct",
                    "dram_active_pct",
                    "DCGM_FI_PROF_DRAM_ACTIVE",
                    "utilization.memory [%]",
                    "mem",
                ),
            )
            component_smfu = as_float_or_none(smfu.get("component_wise"))
            component_smbu = as_float_or_none(smbu.get("component_wise"))
            cap_smfu = as_float_or_none(smfu.get("moe_cap"))
            cap_smbu = as_float_or_none(smbu.get("moe_cap"))
            out.append({
                "slug": slug,
                "batch_size": batch_size,
                "dataset": dataset,
                "phase": phase,
                "observed_sm_util_pct": observed_sm,
                "component_smfu": component_smfu,
                "moe_cap_smfu": cap_smfu,
                "delta_pct_component_smfu_vs_observed": delta_pct(component_smfu, observed_sm),
                "delta_pct_moe_cap_smfu_vs_observed": delta_pct(cap_smfu, observed_sm),
                "observed_memory_util_pct": observed_memory,
                "component_smbu": component_smbu,
                "moe_cap_smbu": cap_smbu,
                "delta_pct_component_smbu_vs_observed": delta_pct(component_smbu, observed_memory),
                "delta_pct_moe_cap_smbu_vs_observed": delta_pct(cap_smbu, observed_memory),
                "note": "lightweight telemetry is coarse; compare trend/magnitude, not exact equality",
            })
    return out


def summary(
    run_rows: list[dict[str, Any]],
    estimator_rows: list[dict[str, Any]],
    profiler_path: Path | None,
    telemetry_path: Path | None,
) -> dict[str, Any]:
    successful_runs = [row for row in run_rows if row.get("status") == "success"]
    comparable = [row for row in estimator_rows if row.get("moe_cap_available")]
    ratios = {}
    for metric in METRICS:
        values = [
            float(row["ratio_component_over_moe_cap"])
            for row in comparable
            if row.get("metric") == metric and row.get("ratio_component_over_moe_cap") not in {None, ""}
        ]
        ratios[metric] = median(values) if values else None
    return {
        "runs_seen": len(run_rows),
        "successful_runs": len(successful_runs),
        "moe_cap_comparable_metric_rows": len(comparable),
        "median_component_over_moe_cap": ratios,
        "profiler_comparison": str(profiler_path) if profiler_path else None,
        "telemetry_comparison": str(telemetry_path) if telemetry_path else None,
    }


def result_values(result: EstimateResult | None) -> dict[str, float]:
    if result is None:
        return {}
    return {
        "prefill_smfu": result.prefill_smfu * 100,
        "prefill_smbu": result.prefill_smbu * 100,
        "decoding_smfu": result.decoding_smfu * 100,
        "decoding_smbu": result.decoding_smbu * 100,
        "prefill_tokens_per_sec": result.prefill_tp,
        "decoding_tokens_per_sec": result.decoding_throughput,
        "ttft": result.ttft,
        "tpot": result.tpot,
        "kv_size": result.kv_size,
    }


def leaf_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    leaves = []
    for path in root.rglob("*"):
        if not path.is_dir() or not latest(path, "metadata_*.json"):
            continue
        rel = path.relative_to(root).parts
        if "validation" in rel:
            continue
        if rel and rel[0] == "component" and root.name != "component":
            continue
        leaves.append(path)
    return leaves


def latest(path: Path, pattern: str) -> Path | None:
    matches = sorted(path.glob(pattern))
    return matches[-1] if matches else None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def parts(root: Path, leaf: Path) -> tuple[str, str, str]:
    rel = leaf.relative_to(root).parts
    slug = rel[0] if len(rel) > 0 else ""
    batch_size = rel[1].removeprefix("bs") if len(rel) > 1 else ""
    dataset = rel[2] if len(rel) > 2 else ""
    return dataset, slug, batch_size


def metric_sample_limit(meta: dict[str, Any], dataset: str) -> int | None:
    cfg = meta.get("dataset_config", {}) or {}
    if cfg.get("runner") != "mini_swe_agent" and dataset != "mini_swe_agent":
        return None
    value = cfg.get("metric_sample_steps", cfg.get("num_samples"))
    if value is None:
        return None
    limit = int(value)
    return limit if limit > 0 else None


def precision_bytes(precision: str) -> float:
    precision = str(precision).lower()
    if "fp8" in precision or "float8" in precision:
        return 1.0
    if "int4" in precision or "4bit" in precision:
        return 0.5
    if "int8" in precision or "8bit" in precision:
        return 1.0
    return 2.0


def ratio(value: Any, baseline: Any) -> float | None:
    value_f = as_float_or_none(value)
    baseline_f = as_float_or_none(baseline)
    if value_f is None or baseline_f in {None, 0.0}:
        return None
    return value_f / baseline_f


def delta_pct(value: Any, baseline: Any) -> float | None:
    value_f = as_float_or_none(value)
    baseline_f = as_float_or_none(baseline)
    if value_f is None or baseline_f in {None, 0.0}:
        return None
    return (value_f - baseline_f) / baseline_f * 100


def as_float_or_none(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def observed_util_pct(row: dict[str, Any], aliases: tuple[str, ...]) -> float | None:
    lowered = {key.lower(): (key, value) for key, value in row.items()}
    for alias in aliases:
        found = lowered.get(alias.lower())
        if not found:
            continue
        key, value = found
        parsed = as_float_or_none(value)
        if parsed is None:
            continue
        return normalize_observed_util(key, parsed)
    return None


def normalize_observed_util(key: str, value: float) -> float:
    key_l = key.lower()
    fraction_like = "dcgm_fi_prof" in key_l or key_l.endswith("_active") or key_l.endswith("_active_pct")
    if fraction_like and 0 <= value <= 1:
        return value * 100
    return value


def median(values: list[float]) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return sorted_values[mid]
    return (sorted_values[mid - 1] + sorted_values[mid]) / 2


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
