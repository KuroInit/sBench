#!/usr/bin/env python3
"""Analyze sBench probe outputs into raw_values.csv."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

from sbench.adapters import resolve_adapter
from sbench.estimator import estimate_component_breakdown, estimate_records, usable_records
from sbench.moe_cap_estimator import estimate_moe_cap_compatible
from sbench.hardware import peak_bandwidth_tb, peak_flops_tf


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python analyze.py <RESULTS_DIR>", file=sys.stderr)
        raise SystemExit(1)
    results_dir = Path(sys.argv[1])
    sweep_config = _load_sweep_config()
    estimator_mode = _estimator_mode(sweep_config)
    rows = []
    breakdown_rows = []
    for leaf in _leaf_dirs(results_dir):
        failure = _latest(leaf, "failure_*.json")
        meta_path = _latest(leaf, "metadata_*.json")
        records_path = _latest(leaf, "server_records_*.jsonl")
        if failure and (not meta_path or failure.stat().st_mtime >= meta_path.stat().st_mtime):
            rows.append(_failure_row(failure))
            continue
        if not meta_path or not records_path:
            if failure:
                rows.append(_failure_row(failure))
            continue
        meta = json.loads(meta_path.read_text())
        if "dataset_config" not in meta:
            dataset_for_cfg = _parts(results_dir, leaf)[0]
            recovered_cfg = _load_effective_dataset_config(leaf, dataset_for_cfg)
            if recovered_cfg:
                meta["dataset_config"] = recovered_cfg
        try:
            records = _read_jsonl(records_path)
        except Exception as exc:
            rows.append(_synthetic_failure_row(leaf, results_dir, f"invalid server_records JSONL: {exc}"))
            continue
        if not records:
            if failure:
                rows.append(_failure_row(failure))
            continue
        dataset, slug, bs = _parts(results_dir, leaf)
        filtered_records = usable_records(records)
        sample_limit = _metric_sample_limit(meta, dataset)
        if sample_limit is not None:
            filtered_records = filtered_records[:sample_limit]
        if not filtered_records:
            rows.append(_synthetic_failure_row(leaf, results_dir, "no usable probe records"))
            continue
        model = meta["model_config"]["model_name"]
        precision = meta["model_config"].get("precision", "bfloat16")
        gpu = os.environ.get("ANALYZE_GPU_TYPE") or meta.get("hardware", {}).get("gpu_type") or records[0].get("gpu_raw_type")
        overrides = {"architecture": meta.get("architecture_overrides", {})}
        cfg = meta.get("hf_config", {})
        if not cfg:
            rows.append(_synthetic_failure_row(leaf, results_dir, "model config.json is required but missing from metadata"))
            continue
        try:
            adapter = resolve_adapter(
                cfg,
                model_name=model,
                overrides=overrides,
                precision_bytes=_precision_bytes(precision),
                num_gpus=int(meta.get("hardware", {}).get("num_gpus", 1)),
                peak_bandwidth_tb=peak_bandwidth_tb(gpu),
                peak_flops_tf=peak_flops_tf(gpu, precision),
            )
            if estimator_mode == "moe-cap":
                result = estimate_moe_cap_compatible(adapter.descriptor, filtered_records)
                estimator_used = "moe-cap"
            else:
                result = estimate_records(adapter.descriptor, filtered_records, components=adapter.components)
                estimator_used = "component-wise"
        except Exception as exc:
            rows.append(_synthetic_failure_row(leaf, results_dir, str(exc)))
            continue
        rows.append({
            "dataset": dataset,
            "slug": slug,
            "batch_size": bs,
            "adapter": adapter.name,
            "estimator": estimator_used,
            "prefill_tokens_per_sec": result.prefill_tp,
            "decoding_tokens_per_sec": result.decoding_throughput,
            "prefill_smfu": result.prefill_smfu * 100,
            "prefill_smbu": result.prefill_smbu * 100,
            "decoding_smfu": result.decoding_smfu * 100,
            "decoding_smbu": result.decoding_smbu * 100,
            "ttft": result.ttft,
            "tpot": result.tpot,
            "kv_size": result.kv_size,
            "metric_sample_records": len(filtered_records),
            "run_status": "success",
        })
        for record in filtered_records:
            for cost in estimate_component_breakdown(adapter.descriptor, record, components=adapter.components):
                breakdown_rows.append({"dataset": dataset, "slug": slug, "batch_size": bs, "forward_pass_id": record.get("forward_pass_id"), "forward_mode": record.get("forward_mode"), "latency": record.get("latency"), "component_name": cost.name, "bandwidth_units": cost.bandwidth_units, "flops_units": cost.flops_units, "cache_units": cost.cache_units, "attention_score_units": cost.attention_score_units})
    _write_csv(results_dir / "raw_values.csv", rows)
    _write_csv(results_dir / "component_breakdown.csv", breakdown_rows)
    _write_plots(results_dir, rows)
    print(f"wrote {results_dir / 'raw_values.csv'}")



def _load_sweep_config() -> dict[str, Any]:
    path = Path(os.environ.get("SWEEP_CONFIG", Path(__file__).resolve().parent / "configs" / "sweep.yaml"))
    if not path.exists():
        return {}
    try:
        import yaml
        return yaml.safe_load(path.read_text()) or {}
    except Exception:
        return {}


def _estimator_mode(config: dict[str, Any]) -> str:
    mode = str(config.get("estimator_mode") or config.get("estimator") or "component-wise").strip().lower()
    aliases = {"component": "component-wise", "component_wise": "component-wise", "components": "component-wise", "moecap": "moe-cap", "moe_cap": "moe-cap"}
    mode = aliases.get(mode, mode)
    if mode not in {"component-wise", "moe-cap"}:
        raise SystemExit(f"unsupported estimator_mode={mode!r}; expected component-wise or moe-cap")
    return mode

def _leaf_dirs(root: Path):
    if not root.exists():
        return []
    return [p for p in root.rglob("*") if p.is_dir() and (_latest(p, "metadata_*.json") or _latest(p, "failure_*.json"))]




def _load_effective_dataset_config(leaf: Path, dataset: str) -> dict[str, Any]:
    path = leaf.parent / f"effective_config_{dataset}.yaml"
    if not path.exists():
        return {}
    try:
        import yaml
        return yaml.safe_load(path.read_text()) or {}
    except Exception:
        return {}


def _metric_sample_limit(meta: dict[str, Any], dataset: str) -> int | None:
    cfg = meta.get("dataset_config", {}) or {}
    if cfg.get("runner") != "mini_swe_agent" and dataset != "mini_swe_agent":
        return None
    value = cfg.get("metric_sample_steps", cfg.get("num_samples"))
    if value is None:
        return None
    limit = int(value)
    return limit if limit > 0 else None

def _latest(path: Path, pattern: str) -> Path | None:
    matches = sorted(path.glob(pattern))
    return matches[-1] if matches else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _failure_row(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    return {"dataset": data.get("dataset"), "slug": data.get("slug"), "batch_size": data.get("batch_size"), "run_status": data.get("status", "failed"), "error": data.get("error", ""), "prefill_smfu": 0, "prefill_smbu": 0, "decoding_smfu": 0, "decoding_smbu": 0}


def _synthetic_failure_row(leaf: Path, root: Path, error: str) -> dict[str, Any]:
    dataset, slug, bs = _parts(root, leaf)
    return {"dataset": dataset, "slug": slug, "batch_size": bs, "run_status": "failed", "error": error, "prefill_smfu": 0, "prefill_smbu": 0, "decoding_smfu": 0, "decoding_smbu": 0}


def _parts(root: Path, leaf: Path) -> tuple[str, str, str]:
    rel = leaf.relative_to(root).parts
    slug = rel[0] if len(rel) > 0 else ""
    bs = rel[1].removeprefix("bs") if len(rel) > 1 else ""
    dataset = rel[2] if len(rel) > 2 else ""
    return dataset, slug, bs


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)



def _write_plots(results_dir: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    _remove_stale_plot_files(results_dir)

    success = [row for row in rows if row.get("run_status") == "success"]
    datasets = sorted({row.get("dataset") for row in success if row.get("dataset")})
    if not datasets:
        return

    plot_specs = [
        ("smbu", "S-MBU", "S-MBU (%)", "prefill_smbu", "decoding_smbu", "Prefill", "Decode"),
        ("smfu", "S-MFU", "S-MFU (%)", "prefill_smfu", "decoding_smfu", "Prefill", "Decode"),
        ("tokens_per_sec", "Throughput", "Tokens/sec", "prefill_tokens_per_sec", "decoding_tokens_per_sec", "Prefill", "Decode"),
        ("latency", "Latency", "Latency (ms)", "ttft", "tpot", "TTFT", "TPOT"),
    ]
    for prefix, title, ylabel, prefill_key, decode_key, prefill_phase, decode_phase in plot_specs:
        for part_idx, dataset_chunk in enumerate(_chunks(datasets, 4), start=1):
            fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False, sharey=False)
            axes = axes.flatten()
            for ax, dataset in zip(axes, dataset_chunk):
                subset = [row for row in success if row.get("dataset") == dataset]
                slugs = sorted({row.get("slug") for row in subset if row.get("slug")})
                for slug in slugs:
                    points = sorted(
                        (
                            int(row.get("batch_size") or 0),
                            _plot_value(row, prefill_key),
                            _plot_value(row, decode_key),
                        )
                        for row in subset
                        if row.get("slug") == slug and int(row.get("batch_size") or 0) > 0
                    )
                    if not points:
                        continue
                    batch = [point[0] for point in points]
                    prefill = [point[1] for point in points]
                    decode = [point[2] for point in points]
                    if len(slugs) == 1:
                        prefill_label = prefill_phase
                        decode_label = decode_phase
                    else:
                        prefill_label = f"{slug} {prefill_phase}"
                        decode_label = f"{slug} {decode_phase}"
                    ax.plot(batch, prefill, marker="o", linewidth=2, label=prefill_label)
                    ax.plot(batch, decode, marker="s", linewidth=2, label=decode_label)
                ax.set_title(str(dataset).replace("_", " ").title())
                ax.set_xscale("log", base=2)
                batch_ticks = sorted({int(row.get("batch_size") or 0) for row in subset if int(row.get("batch_size") or 0) > 0})
                if batch_ticks:
                    ax.set_xticks(batch_ticks)
                    ax.set_xticklabels([str(value) for value in batch_ticks])
                ax.set_xlabel("Batch size (log2 scale)")
                ax.set_ylabel(ylabel)
                ax.grid(True, which="major", alpha=0.3)
                ax.grid(True, which="minor", alpha=0.12)
                ax.legend(title="Phase", loc="best")
            for ax in axes[len(dataset_chunk):]:
                ax.axis("off")
            suffix = "" if len(datasets) <= 4 else f"_part{part_idx}"
            fig.suptitle(f"{title} by Dataset: Prefill vs Decode", fontsize=16)
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            fig.savefig(results_dir / f"{prefix}_all_datasets_xlog{suffix}.png", dpi=180)
            plt.close(fig)


def _plot_value(row: dict[str, Any], key: str) -> float:
    value = float(row.get(key) or 0)
    if key in {"ttft", "tpot"}:
        return value * 1000
    return value


def _remove_stale_plot_files(results_dir: Path) -> None:
    patterns = [
        "prefill_smfu_*.png",
        "prefill_smbu_*.png",
        "decoding_smfu_*.png",
        "decoding_smbu_*.png",
        "prefill_tokens_per_sec_*.png",
        "decoding_tokens_per_sec_*.png",
        "tokens_per_sec_all_datasets_xlog*.png",
        "latency_all_datasets_xlog*.png",
        "smfu_all_datasets_xlog*.png",
        "smbu_all_datasets_xlog*.png",
    ]
    for pattern in patterns:
        for path in results_dir.glob(pattern):
            try:
                path.unlink()
            except OSError:
                pass


def _chunks(values: list[Any], size: int) -> list[list[Any]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def _precision_bytes(precision: str) -> float:
    return 4.0 if precision in {"float32", "fp32"} else 1.0 if precision in {"int8", "fp8"} else 0.5 if precision in {"int4", "fp4"} else 2.0


if __name__ == "__main__":
    main()
