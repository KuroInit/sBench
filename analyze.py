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
from sbench.hardware import peak_bandwidth_tb, peak_flops_tf


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python analyze.py <RESULTS_DIR>", file=sys.stderr)
        raise SystemExit(1)
    results_dir = Path(sys.argv[1])
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
        try:
            records = _read_jsonl(records_path)
        except Exception as exc:
            rows.append(_synthetic_failure_row(leaf, results_dir, f"invalid server_records JSONL: {exc}"))
            continue
        if not records:
            if failure:
                rows.append(_failure_row(failure))
            continue
        filtered_records = usable_records(records)
        if not filtered_records:
            rows.append(_synthetic_failure_row(leaf, results_dir, "no usable probe records"))
            continue
        model = meta["model_config"]["model_name"]
        precision = meta["model_config"].get("precision", "bfloat16")
        gpu = os.environ.get("ANALYZE_GPU_TYPE") or meta.get("hardware", {}).get("gpu_type") or records[0].get("gpu_raw_type")
        overrides = {"architecture": meta.get("architecture_overrides", {})}
        cfg = meta.get("hf_config", {})
        dataset, slug, bs = _parts(results_dir, leaf)
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
            result = estimate_records(adapter.descriptor, filtered_records, components=adapter.components)
        except Exception as exc:
            rows.append(_synthetic_failure_row(leaf, results_dir, str(exc)))
            continue
        rows.append({
            "dataset": dataset,
            "slug": slug,
            "batch_size": bs,
            "adapter": adapter.name,
            "prefill_tokens_per_sec": result.prefill_tp,
            "decoding_tokens_per_sec": result.decoding_throughput,
            "prefill_smfu": result.prefill_smfu * 100,
            "prefill_smbu": result.prefill_smbu * 100,
            "decoding_smfu": result.decoding_smfu * 100,
            "decoding_smbu": result.decoding_smbu * 100,
            "ttft": result.ttft,
            "tpot": result.tpot,
            "kv_size": result.kv_size,
            "run_status": "success",
        })
        for record in filtered_records:
            for cost in estimate_component_breakdown(adapter.descriptor, record, components=adapter.components):
                breakdown_rows.append({"dataset": dataset, "slug": slug, "batch_size": bs, "forward_pass_id": record.get("forward_pass_id"), "forward_mode": record.get("forward_mode"), "latency": record.get("latency"), "component_name": cost.name, "bandwidth_units": cost.bandwidth_units, "flops_units": cost.flops_units, "cache_units": cost.cache_units, "attention_score_units": cost.attention_score_units})
    _write_csv(results_dir / "raw_values.csv", rows)
    _write_csv(results_dir / "component_breakdown.csv", breakdown_rows)
    _write_plots(results_dir, rows)
    print(f"wrote {results_dir / 'raw_values.csv'}")


def _leaf_dirs(root: Path):
    if not root.exists():
        return []
    return [p for p in root.rglob("*") if p.is_dir() and (_latest(p, "metadata_*.json") or _latest(p, "failure_*.json"))]


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
    success = [row for row in rows if row.get("run_status") == "success"]
    datasets = sorted({row.get("dataset") for row in success if row.get("dataset")})
    metrics = [
        ("prefill_smfu", "Prefill S-MFU (%)"),
        ("prefill_smbu", "Prefill S-MBU (%)"),
        ("decoding_smfu", "Decoding S-MFU (%)"),
        ("decoding_smbu", "Decoding S-MBU (%)"),
        ("prefill_tokens_per_sec", "Prefill tokens/sec"),
        ("decoding_tokens_per_sec", "Decoding tokens/sec"),
    ]
    for dataset in datasets:
        subset = [row for row in success if row.get("dataset") == dataset]
        slugs = sorted({row.get("slug") for row in subset})
        for key, label in metrics:
            if not any(row.get(key) not in (None, "", 0) for row in subset):
                continue
            fig, ax = plt.subplots(figsize=(8, 6))
            for slug in slugs:
                points = sorted(
                    (int(row.get("batch_size") or 0), float(row.get(key) or 0))
                    for row in subset
                    if row.get("slug") == slug
                )
                if points:
                    ax.plot([p[0] for p in points], [p[1] for p in points], "o-", label=slug)
            ax.set_xlabel("Batch Size")
            ax.set_ylabel(label)
            ax.set_title(f"{label} - {dataset}")
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(results_dir / f"{key}_{dataset}.png", dpi=150)
            plt.close(fig)

def _precision_bytes(precision: str) -> float:
    return 4.0 if precision in {"float32", "fp32"} else 1.0 if precision in {"int8", "fp8"} else 0.5 if precision in {"int4", "fp4"} else 2.0


if __name__ == "__main__":
    main()
