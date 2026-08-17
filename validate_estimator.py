#!/usr/bin/env python3
"""Post-run estimator validation for sBench results."""

from __future__ import annotations

import argparse
from pathlib import Path

from sbench.validation import run_validation


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate component-wise estimates against reference estimators/profiler summaries.")
    parser.add_argument("results_dir", help="sBench results directory containing metadata_*.json and server_records_*.jsonl files")
    parser.add_argument("--out-dir", default=None, help="Directory for validation outputs. Defaults to <results_dir>/validation")
    parser.add_argument("--profiler-summary", default=None, help="Optional profiler summary CSV to compare against estimated utilization")
    parser.add_argument("--telemetry-summary", default=None, help="Optional DCGM/nvidia-smi summary CSV for lightweight utilization validation")
    parser.add_argument("--sample-limit", type=int, default=None, help="Optional max usable records per run for validation recomputation")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.results_dir) / "validation"
    result = run_validation(
        Path(args.results_dir),
        out_dir=out_dir,
        profiler_summary=Path(args.profiler_summary) if args.profiler_summary else None,
        telemetry_summary=Path(args.telemetry_summary) if args.telemetry_summary else None,
        sample_limit=args.sample_limit,
    )
    print(f"wrote {result.summary_path}")
    print(f"wrote {result.estimator_comparison_path}")
    if result.profiler_comparison_path:
        print(f"wrote {result.profiler_comparison_path}")
    if result.telemetry_comparison_path:
        print(f"wrote {result.telemetry_comparison_path}")


if __name__ == "__main__":
    main()
