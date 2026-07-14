"""Prewarm Singularity/Apptainer cache for mini-SWE-agent SWE-Bench images."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = {
    "lite": "princeton-nlp/SWE-bench_Lite",
    "verified": "princeton-nlp/SWE-bench_Verified",
    "full": "princeton-nlp/SWE-bench",
}


def main() -> int:
    args = parse_args()
    sweep = load_yaml(args.sweep_config)
    configs = mini_swe_configs(sweep)
    if not configs:
        print("[prewarm] mini_swe_agent is not enabled; skipping")
        return 0

    runtime = find_runtime()
    if not runtime:
        print("[prewarm] neither apptainer nor singularity was found", file=sys.stderr)
        return 1

    sif_dir = Path(args.sif_dir or os.environ.get("SBENCH_SIF_CACHE_DIR") or Path(os.environ.get("BASE", ".")) / "sif_cache")
    sif_dir.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    for cfg in configs:
        instance_ids = resolve_instance_ids(cfg)
        if not instance_ids:
            print("[prewarm] no SWE-Bench instance ids resolved; set instance_ids in configs/mini_swe_agent.yaml", file=sys.stderr)
            return 1
        for instance_id in instance_ids:
            if instance_id in seen:
                continue
            seen.add(instance_id)
            image = swebench_image(instance_id)
            sif_path = sif_dir / f"{docker_safe_instance_id(instance_id)}.sif"
            if sif_path.exists() and sif_path.stat().st_size > 0 and not args.force:
                print(f"[prewarm] cached {sif_path}")
                continue
            cmd = [runtime, "pull", "--force", str(sif_path), f"docker://{image}"]
            print("[prewarm]", " ".join(cmd))
            subprocess.run(cmd, check=True)

    print(f"[prewarm] ready: {len(seen)} image(s) in {sif_dir}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-config", default=os.environ.get("SWEEP_CONFIG", str(PROJECT_ROOT / "configs" / "sweep.yaml")))
    parser.add_argument("--sif-dir", default=os.environ.get("SBENCH_SIF_CACHE_DIR"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def mini_swe_configs(sweep: dict[str, Any]) -> list[dict[str, Any]]:
    datasets = (sweep.get("benchmark_types") or {}).get("agentic") or []
    configs: list[dict[str, Any]] = []
    for dataset in datasets:
        path = PROJECT_ROOT / "configs" / f"{dataset}.yaml"
        if not path.exists():
            continue
        cfg = load_yaml(path)
        if cfg.get("runner") == "mini_swe_agent" and str(cfg.get("environment_class", "")).lower() == "singularity":
            configs.append(cfg)
    return configs


def resolve_instance_ids(cfg: dict[str, Any]) -> list[str]:
    explicit = cfg.get("prewarm_instance_ids") or cfg.get("instance_ids")
    if explicit:
        return [str(item) for item in explicit]

    subset = str(cfg.get("subset", "lite")).lower()
    dataset_name = str(cfg.get("prewarm_hf_dataset") or DEFAULT_DATASETS.get(subset, DEFAULT_DATASETS["lite"]))
    split = str(cfg.get("prewarm_split") or cfg.get("split") or "test")
    limit_slice = parse_slice(cfg.get("prewarm_slice") or slice_from_extra_args(cfg.get("extra_args") or []))

    try:
        return load_instance_ids(dataset_name, split, limit_slice)
    except Exception as exc:
        if split != "test":
            print(f"[prewarm] could not load split={split!r} from {dataset_name}: {exc}; trying split='test'", file=sys.stderr)
            return load_instance_ids(dataset_name, "test", limit_slice)
        raise


def load_instance_ids(dataset_name: str, split: str, limit_slice: slice | None) -> list[str]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("install datasets or set instance_ids/prewarm_instance_ids") from exc

    rows = load_dataset(dataset_name, split=split, streaming=True)
    ids: list[str] = []
    start = limit_slice.start if limit_slice else 0
    stop = limit_slice.stop if limit_slice else None
    for idx, row in enumerate(rows):
        if idx < start:
            continue
        if stop is not None and idx >= stop:
            break
        instance_id = row.get("instance_id")
        if instance_id:
            ids.append(str(instance_id))
    return ids


def slice_from_extra_args(extra_args: list[Any]) -> str | None:
    args = [str(arg) for arg in extra_args]
    for idx, arg in enumerate(args):
        if arg == "--slice" and idx + 1 < len(args):
            return args[idx + 1]
        if arg.startswith("--slice="):
            return arg.split("=", 1)[1]
    return None


def parse_slice(value: str | None) -> slice | None:
    if not value:
        return None
    if ":" not in value:
        idx = int(value)
        return slice(idx, idx + 1)
    start, stop = value.split(":", 1)
    return slice(int(start) if start else 0, int(stop) if stop else None)


def swebench_image(instance_id: str) -> str:
    return f"docker.io/swebench/sweb.eval.x86_64.{docker_safe_instance_id(instance_id)}:latest"


def docker_safe_instance_id(instance_id: str) -> str:
    return instance_id.replace("__", "_1776_").replace("/", "_")


def find_runtime() -> str | None:
    preferred = os.environ.get("SBENCH_APPTAINER_BIN")
    if preferred:
        return preferred
    return shutil.which("apptainer") or shutil.which("singularity")


def load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text()) or {}


if __name__ == "__main__":
    raise SystemExit(main())
