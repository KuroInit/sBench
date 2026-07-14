#!/usr/bin/env python3
"""sBench sweep orchestrator for newer SGLang + component estimator."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from sbench.datasets import load_dataset
from sbench.mini_swe_agent_runner import run_mini_swe_agent
from sbench.runner import run_requests, write_request_results

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = os.environ.get("RESULTS_DIR", "./results")
SWEEP_CONFIG = os.environ.get("SWEEP_CONFIG", str(PROJECT_ROOT / "configs" / "sweep.yaml"))
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", os.path.join(RESULTS_DIR, "checkpoint.yaml"))
SUPPORTED_LANES = {"prefill", "chat", "reasoning", "agentic"}


def main() -> None:
    config = load_yaml(SWEEP_CONFIG)
    validate_config(config)
    checkpoint = Checkpoint(CHECKPOINT_PATH)
    run_sweep(config, checkpoint)


def run_sweep(config: dict[str, Any], checkpoint: "Checkpoint") -> None:
    port = int(config.get("port", 30000))
    plan = sweep_plan(config)
    total = len(plan)
    done = 0
    for dataset, model, bs in plan:
        done += 1
        slug = model["slug"]
        dataset_cfg = load_dataset_config(dataset, model)
        signature = run_signature(model, int(bs), dataset, dataset_cfg)
        if checkpoint.is_done(slug, int(bs), dataset, signature):
            print(f"[sweep] [{done}/{total}] skip {slug} bs={bs} {dataset}: done")
            continue
        output_dir = Path(RESULTS_DIR) / slug / f"bs{bs}" / dataset
        leaf_dir = output_dir / safe_model_leaf(model["id"])
        leaf_dir.mkdir(parents=True, exist_ok=True)
        write_yaml(output_dir / f"effective_config_{dataset}.yaml", dataset_cfg)
        probe_path = leaf_dir / f"server_records_{dataset}_{timestamp()}.jsonl"
        print(f"[sweep] [{done}/{total}] {dataset}  {slug}  bs={bs}")
        env = os.environ.copy()
        env["SBENCH_PROBE_RECORD_PATH"] = str(probe_path)
        env["SBENCH_AUTO_INSTALL_PROBE"] = "1"
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(PROJECT_ROOT) if not existing_pythonpath else f"{PROJECT_ROOT}:{existing_pythonpath}"
        env["SBENCH_GPU_TYPE"] = env.get("ANALYZE_GPU_TYPE", env.get("SBENCH_GPU_TYPE", "unknown"))
        proc = start_sglang(model, int(bs), port, env)
        try:
            if not wait_health(port, proc):
                error = f"SGLang failed to start, code={proc.poll()}"
                write_failure(leaf_dir, dataset, model, int(bs), error)
                checkpoint.mark(slug, int(bs), dataset, "failed", signature, error, model["id"])
                continue
            if dataset_cfg.get("runner") == "mini_swe_agent":
                mini_result = run_mini_swe_agent(
                    api_base=f"http://127.0.0.1:{port}",
                    model_id=model["id"],
                    batch_size=int(bs),
                    dataset_cfg=dataset_cfg,
                    output_dir=leaf_dir / "mini_swe_agent",
                    env=env,
                )
                if not mini_result.success:
                    error = mini_result.error or "mini-SWE-agent failed"
                    write_failure(leaf_dir, dataset, model, int(bs), error)
                    checkpoint.mark(slug, int(bs), dataset, "failed", signature, error, model["id"])
                    continue
            else:
                requests = load_dataset(dataset, dataset_cfg, limit=dataset_cfg.get("num_samples"))
                if not requests:
                    error = "dataset produced no requests"
                    write_failure(leaf_dir, dataset, model, int(bs), error)
                    checkpoint.mark(slug, int(bs), dataset, "failed", signature, error, model["id"])
                    continue
                use_chat = lane_for_dataset(config, dataset) == "chat"
                results = asyncio.run(run_requests(f"http://127.0.0.1:{port}", model["id"], requests, int(bs), use_chat_api=use_chat))
                write_request_results(str(leaf_dir / f"detailed_results_{dataset}_{timestamp()}.jsonl"), results)
                ok, error = validate_request_results(results, dataset_cfg)
                if not ok:
                    write_failure(leaf_dir, dataset, model, int(bs), error)
                    checkpoint.mark(slug, int(bs), dataset, "failed", signature, error, model["id"])
                    continue
            write_metadata(leaf_dir, dataset, model, int(bs), dataset_cfg)
            ok, error = validate_probe_file(probe_path)
            if not ok:
                write_failure(leaf_dir, dataset, model, int(bs), error)
                checkpoint.mark(slug, int(bs), dataset, "failed", signature, error, model["id"])
            else:
                clear_failures(leaf_dir)
                checkpoint.mark(slug, int(bs), dataset, "success", signature, model_id=model["id"])
        except Exception as exc:
            write_failure(leaf_dir, dataset, model, int(bs), str(exc))
            checkpoint.mark(slug, int(bs), dataset, "failed", signature, str(exc), model["id"])
        finally:
            stop_process(proc)
    print(f"[sweep] complete {done}/{total}")


def sweep_plan(config: dict[str, Any]) -> list[tuple[str, dict[str, Any], int]]:
    """Return dataset-major execution order: dataset -> model -> batch size."""

    return [
        (dataset, model, int(batch_size))
        for dataset in active_datasets(config)
        for model in config.get("models", [])
        for batch_size in config.get("batch_sizes", [])
    ]


def start_sglang(model: dict[str, Any], batch_size: int, port: int, env: dict[str, str]) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "sbench_probe.sglang_entrypoint",
        "--model-path",
        model["id"],
        "--port",
        str(port),
        "--tp-size",
        str(model.get("tp", 1)),
        "--max-running-requests",
        str(batch_size),
    ]
    if os.environ.get("DISABLE_RADIX_CACHE", "1").lower() not in {"0", "false", "no"}:
        cmd.append("--disable-radix-cache")
    for src, flag in (("chunked_prefill_size", "--chunked-prefill-size"), ("max_prefill_tokens", "--max-prefill-tokens"), ("mem_fraction_static", "--mem-fraction-static"), ("context_length", "--context-length")):
        if model.get(src) is not None:
            cmd += [flag, str(model[src])]
    if model.get("served_model_name"):
        cmd += ["--served-model-name", str(model["served_model_name"])]
    if model.get("dtype"):
        cmd += ["--dtype", str(model["dtype"])]
    if model.get("chat_template"):
        cmd += ["--chat-template", str(model["chat_template"])]
    return subprocess.Popen(cmd, env=env)


def wait_health(port: int, proc: subprocess.Popen, timeout: int = 1500) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(2)
    return False


def stop_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=30)


def active_datasets(config: dict[str, Any]) -> list[str]:
    return list(active_dataset_lanes(config))


def lane_for_dataset(config: dict[str, Any], dataset: str) -> str:
    return active_dataset_lanes(config).get(dataset, "")


def validate_config(config: dict[str, Any]) -> None:
    for dataset, lane in active_dataset_lanes(config).items():
        cfg = load_dataset_config(dataset, {"id": "", "slug": ""})
        declared = cfg.get("benchmark_type")
        if declared and declared != lane:
            raise SystemExit(f"{dataset} is configured as benchmark_type={declared}, but the sweep config places it under {lane}")
        names = cfg.get("dataset_names")
        if names and dataset not in names:
            raise SystemExit(f"{dataset} config declares dataset_names={names}, which does not include {dataset}")
    if not config.get("models"):
        raise SystemExit("models is required")
    if not config.get("batch_sizes"):
        raise SystemExit("batch_sizes is required")


def load_dataset_config(dataset: str, model: dict[str, Any]) -> dict[str, Any]:
    path = PROJECT_ROOT / "configs" / f"{dataset}.yaml"
    cfg = load_yaml(str(path)) if path.exists() else {"dataset_names": [dataset]}
    cfg = dict(cfg)
    overrides = cfg.pop("model_overrides", {}) or {}
    for key in (model.get("config_slug"), model.get("slug"), model.get("id")):
        if key and key in overrides:
            cfg.update(overrides[key] or {})
            break
    if model.get("id"):
        cfg.setdefault("model_id", model["id"])
    if cfg.get("runner") == "mini_swe_agent" and model.get("served_model_name"):
        cfg.setdefault("mini_model_name", f"openai/{model['served_model_name']}")
    return cfg


def active_dataset_lanes(config: dict[str, Any]) -> dict[str, str]:
    lanes: dict[str, str] = {}
    for lane, datasets in (config.get("benchmark_types") or {}).items():
        if lane not in SUPPORTED_LANES:
            raise SystemExit(f"unsupported lane {lane}")
        for dataset in datasets or []:
            if dataset in lanes:
                raise SystemExit(f"dataset {dataset} appears in multiple lanes")
            lanes[dataset] = lane
    if not lanes:
        raise SystemExit("benchmark_types must enable at least one dataset")
    return lanes


def validate_request_results(results: list[Any], cfg: dict[str, Any]) -> tuple[bool, str]:
    if not results:
        return False, "dataset produced no requests"
    successes = sum(1 for result in results if result.success)
    total = len(results)
    min_success_rate = float(cfg.get("min_success_rate", 1.0))
    if successes / total < min_success_rate:
        return False, f"request success rate {successes}/{total} is below required {min_success_rate:.2f}"
    return True, ""


def validate_probe_file(path: Path) -> tuple[bool, str]:
    if not path.exists() or path.stat().st_size == 0:
        return False, "probe produced no server_records file"
    required = {"forward_mode", "latency", "seq_lens_sum", "batch_size"}
    count = 0
    try:
        for line_no, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            missing = sorted(required - set(record))
            if missing:
                return False, f"probe record line {line_no} missing fields: {', '.join(missing)}"
            if record.get("forward_mode") not in {"prefill", "decode"}:
                return False, f"probe record line {line_no} has unsupported forward_mode={record.get('forward_mode')!r}"
            if float(record.get("latency") or 0) <= 0:
                return False, f"probe record line {line_no} has non-positive latency"
            count += 1
    except Exception as exc:
        return False, f"probe server_records file is invalid JSONL: {exc}"
    if count == 0:
        return False, "probe produced no usable records"
    return True, ""


def run_signature(model: dict[str, Any], batch_size: int, dataset: str, dataset_cfg: dict[str, Any]) -> str:
    payload = {
        "model_id": model.get("id"),
        "slug": model.get("slug"),
        "tp": model.get("tp", 1),
        "batch_size": batch_size,
        "dataset": dataset,
        "dataset_cfg": dataset_cfg,
        "hf_config": model.get("hf_config") or model.get("architecture"),
        "probe_schema": 1,
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def safe_model_leaf(model_id: str) -> str:
    cleaned = str(model_id).strip().replace("\\", "/").strip("/")
    if not cleaned:
        return "model"
    return cleaned.replace("/", "__")


def write_metadata(leaf: Path, dataset: str, model: dict[str, Any], batch_size: int, cfg: dict[str, Any]) -> None:
    hf_config = model.get("hf_config") or model.get("architecture") or _try_load_hf_config(model.get("id", ""))
    payload = {
        "model_config": {"model_name": model["id"], "precision": cfg.get("precision", "bfloat16")},
        "hardware": {"num_gpus": model.get("tp", 1), "gpu_type": os.environ.get("SBENCH_GPU_TYPE", "unknown")},
        "system_environment": {"batch_size": batch_size},
        "architecture_overrides": cfg.get("architecture", {}),
        "hf_config": hf_config,
    }
    (leaf / f"metadata_{dataset}_{timestamp()}.json").write_text(json.dumps(payload, indent=2))



def clear_failures(leaf: Path) -> None:
    for failure in leaf.glob("failure_*.json"):
        try:
            failure.unlink()
        except OSError:
            pass

def write_failure(leaf: Path, dataset: str, model: dict[str, Any], batch_size: int, error: str) -> None:
    payload = {"status": "failed", "error": error, "dataset": dataset, "slug": model.get("slug"), "model": model.get("id"), "batch_size": batch_size, "tp": model.get("tp", 1), "timestamp": timestamp()}
    leaf.mkdir(parents=True, exist_ok=True)
    (leaf / f"failure_{dataset}_{timestamp()}.json").write_text(json.dumps(payload, indent=2))



def _try_load_hf_config(model_id: str) -> dict[str, Any]:
    if not model_id:
        return {}
    try:
        from transformers import AutoConfig
        return AutoConfig.from_pretrained(model_id, trust_remote_code=True).to_dict()
    except Exception:
        return {}

class Checkpoint:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.entries = self._load()

    def _load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        data = load_yaml(str(self.path)) or {}
        return data.get("completed", [])

    def is_done(self, slug: str, batch_size: int, dataset: str, signature: str) -> bool:
        return any(
            e.get("slug") == slug
            and e.get("batch_size") == batch_size
            and e.get("dataset") == dataset
            and e.get("status") == "success"
            and e.get("signature") == signature
            for e in self.entries
        )

    def mark(self, slug: str, batch_size: int, dataset: str, status: str, signature: str, error: str | None = None, model_id: str | None = None) -> None:
        self.entries = [e for e in self.entries if not (e.get("slug") == slug and e.get("batch_size") == batch_size and e.get("dataset") == dataset)]
        entry = {"slug": slug, "batch_size": batch_size, "dataset": dataset, "status": status, "signature": signature}
        if error:
            entry["error"] = error
        if model_id:
            entry["model"] = model_id
        self.entries.append(entry)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_yaml(self.path, {"completed": self.entries})


def load_yaml(path: str) -> dict[str, Any]:
    text = Path(path).read_text()
    if path.endswith(".json"):
        return json.loads(text)
    try:
        import yaml
        return yaml.safe_load(text) or {}
    except ImportError:
        return json.loads(text)


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml
        path.write_text(yaml.safe_dump(data, sort_keys=False))
    except ImportError:
        path.write_text(json.dumps(data, indent=2))


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


if __name__ == "__main__":
    main()
