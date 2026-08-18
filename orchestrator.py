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
    estimator_mode = _estimator_mode(config)
    plan = sweep_plan(config)
    model_configs, config_errors = load_model_configs(config.get("models", []))
    total = len(plan)
    done = 0
    for dataset, model, bs in plan:
        done += 1
        config_key = model_config_cache_key(model)
        resolved_config = model_configs.get(config_key)
        if resolved_config is None:
            error = config_errors.get(config_key, "model config.json is required but was not loaded")
            mark_config_failure(checkpoint, model, dataset, int(bs), error)
            continue
        model = dict(model, resolved_hf_config=resolved_config)
        slug = model["slug"]
        base_server_flags = merged_sglang_server_flags(config, model)
        dataset_cfg = load_dataset_config(dataset, model, base_server_flags)
        server_flags = workload_sglang_server_flags(base_server_flags, dataset_cfg)
        model["resolved_sglang_server_flags"] = server_flags
        signature = run_signature(model, int(bs), dataset, dataset_cfg, resolved_config, architecture_overrides=architecture_overrides(model, dataset_cfg))
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
        set_probe_model_env(env, model)
        proc = start_sglang(
            model,
            int(bs),
            port,
            env,
            sglang_server_flags=server_flags,
        )
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
            write_metadata(leaf_dir, dataset, model, int(bs), dataset_cfg, estimator_mode=estimator_mode)
            ok, error = validate_probe_file(
                probe_path,
                minimum_usable_records=required_probe_records(dataset_cfg),
                required_forward_modes=required_forward_modes(dataset_cfg),
            )
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


def load_model_configs(models: list[dict[str, Any]]) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], str]]:
    configs = {}
    failures = {}
    for model in models:
        key = model_config_cache_key(model)
        model_id = model["id"]
        if key in configs or key in failures:
            continue
        try:
            configs[key] = load_required_hf_config(model_id, model.get("config_loader"))
        except Exception as exc:
            failures[key] = f"model config.json is required but could not be loaded: {exc}"
    for (model_id, _), error in failures.items():
        print(f"[sweep] config error for {model_id}: {error}", file=sys.stderr)
    return configs, failures


def model_config_cache_key(model: dict[str, Any]) -> tuple[str, str]:
    loader = json.dumps(model.get("config_loader") or {}, sort_keys=True, default=str)
    return str(model["id"]), loader


def mark_config_failure(checkpoint: "Checkpoint", model: dict[str, Any], dataset: str, batch_size: int, error: str) -> None:
    slug = model.get("slug", safe_model_leaf(model.get("id", "")))
    output_dir = Path(RESULTS_DIR) / slug / f"bs{batch_size}" / dataset
    leaf_dir = output_dir / safe_model_leaf(model.get("id", ""))
    write_failure(leaf_dir, dataset, model, batch_size, error)
    signature = run_signature(model, batch_size, dataset, {"config_error": error}, {})
    checkpoint.mark(slug, batch_size, dataset, "failed", signature, error, model.get("id"))


def sweep_plan(config: dict[str, Any]) -> list[tuple[str, dict[str, Any], int]]:
    """Return dataset-major execution order: dataset -> model -> batch size."""

    return [
        (dataset, model, int(batch_size))
        for dataset in active_datasets(config)
        for model in config.get("models", [])
        for batch_size in config.get("batch_sizes", [])
    ]


def set_probe_model_env(env: dict[str, str], model: dict[str, Any]) -> None:
    cfg = model_config_for_metrics(model)
    num_experts = first_int(cfg, "num_experts", "num_experts_per_layer", "n_routed_experts", "n_experts")
    top_k = first_int(cfg, "num_experts_per_tok", "moe_top_k", "topk", "router_topk")
    if num_experts is not None:
        env["SBENCH_NUM_EXPERTS"] = str(num_experts)
    if top_k is not None:
        env["SBENCH_TOP_K"] = str(top_k)


def first_int(cfg: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = cfg.get(key) if isinstance(cfg, dict) else None
        if value is None:
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    if isinstance(cfg, dict):
        for value in cfg.values():
            if isinstance(value, dict):
                parsed = first_int(value, *keys)
                if parsed is not None:
                    return parsed
    return None


def start_sglang(
    model: dict[str, Any],
    batch_size: int,
    port: int,
    env: dict[str, str],
    *,
    sglang_server_flags: Any = None,
) -> subprocess.Popen:
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
    append_sglang_server_flags(cmd, sglang_server_flags)
    return subprocess.Popen(cmd, env=env)


def append_sglang_server_flags(cmd: list[str], flags: Any) -> None:
    for flag, value in iter_sglang_server_flags(flags):
        if value is False or value is None:
            continue
        cmd.append(flag)
        if value is not True:
            cmd.append(str(value))


def iter_sglang_server_flags(flags: Any) -> list[tuple[str, Any]]:
    if not flags:
        return []
    if isinstance(flags, str):
        return [(flags, True)]
    if isinstance(flags, dict):
        return [(str(flag), value) for flag, value in flags.items()]
    if isinstance(flags, list):
        out: list[tuple[str, Any]] = []
        for item in flags:
            out.extend(iter_sglang_server_flags(item))
        return out
    raise TypeError(f"unsupported sglang_server_flags entry: {flags!r}")


def model_served_name(flags: Any) -> str | None:
    served_name = None
    for flag, value in iter_sglang_server_flags(flags):
        if flag == "--served-model-name" and value not in {None, False, True}:
            served_name = str(value)
    return served_name


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
        validate_dataset_config(dataset, cfg)
    if not config.get("models"):
        raise SystemExit("models is required")
    for model in config.get("models", []):
        if model.get("hf_config") is not None:
            raise SystemExit("models must not define hf_config; model architecture is loaded from config.json")
        validate_architecture_overrides(model.get("architecture"))
        validate_config_loader(model.get("config_loader"))
    if not config.get("batch_sizes"):
        raise SystemExit("batch_sizes is required")


def validate_dataset_config(dataset: str, cfg: dict[str, Any]) -> None:
    for key in ("num_samples", "target_output_tokens"):
        if key in cfg and int(cfg[key]) < 0:
            raise SystemExit(f"{dataset} {key} must be non-negative")
    if int(cfg.get("num_samples", 1)) <= 0:
        raise SystemExit(f"{dataset} num_samples must be positive")
    if "max_input_tokens" in cfg and int(cfg["max_input_tokens"]) <= 0:
        raise SystemExit(f"{dataset} max_input_tokens must be positive")
    if "target_input_tokens" in cfg and int(cfg["target_input_tokens"]) <= 0:
        raise SystemExit(f"{dataset} target_input_tokens must be positive")
    min_success_rate = float(cfg.get("min_success_rate", 1.0))
    if not 0.0 < min_success_rate <= 1.0:
        raise SystemExit(f"{dataset} min_success_rate must be in (0, 1]")
    if "prefix_cache" in cfg and not isinstance(cfg["prefix_cache"], bool):
        raise SystemExit(f"{dataset} prefix_cache must be true or false")
    if dataset == "mmlu_pro" and str(cfg.get("answer_mode", "direct")).lower() not in {"direct", "reasoning"}:
        raise SystemExit("mmlu_pro answer_mode must be direct or reasoning")
    if cfg.get("runner") == "mini_swe_agent":
        if int(cfg.get("metric_sample_steps", cfg.get("num_samples", 0))) <= 0:
            raise SystemExit(f"{dataset} metric_sample_steps must be positive")
        if "issue_count" in cfg and int(cfg["issue_count"]) <= 0:
            raise SystemExit(f"{dataset} issue_count must be positive")


def validate_architecture_overrides(overrides: Any) -> None:
    if overrides is None:
        return
    if not isinstance(overrides, dict):
        raise SystemExit("model architecture overrides must be a mapping")
    allowed = {"attention", "cache", "ffn", "moe", "runtime"}
    invalid = sorted(set(overrides) - allowed)
    if invalid:
        names = ", ".join(invalid)
        raise SystemExit(f"model architecture overrides must use component keys only; invalid keys: {names}")


def validate_config_loader(options: Any) -> None:
    if options is None:
        return
    if not isinstance(options, dict):
        raise SystemExit("model config_loader must be a mapping")
    allowed = {"local_files_only", "revision", "cache_dir", "token", "trust_remote_code"}
    invalid = sorted(set(options) - allowed)
    if invalid:
        names = ", ".join(invalid)
        raise SystemExit(f"model config_loader contains unsupported keys: {names}")


def load_dataset_config(dataset: str, model: dict[str, Any], server_flags: Any = None) -> dict[str, Any]:
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
    served_name = model_served_name(server_flags or model.get("sglang_server_flags"))
    if cfg.get("runner") == "mini_swe_agent" and served_name:
        cfg.setdefault("mini_model_name", f"openai/{served_name}")
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



def _estimator_mode(config: dict[str, Any]) -> str:
    mode = str(config.get("estimator_mode") or config.get("estimator") or "component-wise").strip().lower()
    aliases = {"component": "component-wise", "component_wise": "component-wise", "components": "component-wise", "moecap": "moe-cap", "moe_cap": "moe-cap"}
    mode = aliases.get(mode, mode)
    if mode not in {"component-wise", "moe-cap"}:
        raise SystemExit(f"unsupported estimator_mode={mode!r}; expected component-wise or moe-cap")
    return mode


def validate_request_results(results: list[Any], cfg: dict[str, Any]) -> tuple[bool, str]:
    if not results:
        return False, "dataset produced no requests"
    successes = sum(1 for result in results if result.success)
    total = len(results)
    min_success_rate = float(cfg.get("min_success_rate", 1.0))
    if successes / total < min_success_rate:
        return False, f"request success rate {successes}/{total} is below required {min_success_rate:.2f}"
    return True, ""


def required_probe_records(cfg: dict[str, Any]) -> int | None:
    """Return a strict probe-record requirement for agentic metric sampling."""

    if cfg.get("runner") != "mini_swe_agent":
        return None
    value = cfg.get("metric_sample_steps", cfg.get("num_samples"))
    if value is None:
        return None
    required = int(value)
    return required if required > 0 else None


def required_forward_modes(cfg: dict[str, Any]) -> set[str]:
    """Return the forward phases required for a valid workload trace."""

    if cfg.get("benchmark_type") == "prefill":
        return {"prefill"}
    return {"prefill", "decode"}


def validate_probe_file(
    path: Path,
    *,
    minimum_usable_records: int | None = None,
    required_forward_modes: set[str] | None = None,
) -> tuple[bool, str]:
    if not path.exists() or path.stat().st_size == 0:
        return False, "probe produced no server_records file"
    required = {"forward_mode", "latency", "seq_lens_sum", "batch_size"}
    records = []
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
            records.append(record)
    except Exception as exc:
        return False, f"probe server_records file is invalid JSONL: {exc}"
    if not records:
        return False, "probe produced no usable records"
    if required_forward_modes:
        from sbench.estimator import usable_records

        present = {str(record.get("forward_mode")) for record in usable_records(records)}
        missing_modes = sorted(required_forward_modes - present)
        if missing_modes:
            return False, f"probe is missing required usable forward mode(s): {', '.join(missing_modes)}"
    if minimum_usable_records is not None:
        from sbench.estimator import usable_records

        count = len(usable_records(records))
        if count < minimum_usable_records:
            return False, (
                f"probe produced only {count} usable records; "
                f"{minimum_usable_records} are required for metric sampling"
            )
    return True, ""


def run_signature(
    model: dict[str, Any],
    batch_size: int,
    dataset: str,
    dataset_cfg: dict[str, Any],
    hf_config: dict[str, Any] | None = None,
    architecture_overrides: dict[str, Any] | None = None,
) -> str:
    sweep_cfg = load_yaml(SWEEP_CONFIG)
    payload = {
        "model_id": model.get("id"),
        "slug": model.get("slug"),
        "tp": model.get("tp", 1),
        "batch_size": batch_size,
        "dataset": dataset,
        "dataset_cfg": dataset_cfg,
        "hf_config": hf_config if hf_config is not None else model_config_for_metrics(model),
        "architecture_overrides": architecture_overrides if architecture_overrides is not None else architecture_overrides_for_signature(model, dataset_cfg),
        "probe_schema": 1,
        "estimator_mode": _estimator_mode(sweep_cfg),
        "sglang_server_flags": sweep_cfg.get("sglang_server_flags"),
        "model_sglang_server_flags": model.get("sglang_server_flags"),
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def safe_model_leaf(model_id: str) -> str:
    cleaned = str(model_id).strip().replace("\\", "/").strip("/")
    if not cleaned:
        return "model"
    return cleaned.replace("/", "__")


def write_metadata(
    leaf: Path,
    dataset: str,
    model: dict[str, Any],
    batch_size: int,
    cfg: dict[str, Any],
    *,
    estimator_mode: str,
) -> None:
    hf_config = model_config_for_metrics(model)
    server_flags = model.get("resolved_sglang_server_flags")
    payload = {
        "model_config": {"model_name": model["id"], "precision": model_precision(model, cfg)},
        "hardware": {"num_gpus": model.get("tp", 1), "gpu_type": os.environ.get("SBENCH_GPU_TYPE", "unknown")},
        "system_environment": {"batch_size": batch_size},
        "architecture_overrides": architecture_overrides(model, cfg),
        "estimator_mode": estimator_mode,
        "sglang_server_flags": server_flags,
        "hf_config": hf_config,
        "dataset_config": cfg,
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



def model_config_for_metrics(model: dict[str, Any]) -> dict[str, Any]:
    cfg = model.get("resolved_hf_config")
    if isinstance(cfg, dict) and cfg:
        return cfg
    raise ValueError(f"model config.json has not been loaded for {model.get('id')!r}")


def load_required_hf_config(model_id: str, loader_options: dict[str, Any] | None = None) -> dict[str, Any]:
    if not model_id:
        raise ValueError("model id/path is empty")
    local_cfg = read_local_config_json(model_id)
    if local_cfg:
        return local_cfg
    errors = []
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_id, **auto_config_kwargs(loader_options)).to_dict()
        if cfg:
            return cfg
    except Exception as exc:
        errors.append(f"AutoConfig failed: {exc}")
    try:
        cfg = load_raw_hf_config_json(model_id, loader_options)
        if cfg:
            return cfg
    except Exception as exc:
        errors.append(f"raw config.json failed: {exc}")
    detail = "; ".join(errors) if errors else "empty config.json"
    raise ValueError(f"failed to load config.json for {model_id!r}: {detail}")


def read_local_config_json(model_id: str) -> dict[str, Any]:
    path = Path(os.path.expanduser(os.path.expandvars(model_id)))
    if path.is_dir():
        path = path / "config.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        raise ValueError(f"failed to parse local config.json at {path}: {exc}") from exc
    if not isinstance(data, dict) or not data:
        raise ValueError(f"local config.json at {path} is empty or invalid")
    return data


def load_raw_hf_config_json(model_id: str, loader_options: dict[str, Any] | None = None) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=model_id, filename="config.json", **hf_hub_download_kwargs(loader_options))
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or not data:
        raise ValueError("downloaded config.json is empty or invalid")
    return data


def auto_config_kwargs(options: dict[str, Any] | None = None) -> dict[str, Any]:
    options = options or {}
    kwargs = {"trust_remote_code": bool(options.get("trust_remote_code", True))}
    for key in ("revision", "cache_dir", "token"):
        value = options.get(key)
        if isinstance(value, str):
            value = os.path.expanduser(os.path.expandvars(value))
        if value not in {None, ""}:
            kwargs[key] = value
    if "local_files_only" in options:
        kwargs["local_files_only"] = bool(options["local_files_only"])
    return kwargs


def hf_hub_download_kwargs(options: dict[str, Any] | None = None) -> dict[str, Any]:
    options = options or {}
    kwargs = {}
    for key in ("revision", "cache_dir", "token"):
        value = options.get(key)
        if isinstance(value, str):
            value = os.path.expanduser(os.path.expandvars(value))
        if value not in {None, ""}:
            kwargs[key] = value
    if "local_files_only" in options:
        kwargs["local_files_only"] = bool(options["local_files_only"])
    return kwargs


def merged_sglang_server_flags(config: dict[str, Any], model: dict[str, Any]) -> list[Any]:
    flags = []
    flags.extend(config.get("sglang_server_flags") or [])
    flags.extend(model.get("sglang_server_flags") or [])
    return flags


def workload_sglang_server_flags(flags: Any, dataset_cfg: dict[str, Any]) -> list[Any]:
    """Resolve one unambiguous SGLang flag set for a workload.

    Radix caching is useful for agentic tool loops with a shared issue prefix,
    but distorts the independent-request chat, reasoning, and prefill lanes.
    Dataset YAML therefore owns this one server policy.
    """

    ordered: dict[str, Any] = {}
    for flag, value in iter_sglang_server_flags(flags):
        ordered[flag] = value
    prefix_cache = dataset_cfg.get("prefix_cache")
    if prefix_cache is True:
        ordered.pop("--disable-radix-cache", None)
    elif prefix_cache is False:
        ordered["--disable-radix-cache"] = True
    return [flag if value is True else {flag: value} for flag, value in ordered.items() if value is not None and value is not False]


def model_precision(model: dict[str, Any], dataset_cfg: dict[str, Any]) -> str:
    precision = None
    for flag, value in iter_sglang_server_flags(model.get("resolved_sglang_server_flags")):
        if flag == "--dtype" and value not in {None, False, True}:
            precision = str(value)
    return precision or str(dataset_cfg.get("precision", "bfloat16"))


def architecture_overrides(model: dict[str, Any], dataset_cfg: dict[str, Any]) -> dict[str, Any]:
    return model.get("architecture") or dataset_cfg.get("architecture", {}) or {}


def architecture_overrides_for_signature(model: dict[str, Any], dataset_cfg: dict[str, Any]) -> dict[str, Any]:
    return architecture_overrides(model, dataset_cfg)

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
