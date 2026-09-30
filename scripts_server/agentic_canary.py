#!/usr/bin/env python3
"""Preflight one mini-SWE-agent issue against a running SGLang server."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sbench.mini_swe_agent_runner import run_mini_swe_agent  # noqa: E402


def normalize_api_base(api_base: str) -> str:
    value = api_base.rstrip("/")
    parts = urlsplit(value)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise ValueError("--api-base must be an absolute http(s) URL")
    path = parts.path[:-3] if parts.path.endswith("/v1") else parts.path
    return urlunsplit((parts.scheme, parts.netloc, path.rstrip("/"), "", ""))


def check_server(api_base: str, model_id: str, timeout: float = 10, api_key: str = "") -> list[str]:
    """Require a healthy server and an OpenAI model entry matching model_id."""
    base = normalize_api_base(api_base)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        with urlopen(Request(f"{base}/health", headers=headers), timeout=timeout) as response:
            if response.status != 200:
                raise RuntimeError(f"server health check returned HTTP {response.status}")
        with urlopen(Request(f"{base}/v1/models", headers=headers), timeout=timeout) as response:
            payload = json.load(response)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"SGLang preflight failed for {base}: {exc}") from exc

    entries = payload.get("data") if isinstance(payload, dict) else None
    model_ids = [str(entry["id"]) for entry in entries or [] if isinstance(entry, dict) and entry.get("id")]
    if not model_ids:
        raise RuntimeError("/v1/models returned no model IDs")
    if model_id not in model_ids:
        raise RuntimeError(f"model {model_id!r} is not served; available: {', '.join(model_ids)}")
    return model_ids


def build_canary_config(
    source: dict,
    instance_id: str,
    model_id: str,
    model_name: str | None = None,
) -> dict:
    """Copy the agent config and constrain this invocation to one known issue."""
    if not instance_id.strip():
        raise ValueError("instance_id must not be empty")
    if not model_id.strip():
        raise ValueError("model_id must not be empty")
    config = copy.deepcopy(source)
    overrides = config.pop("model_overrides", {})
    if isinstance(overrides, dict):
        override = overrides.get(model_id)
        if isinstance(override, dict):
            config.update(copy.deepcopy(override))
    config["instance_ids"] = [instance_id]
    config["issue_count"] = 1
    config["workers"] = 1
    config["max_issue_count"] = 1
    config["max_workers"] = 1
    config["mini_model_name"] = model_name or f"openai/{model_id}"
    mini_swe_configs = config.get("mini_swe_configs") or []
    if isinstance(mini_swe_configs, str):
        mini_swe_configs = [mini_swe_configs]
    else:
        mini_swe_configs = list(mini_swe_configs)
    compact_config = "configs/mini_swe_agent_compact.yaml"
    if compact_config not in mini_swe_configs:
        mini_swe_configs.append(compact_config)
    config["mini_swe_configs"] = mini_swe_configs
    if str(config.get("environment_class", "docker")).lower() == "singularity":
        config.setdefault("sandbox_tmpdir", "/tmp")
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base", required=True, help="SGLang root URL, e.g. http://127.0.0.1:30000")
    parser.add_argument("--model-id", required=True, help="Exact model ID listed by GET /v1/models")
    parser.add_argument("--instance-id", required=True, help="One SWE-bench instance ID to run")
    parser.add_argument("--model-name", help="mini-SWE model name; defaults to openai/<model-id>")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "mini_swe_agent.yaml",
        help="mini-SWE-agent dataset YAML (default: configs/mini_swe_agent.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "agentic_canary",
        help="Directory for this canary's logs and predictions",
    )
    parser.add_argument("--timeout", type=float, default=10, help="HTTP preflight timeout in seconds")
    args = parser.parse_args(argv)

    try:
        import yaml

        source = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        if not isinstance(source, dict):
            raise RuntimeError(f"agent config must contain a YAML mapping: {args.config}")
        base = normalize_api_base(args.api_base)
        api_key = os.environ.get("OPENAI_API_KEY", "")
        model_ids = check_server(base, args.model_id, timeout=args.timeout, api_key=api_key)
        print(f"SGLang reachable; served models: {', '.join(model_ids)}")

        config = build_canary_config(source, args.instance_id, args.model_id, args.model_name)
        result = run_mini_swe_agent(
            api_base=base,
            model_id=args.model_id,
            batch_size=1,
            dataset_cfg=config,
            output_dir=args.output_dir,
        )
        print(f"Agent canary return code: {result.returncode}")
        print(f"Canary output: {result.output_dir}")
        if not result.success:
            print(f"FAIL: {result.error}", file=sys.stderr)
            print(f"Inspect stderr: {result.stderr_path}", file=sys.stderr)
            return 1
        print("PASS: one issue completed and produced a valid non-empty model_patch")
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
