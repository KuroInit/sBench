"""mini-SWE-agent integration for real agentic SWE-bench runs."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SUPPORTED_ENVIRONMENT_CLASSES = {"docker", "singularity"}


@dataclass(frozen=True)
class MiniSweAgentResult:
    success: bool
    command: list[str]
    returncode: int
    output_dir: str
    stdout_path: str
    stderr_path: str
    error: str = ""


def run_mini_swe_agent(
    *,
    api_base: str,
    model_id: str,
    batch_size: int,
    dataset_cfg: dict[str, Any],
    output_dir: Path,
    env: dict[str, str] | None = None,
) -> MiniSweAgentResult:
    """Run mini-SWE-agent's SWE-bench entrypoint against the local model server.

    The runner intentionally shells out to mini-SWE-agent rather than importing
    private APIs. This keeps the integration stable across mini-SWE-agent
    releases and lets users choose Docker or Singularity in YAML.
    """

    mini_env = os.environ.copy()
    if env:
        mini_env.update(env)

    output_dir.mkdir(parents=True, exist_ok=True)
    # Each invocation owns a fresh directory. Reusing task artifacts lets a
    # failed retry be mistaken for a successful one.
    attempt_dir = output_dir / f"attempt_{time.time_ns()}"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    openai_api_base = resolved_openai_api_base(api_base, dataset_cfg)
    configure_openai_env(mini_env, api_base, model_id, dataset_cfg)
    command = build_mini_swe_agent_command(
        model_id=model_id,
        batch_size=batch_size,
        dataset_cfg=dataset_cfg,
        output_dir=attempt_dir,
        openai_api_base=openai_api_base,
    )

    stdout_path = attempt_dir / f"mini_swe_agent_stdout_{timestamp()}.log"
    stderr_path = attempt_dir / f"mini_swe_agent_stderr_{timestamp()}.log"
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            proc = subprocess.Popen(
                command,
                env=mini_env,
                stdout=stdout,
                stderr=stderr,
                text=True,
                start_new_session=True,
            )
            try:
                proc.communicate(timeout=int(dataset_cfg.get("timeout_seconds", 0)) or None)
            except subprocess.TimeoutExpired as exc:
                _terminate_process_group(proc)
                result = MiniSweAgentResult(False, command, 124, str(attempt_dir), str(stdout_path), str(stderr_path), f"mini-SWE-agent timed out after {exc.timeout}s")
                write_mini_swe_agent_metadata(attempt_dir, dataset_cfg, result)
                return result
    except FileNotFoundError as exc:
        result = MiniSweAgentResult(False, command, 127, str(attempt_dir), str(stdout_path), str(stderr_path), str(exc))
        write_mini_swe_agent_metadata(attempt_dir, dataset_cfg, result)
        return result

    error = "" if proc.returncode == 0 else f"mini-SWE-agent exited with code {proc.returncode}"
    if proc.returncode == 0:
        error = validate_mini_swe_predictions(attempt_dir, dataset_cfg)
    result = MiniSweAgentResult(proc.returncode == 0, command, proc.returncode, str(attempt_dir), str(stdout_path), str(stderr_path), error)
    if error:
        result = MiniSweAgentResult(False, command, proc.returncode, str(attempt_dir), str(stdout_path), str(stderr_path), error)
    write_mini_swe_agent_metadata(attempt_dir, dataset_cfg, result)
    return result


def build_mini_swe_agent_command(
    *,
    model_id: str,
    batch_size: int,
    dataset_cfg: dict[str, Any],
    output_dir: Path,
    openai_api_base: str | None = None,
) -> list[str]:
    env_class = str(dataset_cfg.get("environment_class", "docker")).lower()
    if env_class not in SUPPORTED_ENVIRONMENT_CLASSES:
        supported = ", ".join(sorted(SUPPORTED_ENVIRONMENT_CLASSES))
        raise ValueError(f"unsupported mini-SWE-agent environment_class={env_class!r}; expected one of: {supported}")

    binary = str(dataset_cfg.get("binary") or os.environ.get("SBENCH_MINI_SWE_AGENT_BIN") or "mini-extra")
    command = [
        binary,
        "swebench",
        "--model",
        str(dataset_cfg.get("mini_model_name") or os.environ.get("SBENCH_MINI_MODEL") or f"openai/{model_id}"),
        "--subset",
        str(dataset_cfg.get("subset", "lite")),
        "--split",
        str(dataset_cfg.get("split", "dev")),
        "--workers",
        str(dataset_cfg.get("workers", batch_size)),
        "--environment-class",
        env_class,
        str(dataset_cfg.get("output_flag", "--output")),
        str(output_dir),
    ]
    model_class = dataset_cfg.get("model_class")
    if model_class:
        command.extend(["--model-class", str(model_class)])
    for config_name in mini_swe_config_args(dataset_cfg):
        command.extend(["-c", config_name])
    for config_spec in mini_swe_local_model_config_args(dataset_cfg, openai_api_base):
        command.extend(["-c", config_spec])
    instance_ids = dataset_cfg.get("instance_ids")
    if instance_ids:
        for instance_id in instance_ids:
            command += ["-i", str(instance_id)]
    extra_args = dataset_cfg.get("extra_args") or []
    if dataset_cfg.get("issue_count") is not None and not instance_ids and slice_from_extra_args(extra_args) is None:
        command.extend(["--slice", f"0:{int(dataset_cfg['issue_count'])}"])
    if extra_args:
        command.extend(str(arg) for arg in extra_args)
    return command


def mini_swe_config_args(dataset_cfg: dict[str, Any]) -> list[str]:
    """Return mini-SWE config files to load via repeated ``-c`` flags."""

    values = []
    single = dataset_cfg.get("mini_swe_config")
    multiple = dataset_cfg.get("mini_swe_configs")
    if single:
        values.append(single)
    if multiple:
        if isinstance(multiple, (str, os.PathLike)):
            values.append(multiple)
        else:
            values.extend(multiple)
    return [str(value) for value in values if str(value).strip()]


def mini_swe_local_model_config_args(dataset_cfg: dict[str, Any], openai_api_base: str | None) -> list[str]:
    """Return mini-SWE config overrides for a local OpenAI-compatible server."""

    if not bool(dataset_cfg.get("local_openai", True)) or not openai_api_base:
        return []
    provider = str(dataset_cfg.get("custom_llm_provider") or dataset_cfg.get("litellm_provider") or "openai")
    return [
        f"model.model_kwargs.custom_llm_provider={provider}",
        f"model.model_kwargs.api_base={openai_api_base}",
    ]


def resolved_openai_api_base(api_base: str, dataset_cfg: dict[str, Any]) -> str:
    return str(dataset_cfg.get("openai_api_base") or f"{api_base.rstrip('/')}/v1")


def configure_openai_env(env: dict[str, str], api_base: str, model_id: str, dataset_cfg: dict[str, Any]) -> None:
    env["OPENAI_API_BASE"] = resolved_openai_api_base(api_base, dataset_cfg)
    env["OPENAI_BASE_URL"] = env["OPENAI_API_BASE"]
    env.setdefault("OPENAI_API_KEY", str(dataset_cfg.get("openai_api_key") or os.environ.get("OPENAI_API_KEY") or "EMPTY"))
    env.setdefault("SBENCH_MINI_MODEL", str(dataset_cfg.get("mini_model_name") or f"openai/{model_id}"))
    env["MSWEA_COST_TRACKING"] = str(dataset_cfg.get("cost_tracking") or "ignore_errors")


def mini_swe_outputs_exist(output_dir: Path, dataset_cfg: dict[str, Any]) -> bool:
    return not validate_mini_swe_predictions(output_dir, dataset_cfg)


def validate_mini_swe_predictions(output_dir: Path, dataset_cfg: dict[str, Any]) -> str:
    """Validate mini-SWE batch-mode submissions rather than arbitrary artifacts.

    A valid prediction proves a patch was submitted, not that it passes the
    official SWE-Bench evaluator. That distinction keeps performance tracing
    honest while allowing correctness evaluation as a separate workload.
    """

    if not bool(dataset_cfg.get("require_output_artifacts", True)):
        return ""
    predictions_path = _prediction_path(output_dir, dataset_cfg)
    if predictions_path is None:
        return "mini-SWE-agent exited successfully but produced no preds.json submission file"
    try:
        payload = json.loads(predictions_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return f"mini-SWE-agent produced invalid predictions JSON: {exc}"
    if not isinstance(payload, dict) or not payload:
        return "mini-SWE-agent predictions file is empty"

    expected_ids = {str(value) for value in dataset_cfg.get("instance_ids") or []}
    expected_count = expected_prediction_count(dataset_cfg)
    if len(payload) != expected_count:
        return f"mini-SWE-agent produced {len(payload)} prediction(s); expected {expected_count}"
    if expected_ids and set(map(str, payload)) != expected_ids:
        return "mini-SWE-agent predictions do not match configured instance_ids"
    for instance_id, prediction in payload.items():
        if not isinstance(prediction, dict):
            return f"prediction for {instance_id!r} is not an object"
        if str(prediction.get("instance_id", "")) != str(instance_id):
            return f"prediction key and instance_id disagree for {instance_id!r}"
        if not str(prediction.get("model_patch", "")).strip():
            return f"prediction for {instance_id!r} has no model_patch submission"
    return ""


def expected_prediction_count(dataset_cfg: dict[str, Any]) -> int:
    expected_ids = dataset_cfg.get("instance_ids") or []
    if expected_ids:
        return len(expected_ids)
    slice_value = slice_from_extra_args(dataset_cfg.get("extra_args") or [])
    if slice_value:
        count = count_from_slice(slice_value)
        if count is not None:
            return count
    return int(dataset_cfg.get("issue_count", 1))


def slice_from_extra_args(extra_args: list[Any]) -> str | None:
    args = [str(arg) for arg in extra_args]
    for idx, arg in enumerate(args):
        if arg == "--slice" and idx + 1 < len(args):
            return args[idx + 1]
        if arg.startswith("--slice="):
            return arg.split("=", 1)[1]
    return None


def count_from_slice(value: str) -> int | None:
    if ":" not in value:
        return 1
    start_text, stop_text = value.split(":", 1)
    if not stop_text:
        return None
    start = int(start_text) if start_text else 0
    stop = int(stop_text)
    return max(stop - start, 0)


def _prediction_path(output_dir: Path, dataset_cfg: dict[str, Any]) -> Path | None:
    configured = dataset_cfg.get("predictions_file")
    candidates = [output_dir / str(configured)] if configured else sorted(output_dir.glob("preds*.json"))
    return next((path for path in candidates if path.is_file() and path.stat().st_size > 0), None)


def _terminate_process_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=30)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            proc.kill()
        proc.wait(timeout=30)


def write_mini_swe_agent_metadata(output_dir: Path, dataset_cfg: dict[str, Any], result: MiniSweAgentResult) -> None:
    payload = {"runner": "mini_swe_agent", "dataset_config": dataset_cfg, "result": asdict(result)}
    (output_dir / f"mini_swe_agent_run_{timestamp()}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")
