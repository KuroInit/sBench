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
    command = build_mini_swe_agent_command(
        model_id=model_id,
        batch_size=batch_size,
        dataset_cfg=dataset_cfg,
        output_dir=output_dir,
    )
    configure_openai_env(mini_env, api_base, model_id, dataset_cfg)

    stdout_path = output_dir / f"mini_swe_agent_stdout_{timestamp()}.log"
    stderr_path = output_dir / f"mini_swe_agent_stderr_{timestamp()}.log"
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
                result = MiniSweAgentResult(False, command, 124, str(output_dir), str(stdout_path), str(stderr_path), f"mini-SWE-agent timed out after {exc.timeout}s")
                write_mini_swe_agent_metadata(output_dir, dataset_cfg, result)
                return result
    except FileNotFoundError as exc:
        result = MiniSweAgentResult(False, command, 127, str(output_dir), str(stdout_path), str(stderr_path), str(exc))
        write_mini_swe_agent_metadata(output_dir, dataset_cfg, result)
        return result

    error = "" if proc.returncode == 0 else f"mini-SWE-agent exited with code {proc.returncode}"
    if proc.returncode == 0 and not mini_swe_outputs_exist(output_dir, dataset_cfg):
        error = "mini-SWE-agent exited successfully but produced no expected output artifacts"
    result = MiniSweAgentResult(proc.returncode == 0, command, proc.returncode, str(output_dir), str(stdout_path), str(stderr_path), error)
    if error:
        result = MiniSweAgentResult(False, command, proc.returncode, str(output_dir), str(stdout_path), str(stderr_path), error)
    write_mini_swe_agent_metadata(output_dir, dataset_cfg, result)
    return result


def build_mini_swe_agent_command(
    *,
    model_id: str,
    batch_size: int,
    dataset_cfg: dict[str, Any],
    output_dir: Path,
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
    instance_ids = dataset_cfg.get("instance_ids")
    if instance_ids:
        for instance_id in instance_ids:
            command += ["-i", str(instance_id)]
    extra_args = dataset_cfg.get("extra_args") or []
    if extra_args:
        command.extend(str(arg) for arg in extra_args)
    return command


def configure_openai_env(env: dict[str, str], api_base: str, model_id: str, dataset_cfg: dict[str, Any]) -> None:
    env["OPENAI_API_BASE"] = str(dataset_cfg.get("openai_api_base") or f"{api_base.rstrip('/')}/v1")
    env["OPENAI_BASE_URL"] = env["OPENAI_API_BASE"]
    env.setdefault("OPENAI_API_KEY", str(dataset_cfg.get("openai_api_key") or os.environ.get("OPENAI_API_KEY") or "EMPTY"))
    env.setdefault("SBENCH_MINI_MODEL", str(dataset_cfg.get("mini_model_name") or f"openai/{model_id}"))


def mini_swe_outputs_exist(output_dir: Path, dataset_cfg: dict[str, Any]) -> bool:
    if not bool(dataset_cfg.get("require_output_artifacts", True)):
        return True
    globs = dataset_cfg.get("expected_output_globs") or ["*.json", "*.jsonl", "*.traj", "*.patch", "preds*", "predictions*"]
    ignored_prefixes = ("mini_swe_agent_run_", "mini_swe_agent_stdout_", "mini_swe_agent_stderr_")
    for pattern in globs:
        for path in output_dir.rglob(str(pattern)):
            if path.is_file() and not path.name.startswith(ignored_prefixes) and path.stat().st_size > 0:
                return True
    return False


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
