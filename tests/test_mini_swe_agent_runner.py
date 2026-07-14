from pathlib import Path

import pytest

from sbench.mini_swe_agent_runner import build_mini_swe_agent_command, configure_openai_env, mini_swe_outputs_exist


def test_builds_docker_command_with_default_openai_model(tmp_path):
    cfg = {"environment_class": "docker", "subset": "lite", "split": "dev", "workers": 2}
    command = build_mini_swe_agent_command(model_id="Qwen/Test", batch_size=4, dataset_cfg=cfg, output_dir=tmp_path)
    assert command[:2] == ["mini-extra", "swebench"]
    assert command[command.index("--model") + 1] == "openai/Qwen/Test"
    assert command[command.index("--environment-class") + 1] == "docker"
    assert command[command.index("--workers") + 1] == "2"


def test_supports_configurable_output_flag(tmp_path):
    cfg = {"environment_class": "docker", "output_flag": "--output-dir"}
    command = build_mini_swe_agent_command(model_id="Qwen/Test", batch_size=1, dataset_cfg=cfg, output_dir=tmp_path)
    assert command[command.index("--output-dir") + 1] == str(tmp_path)


def test_builds_singularity_command_and_instance_filter(tmp_path):
    cfg = {
        "environment_class": "singularity",
        "subset": "verified",
        "split": "test",
        "instance_ids": ["sympy__sympy-15599"],
        "mini_model_name": "openai/local-model",
    }
    command = build_mini_swe_agent_command(model_id="Qwen/Test", batch_size=1, dataset_cfg=cfg, output_dir=Path(tmp_path))
    assert command[command.index("--environment-class") + 1] == "singularity"
    assert command[command.index("--model") + 1] == "openai/local-model"
    assert command[-2:] == ["-i", "sympy__sympy-15599"]


def test_rejects_unknown_environment_class(tmp_path):
    with pytest.raises(ValueError, match="environment_class"):
        build_mini_swe_agent_command(model_id="Qwen/Test", batch_size=1, dataset_cfg={"environment_class": "podman"}, output_dir=tmp_path)


def test_configure_openai_env_points_at_local_sglang():
    env = {}
    configure_openai_env(env, "http://127.0.0.1:30000", "Qwen/Test", {})
    assert env["OPENAI_API_BASE"] == "http://127.0.0.1:30000/v1"
    assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:30000/v1"
    assert env["OPENAI_API_KEY"] == "EMPTY"
    assert env["SBENCH_MINI_MODEL"] == "openai/Qwen/Test"


def test_mini_swe_output_validation_ignores_internal_logs(tmp_path):
    (tmp_path / "mini_swe_agent_run_1.json").write_text("{}")
    assert not mini_swe_outputs_exist(tmp_path, {})
    (tmp_path / "preds.jsonl").write_text("{}\n")
    assert mini_swe_outputs_exist(tmp_path, {})
