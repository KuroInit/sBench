from pathlib import Path

import pytest

from sbench.mini_swe_agent_runner import build_mini_swe_agent_command, configure_openai_env, expected_prediction_count, mini_swe_outputs_exist, run_mini_swe_agent, validate_mini_swe_predictions


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


def test_issue_count_becomes_a_single_issue_slice(tmp_path):
    command = build_mini_swe_agent_command(
        model_id="Qwen/Test",
        batch_size=1,
        dataset_cfg={"environment_class": "singularity", "issue_count": 1},
        output_dir=tmp_path,
    )
    assert command[command.index("--slice") + 1] == "0:1"


def test_explicit_slice_arg_prevents_duplicate_issue_count_slice(tmp_path):
    command = build_mini_swe_agent_command(
        model_id="Qwen/Test",
        batch_size=1,
        dataset_cfg={"environment_class": "singularity", "issue_count": 1, "extra_args": ["--slice=0:8"]},
        output_dir=tmp_path,
    )
    assert "--slice=0:8" in command
    assert "--slice" not in command


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
    assert env["MSWEA_COST_TRACKING"] == "ignore_errors"


def test_configure_openai_env_allows_cost_tracking_override():
    env = {}
    configure_openai_env(env, "http://127.0.0.1:30000", "Qwen/Test", {"cost_tracking": "enabled"})
    assert env["MSWEA_COST_TRACKING"] == "enabled"


def test_configure_openai_env_overrides_inherited_cost_tracking():
    env = {"MSWEA_COST_TRACKING": "bad-inherited-value"}
    configure_openai_env(env, "http://127.0.0.1:30000", "Qwen/Test", {})
    assert env["MSWEA_COST_TRACKING"] == "ignore_errors"


def test_expected_prediction_count_honors_slice_extra_args():
    assert expected_prediction_count({"extra_args": ["--slice", "0:8"]}) == 8
    assert expected_prediction_count({"extra_args": ["--slice=23"]}) == 1
    assert expected_prediction_count({"issue_count": 3}) == 3


def test_mini_swe_output_validation_requires_valid_submission(tmp_path):
    (tmp_path / "mini_swe_agent_run_1.json").write_text("{}")
    assert not mini_swe_outputs_exist(tmp_path, {})
    (tmp_path / "preds.json").write_text('{"repo__issue": {"instance_id": "repo__issue", "model_patch": "diff --git a/a b/a"}}')
    assert mini_swe_outputs_exist(tmp_path, {})


def test_mini_swe_submission_rejects_missing_patch(tmp_path):
    (tmp_path / "preds.json").write_text('{"repo__issue": {"instance_id": "repo__issue", "model_patch": ""}}')
    assert "no model_patch" in validate_mini_swe_predictions(tmp_path, {"issue_count": 1})


def test_mini_swe_attempt_does_not_accept_parent_artifacts(tmp_path, monkeypatch):
    (tmp_path / "preds.jsonl").write_text("old result\n")

    class CompletedProcess:
        returncode = 0

        def communicate(self, timeout=None):
            return None

    monkeypatch.setattr("sbench.mini_swe_agent_runner.subprocess.Popen", lambda *args, **kwargs: CompletedProcess())
    result = run_mini_swe_agent(
        api_base="http://127.0.0.1:30000",
        model_id="Qwen/Test",
        batch_size=1,
        dataset_cfg={"environment_class": "docker"},
        output_dir=tmp_path,
    )
    assert not result.success
    assert "produced no preds.json submission file" in result.error
    assert Path(result.output_dir).parent == tmp_path
