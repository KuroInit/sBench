from sbench.mini_swe_prewarm import docker_safe_instance_id, issue_count_slice, parse_slice, resolve_instance_ids, slice_from_extra_args, swebench_image


def test_extracts_slice_from_extra_args():
    assert slice_from_extra_args(["--slice", "0:8", "--redo-existing"]) == "0:8"
    assert slice_from_extra_args(["--slice=2:4"]) == "2:4"


def test_parses_slice_values():
    assert parse_slice("0:8") == slice(0, 8)
    assert parse_slice("3") == slice(3, 4)


def test_issue_count_limits_prewarm_to_the_runner_selection(monkeypatch):
    captured = {}

    def load_ids(_dataset, _split, selection):
        captured["selection"] = selection
        return ["repo__issue"]

    monkeypatch.setattr("sbench.mini_swe_prewarm.load_instance_ids", load_ids)
    assert issue_count_slice(1) == "0:1"
    assert resolve_instance_ids({"subset": "lite", "issue_count": 1}) == ["repo__issue"]
    assert captured["selection"] == slice(0, 1)


def test_swebench_docker_image_name():
    instance_id = "sympy__sympy-15599"
    assert docker_safe_instance_id(instance_id) == "sympy_1776_sympy-15599"
    assert swebench_image(instance_id) == "docker.io/swebench/sweb.eval.x86_64.sympy_1776_sympy-15599:latest"
