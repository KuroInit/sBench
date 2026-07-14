from sbench.mini_swe_prewarm import docker_safe_instance_id, parse_slice, slice_from_extra_args, swebench_image


def test_extracts_slice_from_extra_args():
    assert slice_from_extra_args(["--slice", "0:8", "--redo-existing"]) == "0:8"
    assert slice_from_extra_args(["--slice=2:4"]) == "2:4"


def test_parses_slice_values():
    assert parse_slice("0:8") == slice(0, 8)
    assert parse_slice("3") == slice(3, 4)


def test_swebench_docker_image_name():
    instance_id = "sympy__sympy-15599"
    assert docker_safe_instance_id(instance_id) == "sympy_1776_sympy-15599"
    assert swebench_image(instance_id) == "docker.io/swebench/sweb.eval.x86_64.sympy_1776_sympy-15599:latest"
