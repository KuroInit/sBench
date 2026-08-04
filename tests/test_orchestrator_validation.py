import json
import os
import time
from types import SimpleNamespace

import pytest

from orchestrator import (
    Checkpoint,
    append_sglang_server_flags,
    auto_config_kwargs,
    hf_hub_download_kwargs,
    load_dataset_config,
    load_required_hf_config,
    merged_sglang_server_flags,
    model_precision,
    run_signature,
    sweep_plan,
    validate_config,
    validate_probe_file,
    validate_request_results,
)


def test_partial_request_failures_fail_by_default():
    results = [
        SimpleNamespace(success=True),
        SimpleNamespace(success=False),
    ]
    ok, error = validate_request_results(results, {})
    assert not ok
    assert "1/2" in error


def test_request_success_rate_can_be_relaxed():
    results = [
        SimpleNamespace(success=True),
        SimpleNamespace(success=False),
    ]
    ok, _ = validate_request_results(results, {"min_success_rate": 0.5})
    assert ok


def test_probe_file_must_be_valid_jsonl(tmp_path):
    path = tmp_path / "server_records.jsonl"
    path.write_text("{not-json}\n")
    ok, error = validate_probe_file(path)
    assert not ok
    assert "invalid JSONL" in error


def test_probe_file_requires_schema_fields(tmp_path):
    path = tmp_path / "server_records.jsonl"
    path.write_text(json.dumps({"forward_mode": "prefill", "latency": 1.0}) + "\n")
    ok, error = validate_probe_file(path)
    assert not ok
    assert "missing fields" in error


def test_checkpoint_signature_prevents_stale_skip(tmp_path):
    checkpoint = Checkpoint(str(tmp_path / "checkpoint.yaml"))
    model = {"id": "Qwen/A", "slug": "qwen", "tp": 1}
    hf_config = {"num_hidden_layers": 1, "hidden_size": 8}
    sig_a = run_signature(model, 2, "batched_prefill", {"target_input_tokens": 128}, hf_config)
    sig_b = run_signature(model, 2, "batched_prefill", {"target_input_tokens": 256}, hf_config)
    checkpoint.mark("qwen", 2, "batched_prefill", "success", sig_a, model_id="Qwen/A")
    assert checkpoint.is_done("qwen", 2, "batched_prefill", sig_a)
    assert not checkpoint.is_done("qwen", 2, "batched_prefill", sig_b)


def test_sweep_plan_is_dataset_major():
    config = {
        "batch_sizes": [2, 4],
        "benchmark_types": {"reasoning": ["mmlu_pro"], "chat": ["azure_chat"]},
        "models": [{"slug": "a"}, {"slug": "b"}],
    }
    plan = [(dataset, model["slug"], bs) for dataset, model, bs in sweep_plan(config)]
    assert plan == [
        ("mmlu_pro", "a", 2),
        ("mmlu_pro", "a", 4),
        ("mmlu_pro", "b", 2),
        ("mmlu_pro", "b", 4),
        ("azure_chat", "a", 2),
        ("azure_chat", "a", 4),
        ("azure_chat", "b", 2),
        ("azure_chat", "b", 4),
    ]


def test_validate_config_rejects_hardcoded_model_hf_config():
    config = {
        "batch_sizes": [2],
        "benchmark_types": {"prefill": ["batched_prefill"]},
        "models": [{"id": "Qwen/Test", "slug": "qwen", "hf_config": {"num_hidden_layers": 1}}],
    }
    with pytest.raises(SystemExit, match="must not define hf_config"):
        validate_config(config)


def test_validate_config_rejects_architecture_as_full_config():
    config = {
        "batch_sizes": [2],
        "benchmark_types": {"prefill": ["batched_prefill"]},
        "models": [{"id": "Qwen/Test", "slug": "qwen", "architecture": {"num_hidden_layers": 1}}],
    }
    with pytest.raises(SystemExit, match="component keys only"):
        validate_config(config)


def test_config_loader_options_expand_paths(monkeypatch):
    monkeypatch.setenv("HF_HOME", "/tmp/hf-cache")
    kwargs = auto_config_kwargs({"local_files_only": True, "revision": "main", "cache_dir": "$HF_HOME", "trust_remote_code": False})
    assert kwargs == {
        "trust_remote_code": False,
        "revision": "main",
        "cache_dir": "/tmp/hf-cache",
        "local_files_only": True,
    }


def test_hf_hub_download_options_exclude_auto_config_only_options(monkeypatch):
    monkeypatch.setenv("HF_HOME", "/tmp/hf-cache")
    kwargs = hf_hub_download_kwargs({"local_files_only": True, "cache_dir": "$HF_HOME", "trust_remote_code": True})
    assert kwargs == {"cache_dir": "/tmp/hf-cache", "local_files_only": True}


def test_load_required_hf_config_reads_raw_local_config_for_unknown_model_type(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model_type": "qwen3_5_moe",
                "num_hidden_layers": 48,
                "hidden_size": 4096,
                "num_experts": 128,
            }
        )
    )
    cfg = load_required_hf_config(str(tmp_path))
    assert cfg["model_type"] == "qwen3_5_moe"
    assert cfg["num_experts"] == 128


def test_global_and_model_server_flags_are_merged_for_metadata_and_mini_swe():
    config = {"sglang_server_flags": [{"--dtype": "float16"}, {"--served-model-name": "global-name"}]}
    model = {"id": "Qwen/Test", "slug": "qwen", "sglang_server_flags": [{"--served-model-name": "model-name"}]}
    flags = merged_sglang_server_flags(config, model)
    assert model_precision({"resolved_sglang_server_flags": flags}, {}) == "float16"
    dataset_cfg = load_dataset_config("mini_swe_agent", model, flags)
    assert dataset_cfg["mini_model_name"] == "openai/model-name"


def test_sglang_server_flags_support_strings_and_key_values():
    cmd = ["python"]
    append_sglang_server_flags(
        cmd,
        [
            "--disable-custom-all-reduce",
            {"--watchdog-timeout": 600},
            {"--log-requests": False},
            {"--mem-fraction-static": 0.85},
        ],
    )
    assert cmd == [
        "python",
        "--disable-custom-all-reduce",
        "--watchdog-timeout",
        "600",
        "--mem-fraction-static",
        "0.85",
    ]
