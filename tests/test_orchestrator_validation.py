import json
import os
import subprocess
import time
from types import SimpleNamespace

import pytest

from orchestrator import (
    Checkpoint,
    append_sglang_server_flags,
    auto_config_kwargs,
    hf_hub_download_kwargs,
    iter_sglang_server_flags,
    load_dataset_config,
    load_required_hf_config,
    merged_sglang_server_flags,
    model_precision,
    model_uses_moe,
    required_forward_modes,
    resolved_num_gpus,
    run_signature,
    server_startup_timeout,
    required_probe_records,
    start_sglang,
    sweep_plan,
    validate_config,
    validate_probe_file,
    validate_request_results,
    workload_sglang_server_flags,
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


def test_server_startup_timeout_is_configurable():
    assert server_startup_timeout({}) == 1500
    assert server_startup_timeout({"server_startup_timeout_seconds": 3600}) == 3600
    with pytest.raises(SystemExit, match="must be positive"):
        server_startup_timeout({"server_startup_timeout_seconds": 0})


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


def test_agentic_probe_file_enforces_metric_sample_floor(tmp_path):
    path = tmp_path / "server_records.jsonl"
    record = {
        "forward_mode": "decode",
        "latency": 1.0,
        "seq_lens_sum": 100,
        "batch_size": 1,
    }
    path.write_text("".join(json.dumps(record) + "\n" for _ in range(3)))
    assert required_probe_records({"runner": "mini_swe_agent", "metric_sample_steps": 4}) == 4
    ok, error = validate_probe_file(path, minimum_usable_records=4)
    assert not ok
    assert "only 3 usable records" in error


def test_probe_file_requires_prefill_and_decode_when_requested(tmp_path):
    path = tmp_path / "server_records.jsonl"
    path.write_text(json.dumps({"forward_mode": "prefill", "latency": 1.0, "seq_lens_sum": 100, "batch_size": 1}) + "\n")
    ok, error = validate_probe_file(path, required_forward_modes={"prefill", "decode"})
    assert not ok
    assert "decode" in error


def test_prefill_workload_requires_only_prefill_probe_phase():
    assert required_forward_modes({"benchmark_type": "prefill", "target_output_tokens": 1}) == {"prefill"}
    assert required_forward_modes({"benchmark_type": "chat", "target_output_tokens": 1}) == {"prefill", "decode"}
    assert required_forward_modes({}, "batched_prefill") == {"prefill"}


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


def test_dense_model_strips_moe_expert_probe_flags():
    config = {
        "sglang_server_flags": [
            "--disable-custom-all-reduce",
            "--expert-distribution-recorder-mode",
            "--enable-return-routed-experts",
            "--enable-expert-distribution-metrics",
        ]
    }
    model = {"id": "Qwen/Qwen3-8B", "slug": "qwen3_8b", "tp": 4}
    hf_config = {
        "model_type": "qwen3",
        "num_hidden_layers": 36,
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
    }
    flags = merged_sglang_server_flags(config, model, hf_config)
    flat_flags = [flag for flag, _ in iter_sglang_server_flags(flags)]
    assert "--disable-custom-all-reduce" in flat_flags
    assert "--expert-distribution-recorder-mode" not in flat_flags
    assert "--enable-return-routed-experts" not in flat_flags
    assert "--enable-expert-distribution-metrics" not in flat_flags
    assert not model_uses_moe(model, hf_config)


def test_moe_model_gets_routed_expert_capture_flag():
    config = {"sglang_server_flags": ["--disable-custom-all-reduce"]}
    model = {"id": "Qwen/Qwen3-30B-A3B", "slug": "qwen3_30b_a3b", "tp": 4}
    hf_config = {
        "model_type": "qwen3_moe",
        "num_hidden_layers": 48,
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "moe_intermediate_size": 768,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "num_experts": 128,
        "num_experts_per_tok": 8,
    }
    flags = merged_sglang_server_flags(config, model, hf_config)
    flat_flags = [flag for flag, _ in iter_sglang_server_flags(flags)]
    assert "--disable-custom-all-reduce" in flat_flags
    assert "--enable-return-routed-experts" in flat_flags
    assert model_uses_moe(model, hf_config)


def test_resolved_num_gpus_includes_pipeline_parallel_size():
    assert resolved_num_gpus({"tp": 2, "pp": 3}, []) == 6
    assert resolved_num_gpus({"tp": 2}, [{"--pp-size": 4}]) == 8


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


def test_workload_prefix_cache_overrides_global_radix_cache_setting():
    flags = ["--disable-radix-cache", {"--dtype": "bfloat16"}]
    assert "--disable-radix-cache" not in workload_sglang_server_flags(flags, {"prefix_cache": True})
    assert "--disable-radix-cache" in workload_sglang_server_flags([], {"prefix_cache": False})


def test_start_sglang_includes_model_pipeline_parallel_size(monkeypatch):
    captured = {}

    def fake_popen(cmd, env):
        captured["cmd"] = cmd
        return SimpleNamespace()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    start_sglang({"id": "Qwen/Test", "tp": 2, "pp": 3}, 8, 30000, {}, sglang_server_flags=[])
    assert captured["cmd"][captured["cmd"].index("--tp-size") + 1] == "2"
    assert captured["cmd"][captured["cmd"].index("--pp-size") + 1] == "3"


def test_start_sglang_uses_resolved_flags_once(monkeypatch):
    captured = {}

    def fake_popen(cmd, env):
        captured["cmd"] = cmd
        captured["env"] = env
        return SimpleNamespace()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    start_sglang(
        {"id": "Qwen/Test", "tp": 4, "sglang_server_flags": [{"--context-length": 4096}]},
        8,
        30000,
        {},
        sglang_server_flags=[
            {"--context-length": 40960},
            {"--chunked-prefill-size": 16384},
        ],
    )
    assert captured["cmd"].count("--context-length") == 1
    assert captured["cmd"].count("--chunked-prefill-size") == 1
    assert "40960" in captured["cmd"]
    assert "4096" not in captured["cmd"]


def test_start_sglang_deduplicates_tp_size_from_flags(monkeypatch):
    captured = {}

    def fake_popen(cmd, env):
        captured["cmd"] = cmd
        return SimpleNamespace()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    start_sglang({"id": "Qwen/Test", "tp": 2}, 8, 30000, {}, sglang_server_flags=[{"--tp-size": 4}])
    assert captured["cmd"].count("--tp-size") == 1
    assert captured["cmd"][captured["cmd"].index("--tp-size") + 1] == "4"
