import json
import os
import time
from types import SimpleNamespace

from orchestrator import Checkpoint, run_signature, sweep_plan, validate_probe_file, validate_request_results


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
    sig_a = run_signature(model, 2, "batched_prefill", {"target_input_tokens": 128})
    sig_b = run_signature(model, 2, "batched_prefill", {"target_input_tokens": 256})
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
