import json
import subprocess
import sys
import os
from pathlib import Path


def test_analyze_writes_raw_values_for_synthetic_probe(tmp_path):
    leaf = tmp_path / "qwen" / "bs2" / "batched_prefill" / "Qwen" / "Model"
    leaf.mkdir(parents=True)
    meta = {
        "model_config": {"model_name": "Qwen/Test", "precision": "bfloat16"},
        "hardware": {"num_gpus": 1, "gpu_type": "NVIDIA-A100-SXM4-40GB"},
        "hf_config": {"num_hidden_layers": 1, "hidden_size": 8, "num_attention_heads": 1, "head_dim": 8, "intermediate_size": 16},
        "architecture_overrides": {},
    }
    (leaf / "metadata_batched_prefill_1.json").write_text(json.dumps(meta))
    (leaf / "server_records_batched_prefill_1.jsonl").write_text(json.dumps({"forward_pass_id": 1, "forward_mode": "prefill", "latency": 1.0, "seq_lens_sum": 100, "batch_size": 1, "expert_activation": 0}) + "\n")
    env = os.environ.copy()
    env.pop("ANALYZE_GPU_TYPE", None)
    subprocess.check_call([sys.executable, str(Path(__file__).parents[1] / "analyze.py"), str(tmp_path)], env=env)
    raw = (tmp_path / "raw_values.csv").read_text()
    assert "prefill_smfu" in raw
    assert "batched_prefill" in raw


def test_analyze_reports_newer_failure_after_metadata(tmp_path):
    leaf = tmp_path / "qwen" / "bs2" / "batched_prefill" / "Qwen" / "Model"
    leaf.mkdir(parents=True)
    meta = {
        "model_config": {"model_name": "Qwen/Test", "precision": "bfloat16"},
        "hardware": {"num_gpus": 1, "gpu_type": "NVIDIA-A100-SXM4-40GB"},
        "hf_config": {"num_hidden_layers": 1, "hidden_size": 8, "num_attention_heads": 1, "head_dim": 8, "intermediate_size": 16},
        "architecture_overrides": {},
    }
    meta_path = leaf / "metadata_batched_prefill_1.json"
    fail_path = leaf / "failure_batched_prefill_2.json"
    meta_path.write_text(json.dumps(meta))
    fail_path.write_text(json.dumps({"dataset": "batched_prefill", "slug": "qwen", "batch_size": 2, "status": "failed", "error": "probe produced no server_records file"}))
    os.utime(meta_path, (time := 1_700_000_000, time))
    os.utime(fail_path, (time + 1, time + 1))
    env = os.environ.copy()
    env.pop("ANALYZE_GPU_TYPE", None)
    subprocess.check_call([sys.executable, str(Path(__file__).parents[1] / "analyze.py"), str(tmp_path)], env=env)
    raw = (tmp_path / "raw_values.csv").read_text()
    assert "probe produced no server_records file" in raw


def test_analyze_reports_unknown_gpu_as_failed_row(tmp_path):
    leaf = tmp_path / "qwen" / "bs2" / "batched_prefill" / "Qwen" / "Model"
    leaf.mkdir(parents=True)
    meta = {
        "model_config": {"model_name": "Qwen/Test", "precision": "bfloat16"},
        "hardware": {"num_gpus": 1, "gpu_type": "unknown"},
        "hf_config": {"num_hidden_layers": 1, "hidden_size": 8, "num_attention_heads": 1, "head_dim": 8, "intermediate_size": 16},
        "architecture_overrides": {},
    }
    (leaf / "metadata_batched_prefill_1.json").write_text(json.dumps(meta))
    (leaf / "server_records_batched_prefill_1.jsonl").write_text(json.dumps({"forward_pass_id": 1, "forward_mode": "prefill", "latency": 1.0, "seq_lens_sum": 100, "batch_size": 1, "expert_activation": 0}) + "\n")
    env = os.environ.copy()
    env.pop("ANALYZE_GPU_TYPE", None)
    subprocess.check_call([sys.executable, str(Path(__file__).parents[1] / "analyze.py"), str(tmp_path)], env=env)
    raw = (tmp_path / "raw_values.csv").read_text()
    assert "unknown GPU type" in raw


def test_analyze_honors_moe_cap_estimator_mode(tmp_path):
    leaf = tmp_path / "qwen" / "bs2" / "batched_prefill" / "Qwen" / "Model"
    leaf.mkdir(parents=True)
    sweep = tmp_path / "sweep.yaml"
    sweep.write_text("estimator_mode: moe-cap\n")
    meta = {
        "model_config": {"model_name": "Qwen/Test-MoE", "precision": "bfloat16"},
        "hardware": {"num_gpus": 1, "gpu_type": "NVIDIA-A100-SXM4-40GB"},
        "hf_config": {
            "model_type": "qwen2_moe",
            "num_hidden_layers": 2,
            "hidden_size": 16,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "intermediate_size": 32,
            "moe_intermediate_size": 4,
            "shared_expert_intermediate_size": 4,
            "num_experts": 8,
            "num_experts_per_tok": 2,
        },
        "architecture_overrides": {},
    }
    (leaf / "metadata_batched_prefill_1.json").write_text(json.dumps(meta))
    (leaf / "server_records_batched_prefill_1.jsonl").write_text(json.dumps({"forward_pass_id": 1, "forward_mode": "prefill", "latency": 1.0, "seq_lens_sum": 100, "batch_size": 1, "processed_tokens": 100, "expert_activation": 2, "raw_probe_source": "expert_distribution_metrics"}) + "\n")
    env = os.environ.copy()
    env.pop("ANALYZE_GPU_TYPE", None)
    env["SWEEP_CONFIG"] = str(sweep)
    subprocess.check_call([sys.executable, str(Path(__file__).parents[1] / "analyze.py"), str(tmp_path)], env=env)
    raw = (tmp_path / "raw_values.csv").read_text()
    assert "moe-cap" in raw
