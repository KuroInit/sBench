import csv
import json
from pathlib import Path

from sbench.validation import leaf_dirs, run_validation


def test_run_validation_recomputes_component_and_moe_cap(tmp_path):
    leaf = tmp_path / "qwen3_30b" / "bs2" / "mmlu_pro" / "Qwen__Qwen3-30B-A3B"
    leaf.mkdir(parents=True)
    _write_metadata(leaf / "metadata_mmlu_pro.json")
    (leaf / "server_records_mmlu_pro.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "forward_pass_id": 1,
                        "forward_mode": "prefill",
                        "latency": 0.1,
                        "seq_lens_sum": 64,
                        "processed_tokens": 64,
                        "batch_size": 2,
                        "expert_activation": 4,
                        "raw_probe_source": "recorder_dump",
                        "per_req_info": [{"extend_len": 32, "total_len": 32}, {"extend_len": 32, "total_len": 32}],
                    }
                ),
                json.dumps(
                    {
                        "forward_pass_id": 2,
                        "forward_mode": "decode",
                        "latency": 0.01,
                        "seq_lens_sum": 64,
                        "batch_size": 2,
                        "expert_activation": 4,
                        "raw_probe_source": "recorder_dump",
                    }
                ),
            ]
        )
    )

    result = run_validation(tmp_path, out_dir=tmp_path / "validation")

    assert result.summary_path.exists()
    assert result.estimator_comparison_path.exists()
    rows = list(csv.DictReader(result.estimator_comparison_path.open()))
    assert {row["metric"] for row in rows} >= {"prefill_smfu", "decoding_smbu", "kv_size"}
    prefill = next(row for row in rows if row["metric"] == "prefill_smfu")
    assert prefill["component_wise"]
    assert prefill["moe_cap"]
    assert prefill["ratio_component_over_moe_cap"]


def test_run_validation_keeps_component_rows_when_moe_cap_lacks_activation(tmp_path):
    leaf = tmp_path / "qwen3_30b" / "bs2" / "mmlu_pro" / "Qwen__Qwen3-30B-A3B"
    leaf.mkdir(parents=True)
    _write_metadata(leaf / "metadata_mmlu_pro.json")
    (leaf / "server_records_mmlu_pro.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"forward_pass_id": 1, "forward_mode": "prefill", "latency": 0.1, "seq_lens_sum": 64, "processed_tokens": 64, "batch_size": 2, "expert_activation": 0, "raw_probe_source": "timing_only", "per_req_info": [{"extend_len": 32, "total_len": 32}, {"extend_len": 32, "total_len": 32}]}),
                json.dumps({"forward_pass_id": 2, "forward_mode": "decode", "latency": 0.01, "seq_lens_sum": 64, "batch_size": 2, "expert_activation": 0, "raw_probe_source": "timing_only"}),
            ]
        )
    )

    result = run_validation(tmp_path, out_dir=tmp_path / "validation")

    rows = list(csv.DictReader(result.estimator_comparison_path.open()))
    assert rows
    assert all(row["component_wise"] for row in rows if row["metric"] in {"prefill_smfu", "decoding_smfu"})
    assert all(row["moe_cap_available"] == "False" for row in rows)
    summary = json.loads(result.summary_path.read_text())
    assert summary["successful_runs"] == 1


def test_run_validation_compares_optional_profiler_summary(tmp_path):
    leaf = tmp_path / "qwen3_30b" / "bs2" / "mmlu_pro" / "Qwen__Qwen3-30B-A3B"
    leaf.mkdir(parents=True)
    _write_metadata(leaf / "metadata_mmlu_pro.json")
    (leaf / "server_records_mmlu_pro.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"forward_pass_id": 1, "forward_mode": "prefill", "latency": 0.1, "seq_lens_sum": 64, "processed_tokens": 64, "batch_size": 2, "expert_activation": 4, "raw_probe_source": "recorder_dump", "per_req_info": [{"extend_len": 32, "total_len": 32}, {"extend_len": 32, "total_len": 32}]}),
                json.dumps({"forward_pass_id": 2, "forward_mode": "decode", "latency": 0.01, "seq_lens_sum": 64, "batch_size": 2, "expert_activation": 4, "raw_probe_source": "recorder_dump"}),
            ]
        )
    )
    profiler = tmp_path / "profiler_summary.csv"
    profiler.write_text("slug,batch_size,dataset,phase,profiled_smfu,profiled_smbu\nqwen3_30b,2,mmlu_pro,prefill,1.0,2.0\n")

    result = run_validation(tmp_path, out_dir=tmp_path / "validation", profiler_summary=profiler)

    assert result.profiler_comparison_path is not None
    rows = list(csv.DictReader(result.profiler_comparison_path.open()))
    assert len(rows) == 1
    assert rows[0]["profiled_smfu"] == "1.0"
    assert rows[0]["estimated_smfu"]


def test_run_validation_compares_optional_lightweight_telemetry(tmp_path):
    leaf = tmp_path / "qwen3_30b" / "bs2" / "mmlu_pro" / "Qwen__Qwen3-30B-A3B"
    leaf.mkdir(parents=True)
    _write_metadata(leaf / "metadata_mmlu_pro.json")
    (leaf / "server_records_mmlu_pro.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"forward_pass_id": 1, "forward_mode": "prefill", "latency": 0.1, "seq_lens_sum": 64, "processed_tokens": 64, "batch_size": 2, "expert_activation": 4, "raw_probe_source": "recorder_dump", "per_req_info": [{"extend_len": 32, "total_len": 32}, {"extend_len": 32, "total_len": 32}]}),
                json.dumps({"forward_pass_id": 2, "forward_mode": "decode", "latency": 0.01, "seq_lens_sum": 64, "batch_size": 2, "expert_activation": 4, "raw_probe_source": "recorder_dump"}),
            ]
        )
    )
    telemetry = tmp_path / "telemetry_summary.csv"
    telemetry.write_text(
        "slug,batch_size,dataset,phase,DCGM_FI_PROF_SM_ACTIVE,DCGM_FI_PROF_DRAM_ACTIVE\n"
        "qwen3_30b,2,mmlu_pro,prefill,0.42,0.18\n"
    )

    result = run_validation(tmp_path, out_dir=tmp_path / "validation", telemetry_summary=telemetry)

    assert result.telemetry_comparison_path is not None
    rows = list(csv.DictReader(result.telemetry_comparison_path.open()))
    assert len(rows) == 1
    assert rows[0]["observed_sm_util_pct"] == "42.0"
    assert rows[0]["observed_memory_util_pct"] == "18.0"
    assert rows[0]["component_smfu"]


def test_leaf_dirs_ignores_validation_and_nested_component_outputs(tmp_path):
    valid = tmp_path / "qwen3_30b" / "bs2" / "mmlu_pro" / "Qwen__Qwen3-30B-A3B"
    validation = tmp_path / "validation" / "qwen3_30b" / "bs2" / "mmlu_pro" / "Qwen__Qwen3-30B-A3B"
    nested_component = tmp_path / "component" / "qwen3_30b" / "bs2" / "mmlu_pro" / "Qwen__Qwen3-30B-A3B"
    for path in (valid, validation, nested_component):
        path.mkdir(parents=True)
        _write_metadata(path / "metadata_mmlu_pro.json")

    assert leaf_dirs(tmp_path) == [valid]
    assert leaf_dirs(tmp_path / "component") == [nested_component]


def _write_metadata(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "model_config": {"model_name": "Qwen/Qwen3-30B-A3B", "precision": "bfloat16"},
                "hardware": {"num_gpus": 1, "gpu_type": "NVIDIA A100-SXM4-40GB"},
                "architecture_overrides": {},
                "hf_config": {
                    "model_type": "qwen3_moe",
                    "num_hidden_layers": 4,
                    "hidden_size": 64,
                    "intermediate_size": 128,
                    "moe_intermediate_size": 32,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "head_dim": 16,
                    "num_experts": 8,
                    "num_experts_per_tok": 2,
                    "decoder_sparse_step": 1,
                    "mlp_only_layers": [],
                },
                "dataset_config": {"benchmark_type": "reasoning", "dataset_names": ["mmlu_pro"]},
            }
        )
    )
