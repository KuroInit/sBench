import sys
from types import ModuleType, SimpleNamespace

from sbench_probe.sglang_probe import append_probe_error, build_probe_record, install_probe


class Mode:
    name = "EXTEND"
    def is_idle(self): return False
    def is_decode(self): return False
    def is_extend(self): return True


class DecodeMode:
    name = "DECODE"
    def is_idle(self): return False
    def is_decode(self): return True
    def is_extend(self): return False


def test_probe_builds_prefill_record_with_per_req_info():
    runner = SimpleNamespace(forward_pass_id=7, tp_size=2, pp_size=1, tp_rank=0, server_args=SimpleNamespace(chunked_prefill_size=128))
    batch = SimpleNamespace(forward_mode=Mode(), batch_size=2, seq_lens_sum=300, req_pool_indices=[4, 5], extend_seq_lens_cpu=[128, 44], seq_lens_cpu=[128, 172], rids=["a", "b"])
    output = SimpleNamespace(expert_distribution_metrics=SimpleNamespace(average_expert_activation=3.5, expert_utilization=0.2))
    record = build_probe_record(runner, batch, output, 0.25)
    assert record.forward_mode == "prefill"
    assert record.forward_pass_id == 7
    assert record.expert_activation == 3.5
    assert record.processed_tokens == 172
    assert record.per_req_info[0]["is_last_chunk"] is False
    assert record.per_req_info[1]["is_last_chunk"] is True


def test_probe_decode_without_expert_data_is_timing_only_but_analyzable():
    runner = SimpleNamespace(forward_pass_id=8, tp_size=1, pp_size=1, tp_rank=0, server_args=SimpleNamespace())
    batch = SimpleNamespace(forward_mode=DecodeMode(), batch_size=4, seq_lens_sum=400)
    record = build_probe_record(runner, batch, SimpleNamespace(), 0.1)
    assert record.forward_mode == "decode"
    assert record.expert_activation == 0
    assert record.raw_probe_source == "timing_only"
    assert record.processed_tokens == 4


def test_probe_profiling_only_sets_activation_zero():
    runner = SimpleNamespace(forward_pass_id=8, tp_size=1, pp_size=1, tp_rank=0, server_args=SimpleNamespace())
    batch = SimpleNamespace(forward_mode=DecodeMode(), batch_size=4, seq_lens_sum=400)
    record = build_probe_record(runner, batch, SimpleNamespace(), 0.1, profiling_only=True)
    assert record.expert_activation == 0
    assert record.raw_probe_source == "profiling_only"


def test_probe_extracts_activation_from_topk_output():
    runner = SimpleNamespace(forward_pass_id=9, tp_size=1, pp_size=1, tp_rank=0, server_args=SimpleNamespace())
    batch = SimpleNamespace(forward_mode=DecodeMode(), batch_size=2, seq_lens_sum=50)
    output = SimpleNamespace(routed_experts_output=SimpleNamespace(topk=[[1, 2], [2, 3]]))
    record = build_probe_record(runner, batch, output, 0.2)
    assert record.raw_probe_source == "routed_experts_output"
    assert record.expert_activation == 3


def test_probe_extracts_activation_from_indexer_topk_output():
    runner = SimpleNamespace(forward_pass_id=10, tp_size=1, pp_size=1, tp_rank=0, server_args=SimpleNamespace())
    batch = SimpleNamespace(forward_mode=DecodeMode(), batch_size=2, seq_lens_sum=50)
    output = SimpleNamespace(indexer_topk_output={"topk": [[4, 4], [5, -1]]})
    record = build_probe_record(runner, batch, output, 0.2)
    assert record.raw_probe_source == "indexer_topk_output"
    assert record.expert_activation == 2


def test_probe_scales_fractional_expert_distribution_metric():
    runner = SimpleNamespace(forward_pass_id=11, tp_size=1, pp_size=1, tp_rank=0, server_args=SimpleNamespace())
    batch = SimpleNamespace(forward_mode=DecodeMode(), batch_size=4, seq_lens_sum=80)
    metrics = SimpleNamespace(expert_activation=0.05, expert_utilization=0.05, num_experts=60)
    record = build_probe_record(runner, batch, SimpleNamespace(expert_distribution_metrics=metrics), 0.2)
    assert record.raw_probe_source == "expert_distribution_metrics_scaled"
    assert record.expert_activation == 3.0


def test_probe_does_not_treat_utilization_as_activation_count():
    runner = SimpleNamespace(forward_pass_id=12, tp_size=1, pp_size=1, tp_rank=0, server_args=SimpleNamespace())
    batch = SimpleNamespace(forward_mode=DecodeMode(), batch_size=4, seq_lens_sum=80)
    metrics = SimpleNamespace(expert_utilization=0.4, num_experts=60)
    record = build_probe_record(runner, batch, SimpleNamespace(expert_distribution_metrics=metrics), 0.2)
    assert record.raw_probe_source == "timing_only"
    assert record.expert_activation == 0.0


def test_probe_topk_tensor_uses_token_layer_topk_shape():
    runner = SimpleNamespace(forward_pass_id=12, tp_size=1, pp_size=1, tp_rank=0, server_args=SimpleNamespace())
    batch = SimpleNamespace(forward_mode=DecodeMode(), batch_size=2, seq_lens_sum=80)
    # Official routed-expert API shape is [tokens, layers, top_k].
    topk = [
        [[1, 2], [4, 5]],
        [[2, 3], [5, 6]],
    ]
    record = build_probe_record(runner, batch, SimpleNamespace(routed_experts_output={"topk": topk}), 0.2)
    assert record.raw_probe_source == "routed_experts_output"
    assert record.expert_activation == 3.0


def test_probe_ignores_empty_stat_recorder_buffer_slots():
    import torch

    runner = SimpleNamespace(forward_pass_id=13, tp_size=1, pp_size=1, tp_rank=0, server_args=SimpleNamespace())
    batch = SimpleNamespace(forward_mode=DecodeMode(), batch_size=1, seq_lens_sum=8)
    logical_count = torch.zeros((1000, 24, 60), dtype=torch.int32)
    logical_count[0, :, :4] = 1
    output = SimpleNamespace(routed_experts_output={"logical_count": logical_count})
    record = build_probe_record(runner, batch, output, 0.2)
    assert record.expert_activation == 4.0


def test_install_probe_wraps_forward_once_and_preserves_output(tmp_path, monkeypatch):
    path = tmp_path / "server_records.jsonl"
    calls = {"count": 0}

    class FakeModelRunner:
        def __init__(self):
            self.forward_pass_id = 123
            self.tp_size = 1
            self.pp_size = 1
            self.tp_rank = 0
            self.server_args = SimpleNamespace()

        def forward(self, forward_batch):
            calls["count"] += 1
            return SimpleNamespace(answer="original")

    module = ModuleType("sglang.srt.model_executor.model_runner")
    module.ModelRunner = FakeModelRunner
    monkeypatch.setitem(sys.modules, "sglang", ModuleType("sglang"))
    monkeypatch.setitem(sys.modules, "sglang.srt", ModuleType("sglang.srt"))
    monkeypatch.setitem(sys.modules, "sglang.srt.model_executor", ModuleType("sglang.srt.model_executor"))
    monkeypatch.setitem(sys.modules, "sglang.srt.model_executor.model_runner", module)

    assert install_probe(str(path)) is True
    wrapped = FakeModelRunner.forward
    assert install_probe(str(path)) is True
    assert FakeModelRunner.forward is wrapped

    runner = FakeModelRunner()
    batch = SimpleNamespace(forward_mode=DecodeMode(), batch_size=2, seq_lens_sum=20)
    output = runner.forward(batch)
    assert output.answer == "original"
    assert calls["count"] == 1
    lines = path.read_text().splitlines()
    assert len(lines) == 1
    assert '"forward_pass_id": 123' in lines[0]


def test_install_probe_writes_rank0_only(tmp_path, monkeypatch):
    path = tmp_path / "server_records.jsonl"

    class FakeModelRunner:
        tp_size = 2
        pp_size = 1
        tp_rank = 1
        forward_pass_id = 1
        server_args = SimpleNamespace()

        def forward(self, forward_batch):
            return SimpleNamespace()

    module = ModuleType("sglang.srt.model_executor.model_runner")
    module.ModelRunner = FakeModelRunner
    monkeypatch.setitem(sys.modules, "sglang", ModuleType("sglang"))
    monkeypatch.setitem(sys.modules, "sglang.srt", ModuleType("sglang.srt"))
    monkeypatch.setitem(sys.modules, "sglang.srt.model_executor", ModuleType("sglang.srt.model_executor"))
    monkeypatch.setitem(sys.modules, "sglang.srt.model_executor.model_runner", module)

    assert install_probe(str(path)) is True
    FakeModelRunner().forward(SimpleNamespace(forward_mode=DecodeMode(), batch_size=1, seq_lens_sum=20))
    assert not path.exists()


def test_probe_error_writes_sidecar_log(tmp_path):
    path = tmp_path / "server_records.jsonl"
    append_probe_error(str(path), RuntimeError("probe broke"))
    error_path = tmp_path / "server_records.jsonl.errors.log"
    assert "RuntimeError: probe broke" in error_path.read_text()
