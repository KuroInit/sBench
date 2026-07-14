"""Thin newer-SGLang forward probe.

The probe wraps ModelRunner.forward but calls SGLang's original implementation
unchanged. It records latency and stable batch metadata after the original
forward returns.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .record_schema import ProbeRecord

_ORIGINAL_FORWARD = None
_PREFILL_REQ_COUNTER = 0


def install_probe(output_path: str | None = None, profiling_only: bool | None = None) -> bool:
    """Install the SGLang ModelRunner.forward wrapper.

    Returns True when SGLang was importable and the wrapper was installed.
    Unit tests can call `build_probe_record` directly without SGLang.
    """
    global _ORIGINAL_FORWARD
    try:
        from sglang.srt.model_executor.model_runner import ModelRunner
    except Exception:
        return False
    if getattr(ModelRunner.forward, "_sbench_probe", False):
        return True
    _ORIGINAL_FORWARD = ModelRunner.forward
    path = output_path or os.environ.get("SBENCH_PROBE_RECORD_PATH", "probe_records.jsonl")
    explicit_profiling = _profiling_only_enabled() if profiling_only is None else profiling_only

    def probed_forward(self, forward_batch, *args, **kwargs):
        start = _sync_and_time()
        output = _ORIGINAL_FORWARD(self, forward_batch, *args, **kwargs)
        latency = _sync_and_time() - start
        try:
            record = build_probe_record(self, forward_batch, output, latency, profiling_only=explicit_profiling)
            if record is not None and _is_rank0(self):
                append_record(path, record)
        except Exception:
            pass
        return output

    probed_forward._sbench_probe = True
    ModelRunner.forward = probed_forward
    return True


def build_probe_record(model_runner: Any, forward_batch: Any, output: Any, latency: float, *, profiling_only: bool = False) -> ProbeRecord | None:
    mode, raw_mode = _classify_forward_mode(forward_batch)
    if mode == "idle":
        return None
    batch_size = int(getattr(forward_batch, "batch_size", 0) or 0)
    seq_lens_sum = int(getattr(forward_batch, "seq_lens_sum", 0) or 0)
    per_req_info = _build_per_req_info(forward_batch, getattr(model_runner, "server_args", None)) if mode == "prefill" else []
    processed = sum(int(item.get("extend_len", 0)) for item in per_req_info) or _out_cache_tokens(forward_batch)
    if not processed and mode == "decode":
        processed = batch_size
    activation, utilization, source = _extract_expert_activation(output, profiling_only=profiling_only)
    return ProbeRecord(
        forward_pass_id=int(getattr(model_runner, "forward_pass_id", 0) or 0),
        forward_mode=mode,
        latency=float(latency),
        seq_lens_sum=seq_lens_sum,
        batch_size=batch_size,
        num_gpus=int(getattr(model_runner, "tp_size", 1) or 1) * int(getattr(model_runner, "pp_size", 1) or 1),
        gpu_num=int(getattr(model_runner, "tp_size", 1) or 1) * int(getattr(model_runner, "pp_size", 1) or 1),
        gpu_raw_type=os.environ.get("SBENCH_GPU_TYPE", "unknown"),
        expert_activation=activation,
        expert_utilization=utilization,
        processed_tokens=processed if processed else None,
        per_req_info=per_req_info,
        raw_forward_mode=raw_mode,
        raw_probe_source=source,
    )


def append_record(path: str, record: ProbeRecord) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.to_dict()) + "\n")


def _sync_and_time() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass
    return time.perf_counter()


def _classify_forward_mode(forward_batch: Any) -> tuple[str, str | None]:
    mode = getattr(forward_batch, "forward_mode", None)
    raw = getattr(mode, "name", None) or str(mode)
    try:
        if mode is not None and mode.is_idle():
            return "idle", raw
        if mode is not None and mode.is_decode():
            return "decode", raw
        if mode is not None and mode.is_extend():
            return "prefill", raw
    except Exception:
        pass
    low = raw.lower() if raw else ""
    if "idle" in low:
        return "idle", raw
    if "decode" in low:
        return "decode", raw
    return "prefill", raw


def _build_per_req_info(forward_batch: Any, server_args: Any) -> list[dict[str, Any]]:
    global _PREFILL_REQ_COUNTER
    req_indices = _to_list(getattr(forward_batch, "req_pool_indices", None))
    extend_lens = _to_list(getattr(forward_batch, "extend_seq_lens_cpu", None)) or _to_list(getattr(forward_batch, "extend_seq_lens", None))
    total_lens = _to_list(getattr(forward_batch, "seq_lens_cpu", None)) or _to_list(getattr(forward_batch, "seq_lens", None))
    rids = _to_list(getattr(forward_batch, "rids", None))
    chunk_size = getattr(server_args, "chunked_prefill_size", None) if server_args is not None else None
    chunk_size = int(chunk_size) if chunk_size else None
    rows = []
    for offset, extend_len in enumerate(extend_lens):
        total_len = total_lens[offset] if offset < len(total_lens) else extend_len
        req_pool_idx = req_indices[offset] if offset < len(req_indices) else offset
        rid = rids[offset] if offset < len(rids) else f"{int(req_pool_idx)}_{_PREFILL_REQ_COUNTER}"
        is_last = True if not chunk_size else int(extend_len) < chunk_size
        rows.append({
            "req_pool_idx": int(req_pool_idx),
            "req_id": str(rid),
            "extend_len": int(extend_len),
            "total_len": int(total_len),
            "is_last_chunk": is_last,
        })
        if is_last:
            _PREFILL_REQ_COUNTER += 1
    return rows


def _extract_expert_activation(output: Any, *, profiling_only: bool) -> tuple[float, float | None, str]:
    if profiling_only:
        return 0.0, None, "profiling_only"
    metrics = getattr(output, "expert_distribution_metrics", None)
    if metrics is not None:
        activation = _field(metrics, "average_expert_activation", "avg_expert_activation", "expert_activation")
        utilization = _field(metrics, "expert_utilization", "average_expert_utilization")
        if activation is not None:
            return float(activation), float(utilization) if utilization is not None else None, "expert_distribution_metrics"
    routed = getattr(output, "routed_experts_output", None)
    activation = _activation_from_routed_output(routed)
    if activation is not None:
        return activation, None, "routed_experts_output"
    indexer = getattr(output, "indexer_topk_output", None)
    activation = _activation_from_routed_output(indexer)
    if activation is not None:
        return activation, None, "indexer_topk_output"
    activation = _activation_from_recorder()
    if activation is not None:
        return activation, None, "recorder_dump"
    return 0.0, None, "timing_only"


def _activation_from_routed_output(value: Any) -> float | None:
    if value is None:
        return None
    counts = _field(value, "expert_counts", "expert_count", "logical_count", "topk", "topk_ids")
    if counts is None:
        return None
    is_topk = hasattr(value, "topk") or (isinstance(value, dict) and "topk" in value)
    try:
        import torch
        tensor = counts if isinstance(counts, torch.Tensor) else torch.as_tensor(counts)
        if tensor.numel() == 0:
            return None
        if tensor.ndim >= 2 and is_topk:
            valid = tensor[tensor >= 0]
            active = valid.unique().numel() if valid.numel() else 0
        elif tensor.ndim >= 2:
            active = (tensor > 0).float().sum(dim=-1).float().mean()
        else:
            active = (tensor > 0).float().sum()
        return float(active.item() if hasattr(active, "item") else active)
    except Exception:
        if is_topk:
            flat = _flatten(counts)
            return float(len({int(item) for item in flat if int(item) >= 0})) if flat else None
        rows = [row for row in counts if isinstance(row, (list, tuple))] if isinstance(counts, (list, tuple)) else []
        if rows:
            return sum(sum(1 for item in row if float(item) > 0) for row in rows) / len(rows)
        return None




def _flatten(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(_flatten(item))
        return out
    return [value]

def _activation_from_recorder() -> float | None:
    try:
        from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
        recorder = get_global_expert_distribution_recorder()
        data = recorder.dump_record(output_mode="object")
    except Exception:
        return None
    activation = _activation_from_routed_output(data)
    return activation

def _field(obj: Any, *names: str) -> Any:
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def _to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    try:
        import torch
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
    except Exception:
        pass
    if isinstance(value, (list, tuple)):
        return list(value)
    try:
        return list(value)
    except TypeError:
        return [value]


def _out_cache_tokens(forward_batch: Any) -> int:
    out_cache = getattr(forward_batch, "out_cache_loc", None)
    shape = getattr(out_cache, "shape", None)
    if shape:
        return int(shape[0])
    return 0


def _is_rank0(model_runner: Any) -> bool:
    return int(getattr(model_runner, "tp_rank", 0) or 0) == 0


def _profiling_only_enabled() -> bool:
    return os.environ.get("SBENCH_PROFILING_ONLY", "0").lower() in {"1", "true", "yes"}
