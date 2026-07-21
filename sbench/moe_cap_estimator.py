"""MoE-CAP-compatible estimator formulas for calibration runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .components import AttentionComponent, CacheComponent, processed_tokens
from .descriptor import ArchitectureDescriptor
from .estimator import EstimateResult, usable_records


@dataclass(frozen=True)
class MoeCapSupport:
    supported: bool
    reason: str = ""


def support_status(arch: ArchitectureDescriptor) -> MoeCapSupport:
    model_type = arch.model_type.lower()
    model_name = arch.model_name.lower()
    if not arch.moe.enabled:
        return MoeCapSupport(False, "MoE-CAP compatibility mode currently requires a MoE descriptor")
    if "qwen" not in model_name and model_type not in {"qwen_moe", "qwen2_moe"}:
        return MoeCapSupport(False, "MoE-CAP compatibility mode currently supports Qwen MoE models first")
    if arch.moe.shared_expert_intermediate_size is None and arch.moe.shared_experts <= 0:
        return MoeCapSupport(False, "Qwen MoE-CAP formula requires shared expert size")
    return MoeCapSupport(True)


def estimate_moe_cap_compatible(arch: ArchitectureDescriptor, records: Iterable[dict]) -> EstimateResult:
    status = support_status(arch)
    if not status.supported:
        raise ValueError(status.reason)

    prefill_smbu: list[tuple[float, float]] = []
    prefill_smfu: list[tuple[float, float]] = []
    decode_smbu: list[tuple[float, float]] = []
    decode_smfu: list[tuple[float, float]] = []
    prefill_tps: list[float] = []
    decode_tps: list[float] = []
    ttfts: list[float] = []
    tpots: list[float] = []
    kv_sizes: list[float] = []

    constants = _qwen_constants(arch)
    rt = arch.runtime

    for record in usable_records(records):
        latency = float(record.get("latency", 0) or 0)
        mode = record.get("forward_mode")
        kv_size = _kv_size(arch, record)
        attention_score = AttentionComponent().attention_score_units(arch, record)
        activation = _activation(arch, record)
        kv_sizes.append(_true_kv_size_mb(arch, record))

        bandwidth_units = constants["layers"] * (
            activation * constants["expert_size"]
            + constants["shared_experts_size_total"]
            + constants["attention_size_per_token"]
        ) + kv_size
        flops_units = constants["layers"] * (
            constants["attention_size_per_token"]
            + constants["expert_size"]
            + constants["shared_experts_size_total"]
        ) + attention_score

        if mode == "prefill":
            throughput = int(record.get("seq_lens_sum", 0)) / latency
            ttfts.append(latency)
            prefill_tps.append(throughput)
            prefill_smbu.append((_smbu(bandwidth_units, rt.precision_bytes, latency, rt.num_gpus, rt.peak_bandwidth_tb), latency))
            prefill_smfu.append((_smfu(flops_units, throughput, rt.num_gpus, rt.peak_flops_tf), latency))
        else:
            throughput = int(record.get("batch_size", 1)) / latency
            tpots.append(latency)
            decode_tps.append(throughput)
            decode_smbu.append((_smbu(bandwidth_units, rt.precision_bytes, latency, rt.num_gpus, rt.peak_bandwidth_tb), latency))
            decode_smfu.append((_smfu(flops_units, throughput, rt.num_gpus, rt.peak_flops_tf), latency))

    return EstimateResult(
        prefill_smbu=_weighted(prefill_smbu),
        prefill_smfu=_weighted(prefill_smfu),
        decoding_smbu=_weighted(decode_smbu),
        decoding_smfu=_weighted(decode_smfu),
        prefill_tp=sum(prefill_tps) / len(prefill_tps) if prefill_tps else 0.0,
        decoding_throughput=sum(decode_tps) / len(decode_tps) if decode_tps else 0.0,
        ttft=sum(ttfts) / len(ttfts) if ttfts else 0.0,
        tpot=sum(tpots) / len(tpots) if tpots else 0.0,
        kv_size=sum(kv_sizes) / len(kv_sizes) if kv_sizes else 0.0,
    )


def _qwen_constants(arch: ArchitectureDescriptor) -> dict[str, float]:
    expert_size = arch.moe.expert_intermediate_size * 3 * arch.moe.hidden_size / 1e12
    if arch.moe.shared_expert_intermediate_size is not None:
        shared = arch.moe.shared_expert_intermediate_size * 3 * arch.moe.hidden_size / 1e12
    else:
        shared = arch.moe.shared_experts * expert_size
    return {
        "layers": float(arch.attention.num_layers),
        "expert_size": expert_size,
        "shared_experts_size_total": shared,
        "attention_size_per_token": AttentionComponent().projection_units(arch),
    }


def _activation(arch: ArchitectureDescriptor, record: dict) -> float:
    value = record.get("expert_activation")
    if value is None or float(value or 0) <= 0:
        return float(arch.moe.top_k or 0)
    return max(float(value), 0.0)


def _kv_size(arch: ArchitectureDescriptor, record: dict) -> float:
    return processed_tokens(record) * CacheComponent().per_token_units(arch) / 1e12


def _true_kv_size_mb(arch: ArchitectureDescriptor, record: dict) -> float:
    per_token = CacheComponent().per_token_units(arch)
    return (processed_tokens(record) * per_token + per_token) / 1e6


def _smbu(units: float, precision_bytes: float, latency: float, num_gpus: int, peak_bandwidth_tb: float) -> float:
    return (units * precision_bytes / latency) / (num_gpus * peak_bandwidth_tb)


def _smfu(units: float, throughput: float, num_gpus: int, peak_flops_tf: float) -> float:
    return (units * 2 * throughput) / (num_gpus * peak_flops_tf / 2)


def _weighted(values: list[tuple[float, float]]) -> float:
    weight = sum(w for _, w in values)
    return sum(value * w for value, w in values) / weight if weight else 0.0
