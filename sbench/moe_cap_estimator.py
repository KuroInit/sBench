"""MoE-CAP-compatible estimator formulas for calibration runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .components import (
    AttentionComponent,
    CacheComponent,
    _full_attention_projection_units,
    _linear_attention_projection_units,
    prefill_context_mass,
    processed_tokens,
    real_expert_activation,
)
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
    if "qwen" not in model_name and model_type not in {"qwen_moe", "qwen2_moe", "qwen3_moe", "qwen3_5_moe", "qwen3_5_moe_text"}:
        return MoeCapSupport(False, "MoE-CAP compatibility mode currently supports Qwen MoE models first")
    if not _is_qwen3_or_newer(arch) and arch.moe.shared_expert_intermediate_size is None and arch.moe.shared_experts <= 0:
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
        attention_score = _attention_score(arch, record)
        activation = _activation(arch, record)
        kv_sizes.append(_true_kv_size_mb(arch, record))

        if constants["is_qwen35"]:
            bandwidth_units = (
                constants["moe_layers"] * (activation * constants["expert_size"] + constants["shared_experts_size_total"])
                + constants["hybrid_attention_size_total"]
                + constants["router_size"]
                + kv_size
            )
            flops_units = (
                constants["moe_layers"] * (constants["top_k"] * constants["expert_size"] + constants["shared_experts_size_total"])
                + constants["hybrid_attention_size_total"]
                + constants["router_size"]
                + attention_score
            )
        elif constants["is_qwen3"]:
            bandwidth_units = (
                constants["moe_layers"] * activation * constants["expert_size"]
                + constants["dense_layers"] * constants["dense_ffn_size"]
                + constants["layers"] * constants["attention_size_per_token"]
                + kv_size
            )
            flops_units = (
                constants["moe_layers"] * constants["expert_size"]
                + constants["dense_layers"] * constants["dense_ffn_size"]
                + constants["layers"] * constants["attention_size_per_token"]
                + attention_score
            )
        else:
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
    dense_ffn_size = arch.ffn.dense_intermediate_size * 3 * arch.ffn.hidden_size / 1e12
    if arch.moe.shared_expert_intermediate_size is not None:
        shared = arch.moe.shared_expert_intermediate_size * 3 * arch.moe.hidden_size / 1e12
    else:
        shared = arch.moe.shared_experts * expert_size
    return {
        "is_qwen35": _is_qwen35(arch),
        "is_qwen3": _is_qwen3(arch),
        "layers": float(arch.attention.num_layers),
        "moe_layers": float(arch.moe.moe_layers),
        "dense_layers": float(arch.ffn.dense_layers),
        "top_k": float(max(arch.moe.top_k, 1)),
        "expert_size": expert_size,
        "dense_ffn_size": dense_ffn_size,
        "shared_experts_size_total": shared,
        "attention_size_per_token": AttentionComponent().projection_units(arch),
        "hybrid_attention_size_total": _hybrid_attention_projection_units(arch),
        "router_size": arch.moe.moe_layers * arch.moe.hidden_size * arch.moe.routed_experts / 1e12,
    }


def _is_qwen35(arch: ArchitectureDescriptor) -> bool:
    model_type = arch.model_type.lower()
    model_name = arch.model_name.lower()
    return "qwen3.5" in model_name or "qwen3_5" in model_type


def _is_qwen3_or_newer(arch: ArchitectureDescriptor) -> bool:
    return _is_qwen35(arch) or _is_qwen3(arch)


def _is_qwen3(arch: ArchitectureDescriptor) -> bool:
    return not _is_qwen35(arch) and (arch.model_type.lower() == "qwen3_moe" or "qwen3" in arch.model_name.lower())


def _activation(arch: ArchitectureDescriptor, record: dict) -> float:
    return real_expert_activation(record)


def _attention_score(arch: ArchitectureDescriptor, record: dict) -> float:
    attn = arch.attention
    ctx = int(record.get("seq_lens_sum", 0) or 0)
    if _is_qwen35(arch) and attn.type == "hybrid":
        prefill = record.get("forward_mode") == "prefill"
        token_count = max(processed_tokens(record) if prefill else int(record.get("batch_size", 1) or 1), 1)
        context_mass = prefill_context_mass(record) if prefill else ctx
        full_layers = attn.full_attention_layers or 0
        linear_layers = attn.linear_attention_layers or max(attn.num_layers - full_layers, 0)
        linear_heads = (attn.linear_num_key_heads or 0) + (attn.linear_num_value_heads or 0)
        linear_dim = (attn.linear_key_head_dim or 0) + (attn.linear_value_head_dim or 0)
        full_score = context_mass * full_layers * attn.num_attention_heads * attn.head_dim * 2
        linear_score = token_count * linear_layers * max(linear_heads, 1) * max(linear_dim, attn.hidden_size)
        return (full_score + linear_score) / max(token_count, 1) / 1e12
    if attn.type == "mla" and attn.qk_rope_head_dim is not None:
        q_head_dim = attn.qk_rope_head_dim + (attn.qk_nope_head_dim or 0)
        k_size = attn.num_layers * attn.num_attention_heads * q_head_dim
        v_size = attn.num_layers * attn.num_attention_heads * (attn.v_head_dim or attn.head_dim)
        score = ctx * k_size + ctx * v_size
    else:
        kv_size = attn.num_layers * attn.num_attention_heads * attn.head_dim
        score = ctx * kv_size * 2
    if record.get("forward_mode") == "prefill":
        score /= max(ctx, 1)
    else:
        score /= max(int(record.get("batch_size", 1) or 1), 1)
    return score / 1e12


def _hybrid_attention_projection_units(arch: ArchitectureDescriptor) -> float:
    attn = arch.attention
    if attn.type != "hybrid":
        return 0.0
    full = (attn.full_attention_layers or 0) * _full_attention_projection_units(attn)
    linear = (attn.linear_attention_layers or 0) * _linear_attention_projection_units(attn)
    if not full and not linear:
        return arch.attention.num_layers * AttentionComponent().projection_units(arch)
    return (full + linear) / 1e12


def _kv_size(arch: ArchitectureDescriptor, record: dict) -> float:
    return CacheComponent().estimate(arch, record).cache_units


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
