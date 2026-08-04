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


@dataclass(frozen=True)
class FormulaCost:
    bandwidth_units: float
    flops_units: float


def support_status(arch: ArchitectureDescriptor) -> MoeCapSupport:
    model_type = arch.model_type.lower()
    model_name = arch.model_name.lower()
    if not arch.moe.enabled:
        return MoeCapSupport(False, "MoE-CAP compatibility mode currently requires a MoE descriptor")
    if "qwen" not in model_name and model_type not in {"qwen_moe", "qwen2_moe", "qwen3_moe", "qwen3_5_moe", "qwen3_5_moe_text"}:
        return MoeCapSupport(False, "MoE-CAP compatibility mode currently supports Qwen MoE models first")
    if _is_qwen35_or_36(arch):
        if arch.attention.type != "hybrid":
            return MoeCapSupport(False, "Qwen3.5/Qwen3.6 MoE-CAP formula requires hybrid attention")
        if arch.attention.full_attention_layers <= 0 or arch.attention.linear_attention_layers <= 0:
            return MoeCapSupport(False, "Qwen3.5/Qwen3.6 MoE-CAP formula requires full and linear attention layer counts")
        if arch.attention.full_attention_layers + arch.attention.linear_attention_layers != arch.attention.num_layers:
            return MoeCapSupport(False, "Qwen3.5/Qwen3.6 hybrid layer counts must sum to num_hidden_layers")
        if arch.moe.top_k <= 0 or arch.moe.routed_experts <= 0:
            return MoeCapSupport(False, "Qwen3.5/Qwen3.6 MoE-CAP formula requires top_k and routed expert counts")
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

    formula = _qwen_formula_family(arch)
    constants = _qwen_constants(arch)
    rt = arch.runtime

    for record in usable_records(records):
        latency = float(record.get("latency", 0) or 0)
        mode = record.get("forward_mode")
        kv_size = _kv_size(arch, record)
        attention_score = _attention_score(arch, record)
        activation = _activation(arch, record)
        kv_sizes.append(_true_kv_size_mb(arch, record))

        cost = _qwen_formula_cost(formula, constants, activation, kv_size, attention_score)

        if mode == "prefill":
            throughput = int(record.get("seq_lens_sum", 0)) / latency
            ttfts.append(latency)
            prefill_tps.append(throughput)
            prefill_smbu.append((_smbu(cost.bandwidth_units, rt.precision_bytes, latency, rt.num_gpus, rt.peak_bandwidth_tb), latency))
            prefill_smfu.append((_smfu(cost.flops_units, throughput, rt.num_gpus, rt.peak_flops_tf), latency))
        else:
            throughput = int(record.get("batch_size", 1)) / latency
            tpots.append(latency)
            decode_tps.append(throughput)
            decode_smbu.append((_smbu(cost.bandwidth_units, rt.precision_bytes, latency, rt.num_gpus, rt.peak_bandwidth_tb), latency))
            decode_smfu.append((_smfu(cost.flops_units, throughput, rt.num_gpus, rt.peak_flops_tf), latency))

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


def _qwen_formula_family(arch: ArchitectureDescriptor) -> str:
    """Model-specific MoE-CAP formula dispatch.

    Add new MoE-CAP-style model formulas here, then implement a matching
    _<family>_cost function below. Component-wise support belongs in
    sbench/adapters.py and sbench/components.py.
    """
    if _is_qwen35_or_36(arch):
        return "qwen3_5_or_3_6_hybrid_moe"
    if _is_qwen3(arch):
        return "qwen3_moe"
    return "qwen_legacy_moe"


def _qwen_formula_cost(
    formula: str,
    constants: dict[str, float],
    activation: float,
    kv_size: float,
    attention_score: float,
) -> FormulaCost:
    if formula == "qwen3_5_or_3_6_hybrid_moe":
        return _qwen35_qwen36_hybrid_moe_cost(constants, activation, kv_size, attention_score)
    if formula == "qwen3_moe":
        return _qwen3_moe_cost(constants, activation, kv_size, attention_score)
    if formula == "qwen_legacy_moe":
        return _qwen_legacy_moe_cost(constants, activation, kv_size, attention_score)
    raise ValueError(f"unsupported Qwen MoE-CAP formula family: {formula}")


def _qwen35_qwen36_hybrid_moe_cost(
    constants: dict[str, float],
    activation: float,
    kv_size: float,
    attention_score: float,
) -> FormulaCost:
    """Qwen3.5/Qwen3.6 35B-A3B: hybrid linear/full attention + shared expert.

    Used for Qwen/Qwen3.5-35B-A3B and Qwen/Qwen3.6-35B-A3B. These models use
    qwen3_5_moe_text configs with full-attention layers interleaved with linear
    attention layers, so attention projection and score costs are not the legacy
    all-full-attention term.
    """
    bandwidth_units = (
        constants["moe_layers"] * (activation * constants["expert_size"] + constants["shared_experts_size_total"])
        + constants["hybrid_attention_size_total"]
        + kv_size
    )
    flops_units = (
        constants["moe_layers"] * (constants["expert_size"] + constants["shared_experts_size_total"])
        + constants["hybrid_attention_size_total"]
        + attention_score
    )
    return FormulaCost(bandwidth_units, flops_units)


def _qwen3_moe_cost(
    constants: dict[str, float],
    activation: float,
    kv_size: float,
    attention_score: float,
) -> FormulaCost:
    """Qwen3 MoE such as Qwen/Qwen3-30B-A3B.

    Uses explicit dense-vs-MoE layer split from mlp_only_layers and
    decoder_sparse_step. Shared expert cost is not included unless represented
    by the model config.
    """
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
    return FormulaCost(bandwidth_units, flops_units)


def _qwen_legacy_moe_cost(
    constants: dict[str, float],
    activation: float,
    kv_size: float,
    attention_score: float,
) -> FormulaCost:
    """Qwen1.5/Qwen2-style MoE-CAP formula.

    Used for Qwen/Qwen1.5-MoE-A2.7B-Chat and older Qwen MoE layouts where every
    layer is treated as the same attention + routed expert + shared expert unit.
    """
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
    return FormulaCost(bandwidth_units, flops_units)


def _qwen_constants(arch: ArchitectureDescriptor) -> dict[str, float]:
    expert_size = arch.moe.expert_intermediate_size * 3 * arch.moe.hidden_size / 1e12
    dense_ffn_size = arch.ffn.dense_intermediate_size * 3 * arch.ffn.hidden_size / 1e12
    if arch.moe.shared_expert_intermediate_size is not None:
        shared = arch.moe.shared_expert_intermediate_size * 3 * arch.moe.hidden_size / 1e12
    else:
        shared = arch.moe.shared_experts * expert_size
    return {
        "layers": float(arch.attention.num_layers),
        "moe_layers": float(arch.moe.moe_layers),
        "dense_layers": float(arch.ffn.dense_layers),
        "top_k": float(max(arch.moe.top_k, 1)),
        "expert_size": expert_size,
        "dense_ffn_size": dense_ffn_size,
        "shared_experts_size_total": shared,
        "attention_size_per_token": AttentionComponent().projection_units(arch),
        "hybrid_attention_size_total": _hybrid_attention_projection_units(arch),
    }


def _is_qwen35_or_36(arch: ArchitectureDescriptor) -> bool:
    model_type = arch.model_type.lower()
    model_name = arch.model_name.lower()
    return "qwen3.5" in model_name or "qwen3.6" in model_name or "qwen3_5" in model_type


def _is_qwen3_or_newer(arch: ArchitectureDescriptor) -> bool:
    return _is_qwen35_or_36(arch) or _is_qwen3(arch)


def _is_qwen3(arch: ArchitectureDescriptor) -> bool:
    return not _is_qwen35_or_36(arch) and (arch.model_type.lower() == "qwen3_moe" or "qwen3" in arch.model_name.lower())


def _activation(arch: ArchitectureDescriptor, record: dict) -> float:
    return real_expert_activation(record)


def _attention_score(arch: ArchitectureDescriptor, record: dict) -> float:
    attn = arch.attention
    ctx = int(record.get("seq_lens_sum", 0) or 0)
    if _is_qwen35_or_36(arch) and attn.type == "hybrid":
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
    return CacheComponent().estimate(arch, record).cache_units * 1e6


def _smbu(units: float, precision_bytes: float, latency: float, num_gpus: int, peak_bandwidth_tb: float) -> float:
    return (units * precision_bytes / latency) / (num_gpus * peak_bandwidth_tb)


def _smfu(units: float, throughput: float, num_gpus: int, peak_flops_tf: float) -> float:
    return (units * 2 * throughput) / (num_gpus * peak_flops_tf / 2)


def _weighted(values: list[tuple[float, float]]) -> float:
    weight = sum(w for _, w in values)
    return sum(value * w for value, w in values) / weight if weight else 0.0
