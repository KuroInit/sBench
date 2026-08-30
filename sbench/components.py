"""Composable utilization cost components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .descriptor import ArchitectureDescriptor
from .trace import has_real_expert_activation


@dataclass(frozen=True)
class ComponentCost:
    name: str
    bandwidth_units: float = 0.0
    flops_units: float = 0.0
    cache_units: float = 0.0
    attention_score_units: float = 0.0

    def plus(self, other: "ComponentCost") -> "ComponentCost":
        return ComponentCost(
            name="total",
            bandwidth_units=self.bandwidth_units + other.bandwidth_units,
            flops_units=self.flops_units + other.flops_units,
            cache_units=self.cache_units + other.cache_units,
            attention_score_units=self.attention_score_units + other.attention_score_units,
        )


class CostComponent(Protocol):
    name: str

    def estimate(self, arch: ArchitectureDescriptor, record: dict) -> ComponentCost:
        ...


def processed_tokens(record: dict) -> int:
    if record.get("processed_tokens") is not None:
        return int(record["processed_tokens"])
    per_req = record.get("per_req_info") or []
    total = sum(int(req.get("extend_len", 0)) for req in per_req)
    return total or int(record.get("seq_lens_sum", 0))


def real_expert_activation(record: dict) -> float:
    value = record.get("expert_activation")
    source = record.get("raw_probe_source", "unknown")
    if value is None or float(value or 0) <= 0:
        raise ValueError(
            "MoE estimator requires real expert_activation from the probe; "
            f"got expert_activation={value!r}, raw_probe_source={source!r}. "
            "Run SGLang with routed expert capture enabled."
        )
    return max(float(value), 0.0)


def prefill_context_mass(record: dict) -> int:
    per_req = record.get("per_req_info") or []
    total = 0
    for req in per_req:
        extend_len = int(req.get("extend_len", 0) or 0)
        seq_len = int(req.get("seq_len") or req.get("total_len") or 0)
        if extend_len > 0 and seq_len > 0:
            total += extend_len * seq_len
    if total:
        return total
    if int(record.get("batch_size", 1) or 1) > 1:
        raise ValueError(
            "prefill attention estimation requires per_req_info for multi-request batches; "
            "the aggregate fallback would overcount cross-request context mass"
        )
    tokens = max(processed_tokens(record), 1)
    return tokens * int(record.get("seq_lens_sum", 0))


class CacheComponent:
    name = "cache"

    def per_token_units(self, arch: ArchitectureDescriptor) -> float:
        cache = arch.cache
        if cache.type == "latent_kv" and cache.kv_lora_rank is not None and cache.qk_rope_head_dim is not None:
            return cache.num_layers * (cache.kv_lora_rank + cache.qk_rope_head_dim)
        if cache.type == "recurrent_state":
            return cache.num_layers * float(cache.recurrent_state_size or 0)
        if cache.type == "hybrid":
            full_layers = cache.full_attention_layers or cache.num_layers
            return 2 * full_layers * cache.head_dim * cache.num_key_value_heads
        scale = cache.sparse_factor if cache.type == "sparse_kv" else 1.0
        return scale * 2 * cache.num_layers * cache.head_dim * cache.num_key_value_heads

    def estimate(self, arch: ArchitectureDescriptor, record: dict) -> ComponentCost:
        units = processed_tokens(record) * self.per_token_units(arch)
        if arch.cache.type == "hybrid":
            units += max(int(record.get("batch_size", 1) or 1), 1) * self.state_units(arch)
        units /= 1e12
        return ComponentCost(name=self.name, cache_units=units)

    def state_units(self, arch: ArchitectureDescriptor) -> float:
        cache = arch.cache
        if cache.type != "hybrid":
            return 0.0
        return cache.linear_attention_layers * _linear_state_units(
            cache.linear_num_key_heads,
            cache.linear_num_value_heads,
            cache.linear_key_head_dim,
            cache.linear_value_head_dim,
        )


class AttentionComponent:
    name = "attention"

    def projection_units(self, arch: ArchitectureDescriptor) -> float:
        """Return one full-attention layer's projection units for legacy formulas."""
        attn = arch.attention
        if attn.type == "mla" and attn.kv_lora_rank is not None and attn.qk_rope_head_dim is not None:
            q_head_dim = attn.qk_rope_head_dim + (attn.qk_nope_head_dim or 0)
            v_head_dim = attn.v_head_dim or attn.head_dim
            base = (
                attn.hidden_size * (attn.kv_lora_rank + attn.qk_rope_head_dim)
                + attn.kv_lora_rank * attn.num_attention_heads * (q_head_dim - attn.qk_rope_head_dim + v_head_dim)
                + v_head_dim * attn.num_attention_heads * attn.hidden_size
            )
            if attn.q_lora_rank:
                q_size = attn.hidden_size * attn.q_lora_rank + attn.q_lora_rank * attn.num_attention_heads * q_head_dim
            else:
                q_size = attn.hidden_size * attn.num_attention_heads * q_head_dim
            return (base + q_size) / 1e12
        return (
            attn.hidden_size * (attn.num_attention_heads * attn.head_dim + attn.num_key_value_heads * attn.head_dim * 2)
            + attn.num_attention_heads * attn.head_dim * attn.hidden_size
        ) / 1e12

    def total_projection_units(self, arch: ArchitectureDescriptor) -> float:
        attn = arch.attention
        if attn.type == "hybrid" and (attn.linear_key_head_dim or attn.linear_value_head_dim or attn.linear_attention_layers):
            full = (attn.full_attention_layers or 0) * _full_attention_projection_units(attn)
            linear = (attn.linear_attention_layers or 0) * _linear_attention_projection_units(attn)
            if not full and not linear:
                return attn.num_layers * self.projection_units(arch)
            return (full + linear) / 1e12
        return attn.num_layers * self.projection_units(arch)

    def attention_score_units(self, arch: ArchitectureDescriptor, record: dict) -> float:
        attn = arch.attention
        ctx = int(record.get("seq_lens_sum", 0))
        batch = max(int(record.get("batch_size", 1)), 1)
        prefill = record.get("forward_mode") == "prefill"
        token_count = max(processed_tokens(record) if prefill else batch, 1)
        context_mass = prefill_context_mass(record) if prefill else ctx
        if attn.type == "linear":
            total = token_count * attn.num_layers * attn.hidden_size
        elif attn.type == "hybrid" and (attn.linear_key_head_dim or attn.linear_value_head_dim):
            linear_heads = (attn.linear_num_key_heads or 0) + (attn.linear_num_value_heads or 0)
            linear_dim = (attn.linear_key_head_dim or 0) + (attn.linear_value_head_dim or 0)
            full_layers = attn.full_attention_layers or 0
            linear_layers = attn.linear_attention_layers or max(attn.num_layers - full_layers, 0)
            dense_part = context_mass * full_layers * attn.num_attention_heads * attn.head_dim * 2
            linear_part = token_count * linear_layers * max(linear_heads, 1) * max(linear_dim, attn.hidden_size)
            total = dense_part + linear_part
        elif attn.type == "mla" and attn.qk_rope_head_dim is not None:
            q_head_dim = attn.qk_rope_head_dim + (attn.qk_nope_head_dim or 0)
            k_size = attn.num_layers * attn.num_attention_heads * q_head_dim
            v_size = attn.num_layers * attn.num_attention_heads * (attn.v_head_dim or attn.head_dim)
            total = context_mass * k_size + context_mass * v_size
        else:
            total = context_mass * attn.num_layers * attn.num_attention_heads * attn.head_dim * 2
        if attn.type in {"sparse", "hybrid"}:
            total *= attn.sparse_factor
        units = total / 1e12
        return units / token_count

    def estimate(self, arch: ArchitectureDescriptor, record: dict) -> ComponentCost:
        projection = self.total_projection_units(arch)
        return ComponentCost(
            name=self.name,
            bandwidth_units=projection,
            flops_units=projection,
            attention_score_units=self.attention_score_units(arch, record),
        )


class DenseFFNComponent:
    name = "dense_ffn"

    def estimate(self, arch: ArchitectureDescriptor, record: dict) -> ComponentCost:
        ffn = arch.ffn
        units = ffn.dense_layers * ffn.dense_intermediate_size * 3 * ffn.hidden_size / 1e12
        return ComponentCost(name=self.name, bandwidth_units=units, flops_units=units)


class MoEComponent:
    name = "moe"

    def estimate(self, arch: ArchitectureDescriptor, record: dict) -> ComponentCost:
        moe = arch.moe
        if not moe.enabled:
            return ComponentCost(name=self.name)
        expert_size = moe.expert_intermediate_size * 3 * moe.hidden_size / 1e12
        if moe.shared_expert_intermediate_size is not None:
            shared = moe.shared_expert_intermediate_size * 3 * moe.hidden_size / 1e12
        else:
            shared = moe.shared_experts * expert_size
        activation = real_expert_activation(record) if has_real_expert_activation(record) else 0.0
        return ComponentCost(
            name=self.name,
            bandwidth_units=moe.moe_layers * (activation * expert_size + shared),
            flops_units=moe.moe_layers * (max(moe.top_k, 1) * expert_size + shared),
        )


class RouterComponent:
    name = "router"

    def estimate(self, arch: ArchitectureDescriptor, record: dict) -> ComponentCost:
        moe = arch.moe
        if not moe.enabled or not moe.routed_experts:
            return ComponentCost(name=self.name)
        # Return per-token router work. The estimator multiplies FLOP units by
        # tokens/sec, so including token count here double-counts throughput.
        units = moe.moe_layers * moe.hidden_size * moe.routed_experts / 1e12
        return ComponentCost(name=self.name, flops_units=units)


class LinearAttentionComponent(AttentionComponent):
    name = "linear_attention"


class SparseAttentionComponent(AttentionComponent):
    name = "sparse_attention"


class HybridAttentionComponent(AttentionComponent):
    name = "hybrid_attention"


DEFAULT_COMPONENTS: tuple[CostComponent, ...] = (
    AttentionComponent(),
    CacheComponent(),
    DenseFFNComponent(),
    MoEComponent(),
    RouterComponent(),
)


def _full_attention_projection_units(attn) -> float:
    q = attn.num_attention_heads * attn.head_dim
    kv = attn.num_key_value_heads * attn.head_dim
    return attn.hidden_size * (q + kv + kv + q)


def _linear_attention_projection_units(attn) -> float:
    key_heads = attn.linear_num_key_heads or attn.num_attention_heads
    value_heads = attn.linear_num_value_heads or attn.num_attention_heads
    key_dim = attn.linear_key_head_dim or attn.head_dim
    value_dim = attn.linear_value_head_dim or attn.head_dim
    key_units = key_heads * key_dim
    value_units = value_heads * value_dim
    conv = (attn.linear_conv_kernel_dim or 0) * (2 * key_units + value_units)
    return attn.hidden_size * (2 * key_units + 2 * value_units) + attn.hidden_size * (2 * value_heads) + value_units * attn.hidden_size + conv


def _linear_state_units(
    key_heads: int | None,
    value_heads: int | None,
    key_dim: int | None,
    value_dim: int | None,
) -> float:
    heads = max(value_heads or key_heads or 0, 0)
    return float(2 * heads * (key_dim or 0) * (value_dim or 0))
