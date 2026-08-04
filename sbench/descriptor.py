"""Architecture descriptors for component-wise S-MFU/S-MBU estimates."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping


def deep_get(data: Any, *keys: str) -> Any:
    if not isinstance(data, Mapping):
        return None
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    for value in data.values():
        if isinstance(value, Mapping):
            found = deep_get(value, *keys)
            if found is not None:
                return found
    return None


def merge_dict(base: Mapping[str, Any], override: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = dict(base)
    if not override:
        return merged
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def as_int(value: Any, default: int = 0) -> int:
    return default if value is None else int(value)


def as_float(value: Any, default: float = 0.0) -> float:
    return default if value is None else float(value)


@dataclass(frozen=True)
class AttentionDescriptor:
    type: str = "mha"
    num_layers: int = 0
    hidden_size: int = 0
    num_attention_heads: int = 0
    num_key_value_heads: int = 0
    head_dim: int = 0
    kv_lora_rank: int | None = None
    q_lora_rank: int | None = None
    qk_rope_head_dim: int | None = None
    qk_nope_head_dim: int | None = None
    v_head_dim: int | None = None
    linear_key_head_dim: int | None = None
    linear_value_head_dim: int | None = None
    linear_num_key_heads: int | None = None
    linear_num_value_heads: int | None = None
    linear_conv_kernel_dim: int | None = None
    layer_types: tuple[str, ...] = ()
    full_attention_layers: int = 0
    linear_attention_layers: int = 0
    sparse_factor: float = 1.0


@dataclass(frozen=True)
class CacheDescriptor:
    type: str = "kv"
    num_layers: int = 0
    head_dim: int = 0
    num_key_value_heads: int = 0
    kv_lora_rank: int | None = None
    qk_rope_head_dim: int | None = None
    recurrent_state_size: int | None = None
    full_attention_layers: int = 0
    linear_attention_layers: int = 0
    linear_key_head_dim: int | None = None
    linear_value_head_dim: int | None = None
    linear_num_key_heads: int | None = None
    linear_num_value_heads: int | None = None
    linear_conv_kernel_dim: int | None = None
    sparse_factor: float = 1.0


@dataclass(frozen=True)
class FFNDescriptor:
    dense_layers: int = 0
    hidden_size: int = 0
    dense_intermediate_size: int = 0


@dataclass(frozen=True)
class MoEDescriptor:
    enabled: bool = False
    moe_layers: int = 0
    routed_experts: int = 0
    shared_experts: int = 0
    top_k: int = 0
    hidden_size: int = 0
    expert_intermediate_size: int = 0
    shared_expert_intermediate_size: int | None = None


@dataclass(frozen=True)
class RuntimeDescriptor:
    precision_bytes: float = 2.0
    num_gpus: int = 1
    peak_bandwidth_tb: float = 1.0
    peak_flops_tf: float = 1.0


@dataclass(frozen=True)
class ArchitectureDescriptor:
    model_name: str = ""
    model_type: str = "unknown"
    attention: AttentionDescriptor = AttentionDescriptor()
    cache: CacheDescriptor = CacheDescriptor()
    ffn: FFNDescriptor = FFNDescriptor()
    moe: MoEDescriptor = MoEDescriptor()
    runtime: RuntimeDescriptor = RuntimeDescriptor()


def descriptor_from_config(
    config: Mapping[str, Any],
    *,
    model_name: str = "",
    overrides: Mapping[str, Any] | None = None,
    precision_bytes: float = 2.0,
    num_gpus: int = 1,
    peak_bandwidth_tb: float = 1.0,
    peak_flops_tf: float = 1.0,
) -> ArchitectureDescriptor:
    """Build a descriptor from HF-style config plus optional architecture overrides."""
    arch_override = (overrides or {}).get("architecture") if overrides else None
    cfg = merge_dict(config, {k: v for k, v in (overrides or {}).items() if k != "architecture"})
    text_cfg = cfg.get("text_config") if isinstance(cfg.get("text_config"), Mapping) else None
    metric_cfg = merge_dict(cfg, text_cfg or {})
    model_l = model_name.lower()

    layers = as_int(deep_get(metric_cfg, "num_hidden_layers", "n_layers"))
    hidden = as_int(deep_get(metric_cfg, "hidden_size", "d_model"))
    heads = as_int(deep_get(metric_cfg, "num_attention_heads", "n_heads", "num_heads"))
    kv_heads = as_int(deep_get(metric_cfg, "num_key_value_heads", "num_kv_heads", "kv_n_heads"), heads)
    head_dim = as_int(deep_get(metric_cfg, "head_dim"), hidden // heads if hidden and heads else 0)
    kv_lora_rank = _optional_int(deep_get(metric_cfg, "kv_lora_rank"))
    q_lora_rank = _optional_int(deep_get(metric_cfg, "q_lora_rank"))
    qk_rope = _optional_int(deep_get(metric_cfg, "qk_rope_head_dim"))
    qk_nope = _optional_int(deep_get(metric_cfg, "qk_nope_head_dim"))
    v_head_dim = _optional_int(deep_get(metric_cfg, "v_head_dim"))
    linear_key_head_dim = _optional_int(deep_get(metric_cfg, "linear_key_head_dim"))
    linear_value_head_dim = _optional_int(deep_get(metric_cfg, "linear_value_head_dim"))
    linear_num_key_heads = _optional_int(deep_get(metric_cfg, "linear_num_key_heads"))
    linear_num_value_heads = _optional_int(deep_get(metric_cfg, "linear_num_value_heads"))
    linear_conv_kernel_dim = _optional_int(deep_get(metric_cfg, "linear_conv_kernel_dim"))
    layer_types = tuple(str(item) for item in (deep_get(metric_cfg, "layer_types") or ()) if item)
    full_attention_layers = sum(1 for item in layer_types if item == "full_attention")
    linear_attention_layers = sum(1 for item in layer_types if item == "linear_attention")
    if not layer_types:
        full_interval = as_int(deep_get(metric_cfg, "full_attention_interval"))
        if full_interval > 0:
            full_attention_layers = sum(1 for idx in range(layers) if (idx + 1) % full_interval == 0)
            linear_attention_layers = max(layers - full_attention_layers, 0)

    attention_type = str(deep_get(metric_cfg, "attention_type") or "").lower()
    cache_type = str(deep_get(metric_cfg, "cache_type") or "").lower()
    if not attention_type:
        if kv_lora_rank and qk_rope:
            attention_type = "mla"
        elif linear_attention_layers or linear_key_head_dim or linear_value_head_dim or "next" in model_l or "hybrid" in model_l:
            attention_type = "hybrid"
        elif kv_heads != heads:
            attention_type = "gqa"
        else:
            attention_type = "mha"
    if not cache_type:
        cache_type = {
            "mla": "latent_kv",
            "linear": "recurrent_state",
            "hybrid": "hybrid",
            "sparse": "sparse_kv",
        }.get(attention_type, "kv")

    moe_intermediate = as_int(deep_get(metric_cfg, "moe_intermediate_size"))
    dense_intermediate = as_int(deep_get(metric_cfg, "intermediate_size", "ffn_hidden_size"))
    first_dense = as_int(deep_get(metric_cfg, "first_k_dense_replace"))
    mlp_only = set(deep_get(metric_cfg, "mlp_only_layers") or [])
    sparse_step = max(as_int(deep_get(metric_cfg, "decoder_sparse_step"), 1), 1)
    if moe_intermediate:
        if first_dense:
            dense_layers = first_dense
            moe_layers = max(layers - first_dense, 0)
        else:
            moe_layers = sum(1 for idx in range(layers) if idx not in mlp_only and (idx + 1) % sparse_step == 0)
            dense_layers = max(layers - moe_layers, 0)
    else:
        moe_layers = 0
        dense_layers = layers

    desc = ArchitectureDescriptor(
        model_name=model_name,
        model_type=str(deep_get(metric_cfg, "model_type") or cfg.get("model_type") or "unknown"),
        attention=AttentionDescriptor(
            type=attention_type,
            num_layers=layers,
            hidden_size=hidden,
            num_attention_heads=heads,
            num_key_value_heads=kv_heads,
            head_dim=head_dim,
            kv_lora_rank=kv_lora_rank,
            q_lora_rank=q_lora_rank,
            qk_rope_head_dim=qk_rope,
            qk_nope_head_dim=qk_nope,
            v_head_dim=v_head_dim,
            linear_key_head_dim=linear_key_head_dim,
            linear_value_head_dim=linear_value_head_dim,
            linear_num_key_heads=linear_num_key_heads,
            linear_num_value_heads=linear_num_value_heads,
            linear_conv_kernel_dim=linear_conv_kernel_dim,
            layer_types=layer_types,
            full_attention_layers=full_attention_layers,
            linear_attention_layers=linear_attention_layers,
            sparse_factor=as_float(deep_get(metric_cfg, "sparse_factor"), 1.0),
        ),
        cache=CacheDescriptor(
            type=cache_type,
            num_layers=layers,
            head_dim=head_dim,
            num_key_value_heads=kv_heads,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope,
            recurrent_state_size=_optional_int(deep_get(metric_cfg, "recurrent_state_size")),
            full_attention_layers=full_attention_layers,
            linear_attention_layers=linear_attention_layers,
            linear_key_head_dim=linear_key_head_dim,
            linear_value_head_dim=linear_value_head_dim,
            linear_num_key_heads=linear_num_key_heads,
            linear_num_value_heads=linear_num_value_heads,
            linear_conv_kernel_dim=linear_conv_kernel_dim,
            sparse_factor=as_float(deep_get(metric_cfg, "cache_sparse_factor"), 1.0),
        ),
        ffn=FFNDescriptor(dense_layers=dense_layers, hidden_size=hidden, dense_intermediate_size=dense_intermediate),
        moe=MoEDescriptor(
            enabled=bool(moe_intermediate),
            moe_layers=moe_layers,
            routed_experts=as_int(deep_get(metric_cfg, "num_experts", "num_experts_per_layer", "n_routed_experts", "n_experts")),
            shared_experts=as_int(deep_get(cfg, "n_shared_experts", "num_shared_experts", "shared_experts")),
            top_k=as_int(deep_get(metric_cfg, "num_experts_per_tok", "moe_top_k", "topk", "router_topk")),
            hidden_size=hidden,
            expert_intermediate_size=moe_intermediate,
            shared_expert_intermediate_size=_optional_int(deep_get(metric_cfg, "shared_expert_intermediate_size")),
        ),
        runtime=RuntimeDescriptor(
            precision_bytes=precision_bytes,
            num_gpus=num_gpus,
            peak_bandwidth_tb=peak_bandwidth_tb,
            peak_flops_tf=peak_flops_tf,
        ),
    )
    if isinstance(arch_override, Mapping):
        desc = apply_architecture_overrides(desc, arch_override)
    return desc


def apply_architecture_overrides(desc: ArchitectureDescriptor, overrides: Mapping[str, Any]) -> ArchitectureDescriptor:
    return replace(
        desc,
        attention=_patch(desc.attention, overrides.get("attention")),
        cache=_patch(desc.cache, overrides.get("cache")),
        ffn=_patch(desc.ffn, overrides.get("ffn")),
        moe=_patch(desc.moe, overrides.get("moe")),
        runtime=_patch(desc.runtime, overrides.get("runtime")),
    )


def _patch(obj: Any, values: Any) -> Any:
    if not isinstance(values, Mapping):
        return obj
    return replace(obj, **dict(values))


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)
