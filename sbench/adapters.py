"""Adapter registry for assembling architecture descriptors and component mixes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .components import (
    AttentionComponent,
    CacheComponent,
    CostComponent,
    DenseFFNComponent,
    HybridAttentionComponent,
    LinearAttentionComponent,
    MoEComponent,
    RouterComponent,
    SparseAttentionComponent,
)
from .descriptor import ArchitectureDescriptor, descriptor_from_config


@dataclass(frozen=True)
class AdapterResult:
    name: str
    descriptor: ArchitectureDescriptor
    components: tuple[CostComponent, ...]


class BaseAdapter:
    name = "base"

    def matches(self, config: Mapping[str, Any], model_name: str, overrides: Mapping[str, Any] | None) -> bool:
        return False

    def build(self, config: Mapping[str, Any], model_name: str, **kwargs: Any) -> AdapterResult:
        descriptor = descriptor_from_config(config, model_name=model_name, **kwargs)
        return AdapterResult(self.name, descriptor, self.components_for(descriptor))

    def components_for(self, descriptor: ArchitectureDescriptor) -> tuple[CostComponent, ...]:
        attention: CostComponent
        if descriptor.attention.type == "linear":
            attention = LinearAttentionComponent()
        elif descriptor.attention.type in {"sparse", "dsa"}:
            attention = SparseAttentionComponent()
        elif descriptor.attention.type == "hybrid":
            attention = HybridAttentionComponent()
        else:
            attention = AttentionComponent()
        return (attention, CacheComponent(), DenseFFNComponent(), MoEComponent(), RouterComponent())


class DenseTransformerAdapter(BaseAdapter):
    name = "dense_transformer"

    def matches(self, config: Mapping[str, Any], model_name: str, overrides: Mapping[str, Any] | None) -> bool:
        return True


class QwenMoEAdapter(BaseAdapter):
    name = "qwen_moe"

    def matches(self, config: Mapping[str, Any], model_name: str, overrides: Mapping[str, Any] | None) -> bool:
        return "qwen" in model_name.lower() and _has(config, "moe_intermediate_size")


class QwenHybridAdapter(BaseAdapter):
    name = "qwen_hybrid"

    def matches(self, config: Mapping[str, Any], model_name: str, overrides: Mapping[str, Any] | None) -> bool:
        attn_type = _override_attention_type(overrides) or _config_attention_type(config)
        return "qwen" in model_name.lower() and ("next" in model_name.lower() or _has(config, "linear_key_head_dim") or attn_type in {"hybrid", "linear", "sparse"})


class DeepSeekMLAAdapter(BaseAdapter):
    name = "deepseek_mla"

    def matches(self, config: Mapping[str, Any], model_name: str, overrides: Mapping[str, Any] | None) -> bool:
        return _has(config, "kv_lora_rank") and _has(config, "qk_rope_head_dim")


class DeepSeekSparseAdapter(BaseAdapter):
    name = "deepseek_sparse"

    def matches(self, config: Mapping[str, Any], model_name: str, overrides: Mapping[str, Any] | None) -> bool:
        attn_type = _override_attention_type(overrides)
        return "deepseek" in model_name.lower() and attn_type in {"sparse", "dsa"}


class LinearAttentionAdapter(BaseAdapter):
    name = "linear_attention"

    def matches(self, config: Mapping[str, Any], model_name: str, overrides: Mapping[str, Any] | None) -> bool:
        return _override_attention_type(overrides) == "linear" or _config_attention_type(config) == "linear"


ADAPTERS: tuple[BaseAdapter, ...] = (
    DeepSeekSparseAdapter(),
    LinearAttentionAdapter(),
    QwenHybridAdapter(),
    DeepSeekMLAAdapter(),
    QwenMoEAdapter(),
    DenseTransformerAdapter(),
)


def resolve_adapter(config: Mapping[str, Any], model_name: str = "", overrides: Mapping[str, Any] | None = None, **kwargs: Any) -> AdapterResult:
    for adapter in ADAPTERS:
        if adapter.matches(config, model_name, overrides):
            return adapter.build(config, model_name, overrides=overrides, **kwargs)
    raise AssertionError("DenseTransformerAdapter should always match")


def _has(data: Any, key: str) -> bool:
    if isinstance(data, Mapping):
        if key in data and data[key] is not None:
            return True
        return any(_has(value, key) for value in data.values())
    return False


def _config_attention_type(config: Mapping[str, Any]) -> str:
    value = _find(config, "attention_type")
    return str(value).lower() if value else ""


def _find(data: Any, key: str) -> Any:
    if isinstance(data, Mapping):
        if key in data and data[key] is not None:
            return data[key]
        for value in data.values():
            found = _find(value, key)
            if found is not None:
                return found
    return None


def _override_attention_type(overrides: Mapping[str, Any] | None) -> str:
    if not overrides:
        return ""
    arch = overrides.get("architecture") if isinstance(overrides, Mapping) else None
    attention = arch.get("attention") if isinstance(arch, Mapping) else None
    return str(attention.get("type", "")).lower() if isinstance(attention, Mapping) else ""
