"""Component-wise S-MFU/S-MBU estimator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .components import DEFAULT_COMPONENTS, ComponentCost, CostComponent
from .descriptor import ArchitectureDescriptor


@dataclass(frozen=True)
class EstimateResult:
    prefill_smbu: float
    prefill_smfu: float
    decoding_smbu: float
    decoding_smfu: float
    prefill_tp: float
    decoding_throughput: float
    ttft: float
    tpot: float
    kv_size: float


def estimate_records(
    arch: ArchitectureDescriptor,
    records: Iterable[dict],
    *,
    components: Iterable[CostComponent] = DEFAULT_COMPONENTS,
) -> EstimateResult:
    prefill_smbu: list[tuple[float, float]] = []
    prefill_smfu: list[tuple[float, float]] = []
    decode_smbu: list[tuple[float, float]] = []
    decode_smfu: list[tuple[float, float]] = []
    prefill_tps: list[float] = []
    decode_tps: list[float] = []
    ttfts: list[float] = []
    tpots: list[float] = []
    kv_sizes: list[float] = []
    comps = tuple(components)

    for record in usable_records(records):
        latency = float(record.get("latency", 0) or 0)
        costs = estimate_component_breakdown(arch, record, components=comps)
        total = _total_cost(costs)
        kv_sizes.append(total.cache_units * 1e6)
        if record.get("forward_mode") == "prefill":
            throughput = int(record.get("seq_lens_sum", 0)) / latency
            ttfts.append(latency)
            prefill_tps.append(throughput)
            prefill_smbu.append((_smbu(arch, total, latency), latency))
            prefill_smfu.append((_smfu(arch, total, throughput), latency))
        else:
            throughput = int(record.get("batch_size", 1)) / latency
            tpots.append(latency)
            decode_tps.append(throughput)
            decode_smbu.append((_smbu(arch, total, latency), latency))
            decode_smfu.append((_smfu(arch, total, throughput), latency))

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


def estimate_component_breakdown(
    arch: ArchitectureDescriptor,
    record: dict,
    *,
    components: Iterable[CostComponent] = DEFAULT_COMPONENTS,
) -> list[ComponentCost]:
    return [component.estimate(arch, record) for component in components]


def usable_records(records: Iterable[dict]) -> list[dict]:
    out = []
    for source in records:
        record = dict(source)
        if record.get("forward_mode") == "prefill" and int(record.get("seq_lens_sum", 0)) <= 10:
            continue
        latency = float(record.get("latency", 0) or 0)
        if latency <= 0:
            continue
        out.append(record)
    return out


def _total_cost(costs: list[ComponentCost]) -> ComponentCost:
    total = ComponentCost(name="total")
    for cost in costs:
        total = total.plus(cost)
    return total


def _smbu(arch: ArchitectureDescriptor, cost: ComponentCost, latency: float) -> float:
    rt = arch.runtime
    return ((cost.bandwidth_units + cost.cache_units) * rt.precision_bytes / latency) / (rt.num_gpus * rt.peak_bandwidth_tb)


def _smfu(arch: ArchitectureDescriptor, cost: ComponentCost, throughput: float) -> float:
    rt = arch.runtime
    return ((cost.flops_units + cost.attention_score_units) * 2 * throughput) / (rt.num_gpus * rt.peak_flops_tf / 2)


def _weighted(values: list[tuple[float, float]]) -> float:
    weight = sum(w for _, w in values)
    return sum(value * w for value, w in values) / weight if weight else 0.0
