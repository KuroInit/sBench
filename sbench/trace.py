"""Shared probe-trace rules used by orchestration, analysis, and validation."""

from __future__ import annotations

from typing import Any


def required_forward_modes(dataset_config: dict[str, Any] | None = None, dataset: str | None = None) -> set[str]:
    """Return forward phases required for a valid metric sample.

    Prefill-only workloads are valid with prefill records only. Other request
    workloads need both prefill and decode records so TTFT and TPOT estimates are
    based on observed phases.
    """

    cfg = dataset_config or {}
    if cfg.get("benchmark_type") == "prefill" or dataset == "batched_prefill":
        return {"prefill"}
    return {"prefill", "decode"}


def has_real_expert_activation(record: dict[str, Any]) -> bool:
    """Whether a record contains a positive, routed-expert activation count."""

    try:
        return float(record.get("expert_activation") or 0) > 0
    except (TypeError, ValueError):
        return False


def moe_activation_note(records: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> str:
    """Return a user-facing note for MoE traces with missing activation data."""

    if not records:
        return ""
    missing_sources = sorted(
        {
            str(record.get("raw_probe_source", "unknown"))
            for record in records
            if not has_real_expert_activation(record)
        }
    )
    if not missing_sources:
        return ""
    return (
        "MoE expert activation missing for some records; MoE bandwidth is "
        "under-estimated as zero for those records. raw_probe_source="
        + ",".join(missing_sources)
    )
