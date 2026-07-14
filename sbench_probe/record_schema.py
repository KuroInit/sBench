"""Stable probe record schema."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ProbeRecord:
    forward_pass_id: int
    forward_mode: str
    latency: float
    seq_lens_sum: int
    batch_size: int
    schema_version: int = 1
    num_gpus: int = 1
    gpu_num: int | None = None
    gpu_raw_type: str = "unknown"
    expert_activation: float = -1.0
    expert_utilization: float | None = None
    processed_tokens: int | None = None
    per_req_info: list[dict[str, Any]] = field(default_factory=list)
    raw_forward_mode: str | None = None
    raw_probe_source: str = "timing_only"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
