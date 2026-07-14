"""YAML/JSON override loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_overrides(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    source = Path(path)
    text = source.read_text()
    if source.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("YAML overrides require PyYAML; use JSON or install pyyaml") from exc
    return yaml.safe_load(text) or {}
