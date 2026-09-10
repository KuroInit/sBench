"""Concurrent DCGM hardware profiling for sBench runs.

Spawns `dcgmi dmon` alongside each benchmark run, streams per-GPU profiler
field samples (SM active, DRAM active, ...) to a per-run CSV, and joins those
samples with probe-record timelines to produce per-phase (prefill/decode)
utilization aggregates. The aggregates form `telemetry_summary.csv`, the
lightweight validation input already understood by
`sbench.validation.telemetry_comparison_rows`.

The sampler is strictly optional and failure-tolerant: missing binaries,
missing host engine, or unsupported fields degrade to a disabled sampler and
never fail a run. DCGM targets datacenter GPUs; consumer GPUs (GeForce) are
expected to report the sampler as unavailable.
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import threading
import time
from bisect import bisect_right
from pathlib import Path
from typing import Any, Iterable

# DCGM field IDs (dcgm_fields.h). 100x IDs are DCGM_FI_PROF_* ratio fields in [0, 1].
DEFAULT_FIELDS: tuple[int, ...] = (1001, 1002, 1003, 1004, 1005)

FIELD_ID_TO_NAME: dict[int, str] = {
    1001: "gr_engine_active",
    1002: "sm_active",
    1003: "sm_occupancy",
    1004: "pipe_tensor_active",
    1005: "dram_active",
    1006: "pipe_fp64_active",
    1007: "pipe_fp32_active",
    1008: "pipe_fp16_active",
    1009: "pcie_tx_throughput",
    1010: "pcie_rx_throughput",
}

# `dcgmi dmon -e` headers print DCGM field short names; the default (no -e)
# dmon table mirrors nvidia-smi dmon and prints bare "sm"/"mem" columns in
# percent. Both map onto canonical fraction columns here.
SHORT_NAME_TO_FIELD: dict[str, int] = {
    "gr_engine_active": 1001,
    "sm_active": 1002,
    "sm_occupancy": 1003,
    "pipe_tensor_active": 1004,
    "dram_active": 1005,
    "pipe_fp64_active": 1006,
    "pipe_fp32_active": 1007,
    "pipe_fp16_active": 1008,
}

_PERCENT_ALIASES: dict[str, str] = {
    # nvidia-smi-style default dmon columns: percent 0-100, scaled to [0, 1]
    "sm": "sm_active",
    "mem": "dram_active",
}

# Columns the summary exposes; aliases kept in one place.
SM_COLUMN = "DCGM_FI_PROF_SM_ACTIVE"
DRAM_COLUMN = "DCGM_FI_PROF_DRAM_ACTIVE"

DEFAULT_INTERVAL_MS = 100
DCGM_CSV_PREFIX = "dcgm_dmon_"

_DMON_UNIT_LINE_MARK = "ID"


class DmonParser:
    """Incremental parser for `dcgmi dmon` stdout.

    Feed every stdout line; when a data row completes it returns a sample dict
    with a wall-clock `ts` plus one canonical fraction column per recognized
    field. Header/unit/repeat-header lines and unparseable rows return None.
    """

    def __init__(self, requested_fields: Iterable[int] = DEFAULT_FIELDS) -> None:
        self.requested_fields = tuple(requested_fields)
        # Ordered value columns: (canonical name or None to ignore, percent-scaled)
        self.columns: list[tuple[str | None, bool]] = []

    def feed(self, line: str, *, ts: float | None = None) -> dict[str, Any] | None:
        text = line.strip()
        if not text:
            return None
        if text.startswith("#"):
            self._parse_header(text)
            return None
        tokens = text.split()
        if not tokens:
            return None
        if tokens[0].upper() == _DMON_UNIT_LINE_MARK:
            return None  # the "% % % ..." unit line under the header
        if not self.columns:
            return None
        entity_id, value_tokens = self._split_entity(tokens)
        if entity_id is None or len(value_tokens) < len(self.columns):
            return None
        # Versions differ in leading entity/id tokens; values are the final columns.
        values = value_tokens[-len(self.columns):]
        sample: dict[str, Any] = {"gpu_id": entity_id, "ts": ts if ts is not None else time.time()}
        recognized = False
        for (name, percent), raw in zip(self.columns, values):
            if name is None:
                continue
            value = _as_float(raw)
            if value is None:
                continue
            if percent:
                value = value / 100.0  # default dmon sm/mem columns are 0-100
            sample[name] = value
            recognized = True
        return sample if recognized else None

    def _parse_header(self, text: str) -> None:
        tokens = text.lstrip("#").split()
        if not tokens or tokens[0].rstrip(":").lower() not in {"entity"}:
            return
        columns: list[tuple[str | None, bool]] = []
        for token in tokens[1:]:
            lowered = token.strip().strip("%").lower()
            if lowered in _PERCENT_ALIASES:
                columns.append((_PERCENT_ALIASES[lowered], True))
            else:
                columns.append((self._canonical_name(token), False))
        if columns:
            self.columns = columns

    @staticmethod
    def _canonical_name(token: str) -> str | None:
        token = token.strip().strip("%")
        if not token:
            return None
        if token.isdigit():
            return FIELD_ID_TO_NAME.get(int(token))
        return FIELD_ID_TO_NAME.get(SHORT_NAME_TO_FIELD.get(token.lower(), -1))

    @staticmethod
    def _split_entity(tokens: list[str]) -> tuple[int | None, list[str]]:
        """Entity column is `GPU 0` / `GPU_I 3` / plain `0` depending on version."""
        if tokens and tokens[0].upper() in {"GPU", "GPU_I", "GPU_CI", "SW", "SW_I", "SW_CI", "CPU", "LINK"}:
            if len(tokens) >= 2 and tokens[1].isdigit():
                return int(tokens[1]), tokens[2:]
            return None, []
        if tokens and tokens[0].isdigit():
            return int(tokens[0]), tokens[1:]
        return None, []


def _as_float(token: str) -> float | None:
    try:
        return float(token)
    except ValueError:
        return None


class DcgmSampler:
    """Runs `dcgmi dmon` for the duration of one benchmark run.

    `start()` launches the dmon process and a background reader thread that
    streams parsed samples into `out_path` (CSV, one row per GPU per tick).
    `stop()` is idempotent and returns a status dict for metadata.
    """

    def __init__(
        self,
        out_path: Path,
        *,
        interval_ms: int = DEFAULT_INTERVAL_MS,
        fields: Iterable[int] = DEFAULT_FIELDS,
        dcgm_bin: str = "dcgmi",
        hostengine_bin: str = "nv-hostengine",
        ready_timeout_s: float = 20.0,
    ) -> None:
        self.out_path = Path(out_path)
        self.interval_ms = max(int(interval_ms), 10)
        self.fields = tuple(int(field) for field in fields) or DEFAULT_FIELDS
        self.dcgm_bin = dcgm_bin
        self.hostengine_bin = hostengine_bin
        self.ready_timeout_s = float(ready_timeout_s)
        self.status: str = "not_started"
        self._reader_error: str = ""
        self.samples: list[dict[str, Any]] = []
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._result: dict[str, Any] | None = None

    @property
    def started(self) -> bool:
        return self._proc is not None

    def start(self) -> bool:
        if self._proc is not None:
            return True
        if shutil.which(self.dcgm_bin) is None:
            self.status = "unavailable: dcgmi not found"
            return False
        try:
            self.out_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self.status = f"unavailable: cannot create {self.out_path.parent}: {exc}"
            return False
        if not self._launch():
            detail = self._reader_error or "dcgmi dmon produced no samples"
            self.status = f"unavailable: {detail}"
            return False
        self.status = "ok"
        return True

    def _launch(self) -> bool:
        if self._try_launch():
            return True
        # DCGM client tools need nv-hostengine; best-effort start and retry once.
        engine = shutil.which(self.hostengine_bin)
        if engine is None:
            return False
        try:
            subprocess.run([engine], capture_output=True, timeout=10, check=False)
        except (OSError, subprocess.TimeoutExpired):
            return False
        return self._try_launch()

    def _try_launch(self) -> bool:
        cmd = [
            self.dcgm_bin,
            "dmon",
            "-d",
            str(self.interval_ms),
            "-e",
            ",".join(str(field) for field in self.fields),
        ]
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            self._reader_error = str(exc)
            self._proc = None
            return False
        self._reader_error = ""
        self._thread = threading.Thread(target=self._read_loop, name="sbench-dcgm", daemon=True)
        self._thread.start()
        proc = self._proc
        deadline = time.monotonic() + self.ready_timeout_s
        while time.monotonic() < deadline:
            if self._stop_event.is_set():
                return False
            with self._lock:
                if self.samples:
                    return True
            if proc.poll() is not None:
                self._finalize_process()
                return False
            if self._thread is None or not self._thread.is_alive():
                self._finalize_process()
                return False
            time.sleep(0.1)
        self._finalize_process()
        return False

    def _read_loop(self) -> None:
        parser = DmonParser(self.fields)
        proc = self._proc
        assert proc is not None and proc.stdout is not None
        try:
            with self.out_path.open("w", newline="", encoding="utf-8") as handle:
                writer: csv.DictWriter | None = None
                for line in proc.stdout:
                    if self._stop_event.is_set():
                        break
                    sample = parser.feed(line)
                    if sample is None:
                        continue
                    if writer is None:
                        columns = ["ts", "gpu_id"] + sorted(k for k in sample if k not in {"ts", "gpu_id"})
                        writer = csv.DictWriter(handle, fieldnames=columns, restval="")
                        writer.writeheader()
                    writer.writerow({k: sample.get(k, "") for k in writer.fieldnames})
                    handle.flush()
                    with self._lock:
                        self.samples.append(sample)
        except Exception as exc:  # never kill the run over telemetry
            self._reader_error = f"reader failed: {exc}"
        finally:
            self._finalize_process()

    def _finalize_process(self) -> None:
        proc = self._proc
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        if proc.stdout:
            proc.stdout.close()
        self._proc = None

    def stop(self) -> dict[str, Any]:
        if self._result is not None:
            return self._result
        self._stop_event.set()
        self._finalize_process()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None
        gpus = sorted({sample["gpu_id"] for sample in self.samples if "gpu_id" in sample})
        self._result = {
            "status": self.status if self.samples else ("ok" if self.status == "ok" else self.status),
            "interval_ms": self.interval_ms,
            "fields": list(self.fields),
            "samples": len(self.samples),
            "gpus": gpus,
            "path": self.out_path.name,
        }
        return self._result


def sampler_from_config(cfg: dict[str, Any] | None, out_path: Path) -> DcgmSampler | None:
    """Build a sampler from the sweep `dcgm:` config block. None when disabled."""
    cfg = cfg or {}
    if cfg.get("enabled", True) is False:
        return None
    interval = cfg.get("interval_ms", DEFAULT_INTERVAL_MS)
    try:
        interval = int(interval)
    except (TypeError, ValueError):
        interval = DEFAULT_INTERVAL_MS
    fields = cfg.get("fields") or DEFAULT_FIELDS
    try:
        fields = tuple(int(field) for field in fields)
    except (TypeError, ValueError):
        fields = DEFAULT_FIELDS
    return DcgmSampler(out_path, interval_ms=interval, fields=fields)


def stop_sampler(sampler: DcgmSampler | None) -> dict[str, Any] | None:
    """Idempotently stop a sampler and return its metadata status dict."""
    if sampler is None:
        return None
    return sampler.stop()


def load_dcgm_csv(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows: list[dict[str, Any]] = []
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = {}
            for key, value in raw.items():
                if key in {"ts", "gpu_id"}:
                    row[key] = float(value) if key == "ts" else int(float(value))
                    continue
                if key is None:
                    continue
                value_f = _as_float(value or "")
                if value_f is not None:
                    row[key] = value_f
            rows.append(row)
        return rows


def phase_windows(records: Iterable[dict[str, Any]]) -> list[tuple[str, float, float]]:
    """Reconstruct (phase, start, end) wall-clock windows from probe records.

    Probe records carry `ts` (unix seconds at forward completion) and `latency`
    (seconds), so each forward pass covers [ts - latency, ts]. Records are
    sequential on the serving rank, so the windows tile the workload.
    """
    windows: list[tuple[str, float, float]] = []
    ordered = sorted(records, key=lambda rec: (rec.get("ts") or 0.0, rec.get("forward_pass_id") or 0))
    for record in ordered:
        ts = record.get("ts")
        mode = record.get("forward_mode")
        latency = record.get("latency") or 0
        try:
            ts = float(ts)
            latency = float(latency)
        except (TypeError, ValueError):
            continue
        if mode not in {"prefill", "decode"} or latency <= 0:
            continue
        windows.append((mode, ts - latency, ts))
    return windows


def aggregate_by_phase(
    samples: Iterable[dict[str, Any]],
    windows: list[tuple[str, float, float]],
) -> dict[str, dict[str, float]]:
    """Mean SM/DRAM/tensor activity per phase for samples inside phase windows.

    Rows with multiple GPUs (TP>1) are averaged together; samples outside every
    window (startup, idle gaps) are excluded from phase means.
    """
    starts = [start for _, start, _ in windows]
    totals: dict[str, dict[str, list[float]]] = {}
    counts: dict[str, int] = {}
    for sample in samples:
        ts = sample.get("ts")
        if ts is None:
            continue
        idx = bisect_right(starts, ts) - 1
        if idx < 0 or not (windows[idx][1] <= ts <= windows[idx][2]):
            continue
        phase = windows[idx][0]
        counts[phase] = counts.get(phase, 0) + 1
        bucket = totals.setdefault(phase, {})
        for key, value in sample.items():
            if key in {"ts", "gpu_id"} or not isinstance(value, (int, float)):
                continue
            bucket.setdefault(key, []).append(float(value))
    out: dict[str, dict[str, float]] = {}
    for phase, values in totals.items():
        row: dict[str, float] = {}
        for key, nums in values.items():
            row[f"{key}_mean"] = sum(nums) / len(nums)
        row["samples"] = counts.get(phase, 0)
        out[phase] = row
    return out


def telemetry_rows_for_run(
    *,
    slug: str,
    batch_size: int | str,
    dataset: str,
    dcgm_csv: Path,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Per-phase summary rows for one run, in the validate_estimator schema."""
    try:
        samples = load_dcgm_csv(dcgm_csv)
    except (OSError, ValueError):
        return []
    if not samples:
        return []
    windows = phase_windows(records)
    aggregated = aggregate_by_phase(samples, windows)
    has_phase_rows = bool(aggregated)
    rows: list[dict[str, Any]] = []
    if not has_phase_rows:
        # Legacy probe records without `ts`: whole-run fallback row.
        sm = [s["sm_active"] for s in samples if isinstance(s.get("sm_active"), (int, float))]
        dram = [s["dram_active"] for s in samples if isinstance(s.get("dram_active"), (int, float))]
        aggregated = {"run": {"sm_active_mean": sum(sm) / len(sm) if sm else None, "dram_active_mean": sum(dram) / len(dram) if dram else None, "samples": len(samples)}}
    for phase, stats in aggregated.items():
        sm_mean = stats.get("sm_active_mean")
        dram_mean = stats.get("dram_active_mean")
        row: dict[str, Any] = {
            "slug": slug,
            "batch_size": str(batch_size),
            "dataset": dataset,
            "phase": phase,
            "samples": int(stats.get("samples", 0)),
        }
        row["gpu_util_pct"] = round(sm_mean * 100, 3) if sm_mean is not None else ""
        row["memory_util_pct"] = round(dram_mean * 100, 3) if dram_mean is not None else ""
        row[SM_COLUMN] = round(sm_mean, 4) if sm_mean is not None else ""
        row[DRAM_COLUMN] = round(dram_mean, 4) if dram_mean is not None else ""
        tensor_mean = stats.get("pipe_tensor_active_mean")
        if tensor_mean is not None:
            row["DCGM_FI_PROF_PIPE_TENSOR_ACTIVE"] = round(tensor_mean, 4)
        rows.append(row)
    return rows


def iter_run_leaves(results_dir: Path) -> Iterable[tuple[str, str, str, Path]]:
    """Yield (slug, batch size N as str, dataset, leaf_dir) for leaves holding DCGM CSVs."""
    root = Path(results_dir)
    if not root.exists():
        return
    for dcgm_csv in sorted(root.rglob(f"{DCGM_CSV_PREFIX}*.csv")):
        leaf = dcgm_csv.parent
        rel = leaf.relative_to(root).parts
        # results/<slug>/bs<N>/<dataset>/<model_leaf>/dcgm_dmon_*.csv
        if len(rel) < 4:
            continue
        model_leaf, dataset, bs_dir, slug = rel[-1], rel[-2], rel[-3], rel[-4]
        if not bs_dir.startswith("bs") or not bs_dir[2:].isdigit():
            continue
        yield slug, bs_dir[2:], dataset, leaf


def telemetry_summary_rows(results_dir: Path, sample_limit: int | None = None) -> list[dict[str, Any]]:
    """Build the `telemetry_summary.csv` rows for an entire results tree."""
    rows: list[dict[str, Any]] = []
    for slug, batch_size, dataset, leaf in iter_run_leaves(results_dir):
        records_path = _latest(leaf, "server_records_*.jsonl")
        dcgm_csv = _latest(leaf, f"{DCGM_CSV_PREFIX}*.csv")
        if records_path is None or dcgm_csv is None:
            continue
        try:
            records = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
        except (OSError, json.JSONDecodeError):
            continue
        rows.extend(
            telemetry_rows_for_run(
                slug=slug,
                batch_size=batch_size,
                dataset=dataset,
                dcgm_csv=dcgm_csv,
                records=records,
            )
        )
    return rows


def _latest(path: Path, pattern: str) -> Path | None:
    matches = sorted(path.glob(pattern))
    return matches[-1] if matches else None
