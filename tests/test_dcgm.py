import csv
import stat
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sbench import dcgm

V3_HEADER = "#Entity        ID             GR_ENGINE_ACTIVE   SM_ACTIVE   SM_OCCUPANCY   PIPE_TENSOR_ACTIVE   DRAM_ACTIVE"
V3_UNITS = "               %              %                  %           %              %                    %"


def test_parser_accepts_v3_named_headers():
    parser = dcgm.DmonParser()
    assert parser.feed(V3_HEADER) is None
    assert parser.feed(V3_UNITS) is None
    sample = parser.feed("GPU 0          0              0.423              0.412       0.281          0.301                0.187")
    assert sample["gpu_id"] == 0
    assert sample["sm_active"] == pytest.approx(0.412)
    assert sample["dram_active"] == pytest.approx(0.187)
    assert sample["pipe_tensor_active"] == pytest.approx(0.301)
    assert "ts" in sample


def test_parser_accepts_numeric_field_id_headers():
    parser = dcgm.DmonParser()
    assert parser.feed("# Entity        id      1002     1005") is None
    sample = parser.feed("GPU 0           0       0.412    0.187")
    assert sample["sm_active"] == pytest.approx(0.412)
    assert sample["dram_active"] == pytest.approx(0.187)


def test_parser_scales_default_percent_columns():
    parser = dcgm.DmonParser()
    assert parser.feed("# Entity        id      pwr    gtemp    mtemp    sm     mem    enc    dec    mclk    pclk") is None
    sample = parser.feed("GPU 0           0       67     37       34       99     62     0      0      877     1200")
    assert sample["sm_active"] == pytest.approx(0.99)
    assert sample["dram_active"] == pytest.approx(0.62)
    assert "pwr" not in sample




def test_sampler_streams_samples_to_csv(tmp_path):
    dcgmi = _fake_dcgmi(tmp_path)
    out = tmp_path / "leaf" / "dcgm_dmon_sharegpt_1.csv"
    sampler = dcgm.DcgmSampler(out, interval_ms=20, dcgm_bin=str(dcgmi), ready_timeout_s=10)
    assert sampler.start() is True
    time.sleep(0.25)
    result = sampler.stop()
    assert result["status"] == "ok"
    assert result["samples"] > 0
    assert result["gpus"] == [0]
    assert result["path"] == out.name
    rows = dcgm.load_dcgm_csv(out)
    assert len(rows) == result["samples"]
    assert {"ts", "gpu_id", "sm_active", "dram_active"} <= set(rows[0])
    assert 0.0 <= rows[0]["sm_active"] <= 1.0
    assert sampler.stop() == result
def test_parser_skips_na_rows_and_repeated_headers():
    parser = dcgm.DmonParser()
    parser.feed("#Entity        ID             GR_ENGINE_ACTIVE   SM_ACTIVE   DRAM_ACTIVE")
    first = parser.feed("GPU 0          0              N/A                0.500       0.200")
    assert first["sm_active"] == 0.5 and first["dram_active"] == 0.2
    assert "gr_engine_active" not in first and "ts" in first
    parser.feed("#Entity        ID             GR_ENGINE_ACTIVE   SM_ACTIVE   DRAM_ACTIVE")
    again = parser.feed("GPU 0          0              0.300              0.600       0.250")
    assert again["gr_engine_active"] == 0.3
    assert again["sm_active"] == 0.6
def test_sampler_reports_missing_binary(tmp_path):
    sampler = dcgm.DcgmSampler(tmp_path / "out.csv", dcgm_bin="dcgmi-definitely-missing", ready_timeout_s=2)
    assert sampler.start() is False
    assert "not found" in sampler.status
    result = sampler.stop()
    assert result["samples"] == 0
    assert "unavailable" in result["status"]


def test_sampler_survives_dying_dmon(tmp_path):
    dcgmi = tmp_path / "dcgmi-crash"
    dcgmi.write_text("#!/usr/bin/env python3\nprint('boom')\n")
    dcgmi.chmod(dcgmi.stat().st_mode | stat.S_IEXEC)
    sampler = dcgm.DcgmSampler(tmp_path / "out.csv", interval_ms=20, dcgm_bin=str(dcgmi), ready_timeout_s=3)
    assert sampler.start() is False
    assert sampler.stop()["samples"] == 0


def test_sampler_from_config_defaults_and_overrides(tmp_path):
    assert dcgm.sampler_from_config(None, tmp_path / "a.csv") is not None
    assert dcgm.sampler_from_config({"enabled": False}, tmp_path / "b.csv") is None
    sampler = dcgm.sampler_from_config({"interval_ms": 250, "fields": [1002, 1005]}, tmp_path / "c.csv")
    assert sampler.interval_ms == 250
    assert sampler.fields == (1002, 1005)
    fallback = dcgm.sampler_from_config({"interval_ms": "junk", "fields": "junk"}, tmp_path / "d.csv")
    assert fallback.interval_ms == dcgm.DEFAULT_INTERVAL_MS
    assert fallback.fields == dcgm.DEFAULT_FIELDS


def test_phase_windows_tile_forward_passes():
    t = 1000.0
    records = [
        {"forward_pass_id": 1, "forward_mode": "prefill", "latency": 1.0, "ts": t + 1.0},
        {"forward_pass_id": 2, "forward_mode": "decode", "latency": 0.5, "ts": t + 2.0},
        {"forward_pass_id": 3, "forward_mode": "idle", "latency": 1.0, "ts": t + 3.0},
        {"forward_pass_id": 4, "forward_mode": "decode", "latency": 0, "ts": t + 4.0},
    ]
    windows = dcgm.phase_windows(records)
    assert windows == [("prefill", t, t + 1.0), ("decode", t + 1.5, t + 2.0)]


def test_aggregate_by_phase_joins_samples_to_windows():
    t = 1000.0
    windows = [("prefill", t, t + 1.0), ("decode", t + 1.5, t + 2.0)]
    samples = [
        {"ts": t + 0.5, "gpu_id": 0, "sm_active": 0.8, "dram_active": 0.3},
        {"ts": t + 0.75, "gpu_id": 1, "sm_active": 0.6, "dram_active": 0.5},
        {"ts": t + 1.75, "gpu_id": 0, "sm_active": 0.4, "dram_active": 0.7},
        {"ts": t + 3.0, "gpu_id": 0, "sm_active": 0.1, "dram_active": 0.1},
    ]
    agg = dcgm.aggregate_by_phase(samples, windows)
    assert agg["prefill"]["sm_active_mean"] == pytest.approx(0.7)
    assert agg["prefill"]["dram_active_mean"] == pytest.approx(0.4)
    assert agg["prefill"]["samples"] == 2
    assert agg["decode"]["sm_active_mean"] == pytest.approx(0.4)
    assert agg["decode"]["samples"] == 1


def test_telemetry_rows_for_run_schema(tmp_path):
    out = tmp_path / "dcgm_dmon_sharegpt_1.csv"
    _write_dcgm_csv(out, [
        (1000.5, 0, 0.8, 0.3),
        (1001.5, 0, 0.4, 0.7),
    ])
    records = [
        {"forward_pass_id": 1, "forward_mode": "prefill", "latency": 1.0, "ts": 1001.0},
        {"forward_pass_id": 2, "forward_mode": "decode", "latency": 0.5, "ts": 1002.0},
    ]
    rows = dcgm.telemetry_rows_for_run(slug="qwen3_30b", batch_size=2, dataset="sharegpt", dcgm_csv=out, records=records)
    by_phase = {row["phase"]: row for row in rows}
    assert set(by_phase) == {"prefill", "decode"}
    prefill = by_phase["prefill"]
    assert prefill["slug"] == "qwen3_30b"
    assert prefill["batch_size"] == "2"
    assert prefill["dataset"] == "sharegpt"
    assert prefill["gpu_util_pct"] == pytest.approx(80.0)
    assert prefill["memory_util_pct"] == pytest.approx(30.0)
    assert prefill["DCGM_FI_PROF_SM_ACTIVE"] == pytest.approx(0.8)
    assert prefill["DCGM_FI_PROF_DRAM_ACTIVE"] == pytest.approx(0.3)
    assert prefill["samples"] == 1
    assert by_phase["decode"]["gpu_util_pct"] == pytest.approx(40.0)


def test_telemetry_rows_fall_back_to_whole_run_without_ts(tmp_path):
    out = tmp_path / "dcgm_dmon_mmlu_pro_1.csv"
    _write_dcgm_csv(out, [
        (10.0, 0, 0.5, 0.25),
        (10.2, 0, 0.7, 0.35),
    ])
    records = [{"forward_pass_id": 1, "forward_mode": "prefill", "latency": 0.1}]
    rows = dcgm.telemetry_rows_for_run(slug="s", batch_size="8", dataset="mmlu_pro", dcgm_csv=out, records=records)
    assert len(rows) == 1
    assert rows[0]["phase"] == "run"
    assert rows[0]["gpu_util_pct"] == pytest.approx(60.0)
    assert rows[0]["memory_util_pct"] == pytest.approx(30.0)


def test_telemetry_summary_rows_scans_results_tree(tmp_path):
    leaf = tmp_path / "qwen3_30b" / "bs2" / "sharegpt" / "Qwen__Qwen3-30B-A3B"
    leaf.mkdir(parents=True)
    records = [
        {"forward_pass_id": 1, "forward_mode": "prefill", "latency": 1.0, "ts": 1001.0},
        {"forward_pass_id": 2, "forward_mode": "decode", "latency": 0.5, "ts": 1002.0},
    ]
    (leaf / "server_records_sharegpt_1.jsonl").write_text(
        "\n".join(__import__("json").dumps(r) for r in records) + "\n"
    )
    _write_dcgm_csv(leaf / "dcgm_dmon_sharegpt_1.csv", [(1000.5, 0, 0.8, 0.3), (1001.5, 0, 0.4, 0.7)])
    rows = dcgm.telemetry_summary_rows(tmp_path)
    phases = {row["phase"] for row in rows}
    assert phases == {"prefill", "decode"}
    assert all(row["slug"] == "qwen3_30b" and row["batch_size"] == "2" and row["dataset"] == "sharegpt" for row in rows)


def _fake_dcgmi(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir(exist_ok=True)
    dcgmi = bin_dir / "dcgmi"
    dcgmi.write_text(
        "#!/usr/bin/env python3\n"
        "import time\n"
        f"print({V3_HEADER!r})\n"
        f"print({V3_UNITS!r})\n"
        "i = 0\n"
        "while True:\n"
        "    sm = 0.2 + 0.1 * (i % 5)\n"
        "    dram = 0.8 - 0.1 * (i % 5)\n"
        "    print(f'GPU 0          0              0.500              {sm:.3f}       0.300          0.400                {dram:.3f}', flush=True)\n"
        "    i += 1\n"
        "    time.sleep(0.02)\n"
    )
    dcgmi.chmod(dcgmi.stat().st_mode | stat.S_IEXEC)
    return dcgmi


def _write_dcgm_csv(path: Path, rows: list[tuple[float, int, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ts", "gpu_id", "gr_engine_active", "sm_active", "sm_occupancy", "pipe_tensor_active", "dram_active"])
        for ts, gpu, _gr, sm, _occ, _pt, dram in [(t, g, 0.5, s, 0.3, 0.4, d) for t, g, s, d in rows]:
            writer.writerow([f"{ts:.3f}", gpu, f"{_gr:.3f}", f"{sm:.3f}", f"{_occ:.3f}", f"{_pt:.3f}", f"{dram:.3f}"])
