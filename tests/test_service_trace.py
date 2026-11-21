"""Offline checks for the matched-workload service capture."""
import asyncio
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from simverse.execute.request_trace import summarize_request_trace
from simverse.execute.service_trace import (METHODS, collect_service_traces,
                                             speculative_config)
from simverse.execute.service_trace_vllm import capture_requests

FIXTURE = Path(__file__).with_name("fixtures") / "r07b_r09_request_trace.json"


@pytest.fixture
def offline_capture(tmp_path):
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    plan = {
        "schema_version": "1.0", "run_id": "r07b-r09-service-fixture", "tp": 1,
        "model_revision": "a" * 40, "tokenizer_revision": "a" * 40, "draft_revision": "b" * 40,
        "runtime_identity": "offline fixtures", "topology": "single GPU fixture",
        "minimum_device_memory_bytes": 1, "enable_expert_parallel": False,
        "gpu_memory_utilization": 0.9, "dflash_k": 15, "mtp_k": 1,
        "max_model_len": 128, "max_num_seqs": 8, "arrival_tolerance_ns": 0,
        "window": {k: fixture["window"][k] for k in ("start_ns", "end_ns", "drain_end_ns")},
        "slo": fixture["slo"],
        "sampling": {"max_tokens": 8, "temperature": 0, "seed": 3, "ignore_eos": False},
        "requests": [
            {"request_id": r["request_id"], "sample_id": r["sample_id"],
             "prompt_token_ids": list(range(r["input_tokens"])), "planned_arrival_ns": r["arrival_ns"]}
            for r in fixture["requests"]
        ],
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    plan_hash = "sha256:" + hashlib.sha256(plan_path.read_bytes()).hexdigest()
    replay = tmp_path / "raw"
    for method in METHODS:
        raw = {
            "method": method, "source_kind": "synthetic", "clock_id": "offline_server",
            "plan_sha256": plan_hash, "runtime": {"device": "offline"},
            "speculative_config": speculative_config(method, plan), "requests": [],
        }
        for req in fixture["requests"]:
            row = {key: req[key] for key in ("request_id", "arrival_ns", "status", "terminal_ns", "end_reason")}
            row["prompt_token_ids"] = list(range(req["input_tokens"]))
            row["chunks"] = [
                {"time_ns": chunk["time_ns"], "token_ids": [1000 + i for i in chunk["token_indices"]]}
                for chunk in req["emissions"]
            ]
            if req["request_id"] == "A":
                row["phase_spans"] = {"prefill": {"start_ns": 2_000_000, "end_ns": 8_000_000}}
                if method == "dflash_v1":
                    row["native_speculation"] = {
                        "per_step_drafted": [3], "per_step_verified": [3], "per_step_accepted": [2],
                    }
                elif method == "mtp":
                    row["native_speculation"] = {
                        "per_step_drafted": [1, 1], "per_step_verified": [1, 1], "per_step_accepted": [0, 1],
                    }
            raw["requests"].append(row)
        folder = replay / method
        folder.mkdir(parents=True)
        (folder / "server.json").write_text(json.dumps(raw), encoding="utf-8")
    return plan_path, replay


def test_cli_replays_three_modes_and_uses_trace_summary(offline_capture, tmp_path):
    plan, replay = offline_capture
    out = tmp_path / "report"
    proc = subprocess.run(
        [sys.executable, "-m", "simverse.cli", "collect-service-trace", str(plan), str(out),
         "--replay", str(replay)],
        cwd=FIXTURE.parents[2], capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    assert result["paired"] is True
    assert result["output_mismatches"] == {"dflash_v1": [], "mtp": []}
    assert result["modes"]["baseline"]["native_acceptance_histogram"] is None
    assert result["modes"]["dflash_v1"]["native_acceptance_histogram"] == {"2": 1}
    assert result["modes"]["mtp"]["native_acceptance_histogram"] == {"0": 1, "1": 1}
    for method in METHODS:
        trace = out / method / "trace.json"
        summary = summarize_request_trace(trace)
        assert summary == json.loads((out / method / "summary.json").read_text(encoding="utf-8"))
        assert summary["manifest"]["source_kind"] == "synthetic"
        assert summary["manifest"]["paired_plan_sha256"] == result["plan_sha256"]
        assert summary["window"]["time_basis"] == "server_emitted"
        assert summary["completed_token_per_s"] == 100
        assert summary["event_token_per_s"] == 175
        assert summary["ttft"]["p50_ms"] == 4
        assert summary["ttft"]["p95_ms"] == 6
        assert summary["goodput"] == {"requests_per_s": 25, "token_per_s": 25}
        assert summary["requests"]["A"]["itl_ms"] == [12, 0, 30]
        assert (out / method / "server.json").read_bytes() == (replay / method / "server.json").read_bytes()


@pytest.mark.parametrize(("field", "value", "message"), [
    ("plan_sha256", "different", "计划"),
    ("method", "baseline", "方法"),
    ("speculative_config", None, "投机配置"),
])
def test_replay_rejects_mismatched_plan_or_mode(offline_capture, tmp_path, field, value, message):
    plan, replay = offline_capture
    path = replay / "mtp" / "server.json"
    server = json.loads(path.read_text(encoding="utf-8"))
    server[field] = value
    path.write_text(json.dumps(server), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        collect_service_traces(plan, tmp_path / "out", replay)


def test_comparison_preserves_arrival_and_output_mismatches(offline_capture, tmp_path):
    plan, replay = offline_capture
    path = replay / "mtp" / "server.json"
    server = json.loads(path.read_text(encoding="utf-8"))
    server["requests"][1]["arrival_ns"] += 1
    server["requests"][1]["chunks"][0]["token_ids"] = [2026]
    path.write_text(json.dumps(server), encoding="utf-8")
    result = collect_service_traces(plan, tmp_path / "out", replay)
    assert result["paired"] is False
    assert result["modes"]["mtp"]["max_arrival_drift_ns"] == 1
    assert result["output_mismatches"]["mtp"] == ["B"]


def test_replay_rejects_invalid_native_acceptance(offline_capture, tmp_path):
    plan, replay = offline_capture
    path = replay / "dflash_v1" / "server.json"
    server = json.loads(path.read_text(encoding="utf-8"))
    server["requests"][0]["native_speculation"]["per_step_accepted"] = [4]
    path.write_text(json.dumps(server), encoding="utf-8")
    with pytest.raises(ValueError, match="投机统计"):
        collect_service_traces(plan, tmp_path / "out", replay)


def test_service_emitter_records_delta_groups_and_native_timing():
    epoch = 1_000_000_000
    now = iter([epoch, epoch + 1_000_000, epoch + 5_000_000, epoch + 9_000_000]).__next__
    request = {"request_id": "A", "prompt_token_ids": [1, 2], "planned_arrival_ns": 0}
    plan = {"window": {"drain_end_ns": 20_000_000}, "requests": [request]}
    native_stats = SimpleNamespace(queued_ts=1.0015, scheduled_ts=1.002, first_token_ts=1.004,
                                   last_token_ts=1.008, num_preemptions=0, num_generation_tokens=3)
    native_spec = SimpleNamespace(to_dict=lambda: {"per_step_accepted": [1]})

    class Engine:
        async def generate(self, prompt, params, request_id):
            assert prompt["prompt_token_ids"] == [1, 2] and request_id == "A"
            first = SimpleNamespace(token_ids=[10], finish_reason=None,
                                    stop_reason=None, spec_decode_metrics=None)
            yield SimpleNamespace(metrics=native_stats, outputs=[first], finished=False, prompt_token_ids=[1, 2])
            last = SimpleNamespace(token_ids=[11, 12], finish_reason="length",
                                   stop_reason=None, spec_decode_metrics=native_spec)
            yield SimpleNamespace(metrics=native_stats, outputs=[last], finished=True, prompt_token_ids=[1, 2])

    [row] = asyncio.run(capture_requests(Engine(), object(), plan, epoch, now))
    assert row["arrival_ns"] == 1_000_000
    assert row["chunks"] == [{"time_ns": 5_000_000, "token_ids": [10]},
                             {"time_ns": 9_000_000, "token_ids": [11, 12]}]
    assert row["terminal_ns"] == 9_000_000
    assert row["status"] == "completed"
    assert row["phase_spans"] == {
        "prefill": {"start_ns": 2_000_000, "end_ns": 4_000_000},
        "decode": {"start_ns": 4_000_000, "end_ns": 8_000_000},
    }
    assert row["native_speculation"] == {"per_step_accepted": [1]}
