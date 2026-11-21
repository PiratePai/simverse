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
from simverse.execute.service_trace_cuda import _RankCapture
from simverse.execute import service_trace_vllm
from simverse.execute.service_trace_vllm import capture_requests

FIXTURE = Path(__file__).with_name("fixtures") / "r07b_r09_request_trace.json"
ORIGIN = 1_000_000_000


class FakeCuda:
    def __init__(self, rank):
        self.ms = 0
        self.backend = "native"
        self.stream = SimpleNamespace(cuda_stream=7)
        start = 100 * (rank + 1)
        self.memory = {"allocated": start, "reserved": 2 * start}
        self.peak = None
        self.calls = []

    def get_allocator_backend(self):
        return self.backend

    def synchronize(self, device):
        self.calls.append(("sync", device.index))

    def reset_peak_memory_stats(self, device):
        self.calls.append(("reset", device.index))
        self.peak = {name: value + 50 for name, value in self.memory.items()}

    def memory_allocated(self, device):
        return self.memory["allocated"]

    def memory_reserved(self, device):
        return self.memory["reserved"]

    def max_memory_allocated(self, device):
        return self.peak["allocated"]

    def max_memory_reserved(self, device):
        return self.peak["reserved"]

    def current_stream(self, device):
        return self.stream

    def Event(self, *, enable_timing):
        assert enable_timing
        event = SimpleNamespace()

        def record(stream):
            assert stream is self.stream
            event.ms = self.ms

        event.record = record
        event.elapsed_time = lambda end: end.ms - event.ms
        return event


def _capture_fixture_rank(rank, speculative, backend="native"):
    cuda = FakeCuda(rank)
    cuda.backend = backend
    host = [ORIGIN]
    runner = SimpleNamespace(execute_model_state=None)

    def work(ms):
        cuda.ms += ms * (rank + 1)
        host[0] += 100_000

    runner.kv_connector = SimpleNamespace(pre_forward=lambda *_: None, finish_forward=lambda: None)
    runner.speculator = SimpleNamespace(propose=lambda *, ms, **_: work(ms))
    runner.postprocess_sampled = work

    def execute(scheduler, intermediate_tensors=None, dummy_run=False):
        runner.kv_connector.pre_forward(scheduler)
        work(scheduler.verify_ms)
        runner.kv_connector.finish_forward()
        runner.execute_model_state = SimpleNamespace(num_spec_tokens_to_schedule=scheduler.k)

    def sample(scheduler):
        runner.postprocess_sampled(scheduler.recovery_ms)
        runner.speculator.propose(ms=scheduler.draft_ms, num_speculative_tokens=scheduler.k)
        runner.execute_model_state = None

    runner.execute_model, runner.sample_tokens = execute, sample
    worker = SimpleNamespace(rank=rank, device=SimpleNamespace(index=rank), model_runner=runner,
                             requested_memory=900 - rank * 100, available_kv_cache_memory_bytes=500)
    capture = _RankCapture(worker, cuda, speculative, lambda: host[0])
    host[0] += 20_000_000
    for ids, spec, k, timings in ((["A-native"], {}, 1, (1, 1, 2)),
                                  (["A-native", "B-native"], {"A-native": [51]}, 1, (3, 1, 4)),
                                  (["B-native"], {"B-native": [52]}, 0, (5, 2, 0))):
        verify, recovery, draft = timings
        scheduler = SimpleNamespace(num_scheduled_tokens=dict.fromkeys(ids, 1), total_num_scheduled_tokens=len(ids),
                                    scheduled_spec_decode_tokens=spec, k=k, verify_ms=verify,
                                    recovery_ms=recovery, draft_ms=draft)
        runner.execute_model(scheduler)
        runner.sample_tokens(scheduler)
    cuda.memory["allocated"] -= 10
    host[0] += 80_000_000
    report = capture.finish()
    assert runner.execute_model is execute and runner.sample_tokens is sample
    expected_calls = [("sync", rank)] + ([("reset", rank)] if backend == "native" else []) + [("sync", rank)]
    assert cuda.calls == expected_calls
    return report


@pytest.fixture
def offline_capture(tmp_path):
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    plan = {
        "schema_version": "1.0", "run_id": "r07b-r09-service-fixture", "tp": 2,
        "model_revision": "a" * 40, "tokenizer_revision": "a" * 40, "draft_revision": "b" * 40,
        "runtime_identity": "offline fixtures", "topology": "two GPU fixture",
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
            "clock_origin_ns": ORIGIN,
            "rank_observations": [_capture_fixture_rank(rank, method != "baseline") for rank in (1, 0)],
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
        runtime = summary["runtime_observations"]
        assert runtime["complete"] is result["modes"][method]["runtime_observations_complete"] is True
        assert runtime["whole_device_peak"]["value"] is None
        first, second = runtime["ranks"]
        assert [rank["rank"] for rank in (first, second)] == [0, 1]
        assert first["memory"]["allocated"] == {"start_bytes": 100, "end_bytes": 90, "peak_bytes": 150}
        assert second["memory"]["reserved"]["peak_bytes"] == 450
        assert second["requested_memory_bytes"]["value"] == 800
        if method == "baseline":
            assert runtime["phase_applicable"] is False
            assert first["phases"] == {}
        else:
            assert runtime["phase_boundaries"]["recovery"]["boundary"] == "postprocess_sampled"
            assert runtime["slowest_rank_by_phase_p95"]["verify"] == {"rank": 1, "p95_ms": 10}
            assert first["phases"]["draft"]["observed_p50_ms"] == 2
            assert first["phases"]["verify"]["observed_p95_ms"] == 5
            assert first["phases"]["recovery"]["observed_p50_ms"] == 1
            assert second["phases"]["verify"]["observed_p50_ms"] == 6
            assert first["observed_stream_busy_ms"] == {"7": 17}
            assert second["observed_stream_busy_ms"] == {"7": 34}


def _edit_capture(replay, method, edit):
    path = replay / method / "server.json"
    server = json.loads(path.read_text(encoding="utf-8"))
    edit(server)
    path.write_text(json.dumps(server), encoding="utf-8")


def test_runtime_replay_reports_gaps_and_uses_stream_union(offline_capture, tmp_path):
    plan, replay = offline_capture

    def edit(server):
        _, rank = server["rank_observations"]
        server["rank_observations"] = [rank]
        del rank["memory"]["reserved"]
        del rank["steps"][1]["phases"]["recovery"]
        # Turn the adjacent [0, 2) and [2, 5) markers into overlapping intervals.
        rank["steps"][1]["phases"]["verify"]["start_ns"] = 1_000_000

    _edit_capture(replay, "dflash_v1", edit)
    _edit_capture(replay, "baseline", lambda raw: raw.pop("rank_observations"))
    def unsupported_allocator(server):
        server["rank_observations"][0] = _capture_fixture_rank(1, True, backend="cudaMallocAsync")

    _edit_capture(replay, "mtp", unsupported_allocator)
    out = tmp_path / "gaps"
    collect_service_traces(plan, out, replay)
    result = summarize_request_trace(out / "dflash_v1" / "trace.json")
    runtime = result["runtime_observations"]
    assert result["completed_token_per_s"] == 100
    assert runtime["complete"] is False
    assert runtime["slowest_rank_by_phase_p95"]["verify"]["rank"] is None
    first, second = runtime["ranks"]
    assert first["memory"]["reserved"]["peak_bytes"] is None
    assert first["memory"]["reserved"]["reason"]
    assert first["phases"]["recovery"] == {
        "sample_count": 1, "expected_count": 2, "missing_steps": [2],
        "observed_p50_ms": 2, "observed_p95_ms": 2, "reason": "缺少轮内阶段时间",
    }
    assert first["observed_stream_busy_ms"] == {"7": 16}
    assert second["phases"]["verify"]["missing_steps"] == [2, 3]
    assert second["phases"]["verify"]["observed_p95_ms"] is None
    assert second["memory"]["allocated"]["peak_bytes"] is None
    assert second["source_row"] is None
    baseline = summarize_request_trace(out / "baseline" / "trace.json")["runtime_observations"]
    assert baseline["complete"] is False
    assert baseline["ranks"][0]["window"] is None
    mtp = summarize_request_trace(out / "mtp" / "trace.json")["runtime_observations"]
    assert mtp["complete"] is False
    assert mtp["ranks"][1]["memory"]["allocated"]["reason"] == "分配器不支持此计数口径"
    assert mtp["ranks"][1]["phases"]["verify"]["observed_p50_ms"] == 6


@pytest.mark.parametrize(("field", "value", "message"), [
    (("rank",), 0, "重复"),
    (("window", "end_ns"), ORIGIN + 20_000_000, "未覆盖"),
    (("memory", "allocated", "peak_bytes"), 1, "低于窗口"),
    (("steps", 1, "verify_request_ids"), ["B-native"], "跨 rank"),
    (("steps", 1, "phases", "verify", "host_end_ns"), ORIGIN + 900_000_000, "主机窗口"),
    (("steps", 1, "draft_tokens_to_schedule"), 0, "未调度"),
])
def test_runtime_replay_rejects_inconsistent_evidence(offline_capture, tmp_path, field, value, message):
    plan, replay = offline_capture

    def edit(server):
        obj = server["rank_observations"][0]
        for key in field[:-1]:
            obj = obj[key]
        obj[field[-1]] = value

    _edit_capture(replay, "dflash_v1", edit)
    with pytest.raises(ValueError, match=message):
        collect_service_traces(plan, tmp_path / "invalid", replay)


def test_service_window_arms_after_warmup_and_finishes_after_core_drain(monkeypatch):
    order = []

    async def pause(mode, clear_cache):
        order.append((mode, clear_cache))

    async def rpc(method, timeout, args=()):
        order.append((method, args))
        return ["rank-data"]

    async def resume():
        order.append("resume")

    async def requests(engine, params, plan, origin):
        order.append("requests")
        assert origin > 0
        return ["request-data"]

    engine = SimpleNamespace(pause_generation=pause, collective_rpc=rpc, resume_generation=resume)
    monkeypatch.setattr(service_trace_vllm, "capture_requests", requests)
    _, rows, ranks = asyncio.run(service_trace_vllm.capture_service_window(engine, None, {}, True))
    assert (rows, ranks) == (["request-data"], ["rank-data"])
    assert order == [("wait", False), ("simverse_begin_trace", (True,)), "resume", "requests",
                     ("wait", False), ("simverse_end_trace", ())]


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
