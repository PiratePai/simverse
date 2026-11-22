"""Offline checks for the matched-workload service capture."""
import asyncio
import hashlib
import json
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from simverse.execute.request_trace import summarize_request_trace
from simverse.execute.service_trace import (METHODS, collect_service_traces,
                                             speculative_config)
from simverse.execute.service_trace_cuda import _RankCapture
from simverse.execute import service_trace_nvml, service_trace_scheduler, service_trace_vllm
from simverse.execute.service_trace_scheduler import _SchedulerCapture, _instrument_rollback
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
    runner.rejection_sampler = work
    runner.postprocess_sampled = work

    def execute(scheduler, intermediate_tensors=None, dummy_run=False):
        runner.kv_connector.pre_forward(scheduler)
        work(scheduler.verify_ms)
        runner.kv_connector.finish_forward()
        runner.execute_model_state = SimpleNamespace(num_spec_tokens_to_schedule=scheduler.k)

    def sample(scheduler):
        if scheduler.scheduled_spec_decode_tokens:
            runner.rejection_sampler(scheduler.rejection_ms)
        runner.postprocess_sampled(scheduler.recovery_ms)
        runner.speculator.propose(ms=scheduler.draft_ms, num_speculative_tokens=scheduler.k)
        runner.execute_model_state = None

    runner.execute_model, runner.sample_tokens = execute, sample
    worker = SimpleNamespace(rank=rank, device=SimpleNamespace(index=rank), model_runner=runner,
                             requested_memory=900 - rank * 100, available_kv_cache_memory_bytes=500)
    capture = _RankCapture(worker, cuda, speculative, lambda: host[0])
    host[0] += 20_000_000
    for ids, spec, k, timings in ((["A-native"], {}, 1, (1, 0, 1, 2)),
                                  (["A-native", "B-native"], {"A-native": [51]}, 1, (3, 2, 1, 4)),
                                  (["B-native"], {"B-native": [52]}, 0, (5, 3, 2, 0))):
        verify, rejection, recovery, draft = timings
        scheduler = SimpleNamespace(num_scheduled_tokens=dict.fromkeys(ids, 1), total_num_scheduled_tokens=len(ids),
                                    scheduled_spec_decode_tokens=spec, k=k, verify_ms=verify,
                                    rejection_ms=rejection, recovery_ms=recovery, draft_ms=draft)
        runner.execute_model(scheduler)
        runner.sample_tokens(scheduler)
    cuda.memory["allocated"] -= 10
    host[0] += 80_000_000
    report = capture.finish()
    assert runner.execute_model is execute and runner.sample_tokens is sample
    expected_calls = [("sync", rank)] + ([("reset", rank)] if backend == "native" else []) + [("sync", rank)]
    assert cuda.calls == expected_calls
    report["device_memory"] = {
        "device_uuid": f"GPU-offline-{rank}", "sample_interval_ns": 20_000_000, "reason": None,
        "samples": [{"time_ns": ORIGIN + delta_ms * 1_000_000, "used_bytes": used + rank * 100,
                     "total_bytes": 2000} for delta_ms, used in ((0, 500), (20, 780), (45, 630), (65, 550))],
    }
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
        if method != "baseline":
            raw["scheduler_observations"] = {
                "window": {"start_ns": ORIGIN, "end_ns": ORIGIN + 60_000_000},
                "step_count": 3, "rollback_count": 2, "reason": None,
                "rollbacks": [{"step": step, "request_id": rid, "rejected_tokens": 1,
                               "start_ns": ORIGIN + start, "end_ns": ORIGIN + end}
                              for step, rid, start, end in ((2, "A-native", 34_000_000, 34_002_000),
                                                            (3, "B-native", 44_000_000, 44_005_000))],
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
    assert result["comparison_eligibility"]["pairs"]["dflash_v1"]["checks"]["unattributed_limit"] == "计划未登记未归属时长门槛"
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
        first, second = runtime["ranks"]
        assert [rank["rank"] for rank in (first, second)] == [0, 1]
        assert first["memory"]["allocated"] == {"start_bytes": 100, "end_bytes": 90, "peak_bytes": 150}
        assert second["memory"]["reserved"]["peak_bytes"] == 450
        assert first["device_memory"]["observed_peak_bytes"] == 780
        assert second["device_memory"]["observed_peak_bytes"] == 880
        assert second["device_memory"]["max_sample_gap_ms"] == 25
        assert second["requested_memory_bytes"]["value"] == 800
        if method == "baseline":
            assert runtime["phase_applicable"] is False
            assert runtime["scheduler_rollback"] is None
            assert first["phases"] == {}
        else:
            scheduler = runtime["scheduler_rollback"]
            assert scheduler["observed_p50_ms"] == 0.002
            assert scheduler["observed_p95_ms"] == 0.005
            assert scheduler["observed_busy_ms"] == 0.007
            assert runtime["phase_boundaries"]["recovery"]["boundary"] == "postprocess_sampled"
            assert runtime["slowest_rank_by_phase_p95"]["verify"] == {"rank": 1, "p95_ms": 10}
            assert first["phases"]["draft"]["observed_p50_ms"] == 2
            assert first["phases"]["verify"]["observed_p95_ms"] == 5
            assert first["phases"]["rejection_sampling"]["observed_p50_ms"] == 2
            assert first["phases"]["rejection_sampling"]["observed_p95_ms"] == 3
            assert first["phases"]["recovery"]["observed_p50_ms"] == 1
            assert second["phases"]["verify"]["observed_p50_ms"] == 6
            assert first["observed_stream_busy_ms"] == {"7": 22}
            assert second["observed_stream_busy_ms"] == {"7": 44}


def _edit_capture(replay, method, edit):
    path = replay / method / "server.json"
    server = json.loads(path.read_text(encoding="utf-8"))
    edit(server)
    path.write_text(json.dumps(server), encoding="utf-8")


@pytest.fixture
def round_commit_capture(offline_capture):
    plan, replay = offline_capture

    def round_row(rid, step, start, counts, ids, end=None):
        proposed, verified, accepted = counts
        row = {"step": step, "backend_request_id": f"{rid}-native", "token_start": start,
               "proposed": proposed, "verified": verified, "accepted": accepted,
               "target_offset": accepted, "backend_token_ids": ids}
        if end is not None:
            row["backend_finish_reason"] = end
        return row

    def edit(raw):
        a, b, *_ = raw["requests"]
        a["chunks"][1:] = [{"time_ns": a["terminal_ns"], "token_ids": [1001, 1001, 1003, 1004]}]
        b["chunks"][1:] = [{"time_ns": b["terminal_ns"], "token_ids": [1001, 1002]}]
        b["end_reason"] = "length"
        for request in raw["requests"]:
            offset = 0
            for group in request["chunks"]:
                group["token_start"] = offset
                offset += len(group["token_ids"])
            request["emitted_token_count"] = offset
        if raw["method"] == "baseline":
            return
        scheduler = raw["scheduler_observations"]
        for rank in raw["rank_observations"]:
            for step in rank["steps"][1:]:
                step["batch_request_ids"] = ["A-native", "B-native"]
                step["verify_request_ids"] = ["A-native", "B-native"]
        if raw["method"] == "dflash_v1":
            rounds = [round_row("B", 2, 1, (3, 3, 0), [1001]),
                      round_row("A", 3, 1, (3, 3, 3), [1001, 1001, 1003, 1004], "stop"),
                      round_row("B", 3, 2, (2, 2, 1), [1002], "length")]
            scheduler["rollbacks"][0]["request_id"] = "B-native"
        else:
            rounds = [round_row("A", 2, 2, (1, 1, 0), [1001]),
                      round_row("A", 3, 3, (1, 1, 1), [1003, 1004], "stop")]
            scheduler["rollbacks"].pop()
            scheduler["rollback_count"] = 1
        scheduler["rounds"] = rounds
        for request in raw["requests"]:
            backend_id = f"{request['request_id']}-native"
            request["backend_request_id"] = backend_id
            observed = [row for row in rounds if row["backend_request_id"] == backend_id]
            request["native_speculation"] = {"num_spec_steps": len(observed), **{
                f"per_step_{key}": [row[field] for row in observed]
                for key, field in (("drafted", "proposed"), ("verified", "verified"), ("accepted", "accepted"))
            }}

    for method in METHODS:
        _edit_capture(replay, method, edit)
    return plan, replay


def test_round_commits_link_merge_truncation_and_exact_bonus(round_commit_capture, tmp_path):
    plan, replay = round_commit_capture
    out = tmp_path / "round-report"
    report = collect_service_traces(plan, out, replay)
    assert report["output_mismatches"] == {"dflash_v1": [], "mtp": []}
    for mode in METHODS:
        summary = summarize_request_trace(out / mode / "trace.json")
        assert summary["completed_token_per_s"] == 100
        if mode == "baseline":
            assert "round_commits" not in summary
            continue
        assert report["modes"][mode]["round_commit_reason"] is None
        linked = summary["round_commits"]
        assert linked["complete"] and linked["unassigned_backend_round_count"] == 0
        rounds = {row["round_id"]: row for row in linked["rounds"]}
        assert linked["requests"]["C"]["expected_round_count"] == 0
        if mode == "dflash_v1":
            assert linked["native_acceptance_histogram"] == {"0": 1, "1": 1, "3": 1}
            assert {tuple(r[key] for key in ("proposed", "verified", "accepted", "committed"))
                    for r in linked["joint_distribution"]} == {(3, 3, 0, 1), (3, 3, 3, 4), (2, 2, 1, 1)}
            assert rounds["B:2"]["emission_group_indices"] == rounds["B:3"]["emission_group_indices"] == [1]
            assert rounds["B:2"]["target_bonus"] == {"kind": "replacement", "token_index": 1, "token_id": 1001}
            assert rounds["B:3"]["token_indices"] == [2]
            assert rounds["B:3"]["backend_finish_reason"] == "length"
            assert rounds["B:3"]["target_bonus_status"] == "not_emitted"
            assert rounds["B:3"]["target_bonus"] is None
            assert rounds["A:3"]["target_bonus"] == {"kind": "bonus", "token_index": 4, "token_id": 1004}
        else:
            assert rounds["A:2"]["token_indices"] == [2]
            assert rounds["A:3"]["token_indices"] == [3, 4]
            assert rounds["A:2"]["emission_group_indices"] == rounds["A:3"]["emission_group_indices"] == [1]
            assert rounds["A:2"]["target_bonus"]["token_index"] == 2
            assert rounds["A:3"]["target_bonus"]["token_index"] == 4


def test_round_commits_report_gaps_without_guessing_from_repeated_tokens(round_commit_capture, tmp_path):
    plan, replay = round_commit_capture

    def partial(raw):
        rounds = raw["scheduler_observations"]["rounds"]
        del rounds[0]["backend_token_ids"]
        del rounds[1]["target_offset"]
        # The scheduler emitted a target token that the service stopped before delivering.
        rounds[2]["backend_token_ids"].append(1003)

    def unbound(raw):
        del raw["requests"][0]["backend_request_id"]

    _edit_capture(replay, "dflash_v1", partial)
    _edit_capture(replay, "mtp", unbound)
    out = tmp_path / "round-gaps"
    collect_service_traces(plan, out, replay)
    partial = summarize_request_trace(out / "dflash_v1" / "trace.json")["round_commits"]
    assert not partial["complete"]
    assert partial["requests"]["B"]["linked_round_count"] == 1
    rounds = {row["round_id"]: row for row in partial["rounds"]}
    assert rounds["B:2"]["committed"] is None and rounds["B:2"]["token_indices"] is None
    assert rounds["A:3"]["committed"] == 4 and rounds["A:3"]["target_bonus_status"] == "unknown"
    assert rounds["B:3"]["backend_committed"] == 2 and rounds["B:3"]["committed"] == 1
    assert rounds["B:3"]["target_bonus_status"] == "not_emitted"
    unbound = summarize_request_trace(out / "mtp" / "trace.json")["round_commits"]
    assert unbound["unassigned_backend_round_count"] == 2
    assert unbound["requests"]["A"]["reason"] == "缺少后端请求 ID"
    assert unbound["rounds"] == []


@pytest.mark.parametrize(("key", "value", "message"), [
    ("token_start", 1, "提交序号重复"),
    ("backend_token_ids", [4444], "与服务端输出序号上的 token 不符"),
    ("target_offset", 0, "与接受前缀不符"),
    ("verified", 1, "与原生投机统计不符"),
])
def test_round_commits_reject_inconsistent_source(round_commit_capture, tmp_path, key, value, message):
    plan, replay = round_commit_capture

    def edit(raw):
        raw["scheduler_observations"]["rounds"][2][key] = value

    _edit_capture(replay, "dflash_v1", edit)
    with pytest.raises(ValueError, match=message):
        collect_service_traces(plan, tmp_path / "invalid-round", replay)


def test_missing_backend_round_does_not_shift_later_indices(round_commit_capture, tmp_path):
    plan, replay = round_commit_capture
    _edit_capture(replay, "mtp", lambda raw: raw["scheduler_observations"]["rounds"].pop(0))
    out = tmp_path / "missing-round"
    collect_service_traces(plan, out, replay)
    summary = summarize_request_trace(out / "mtp" / "trace.json")["round_commits"]
    assert summary["requests"]["A"] == {
        "expected_round_count": 2, "observed_round_count": 1, "linked_round_count": 1,
        "complete": False, "reason": "缺少后端投机轮",
    }
    [later] = summary["rounds"]
    assert later["round_id"] == "A:3" and later["token_indices"] == [3, 4]
    assert later["target_bonus"]["token_index"] == 4


@pytest.mark.parametrize(("index", "message"), [(0, "连续输出序号"), (1, "emitted_token_count")])
def test_replay_rejects_dropped_service_groups(round_commit_capture, tmp_path, index, message):
    plan, replay = round_commit_capture
    _edit_capture(replay, "mtp", lambda raw: raw["requests"][0]["chunks"].pop(index))
    with pytest.raises(ValueError, match=message):
        collect_service_traces(plan, tmp_path / "missing-group", replay)


def _rewrite_plan(plan, replay, **changes):
    updated = json.loads(plan.read_text(encoding="utf-8"))
    updated.update(changes)
    plan.write_text(json.dumps(updated), encoding="utf-8")
    plan_sha = "sha256:" + hashlib.sha256(plan.read_bytes()).hexdigest()
    for method in METHODS:
        _edit_capture(replay, method, lambda raw: raw.update(plan_sha256=plan_sha))


@pytest.fixture
def matched_attribution(round_commit_capture):
    plan, replay = round_commit_capture
    _rewrite_plan(plan, replay, max_unattributed_ns=15_000_000)

    def span(span_id, stage, start, end, resource, rank=None):
        row = {"span_id": span_id, "request_id": "A", "stage": stage, "role": "target",
               "resource": resource, "timing_boundary": "synthetic_aligned_resource",
               "start_ns": start * 1_000_000, "end_ns": end * 1_000_000}
        if rank is not None:
            row["rank"] = rank
        return row

    def edit(raw):
        for request in raw["requests"]:
            chunks = request["chunks"]
            if chunks:
                request["phase_spans"] = {"prefill": {"start_ns": request["arrival_ns"],
                                                      "end_ns": chunks[0]["time_ns"]}}
                if request["emitted_token_count"] > 1:
                    request["phase_spans"]["decode"] = {"start_ns": chunks[0]["time_ns"],
                                                        "end_ns": chunks[-1]["time_ns"]}
        spans = [span("lane-0-prefill", "prefill", 8, 20, "stream-0", 0),
                 span("lane-0-decode", "decode", 18, 25, "stream-0", 0),
                 span("lane-1-decode", "decode", 40, 55, "stream-0", 1)]
        if raw["method"] != "baseline":
            for index, stage in enumerate(("draft", "verify", "recovery", "rejection_sampling"), 10):
                spans.append(span(stage, stage, index, index + 1, "stream-0", 0))
            spans.append(span("cpu-rollback", "scheduler_rollback", 43, 44, "scheduler"))
        raw["stage_spans"] = spans
        ledger = {}
        for row in spans:
            ledger.setdefault(row["stage"], []).append(row["span_id"])
        raw["stage_attribution"] = ledger

    for method in METHODS:
        _edit_capture(replay, method, edit)
    return plan, replay


def test_matched_reconciliation_unions_overlaps_and_keeps_eligibility_scoped(matched_attribution, tmp_path):
    plan, replay = matched_attribution
    out = tmp_path / "reconciliation"
    result = collect_service_traces(plan, out, replay)
    assert result["paired"] is True
    eligibility = result["comparison_eligibility"]
    assert eligibility["eligible"] is True and eligibility["hardware_eligible"] is False
    for method in METHODS:
        summary = summarize_request_trace(out / method / "trace.json")
        mode = result["modes"][method]
        reconciled = summary["stage_reconciliation"]
        assert mode["stage_reconciliation"] == reconciled
        assert mode["metrics"]["completed_token_per_s"] == 100
        assert reconciled["phase_coverage_complete"]
        assert reconciled["request_phases"]["prefill"]["expected_count"] == 4
        assert reconciled["request_phases"]["decode"]["expected_count"] == 2
        residual = reconciled["attribution"]
        assert (residual["wall_ns"], residual["observed_aligned_ns"], residual["unattributed_ns"]) == (
            40_000_000, 25_000_000, 15_000_000,
        )
        assert residual["coverage"]["ratio"] == 1
        assert reconciled["observed_request_wall_ns"] == 40_000_000
        lanes = {row["rank"]: row for row in residual["resources"] if row["rank"] is not None}
        assert lanes[0]["observed_busy_ns"] == 15_000_000
        assert lanes[1]["observed_busy_ns"] == 10_000_000
        assert lanes[1]["window"] == {"start_ns": 40_000_000, "end_ns": 50_000_000}
        if method != "baseline":
            assert reconciled["runtime_phases"]["verify"]["expected_count"] == 4
            pair = eligibility["pairs"][method]
            assert pair["service_metrics_comparable"] and not pair["hardware_eligible"]
            assert pair["candidate_minus_baseline"]["completed_token_per_s"] == 0
            assert pair["hardware_reason"] == "两路须均为硬件采集记录"


def test_cross_clock_attribution_keeps_local_busy_but_not_residual(matched_attribution, tmp_path):
    plan, replay = matched_attribution
    _edit_capture(replay, "mtp", lambda raw: raw["stage_spans"][2].update(clock_id="cuda_rank1_stream0"))
    result = collect_service_traces(plan, tmp_path / "cross-clock", replay)
    residual = result["modes"]["mtp"]["stage_reconciliation"]["attribution"]
    assert residual["observed_aligned_ns"] == 16_000_000
    assert residual["unattributed_ns"] is None and residual["unaligned_span_ids"] == ["lane-1-decode"]
    local = next(row for row in residual["resources"] if row["scope"] == "local_clock")
    assert local["observed_busy_ns"] == 15_000_000
    pairs = result["comparison_eligibility"]["pairs"]
    assert pairs["dflash_v1"]["eligible"]
    assert pairs["mtp"]["service_metrics_comparable"] and not pairs["mtp"]["eligible"]
    assert pairs["mtp"]["checks"]["unattributed_limit"] == "未归属时长未知"


def test_cross_mode_clock_suppresses_paired_deltas(matched_attribution, tmp_path):
    plan, replay = matched_attribution
    _edit_capture(replay, "mtp", lambda raw: raw.update(clock_id="another_observation_clock"))
    result = collect_service_traces(plan, tmp_path / "cross-mode-clock", replay)
    pair = result["comparison_eligibility"]["pairs"]["mtp"]
    assert not pair["service_metrics_comparable"] and pair["max_arrival_delta_ns"] is None
    assert all(value is None for value in pair["candidate_minus_baseline"].values())
    assert pair["checks"]["window_and_clock"]


def test_missing_resource_and_runtime_phase_leave_residual_unknown(matched_attribution, tmp_path):
    plan, replay = matched_attribution
    _edit_capture(replay, "dflash_v1", lambda raw: raw["stage_spans"].pop(2))
    _edit_capture(replay, "mtp", lambda raw: raw["rank_observations"][0]["steps"][1]["phases"].pop("recovery"))
    result = collect_service_traces(plan, tmp_path / "stage-gaps", replay)
    first = result["modes"]["dflash_v1"]["stage_reconciliation"]
    assert first["phase_coverage_complete"]
    assert first["attribution"]["missing_span_ids"] == ["lane-1-decode"]
    assert first["attribution"]["unattributed_ns"] is None
    second = result["modes"]["mtp"]["stage_reconciliation"]
    assert second["runtime_phases"]["recovery"]["ratio"] == 0.75
    assert second["attribution"]["observed_aligned_ns"] == 25_000_000
    assert second["attribution"]["unattributed_ns"] is None
    assert result["comparison_eligibility"]["pairs"]["mtp"]["checks"]["stage_coverage"]


def test_eligibility_reports_residual_threshold_memory_and_round_gaps(matched_attribution, tmp_path):
    plan, replay = matched_attribution
    _rewrite_plan(plan, replay, max_unattributed_ns=14_999_999)
    _edit_capture(replay, "dflash_v1", lambda raw: raw["rank_observations"][0].pop("device_memory"))
    _edit_capture(replay, "mtp", lambda raw: raw["scheduler_observations"]["rounds"].pop(0))
    result = collect_service_traces(plan, tmp_path / "evidence-gates", replay)
    pairs = result["comparison_eligibility"]["pairs"]
    assert not result["comparison_eligibility"]["eligible"]
    assert pairs["dflash_v1"]["checks"]["memory_coverage"]
    assert pairs["mtp"]["checks"]["round_commits"]
    assert all(pair["checks"]["unattributed_limit"] == "未归属时长超过计划门槛" for pair in pairs.values())


def test_native_rounds_prevent_empty_worker_inventory_from_qualifying(matched_attribution, tmp_path):
    plan, replay = matched_attribution

    def edit(raw):
        for rank in raw["rank_observations"]:
            rank["steps"] = []

    _edit_capture(replay, "mtp", edit)
    result = collect_service_traces(plan, tmp_path / "empty-worker-rounds", replay)
    reconciliation = result["modes"]["mtp"]["stage_reconciliation"]
    verify = reconciliation["runtime_phases"]["verify"]
    assert verify["expected_count"] is None
    assert verify["reason"] == "原生投机轮缺少对应阶段的预期记录"
    assert reconciliation["attribution"]["unattributed_ns"] is None


def test_pair_checks_observed_arrival_drift_between_modes(matched_attribution, tmp_path):
    plan, replay = matched_attribution
    _rewrite_plan(plan, replay, arrival_tolerance_ns=1)

    def edit(raw, deviation):
        request = raw["requests"][1]
        request["arrival_ns"] += deviation
        request["phase_spans"]["prefill"]["start_ns"] += deviation

    _edit_capture(replay, "baseline", lambda raw: edit(raw, -1))
    _edit_capture(replay, "mtp", lambda raw: edit(raw, 1))
    result = collect_service_traces(plan, tmp_path / "pair-drift", replay)
    assert result["paired"]
    pair = result["comparison_eligibility"]["pairs"]["mtp"]
    assert pair["max_arrival_delta_ns"] == 2 and not pair["service_metrics_comparable"]
    assert pair["checks"]["arrival_alignment"]


@pytest.mark.parametrize("missing", ["baseline", "mtp"])
def test_reconciliation_retains_available_modes_when_a_record_is_missing(matched_attribution, tmp_path, missing):
    plan, replay = matched_attribution
    (replay / missing / "server.json").unlink()
    out = tmp_path / "missing-mode"
    result = collect_service_traces(plan, out, replay)
    assert result["missing_modes"] == {missing: "缺少原始 server.json"}
    assert not result["paired"] and not result["comparison_eligibility"]["eligible"]
    for method in set(METHODS) - {missing}:
        assert summarize_request_trace(out / method / "trace.json")["completed_token_per_s"] == 100
    pairs = result["comparison_eligibility"]["pairs"]
    assert pairs["mtp"]["missing_modes"] == [missing]
    assert result["output_mismatches"]["mtp"] is None
    assert pairs["dflash_v1"]["eligible"] == (missing != "baseline")


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
    assert first["observed_stream_busy_ms"] == {"7": 21}
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


def test_runtime_details_keep_missing_partial_and_zero_distinct(offline_capture, tmp_path):
    plan, replay = offline_capture

    def partial(raw):
        first, second = raw["rank_observations"]
        del first["steps"][1]["phases"]["rejection_sampling"]
        first["device_memory"]["samples"] = first["device_memory"]["samples"][1:3]
        second["device_memory"]["reason"] = "read failed"
        raw["scheduler_observations"]["rollbacks"].pop()

    def zero(raw):
        raw["scheduler_observations"].update(rollback_count=0, rollbacks=[])

    _edit_capture(replay, "dflash_v1", partial)
    _edit_capture(replay, "mtp", zero)
    out = tmp_path / "more-gaps"
    comparison = collect_service_traces(plan, out, replay)
    assert comparison["modes"]["dflash_v1"]["runtime_observations_complete"] is False
    assert comparison["modes"]["mtp"]["runtime_observations_complete"] is True
    partial = summarize_request_trace(out / "dflash_v1" / "trace.json")["runtime_observations"]
    assert partial["ranks"][1]["phases"]["rejection_sampling"]["missing_steps"] == [2]
    assert partial["ranks"][1]["device_memory"]["observed_peak_bytes"] == 880
    assert partial["ranks"][1]["device_memory"]["reason"] == "整卡显存采样未覆盖请求的观测区间"
    assert partial["ranks"][0]["device_memory"]["reason"] == "read failed"
    assert partial["ranks"][0]["memory"]["allocated"]["peak_bytes"] == 150
    assert partial["scheduler_rollback"]["observed_p95_ms"] == 0.002
    assert partial["scheduler_rollback"]["expected_count"] == 2
    assert partial["scheduler_rollback"]["complete"] is False
    zero = summarize_request_trace(out / "mtp" / "trace.json")["runtime_observations"]["scheduler_rollback"]
    assert zero["complete"] is True and zero["expected_count"] == 0
    assert zero["observed_busy_ms"] == 0 and zero["observed_p95_ms"] is None


def test_runtime_replays_legacy_worker_rows_with_explicit_missing_detail(offline_capture, tmp_path):
    plan, replay = offline_capture

    def legacy(raw):
        del raw["scheduler_observations"]
        for rank in raw["rank_observations"]:
            del rank["device_memory"]
            for step in rank["steps"]:
                step["phases"].pop("rejection_sampling", None)

    _edit_capture(replay, "dflash_v1", legacy)
    out = tmp_path / "legacy"
    collect_service_traces(plan, out, replay)
    runtime = summarize_request_trace(out / "dflash_v1" / "trace.json")["runtime_observations"]
    assert not runtime["complete"]
    assert runtime["scheduler_rollback"]["expected_count"] is None
    assert runtime["scheduler_rollback"]["observed_busy_ms"] is None
    rank = runtime["ranks"][0]
    assert rank["device_memory"]["observed_peak_bytes"] is None
    assert rank["phases"]["rejection_sampling"]["missing_steps"] == [2, 3]
    assert rank["phases"]["verify"]["observed_p95_ms"] == 5


@pytest.mark.parametrize(("path", "value", "message"), [
    (("rank_observations", 0, "device_memory", "samples", 1, "used_bytes"), 3000, "整卡显存用量"),
    (("rank_observations", 0, "device_memory", "samples", 2, "time_ns"), ORIGIN, "采样时间逆序"),
    (("scheduler_observations", "rollbacks", 0, "request_id"), "B-native", "CPU 回退批次"),
    (("scheduler_observations", "rollbacks", 0, "end_ns"), ORIGIN + 70_000_000, "CPU 回退时间"),
    (("scheduler_observations", "rollback_count"), 1, "超出声明次数"),
])
def test_independent_telemetry_rejects_invalid_samples(offline_capture, tmp_path, path, value, message):
    plan, replay = offline_capture

    def bad(raw):
        item = raw
        for key in path[:-1]:
            item = item[key]
        item[path[-1]] = value

    _edit_capture(replay, "dflash_v1", bad)
    with pytest.raises(ValueError, match=message):
        collect_service_traces(plan, tmp_path / "invalid-detail", replay)


def _scheduler_update_fixture(self, scheduler_output, model_runner_output):
    req_id = scheduler_output.request_id
    request, output_is_stale, num_rejected = model_runner_output
    num_draft_tokens = num_verified = 4
    num_accepted, num_sampled = num_verified - num_rejected, 1
    self.clock[0] += 100
    if not output_is_stale:
        if request.num_computed_tokens > 0:
            request.num_computed_tokens -= num_rejected
        if request.num_output_placeholders > 0:
            request.num_output_placeholders -= num_rejected
    self.clock[0] += 200
    output = SimpleNamespace(request_id=req_id, new_token_ids=[123], finish_reason=None)
    return {0: SimpleNamespace(outputs=[output])}


def test_scheduler_hook_times_only_native_counter_updates():
    clock = [1000]

    class Request(SimpleNamespace):
        def __setattr__(self, name, value):
            clock[0] += 10
            super().__setattr__(name, value)

    request = Request(num_computed_tokens=12, num_output_placeholders=4, num_output_tokens=1)
    capture = _SchedulerCapture(lambda: clock[0])
    scheduler = SimpleNamespace(clock=clock, _simverse_rollback=capture)
    capture.steps = 2
    update = _instrument_rollback(_scheduler_update_fixture)
    batch = SimpleNamespace(request_id="A", total_num_scheduled_tokens=2)
    start = clock[0] + 100
    update(scheduler, batch, (request, False, 2))
    assert request.num_computed_tokens == 10
    assert request.num_output_placeholders == 2
    update(scheduler, batch, (request, True, 3))
    assert request.num_computed_tokens == 10
    update(scheduler, batch, (request, False, 0))
    assert request.num_computed_tokens == 10
    assert capture.finish(None)["rollbacks"] == [
        {"step": 2, "request_id": "A", "rejected_tokens": 2, "start_ns": start, "end_ns": start + 20},
    ]
    with pytest.raises(ValueError, match="回退代码块"):
        _instrument_rollback(test_scheduler_hook_times_only_native_counter_updates)


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_scheduler_factory_drains_native_variant_through_core_utility(monkeypatch, async_scheduling):
    class Scheduler:
        update_from_output = _scheduler_update_fixture

        def __init__(self, *, vllm_config):
            self.clock = [1000]

    class AsyncScheduler(Scheduler):
        pass

    class Core:
        pass

    monkeypatch.setitem(sys.modules, "vllm.v1.core.sched.scheduler", SimpleNamespace(Scheduler=Scheduler))
    monkeypatch.setitem(sys.modules, "vllm.v1.core.sched.async_scheduler", SimpleNamespace(AsyncScheduler=AsyncScheduler))
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.core", SimpleNamespace(EngineCore=Core))
    config = SimpleNamespace(scheduler_config=SimpleNamespace(async_scheduling=async_scheduling))
    core = Core()
    core.scheduler = service_trace_scheduler.TraceScheduler(vllm_config=config)
    assert isinstance(core.scheduler, AsyncScheduler) == async_scheduling
    batch = SimpleNamespace(request_id="A", total_num_scheduled_tokens=2)
    warmup = SimpleNamespace(num_computed_tokens=6, num_output_placeholders=2, num_output_tokens=1)
    core.scheduler.update_from_output(batch, (warmup, False, 1))
    assert warmup.num_computed_tokens == 5
    core.simverse_scheduler_trace(True)
    request = SimpleNamespace(num_computed_tokens=12, num_output_placeholders=4, num_output_tokens=4)
    core.scheduler.update_from_output(batch, (request, False, 2))
    assert request.num_computed_tokens == 10
    report = core.simverse_scheduler_trace(False)
    assert report["reason"] is None and report["step_count"] == report["rollback_count"] == 1
    assert report["rollbacks"][0]["request_id"] == "A"
    assert report["rounds"] == [{
        "step": 1, "backend_request_id": "A", "token_start": 4, "proposed": 4, "verified": 4,
        "accepted": 2, "target_offset": 2, "backend_token_ids": [123],
    }]


@pytest.mark.parametrize("poll_fails", [False, True])
def test_nvml_samples_the_physical_parent_in_a_background_thread(poll_fails):
    polled = threading.Event()
    calls = []
    read_count = 0

    class NVMLError(Exception):
        value = 1

    def handle(uuid):
        calls.append(uuid)
        if uuid.startswith("GPU-"):
            raise NVMLError()
        return "mig"

    def parent(mig):
        assert mig == "mig"
        return "physical"

    def memory(device):
        nonlocal read_count
        assert device == "physical"
        read_count += 1
        if threading.current_thread().name == "simverse-nvml":
            polled.set()
            if poll_fails:
                raise NVMLError()
            return SimpleNamespace(used=880, total=2000)
        return SimpleNamespace(used=500, total=2000)

    nvml = SimpleNamespace(NVMLError=NVMLError, NVML_ERROR_NOT_FOUND=1, NVML_ERROR_INVALID_ARGUMENT=2,
                           nvmlInit=lambda: calls.append("init"), nvmlShutdown=lambda: calls.append("shutdown"),
                           nvmlDeviceGetHandleByUUID=handle, nvmlDeviceGetDeviceHandleFromMigDeviceHandle=parent,
                           nvmlDeviceGetUUID=lambda _: b"GPU-parent", nvmlDeviceGetMemoryInfo=memory)
    sampler = service_trace_nvml._DeviceMemorySampler(nvml, "a-device")
    try:
        assert polled.wait(2)
    finally:
        report = sampler.finish()
    assert calls == ["init", "GPU-a-device", "MIG-a-device", "shutdown"]
    assert report["device_uuid"] == "GPU-parent"
    assert read_count >= 3
    if poll_fails:
        assert report["reason"] and len(report["samples"]) == 2
    else:
        assert report["reason"] is None and max(row["used_bytes"] for row in report["samples"]) == 880


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

    async def utility(method, arg):
        order.append((method, arg))
        return "scheduler-data"

    async def resume():
        order.append("resume")

    async def requests(engine, params, plan, origin):
        order.append("requests")
        assert origin > 0
        engine.output_processor.add_request(SimpleNamespace(request_id="server-id", external_req_id="A"))
        return [{"request_id": "A"}]

    def add_request(request):
        assert request.request_id == "server-id"

    engine = SimpleNamespace(pause_generation=pause, collective_rpc=rpc, resume_generation=resume,
                             engine_core=SimpleNamespace(call_utility_async=utility),
                             output_processor=SimpleNamespace(add_request=add_request))
    monkeypatch.setattr(service_trace_vllm, "capture_requests", requests)
    _, rows, ranks, scheduler = asyncio.run(service_trace_vllm.capture_service_window(engine, None, {}, True))
    assert (rows, ranks, scheduler) == ([{"request_id": "A", "backend_request_id": "server-id"}],
                                       ["rank-data"], "scheduler-data")
    assert engine.output_processor.add_request is add_request
    assert order == [("wait", False), ("simverse_begin_trace", (True,)), ("simverse_scheduler_trace", True),
                     "resume", "requests", ("wait", False), ("simverse_scheduler_trace", False),
                     ("simverse_end_trace", ())]


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
    pair = result["comparison_eligibility"]["pairs"]["mtp"]
    assert pair["checks"]["output_consistency"]
    assert all(value is None for value in pair["candidate_minus_baseline"].values())


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
    assert row["chunks"] == [{"time_ns": 5_000_000, "token_start": 0, "token_ids": [10]},
                             {"time_ns": 9_000_000, "token_start": 1, "token_ids": [11, 12]}]
    assert row["emitted_token_count"] == 3
    assert row["terminal_ns"] == 9_000_000
    assert row["status"] == "completed"
    assert row["phase_spans"] == {
        "prefill": {"start_ns": 2_000_000, "end_ns": 4_000_000},
        "decode": {"start_ns": 4_000_000, "end_ns": 8_000_000},
    }
    assert row["native_speculation"] == {"per_step_accepted": [1]}
