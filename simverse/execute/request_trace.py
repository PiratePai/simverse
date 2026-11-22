"""Offline R07b/R09 request trace validation and server-side reduction."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

from ..contracts import TimeBasis, digest
from .reduce import (busy_integral, completed_request_ids_in_window, quantile_nearest_rank,
                     request_mean_itl, throughput_tokens_per_ms,
                     window_token_count)
from .service_round_commits import summarize_round_commits
from .stage_reconciliation import summarize_stage_reconciliation
from .trace_fields import _check_round_counts, _integer, _list, _object, _text

NS_PER_MS = 1_000_000
NS_PER_S = 1_000_000_000
STATUSES = ("completed", "cancelled", "failed", "in_flight")
METHODS = ("baseline", "dflash_v1", "mtp")
ROUND_PHASES = {
    "draft": {"role": "draft", "boundary": "speculator.propose"},
    "verify": {"role": "target", "boundary": "kv_connector.pre_forward.entry_to_finish_forward.entry"},
    "rejection_sampling": {"role": "target", "boundary": "rejection_sampler.__call__"},
    "recovery": {"role": "target", "boundary": "postprocess_sampled"},
}
MEMORY_COUNTERS = ("allocated", "reserved")


@dataclass(frozen=True)
class _Request:
    sample_id: str
    input_tokens: int
    arrival_ns: int
    status: str
    emissions: tuple[int, ...]


def _check_timing(obj, window, field, *, local_clock=False):
    if local_clock and "clock_id" in obj:
        _text(obj, "clock_id", field)
    elif obj.get("clock_id", window["clock_id"]) != window["clock_id"]:
        raise ValueError(f"{field}.clock_id 与服务端观测窗口不一致")
    if obj.get("time_basis", window["time_basis"]) != window["time_basis"]:
        raise ValueError(f"{field}.time_basis 与服务端观测窗口不一致")


def _sha256(obj, key, field):
    value = _text(obj, key, field)
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
        raise ValueError(f"{field}.{key} 必须是完整 SHA256")
    return value


def _provenance(manifest, directory: Path):
    _text(manifest, "run_id", "manifest")
    kind = manifest.get("source_kind")
    if kind not in ("synthetic", "captured"):
        raise ValueError("manifest.source_kind 必须是 synthetic 或 captured")
    if manifest.get("method") not in METHODS:
        raise ValueError("manifest.method 必须是 baseline、dflash_v1 或 mtp")
    raw_files = _object(manifest.get("raw_files", {}), "manifest.raw_files")
    if kind == "captured":
        for key in ("backend_revision", "model_revision", "tokenizer_revision",
                    "device", "topology", "precision", "command", "runtime_identity"):
            _text(manifest, key, "manifest")
        for key in ("tp", "ep"):
            if _integer(manifest, key, "manifest") == 0:
                raise ValueError(f"manifest.{key} 必须大于零")
        method = _object(manifest.get("method_config"), "manifest.method_config")
        if manifest["method"] != "baseline":
            _text(method, "revision", "manifest.method_config")
        if not raw_files:
            raise ValueError("manifest.raw_files 缺少采集原始文件")
    for name in raw_files:
        expected = _sha256(raw_files, name, "manifest.raw_files")
        reference = Path(name)
        path = (directory / reference).resolve()
        if reference.is_absolute() or not path.is_relative_to(directory) or not path.is_file():
            raise ValueError(f"manifest.raw_files 缺少目录内的原始文件 {name}")
        content = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(128 * 1024), b""):
                content.update(chunk)
        if "sha256:" + content.hexdigest() != expected:
            raise ValueError(f"manifest.raw_files 的 SHA256 不匹配：{name}")


def _parse_request(obj, field, window):
    _check_timing(obj, window, field)
    sample = _text(obj, "sample_id", field)
    arrival = _integer(obj, "arrival_ns", field)
    admit = arrival
    if obj.get("admit_ns") is not None:
        admit = _integer(obj, "admit_ns", field)
        if admit < arrival:
            raise ValueError(f"{field}.admit_ns 早于到达")
    status = obj.get("status")
    if status not in STATUSES:
        raise ValueError(f"{field}.status 无效")
    terminal = None
    if status != "in_flight":
        terminal = _integer(obj, "terminal_ns", field)
        _text(obj, "end_reason", field)
    elif obj.get("terminal_ns") is not None:
        raise ValueError(f"{field}.terminal_ns 必须为空")
    emissions = []
    previous = admit
    for i, row in enumerate(_list(obj.get("emissions"), f"{field}.emissions")):
        location = f"{field}.emissions[{i}]"
        group = _object(row, location)
        _check_timing(group, window, location)
        ns = _integer(group, "time_ns", location)
        if ns < previous:
            raise ValueError(f"{location}.time_ns 逆序或早于到达/准入")
        previous = ns
        indices = _list(group.get("token_indices"), f"{location}.token_indices")
        if not indices or (window["granularity"] == "token" and len(indices) != 1):
            raise ValueError(f"{location}.token_indices 与提交粒度不符")
        for expected, index in enumerate(indices, start=len(emissions)):
            if type(index) is not int or index != expected:
                raise ValueError(f"{location}.token_indices 须从 0 连续递增，无重复")
        emissions.extend([ns] * len(indices))
    if terminal is not None and terminal < previous:
        raise ValueError(f"{field}.terminal_ns 早于到达、准入或最后提交")
    if len(emissions) != _integer(obj, "output_tokens", field):
        raise ValueError(f"{field}.output_tokens 与提交序号不符")
    return _Request(sample, _integer(obj, "input_tokens", field), arrival, status, tuple(emissions))


def _check_details(data, requests, window, manifest):
    rounds = {}
    committed_indices = {rid: set() for rid in requests}
    for i, row in enumerate(_list(data.get("rounds", []), "rounds")):
        field = f"rounds[{i}]"
        current = _object(row, field)
        _check_timing(current, window, field)
        rid = _text(current, "request_id", field)
        round_id = _text(current, "round_id", field)
        if manifest["method"] == "baseline" or rid not in requests or round_id in rounds:
            raise ValueError(f"{field} 的方法、请求引用或轮 ID 无效")
        rounds[round_id] = rid
        limit, proposed, verified, accepted, committed = (
            _integer(current, key, field) for key in
            ("configured_k", "proposed", "verified", "accepted", "committed")
        )
        _check_round_counts(proposed, verified, accepted, limit, field)
        if not proposed:
            _text(current, "end_reason", field)
        first = _integer(current, "token_start", field)
        if first + committed > len(requests[rid].emissions):
            raise ValueError(f"{field}.committed 与提交序号不符")
        indices = range(first, first + committed)
        if not committed_indices[rid].isdisjoint(indices):
            raise ValueError(f"{field}.committed 与其他轮重复归属")
        committed_indices[rid].update(indices)
    batches = _object(data.get("batches", {}), "batches")
    for batch_id, members in batches.items():
        field = f"batches.{batch_id}"
        ids = _list(members, field)
        if not batch_id.strip() or not ids or any(
            not isinstance(rid, str) or rid not in requests for rid in ids
        ) or len(set(ids)) != len(ids):
            raise ValueError(f"{field} 含无效或重复的请求引用")
    span_ids = set()
    for i, row in enumerate(_list(data.get("spans", []), "spans")):
        field = f"spans[{i}]"
        span = _object(row, field)
        _check_timing(span, window, field, local_clock=data.get("stage_attribution") is not None
                      and span.get("resource") != "request_wall")
        span_id = _text(span, "span_id", field)
        if span_id in span_ids:
            raise ValueError(f"{field}.span_id 重复")
        span_ids.add(span_id)
        rid = span.get("request_id")
        batch_id = span.get("batch_id")
        if rid is not None and batch_id is None:
            rid = _text(span, "request_id", field)
            if rid not in requests:
                raise ValueError(f"{field}.request_id 引用缺失")
            members = [rid]
        elif batch_id is not None and rid is None:
            members = batches.get(_text(span, "batch_id", field))
            if members is None:
                raise ValueError(f"{field}.batch_id 引用缺失")
        else:
            raise ValueError(f"{field} 须提供 request_id 或 batch_id 之一")
        if span.get("round_id") is not None:
            if rounds.get(_text(span, "round_id", field)) not in members:
                raise ValueError(f"{field}.round_id 引用缺失或归属不符")
        for key in ("stage", "role", "resource", "timing_boundary"):
            _text(span, key, field)
        if span.get("start_ns") is None and span.get("end_ns") is None:
            _text(span, "timing_reason", field)
        elif _integer(span, "end_ns", field) < _integer(span, "start_ns", field):
            raise ValueError(f"{field} 的阶段起止时间逆序")


def _ms(value):
    return value / NS_PER_MS if value is not None else None


def _runtime_memory(row, backend, field):
    memory = row.get("memory")
    if memory is not None and backend != "native":
        raise ValueError(f"{field}.memory 仅接受 native CUDA allocator 的计数")
    memory = {} if memory is None else _object(memory, f"{field}.memory")
    reason = "分配器不支持此计数口径" if backend != "native" else "缺少运行时显存计数"
    counters = {}
    keys = ("start_bytes", "end_bytes", "peak_bytes")
    for name in MEMORY_COUNTERS:
        if memory.get(name) is None:
            counters[name] = {"peak_bytes": None, "reason": reason}
            continue
        counter_field = f"{field}.memory.{name}"
        counter = _object(memory[name], counter_field)
        counters[name] = {key: _integer(counter, key, counter_field) for key in keys}
        start, end, peak = counters[name].values()
        if peak < max(start, end):
            raise ValueError(f"{counter_field}.peak_bytes 低于窗口边界读数")
    for key in keys:
        allocated, reserved = (counters[name].get(key) for name in MEMORY_COUNTERS)
        if allocated is not None and reserved is not None and allocated > reserved:
            raise ValueError(f"{field}.memory.{key} 的 allocated 超过 reserved")
    return counters


def _runtime_step(row, field, start, end):
    step = _integer(row, "step", field)
    ids = {}
    for key in ("batch_request_ids", "verify_request_ids"):
        values = _list(row.get(key), f"{field}.{key}")
        if any(not isinstance(v, str) or not v.strip() for v in values) or len(set(values)) != len(values):
            raise ValueError(f"{field}.{key} 须为不重复的请求 ID")
        ids[key] = tuple(sorted(values))
    batch, verified = ids.values()
    if not step or not batch or not set(verified).issubset(batch):
        raise ValueError(f"{field} 的轮 ID 或校验请求归属无效")
    draft_count = _integer(row, "draft_tokens_to_schedule", field)
    expected = ({"draft"} if draft_count else set()) | (
        {"verify", "rejection_sampling", "recovery"} if verified else set()
    )
    spans = _object(row.get("phases"), f"{field}.phases")
    if spans.keys() - expected:
        raise ValueError(f"{field}.phases 存在未调度的阶段")
    observations = {}
    for phase, span in spans.items():
        span_field = f"{field}.phases.{phase}"
        span = _object(span, span_field)
        stream = _text(span, "stream_id", span_field)
        lo, hi, host_lo, host_hi = (_integer(span, key, span_field) for key in
                                    ("start_ns", "end_ns", "host_start_ns", "host_end_ns"))
        if lo > hi or not start <= host_lo <= host_hi <= end:
            raise ValueError(f"{span_field} 的 CUDA 区间或主机窗口无效")
        observations[phase] = (stream, lo, hi)
    return step, (batch, verified, draft_count), expected, observations


def _timing_quantiles(durations):
    return {f"observed_p{p}_ms": _ms(quantile_nearest_rank(durations, p / 100)) for p in (50, 95)}


def _rank_timings(observed, expected_steps, reason):
    summary = {}
    for phase, expected in expected_steps.items():
        samples = {step: phases[phase] for step, phases in observed.items() if phase in phases}
        missing = sorted(expected - samples.keys())
        durations = [hi - lo for _, lo, hi in samples.values()]
        gap = None
        if missing or not samples:
            gap = reason or ("缺少轮内阶段时间" if missing else "无适用轮次")
        summary[phase] = {
            "sample_count": len(samples), "expected_count": len(expected), "missing_steps": missing,
            **_timing_quantiles(durations), "reason": gap,
        }
    streams = {}
    for phases in observed.values():
        for stream, lo, hi in phases.values():
            streams.setdefault(stream, []).append((lo, hi))
    return summary, {stream: _ms(busy_integral(intervals)) for stream, intervals in sorted(streams.items())}


def _runtime_window(row, field, request_bounds):
    field += ".window"
    window = _object(row.get("window"), field)
    start, end = (_integer(window, key, field) for key in ("start_ns", "end_ns"))
    if start >= end or (request_bounds and not start <= request_bounds[0] <= request_bounds[1] <= end):
        raise ValueError(f"{field} 未覆盖请求的观测区间")
    return {"start_ns": start, "end_ns": end}


def _whole_device_memory(value, field, request_bounds):
    if value is None:
        return {"observed_peak_bytes": None, "sample_count": 0, "complete": False,
                "reason": "缺少整卡显存采样"}
    row = _object(value, field)
    period = _integer(row, "sample_interval_ns", field)
    if not period:
        raise ValueError(f"{field} 的采样间隔须大于零")
    reason = _text(row, "reason", field) if row.get("reason") is not None else None
    samples = _list(row.get("samples"), f"{field}.samples")
    device_uuid = _text(row, "device_uuid", field) if samples or row.get("device_uuid") is not None else None
    if device_uuid and not device_uuid.startswith("GPU-"):
        raise ValueError(f"{field}.device_uuid 须为 NVML 物理 GPU UUID")
    times, used = [], []
    total = None
    for i, sample in enumerate(samples):
        sample_field = f"{field}.samples[{i}]"
        sample = _object(sample, sample_field)
        ns, occupied, capacity = (_integer(sample, key, sample_field) for key in
                                  ("time_ns", "used_bytes", "total_bytes"))
        if not capacity or occupied > capacity or (total is not None and capacity != total):
            raise ValueError(f"{sample_field} 的整卡显存用量或容量无效")
        if times and ns < times[-1]:
            raise ValueError(f"{sample_field} 的采样时间逆序")
        times.append(ns)
        used.append(occupied)
        total = capacity
    if not reason:
        if len(times) < 2 or times[0] == times[-1]:
            reason = "整卡显存样本不足"
        elif request_bounds and not times[0] <= request_bounds[0] <= request_bounds[1] <= times[-1]:
            reason = "整卡显存采样未覆盖请求的观测区间"
    return {"device_uuid": device_uuid, "total_bytes": total,
            "sample_count": len(samples), "sample_interval_ms": _ms(period),
            "window": {"start_ns": times[0], "end_ns": times[-1]} if times else None,
            "max_sample_gap_ms": _ms(max((b - a for a, b in zip(times, times[1:])), default=None)),
            "observed_peak_bytes": max(used, default=None), "complete": reason is None, "reason": reason}


def _scheduler_rollback(data, field, request_bounds, gpu_steps):
    boundary = {"resource": "engine_core_cpu_thread", "boundary": "update_from_output.rejection_counter_rollback"}
    if data is None:
        return {**boundary, **_timing_quantiles([]), "sample_count": 0, "expected_count": None,
                "observed_busy_ms": None, "complete": False, "reason": "缺少 CPU scheduler 观测"}
    row = _object(data, field)
    window = _runtime_window(row, field, request_bounds)
    step_count = _integer(row, "step_count", field)
    reason = _text(row, "reason", field) if row.get("reason") is not None else None
    expected = _integer(row, "rollback_count", field) if row.get("rollback_count") is not None else None
    seen, intervals = set(), []
    for i, item in enumerate(_list(row.get("rollbacks"), f"{field}.rollbacks")):
        item_field = f"{field}.rollbacks[{i}]"
        rollback = _object(item, item_field)
        step = _integer(rollback, "step", item_field)
        rid = _text(rollback, "request_id", item_field)
        if not 1 <= step <= step_count or (step, rid) in seen or (
            step in gpu_steps and rid not in gpu_steps[step][1]
        ):
            raise ValueError(f"{item_field} 的 CPU 回退批次或请求归属无效")
        seen.add((step, rid))
        if not _integer(rollback, "rejected_tokens", item_field):
            raise ValueError(f"{item_field} 的回退 token 数须大于零")
        lo, hi = (_integer(rollback, key, item_field) for key in ("start_ns", "end_ns"))
        if not window["start_ns"] <= lo <= hi <= window["end_ns"]:
            raise ValueError(f"{item_field} 的 CPU 回退时间超出窗口或逆序")
        intervals.append((lo, hi))
    observed = len(intervals)
    if expected is not None and expected < observed:
        raise ValueError(f"{field} 的 CPU 回退记录超出声明次数")
    if reason is None and (expected is None or expected > observed or step_count < max(gpu_steps, default=0)):
        reason = "CPU scheduler 轮次或回退记录缺失"
    busy = _ms(busy_integral(intervals)) if observed or reason is None else None
    return {**boundary, **_timing_quantiles([hi - lo for lo, hi in intervals]), "window": window,
            "step_count": step_count, "sample_count": observed, "expected_count": expected,
            "observed_busy_ms": busy, "complete": reason is None,
            "reason": "采集窗口内未执行回退" if reason is None and not intervals else reason}


def _summarize_runtime(data, manifest):
    field = "runtime_observations"
    data_runtime = _object(data[field], field)
    tp = _integer(manifest, "tp", "manifest")
    if not tp:
        raise ValueError("manifest.tp 必须大于零")
    ranks = _list(data_runtime.get("ranks"), f"{field}.ranks")
    scheduler_data = data_runtime.get("scheduler")
    origin = _integer(data_runtime, "clock_origin_ns", field) if ranks or scheduler_data is not None else None
    speculative = manifest["method"] != "baseline"
    request_bounds = None
    if origin is not None and data["requests"]:
        first_arrival = min(row["arrival_ns"] for row in data["requests"])
        last_terminal = max(row.get("terminal_ns") or row["arrival_ns"] for row in data["requests"])
        request_bounds = origin + first_arrival, origin + last_terminal
    if scheduler_data is not None and not speculative:
        raise ValueError("baseline 含 CPU 投机回退记录")
    rank_summaries, rank_steps, all_steps = {}, {}, {}
    expected_steps = {phase: set() for phase in ROUND_PHASES} if speculative else {}
    for i, row in enumerate(ranks):
        row_field = f"{field}.ranks[{i}]"
        row = _object(row, row_field)
        rank = _integer(row, "rank", row_field)
        if rank >= tp or rank in rank_summaries:
            raise ValueError(f"{row_field}.rank 超出 TP 范围或重复")
        window = _runtime_window(row, row_field, request_bounds)
        backend = _text(row, "allocator_backend", row_field)
        phase_reason = row.get("phase_reason")
        if phase_reason is not None:
            phase_reason = _text(row, "phase_reason", row_field)
        elif speculative and "steps" not in row:
            phase_reason = "缺少后端轮次清单"
        rank_summaries[rank] = {
            "rank": rank, "device_index": _integer(row, "device_index", row_field), "source_row": i,
            "allocator_backend": backend, "window": window,
            "memory": _runtime_memory(row, backend, row_field),
            "device_memory": _whole_device_memory(row.get("device_memory"), f"{row_field}.device_memory", request_bounds),
            "phase_reason": phase_reason,
            **{key: {"value": _integer(row, key, row_field)} if row.get(key) is not None else
               {"value": None, "reason": "缺少 vLLM 的初始化预算"}
               for key in ("requested_memory_bytes", "kv_cache_budget_bytes")},
        }
        rank_steps[rank] = observed = {}
        for j, step in enumerate(_list(row.get("steps", []), f"{row_field}.steps")):
            step_field = f"{row_field}.steps[{j}]"
            if not speculative:
                raise ValueError("baseline 含投机轮内记录")
            index, identity, expected, phases = _runtime_step(
                _object(step, step_field), step_field, window["start_ns"], window["end_ns"],
            )
            if index in observed or all_steps.setdefault(index, identity) != identity:
                raise ValueError(f"{step_field} 的轮 ID 重复或跨 rank 的批次归属不一致")
            observed[index] = phases
            for phase in expected:
                expected_steps[phase].add(index)

    scheduler = _scheduler_rollback(scheduler_data, f"{field}.scheduler", request_bounds, all_steps) if speculative else None
    complete = scheduler is None or scheduler["complete"]
    for rank in range(tp):
        missing_reason = "缺少 rank 的运行时记录"
        result = rank_summaries.setdefault(rank, {
            "rank": rank, "device_index": None, "source_row": None, "allocator_backend": None, "window": None,
            "memory": {name: {"peak_bytes": None, "reason": missing_reason} for name in MEMORY_COUNTERS},
            "device_memory": {"observed_peak_bytes": None, "sample_count": 0, "complete": False,
                              "reason": missing_reason},
            "requested_memory_bytes": {"value": None, "reason": missing_reason},
            "kv_cache_budget_bytes": {"value": None, "reason": missing_reason},
            "phase_reason": missing_reason,
        })
        timings, busy = _rank_timings(rank_steps.get(rank, {}), expected_steps, result["phase_reason"])
        result.update(phases=timings, observed_stream_busy_ms=busy)
        if not result["device_memory"]["complete"] or any(
            metric["peak_bytes"] is None for metric in result["memory"].values()
        ) or any(
            phase["reason"] is not None for phase in timings.values()
        ):
            complete = False
    slowest = {}
    for phase in expected_steps:
        samples = [(rank, summary["phases"][phase]) for rank, summary in rank_summaries.items()]
        if any(timing["reason"] is not None for _, timing in samples):
            slowest[phase] = {"rank": None, "p95_ms": None, "reason": "缺少可对照的各 rank 轮内时间"}
            continue
        peak = max(timing["observed_p95_ms"] for _, timing in samples)
        rank = min(rank for rank, timing in samples if timing["observed_p95_ms"] == peak)
        slowest[phase] = {"rank": rank, "p95_ms": peak}
    return {"complete": complete, "phase_applicable": speculative, "estimator": "nearest_rank",
            "phase_boundaries": ROUND_PHASES if speculative else {}, "slowest_rank_by_phase_p95": slowest,
            "phase_clock": "cuda_event_per_rank_stream", "phase_host_clock": data["window"]["clock_id"],
            "memory_scope": "pytorch_cuda_allocator", "clock_origin_ns": origin,
            "device_memory_scope": "nvml_sampled_physical_gpu", "scheduler_rollback": scheduler,
            "ranks": [rank_summaries[rank] for rank in range(tp)]}


def _summarize(requests, window, slo, start, end):
    # Reducers consume a common time unit. Keep integer ns for exact window comparisons.
    emissions = {rid: req.emissions for rid, req in requests.items()}
    completed = [rid for rid, req in requests.items() if req.status == "completed"]
    window_completed = set(completed_request_ids_in_window(emissions, completed, start, end))
    per_request = {}
    ttft_samples = []
    qualified = []
    for rid, req in requests.items():
        times = req.emissions
        ttft = times[0] - req.arrival_ns if times else None
        itl = request_mean_itl(times)
        per_request[rid] = {
            "status": req.status, "output_tokens": len(times),
            "ttft_ms": _ms(ttft), "ttft_reason": None if times else "无服务端输出",
            "e2e_ms": _ms(times[-1] - req.arrival_ns) if times else None,
            "itl_ms": [_ms(b - a) for a, b in zip(times, times[1:])],
            "mean_itl_ms": _ms(itl),
        }
        if rid in window_completed:
            ttft_samples.append(ttft)
            if slo is not None and ttft <= slo["ttft_max_ns"] and (
                itl is None or itl <= slo["mean_itl_max_ns"]
            ):
                qualified.append(rid)
    result = {
        "window": window,
        "status_counts": dict(Counter(req.status for req in requests.values())),
        "requests": per_request,
        "window_completed_requests": len(window_completed),
        "window_token_events": window_token_count(emissions, start, end),
        "completed_token_per_s": throughput_tokens_per_ms(emissions, completed, start, end) * NS_PER_S,
        "event_token_per_s": throughput_tokens_per_ms(emissions, completed, start, end, "token_events") * NS_PER_S,
        "ttft": {
            "sample_count": len(ttft_samples), "estimator": "nearest_rank",
            "p50_ms": _ms(quantile_nearest_rank(ttft_samples, 0.50)),
            "p95_ms": _ms(quantile_nearest_rank(ttft_samples, 0.95)),
            "reason": None if ttft_samples else "窗口内无有输出的成功请求",
        },
        "slo": slo,
        "goodput": None,
    }
    if slo is not None:
        result["goodput"] = {
            "requests_per_s": len(qualified) * NS_PER_S / (end - start),
            "token_per_s": throughput_tokens_per_ms(emissions, qualified, start, end) * NS_PER_S,
        }
    return result


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON 键重复：{key}")
        result[key] = value
    return result


def _finite_float(text):
    number = float(text)
    if not isfinite(number):
        raise ValueError(f"JSON 数值必须有限：{text}")
    return number


def summarize_request_trace(path: str | Path) -> dict:
    """Read one trace and report server-side metrics without changing the inputs."""
    path = Path(path)
    raw = path.read_bytes()
    data = _object(json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object,
                              parse_float=_finite_float, parse_constant=_finite_float), "trace")
    if data.get("schema_version") != "1.0":
        raise ValueError("schema_version 必须是 1.0")
    manifest = _object(data.get("manifest"), "manifest")
    window = _object(data.get("window"), "window")
    _text(window, "clock_id", "window")
    if window.get("time_basis") not in (TimeBasis.COMMITTED.value, TimeBasis.SERVER_EMITTED.value):
        raise ValueError("window.time_basis 须为 committed 或 server_emitted")
    granularity = window.get("granularity")
    if granularity not in ("token", "commit_group"):
        raise ValueError("window.granularity 须为 token 或 commit_group")
    start = _integer(window, "start_ns", "window")
    end = _integer(window, "end_ns", "window")
    if start >= end:
        raise ValueError("window 须满足 start_ns < end_ns")
    for key, lower, upper in (("warmup_end_ns", 0, start), ("drain_end_ns", end, 2**63 - 1)):
        if key in window and not lower <= _integer(window, key, "window") <= upper:
            raise ValueError(f"window.{key} 超出允许范围")
    slo = data.get("slo")
    if slo is not None:
        _object(slo, "slo")
        _integer(slo, "ttft_max_ns", "slo")
        _integer(slo, "mean_itl_max_ns", "slo")
    requests = {}
    for i, row in enumerate(_list(data.get("requests"), "requests")):
        field = f"requests[{i}]"
        obj = _object(row, field)
        rid = _text(obj, "request_id", field)
        if rid in requests:
            raise ValueError(f"{field}.request_id 重复")
        requests[rid] = _parse_request(obj, field, window)
    workload = [{"request_id": rid, "sample_id": req.sample_id, "input_tokens": req.input_tokens,
                 "arrival_ns": req.arrival_ns} for rid, req in sorted(requests.items())]
    if _sha256(manifest, "workload_sha256", "manifest") != digest(workload):
        raise ValueError("manifest.workload_sha256 与请求清单不匹配")
    _provenance(manifest, path.resolve().parent)
    _check_details(data, requests, window, manifest)
    runtime = {}
    if "runtime_observations" in data:
        runtime["runtime_observations"] = _summarize_runtime(data, manifest)
        if manifest["method"] != "baseline":
            runtime["round_commits"] = summarize_round_commits(data, manifest)
    if "runtime_observations" in data or "stage_attribution" in data:
        runtime["stage_reconciliation"] = summarize_stage_reconciliation(
            data, manifest, runtime.get("runtime_observations", {}), runtime.get("round_commits", {}), ROUND_PHASES,
        )
    return {
        "schema_version": "1.0",
        "trace_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
        "manifest": manifest,
        **_summarize(requests, window, slo, start, end),
        **runtime,
    }
