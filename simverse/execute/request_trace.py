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
from .reduce import (completed_request_ids_in_window, quantile_nearest_rank,
                     request_mean_itl, throughput_tokens_per_ms,
                     window_token_count)

NS_PER_MS = 1_000_000
NS_PER_S = 1_000_000_000
STATUSES = ("completed", "cancelled", "failed", "in_flight")
METHODS = ("baseline", "dflash_v1", "mtp")


@dataclass(frozen=True)
class _Request:
    sample_id: str
    input_tokens: int
    arrival_ns: int
    status: str
    emissions: tuple[int, ...]


def _object(value, field):
    if not isinstance(value, dict):
        raise ValueError(f"{field} 必须是对象")
    return value


def _list(value, field):
    if not isinstance(value, list):
        raise ValueError(f"{field} 必须是数组")
    return value


def _text(obj, key, field):
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field}.{key} 必须是非空字符串")
    return value


def _integer(obj, key, field):
    value = obj.get(key)
    if type(value) is not int or not 0 <= value < 2**63:
        raise ValueError(f"{field}.{key} 必须是非负 64 位整数")
    return value


def _check_timing(obj, window, field):
    for key in ("clock_id", "time_basis"):
        if obj.get(key, window[key]) != window[key]:
            raise ValueError(f"{field}.{key} 与服务端观测窗口不一致")


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
        if not accepted <= verified <= proposed <= limit:
            raise ValueError(f"{field} 须满足 accepted ≤ verified ≤ proposed ≤ configured_k")
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
        _check_timing(span, window, field)
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
    return {
        "schema_version": "1.0",
        "trace_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
        "manifest": manifest,
        **_summarize(requests, window, slo, start, end),
    }
