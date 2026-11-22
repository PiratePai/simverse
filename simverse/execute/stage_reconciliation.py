"""Stage coverage and interval accounting in the service trace's clock domains."""
from .reduce import busy_integral
from .trace_fields import _integer, _list, _object


REQUEST_PHASES = {"prefill": 1, "decode": 2}


def _coverage(observed, expected, reason=None):
    if reason is None and (expected is None or observed < expected):
        reason = "缺少阶段观测"
    return {"observed_count": observed, "expected_count": expected,
            "ratio": observed / expected if expected else None, "complete": reason is None, "reason": reason}


def _clip(start, end, window):
    lo, hi = max(start, window["start_ns"]), min(end, window["end_ns"])
    return [(lo, hi)] if lo < hi else []


def _request_coverage(requests, spans, window):
    result, intervals = {}, []
    for stage, minimum in REQUEST_PHASES.items():
        expected = {row["request_id"] for row in requests if row["output_tokens"] >= minimum}
        observed = set()
        for span in spans:
            if span["stage"] == stage and span["resource"] == "request_wall" and (
                span.get("request_id") in expected and span.get("start_ns") is not None
            ):
                observed.add(span["request_id"])
                intervals.extend(_clip(span["start_ns"], span["end_ns"], window))
        result[stage] = _coverage(len(observed), len(expected))
    return result, busy_integral(intervals) if any(row["observed_count"] for row in result.values()) else None


def _runtime_coverage(runtime, speculative, round_commits, round_phases):
    if not speculative:
        return {}
    ranks = runtime.get("ranks", [])
    complete_inventory = bool(ranks) and all(r["source_row"] is not None and not r["phase_reason"] for r in ranks)
    has_verified_round = any(row["verified"] for row in round_commits.get("native_round_distribution", []))
    result = {}
    for stage in round_phases:
        rank_rows = [r["phases"][stage] for r in ranks]
        observed = sum(row["sample_count"] for row in rank_rows)
        expected = sum(row["expected_count"] for row in rank_rows)
        reason = None
        if not complete_inventory:
            reason = "缺少 rank 或后端轮次清单"
        elif has_verified_round and not expected:
            reason = "原生投机轮缺少对应阶段的预期记录"
        result[stage] = _coverage(observed, expected if expected or reason is None else None, reason)
    scheduler = runtime.get("scheduler_rollback") or {}
    result["scheduler_rollback"] = _coverage(scheduler.get("sample_count", 0), scheduler.get("expected_count"),
                                             None if scheduler.get("complete") else scheduler.get("reason"))
    return result


def _attribution(data, spans, required, manifest, phase_complete):
    field = "stage_attribution"
    ledger = data.get(field)
    window = data["window"]
    wall = window["end_ns"] - window["start_ns"]
    declared = {}
    for stage, ids in ({} if ledger is None else _object(ledger, field)).items():
        ids = _list(ids, f"{field}.{stage}")
        if not stage.strip() or any(not isinstance(i, str) or not i.strip() or i in declared for i in ids) or (
            len(ids) != len(set(ids))
        ):
            raise ValueError(f"{field}.{stage} 含无效或重复的 span ID")
        declared.update(dict.fromkeys(ids, stage))
    by_id = {s["span_id"]: s for s in spans if s["resource"] != "request_wall"}
    if ledger is not None and by_id.keys() - declared.keys():
        raise ValueError("阶段资源 span 缺少 stage_attribution 声明")

    missing, unaligned = [], []
    timed_stages, aligned, resources = set(), [], {}
    aligned_count = 0
    for span_id, stage in declared.items():
        span = by_id.get(span_id)
        if span is None:
            missing.append(span_id)
            continue
        if span["stage"] != stage:
            raise ValueError(f"{field}.{stage} 与 {span_id} 的阶段不符")
        rank = _integer(span, "rank", f"span {span_id}") if span.get("rank") is not None else None
        if rank is not None and "tp" in manifest and rank >= _integer(manifest, "tp", "manifest"):
            raise ValueError(f"span {span_id} 的 rank 超出 TP 范围")
        if span.get("start_ns") is None:
            missing.append(span_id)
            continue
        timed_stages.add(stage)
        clock = span.get("clock_id", window["clock_id"])
        lo, hi = span["start_ns"], span["end_ns"]
        if clock == window["clock_id"]:
            aligned_count += 1
            intervals = _clip(lo, hi, window)
            aligned.extend(intervals)
        else:
            unaligned.append(span_id)
            intervals = [(lo, hi)]
        resources.setdefault((clock, rank, span["resource"]), []).extend(intervals)

    observed = busy_integral(aligned) if aligned_count else None
    missing_stages = sorted(required - timed_stages)
    reasons = []
    if not declared:
        reasons.append("缺少阶段资源覆盖清单")
    if declared and (missing or missing_stages):
        reasons.append("缺少阶段资源区间")
    if unaligned:
        reasons.append("阶段资源时钟无法与观测窗口对齐")
    if not phase_complete:
        reasons.append("请求或运行时阶段观测不完整")
    per_resource = [
        {"clock_id": clock, "rank": rank, "resource": resource,
         "scope": "observation_window" if clock == window["clock_id"] else "local_clock",
         "window": {"start_ns": min(lo for lo, _ in values), "end_ns": max(hi for _, hi in values)} if values else None,
         "observed_busy_ns": busy_integral(values)}
        for (clock, rank, resource), values in sorted(resources.items(), key=lambda item: (
            item[0][0], -1 if item[0][1] is None else item[0][1], item[0][2],
        ))
    ]
    return {"wall_ns": wall, "observed_aligned_ns": observed,
            "unattributed_ns": wall - observed if not reasons else None, "reasons": reasons,
            "coverage": _coverage(aligned_count, len(declared) if ledger is not None else None,
                                  reasons[0] if reasons else None),
            "missing_stages": missing_stages, "missing_span_ids": missing, "unaligned_span_ids": unaligned,
            "resources": per_resource}


def summarize_stage_reconciliation(data, manifest, runtime, round_commits, round_phases):
    spans = data.get("spans", [])
    request, request_wall = _request_coverage(data["requests"], spans, data["window"])
    worker = _runtime_coverage(runtime, manifest["method"] != "baseline", round_commits, round_phases)
    combined = {**request, **worker}
    complete = all(row["complete"] for row in combined.values())
    required = {stage for stage, row in combined.items() if row["expected_count"] != 0 or not row["complete"]}
    return {"request_phases": request, "runtime_phases": worker,
            "phase_coverage_complete": complete,
            "observed_request_wall_ns": request_wall,
            "attribution": _attribution(data, spans, required, manifest, complete)}
