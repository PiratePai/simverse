"""Link scheduler decode rounds to service output sequence positions."""
from collections import Counter

from .trace_fields import _check_round_counts, _integer, _list, _object, _text, _tokens


def _native_acceptance(request, field, limit):
    stats = request.get("native_speculation")
    if stats is None:
        return None
    stats = _object(stats, f"{field}.native_speculation")
    keys = ("per_step_drafted", "per_step_verified", "per_step_accepted")
    if not any(key in stats for key in keys) and type(stats.get("num_spec_steps")) is int and stats["num_spec_steps"] == 0:
        return []
    counts = [_list(stats.get(key), f"{field}.native_speculation.{key}") for key in keys]
    if len({len(values) for values in counts}) != 1:
        raise ValueError(f"{field} 的投机轮数组长度不符")
    steps = list(zip(*counts))
    if "num_spec_steps" in stats and _integer(stats, "num_spec_steps", f"{field}.native_speculation") != len(steps):
        raise ValueError(f"{field} 的原生投机轮数与逐轮数组不符")
    for proposed, verified, accepted in steps:
        _check_round_counts(proposed, verified, accepted, limit, f"{field} 的原生投机统计")
    return steps


def _backend_rounds(scheduler, limit):
    field = "runtime_observations.scheduler"
    if scheduler is None or "rounds" not in scheduler:
        return {}, "缺少后端投机轮记录"
    reason = _text(scheduler, "reason", field) if scheduler.get("reason") is not None else None
    batches = _integer(scheduler, "step_count", field)
    grouped = {}
    for i, item in enumerate(_list(scheduler["rounds"], f"{field}.rounds")):
        location = f"{field}.rounds[{i}]"
        row = _object(item, location)
        rid = _text(row, "backend_request_id", location)
        step = _integer(row, "step", location)
        start = _integer(row, "token_start", location)
        counts = tuple(_integer(row, key, location) for key in ("proposed", "verified", "accepted"))
        _check_round_counts(*counts, limit, location)
        accepted = counts[2]
        target = row.get("target_offset")
        if target is not None and _integer(row, "target_offset", location) != accepted:
            raise ValueError(f"{location}.target_offset 与接受前缀不符")
        ids = row.get("backend_token_ids")
        if ids is not None and len(_tokens(ids, f"{location}.backend_token_ids")) > accepted + 1:
            raise ValueError(f"{location} 的后端提交数超过接受前缀加一个目标 token")
        if row.get("backend_finish_reason") is not None:
            _text(row, "backend_finish_reason", location)
        rounds = grouped.setdefault(rid, [])
        if not 1 <= step <= batches or (rounds and (
            step <= rounds[-1]["step"] or
            start < rounds[-1]["token_start"] + len(rounds[-1].get("backend_token_ids") or [])
        )):
            raise ValueError(f"{location} 的后端轮或提交序号重复、逆序或越界")
        rounds.append(row)
    return grouped, reason


def _output_sequence(request, field):
    ids, groups = [], []
    for i, emission in enumerate(request["emissions"]):
        location = f"{field}.emissions[{i}].token_ids"
        count = len(emission["token_indices"])
        tokens = emission.get("token_ids")
        if tokens is None:
            tokens = [None] * count
        elif len(_tokens(tokens, location)) != count:
            raise ValueError(f"{location} 与输出序号数不符")
        ids.extend(tokens)
        groups.extend([i] * count)
    return ids, groups


def _link_round(row, request_id, ids, groups, limit):
    first = row["token_start"]
    backend = row.get("backend_token_ids")
    accepted, verified = row["accepted"], row["verified"]
    result = {
        "request_id": request_id, "round_id": f"{request_id}:{row['step']}",
        "backend_request_id": row["backend_request_id"], "backend_step": row["step"],
        "backend_token_start": first, "backend_committed": len(backend) if backend is not None else None,
        "configured_k": limit, "proposed": row["proposed"], "verified": verified, "accepted": accepted,
        "committed": None, "token_start": None, "token_indices": None, "emission_group_indices": None,
        "target_bonus": None, "target_bonus_status": "unknown", "reason": None,
    }
    if row.get("backend_finish_reason") is not None:
        result["backend_finish_reason"] = row["backend_finish_reason"]
    if not row["proposed"]:
        result["end_reason"] = "后端记录本轮提议数为零"
    if backend is None:
        result["reason"] = "缺少后端提交 token"
        return result
    stop = min(first + len(backend), len(ids))
    emitted = ids[first:stop]
    if any(actual is not None and actual != expected for actual, expected in zip(emitted, backend)):
        raise ValueError(f"{result['round_id']} 的后端 token 与服务端输出序号上的 token 不符")
    if None in emitted:
        result["reason"] = "缺少服务端输出组的 token ID"
        return result
    result.update(committed=len(emitted), token_start=min(first, len(ids)), token_indices=list(range(first, stop)),
                  emission_group_indices=list(dict.fromkeys(groups[first:stop])))
    offset = row.get("target_offset")
    if not emitted or (offset is not None and offset >= len(emitted)):
        result["target_bonus_status"] = "not_emitted"
    elif offset is None:
        result["reason"] = "缺少目标 token 的轮内位置"
    else:
        result["target_bonus_status"] = "emitted"
        result["target_bonus"] = {"token_index": first + offset, "token_id": emitted[offset],
                                  "kind": "bonus" if accepted == verified else "replacement"}
    return result


def _distribution(counts, names):
    return [{**dict(zip(names, sample)), "count": count} for sample, count in sorted(counts.items())]


def summarize_round_commits(data, manifest):
    config = _object(manifest.get("method_config"), "manifest.method_config")
    limit = _integer(config, "num_speculative_tokens", "manifest.method_config")
    grouped, source_reason = _backend_rounds(data["runtime_observations"].get("scheduler"), limit)
    summaries, linked_rounds, mapped = {}, [], set()
    names = ("proposed", "verified", "accepted")
    native, joint, histogram = Counter(), Counter(), Counter()
    native_requests = 0
    for i, request in enumerate(data["requests"]):
        field = f"requests[{i}]"
        rid = request["request_id"]
        backend_id = request.get("backend_request_id")
        if backend_id is not None:
            backend_id = _text(request, "backend_request_id", field)
            if backend_id in mapped:
                raise ValueError(f"{field}.backend_request_id 重复归属")
            mapped.add(backend_id)
        observed = grouped.pop(backend_id, [])
        expected = _native_acceptance(request, field, limit)
        reason = None
        if expected is not None:
            native_requests += 1
            native.update(expected)
            histogram.update(accepted for _, _, accepted in expected)
            ordered = iter(expected)
            for row in observed:
                if tuple(row[key] for key in names) not in ordered:
                    raise ValueError(f"{field} 的后端轮计数或顺序与原生投机统计不符")
            if len(observed) < len(expected):
                reason = source_reason or ("缺少后端请求 ID" if backend_id is None else "缺少后端投机轮")
        else:
            reason = "缺少终态原生投机统计"
        ids, groups = _output_sequence(request, field)
        rounds = [_link_round(row, rid, ids, groups, limit) for row in observed]
        linked = [row for row in rounds if row["committed"] is not None]
        joint.update(tuple(row[key] for key in (*names, "committed")) for row in linked)
        linked_rounds.extend(rounds)
        reason = reason or next((row["reason"] for row in rounds if row["reason"]), None)
        summaries[rid] = {"expected_round_count": len(expected) if expected is not None else None,
                          "observed_round_count": len(rounds), "linked_round_count": len(linked),
                          "complete": reason is None, "reason": reason}
    unassigned = sum(map(len, grouped.values()))
    complete = not unassigned and all(row["complete"] for row in summaries.values())
    return {"complete": complete, "reason": None if complete else "逐轮提交证据不完整",
            "unassigned_backend_round_count": unassigned, "requests": summaries, "rounds": linked_rounds,
            "native_request_count": native_requests, "native_acceptance_histogram": {str(k): v for k, v in sorted(histogram.items())}
            if native_requests else None, "native_round_distribution": _distribution(native, names),
            "joint_distribution": _distribution(joint, (*names, "committed"))}
