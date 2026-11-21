"""Matched-workload service captures and replay for R07b/R09."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from ..contracts import digest
from .request_trace import (METHODS, _finite_float, _integer, _list, _object,
                            _text, _unique_object, summarize_request_trace)

VLLM_REVISION = "11aa1369e297f67bc4dc26e119d10830f39956c6"
SERVER_CLOCK_ID = "linux_host_monotonic"
TARGET_MODEL = "Qwen/Qwen3.6-35B-A3B"
DRAFT_MODEL = "z-lab/Qwen3.6-35B-A3B-DFlash"


def _json(raw):
    return json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object,
                      parse_float=_finite_float, parse_constant=_finite_float)


def _sha(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                    encoding="utf-8")


def _tokens(value, field):
    tokens = _list(value, field)
    if any(type(token) is not int or not 0 <= token < 2**31 for token in tokens):
        raise ValueError(f"{field} 含无效 token ID")
    return tokens


def read_plan(path):
    raw = Path(path).read_bytes()
    plan = _object(_json(raw), "plan")
    extra = set(plan) - {
        "schema_version", "run_id", "model_revision", "tokenizer_revision", "draft_revision",
        "runtime_identity", "topology", "tp", "minimum_device_memory_bytes", "max_model_len",
        "max_num_seqs", "dflash_k", "mtp_k", "enable_expert_parallel", "gpu_memory_utilization",
        "sampling", "window", "arrival_tolerance_ns", "warmup_runs", "warmup_timeout_ns",
        "requests", "slo",
    }
    if extra:
        raise ValueError(f"plan 含未支持的字段：{', '.join(sorted(extra))}")
    if plan.get("schema_version") != "1.0":
        raise ValueError("plan.schema_version 必须是 1.0")
    for key in ("run_id", "model_revision", "tokenizer_revision", "runtime_identity", "topology"):
        _text(plan, key, "plan")
    for key in ("model_revision", "tokenizer_revision", "draft_revision"):
        revision = _text(plan, key, "plan")
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError(f"plan.{key} 必须是完整的小写提交哈希")
    for key in ("tp", "minimum_device_memory_bytes", "max_model_len", "max_num_seqs",
                "dflash_k", "mtp_k"):
        if _integer(plan, key, "plan") == 0:
            raise ValueError(f"plan.{key} 必须大于零")
    if type(plan.get("enable_expert_parallel")) is not bool:
        raise ValueError("plan.enable_expert_parallel 必须是布尔值")
    utilization = plan.get("gpu_memory_utilization")
    if type(utilization) not in (int, float) or not 0 < utilization <= 1:
        raise ValueError("plan.gpu_memory_utilization 必须在 (0, 1] 内")
    sampling = _object(plan.get("sampling"), "sampling")
    if _integer(sampling, "max_tokens", "sampling") == 0:
        raise ValueError("sampling.max_tokens 必须大于零")
    _integer(sampling, "seed", "sampling")
    temperature = sampling.get("temperature")
    if type(temperature) not in (int, float) or not 0 <= temperature <= 2:
        raise ValueError("sampling.temperature 必须在 [0, 2] 内")
    if type(sampling.get("ignore_eos")) is not bool:
        raise ValueError("sampling.ignore_eos 必须是布尔值")
    if set(sampling) != {"temperature", "seed", "max_tokens", "ignore_eos"}:
        raise ValueError("sampling 含未支持的配置")
    window = _object(plan.get("window"), "window")
    if set(window) != {"start_ns", "end_ns", "drain_end_ns"}:
        raise ValueError("window 须提供 start_ns、end_ns 和 drain_end_ns")
    start, end, drain = (_integer(window, key, "window") for key in
                         ("start_ns", "end_ns", "drain_end_ns"))
    if not start < end <= drain:
        raise ValueError("window 须满足 start_ns < end_ns ≤ drain_end_ns")
    if plan.get("slo") is not None:
        slo = _object(plan["slo"], "slo")
        for key in ("ttft_max_ns", "mean_itl_max_ns"):
            _integer(slo, key, "slo")
    _integer(plan, "arrival_tolerance_ns", "plan")
    for key in ("warmup_runs", "warmup_timeout_ns"):
        if key in plan and _integer(plan, key, "plan") == 0:
            raise ValueError(f"plan.{key} 必须大于零")
    requests = _list(plan.get("requests"), "requests")
    if not requests:
        raise ValueError("requests 不能为空")
    ids = set()
    for i, item in enumerate(requests):
        field = f"requests[{i}]"
        request = _object(item, field)
        if set(request) != {"request_id", "sample_id", "prompt_token_ids", "planned_arrival_ns"}:
            raise ValueError(f"{field} 含未支持或缺失的字段")
        rid = _text(request, "request_id", field)
        _text(request, "sample_id", field)
        if rid in ids:
            raise ValueError(f"{field}.request_id 重复")
        ids.add(rid)
        tokens = _tokens(request.get("prompt_token_ids"), f"{field}.prompt_token_ids")
        if not tokens or len(tokens) + sampling["max_tokens"] > plan["max_model_len"]:
            raise ValueError(f"{field} 的输入与输出预算超出可用长度")
        if _integer(request, "planned_arrival_ns", field) >= end:
            raise ValueError(f"{field}.planned_arrival_ns 必须早于窗口结束")
    return plan, raw


def speculative_config(method, plan):
    if method == "baseline":
        return None
    if method == "dflash_v1":
        return {"method": "dflash", "model": DRAFT_MODEL, "revision": plan["draft_revision"],
                "num_speculative_tokens": plan["dflash_k"]}
    if method == "mtp":
        return {"method": "mtp", "num_speculative_tokens": plan["mtp_k"]}
    raise ValueError("不支持的采集方法")


def _acceptance(row, field, limit):
    stats = row.get("native_speculation")
    if stats is None:
        return None
    stats = _object(stats, f"{field}.native_speculation")
    keys = ("per_step_drafted", "per_step_verified", "per_step_accepted")
    if not all(key in stats for key in keys) and stats.get("num_spec_steps") == 0:
        return []
    counts = [_list(stats.get(key), f"{field}.native_speculation.{key}") for key in keys]
    if len({len(values) for values in counts}) != 1:
        raise ValueError(f"{field} 的投机轮数组长度不符")
    steps = list(zip(*counts))
    if stats.get("num_spec_steps", len(steps)) != len(steps):
        raise ValueError(f"{field} 的原生投机轮数与逐轮数组不符")
    for proposed, verified, accepted in steps:
        if any(type(n) is not int for n in (proposed, verified, accepted)) or not (
            0 <= accepted <= verified <= proposed <= limit
        ):
            raise ValueError(f"{field} 的原生投机统计超出数量边界")
    return steps


def _package_mode(plan, plan_raw, mode, directory, server_raw):
    capture = _object(_json(server_raw), "server")
    if capture.get("method") != mode or capture.get("plan_sha256") != _sha(plan_raw):
        raise ValueError(f"{mode} 的原始记录与计划或方法不匹配")
    kind = capture.get("source_kind")
    if kind not in ("captured", "synthetic"):
        raise ValueError(f"{mode} 的来源类型无效")
    runtime = _object(capture.get("runtime"), "server.runtime")
    native_config = speculative_config(mode, plan)
    if capture.get("speculative_config") != native_config:
        raise ValueError(f"{mode} 的投机配置与计划不匹配")
    if kind == "captured" and runtime.get("backend_revision") != VLLM_REVISION:
        raise ValueError(f"{mode} 的 vLLM 提交与采集适配器不匹配")
    offered = {r["request_id"]: r for r in plan["requests"]}
    requests, spans, outputs = [], [], {}
    all_steps = Counter()
    request_coverage = Counter()
    max_drift = 0
    for i, item in enumerate(_list(capture.get("requests"), "server.requests")):
        field = f"server.requests[{i}]"
        row = _object(item, field)
        rid = _text(row, "request_id", field)
        if rid not in offered or rid in outputs:
            raise ValueError(f"{field}.request_id 未在负载登记或重复")
        source = offered[rid]
        if row.get("prompt_token_ids") != source["prompt_token_ids"]:
            raise ValueError(f"{field}.prompt_token_ids 与负载不符")
        actual_arrival = _integer(row, "arrival_ns", field)
        max_drift = max(max_drift, abs(actual_arrival - source["planned_arrival_ns"]))
        emissions, ids = [], []
        for chunk in _list(row.get("chunks"), f"{field}.chunks"):
            record = _object(chunk, f"{field}.chunks[]")
            tokens = _tokens(record.get("token_ids"), f"{field}.chunks[].token_ids")
            if tokens:
                emissions.append({"time_ns": _integer(record, "time_ns", f"{field}.chunks[]"),
                                  "token_indices": list(range(len(ids), len(ids) + len(tokens)))})
                ids.extend(tokens)
        if len(ids) > plan["sampling"]["max_tokens"]:
            raise ValueError(f"{field} 的实际输出超出计划上限")
        outputs[rid] = (row.get("status"), row.get("end_reason"), ids)
        requests.append({
            "request_id": rid, "sample_id": source["sample_id"],
            "input_tokens": len(source["prompt_token_ids"]), "arrival_ns": actual_arrival,
            "status": row.get("status"), "terminal_ns": row.get("terminal_ns"),
            "end_reason": row.get("end_reason"), "output_tokens": len(ids),
            "emissions": emissions,
        })
        for phase in ("prefill", "decode"):
            timing = _object(_object(row.get("phase_spans", {}), f"{field}.phase_spans").get(
                phase, {}), f"{field}.phase_spans.{phase}")
            span = {"span_id": f"{rid}:{phase}", "request_id": rid, "stage": phase,
                    "role": "target", "resource": "request_wall",
                    "timing_boundary": "engine_core_monotonic",
                    "start_ns": timing.get("start_ns"), "end_ns": timing.get("end_ns")}
            if span["start_ns"] is None and span["end_ns"] is None:
                span["timing_reason"] = "后端未提供可用的请求阶段时间"
            else:
                request_coverage[phase] += 1
            spans.append(span)
        if native_config is not None:
            steps = _acceptance(row, field, native_config["num_speculative_tokens"])
            if steps is not None:
                request_coverage["native_speculation"] += 1
                all_steps.update(steps)
        elif row.get("native_speculation") is not None:
            raise ValueError("baseline 含原生投机统计")
    if set(outputs) != set(offered):
        raise ValueError(f"{mode} 的原始记录缺少计划请求")

    workload = [{key: r[key] for key in ("request_id", "sample_id", "input_tokens", "arrival_ns")}
                for r in sorted(requests, key=lambda r: r["request_id"])]
    base = plan["window"]
    clock_id = _text(capture, "clock_id", "server")
    if kind == "captured" and clock_id != SERVER_CLOCK_ID:
        raise ValueError(f"{mode} 的采集时钟来源无效")
    trace = {
        "schema_version": "1.0",
        "manifest": {
            "run_id": f'{plan["run_id"]}:{mode}', "source_kind": kind, "method": mode,
            "workload_sha256": digest(workload), "paired_plan_sha256": _sha(plan_raw),
            "raw_files": {"plan.json": _sha(plan_raw), "server.json": _sha(server_raw)},
            "model_revision": plan["model_revision"], "tokenizer_revision": plan["tokenizer_revision"],
            "topology": plan["topology"], "precision": "bfloat16", "tp": plan["tp"],
            "ep": plan["tp"] if plan["enable_expert_parallel"] else 1,
            "backend_revision": runtime.get("backend_revision"), "device": runtime.get("device"),
            "runtime_identity": plan["runtime_identity"], "command": capture.get("command"),
            "method_config": {**(native_config or {}),
                              "revision": plan["draft_revision"] if mode == "dflash_v1" else plan["model_revision"]},
        },
        "window": {**base, "warmup_end_ns": base["start_ns"], "clock_id": clock_id,
                   "time_basis": "server_emitted", "granularity": "commit_group"},
        "requests": requests, "spans": spans, "slo": plan.get("slo"),
        "runtime_observations": {"clock_origin_ns": capture.get("clock_origin_ns"),
                                 "ranks": capture.get("rank_observations", []),
                                 "scheduler": capture.get("scheduler_observations")},
    }
    (directory / "server.json").write_bytes(server_raw)
    (directory / "plan.json").write_bytes(plan_raw)
    _save(directory / "trace.json", trace)
    summary = summarize_request_trace(directory / "trace.json")
    _save(directory / "summary.json", summary)
    histogram = Counter()
    for (_, _, accepted), count in all_steps.items():
        histogram[accepted] += count
    return {
        "source_kind": kind, "runtime": runtime, "max_arrival_drift_ns": max_drift,
        "arrivals_within_tolerance": max_drift <= plan["arrival_tolerance_ns"],
        "observed_phase_requests": dict(request_coverage),
        "native_acceptance_histogram": dict(sorted(histogram.items())) if request_coverage["native_speculation"] else None,
        "native_round_distribution": [{"proposed": p, "verified": v, "accepted": a, "count": count}
                                      for (p, v, a), count in sorted(all_steps.items())],
        "runtime_observations_complete": summary["runtime_observations"]["complete"],
        "round_commit_reason": "服务端输出组与后端轮缺少关联" if native_config is not None else None,
        "status_counts": summary["status_counts"],
    }, outputs


def collect_service_traces(plan_path: Path, output: Path, replay: Path | None = None):
    plan, raw = read_plan(plan_path)
    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_bytes(raw)
    modes, output_sequences = {}, {}
    for method in METHODS:
        directory = output / method
        directory.mkdir()
        if replay is None:
            spool = directory / "captured.json"
            log_path = directory / "backend.log"
            with log_path.open("wb") as log:
                worker = subprocess.run([sys.executable, "-m", "simverse.execute.service_trace_vllm",
                                         str(output / "plan.json"), method, str(spool)],
                                        stdout=log, stderr=subprocess.STDOUT)
            if worker.returncode:
                raise ValueError(f"{method} 的服务进程退出码 {worker.returncode}，详见 {log_path}")
            server_raw = spool.read_bytes()
        else:
            server_raw = (replay / method / "server.json").read_bytes()
        modes[method], output_sequences[method] = _package_mode(plan, raw, method, directory, server_raw)
        if replay is None:
            spool.unlink()

    baseline = modes["baseline"]
    result = {
        "run_id": plan["run_id"], "plan_sha256": _sha(raw), "modes": modes,
        "same_source_and_runtime": all(m["source_kind"] == baseline["source_kind"]
                                       and m["runtime"] == baseline["runtime"] for m in modes.values()),
        "arrivals_within_tolerance": all(m["arrivals_within_tolerance"] for m in modes.values()),
        "output_mismatches": {
            mode: [rid for rid, observed in output_sequences[mode].items()
                   if observed != output_sequences["baseline"][rid]]
            for mode in METHODS if mode != "baseline"
        },
    }
    result["paired"] = result["same_source_and_runtime"] and result["arrivals_within_tolerance"]
    _save(output / "comparison.json", result)
    return result
