"""vLLM server-process worker for matched R07b/R09 captures.

vLLM and torch are optional runtime dependencies for this worker only.
"""
from __future__ import annotations

import argparse
import asyncio
import shlex
import subprocess
import sys
import time
from contextlib import aclosing
from math import isfinite
from pathlib import Path

from .request_trace import METHODS, NS_PER_S
from .service_trace import (SERVER_CLOCK_ID, TARGET_MODEL, VLLM_REVISION, _save, _sha,
                            read_plan, speculative_config)

def _preflight(plan):
    if sys.platform != "linux":
        raise ValueError("实采需要 Linux CUDA 环境")
    import torch
    import vllm

    if not torch.cuda.is_available() or torch.cuda.device_count() != plan["tp"]:
        raise ValueError("可用 CUDA 设备数须等于计划 TP")
    checkout = Path(vllm.__file__).resolve().parents[1]
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("需要从固定提交的 vLLM 源码检出运行") from exc
    if revision != VLLM_REVISION or status:
        raise ValueError(f"需要干净的 vLLM {VLLM_REVISION} 源码检出")
    devices, free_bytes = [], []
    for rank in range(plan["tp"]):
        properties = torch.cuda.get_device_properties(rank)
        free, _ = torch.cuda.mem_get_info(rank)
        if free < plan["minimum_device_memory_bytes"]:
            raise ValueError(f"rank {rank} 的可用显存低于计划门槛")
        free_bytes.append(free)
        devices.append({"name": properties.name, "uuid": str(getattr(properties, "uuid", "")),
                        "total_bytes": properties.total_memory})
    return {"backend_revision": revision, "vllm_version": vllm.__version__,
            "torch_version": str(torch.__version__), "torch_cuda": torch.version.cuda,
            "device": ", ".join(d["name"] for d in devices), "devices": devices}, free_bytes


def _native_timing(stats, epoch_ns, arrival_ns, terminal_ns):
    """Retain request-wall spans only when native events align to the local origin."""
    if stats is None:
        return {}, None
    fields = {key: getattr(stats, key, None) for key in
              ("queued_ts", "scheduled_ts", "first_token_ts", "last_token_ts",
               "num_preemptions", "num_generation_tokens")}
    spans = {}
    for phase, keys in (("prefill", ("scheduled_ts", "first_token_ts")),
                        ("decode", ("first_token_ts", "last_token_ts"))):
        start, end = (fields[key] for key in keys)
        if all(isinstance(v, (float, int)) and isfinite(v) and v > 0 for v in (start, end)):
            lo, hi = (round(v * NS_PER_S) - epoch_ns for v in (start, end))
            if arrival_ns <= lo <= hi <= terminal_ns:
                spans[phase] = {"start_ns": lo, "end_ns": hi}
    return spans, fields


async def capture_requests(engine, params, plan, epoch_ns, now=time.monotonic_ns):
    """Drive requests on one service clock. Output deltas are observed at the emitter."""
    drain = plan["window"]["drain_end_ns"]

    async def capture(request):
        rid = request["request_id"]
        planned = request["planned_arrival_ns"]
        await asyncio.sleep(max(0, epoch_ns + planned - now()) / NS_PER_S)
        arrival = now() - epoch_ns
        row = {"request_id": rid, "prompt_token_ids": request["prompt_token_ids"],
               "arrival_ns": arrival, "chunks": [], "status": "failed",
               "end_reason": "stream ended without final output"}
        stats = None

        async def stream():
            nonlocal stats
            prompt = {"prompt_token_ids": request["prompt_token_ids"]}
            async with aclosing(engine.generate(prompt, params, rid)) as outputs:
                async for output in outputs:
                    observed = now() - epoch_ns
                    stats = output.metrics
                    if not output.outputs and not output.finished:
                        continue
                    if output.prompt_token_ids is not None and list(output.prompt_token_ids) != prompt["prompt_token_ids"]:
                        raise ValueError("vLLM 实际输入与计划 token 不同")
                    if len(output.outputs) != 1:
                        raise ValueError("采集要求每次输出恰有一个序列")
                    completion = output.outputs[0]
                    tokens = list(completion.token_ids)
                    if tokens:
                        row["chunks"].append({"time_ns": observed, "token_ids": tokens})
                    if not output.finished:
                        continue
                    reason = completion.finish_reason
                    row["status"] = {"abort": "cancelled", "stop": "completed",
                                     "length": "completed"}.get(reason, "failed")
                    row["end_reason"] = reason or "backend finished without finish_reason"
                    if completion.stop_reason is not None:
                        row["stop_reason"] = completion.stop_reason
                    if completion.spec_decode_metrics is not None:
                        row["native_speculation"] = completion.spec_decode_metrics.to_dict()
                    row["terminal_ns"] = observed
                    return

        try:
            await asyncio.wait_for(stream(), timeout=max(0, drain - arrival) / NS_PER_S)
        except asyncio.TimeoutError:
            row.update(status="cancelled", end_reason="drain deadline")
        except Exception as exc:
            row.update(status="failed", end_reason=f"{type(exc).__name__}: {exc}")
        if "terminal_ns" not in row:
            row["terminal_ns"] = now() - epoch_ns
        row["phase_spans"], row["native_request_metrics"] = _native_timing(
            stats, epoch_ns, arrival, row["terminal_ns"],
        )
        return row

    ordered = sorted(plan["requests"], key=lambda request: request["planned_arrival_ns"])
    return await asyncio.gather(*(capture(request) for request in ordered))


async def _run(plan, plan_raw, method, command):
    runtime, free_bytes = _preflight(plan)
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM

    spec = speculative_config(method, plan)
    args = AsyncEngineArgs(
        model=TARGET_MODEL, revision=plan["model_revision"],
        tokenizer=TARGET_MODEL, tokenizer_revision=plan["tokenizer_revision"],
        dtype="bfloat16", tensor_parallel_size=plan["tp"],
        seed=plan["sampling"]["seed"], generation_config="vllm",
        enable_expert_parallel=plan["enable_expert_parallel"],
        max_model_len=plan["max_model_len"], max_num_seqs=plan["max_num_seqs"],
        gpu_memory_utilization=plan["gpu_memory_utilization"], enable_prefix_caching=False,
        trust_remote_code=False, speculative_config=spec,
        disable_log_stats=False, per_request_spec_decode_metrics="detailed" if spec else "none",
    )
    engine = AsyncLLM.from_engine_args(args)
    try:
        params = SamplingParams(output_kind=RequestOutputKind.DELTA, **plan["sampling"])
        # Warmup drains before arming the common offered-load schedule.
        warmup_runs = plan.get("warmup_runs", 1)
        timeout_ns = plan.get("warmup_timeout_ns", 300 * NS_PER_S)
        for i in range(warmup_runs):
            warmup_plan = {**plan, "window": {"drain_end_ns": timeout_ns},
                           "requests": [{**plan["requests"][i % len(plan["requests"])],
                                         "planned_arrival_ns": 0, "request_id": f"_warmup-{i}"}]}
            [warmup] = await capture_requests(engine, params, warmup_plan, time.monotonic_ns())
            if warmup["status"] != "completed":
                raise ValueError(f"预热失败：{warmup['end_reason']}")
        # Let the first scheduled tasks arm their timers before the capture origin.
        origin = time.monotonic_ns() + 100_000_000
        rows = await capture_requests(engine, params, plan, origin)
    finally:
        engine.shutdown()
    return {"method": method, "plan_sha256": _sha(plan_raw), "source_kind": "captured",
            "runtime": runtime, "clock_id": SERVER_CLOCK_ID, "clock_origin_ns": origin,
            "preflight_free_bytes": free_bytes, "warmup_runs": warmup_runs, "command": command,
            "speculative_config": spec, "requests": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("method", choices=METHODS)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plan, raw = read_plan(args.plan)
    command = shlex.join([sys.executable, "-m", __spec__.name, *sys.argv[1:]])
    _save(args.output, asyncio.run(_run(plan, raw, args.method, command)))


if __name__ == "__main__":
    main()
