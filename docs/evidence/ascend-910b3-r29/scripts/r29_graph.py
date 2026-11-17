"""R29-A4：graph 驻留分阶段复核（修正旧 p1a 的归因错误）。

旧结论的问题：`before−after` 与 `before−baseline` 同为 35,651,584 B，
即预热后 baseline 与 after 相等，可复算净差为 0，却被写成"graph 额外驻留"。
本轮把 eager 预热、baseline、capture、replay×1、replay×10、release 各自快照，
并把每段差值与"相对 baseline"的差值分别列出，不做跨阶段混算。
"""

from __future__ import annotations

import json
import sys

import torch
import torch_npu  # noqa: F401

from r29_common import base_record, write_json


def snap(dev: int, stage: str) -> dict:
    free, total = torch.npu.mem_get_info(dev)
    stats = torch.npu.memory_stats(dev) or {}
    return {
        "stage": stage,
        "free_bytes": int(free),
        "total_bytes": int(total),
        "used_bytes": int(total) - int(free),
        "allocated_bytes": int(torch.npu.memory_allocated(dev)),
        "reserved_bytes": int(torch.npu.memory_reserved(dev)),
        "active_bytes": int(stats.get("active_bytes.all.current", -1)),
    }


def delta(a: dict, b: dict, label: str) -> dict:
    return {
        "label": label,
        "free_delta_bytes": a["free_bytes"] - b["free_bytes"],
        "allocated_delta_bytes": b["allocated_bytes"] - a["allocated_bytes"],
        "reserved_delta_bytes": b["reserved_bytes"] - a["reserved_bytes"],
        "from": a["stage"],
        "to": b["stage"],
    }


def main() -> int:
    dev = 0
    torch.npu.set_device(dev)
    rec = base_record("r29_graph_pools")
    rec["device"] = torch.npu.get_device_name(dev)
    rec["workload"] = "20 x (relu(x @ w)) with 2048x2048 bf16 weights on npu:0"
    rec["snapshots"] = []
    rec["deltas"] = []
    rec["errors"] = []
    rec["boundary"] = (
        "allocator/free snapshots across phases of one process; "
        "not a full-model peak budget and not comparable to ACL snapshots"
    )

    s = snap(dev, "startup")
    rec["snapshots"].append(s)

    w = torch.randn(2048, 2048, dtype=torch.bfloat16, device=f"npu:{dev}")
    x = torch.randn(2048, 2048, dtype=torch.bfloat16, device=f"npu:{dev}")

    def step(y):
        for _ in range(20):
            y = torch.relu(y @ w)
        return y

    y = step(x)
    torch.npu.synchronize()
    rec["snapshots"].append(snap(dev, "eager_after_first_step"))

    for _ in range(5):
        y = step(x)
    torch.npu.synchronize()
    rec["snapshots"].append(snap(dev, "eager_after_warmup"))

    torch.npu.empty_cache()
    rec["snapshots"].append(snap(dev, "baseline_after_warmup_and_empty_cache"))

    graph_supported = hasattr(torch.npu, "NPUGraph") and hasattr(torch.npu, "graph")
    rec["graph_api_present"] = graph_supported
    rec["graph_api"] = "torch.npu.NPUGraph + torch.npu.graph(...) context"
    g = None
    static_in = x.clone()
    static_out = None
    try:
        g = torch.npu.NPUGraph()
        with torch.npu.graph(g):
            static_out = step(static_in)
        torch.npu.synchronize()
        rec["snapshots"].append(snap(dev, "after_capture"))
        g.replay()
        torch.npu.synchronize()
        rec["snapshots"].append(snap(dev, "after_replay_1"))
        for _ in range(9):
            g.replay()
        torch.npu.synchronize()
        rec["snapshots"].append(snap(dev, "after_replay_10"))
    except Exception as exc:  # noqa: BLE001
        rec["errors"].append({"stage": "graph_capture_or_replay", "error": f"{type(exc).__name__}: {exc}"})
        rec["graph_api_present"] = graph_supported and False

    snapshots = rec["snapshots"]
    by_stage = {s["stage"]: s for s in snapshots}
    baseline = by_stage.get("baseline_after_warmup_and_empty_cache")
    startup = by_stage["startup"]
    if baseline:
        rec["deltas"].append(delta(startup, baseline, "eager_net_vs_startup"))
        for stage in ("after_capture", "after_replay_1", "after_replay_10"):
            if stage in by_stage:
                rec["deltas"].append(delta(baseline, by_stage[stage], f"{stage}_vs_baseline"))
        if "after_capture" in by_stage:
            rec["deltas"].append(delta(by_stage["eager_after_warmup"], by_stage["after_capture"], "capture_vs_warmup"))

    # 旧报告的关键可复算等式：预热后 baseline 与 after 是否相等
    if "eager_after_warmup" in by_stage and "after_capture" in by_stage:
        warm, cap = by_stage["eager_after_warmup"], by_stage["after_capture"]
        rec["old_report_recompute"] = {
            "warmup_free": warm["free_bytes"],
            "capture_free": cap["free_bytes"],
            "warmup_minus_capture_free": warm["free_bytes"] - cap["free_bytes"],
            "note": "旧 p1a 中 before−after 与 before−baseline 同值；这里给出可复算的等价量",
        }

    if g is not None:
        try:
            del g, static_out, static_in
            torch.npu.synchronize()
            torch.npu.empty_cache()
            rec["snapshots"].append(snap(dev, "after_release_and_empty_cache"))
        except Exception as exc:  # noqa: BLE001
            rec["errors"].append({"stage": "release", "error": f"{type(exc).__name__}: {exc}"})

    path = write_json("graph_pools.json", rec)
    print(json.dumps({"wrote": str(path), "stages": [s["stage"] for s in rec["snapshots"]], "errors": len(rec["errors"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
