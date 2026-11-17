"""Distill reviewed archive observations; no device imports or execution.

These sets are conditional cost experiments, not admitted measured profiles.
R28 and R29 remain separate software/runtime domains.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = Path("assets/cost_records")
SETS = {"ascend-r29-hccl": "ascend_r29_hccl.json",
        "ascend-r28-gemm": "ascend_r28_gemm.json"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distill(root=ROOT):
    def source(path):
        return {"path": path, "sha256": sha(root / path)}

    def read(path):
        return json.loads((root / path).read_text(encoding="utf-8"))

    r29 = "docs/evidence/ascend-910b3-r29/"
    ranks = [read(r29 + f"raw/collectives_rank{rank}.json") for rank in (0, 1)]
    out = {
        "schema_version": 1, "id": "ascend-r29-hccl",
        "label": "R29 双卡 HCCL 记录对照（有条件，非校准）",
        "source_commit": "7de467d604d83cbee5eef76c2d12679356f3cad1",
        "source_kind": "assumed-from-archived-observation",
        "admission": "conditional-only", "measured_profile_admitted": False,
        "domain": {"device": "ascend_910b3", "world_size": 2,
                   "dtype": "bfloat16", "physical_cards": [5, 6],
                   "torch": "2.10.0+cpu", "torch_npu": "2.10.0.post4",
                   "implementation": "torch.distributed/hccl isolated collective",
                   "transport": "unknown", "cann": "unverified",
                   "timing_boundary": ranks[0]["timing_boundary"]},
        "limits": ["限定卡5/6采集条件；仅选用该档视为显式条件假设，未核实际部署拓扑/运行版本",
                   "两rank中位数取较大值是保守建模选择，不是同步窗口的最慢rank实测",
                   "包括host下发与同步摊销，替换整笔collective成本，不再加启动常数",
                   "不能当HCCS线缆带宽；不能外推TP4/8或跨机；不解除部署排名",
                   "邻接尺寸内按字节线性插值；留一尺寸误差为曲线诊断，非独立运行留出验收"],
        "sources": [source(r29 + f"raw/collectives_rank{rank}.json") for rank in (0, 1)]
                   + [source(r29 + "scripts/r29_collectives.py"), source(r29 + "raw/env_identity.json")],
        "curves": {}, "excluded": [],
    }
    for rank, data in enumerate(ranks):
        if data["rank"] != rank or data["world_size"] != 2 or data["errors"]:
            raise ValueError("R29 rank identity/errors invalid")
        if data["dtype"] != "bfloat16" or data["device_name"] != "Ascend910B3":
            raise ValueError("R29 device/dtype invalid")
    for row in ranks[0]["results"]:
        op, n = row["op"], row["numel_per_rank"] * 2
        others = [next(r for r in d["results"] if r["op"] == op
                       and r["numel_per_rank"] * 2 == n) for d in ranks]
        for r in others:
            samples = r["batches_ms"]
            if r["correctness"] != "exact" or len(samples) != 5 or r.get("error"):
                raise ValueError("Unverified collective cannot be distilled")
            if not all(math.isfinite(x) and x > 0 for x in samples):
                raise ValueError("Invalid timing")
            if abs(statistics.median(samples) - r["median_ms"]) > 1e-6:
                raise ValueError("Median does not replay")
        # Key by actual per-rank INPUT bytes, not wire or uniform scale.
        input_bytes = n * 2 if op == "reduce_scatter" else n
        medians = [r["median_ms"] for r in others]
        out["curves"].setdefault(op, []).append({
            "id": f"r29:{op}:{input_bytes}", "input_bytes": input_bytes,
            "output_bytes": n * 2 if op == "all_gather" else n,
            "rank_medians_ms": medians, "rank_batches_ms": [r["batches_ms"] for r in others],
            "selected_ms": max(medians),
            "rank_policy": "max-of-rank-medians-not-joint-window",
        })
    for op, points in out["curves"].items():
        points.sort(key=lambda p: p["input_bytes"])
        errors = []
        for lo, mid, hi in zip(points, points[1:], points[2:]):
            f = (mid["input_bytes"] - lo["input_bytes"]) / (hi["input_bytes"] - lo["input_bytes"])
            estimate = lo["selected_ms"] + f * (hi["selected_ms"] - lo["selected_ms"])
            errors.append(abs(estimate - mid["selected_ms"]) / mid["selected_ms"])
        out.setdefault("curve_diagnostics", {})[op] = {
            "method": "leave-one-interior-size-out; endpoints excluded",
            "count": len(errors), "median_relative_error": statistics.median(errors),
            "max_relative_error": max(errors), "independent_run_holdout": False}

    r28 = "docs/evidence/ascend-910b3/"
    data = read(r28 + "p3_per_op_qwen.json")
    operators = {"in_proj_qkv", "in_proj_z", "in_proj_ba", "out_proj", "moe_router",
                 "q_proj", "k_proj", "v_proj", "o_proj", "lm_head"}
    gemm = {
        "schema_version": 1, "id": "ascend-r28-gemm",
        "label": "R28 独立 BF16 GEMM 记录对照（有条件，非校准）",
        "source_commit": "6a16fbed8e6dc0982ed9abdea0ea1df6c006e429",
        "source_kind": "assumed-from-archived-observation",
        "admission": "conditional-only", "measured_profile_admitted": False,
        "domain": {"device": "ascend_910b3", "model_id": "qwen36_35b_a3b",
                   "dtype": "bfloat16", "torch": "2.9.0+cpu",
                   "torch_npu": "2.9.0.post9", "stack_identity": "collector-declared-incomplete",
                   "implementation": "isolated torch.matmul, contiguous random BF16 operands",
                   "timing_boundary": "host wallclock / iterations, 20 warmup, 5 synchronized batches"},
        "sources": [source(r28 + name) for name in
                    ("p3_per_op_qwen.json", "scripts/p3_per_op.py", "p2_bench_gemm.txt", "env.txt")],
        "limits": ["仅主模型的独立matmul数学/形状匹配；非vLLM融合核、非TP并发实测",
                   "只有min/median/max摘要，无原始五批样本、完整输入清单或kernel版本",
                   "不插值、不外推；缺匹配或边界不同逐项回退；不用于真实profile准入",
                   "R28与R29的软件栈不同，不能拼成同一成本档",
                   "算子留出误差未知；不得用这些单点拟合全局倍率"],
        "operators": [], "excluded": [],
    }
    for row in data["records"]:
        label = row.get("shape_label", "")
        dims = label.split("\u00d7")
        if row["op"] not in operators or row["kind"] != "gemm" or len(dims) != 3:
            gemm["excluded"].append({"op": row["op"], "shape": label,
                                     "reason": "组合/代理/非独立GEMM，缺等价实现证据"})
            continue
        m, k, n = map(int, dims)
        if row["flops"] != 2 * m * k * n or row.get("error"):
            raise ValueError("GEMM FLOPs/shape mismatch")
        gemm["operators"].append({
            "id": f"r28:{row['op']}:{m}:{k}:{n}", "op": row["op"],
            "shape": [m, k, n], "selected_ms": row["median_us"] / 1000,
            "min_ms": row["min_us"] / 1000, "max_ms": row["max_us"] / 1000,
            "sample_count": 5, "samples_available": False,
        })
    for row in data["m_scale"]:
        if row["M"] == 1:
            continue  # independent duplicate; don't pool unrelated windows
        gemm["operators"].append({
            "id": f"r28:m_scale:{row['M']}:2048:1024", "op": "in_proj_qkv",
            "shape": [row["M"], 2048, 1024], "selected_ms": row["median_us"] / 1000,
            "sample_count": 5, "samples_available": False,
        })
    gemm["excluded"].append({"op": "msprof/kernel summaries", "reason":
                             "无逐调用关联/预热剔除与完整身份，不把摘要时长替换独立算子"})
    return {out["id"]: out, gemm["id"]: gemm}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for sid, value in distill().items():
        target = ROOT / DEST / SETS[sid]
        if args.check:
            if json.loads(target.read_text(encoding="utf-8")) != value:
                raise SystemExit(f"Record distillation drift: {sid}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                              encoding="utf-8", newline="\n")
        print(sid, "OK")
    hashes = {name: sha(ROOT / DEST / name) for name in SETS.values()}
    manifest = ROOT / DEST / "manifest.json"
    if args.check:
        if json.loads(manifest.read_text(encoding="utf-8")) != hashes:
            raise SystemExit("Derived manifest drift")
    else:
        manifest.write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
