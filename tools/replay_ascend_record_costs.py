"""Offline coverage and measurement backlog for fixed Ascend scenarios."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simverse.contracts import digest
from simverse.pipeline import run_v2
from simverse.scenario import ScenarioV2

TARGET = ROOT / "docs/evidence/ascend-record-costs/coverage-plan.json"
Q16 = dict(dense_w=16, expert_w=16, act=16, kv=16, state=32)


def replay():
    scenarios = [
        ("r29-tp2-baseline", "ascend-r29-hccl", 2, 1, "baseline"),
        ("r29-tp2-dflash", "ascend-r29-hccl", 2, 1, "dflash_v1"),
        ("r29-tp2-ep2", "ascend-r29-hccl", 2, 2, "baseline"),
        ("r29-tp8-domain-fallback", "ascend-r29-hccl", 8, 1, "baseline"),
        ("r28-tp2-baseline", "ascend-r28-gemm", 2, 1, "baseline"),
        ("r28-tp8-baseline", "ascend-r28-gemm", 8, 1, "baseline"),
        ("r28-tp8-dflash-verify16", "ascend-r28-gemm", 8, 1, "dflash_v1"),
    ]
    result = {"schema_version": 1, "method": "offline replay, no NPU execution",
              "scope": "conditional observation comparison; fallback costs are analytical diagnostics, not measured bottlenecks",
              "cases": []}
    for name, sid, tp, ep, method in scenarios:
        params = dict(name=name, method_id=method, k=15, block_size=16,
                      quant=Q16, draft_quant=Q16, workload=dict(batch=1, ctx_tokens=4096),
                      deployment=dict(mode="baseline" if method == "baseline" else "colocated",
                                      target_device="ascend_910b3", target_tp=tp, target_ep=ep))
        analytic = run_v2(ScenarioV2(**params)).to_json()
        selected = run_v2(ScenarioV2(**params, cost_records=sid)).to_json()
        assert analytic["work"] == selected["work"]
        assert analytic["cost"]["memory"] == selected["cost"]["memory"]
        report = selected["cost"]["record_cost"]
        result["cases"].append({
            "name": name, "scenario": selected["scenario_snapshot"],
            "record_digest": report["content_digest"],
            "work_digest": digest(selected["work"]), "raw_counts_memory_unchanged": True,
            "matched": report["matched"], "total": report["total"],
            "analytical_tpot_ms": selected["metrics"]["M01_tpot_analytical"]["value"],
            "record_tpot_ms": (selected["metrics"]["M01_tpot_selected"]["value"]
                               if report["matched"] else None),
            "coverage_by_stage": report["stages"],
            "prerequisites": report["prerequisites"],
            "unpriced_components": report["unpriced_components"],
            "measurement_plan": report["measurement_plan"],
        })
    return result


def expected_coverage_plan():
    expected = json.loads(TARGET.read_text(encoding="utf-8"))
    for case in expected["cases"]:
        for item in case["unpriced_components"]:
            item["reason"] = item["reason"].replace(
                "时间线以显式 0 占位，仅为已建模阶段下界。",
                "事件时长保持未知，轮耗时为已计价阶段下界。")
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = replay()
    if args.check:
        if expected_coverage_plan() != result:
            raise SystemExit("Coverage/backlog replay drift; review before updating")
    else:
        TARGET.parent.mkdir(parents=True, exist_ok=True)
        TARGET.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8", newline="\n")
    for case in result["cases"]:
        print(case["name"], f"{case['matched']}/{case['total']}",
              case["analytical_tpot_ms"], "->", case["record_tpot_ms"])


if __name__ == "__main__":
    main()
