"""H200 校准闭环固定场景前后对比（docs/evidence/h200_calib_compare.json 生成器）。

可重放：qwen3_30b_a3b baseline decode H200 TP1 BF16，B/ctx 9 点网格，
analytic=T0 roofline 解析、selected=实测组合（measured+fallback+gap）。
用法：py -3.12 tools/h200_calib_compare.py [--out docs/evidence/h200_calib_compare.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from simverse.pipeline import run_v2  # noqa: E402
from simverse.scenario import ScenarioV2  # noqa: E402

SET_ID = "h200-qwen3-30b-a3b-tiny-op-v1"
GRID = [(1, 512), (1, 2048), (1, 4095), (8, 2048), (8, 4095), (32, 512),
        (32, 2048), (32, 4095), (8, 512)]


def scenario(batch: int, ctx: int) -> ScenarioV2:
    return ScenarioV2.from_json({
        "schema_version": "0.2", "name": f"h200-calib-compare-B{batch}-c{ctx}",
        "model_id": "qwen3_30b_a3b", "draft_id": None, "method_id": "baseline",
        "deployment": {"mode": "baseline", "target_device": "h200_sxm",
                       "target_tp": 1, "draft_device": "l40", "draft_tp": 1,
                       "link": "pcie5"},
        "workload": {"batch": batch, "ctx_tokens": ctx},
        "acceptance": {"kind": "geometric", "alpha": 0.75},
        "quant": {"dense_w": 16, "expert_w": 16, "act": 16, "kv": 16, "state": 32},
        "draft_quant": {"dense_w": 16, "expert_w": 16, "act": 16, "kv": 16,
                        "state": 32},
        "cost_profile": SET_ID, "seed": 0})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "docs" / "evidence"
                                         / "h200_calib_compare.json"))
    args = ap.parse_args()
    points = []
    for batch, ctx in GRID:
        b = run_v2(scenario(batch, ctx)).to_json()
        cal = b["cost"]["calibration"]["stages"]["target_decode"]
        analytic = b["metrics"]["M01_tpot_analytical"]["value"]
        selected = b["metrics"]["M01_tpot_profile_calibrated"]["value"]
        points.append({
            "B": batch, "ctx": ctx,
            "analytic_ms": analytic, "selected_ms": selected,
            "measured_ms": cal["measured_ms"], "fallback_ms": cal["fallback_ms"],
            "gap_ms": cal["gap_allowance_ms"], "gap_regions": cal["gap_regions"],
            "kernels": cal["kernels"],
            "coverage_rows": cal["coverage_rows"],
            "coverage_time_frac": cal["coverage_time_frac"],
            "ratio_selected_over_analytic": round(selected / analytic, 2),
        })
    Path(args.out).write_text(
        json.dumps(points, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"OK {len(points)} 点 → {args.out}")
    for p in points:
        print(f"  B{p['B']:>3} ctx{p['ctx']:>4}: analytic {p['analytic_ms']:.3f}"
              f" → selected {p['selected_ms']:.3f}"
              f"（measured {p['measured_ms']:.3f} + fb {p['fallback_ms']:.3f}"
              f" + gap {p['gap_ms']:.3f}×{p['gap_regions']}r）"
              f" ratio {p['ratio_selected_over_analytic']}×"
              f" time-coverage {p['coverage_time_frac']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
