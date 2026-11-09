"""参数扫描 v0.2（E20 修复）：不可变设备/链路快照注入，不触碰全局注册表。

每个扫描点 = 独立 ScenarioV2 → 独立 run_v2 → 独立 run_id；谱系（base digest +
覆盖字段）随行输出。扫描值域显式登记（Q8 假设，来源标注演示默认）。
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional

from ..pipeline import run_v2
from ..scenario import ScenarioV2


def variant(base: ScenarioV2, **over) -> ScenarioV2:
    """生成派生场景：字段级覆盖（deployment/workload/acceptance 为 dict 合并）。"""
    data = base.to_json()
    for key, value in over.items():
        if key in ("deployment", "workload", "acceptance") and isinstance(value, dict):
            data[key] = {**data[key], **value}
        else:
            data[key] = value
    return ScenarioV2.from_json(data)


def sweep_1d(base: ScenarioV2, field_path: str, values: List[Any],
             metric_key: str = "M01_tpot_analytical") -> Dict[str, Any]:
    """一维扫描。field_path 用点号（acceptance.alpha / workload.batch / ...）。"""
    points = []
    for v in values:
        if field_path == "acceptance.alpha":
            sc = variant(base, acceptance={"kind": "geometric", "alpha": float(v)})
        elif field_path == "workload.batch":
            sc = variant(base, workload={"batch": int(v)})
        elif field_path == "workload.ctx_tokens":
            sc = variant(base, workload={"ctx_tokens": int(v)})
        elif field_path == "k":
            sc = variant(base, k=int(v))
        elif field_path == "deployment.target_tp":
            sc = variant(base, deployment={"target_tp": int(v)})
        elif field_path == "block_size":
            sc = variant(base, block_size=int(v))
        else:
            raise ValueError(f"未知扫描维度 {field_path}")
        bundle = run_v2(sc)
        met = bundle.metrics.get(metric_key)
        baseline = bundle.metrics.get("M01_tpot_baseline")
        points.append({
            "value": v,
            "run_id": bundle.manifest.run_id,
            "status": bundle.manifest.status,
            "tpot_ms": met.value if met else None,
            "reason": met.reason if met else None,
            "baseline_tpot_ms": baseline.value if baseline else None,
            "speedup": (baseline.value / met.value)
            if (met and met.value and baseline and baseline.value) else None,
        })
    return {"schema_version": "0.2", "field": field_path,
            "base_digest": base.scenario_digest(),
            "metric_key": metric_key, "points": points}


def find_crossover(scan: dict) -> Optional[float]:
    """speedup=1 交点（线性插值；无交点返回 None——不造数）。"""
    xs, ys = [], []
    for p in scan["points"]:
        if p["speedup"] is not None:
            xs.append(float(p["value"]))
            ys.append(p["speedup"])
    for i in range(1, len(xs)):
        f0, f1 = ys[i - 1] - 1.0, ys[i] - 1.0
        if f0 == 0:
            return xs[i - 1]
        if f0 * f1 < 0:
            return xs[i - 1] + f0 / (f0 - f1) * (xs[i] - xs[i - 1])
    return None


DEFAULT_SWEEP_RANGES = {
    # Q8：显式假设扫描；数值为可编辑演示默认（非用户确认/实测）
    "acceptance.alpha": [round(0.3 + 0.05 * i, 2) for i in range(13)],
    "workload.batch": [1, 2, 4, 8, 16, 32],
    "workload.ctx_tokens": [4096, 8192, 16384, 32768, 65536, 131072],
    "block_size": [8, 16],
}


def sweep_suite(base: ScenarioV2) -> Dict[str, Any]:
    out: Dict[str, Any] = {"schema_version": "0.2",
                           "base_digest": base.scenario_digest(),
                           "assumption": "扫描值域为演示默认（Q8 hypothesis）；"
                                         "每点独立 run_id"}
    for field, values in DEFAULT_SWEEP_RANGES.items():
        out[field] = sweep_1d(base, field, values)
    out["breakevens"] = {f: find_crossover(out[f]) for f in DEFAULT_SWEEP_RANGES}
    return out
