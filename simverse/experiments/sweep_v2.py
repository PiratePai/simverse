"""参数扫描 v0.2（E20 修复 + R10e 指标口径修复）：不可变设备/链路快照
注入，不触碰全局注册表。

每个扫描点 = 独立 ScenarioV2 → 独立 run_v2 → 独立 run_id；谱系（base digest +
覆盖字段）随行输出。扫描值域显式登记（Q8 假设，来源标注演示默认）。

R10e 指标口径规则（修复前：显式请求 ``M01_tpot_analytical`` 仍被换成
``M01_tpot_selected``，baseline/speedup 口径混搭）：
  * ``metric_key=None``（默认）：用选用成本 ``M01_tpot_selected``，配对
    ``M01_baseline_selected``；旧结果缺选用指标时回退解析口径并在点级
    与扫描级明示 ``fallback_reason``（兼容缺 selected 的历史 bundle）；
  * 显式请求某指标：候选、基线、speedup 全部按该口径配对（analytical
    ↔ ``M01_tpot_baseline``；selected ↔ ``M01_baseline_selected``）；
    指标缺失时保留空值与原因，不静默换口径；
  * 点级保留 run_id、实际口径（``metric_key``/``baseline_metric_key``）、
    成本来源（``metric_source``）与失败原因；扫描级 ``metric_key`` 反映
    实际使用口径（默认回退时为解析口径），另有 ``requested_metric_key``
    与 ``fallback_reason``。
"""
from __future__ import annotations

from math import isfinite
from typing import Any, Dict, List, Optional

from ..pipeline import run_v2
from ..scenario import ScenarioV2
from .compare import metric_compatibility

# 候选指标 → 同口径基线指标（speedup 的分子分母必须同口径）
BASELINE_PAIR = {
    "M01_tpot_selected": "M01_baseline_selected",
    "M01_tpot_analytical": "M01_tpot_baseline",
}
# 默认（metric_key=None）时的首选与回退口径
DEFAULT_METRIC = "M01_tpot_selected"
FALLBACK_METRIC = "M01_tpot_analytical"


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
             metric_key: Optional[str] = None) -> Dict[str, Any]:
    """一维扫描。field_path 用点号（acceptance.alpha / workload.batch / ...）。

    metric_key：None=默认（选用成本，缺则回退解析并明示）；显式传指标名
    则严格按该口径计算候选/基线/speedup。
    """
    if metric_key is not None and metric_key not in BASELINE_PAIR:
        raise ValueError(
            f"未知/无基线配对的扫描指标 {metric_key}；可用：{sorted(BASELINE_PAIR)}")
    points = []
    used_keys: List[str] = []
    sweep_fallback = None
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
        record = bundle.cost.get("record_cost") or {}
        selection = bundle.cost.get("selection") or {}
        metrics = bundle.metrics
        chosen_key = metric_key or DEFAULT_METRIC
        fallback_reason = None
        if metric_key is None:
            missing = [key for key in (DEFAULT_METRIC, BASELINE_PAIR[DEFAULT_METRIC])
                       if metrics.get(key) is None]
            if missing and all(metrics.get(key) is not None for key in
                               (FALLBACK_METRIC, BASELINE_PAIR[FALLBACK_METRIC])):
                chosen_key = FALLBACK_METRIC
                fallback_reason = f"缺选用指标 {', '.join(missing)}，整对回退解析口径"
                sweep_fallback = fallback_reason
        met = metrics.get(chosen_key)
        baseline_key = BASELINE_PAIR[chosen_key]
        baseline = metrics.get(baseline_key)
        source = ("analytic" if chosen_key == FALLBACK_METRIC
                  else selection.get("source", "unknown"))
        source_id = selection.get("source_id") if source != "analytic" else None
        value = met.value if met is not None else None
        base_value = baseline.value if baseline is not None else None
        reason = met.reason if met is not None else f"指标缺失：{chosen_key}"
        speedup = None
        if baseline is None:
            reason = reason or f"指标缺失：{baseline_key}"
        elif base_value is None:
            reason = reason or baseline.reason
        elif value is not None:
            compatible = metric_compatibility(baseline, met)
            if not compatible["compatible"]:
                reason = compatible["reason"]
            elif value > 0 and base_value > 0 and bundle.manifest.status == "completed":
                speedup = base_value / value
            else:
                reason = reason or "扫描点未完成或耗时不为正"
        if chosen_key not in used_keys:
            used_keys.append(chosen_key)
        points.append({
            "value": v,
            "run_id": bundle.manifest.run_id,
            "status": bundle.manifest.status,
            "tpot_ms": value,
            "reason": reason,
            "metric_key": chosen_key,
            "baseline_metric_key": baseline_key,
            "metric_source": source,
            "metric_source_id": source_id,
            "metric_time_basis": met.scope.time_basis if met is not None else None,
            "metric_result_kind": met.result_kind.value if met is not None else None,
            "record_coverage": {"matched": record.get("matched", 0),
                                "total": record.get("total", 0)},
            "baseline_tpot_ms": base_value,
            "speedup": speedup,
            "fallback_reason": fallback_reason,
        })
    # 扫描级元数据反映实际使用口径（主口径=扫描序首个出现；混用时另列全表）
    primary = used_keys[0] if used_keys else (metric_key or DEFAULT_METRIC)
    out: Dict[str, Any] = {
        "schema_version": "0.2", "field": field_path,
        "base_digest": base.scenario_digest(),
        "metric_key": primary,
        "requested_metric_key": metric_key or "default:" + DEFAULT_METRIC,
        "points": points,
    }
    if len(used_keys) > 1:
        out["metric_keys_used"] = list(used_keys)
    if sweep_fallback:
        out["fallback_reason"] = sweep_fallback
    return out


def find_crossover(scan: dict) -> Optional[float]:
    """返回扫描顺序中第一个已知交点或相邻可比点间的插值。"""
    basis_fields = ("metric_key", "baseline_metric_key", "metric_source",
                    "metric_source_id", "metric_time_basis", "metric_result_kind")
    previous = None
    for point in scan.get("points") or []:
        speedup = point.get("speedup")
        if (point.get("status", "completed") != "completed"
                or speedup is None or not isfinite(speedup)):
            previous = None
            continue
        value = float(point["value"])
        basis = tuple(point.get(field) for field in basis_fields)
        delta = speedup - 1.0
        if delta == 0:
            return value
        if previous is not None:
            prev_value, prev_delta, prev_basis = previous
            if prev_basis == basis and prev_delta * delta < 0:
                return prev_value + prev_delta / (prev_delta - delta) * (value - prev_value)
        previous = value, delta, basis
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
