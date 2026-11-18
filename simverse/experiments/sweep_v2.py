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

from dataclasses import replace
from typing import Any, Dict, List, Optional

from ..pipeline import run_v2
from ..scenario import ScenarioV2

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
        # 口径选择：默认=选用成本（缺失回退解析并明示）；显式=严格该口径
        chosen_key = metric_key
        fallback_reason = None
        if chosen_key is None:
            if bundle.metrics.get(DEFAULT_METRIC) is not None:
                chosen_key = DEFAULT_METRIC
            elif bundle.metrics.get(FALLBACK_METRIC) is not None:
                chosen_key = FALLBACK_METRIC
                fallback_reason = (f"缺选用指标 {DEFAULT_METRIC}（旧结果/未走成本"
                                   "选择路径），回退解析口径")
                sweep_fallback = fallback_reason
            else:
                chosen_key = FALLBACK_METRIC  # 两者皆缺：保留空值与原因
        met = bundle.metrics.get(chosen_key)
        baseline_key = BASELINE_PAIR[chosen_key]
        baseline = bundle.metrics.get(baseline_key)
        if baseline is None and chosen_key == DEFAULT_METRIC and \
                bundle.metrics.get(BASELINE_PAIR[FALLBACK_METRIC]) is not None:
            # 选用候选存在但选用基线缺失（旧结果）：基线回退解析并明示
            baseline = bundle.metrics[BASELINE_PAIR[FALLBACK_METRIC]]
            fallback_reason = (f"缺 {baseline_key}，基线回退解析口径"
                               + ("；" + fallback_reason if fallback_reason else ""))
            sweep_fallback = fallback_reason
        if chosen_key not in used_keys:
            used_keys.append(chosen_key)
        points.append({
            "value": v,
            "run_id": bundle.manifest.run_id,
            "status": bundle.manifest.status,
            "tpot_ms": met.value if met else None,
            "reason": met.reason if met else f"指标缺失：{chosen_key}",
            "metric_key": chosen_key,
            "baseline_metric_key": baseline_key,
            "metric_source": selection.get("source") or "unknown",
            "record_coverage": {"matched": record.get("matched", 0),
                                "total": record.get("total", 0)},
            "baseline_tpot_ms": baseline.value if baseline else None,
            "speedup": (baseline.value / met.value)
            if (met and met.value and baseline and baseline.value) else None,
            "fallback_reason": fallback_reason,
        })
    # 扫描级元数据反映实际使用口径（主口径=扫描序首个出现；混用时另列全表）
    primary = used_keys[0] if used_keys else DEFAULT_METRIC
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
    """speedup=1 交点（保留单值接口；R10e 确定性规则）。

    按原始扫描顺序处理，只对**相邻**且口径可比的点插值：
      * 无效点（speedup 缺失/失败）截断区间——不跨未知区间插值；
      * 口径不可比（相邻点 metric_key 不同）同样截断；
      * 扫描顺序上第一个恰好等于 1 的点直接返回其值（连续等于 1 取首个）；
      * 否则第一个相邻符号翻转区间线性插值；多个交点取扫描序首个；
      * 无交点返回 None（不造数）。
    """
    points = scan.get("points") or []
    xs: List[float] = []
    ys: List[Optional[float]] = []
    keys: List[Any] = []
    for p in points:
        xs.append(float(p["value"]))
        ys.append(p.get("speedup"))
        keys.append(p.get("metric_key"))
    n = len(xs)
    for i in range(n):
        if ys[i] is not None and ys[i] == 1.0:
            return xs[i]
        if i == 0 or ys[i - 1] is None or ys[i] is None:
            continue
        if keys[i - 1] != keys[i]:
            continue  # 口径不可比：截断
        f0, f1 = ys[i - 1] - 1.0, ys[i] - 1.0
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
