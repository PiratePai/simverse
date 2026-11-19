"""ComparisonPolicy 与多目标比较（Q4 双资源政策 / Q11 多目标并列）。

  * objectives：并列目标（metric/statistic/scope/time_basis/单位/方向），无默认加权；
  * resource_policy：equal_total_inventory | equal_target_plus_disclosed_delta；
  * ranking_metric_ref：可空的临时排序（切换不改资格与数值）；
  * eligibility：统一走 experiments.eligibility（R10e/R10f；与管线 M02 同一
    套按场景适用域的规则——修复清空 capability.checks 仍 eligible、以及
    声明未检查的 same_time_basis/result_kind_compatible）。

R10e/R10f 比较侧规则：
  * 资格：目标端/草稿端（按场景适用域）/方法执行/阶段成本/容量证据全部
    满足才 eligible；必需检查缺失保留 unknown 与具体原因；baseline 只检查
    其适用项；预算 unknown、假设预算、OOM 均不可排名；
  * 口径核对：相对变化（delta）要求双方指标的单位、时间基准、统计量、
    适用范围（role/phase/aggregation）与结果类型相容——不相容时保留双方
    数值与原因、delta=None（新增 objectives[*].compatibility 字段仅补充
    该核对结论；既有键语义不变；旧行为会给出混口径差值，正是修复点）；
  * 未通过资格/口径的结果保留可用数值与原因，不删除行。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .eligibility import eligibility_issues


@dataclass(frozen=True)
class Objective:
    metric_id: str
    statistic: str
    scope: str
    time_basis: str
    unit: str
    direction: str            # minimize | maximize
    domain: str = ""          # 适用域（如 "analytical-estimate 同类可比"）


@dataclass
class ComparisonPolicy:
    policy_id: str
    version: str
    resource_policy: str      # equal_total_inventory | equal_target_plus_disclosed_delta
    objectives: List[Objective]
    allowed_differences: List[str] = field(default_factory=list)
    quality_requirement: str = "未验证质量等价：接受率为假设输入（Q8）"
    ranking_metric_ref: Optional[Dict[str, str]] = None   # 用户临时排序
    eligibility_predicates: List[str] = field(default_factory=lambda: [
        "structure_complete", "config_coverage_complete",
        "forward_evidence_supported", "cpu_forward_executed",
        "gpu_measurement_present", "deployment_evidence_complete",
        "method_recipe_supported", "stage_cost_complete",
        "memory_feasible_known", "same_time_basis",
        "result_kind_compatible"])

    def to_json(self) -> dict:
        return {
            "policy_id": self.policy_id, "version": self.version,
            "resource_policy": self.resource_policy,
            "objectives": [o.__dict__ for o in self.objectives],
            "allowed_differences": list(self.allowed_differences),
            "quality_requirement": self.quality_requirement,
            "ranking_metric_ref": self.ranking_metric_ref,
            "eligibility_predicates": list(self.eligibility_predicates),
        }


DEFAULT_OBJECTIVES = [
    Objective("M01_tpot_analytical", "expectation", "steady-decode/target",
              "committed", "ms", "minimize", "解析估算同类比较（Q2）"),
    Objective("M02_speedup_vs_baseline", "expectation", "steady-decode/target",
              "committed", "ratio", "maximize", "同基线配对"),
    Objective("M06_memory_live", "exact", "per-rank/steady", "committed",
              "B", "minimize", "标称口径（feasible 判定 unknown 单列）"),
]


def default_policy(resource_policy: str = "equal_total_inventory") -> ComparisonPolicy:
    return ComparisonPolicy(
        policy_id=f"default-{resource_policy}", version="0.2",
        resource_policy=resource_policy, objectives=list(DEFAULT_OBJECTIVES))


def _scope_tuple(metric) -> tuple:
    """适用范围可比键：role/phase/aggregation/time_basis（rank/group 等
    单运行身份字段不进比较）。"""
    sc = getattr(metric, "scope", None)
    if sc is None:
        return ()
    return (getattr(sc, "role", None), getattr(sc, "phase", None),
            getattr(sc, "aggregation", None), getattr(sc, "time_basis", None))


def metric_compatibility(baseline_metric, candidate_metric) -> Dict[str, Any]:
    """口径核对：单位/时间基准/统计量/适用范围/结果类型。

    任一维度不一致 → compatible=False + 具体原因（双方数值保留在调用方）。
    任一侧缺失 → compatible=False（无法核对，不给相对变化）。
    """
    if baseline_metric is None or candidate_metric is None:
        return {"compatible": False,
                "reason": "基线或候选指标缺失，无法核对口径"}
    dims = {
        "unit": (baseline_metric.unit, candidate_metric.unit),
        "statistic": (str(getattr(baseline_metric, "statistic", None)),
                      str(getattr(candidate_metric, "statistic", None))),
        "time_basis": (_scope_tuple(baseline_metric)[3:4],
                       _scope_tuple(candidate_metric)[3:4]),
        "scope": (_scope_tuple(baseline_metric)[:3],
                  _scope_tuple(candidate_metric)[:3]),
        "result_kind": (str(getattr(baseline_metric, "result_kind", None)),
                        str(getattr(candidate_metric, "result_kind", None))),
    }
    mismatch = [f"{name}：{a} ≠ {b}"
                for name, (a, b) in dims.items() if a != b]
    if mismatch:
        return {"compatible": False, "reason": "口径不匹配——" + "；".join(mismatch)}
    return {"compatible": True, "reason": None}


def compare_runs(baseline_bundle: Any, candidate_bundles: List[Any],
                 policy: Optional[ComparisonPolicy] = None) -> Dict[str, Any]:
    """并列多目标比较：绝对值 + 相对基线变化 + 资源披露；不选唯一赢家。

    资格与口径规则见模块 docstring；不合格/口径不匹配时保留数值与原因。
    """
    policy = policy or default_policy()
    rows = []
    base_scenario = getattr(baseline_bundle, "scenario_snapshot", None) or {}
    base_elig_reasons = eligibility_issues(
        baseline_bundle.capability.get("checks", {}), base_scenario)
    base_elig = not base_elig_reasons
    base_metrics = baseline_bundle.metrics
    for cand in candidate_bundles:
        cand_scenario = getattr(cand, "scenario_snapshot", None) or {}
        reasons = eligibility_issues(cand.capability.get("checks", {}),
                                     cand_scenario)
        if base_elig_reasons:
            reasons.append("基线证据未通过：" + "；".join(base_elig_reasons))
        row: Dict[str, Any] = {
            "run_id": cand.manifest.run_id,
            "scenario": cand.scenario_snapshot.get("name"),
            "objectives": {},
            "resource_delta": _resource_delta(baseline_bundle, cand),
        }
        for obj in policy.objectives:
            bm = base_metrics.get(obj.metric_id)
            cm = cand.metrics.get(obj.metric_id)
            if obj.metric_id == "M06_memory_live":
                cm = _first_memory(cand)
                bm = _first_memory(baseline_bundle)
            val_c = cm.value if cm else None
            val_b = bm.value if bm else None
            compat = metric_compatibility(bm, cm)
            if not compat["compatible"]:
                reasons.append(f"{obj.metric_id}：{compat['reason']}")
            elif val_c is None or val_b is None:
                reasons.append(f"{obj.metric_id}：{(cm.reason if val_c is None else bm.reason) or '指标值未知'}")
            delta = (val_c - val_b) if (val_c is not None and val_b is not None
                                        and compat["compatible"]) else None
            row["objectives"][obj.metric_id] = {
                "candidate": val_c, "baseline": val_b,
                "unit": obj.unit, "direction": obj.direction,
                "delta": delta,
                "reason": (cm.reason if cm and cm.value is None else None),
                # 新增（R10e/f）：口径核对结论；不改变既有键的语义。
                # 旧行为在口径不匹配时仍给 delta（混口径差值），已修复。
                "compatibility": compat,
            }
        row["eligible"] = not reasons
        row["ineligible_reason"] = "；".join(reasons) if reasons else None
        rows.append(row)
    return {
        "schema_version": "0.2",
        "policy": policy.to_json(),
        "baseline_run_id": baseline_bundle.manifest.run_id,
        "baseline_eligible": base_elig,
        # baseline 适用项未满足时列出原因（baseline 只检查其适用项）
        "baseline_ineligible_reason": ("；".join(base_elig_reasons)
                                       if base_elig_reasons else None),
        "rows": rows,
        "note": "多目标并列（Q11）：不默认赢家；排序需显式 ranking_metric_ref",
    }


def _first_memory(bundle) -> Any:
    for k, v in bundle.metrics.items():
        if k.startswith("M06_memory"):
            return v
    return None


def _resource_delta(base, cand) -> Dict[str, Any]:
    """资源披露（Q4）：设备数量向量差，不称扩容收益。"""
    def _res(b):
        d = b.scenario_snapshot["deployment"]
        return {"target": f"{d['target_tp']}×{d['target_device']}",
                "draft": (f"{d['draft_tp']}×{d['draft_device']}"
                          if b.scenario_snapshot.get("draft_id") else None)}
    return {"baseline": _res(base), "candidate": _res(cand)}
