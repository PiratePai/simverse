"""ComparisonPolicy 与多目标比较（Q4 双资源政策 / Q11 多目标并列）。

  * objectives：并列目标（metric/statistic/scope/time_basis/单位/方向），无默认加权；
  * resource_policy：equal_total_inventory | equal_target_plus_disclosed_delta；
  * ranking_metric_ref：可空的临时排序（切换不改资格与数值）；
  * eligibility：逐谓词 status/reason/affected；OOM/unknown/外推不进排名。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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


def compare_runs(baseline_bundle: Any, candidate_bundles: List[Any],
                 policy: Optional[ComparisonPolicy] = None) -> Dict[str, Any]:
    """并列多目标比较：绝对值 + 相对基线变化 + 资源披露；不选唯一赢家。

    资格：capability 检查中 structure/memory 全 supported 才 eligible；
    不合格保留数值与原因。
    """
    policy = policy or default_policy()
    rows = []

    def _eligible(bundle) -> tuple:
        checks = bundle.capability.get("checks", {})
        method = checks.get("method_recipe_ready")
        if method and method["status"] != "supported":
            return False, f"方法：{method['reason']}"
        structure = checks.get("model_structure_complete")
        if structure and structure["status"] != "supported":
            return False, f"结构缺口：{structure['reason']}"
        for cid in ("model_config_coverage", "model_forward_evidence",
                    "draft_forward_evidence", "model_cpu_forward",
                    "model_gpu_measurement", "deployment_ranking_evidence"):
            chk = checks.get(cid)
            if chk and chk["status"] != "supported":
                return False, f"证据缺口（{cid}）：{chk['reason']}"
        for cid, chk in checks.items():
            if cid.startswith("memory_") and chk["status"] != "supported":
                return False, f"容量：{chk['reason']}"
        stage = checks.get("stage_cost_complete")
        if stage and stage["status"] != "supported":
            return False, f"阶段成本缺口：{stage['reason']}"
        return True, None

    base_elig, base_reason = _eligible(baseline_bundle)
    base_metrics = baseline_bundle.metrics
    for cand in candidate_bundles:
        elig, reason = _eligible(cand)
        row: Dict[str, Any] = {
            "run_id": cand.manifest.run_id,
            "scenario": cand.scenario_snapshot.get("name"),
            "eligible": elig, "ineligible_reason": reason,
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
            row["objectives"][obj.metric_id] = {
                "candidate": val_c, "baseline": val_b,
                "unit": obj.unit, "direction": obj.direction,
                "delta": (val_c - val_b) if (val_c is not None and val_b is not None)
                else None,
                "reason": (cm.reason if cm and cm.value is None else None),
            }
        rows.append(row)
    return {
        "schema_version": "0.2",
        "policy": policy.to_json(),
        "baseline_run_id": baseline_bundle.manifest.run_id,
        "baseline_eligible": base_elig,
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
