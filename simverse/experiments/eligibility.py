"""统一部署排名资格判定（R10e/R10f）：管线 M02 与 compare_runs 共用一套。

修复前两套判定各自为政：compare_runs 对缺失的必需检查项视而不见——
清空 ``capability.checks`` 后仍 ``eligible=True``；管线侧只看"存在且非
supported"的检查项，同样不追缺失。

必需证据按场景的模型、方法和部署方式确定。缺失的检查保持 unknown。
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

# 目标端必需检查（所有可排名场景都应产生）
TARGET_CHECKS: tuple = (
    "model_structure_complete", "model_config_coverage",
    "model_forward_evidence", "model_cpu_forward",
    "model_gpu_measurement", "deployment_ranking_evidence",
    "parallel_valid",
)

# 方法/草稿端检查：适用方法 → 必需检查项（baseline 无适用项）
METHOD_SCOPED_CHECKS: Dict[str, tuple] = {
    "dflash_v1": ("draft_forward_evidence", "dflash_serving_integration"),
    "mtp": ("mtp_serving_integration", "method_recipe_ready"),
    "chain_draft": ("method_recipe_ready",),
    "chain_moe_proxy": ("method_recipe_ready",),
}

# 存在即检查（管线按条件添加；出现且非 supported 即拦截）。
# 注：engram_host_lookup_latency 不单列——它与 stage_cost_complete 同源
# （Engram host 查表缺口），后者已按适用域拦截，重复计数只会改变冻结的
# M02 reason 措辞（R27 证据锚点），不放松任何判定。
PRESENT_ONLY_CHECKS: tuple = (
    "ascend_device_recipe", "acceptance_known", "cost_profile_domain",
    "archived_cost_records",
)

MEMORY_PREFIX = "memory_"


def _method_of(scenario: Optional[Mapping]) -> Optional[str]:
    if not scenario:
        return None
    return scenario.get("method_id")


def _stage_cost_applies(scenario: Mapping) -> bool:
    deployment = scenario.get("deployment") or {}
    return (scenario.get("method_id") == "dflash_v1"
            or (scenario.get("model_id") == "deepseek_v41_flash"
                and deployment.get("engram_table_placement", "cpu_offload") == "cpu_offload"))


def _memory_checks(scenario: Mapping) -> tuple:
    mode = (scenario.get("deployment") or {}).get("mode")
    if mode == "colocated":
        return ("memory_colocated_r0",)
    if mode == "baseline":
        return ("memory_target_r0",)
    if mode in ("dt_serial", "dt_pipeline_sim"):
        if scenario.get("draft_id") or scenario.get("method_id") == "dflash_v1":
            return ("memory_target_r0", "memory_draft_r0")
        return ("memory_target_r0",)
    return ()


def eligibility_issues(checks: Mapping[str, Any],
                       scenario: Optional[Mapping] = None) -> List[str]:
    """返回不合格原因列表（空列表 = 可排名）。

    checks：bundle.capability["checks"]（或 CapabilityReport.checks）。
    scenario：scenario_snapshot dict；None 时方法/阶段项退化为存在性检查
    （无法判定适用性时不因缺失而拦截——但目标端核心与容量仍必需）。
    """
    issues: List[str] = []
    checks = checks or {}

    def _fail(cid: str, reason: str) -> None:
        issues.append(f"{reason}" if reason else f"检查 {cid} 未通过")

    # 目标端证据：缺失即 unknown（修复"清空 checks 仍 eligible"）
    for cid in TARGET_CHECKS:
        chk = checks.get(cid)
        if chk is None:
            _fail(cid, f"必需证据缺失（unknown）：{cid}——run 未产生该检查，"
                       "不视为通过")
        elif chk["status"] != "supported":
            _fail(cid, str(chk.get("reason") or f"{cid}={chk['status']}"))

    # 方法/草稿端：按适用域必需；无场景信息（合成 cap/旧调用）时存在即检查
    method = _method_of(scenario)
    if method in METHOD_SCOPED_CHECKS:
        for cid in METHOD_SCOPED_CHECKS[method]:
            chk = checks.get(cid)
            if chk is None:
                _fail(cid, f"必需证据缺失（unknown）：{cid}——{method} 场景应"
                           "产生该检查而 run 未提供")
            elif chk["status"] != "supported":
                _fail(cid, str(chk.get("reason") or f"{cid}={chk['status']}"))
    else:
        present_ids = {cid for cids in METHOD_SCOPED_CHECKS.values()
                       for cid in cids}
        for cid in sorted(present_ids):
            chk = checks.get(cid)
            if chk is not None and chk["status"] != "supported":
                _fail(cid, str(chk.get("reason") or f"{cid}={chk['status']}"))

    stage = checks.get("stage_cost_complete")
    if stage is None and _stage_cost_applies(scenario or {}):
        _fail("stage_cost_complete", "必需证据缺失（unknown）：stage_cost_complete")
    elif stage is not None and stage["status"] != "supported":
        _fail("stage_cost_complete", str(stage.get("reason") or "stage_cost_complete"))

    memory = {cid: chk for cid, chk in checks.items() if cid.startswith(MEMORY_PREFIX)}
    required_memory = _memory_checks(scenario or {})
    for cid in required_memory:
        if cid not in memory:
            _fail(cid, f"必需容量证据缺失（unknown）：{cid}")
    if not required_memory and not memory:
        _fail(MEMORY_PREFIX, "必需证据缺失（unknown）：无 memory_* 容量检查")
    for cid, chk in sorted(memory.items()):
        if chk["status"] != "supported":
            _fail(cid, str(chk.get("reason") or f"{cid}={chk['status']}"))

    # 附加条件：存在即检查（管线按条件添加的条件项）
    for cid in PRESENT_ONLY_CHECKS:
        chk = checks.get(cid)
        if chk is not None and chk["status"] != "supported":
            _fail(cid, str(chk.get("reason") or f"{cid}={chk['status']}"))

    return issues


def is_eligible(checks: Mapping[str, Any],
                scenario: Optional[Mapping] = None) -> bool:
    return not eligibility_issues(checks, scenario)
