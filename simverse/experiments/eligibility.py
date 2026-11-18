"""统一部署排名资格判定（R10e/R10f）：管线 M02 与 compare_runs 共用一套。

修复前两套判定各自为政：compare_runs 对缺失的必需检查项视而不见——
清空 ``capability.checks`` 后仍 ``eligible=True``；管线侧只看"存在且非
supported"的检查项，同样不追缺失。

规则（单一事实源，双方按场景适用域取用）：
  * 目标端证据：结构完整、config 覆盖、forward 证据、CPU 执行、GPU 实测、
    部署排名证据、并行配方合法——全部必需，缺失即 unknown 并给出具体
    检查项（不静默跳过）；
  * 草稿端证据：仅当场景带草稿（dflash_v1）时必需 draft_forward_evidence
    与 dflash_serving_integration；mtp 方法对应 mtp_serving_integration
    与 method_recipe_ready；baseline 只检查其适用项（无草稿/方法检查）；
  * 方法执行：method_recipe_ready（chain/mtp 内部研究方法）；
  * 阶段成本：stage_cost_complete——按场景适用域判定（V4.1+cpu_offload
    Engram host 查表时管线必产生该检查；无场景信息时按存在性检查）；
  * 容量：至少存在一条 memory_* 检查且全部为 supported——预算 unknown、
    假设预算（approximated/feasible-conditional）与 OOM（unsupported）
    均不可获得部署排名资格；
  * 附加条件（存在即检查）：ascend_device_recipe、acceptance_known、
    cost_profile_domain、archived_cost_records。
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


def _stage_cost_applies(scenario: Optional[Mapping]) -> Optional[bool]:
    """stage_cost_complete 的适用域：V4.1 + Engram 表 host 驻留（管线
    唯一添加该检查的路径）。无场景信息时返回 None（按存在性检查）。"""
    if not scenario:
        return None
    dep = scenario.get("deployment") or {}
    return (scenario.get("model_id") == "deepseek_v41_flash"
            and dep.get("engram_table_placement") == "cpu_offload")


def required_checks(scenario: Optional[Mapping] = None) -> List[str]:
    """按场景适用域列出必需检查项（含 memory 存在性占位）。"""
    checks = list(TARGET_CHECKS)
    method = _method_of(scenario)
    if method in METHOD_SCOPED_CHECKS:
        checks.extend(METHOD_SCOPED_CHECKS[method])
    if _stage_cost_applies(scenario):
        checks.append("stage_cost_complete")
    return checks


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

    # 阶段成本：适用域内必需；域外/未知按存在性
    stage_applies = _stage_cost_applies(scenario)
    stage = checks.get("stage_cost_complete")
    if stage is None:
        if stage_applies:
            _fail("stage_cost_complete",
                  "必需证据缺失（unknown）：stage_cost_complete——V4.1 Engram "
                  "host 查表场景应产生该检查而 run 未提供")
    elif stage["status"] != "supported":
        _fail("stage_cost_complete",
              str(stage.get("reason") or "stage_cost_complete"))

    # 容量：memory_* 至少一条且全部 supported（预算 unknown/假设/OOM 均拦截）
    memory = {cid: chk for cid, chk in checks.items()
              if cid.startswith(MEMORY_PREFIX)}
    if not memory:
        issues.append("必需证据缺失（unknown）：无任何 memory_* 容量检查——"
                      "可部署性未判定（预算/峰值证据缺失），不得排名")
    else:
        for cid, chk in sorted(memory.items()):
            if chk["status"] != "supported":
                _fail(cid, str(chk.get("reason") or f"{cid}={chk['status']}"))

    # 附加条件：存在即检查（管线按条件添加的条件项）
    for cid in PRESENT_ONLY_CHECKS:
        chk = checks.get(cid)
        if chk is not None and chk["status"] not in ("supported",):
            _fail(cid, str(chk.get("reason") or f"{cid}={chk['status']}"))

    return issues


def is_eligible(checks: Mapping[str, Any],
                scenario: Optional[Mapping] = None) -> bool:
    return not eligibility_issues(checks, scenario)
