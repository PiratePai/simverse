# -*- coding: utf-8 -*-
"""R03a/R03b 预算输入契约与可部署性判定测试。

负例覆盖：unknown/oom/feasible/feasible-conditional 四态、非法输入、
开销不重复扣除、假设预算不进实际部署排名、已知下界超标称即 OOM。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simverse.lowering.memory import (  # noqa: E402
    Allocation, DeviceBudget, RankMemory)
from simverse.scenario import BudgetInput, DeploymentSpec, ScenarioV2  # noqa: E402

GB = 1024 ** 3


def _budget(**kw):
    base = dict(device_id="h100_sxm", nominal_bytes=80 * GB)
    base.update(kw)
    return DeviceBudget(**base)


def _rm(budget, bytes_=1000, unknown=None):
    rm = RankMemory("r", "h100_sxm", budget)
    rm.add(Allocation("w", "weights", "target", bytes_, lifecycle="resident"))
    rm.unknown_components = list(unknown or [])
    return rm


# ---------------------------------------------------------------------------
# R03a：预算输入契约
# ---------------------------------------------------------------------------


def test_budget_input_defaults_unknown():
    b = BudgetInput().validate()
    assert b.allocatable_bytes is None and b.allocatable_kind == "unknown"
    assert b.extra_overhead_bytes == 0


def test_budget_input_rejects_illegal():
    with pytest.raises(ValueError):
        BudgetInput(allocatable_bytes=-1).validate()
    with pytest.raises(ValueError):
        BudgetInput(allocatable_bytes=10, allocatable_kind="measured").validate()
    with pytest.raises(ValueError, match="measured 预算必须带"):
        BudgetInput(allocatable_bytes=10, allocatable_kind="measured",
                    ).validate()
    with pytest.raises(ValueError):
        BudgetInput(allocatable_bytes=10, allocatable_kind="hypnosis").validate()
    with pytest.raises(ValueError):
        BudgetInput(extra_overhead_bytes=5).validate()      # 无来源
    with pytest.raises(ValueError):
        BudgetInput(allocatable_kind="measured",
                    allocatable_source="mem_get_info").validate()  # 无数值


def test_scenario_budget_serialization_and_identity():
    a = ScenarioV2()
    b = ScenarioV2(deployment=DeploymentSpec(
        budget=BudgetInput(allocatable_bytes=70 * GB,
                           allocatable_kind="hypothesis",
                           allocatable_source="test")))
    a.validate()
    b.validate()
    assert a.scenario_digest() != b.scenario_digest()
    # JSON 往返保持预算字段
    rt = ScenarioV2.from_json(b.to_json())
    assert rt.deployment.budget.allocatable_bytes == 70 * GB
    assert rt.deployment.budget.allocatable_kind == "hypothesis"


def test_pipeline_rejects_over_nominal_budget():
    from simverse.pipeline import run_v2
    with pytest.raises(ValueError, match="标称容量"):
        run_v2(ScenarioV2(deployment=DeploymentSpec(
            budget=BudgetInput(allocatable_bytes=200 * GB,
                               allocatable_kind="hypothesis",
                               allocatable_source="x"))))


# ---------------------------------------------------------------------------
# R03b：四态判定（单元层，峰值可控）
# ---------------------------------------------------------------------------


def test_unknown_without_allocatable():
    rep = _rm(_budget()).peak_vs_budget()
    assert rep["status"] == "unknown"
    assert any("可分配预算未知" in r for r in rep["reasons"])


def test_oom_from_nominal_ceiling_without_budget():
    """预算未知，但已知对象下界 > 标称物理上限 → 仍判 OOM。"""
    rep = _rm(_budget(), bytes_=80 * GB + 1).peak_vs_budget()
    assert rep["status"] == "oom"
    assert any("标称容量" in r for r in rep["reasons"])


def test_measured_budget_complete_peak_feasible():
    rep = _rm(_budget(allocatable_bytes=70 * GB, allocatable_kind="measured",
                      allocatable_source="mem_get_info")).peak_vs_budget()
    assert rep["status"] == "feasible"


def test_measured_budget_oom():
    rep = _rm(_budget(allocatable_bytes=70 * GB, allocatable_kind="measured",
                      allocatable_source="mem_get_info"),
              bytes_=70 * GB + 1).peak_vs_budget()
    assert rep["status"] == "oom"
    assert any("峰值" in r for r in rep["reasons"])


def test_hypothesis_budget_conditional_only():
    rep = _rm(_budget(allocatable_bytes=70 * GB, allocatable_kind="hypothesis",
                      allocatable_source="user")).peak_vs_budget()
    assert rep["status"] == "feasible-conditional"
    assert any("条件结论" in r for r in rep["reasons"])


def test_incomplete_peak_unknown_even_with_measured_budget():
    rep = _rm(_budget(allocatable_bytes=70 * GB, allocatable_kind="measured",
                      allocatable_source="mem_get_info"),
              unknown=["workspace 未知"]).peak_vs_budget()
    assert rep["status"] == "unknown"
    assert any("峰值不完整" in r for r in rep["reasons"])


def test_overhead_deducted_once():
    """额外开销从预算扣除一次（gross 语义）；对象字节不再重复扣。"""
    b = _budget(allocatable_bytes=70 * GB, allocatable_kind="measured",
                allocatable_source="t", extra_overhead_bytes=5 * GB,
                overhead_source="framework reserve")
    assert b.effective_budget == 65 * GB
    rep = _rm(b, bytes_=65 * GB).peak_vs_budget()
    assert rep["status"] == "feasible"
    rep2 = _rm(b, bytes_=65 * GB + 1).peak_vs_budget()
    assert rep2["status"] == "oom"


# ---------------------------------------------------------------------------
# R03b：pipeline/比较资格（假设预算不进实际部署排名）
# ---------------------------------------------------------------------------


def test_pipeline_oom_scenario_end_to_end():
    """B×ctx 巨大 → 已知 KV 下界超 80GB 标称 → OOM（不依赖预算证据）。"""
    from simverse.contracts import Feasibility
    from simverse.pipeline import run_v2
    sc = ScenarioV2(workload={"batch": 512, "ctx_tokens": 262144})
    b = run_v2(sc)
    m06 = b.metrics["M06_memory_colocated_r0"]
    assert m06.feasibility == Feasibility.OOM
    assert b.capability["checks"]["memory_colocated_r0"]["status"] == "unsupported"
    # M02 不参与排名（OOM 传播）
    assert "不参与排名" in (b.metrics["M02_speedup_vs_baseline"].reason or "")


def test_pipeline_hypothesis_budget_conditional_metric():
    from simverse.contracts import Feasibility, Support
    from simverse.pipeline import run_v2
    sc = ScenarioV2(deployment=DeploymentSpec(
        budget=BudgetInput(allocatable_bytes=70 * GB,
                           allocatable_kind="hypothesis",
                           allocatable_source="研究假设")))
    b = run_v2(sc)
    chk = b.capability["checks"]["memory_colocated_r0"]
    # 峰值不完整（workspace/graph 缺口）→ 即便假设预算也是 unknown，
    # 原因必须写全（非静默 feasible）
    assert chk["status"] in ("unknown", "approximated")
    reason = chk["reason"]
    assert ("峰值不完整" in reason) or ("可分配预算未知" in reason)
    m06 = b.metrics["M06_memory_colocated_r0"]
    assert m06.feasibility == Feasibility.UNKNOWN


def test_hypothesis_budget_not_eligible_for_ranking():
    from simverse.experiments.compare import compare_runs
    from simverse.pipeline import run_v2
    base = run_v2(ScenarioV2(
        deployment=DeploymentSpec(mode="baseline"),
        method_id="baseline", draft_id=None))
    cand = run_v2(ScenarioV2(deployment=DeploymentSpec(
        budget=BudgetInput(allocatable_bytes=70 * GB,
                           allocatable_kind="hypothesis",
                           allocatable_source="研究假设"))))
    out = compare_runs(base, [cand])
    row = out["rows"][0]
    assert row["eligible"] is False
    assert "容量" in row["ineligible_reason"] or "峰值" in row["ineligible_reason"]
