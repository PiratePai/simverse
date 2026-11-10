"""M2 验证：方法语义、管线闭环、扫描/比较、profile 契约（V05/V06/V09/V10 部分）。
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import simverse.devices as DEV  # noqa: E402
from simverse.contracts import Feasibility, Support  # noqa: E402
from simverse.cost.profile import (CalibrationModel, ProfileRecord,  # noqa: E402
                                   apply_calibration,
                                   synthetic_profile_fixture)
from simverse.experiments import compare_runs, default_policy, sweep_1d, variant  # noqa: E402
from simverse.methods import build_method_plan, acceptance_summary  # noqa: E402
from simverse.model_ir import build_catalog  # noqa: E402
from simverse.pipeline import run_v2  # noqa: E402
from simverse.scenario import ScenarioV2  # noqa: E402

import importlib.util
_spec = importlib.util.spec_from_file_location("golden_v02", ROOT / "golden" / "oracles_v02.py")
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)


def _dflash_sc(**over):
    base = dict(method_id="dflash_v1", k=15,
                deployment={"mode": "colocated", "target_tp": 8},
                workload={"batch": 1, "ctx_tokens": 8192},
                acceptance={"kind": "geometric", "alpha": 0.72})
    base["deployment"].update(over.pop("deployment", {}))
    base["workload"].update(over.pop("workload", {}))
    base["acceptance"].update(over.pop("acceptance", {}))
    base.update(over)
    return ScenarioV2(**base)


# ---------------------------------------------------------------------------
# V05 方法与接受
# ---------------------------------------------------------------------------


def test_dflash_plan_semantics_bound_to_pinned_source():
    cat = build_catalog()
    d = cat["drafts"]["dflash_qwen36_35b_a3b"]
    plan = build_method_plan("dflash_v1", {"kind": "geometric", "alpha": 0.72},
                             64, draft_ir=d)
    assert plan.source_pin.startswith("z-lab/dflash@07ebd93")
    assert [s.step_id for s in plan.steps] == ["draft_block", "d2t", "verify_block",
                                               "commit_rewind", "hidden_extract"]
    assert plan.k_effective == 15 and plan.block_size == 16
    msgs = {m.direction: m for m in plan.messages}
    assert "E[N]·8·2048" in msgs["T->D"].bytes_per_round_expr
    assert plan.cache_policy["target_kv"].find("crop") >= 0


def test_acceptance_endpoints_and_unknowns():
    a1 = acceptance_summary(build_method_plan(
        "chain_draft", {"kind": "geometric", "alpha": 1.0}, 4))
    assert a1["E_N"] == 5.0                       # α=1 → K+1（端点修复）
    a0 = acceptance_summary(build_method_plan(
        "chain_draft", {"kind": "geometric", "alpha": 0.0}, 4))
    assert a0["E_N"] == 1.0                       # α=0 → 1
    am = acceptance_summary(build_method_plan(
        "chain_draft", {"kind": "geometric", "alpha": 0.75}, 4))
    assert am["E_N"] == pytest.approx(G.EN_GEOM_075_K4)   # golden 3.05078125
    ad = acceptance_summary(build_method_plan(
        "chain_draft", {"kind": "dynamic_k"}, 4))
    assert ad["E_N"] is None and ad["reason"]


def test_position_alphas_non_geometric():
    s = acceptance_summary(build_method_plan(
        "chain_draft", {"kind": "position_alphas", "alphas": [0.5, 0.9]}, 2))
    assert s["E_N"] == pytest.approx(1 + 0.5 + 0.5 * 0.9)


def test_mtp_k_over_config_rejected():
    cat = build_catalog()
    with pytest.raises(ValueError, match="MTP"):
        build_method_plan("mtp", {"kind": "geometric", "alpha": 0.7}, 4,
                          mtp_layers=1)


# ---------------------------------------------------------------------------
# 管线闭环
# ---------------------------------------------------------------------------


def test_required_scenario_qwen36_dflash_closed_loop():
    b = run_v2(_dflash_sc())
    assert b.manifest.status == "completed"
    m = b.metrics
    assert m["M03_E_N"].value == pytest.approx(3.552800756624948, rel=1e-9)
    assert m["M03_K"].value == 15
    tpot = m["M01_tpot_analytical"].value
    base = m["M01_tpot_baseline"].value
    assert tpot > 0 and base > tpot          # 该假设下投机盈利（解析估算）
    assert m["M04_round_ms"].value == pytest.approx(
        sum(v.value for v in m["M04_stage_breakdown_ms"].values()), rel=1e-9)
    # 串行口径：轮墙钟 = 阶段和（V07 解析/事件相等的静态部分）
    assert m["M04_round_ms"].value / m["M03_E_N"].value == pytest.approx(tpot, rel=1e-9)


def test_run_id_deterministic_and_digest_identity():
    sc = _dflash_sc()
    a, c = run_v2(sc), run_v2(sc)
    assert a.manifest.run_id == c.manifest.run_id
    sc2 = _dflash_sc(acceptance={"alpha": 0.73})
    b2 = run_v2(sc2)
    assert b2.manifest.run_id != a.manifest.run_id   # 输入变 → 身份变


def test_unknown_acceptance_propagates_to_tpot():
    b = run_v2(_dflash_sc(acceptance={"kind": "dynamic_k"}))
    assert b.metrics["M03_E_N"].value is None
    assert b.metrics["M01_tpot_analytical"].value is None
    assert b.metrics["M01_tpot_analytical"].reason


def test_v41_structure_gap_blocks_ranking():
    sc = ScenarioV2(model_id="deepseek_v41_flash", method_id="chain_draft",
                    draft_id=None, k=4, deployment={"mode": "colocated",
                                                    "target_tp": 8},
                    acceptance={"kind": "geometric", "alpha": 0.75})
    b = run_v2(sc)
    assert b.manifest.status == "completed"
    assert b.capability["checks"]["model_structure_complete"]["status"] == "unsupported"
    m02 = b.metrics["M02_speedup_vs_baseline"]
    assert m02.value is not None and m02.reason   # 数值保留 + 不参与排名


def test_baseline_deployment_plain():
    b = run_v2(ScenarioV2(method_id="baseline", draft_id=None,
                          deployment={"mode": "baseline", "target_tp": 8},
                          workload={"batch": 1, "ctx_tokens": 8192}))
    assert b.manifest.status == "completed"
    assert b.metrics["M03_E_N"].value is None or b.metrics["M03_E_N"].value == 1.0
    assert b.metrics["M01_tpot_analytical"].value == pytest.approx(
        b.metrics["M01_tpot_baseline"].value)


def test_tp_validation_rejects_illegal():
    sc = _dflash_sc(deployment={"target_tp": 5})   # 不整除 16 头
    b = run_v2(sc)
    assert b.manifest.status == "failed"
    # R01：检查项更名 parallel_valid（TP+EP 统一配方判定）
    chk = b.capability["checks"]["parallel_valid"]
    assert chk["status"] == "invalid"
    assert "不整除 Q 头数" in chk["reason"] or "TP5" in chk["reason"]


def test_collectives_included_at_tp8_absent_at_tp1():
    b8 = run_v2(_dflash_sc(deployment={"target_tp": 8}))
    b1 = run_v2(_dflash_sc(deployment={"target_tp": 1}))
    c8 = b8.cost["target_decode"]["collectives"]["total_t_ms"]
    c1 = b1.cost["target_decode"]["collectives"]["total_t_ms"]
    assert c8 > 0 and c1 == 0.0                    # E03 修复：TP 通信入账


def test_dt_serial_link_costs_present():
    b = run_v2(_dflash_sc(deployment={"mode": "dt_serial"}))
    assert b.cost["link"]["t_d2t_ms"] > 0 and b.cost["link"]["t_t2d_ms"] > 0
    assert b.metrics["M19_msg_bytes"]["t2d"].value > 0   # hidden 消息非零


# ---------------------------------------------------------------------------
# V10 扫描隔离（E20）
# ---------------------------------------------------------------------------


def test_sweep_does_not_mutate_global_registry():
    before = {k: v.bw for k, v in DEV.LINKS.items()}
    before_dev = {k: v.hbm_bw for k, v in DEV.DEVICES.items()}
    scan = sweep_1d(_dflash_sc(), "acceptance.alpha", [0.5, 0.6, 0.7])
    assert {k: v.bw for k, v in DEV.LINKS.items()} == before
    assert {k: v.hbm_bw for k, v in DEV.DEVICES.items()} == before_dev
    ids = [p["run_id"] for p in scan["points"]]
    assert len(set(ids)) == len(ids)              # 每点独立身份


def test_sweep_points_have_independent_runs():
    scan = sweep_1d(_dflash_sc(), "block_size", [8, 16])
    vals = [(p["value"], p["tpot_ms"] is not None) for p in scan["points"]]
    assert vals == [(8, True), (16, True)]


# ---------------------------------------------------------------------------
# V09a 比较资格
# ---------------------------------------------------------------------------


def test_comparison_keeps_values_when_ineligible():
    base = run_v2(ScenarioV2(method_id="baseline", draft_id=None,
                             deployment={"mode": "baseline", "target_tp": 8},
                             workload={"batch": 1, "ctx_tokens": 8192}))
    cand = run_v2(_dflash_sc())
    out = compare_runs(base, [cand], default_policy("equal_total_inventory"))
    row = out["rows"][0]
    assert row["eligible"] is False                # 容量 unknown（标称口径）
    assert row["objectives"]["M01_tpot_analytical"]["candidate"] > 0


def test_research_method_proxies_cannot_enter_deployment_ranking():
    base = run_v2(ScenarioV2(method_id="baseline", draft_id=None,
                             deployment={"mode": "baseline"}))
    for method, k in (("mtp", 1), ("chain_draft", 4),
                      ("chain_moe_proxy", 4)):
        candidate = run_v2(ScenarioV2(method_id=method, k=k))
        row = compare_runs(base, [candidate])["rows"][0]
        assert row["eligible"] is False
        assert row["ineligible_reason"].startswith("方法：")
        assert row["objectives"]["M01_tpot_analytical"]["candidate"] > 0


def test_speculative_verify_includes_target_tp_collectives():
    for method, k in (("mtp", 1), ("chain_draft", 4),
                      ("chain_moe_proxy", 4)):
        bundle = run_v2(ScenarioV2(method_id=method, k=k))
        verify = bundle.cost["verify"]
        collective = verify["collectives"]["total_t_ms"]
        assert collective > 0
        assert verify["t_ms"] >= collective
        assert bundle.metrics["M04_stage_breakdown_ms"]["verify"].value == pytest.approx(
            verify["t_ms"])
        assert "method_recipe_ready" in bundle.capability["checks"]
        reason = bundle.metrics["M02_speedup_vs_baseline"].reason
        assert ("MTP" if method == "mtp" else "tiny_gqa") in reason


def test_dflash_draft_tp_collectives_and_pipeline_memory_gate():
    from simverse.scenario import BudgetInput, DeploymentSpec

    colocated = run_v2(ScenarioV2(method_id="dflash_v1"))
    draft_collectives = colocated.cost["draft_block"]["collectives"]
    assert draft_collectives["total_t_ms"] > 0
    assert draft_collectives["counts_by_kind"]["allreduce"] == 12

    measured = BudgetInput(allocatable_bytes=70 * 1024**3,
                           allocatable_kind="measured", allocatable_source="fixture")
    pipeline = run_v2(ScenarioV2(deployment=DeploymentSpec(
        mode="dt_pipeline_sim", target_tp=8, draft_tp=1,
        budget=measured,
        draft_budget=BudgetInput(allocatable_bytes=40 * 1024**3,
                                 allocatable_kind="measured",
                                 allocatable_source="fixture"))))
    assert pipeline.cost["draft_block"]["collectives"]["total_t_ms"] == 0
    assert "draft_r0" in pipeline.cost["memory"]
    assert pipeline.cost["memory"]["draft_r0"]["peak_complete"] is False
    assert pipeline.metrics["M02_speedup_vs_baseline"].reason
    base = run_v2(ScenarioV2(method_id="baseline", draft_id=None,
                             deployment={"mode": "baseline"}))
    assert compare_runs(base, [pipeline])["rows"][0]["eligible"] is False


def test_missing_stage_cost_rejected_and_unpinned_mtp_fails_cleanly():
    from simverse.execute.serial import build_serial_round
    from simverse.methods import build_method_plan

    plan = build_method_plan("baseline", {}, 1)
    with pytest.raises(ValueError, match="decode"):
        build_serial_round(plan, {}, 1.0)
    bundle = run_v2(ScenarioV2(model_id="deepseek_v41_flash",
                               method_id="mtp", k=3))
    assert bundle.manifest.status == "failed"
    assert "MTP" in bundle.metrics["error"].reason


# ---------------------------------------------------------------------------
# V04-cost profile 契约接线（合成 fixture；不是硬件测量）
# ---------------------------------------------------------------------------


def test_profile_record_fields_and_boundary():
    rec = synthetic_profile_fixture()
    assert rec.timing_boundary == "device-kernel"
    assert rec.source_kind == "synthetic"
    with pytest.raises(ValueError):
        ProfileRecord(**{**rec.__dict__, "timing_boundary": "wall-clock-illegal"})


def test_calibration_fallback_to_analytic():
    rec = synthetic_profile_fixture()
    cal = CalibrationModel(
        calibration_id="cal-test", version="1", content_sha256="x",
        train_refs=[], holdout_refs=[], coverage={"gemm-fp8-h100": "in"},
        method="iso-shape-neighbor")
    est = apply_calibration(123.0, rec, cal, "gemm-fp8-h100")
    assert est.selected_ms is None and est.cost_source == "analytic"
    assert est.fallback_reason.find("synthetic") >= 0
    assert est.unmeasured is True                  # Q2：无实测不升级
    est2 = apply_calibration(123.0, rec, None, "gemm-fp8-h100")
    assert est2.cost_source == "analytic" and est2.fallback_reason


def test_bundle_json_serializable_with_metrics():
    from simverse.contracts import js_safe
    import json
    b = run_v2(_dflash_sc())
    payload = js_safe(b.to_json())
    text = json.dumps(payload, allow_nan=False)
    assert '"run_id"' in text and '"M01_tpot_analytical"' in text
