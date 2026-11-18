"""R10 扫描口径/交叉点/部署比较资格回归（批次二）。

* sweep_1d 指标口径：默认=选用成本（配对选用基线）；显式解析则候选/
  基线/speedup 全解析口径；旧结果缺选用指标回退解析并明示；显式请求
  的指标缺失保留空值与原因；点级保留 run_id/实际口径/成本来源/原因，
  扫描级元数据反映实际使用口径；
* find_crossover：原始扫描顺序、只对相邻有效且口径可比的点插值；缺失/
  不可比截断区间；端点恰=1、连续=1、多交点取扫描序首个；无交点 None；
* 统一资格（experiments.eligibility，管线 M02 与 compare_runs 共用）：
  必需证据缺失=unknown 拦截（修复清空 capability.checks 仍 eligible）；
  草稿证据按场景适用域；baseline 只查适用项；预算 unknown/假设/OOM 均
  不可排名；
* compare_runs 口径核对：单位/时间基准/统计量/适用范围/结果类型相容才
  给相对变化；不相容保留数值与原因；证据齐全合成正例可合格比较。

修复前最小复现见 docs/validation/entry-scan-compare-fixes-20261119.md。
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simverse.contracts import Scope, Statistic, metric_from  # noqa: E402
from simverse.experiments import compare_runs, sweep_1d  # noqa: E402
from simverse.experiments.compare import metric_compatibility  # noqa: E402
from simverse.experiments.eligibility import eligibility_issues  # noqa: E402
from simverse.experiments.sweep_v2 import find_crossover  # noqa: E402
from simverse.scenario import ScenarioV2  # noqa: E402

Q16 = {"dense_w": 16, "expert_w": 16, "act": 16, "kv": 16, "state": 32}


def _ascend_records_scenario(tp=2):
    """昇腾归档 HCCL 通信记录 + DFlash 场景：collective 按记录计价、
    其余解析回退 → 选用/解析的候选、基线、speedup 全部真实分叉
    （r28-gemm 只改 baseline 侧；baseline 方法自比恒 1.0 无法暴露）。"""
    return ScenarioV2(**dict(
        method_id="dflash_v1", k=15, block_size=16, quant=Q16.copy(),
        draft_quant=Q16.copy(), workload=dict(batch=1, ctx_tokens=4096),
        cost_records="ascend-r29-hccl",
        deployment=dict(mode="colocated", target_device="ascend_910b3",
                        target_tp=tp)))


# ---------------------------------------------------------------------------
# sweep_1d 指标口径（真实记录档：selected ≠ analytical）
# ---------------------------------------------------------------------------

def test_r10e_records_scenario_actually_diverges_selected_vs_analytical():
    from simverse.pipeline import run_v2
    b = run_v2(_ascend_records_scenario())
    g = lambda k: b.metrics[k].value
    assert g("M01_tpot_selected") != g("M01_tpot_analytical")
    assert g("M01_baseline_selected") != g("M01_tpot_baseline")
    assert g("M02_speedup_selected") != g("M02_speedup_vs_baseline"), \
        "前置：归档记录场景的选用/解析口径必须确实分叉"


def test_r10e_default_sweep_uses_selected_pair_end_to_end():
    scan = sweep_1d(_ascend_records_scenario(), "acceptance.alpha",
                    [0.5, 0.6])
    assert scan["metric_key"] == "M01_tpot_selected"
    assert scan["requested_metric_key"] == "default:M01_tpot_selected"
    for p in scan["points"]:
        assert p["metric_key"] == "M01_tpot_selected"
        assert p["baseline_metric_key"] == "M01_baseline_selected"
        assert p["metric_source"] == "records"
        assert p["run_id"]
        # 手算核对：speedup = 选用基线 / 选用候选
        assert p["speedup"] == pytest.approx(
            p["baseline_tpot_ms"] / p["tpot_ms"])


def test_r10e_explicit_analytical_uses_analytic_pair_everywhere():
    scan = sweep_1d(_ascend_records_scenario(), "acceptance.alpha",
                    [0.5, 0.6], metric_key="M01_tpot_analytical")
    assert scan["metric_key"] == "M01_tpot_analytical"
    default = sweep_1d(_ascend_records_scenario(), "acceptance.alpha",
                       [0.5, 0.6])
    for p, pd in zip(scan["points"], default["points"]):
        assert p["metric_key"] == "M01_tpot_analytical"
        assert p["baseline_metric_key"] == "M01_tpot_baseline"
        # 显式解析口径：候选/基线/speedup 全部解析，且与选用口径确实不同
        assert p["tpot_ms"] != pd["tpot_ms"]
        assert p["baseline_tpot_ms"] != pd["baseline_tpot_ms"]
        assert p["speedup"] == pytest.approx(
            p["baseline_tpot_ms"] / p["tpot_ms"])
        assert p["speedup"] != pytest.approx(pd["speedup"])


def _fake_bundle(metrics, cost=None, status="completed"):
    from types import SimpleNamespace
    return SimpleNamespace(
        manifest=SimpleNamespace(run_id="sha256:fake", status=status),
        metrics=metrics, cost=cost or {})


def _m(value, unit="ms"):
    return metric_from(value, unit, "internal",
                       statistic=Statistic.EXPECTATION,
                       scope=Scope(role="target", phase="steady-decode",
                                   aggregation="per-run", time_basis="committed"))


def test_r10e_default_falls_back_on_old_results_without_selected(monkeypatch):
    """旧结果缺选用指标：默认请求回退解析口径并明示（兼容）。"""
    from simverse.experiments import sweep_v2
    old = _fake_bundle({
        "M01_tpot_analytical": _m(67.0),
        "M01_tpot_baseline": _m(112.0),
    })
    monkeypatch.setattr(sweep_v2, "run_v2", lambda sc: old)
    scan = sweep_v2.sweep_1d(ScenarioV2.from_json(
        {"model_id": "qwen36_35b_a3b", "method_id": "dflash_v1", "k": 4}),
        "acceptance.alpha", [0.5])
    p = scan["points"][0]
    assert scan["metric_key"] == "M01_tpot_analytical"
    assert "回退解析口径" in scan["fallback_reason"]
    assert p["metric_key"] == "M01_tpot_analytical"
    assert "回退解析口径" in p["fallback_reason"]
    assert p["speedup"] == pytest.approx(112.0 / 67.0)


def test_r10e_explicit_missing_metric_keeps_null_with_reason(monkeypatch):
    """显式请求的指标缺失：空值+原因，不静默换口径。"""
    from simverse.experiments import sweep_v2
    only_selected = _fake_bundle({
        "M01_tpot_selected": _m(70.0),
        "M01_baseline_selected": _m(112.0),
    })
    monkeypatch.setattr(sweep_v2, "run_v2", lambda sc: only_selected)
    scan = sweep_v2.sweep_1d(ScenarioV2.from_json(
        {"model_id": "qwen36_35b_a3b", "method_id": "dflash_v1", "k": 4}),
        "acceptance.alpha", [0.5], metric_key="M01_tpot_analytical")
    p = scan["points"][0]
    assert p["tpot_ms"] is None and p["speedup"] is None
    assert "M01_tpot_analytical" in p["reason"]
    assert p["metric_key"] == "M01_tpot_analytical"
    assert "fallback_reason" not in scan  # 未发生回退：显式请求保持空值


def test_r10e_unknown_explicit_metric_rejected():
    with pytest.raises(ValueError, match="无基线配对"):
        sweep_1d(ScenarioV2.from_json(
            {"model_id": "qwen36_35b_a3b", "method_id": "dflash_v1", "k": 4}),
            "acceptance.alpha", [0.5], metric_key="M03_E_N")


# ---------------------------------------------------------------------------
# find_crossover 手算锚点
# ---------------------------------------------------------------------------

def _pts(*pairs, key=None):
    keys = [key] * len(pairs) if key else None
    out = []
    for i, (x, y) in enumerate(pairs):
        p = {"value": x, "speedup": y}
        if keys:
            p["metric_key"] = keys[i]
        out.append(p)
    return {"points": out}


def test_r10e_crossover_normal_flip_interpolates():
    assert find_crossover(_pts((1.0, 0.5), (2.0, 1.5))) == pytest.approx(1.5)
    assert find_crossover(_pts((1.0, 1.5), (3.0, 0.5))) == pytest.approx(2.0)


def test_r10e_crossover_missing_point_truncates_interval():
    # 修复前：过滤缺失点后跨 [1,3] 插值得到 2.0（穿过未知区间）
    assert find_crossover(_pts((1.0, 0.5), (2.0, None), (3.0, 1.5))) is None
    assert find_crossover(_pts((1.0, None), (2.0, 0.5), (3.0, 1.5))) \
        == pytest.approx(2.5)  # 缺失在区间外：相邻有效对仍可插值


def test_r10e_crossover_incomparable_caliber_truncates():
    scan = {"points": [
        {"value": 1.0, "speedup": 0.5, "metric_key": "M01_tpot_selected"},
        {"value": 2.0, "speedup": 1.5, "metric_key": "M01_tpot_analytical"}]}
    assert find_crossover(scan) is None
    same = {"points": [
        {"value": 1.0, "speedup": 0.5, "metric_key": "M01_tpot_selected"},
        {"value": 2.0, "speedup": 1.5, "metric_key": "M01_tpot_selected"}]}
    assert find_crossover(same) == pytest.approx(1.5)


def test_r10e_crossover_endpoint_exactly_one():
    assert find_crossover(_pts((1.0, 0.5), (2.0, 1.0))) == 2.0
    assert find_crossover(_pts((1.0, 1.0))) == 1.0            # 首点恰=1
    assert find_crossover(_pts((1.0, 1.5), (2.0, 1.0))) == 2.0


def test_r10e_crossover_consecutive_ones_return_first():
    assert find_crossover(_pts((1.0, 0.8), (2.0, 1.0), (3.0, 1.0),
                               (4.0, 0.6))) == 2.0


def test_r10e_crossover_multiple_crossings_return_first_in_scan_order():
    assert find_crossover(_pts((1.0, 0.5), (2.0, 1.5), (3.0, 0.5),
                               (4.0, 1.5))) == pytest.approx(1.5)


def test_r10e_crossover_none_when_no_crossing():
    assert find_crossover(_pts((1.0, 2.0), (2.0, 3.0))) is None
    assert find_crossover(_pts((1.0, 0.2), (2.0, 0.4))) is None
    assert find_crossover({"points": []}) is None


def test_r10e_crossover_equality_wins_before_later_crossing():
    # 扫描顺序：i=1 恰=1 先于其后区间 (1,2) 的翻转被返回
    assert find_crossover(_pts((1.0, 0.5), (2.0, 1.0), (3.0, 1.5))) == 2.0
    # 反之，更早区间 (0,1) 的翻转先于点 2 的等值：1 + 0.5/1.1×1
    assert find_crossover(_pts((1.0, 0.5), (2.0, 1.6), (3.0, 1.0))) \
        == pytest.approx(1.4545454545)


# ---------------------------------------------------------------------------
# 统一资格矩阵（experiments.eligibility；管线 M02 与 compare_runs 共用）
# ---------------------------------------------------------------------------

def _full_supported_checks(extra=None, memory_status="supported",
                           memory_reason="feasible；预算 measured"):
    checks = {
        "model_structure_complete": ("supported", "完备"),
        "model_config_coverage": ("supported", "完备"),
        "model_forward_evidence": ("supported", "源码核过"),
        "model_cpu_forward": ("supported", "cpu-executed"),
        "model_gpu_measurement": ("supported", "model-measured"),
        "deployment_ranking_evidence": ("supported", "齐全"),
        "parallel_valid": ("supported", "TP8/EP1"),
        "memory_colocated_r0": (memory_status, memory_reason),
    }
    if extra:
        checks.update(extra)
    return {k: {"status": s, "reason": r} for k, (s, r) in checks.items()}


DFLASH_SCENARIO = {"name": "t", "model_id": "qwen36_35b_a3b",
                   "draft_id": "dflash_qwen36_35b_a3b",
                   "method_id": "dflash_v1",
                   "deployment": {"mode": "colocated",
                                  "target_device": "h100_sxm", "target_tp": 8,
                                  "draft_device": "l40", "draft_tp": 1,
                                  "engram_table_placement": "cpu_offload"}}
BASELINE_SCENARIO = {"name": "b", "model_id": "qwen36_35b_a3b",
                     "draft_id": None, "method_id": "baseline",
                     "deployment": {"mode": "baseline",
                                    "target_device": "h100_sxm",
                                    "target_tp": 8,
                                    "draft_device": "l40", "draft_tp": 1}}


def test_r10f_empty_checks_not_eligible_with_reasons():
    issues = eligibility_issues({}, DFLASH_SCENARIO)
    assert issues
    assert all("必需证据缺失" in i or "memory_" in i or "容量" in i
               for i in issues)
    assert any("model_structure_complete" in i for i in issues)


def test_r10f_single_required_missing_produces_specific_reason():
    checks = _full_supported_checks(extra={
        "draft_forward_evidence": ("supported", "ok"),
        "dflash_serving_integration": ("supported", "ok")})
    del checks["model_gpu_measurement"]
    issues = eligibility_issues(checks, DFLASH_SCENARIO)
    assert len([i for i in issues if "model_gpu_measurement" in i]) == 1
    assert "未产生该检查" in issues[0]


def test_r10f_draft_evidence_required_only_for_draft_scenarios():
    checks = _full_supported_checks()  # 无草稿/服务集成检查
    # baseline：只检查适用项 → 通过
    assert eligibility_issues(checks, BASELINE_SCENARIO) == []
    # dflash：草稿证据缺失 → 拦截并点名
    issues = eligibility_issues(checks, DFLASH_SCENARIO)
    assert any("draft_forward_evidence" in i for i in issues)
    assert any("dflash_serving_integration" in i for i in issues)
    # 补齐草稿证据后通过
    ok = _full_supported_checks(extra={
        "draft_forward_evidence": ("supported", "源码核过"),
        "dflash_serving_integration": ("supported", "双栈核对")})
    assert eligibility_issues(ok, DFLASH_SCENARIO) == []


@pytest.mark.parametrize("status,frag", [
    ("unknown", "unknown"),                       # 预算未知
    ("approximated", "假设"),                      # 假设预算（feasible-conditional）
    ("unsupported", "OOM"),                       # OOM
], ids=["budget-unknown", "budget-hypothesis", "oom"])
def test_r10f_budget_states_all_block_ranking(status, frag):
    extra = {"draft_forward_evidence": ("supported", "源码核过"),
             "dflash_serving_integration": ("supported", "双栈核对"),
             "memory_colocated_r0": (status, f"{frag}；详见 M06")}
    checks = _full_supported_checks(extra=extra)
    issues = eligibility_issues(checks, DFLASH_SCENARIO)
    assert issues
    assert frag in " ".join(issues)   # 容量原因在场且拦截


def test_r10f_stage_cost_applies_only_to_v41_engram_host():
    # qwen/dflash：不适用 stage_cost_complete，其余齐全 → 通过
    assert eligibility_issues(_ok_checks(), DFLASH_SCENARIO) == []
    v41 = dict(DFLASH_SCENARIO, model_id="deepseek_v41_flash",
               draft_id=None, method_id="baseline",
               deployment={"mode": "baseline",
                           "engram_table_placement": "cpu_offload"})
    issues = eligibility_issues(_full_supported_checks(), v41)
    assert any("stage_cost_complete" in i for i in issues)
    gpu_offload = dict(v41, deployment={"mode": "baseline",
                                        "engram_table_placement": "gpu"})
    assert eligibility_issues(_full_supported_checks(), gpu_offload) == []


def _synth_bundle(checks, scenario, tpot=10.0, speedup=2.0, mem=1 << 30,
                  metric_over=None):
    from types import SimpleNamespace
    sc = Scope(role="target", phase="steady-decode", aggregation="per-run",
               time_basis="committed")
    metrics = {
        "M01_tpot_analytical": metric_from(tpot, "ms", "internal",
                                           statistic=Statistic.EXPECTATION,
                                           scope=sc),
        "M02_speedup_vs_baseline": metric_from(speedup, "ratio", "internal",
                                               statistic=Statistic.EXPECTATION,
                                               scope=sc),
        "M06_memory_live": metric_from(mem, "B", "internal",
                                       statistic=Statistic.EXACT, scope=sc),
    }
    if metric_over:
        for k, v in metric_over.items():
            if v is None:
                metrics.pop(k, None)
            else:
                metrics[k] = v
    return SimpleNamespace(
        manifest=SimpleNamespace(run_id="sha256:synth"),
        scenario_snapshot=scenario, capability={"checks": checks},
        metrics=metrics)


def _ok_checks(method="dflash_v1"):
    extra = {"draft_forward_evidence": ("supported", "源码核过"),
             "dflash_serving_integration": ("supported", "双栈核对")} \
        if method == "dflash_v1" else {}
    return _full_supported_checks(extra=extra)


def test_r10f_compare_empty_checks_bundle_not_eligible():
    """修复前最小复现：清空 capability.checks 后 eligible=True。"""
    base = _synth_bundle(_ok_checks("baseline"), BASELINE_SCENARIO)
    cand = _synth_bundle({}, DFLASH_SCENARIO)          # checks 被清空
    row = compare_runs(base, [cand])["rows"][0]
    assert row["eligible"] is False
    assert row["ineligible_reason"]
    # 未通过的结果保留可用数值
    assert row["objectives"]["M01_tpot_analytical"]["candidate"] == 10.0


def test_r10f_synthetic_positive_case_compares_as_eligible():
    base = _synth_bundle(_ok_checks("baseline"), BASELINE_SCENARIO,
                         tpot=20.0, speedup=1.0, mem=2 << 30)
    cand = _synth_bundle(_ok_checks(), DFLASH_SCENARIO,
                         tpot=10.0, speedup=2.0, mem=1 << 30)
    out = compare_runs(base, [cand])
    row = out["rows"][0]
    assert row["eligible"] is True and row["ineligible_reason"] is None
    assert out["baseline_eligible"] is True
    for oid, obj in row["objectives"].items():
        assert obj["compatibility"]["compatible"] is True, oid
        assert obj["delta"] is not None, oid


def test_r10f_time_basis_mismatch_blocks_delta_keeps_values():
    other_basis = Scope(role="target", phase="steady-decode",
                        aggregation="per-run", time_basis="server_emitted")
    cand_metric = metric_from(5.0, "ms", "internal",
                              statistic=Statistic.EXPECTATION,
                              scope=other_basis)
    base = _synth_bundle(_ok_checks("baseline"), BASELINE_SCENARIO,
                         tpot=20.0)
    cand = _synth_bundle(_ok_checks(), DFLASH_SCENARIO,
                         metric_over={"M01_tpot_analytical": cand_metric})
    row = compare_runs(base, [cand])["rows"][0]
    obj = row["objectives"]["M01_tpot_analytical"]
    assert obj["candidate"] == 5.0 and obj["baseline"] == 20.0  # 数值保留
    assert obj["delta"] is None
    assert "time_basis" in obj["compatibility"]["reason"]


@pytest.mark.parametrize("over,frag", [
    # 单位不匹配
    ({"M01_tpot_analytical": metric_from(
        5000.0, "us", "internal", statistic=Statistic.EXPECTATION,
        scope=Scope(role="target", phase="steady-decode",
                    aggregation="per-run", time_basis="committed"))}, "unit"),
    # 统计量不匹配（exact vs expectation）
    ({"M01_tpot_analytical": metric_from(
        5.0, "ms", "internal", statistic=Statistic.EXACT,
        scope=Scope(role="target", phase="steady-decode",
                    aggregation="per-run", time_basis="committed"))},
     "statistic"),
    # 适用范围不匹配（per-token vs per-run）
    ({"M01_tpot_analytical": metric_from(
        5.0, "ms", "internal", statistic=Statistic.EXPECTATION,
        scope=Scope(role="target", phase="steady-decode",
                    aggregation="per-token", time_basis="committed"))},
     "scope"),
    # 指标缺失一侧
    ({"M01_tpot_analytical": None}, "缺失"),
], ids=["unit", "statistic", "scope", "missing"])
def test_r10f_metric_compat_dimensions_block_delta(over, frag):
    base = _synth_bundle(_ok_checks("baseline"), BASELINE_SCENARIO)
    cand = _synth_bundle(_ok_checks(), DFLASH_SCENARIO, metric_over=over)
    row = compare_runs(base, [cand])["rows"][0]
    obj = row["objectives"]["M01_tpot_analytical"]
    assert obj["delta"] is None
    assert frag in (obj["compatibility"]["reason"] or "")


def test_r10f_result_kind_mixing_blocks_delta():
    from simverse.contracts import ResultKind
    measured = metric_from(5.0, "ms", "internal",
                           statistic=Statistic.EXPECTATION,
                           scope=Scope(role="target", phase="steady-decode",
                                       aggregation="per-run",
                                       time_basis="committed"),
                           result_kind=ResultKind.MEASURED_COMPOSITE)
    base = _synth_bundle(_ok_checks("baseline"), BASELINE_SCENARIO)
    cand = _synth_bundle(_ok_checks(), DFLASH_SCENARIO,
                         metric_over={"M01_tpot_analytical": measured})
    obj = compare_runs(base, [cand])["rows"][0]["objectives"][
        "M01_tpot_analytical"]
    assert obj["delta"] is None
    assert "result_kind" in obj["compatibility"]["reason"]


def test_r10f_metric_compatibility_unit_helper():
    a = _m(1.0)
    assert metric_compatibility(a, _m(2.0))["compatible"] is True
    assert metric_compatibility(None, a)["compatible"] is False
    b = metric_from(1.0, "us", "internal", statistic=Statistic.EXPECTATION,
                    scope=a.scope)
    assert metric_compatibility(a, b)["compatible"] is False


def test_r10f_policy_predicates_all_enforced():
    """声明的资格谓词不再空转：same_time_basis/result_kind_compatible 由
    口径核对执行，其余由 eligibility_issues 执行（映射逐项可追溯）。"""
    from simverse.experiments.compare import default_policy
    predicates = set(default_policy().eligibility_predicates)
    assert {"same_time_basis", "result_kind_compatible"} <= predicates
    # 其余谓词对应 eligibility 的检查族（存在性由上述测试覆盖）
    assert {"structure_complete", "memory_feasible_known",
            "gpu_measurement_present"} <= predicates


# ---------------------------------------------------------------------------
# CLI 扫描与 API 扫描各一次：点级指标/缺失原因/输出兼容性
# ---------------------------------------------------------------------------

_SMALL = {"model_id": "qwen36_35b_a3b", "method_id": "dflash_v1", "k": 4,
          "workload": {"ctx_tokens": 8192}}


def test_r10e_api_and_cli_sweep_compatible_point_metadata(tmp_path):
    import json
    import os
    import subprocess

    from simverse.webapp import app
    r = app.test_client().post("/api/v2/sweep", json=_SMALL)
    assert r.status_code == 200
    api = r.get_json()
    out = tmp_path / "sweep_v2.json"
    sc_file = tmp_path / "sc.json"
    sc_file.write_text(json.dumps(_SMALL), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "simverse.cli", "sweep-v2", str(sc_file),
         "-o", str(out)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env={**os.environ, "PYTHONPATH": str(ROOT)})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    cli = json.loads(out.read_text(encoding="utf-8"))
    # 同一基础场景：谱系一致；各维度点值与口径元数据逐点一致
    assert cli["base_digest"] == api["base_digest"]
    for dim in ("acceptance.alpha", "workload.batch", "workload.ctx_tokens",
                "block_size"):
        cp, ap = cli[dim]["points"], api[dim]["points"]
        assert len(cp) == len(ap)
        for pc, pa in zip(cp, ap):
            assert pc["value"] == pa["value"]
            assert pc["metric_key"] == pa["metric_key"]
            assert pc["baseline_metric_key"] == pa["baseline_metric_key"]
            assert pc["tpot_ms"] == pytest.approx(pa["tpot_ms"])
            assert pc["speedup"] == pytest.approx(pa["speedup"])
            # 点级缺失原因字段在（值可为 None）；run 状态携带
            assert "reason" in pc and pc["run_id"]
    assert cli["breakevens"] == api["breakevens"]
