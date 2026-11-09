"""H200 第一条实测校准闭环（V11 首步：算子级 measured 组合）。

来源：pi-viewer @c5909a7 H200 Qwen3-30B-A3B-tiny CUDA_EVENT 数据蒸馏。
断言覆盖：产物完整性拒绝、查表精确性、阶段组合独立复算、域门禁与回退、
指标披露与 result_kind、verify 阶段混合覆盖。
"""
import json
from pathlib import Path

import pytest

from simverse.cost.profile_store import (PROFILE_SETS, ProfileIntegrityError,
                                         load_profile_set)
from simverse.pipeline import run_v2
from simverse.scenario import ScenarioV2

ROOT = Path(__file__).resolve().parent.parent
SET_ID = "h200-qwen3-30b-a3b-tiny-op-v1"
ASSET = ROOT / "assets" / "profiles" / PROFILE_SETS[SET_ID]


def _scenario(**over):
    data = {
        "schema_version": "0.2", "name": "t-calib",
        "model_id": "qwen3_30b_a3b", "draft_id": None, "method_id": "baseline",
        "deployment": {"mode": "baseline", "target_device": "h200_sxm",
                       "target_tp": 1, "draft_device": "l40", "draft_tp": 1,
                       "link": "pcie5"},
        "workload": {"batch": 8, "ctx_tokens": 2048},
        "acceptance": {"kind": "geometric", "alpha": 0.75},
        "quant": {"dense_w": 16, "expert_w": 16, "act": 16, "kv": 16, "state": 32},
        "draft_quant": {"dense_w": 16, "expert_w": 16, "act": 16, "kv": 16,
                        "state": 32},
        "cost_profile": SET_ID, "seed": 0}
    data.update(over)
    return ScenarioV2.from_json(data)


# ---------------------------------------------------------------------------
# 产物与加载
# ---------------------------------------------------------------------------

def test_manifest_admission_and_domain_recorded():
    man = json.loads((ASSET / "manifest.json").read_text(encoding="utf-8"))
    adm = man["admission"]
    assert "linear_op.csv" in adm["admit"] and "attention.csv" in adm["admit"]
    assert "moe.csv" in adm["reject"] and "128" in adm["reject"]["moe.csv"]
    assert "replica_note" in adm
    d = man["domain"]
    assert d["device"] == "h200_sxm" and d["precision"] == "BF16"
    assert d["tensor_parallel"] == 1
    assert d["shape_identity"]["n_q_head"] == 32
    assert d["shape_identity"]["head_dim"] == 128
    # 留出误差报告齐备且有限
    rep = man["holdout"]["error_report"]
    assert rep["linear_ops"]["attn_pre_proj"]["n"] > 10
    for grp in ("linear_ops", "attn_decode"):
        for op, r in rep[grp].items():
            assert 0 <= r["median_rel_err"] < 0.5
            assert r["p90_rel_err"] >= r["median_rel_err"] > 0


def test_load_rejects_tampered_artifact(tmp_path, monkeypatch):
    import shutil
    from simverse.cost import profile_store as ps
    dest = tmp_path / PROFILE_SETS[SET_ID]
    shutil.copytree(ASSET, dest)
    p = dest / "linear_ops.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    data["grid"]["attn_pre_proj"]["8"] = 0.001
    p.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(ps, "PROFILES_ROOT", tmp_path)
    with pytest.raises(ProfileIntegrityError):
        load_profile_set(SET_ID)


def test_exact_grid_lookup_matches_distill():
    pset = load_profile_set(SET_ID)
    lin = json.loads((ASSET / "linear_ops.json").read_text(encoding="utf-8"))
    assert pset.lookup_linear("attn_pre_proj", 8) == \
        pytest.approx(lin["grid"]["attn_pre_proj"]["8"])
    # 域外显式 None
    assert pset.lookup_linear("attn_pre_proj", 999999) is None
    att = json.loads((ASSET / "attn_decode.json").read_text(encoding="utf-8"))
    pt = next(p for p in att["points"] if p["batch"] == 8 and p["kv"] == 2048)
    assert pset.lookup_attn_block(8, 2048) == \
        pytest.approx(sum(pt["median_ms"].values()))
    assert pset.lookup_attn_block(8, 999999) is None  # kv 域外


# ---------------------------------------------------------------------------
# 阶段组合独立复算（半 oracle：从蒸馏文件与场景常量重推 selected）
# ---------------------------------------------------------------------------

def test_stage_composition_arithmetic():
    b = run_v2(_scenario()).to_json()
    cal = b["cost"]["calibration"]["stages"]["target_decode"]
    assert cal["domain_ok"] and cal["reasons"] == []

    lin = json.loads((ASSET / "linear_ops.json").read_text(encoding="utf-8"))
    att = json.loads((ASSET / "attn_decode.json").read_text(encoding="utf-8"))
    man = json.loads((ASSET / "manifest.json").read_text(encoding="utf-8"))
    gap_us = next(g["kernel_gap_us"] for g in man["kernel_gap"]
                  if g["execution_mode"] == "eager")
    L = 48  # Qwen3-30B-A3B 层数
    groups = {
        "attn_pre_proj": L, "attn_rope": L, "attn_post_proj": L,
        "input_layernorm": L, "post_attention_layernorm": L, "emb": 1,
    }
    expect_measured = sum(lin["grid"][g]["8"] * n for g, n in groups.items())
    pt = next(p for p in att["points"] if p["batch"] == 8 and p["kv"] == 2048)
    expect_measured += sum(pt["median_ms"].values()) * L
    assert cal["measured_ms"] == pytest.approx(expect_measured, rel=1e-6)
    # 差额法：selected = measured + (analytic − 被替换 analytic) + gap
    assert cal["selected_ms"] == pytest.approx(
        cal["measured_ms"] + cal["fallback_ms"] + cal["gap_allowance_ms"],
        rel=1e-6)
    # MoE/lm_head 未被 measured 组占有 → 落入回退
    group_names = {g["group"] for g in cal["groups"]}
    assert not any("moe" in g for g in group_names)
    assert cal["fallback_ms"] > 0
    # 间隙补贴只按计时区间边界计（2026-11-10 修复：区间内间隙已在 CUDA_EVENT
    # median 内，按全部 kernel 计费属重复计费——负例见下）
    expect_regions = 5 * L + 1 + 4 * L   # 5 linear 组/层 + emb + 4 attn op/层
    assert cal["gap_regions"] == expect_regions
    assert cal["gap_allowance_ms"] == pytest.approx(
        cal["gap_regions"] * gap_us * 1e-3, rel=1e-6)
    # 时间口径覆盖披露（依赖估计，非真实性度量）
    assert 0 < cal["coverage_time_frac"] < 1
    assert cal["coverage_time_frac"] == pytest.approx(
        cal["measured_ms"] / cal["selected_ms"], rel=1e-3)


def test_gap_policy_no_double_billing():
    """间隙基数必须是计时区间数，不得按全部 kernel 数计（重复计费回归负例）。"""
    b = run_v2(_scenario()).to_json()
    cal = b["cost"]["calibration"]["stages"]["target_decode"]
    assert cal["kernels"] > cal["gap_regions"]  # kernel 数 > 区间数（多 kernel op）
    man = json.loads((ASSET / "manifest.json").read_text(encoding="utf-8"))
    gap_us = next(g["kernel_gap_us"] for g in man["kernel_gap"]
                  if g["execution_mode"] == "eager")
    naive = cal["kernels"] * gap_us * 1e-3
    assert cal["gap_allowance_ms"] < naive  # 修复前=naive（B8 高估 ~0.32ms）


def test_metrics_disclosure_and_result_kind():
    b = run_v2(_scenario()).to_json()
    m = b["metrics"]["M01_tpot_profile_calibrated"]
    cal = b["cost"]["calibration"]["stages"]["target_decode"]
    assert m["value"] == pytest.approx(
        cal["selected_ms"] + cal["collectives_ms"], rel=1e-9)
    assert m["result_kind"] == "measured-composite"
    assert "measured" in m["evidence_refs"] and "assumed" in m["evidence_refs"]
    assert m["assumption_refs"]
    assert "非整步实测" in "".join(m["assumption_refs"])
    cov = b["cost"]["calibration"]["coverage"]
    assert cov["rows_covered"] > 0 and cov["rows_total"] > cov["rows_covered"]
    assert cov["holdout_error_report"]["linear_ops"]
    assert cov["environment"]["gpu_name"] == "NVIDIA H200"
    assert b["manifest"]["cost_digest"]  # 新成本身份入 run 身份
    assert "ev-h200-tiny-distill" in b["evidence_refs"]


# ---------------------------------------------------------------------------
# 域门禁与回退
# ---------------------------------------------------------------------------

def test_domain_reject_tp8_all_fallback_with_reason():
    sc = _scenario(deployment={"mode": "baseline", "target_device": "h200_sxm",
                               "target_tp": 8, "draft_device": "l40",
                               "draft_tp": 1, "link": "pcie5"})
    b = run_v2(sc).to_json()
    cal = b["cost"]["calibration"]["stages"]["target_decode"]
    assert cal["domain_ok"] is False
    assert any("TP" in r for r in cal["reasons"])
    assert cal["selected_ms"] is None and cal["groups"] == []
    assert cal["fallback_ms"] == pytest.approx(cal["analytic_ms"])
    assert b["metrics"]["M01_tpot_analytical"]["value"] > 0
    assert "M01_tpot_profile_calibrated" not in b["metrics"]
    chk = b["capability"]["checks"]["cost_profile_domain"]
    assert chk["status"] == "approximated"


def test_domain_reject_precision_mismatch():
    sc = _scenario(quant={"dense_w": 8, "expert_w": 8, "act": 8, "kv": 8,
                          "state": 32})
    b = run_v2(sc).to_json()
    cal = b["cost"]["calibration"]["stages"]["target_decode"]
    assert cal["domain_ok"] is False
    assert any("精度" in r for r in cal["reasons"])


def test_domain_reject_wrong_model_shape():
    data = _scenario().to_json()
    data["model_id"] = "qwen36_35b_a3b"   # hidden 相同但头布局/GDN 不同
    b = run_v2(ScenarioV2.from_json(data)).to_json()
    cal = b["cost"]["calibration"]["stages"]["target_decode"]
    assert cal["domain_ok"] is False
    assert any("形状" in r or "heads" in r for r in cal["reasons"])


def test_shape_gate_operator_composition_semantics():
    """形状一致的必要条件补强：QKNorm/全旋转/无滑窗三项算子构成核对。

    形状逐维一致但构成不同（无 QKNorm、部分旋转、滑窗层）的模型，实测块的
    算子构成与模型 forward 不符 → 必须显式形状域不符，不得静默套用测量值。
    """
    from simverse.cost.calibrated import _shape_mismatches
    from simverse.model_ir.schema import (AttentionIR, LayerIR, ModelIR,
                                          SourceIdentity)
    ident = load_profile_set(SET_ID).shape_identity
    src = SourceIdentity("t", "t", "0" * 64)

    def model(attn=None):
        a = attn or AttentionIR(heads=32, kv_heads=4, head_dim=128,
                                rotary_dim=128)
        return ModelIR("t", "t", "qwen3_moe", 2048, 151936,
                       [LayerIR(0, "full_attention", attn=a)], src)

    assert _shape_mismatches(model(), ident) == []          # 基线：一致
    no_qknorm = AttentionIR(heads=32, kv_heads=4, head_dim=128, rotary_dim=128,
                            qk_head_norm=False)
    assert any("QKNorm" in r for r in _shape_mismatches(model(no_qknorm), ident))
    partial = AttentionIR(heads=32, kv_heads=4, head_dim=128, rotary_dim=16)
    assert any("旋转" in r for r in _shape_mismatches(model(partial), ident))
    sliding = AttentionIR(heads=32, kv_heads=4, head_dim=128, rotary_dim=128,
                          sliding_window=4096)
    assert any("滑窗" in r for r in _shape_mismatches(model(sliding), ident))


def test_unknown_profile_set_rejected():
    with pytest.raises(ValueError):
        run_v2(_scenario(cost_profile="not-a-profile-set"))


# ---------------------------------------------------------------------------
# verify 阶段：线性算子覆盖、注意力块回退（Q>1 域外）
# ---------------------------------------------------------------------------

def test_chain_draft_verify_mixed_coverage():
    sc = _scenario(
        method_id="chain_draft", draft_id=None,
        deployment={"mode": "colocated", "target_device": "h200_sxm",
                    "target_tp": 1, "draft_device": "h200_sxm", "draft_tp": 1,
                    "link": "pcie5"})
    b = run_v2(sc).to_json()
    stages = b["cost"]["calibration"]["stages"]
    assert "verify" in stages
    v = stages["verify"]
    assert v["domain_ok"]
    groups = {g["group"]: g for g in v["groups"]}
    assert groups["attn_pre_proj"]["source"] == "measured"
    blk = groups["attn_decode_block"]
    assert blk["source"] == "analytic" and "Q=" in blk["reason"]
    # 投机轮仍产出校准 TPOT（draft 解析 + verify 混合）
    assert b["metrics"]["M01_tpot_profile_calibrated"]["value"] > 0


def test_cli_flag_applies_profile():
    from simverse.cli import _v2_scenario
    import simverse.cli as cli
    sc = _v2_scenario(str(ROOT / "assets" / "scenarios" / "h200_calib_loop1.json"))
    assert sc.cost_profile == SET_ID
    class _A:
        pass
    a = _A()
    a.scenario = None
    a.cost_profile = None
    sc2 = _v2_scenario(None)
    assert sc2.cost_profile is None  # 内置默认不带 profile


# ---------------------------------------------------------------------------
# G-C10：UI 精度选择器接线（BF16 档 → H200 profile 域内 → 校准指标出现）
# UI 只拼场景 JSON 提交 /api/v2/run；三通道（进程内/API/CLI）必须同 run 同数。
# ---------------------------------------------------------------------------

BF16_UI_SCENARIO = {   # workbench.js QUANT_PRESETS.bf16 档拼出的场景等价物
    "schema_version": "0.2", "name": "ui-bf16-calib",
    "model_id": "qwen3_30b_a3b", "draft_id": None, "method_id": "baseline",
    "deployment": {"mode": "baseline", "target_device": "h200_sxm",
                   "target_tp": 1, "draft_device": "l40", "draft_tp": 1,
                   "link": "pcie5"},
    "workload": {"batch": 8, "ctx_tokens": 2048},
    "acceptance": {"kind": "geometric", "alpha": 0.75},
    "quant": {"dense_w": 16, "expert_w": 16, "act": 16, "kv": 16, "state": 32},
    "draft_quant": {"dense_w": 16, "expert_w": 16, "act": 16, "kv": 16,
                    "state": 32},
    "cost_profile": SET_ID, "seed": 0}


def test_ui_bf16_preset_enters_calibration_domain():
    # 通道 1：进程内
    inproc = run_v2(ScenarioV2.from_json(BF16_UI_SCENARIO)).to_json()
    m = inproc["metrics"].get("M01_tpot_profile_calibrated")
    assert m is not None and m["value"] > 0
    assert inproc["capability"]["checks"]["cost_profile_domain"]["status"] == \
        "supported"
    cal = inproc["cost"]["calibration"]["stages"]["target_decode"]
    assert cal["domain_ok"] and cal["coverage_time_frac"] > 0.7

    # 通道 2：/api/v2/run（UI 实际通道）——同 run_id 同值
    from simverse.webapp import app
    from simverse.contracts import js_restore
    client = app.test_client()
    r = client.post("/api/v2/run", json=BF16_UI_SCENARIO)
    assert r.status_code == 200, r.data
    web = js_restore(r.get_json())
    assert web["manifest"]["run_id"] == inproc["manifest"]["run_id"]
    assert web["metrics"]["M01_tpot_profile_calibrated"]["value"] == m["value"]

    # 通道 3：CLI 场景文件读同一 JSON
    from simverse.cli import _v2_scenario
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                     encoding="utf-8") as f:
        json.dump(BF16_UI_SCENARIO, f, ensure_ascii=False)
        path = f.name
    sc = _v2_scenario(path)
    assert sc.quant["dense_w"] == 16 and sc.cost_profile == SET_ID
    cli_b = run_v2(sc).to_json()
    assert cli_b["manifest"]["run_id"] == inproc["manifest"]["run_id"]


def test_ui_static_precision_selector_wired():
    """静态接线核对：HTML 有精度选择器、JS 档位与后端契约一致（不冒充运行时测试）。"""
    html = (ROOT / "simverse" / "static" / "workbench.html").read_text(
        encoding="utf-8")
    js = (ROOT / "simverse" / "static" / "workbench.js").read_text(
        encoding="utf-8")
    assert 'id="sel-prec"' in html and "bf16" in html
    assert "QUANT_PRESETS" in js
    for frag in ("dense_w: 16, expert_w: 16, act: 16, kv: 16",   # bf16 档
                 "dense_w: 8, expert_w: 4, act: 8, kv: 8"):      # 默认档保留
        assert frag in js
    # 检查器精度标签改为从场景快照读取（不再写死 8b/4b）
    assert "quantLabel" in js and "权重 8b/4b" not in js
