"""H200 Qwen3-235B-A22B 第二条实测校准闭环（真模型、TP∈{2,8} 域）。

与 30B-tiny 闭环的区别：官方身份已入册（catalog qwen3_235b_a22b）、
TP 集合域、TP1 测的 replicated 算子组并入 TP2/8（tp_invariant 登记）、
MoE 数据在册但 defer（per-rank 分片账本 V04 未统一——不混尺）。
"""
import json
from pathlib import Path

import pytest

from simverse.cost.profile_store import (PROFILE_SETS, ProfileIntegrityError,
                                         load_profile_set)
from simverse.pipeline import run_v2
from simverse.scenario import ScenarioV2

ROOT = Path(__file__).resolve().parent.parent
SET_ID = "h200-qwen3-235b-a22b-op-v1"
ASSET = ROOT / "assets" / "profiles" / PROFILE_SETS[SET_ID]
L = 94   # Qwen3-235B-A22B 层数


def _scenario(**over):
    data = {
        "schema_version": "0.2", "name": "t-calib-235b",
        "model_id": "qwen3_235b_a22b", "draft_id": None,
        "method_id": "baseline",
        "deployment": {"mode": "baseline", "target_device": "h200_sxm",
                       "target_tp": 8, "draft_device": "l40", "draft_tp": 1,
                       "link": "pcie5"},
        "workload": {"batch": 8, "ctx_tokens": 2048},
        "acceptance": {"kind": "geometric", "alpha": 0.75},
        "quant": {"dense_w": 16, "expert_w": 16, "act": 16, "kv": 16,
                  "state": 32},
        "draft_quant": {"dense_w": 16, "expert_w": 16, "act": 16, "kv": 16,
                        "state": 32},
        "cost_profile": SET_ID, "seed": 0}
    data.update(over)
    return ScenarioV2.from_json(data)


# ---------------------------------------------------------------------------
# manifest 准入与身份
# ---------------------------------------------------------------------------

def test_manifest_identity_and_admission():
    man = json.loads((ASSET / "manifest.json").read_text(encoding="utf-8"))
    ident = man["model_identity"]
    assert ident["repo_id"] == "Qwen/Qwen3-235B-A22B"
    assert ident["revision"].startswith("8efa6172")
    adm = man["admission"]
    assert "moe.csv:TP8/EP1" in adm["defer"]
    # 2026-11-10 A组 R01（G-C11）解除后：defer 理由改为数据准入（G-D2），
    # 分片账本混尺（V04）不再是阻塞项
    assert "G-D2" in adm["defer"]["moe.csv:TP8/EP1"]
    assert man["tp_invariant_groups"] == ["emb", "input_layernorm",
                                          "post_attention_layernorm"]
    assert man["domain"]["tensor_parallel"] == [2, 8]
    rep = man["holdout"]["error_report"]
    for grp in ("linear_ops", "attn_decode"):
        assert rep[grp]
        for op, r in rep[grp].items():
            assert 0 < r["median_rel_err"] < 0.05 and r["n"] > 30


def test_tampered_artifact_rejected(tmp_path, monkeypatch):
    import shutil
    from simverse.cost import profile_store as ps
    dest = tmp_path / PROFILE_SETS[SET_ID]
    shutil.copytree(ASSET, dest)
    p = dest / "attn_decode.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    data["points"][0]["median_ms"]["attn_decode"] = 0.001
    p.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(ps, "PROFILES_ROOT", tmp_path)
    with pytest.raises(ProfileIntegrityError):
        load_profile_set(SET_ID)


# ---------------------------------------------------------------------------
# 查表：TP 域与 TP 不变组
# ---------------------------------------------------------------------------

def test_lookup_tp_domain_and_invariance():
    pset = load_profile_set(SET_ID)
    assert pset.tp_allowed(2) and pset.tp_allowed(8)
    assert not pset.tp_allowed(1) and not pset.tp_allowed(4)
    lin = json.loads((ASSET / "linear_ops.json").read_text(encoding="utf-8"))
    # sharded 组：TP2/8 有表，TP1 无
    v8 = pset.lookup_linear("attn_pre_proj", 8, tp=8)
    assert v8 == pytest.approx(lin["grid"]["attn_pre_proj"]["8:8"])
    assert pset.lookup_linear("attn_pre_proj", 8, tp=1) is None
    assert pset.lookup_linear("attn_pre_proj", 999999, tp=8) is None
    # TP 不变组：TP1 测量复制到 2/8（值相等）；与 30B-tiny set 的同名组独立
    v2 = pset.lookup_linear("input_layernorm", 8, tp=2)
    assert v2 == pytest.approx(lin["grid"]["input_layernorm"]["2:8"])
    assert v2 == pytest.approx(lin["grid"]["input_layernorm"]["8:8"])
    # 注意力块 TP8 域外 kv → None
    assert pset.lookup_attn_block(8, 999999, tp=8) is None
    assert pset.lookup_attn_block(8, 2048, tp=1) is None   # TP1 无表


# ---------------------------------------------------------------------------
# 域门禁
# ---------------------------------------------------------------------------

def test_domain_reject_tp1_with_reason():
    sc = _scenario(deployment={"mode": "baseline", "target_device": "h200_sxm",
                               "target_tp": 1, "draft_device": "l40",
                               "draft_tp": 1, "link": "pcie5"})
    b = run_v2(sc).to_json()
    cal = b["cost"]["calibration"]["stages"]["target_decode"]
    assert cal["domain_ok"] is False
    assert any("TP" in r for r in cal["reasons"])
    assert "M01_tpot_profile_calibrated" not in b["metrics"]


def test_domain_reject_precision():
    sc = _scenario(quant={"dense_w": 8, "expert_w": 8, "act": 8, "kv": 8,
                          "state": 32})
    b = run_v2(sc).to_json()
    cal = b["cost"]["calibration"]["stages"]["target_decode"]
    assert cal["domain_ok"] is False
    assert any("精度" in r for r in cal["reasons"])


# ---------------------------------------------------------------------------
# 组合独立复算（TP8、94 层；MoE 必须在回退中）
# ---------------------------------------------------------------------------

def _stage_cal(tp=8):
    dep = {"mode": "baseline", "target_device": "h200_sxm", "target_tp": tp,
           "draft_device": "l40", "draft_tp": 1, "link": "pcie5"}
    b = run_v2(_scenario(deployment=dep)).to_json()
    return b, b["cost"]["calibration"]["stages"]["target_decode"]


def test_stage_composition_tp8():
    b, cal = _stage_cal(8)
    assert cal["domain_ok"] and cal["reasons"] == []
    lin = json.loads((ASSET / "linear_ops.json").read_text(encoding="utf-8"))
    att = json.loads((ASSET / "attn_decode.json").read_text(encoding="utf-8"))
    man = json.loads((ASSET / "manifest.json").read_text(encoding="utf-8"))
    gap_us = next(g["kernel_gap_us"] for g in man["kernel_gap"]
                  if g["execution_mode"] == "eager" and g["tensor_parallel"] == 8)
    groups = {"attn_pre_proj": L, "attn_rope": L, "attn_post_proj": L,
              "input_layernorm": L, "post_attention_layernorm": L, "emb": 1}
    # 全部组在 TP8 键下有值（sharded 为 TP8 实测；TP 不变组为 TP1 值复制）
    expect_measured = sum(lin["grid"][g]["8:8"] * n for g, n in groups.items())
    pt = next(p for p in att["points"]
              if p["tp"] == 8 and p["batch"] == 8 and p["kv"] == 2048)
    expect_measured += sum(pt["median_ms"].values()) * L
    assert cal["measured_ms"] == pytest.approx(expect_measured, rel=1e-6)
    # gap = 区间边界 × TP8 eager gap（区间内间隙不重复计费）
    expect_regions = 5 * L + 1 + 4 * L
    assert cal["gap_regions"] == expect_regions
    assert cal["gap_allowance_ms"] == pytest.approx(
        expect_regions * gap_us * 1e-3, rel=1e-6)
    assert cal["selected_ms"] == pytest.approx(
        cal["measured_ms"] + cal["fallback_ms"] + cal["gap_allowance_ms"],
        rel=1e-6)
    # MoE 按准入裁决回退（defer），不产生 measured 组
    group_names = {g["group"] for g in cal["groups"]}
    assert not any("moe" in g for g in group_names)
    # R01（2026-11-10）后 MoE 解析行为 per-rank 口径（TP8 inter/tp 切分，
    # flops/字节 ≈ 整模型/8）：fallback 从 38.30ms 降至 ~4.85ms，不再
    # "占主导"（旧断言 fallback>0.9×measured 编码的是整模型口径高估）。
    assert cal["fallback_ms"] > 0
    assert cal["fallback_ms"] < cal["measured_ms"]          # per-rank 口径
    assert 0.5 < cal["coverage_time_frac"] < 1.0            # MoE 仍在域外


def test_tp2_also_in_domain():
    b, cal = _stage_cal(2)
    assert cal["domain_ok"]
    assert cal["coverage_rows"][0] > 0
    m = b["metrics"]["M01_tpot_profile_calibrated"]
    assert m["value"] > 0 and m["result_kind"] == "measured-composite"


def test_count_ledger_unchanged_with_without_profile():
    """G7 同型：开/关 cost_profile，work 账本逐字段不变（校准只改成本）。"""
    a = run_v2(_scenario()).to_json()
    b = run_v2(_scenario(cost_profile=None)).to_json()
    ra = a["work"]["target_decode"]["rows"]
    rb = b["work"]["target_decode"]["rows"]
    assert len(ra) == len(rb)
    for x, y in zip(ra, rb):
        assert x == y
    assert a["manifest"]["run_id"] != b["manifest"]["run_id"]  # 成本身份变更
