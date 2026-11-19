"""R27（2026-11-17 cost-integrity 轮）回归：计算路由/绘图元数据/期望字节/
内部 chain 计价/警告保留/运行身份/产品隔离/计数不可变。

证据链（docs/validation/cost-integrity-review-fixes.md；evidence/
cost-integrity-r27/）：修复只改成本口径，不改 Work/Traffic/Memory 原始
计数（本文件用已提交 before 证据钉死）。全部为 hermetic 测试，不执行
vLLM/SGLang 模块、不做 GPU 测量。
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simverse.contracts import digest  # noqa: E402
from simverse.cost.t0 import _ELEMENTWISE, dtype_route, price_row  # noqa: E402
from simverse.count.work import OpRow  # noqa: E402
from simverse.devices import DEVICES  # noqa: E402
from simverse.model_ir import build_catalog  # noqa: E402
from simverse.pipeline import _device_snapshot, run_v2  # noqa: E402
from simverse.scenario import ScenarioV2, V41_QUANT  # noqa: E402


def _dflash_sc(**over):
    base = dict(method_id="dflash_v1", k=15, block_size=16,
                deployment={"mode": "colocated", "target_tp": 8},
                workload={"batch": 1, "ctx_tokens": 32768},
                acceptance={"kind": "geometric", "alpha": 0.72})
    base["deployment"].update(over.pop("deployment", {}))
    base["workload"].update(over.pop("workload", {}))
    base["acceptance"].update(over.pop("acceptance", {}))
    base.update(over)
    return ScenarioV2(**base)


# ---------------------------------------------------------------------------
# A：dtype_route 执行配方路由（理论峰值档，非实测校准）
# ---------------------------------------------------------------------------

def test_dtype_route_recipe_vs_heuristic_tiers():
    """V4.1 配方（extras.quant.gemm_compute_dtype）→ bf16 档；无配方模型
    沿用 dense_w 存储位宽启发（不引入"全看 act"的新全局规则）。"""
    cat = build_catalog()
    v41 = cat["models"]["deepseek_v41_flash"]
    recipe = (v41.extras or {}).get("quant")
    assert recipe and recipe["gemm_compute_dtype"] == "fp16"
    # 配方档：GEMM bf16 档；元素/注意力核仍看 act（=16 → fp16 档）
    assert dtype_route(V41_QUANT, recipe) == ("fp16", "fp16")
    # 无配方：旧启发口径保持不变（V4.1 场景 frozen 档在无配方时本会落 fp8）
    assert dtype_route(V41_QUANT, None) == ("fp8", "fp16")
    # Qwen3.6（无 extras.quant）默认档不受影响
    assert dtype_route({"dense_w": 8, "act": 8}) == ("fp8", "fp8")
    assert dtype_route({"dense_w": 16, "act": 8}) == ("fp16", "fp8")
    qwen = cat["models"]["qwen36_35b_a3b"]
    assert not (qwen.extras or {}).get("quant")


def test_dtype_route_changes_cost_not_counts():
    """同一账本两种计算档：字节字段逐项相等，只有 t_comp 随档位缩放——
    成本修正不得改变原始计数（G7 同型，行级）。"""
    row = OpRow("s", 0, "q_proj", flops=2 * 4096 * 4096,
                weight_read=4096 * 4096)
    dev = DEVICES["h100_sxm"]
    a = price_row(row, dev, "fp8", "fp8")
    b = price_row(row, dev, "fp16", "fp16")
    for k in ("bytes_exact", "bytes_expected", "bytes_host_expected"):
        assert a[k] == b[k] == (4096 * 4096 if k == "bytes_exact" else 0.0)
    # fp8 峰值 1979e12 是 fp16 989.5e12 的 2×：fp8 档 t_comp 恰为一半，t_mem 不变
    assert b["t_comp_ms"] == pytest.approx(2 * a["t_comp_ms"], rel=1e-12)
    assert a["t_mem_ms"] == b["t_mem_ms"]


# ---------------------------------------------------------------------------
# B：Roofline 元数据（同一次计价输出；单位与一致性）
# ---------------------------------------------------------------------------

def test_price_row_metadata_units_and_consistency():
    dev = DEVICES["h100_sxm"]
    gemm = OpRow("s", 0, "q_proj", flops=12345678, weight_read=999)
    ew = OpRow("s", 0, "rope", flops=999999, act_read=42, act_write=42)
    for row, tier in ((gemm, "fp8"), (ew, "fp16")):
        est = price_row(row, dev, "fp8", "fp16")
        assert est["compute_dtype"] == tier
        # 单位：TFLOP/s（十进制 1e12）与 dense 规格；effective = 规格×mfu
        assert est["peak_tflops"] == pytest.approx(
            dev.peak(tier) / 1e12, rel=1e-12)
        assert est["effective_peak_tflops"] == pytest.approx(
            dev.peak(tier) * dev.mfu_cap / 1e12, rel=1e-12)
        assert est["bandwidth_tb_s"] == pytest.approx(dev.hbm_bw / 1e12)
        # 与 roofline 同一计价一致：t_comp = flops / (effective·1e12) ·1e3
        assert est["t_comp_ms"] == pytest.approx(
            row.flops / (est["effective_peak_tflops"] * 1e12) * 1e3, rel=1e-12)
        assert est["t_mem_ms"] == pytest.approx(
            (est["bytes_exact"] + est["bytes_expected"]) / dev.hbm_bw * 1e3,
            rel=1e-12)


def test_roofline_points_carry_pricing_metadata():
    """绘图元数据由实际计价路径输出：V4.1 行 bf16 档 + 配方 basis；无配方
    模型行启发档 basis；row_id 稳定。"""
    v41 = run_v2(ScenarioV2(model_id="deepseek_v41_flash", method_id="baseline",
                            draft_id=None, quant=V41_QUANT,
                            deployment={"mode": "baseline", "target_tp": 8},
                            workload={"batch": 1, "ctx_tokens": 32768}))
    pts = v41.cost["roofline_points"]
    assert pts
    from simverse.cost.t0 import _ELEMENTWISE as EW
    for p in pts:
        assert p["compute_dtype"] in ("fp16", "fp8")
        assert p["peak_tflops"] == pytest.approx(989.5)
        assert p["effective_peak_tflops"] == pytest.approx(989.5 * 0.75)
        assert p["bandwidth_tb_s"] == pytest.approx(3.35)
        # V4.1：配方档（GEMM）与近似档（元素核）都应给出口径说明
        assert p["cost_basis"]
        if p["op"] not in EW:
            assert "执行配方" in p["cost_basis"] and "Marlin" in p["cost_basis"]
        else:
            assert "近似" in p["cost_basis"]
        assert p["row_id"] == f'{p["stage"]}:{p["layer"]}:{p["op"]}'
    ids = [p["row_id"] for p in pts]
    assert len(ids) == len(set(ids))

    qw = run_v2(_dflash_sc(quant={"dense_w": 8, "expert_w": 4, "act": 16,
                                  "kv": 8, "state": 32}))
    qpts = qw.cost["roofline_points"]
    gemm_tiers = {p["compute_dtype"] for p in qpts if p["op"] not in EW}
    ew_tiers = {p["compute_dtype"] for p in qpts if p["op"] in EW}
    # 无配方模型：gemm 看 dense_w=8 → fp8；元素核看 act=16 → fp16（口径分立）
    assert gemm_tiers == {"fp8"} and ew_tiers == {"fp16"}
    assert all("启发" in p["cost_basis"] for p in qpts if p["op"] not in EW)


def test_roofline_metadata_optional_and_old_fields_kept():
    """既有字段保留（前端兼容）；新增为可选字段。"""
    b = run_v2(_dflash_sc()).to_json()
    for p in b["cost"]["roofline_points"]:
        for key in ("row_id", "layer", "op", "stage", "ai_flop_per_byte",
                    "attained_tflops", "bound", "t_ms", "compute_dtype",
                    "peak_tflops", "effective_peak_tflops", "bandwidth_tb_s",
                    "cost_basis"):
            assert key in p


# ---------------------------------------------------------------------------
# C：期望通信字节（E[N] 浮点期望，非整数不截断）
# ---------------------------------------------------------------------------

def test_dflash_t2d_expected_bytes_not_truncated():
    """K15/α0.72 锚点：58209.08759654315（旧代码 int() 截成 58209）。"""
    b = run_v2(_dflash_sc()).to_json()
    t2d = b["cost"]["link"]["bytes"]["t2d"]
    assert isinstance(t2d, float)
    assert t2d == 58209.08759654315
    assert b["cost"]["link"]["bytes"]["d2t"] == 60   # 协议确定字节保持整数
    assert isinstance(b["cost"]["link"]["bytes"]["d2t"], int)
    m19 = b["metrics"]["M19_msg_bytes"]["t2d"]
    assert m19["value"] == t2d and m19["statistic"] == "expectation"
    # 确定字节（d2t）保持整数值（每轮协议消息，统计身份沿用轮期望口径）
    assert isinstance(b["metrics"]["M19_msg_bytes"]["d2t"]["value"], int)


def test_mtp_t2d_expected_bytes_float():
    """MTP t2d = B·E[N]·h·act_bytes：α0.7 K1 → E[N]=1.7 → 1.7·2048 = 3481.6。"""
    b = run_v2(ScenarioV2(method_id="mtp", k=1,
                          acceptance={"kind": "geometric", "alpha": 0.7},
                          deployment={"mode": "colocated", "target_tp": 8},
                          workload={"batch": 1, "ctx_tokens": 1024})).to_json()
    t2d = b["cost"]["link"]["bytes"]["t2d"]
    assert isinstance(t2d, float) and t2d == 3481.6
    assert b["metrics"]["M19_msg_bytes"]["t2d"]["statistic"] == "expectation"


def test_expected_bytes_survive_json_roundtrip():
    """序列化链路（bundle→json）不再截断浮点期望。"""
    b = run_v2(_dflash_sc()).to_json()
    text = json.dumps(b, allow_nan=False)
    assert "58209.08759654315" in text


# ---------------------------------------------------------------------------
# D：内部 chain 计价（草稿精度/设备/隔离）
# ---------------------------------------------------------------------------

def _chain_run(**over):
    base = dict(method_id="chain_draft", k=4,
                deployment={"mode": "colocated", "target_tp": 8},
                workload={"batch": 1, "ctx_tokens": 4096})
    base["deployment"].update(over.pop("deployment", {}))
    base.update(over)
    return run_v2(ScenarioV2(model_id="qwen36_35b_a3b", draft_id=None, **base),
                  allow_internal_methods=True)


def _draft_steps(bundle):
    return sum(v.value for k, v in
               bundle.metrics["M04_stage_breakdown_ms"].items()
               if k.startswith("draft_step_"))


def test_chain_draft_priced_with_draft_quant_not_target():
    """草稿计数与计价同源 dquant：目标 quant 档变化不影响草稿步耗时
    （旧代码误用目标 gemm 档计价草稿行）。"""
    a = _chain_run(quant={"dense_w": 16, "expert_w": 4, "act": 8,
                         "kv": 8, "state": 32})
    b = _chain_run(quant={"dense_w": 8, "expert_w": 4, "act": 8,
                         "kv": 8, "state": 32})
    assert a.manifest.status == "completed" == b.manifest.status
    # 目标侧随档位变化（fp16 档峰值减半 → 解析耗时更高）
    assert a.cost["target_decode"]["t_ms"] != b.cost["target_decode"]["t_ms"]
    # 草稿步只随 dquant 计价：两侧 dquant 相同 → 草稿步逐值相等
    assert _draft_steps(a) == pytest.approx(_draft_steps(b), rel=1e-12)
    # 反向锚：dquant 变档 → 草稿步变化（同源性的正例）
    c = _chain_run(draft_quant={"dense_w": 16, "expert_w": 8, "act": 8,
                                "kv": 8, "state": 32})
    assert _draft_steps(c) != pytest.approx(_draft_steps(b), rel=1e-9)


def test_chain_dt_mode_uses_draft_device():
    """colocated 用目标设备；DT 用草稿设备（旧代码恒用目标设备计价）。"""
    colocated = _chain_run()
    dt_l40 = _chain_run(deployment={"mode": "dt_serial", "target_tp": 8,
                                    "draft_device": "l40"})
    dt_h100 = _chain_run(deployment={"mode": "dt_serial", "target_tp": 8,
                                     "draft_device": "h100_sxm"})
    # 同批同结构：草稿步在 l40（峰值 362 vs 1979）上显著更慢
    assert _draft_steps(dt_l40) > _draft_steps(dt_h100)
    # colocated 草稿步与 target 设备同源（h100）≈ dt_h100 草稿步
    assert _draft_steps(colocated) == pytest.approx(_draft_steps(dt_h100),
                                                    rel=1e-12)


def test_chain_dt_draft_tp_unsupported_rejected():
    """chain 草稿无分片/独立组配方：draft_tp>1 显式拒绝（不静默按 TP1）。"""
    b = _chain_run(deployment={"mode": "dt_serial", "target_tp": 8,
                               "draft_device": "l40", "draft_tp": 2})
    assert b.manifest.status == "failed"
    assert b.assumption_dict.get("error") == "unsupported_combo"


def test_chain_methods_stay_internal():
    """产品入口不放行 chain（内部研究隔离；G8 补充面）。"""
    sc = ScenarioV2(model_id="qwen36_35b_a3b", method_id="chain_moe_proxy",
                    draft_id=None, k=4,
                    deployment={"mode": "colocated", "target_tp": 8})
    b = run_v2(sc)   # 不带 allow_internal_methods
    assert b.manifest.status == "failed"
    assert b.assumption_dict.get("error") == "internal_research_method"
    cat = build_catalog()
    assert "tiny_gqa" not in cat["models"]


# ---------------------------------------------------------------------------
# E：pipeline 模式 Metric 增补不丢警告
# ---------------------------------------------------------------------------

def _v41_chain_pipeline():
    return run_v2(
        ScenarioV2(model_id="deepseek_v41_flash", method_id="chain_draft",
                   draft_id=None, k=4, quant=V41_QUANT,
                   deployment={"mode": "dt_pipeline_sim", "target_tp": 8},
                   acceptance={"kind": "geometric", "alpha": 0.75}),
        allow_internal_methods=True)


def test_host_warning_survives_pipeline_mode_metric_update():
    b = _v41_chain_pipeline()
    assert b.manifest.status == "completed"
    m01 = b.metrics["M01_tpot_analytical"]
    reason = m01.reason or ""
    # 两条口径并存：Host 成本未计 + 串行 TPOT/流水周期不可混用
    assert "Engram host 表查找与预取时延未计" in reason
    assert "dt_pipeline_sim" in reason and "M18" in reason
    # 不重复堆叠：流水注记只出现一次
    assert reason.count("dt_pipeline_sim") == 1
    assert m01.support.value == "approximated"
    assert m01.formula_ref == "v2.roofline"      # 重建不再丢 formula
    assert "M18_cohort_period_ms" in b.metrics
    checks = b.capability["checks"]
    assert checks["engram_host_lookup_latency"]["status"] == "unknown"
    assert checks["stage_cost_complete"]["status"] == "unknown"
    serial = b.execution["rounds"][0]
    assert serial["timing_complete"] is False
    events = {event["stage"]: event for event in serial["events"]}
    assert events["verify"]["t_ms"] > 0
    assert "Engram host" in events["verify"]["reason"]
    assert events["t2d"]["start_ms"] is None
    assert events["t2d"]["start_lower_bound_ms"] > 0
    assert b.cost["pipeline_sim"]["timing_complete"] is False
    for key in ("M04_round_ms", "M04_round_selected", "M18_per_round_throughput_ms"):
        assert b.metrics[key].value > 0
        assert b.metrics[key].support.value == "approximated"
        assert b.metrics[key].feasibility.value == "unknown"
        assert "Engram host" in b.metrics[key].reason
    assert "Engram host" in b.metrics["M04_stage_breakdown_ms"]["verify"].reason


# ---------------------------------------------------------------------------
# F：运行身份（device_snapshot 数值快照 / input_digest / run_id）
# ---------------------------------------------------------------------------

def test_run_id_stable_for_same_inputs_and_snapshot():
    sc = _dflash_sc()
    a, c = run_v2(sc), run_v2(sc)
    assert a.manifest.run_id == c.manifest.run_id
    assert a.manifest.device_snapshot_digest == c.manifest.device_snapshot_digest


def test_device_snapshot_covers_values_and_sources():
    snap = _device_snapshot(_dflash_sc().deployment)
    t = snap["target"]
    assert t["id"] == "h100_sxm"
    assert t["peak_by_dtype"]["fp16"] == 989.5e12
    assert t["hbm_bw"] == 3.35e12 and t["sm_count"] == 132
    assert t["sources"]["peak_fp16"].startswith("spec:")
    assert snap["link"]["bw"] == 64e9
    assert snap["target_tp"] == 8 and snap["target_ep"] == 1
    # 非语义键不入快照
    assert not any(k in snap for k in ("timestamp", "created_at", "progress"))


def test_device_value_change_changes_identity(monkeypatch):
    """同设备 ID 下改带宽/峰值 → 硬件摘要与 run_id 变化。"""
    import dataclasses
    from simverse.devices import LINKS
    sc = _dflash_sc()
    base = run_v2(sc)
    dev = DEVICES["h100_sxm"]
    peak = dict(dev.peak_by_dtype)
    peak["fp8"] = 1900e12
    tuned = dataclasses.replace(dev, hbm_bw=3.0e12, peak_by_dtype=peak)
    monkeypatch.setitem(DEVICES, "h100_sxm", tuned)
    changed = run_v2(sc)
    assert changed.manifest.device_snapshot_digest != \
        base.manifest.device_snapshot_digest
    assert changed.manifest.run_id != base.manifest.run_id
    # 只改链路 fixed_ms（计价消费的数值不动结果，但快照含它 → 身份变）
    monkeypatch.setitem(LINKS, "pcie5",
                        dataclasses.replace(LINKS["pcie5"], fixed_ms=0.2))
    relinked = run_v2(sc)
    assert relinked.manifest.run_id != changed.manifest.run_id


def test_deployment_params_enter_full_identity():
    """关键部署参数（target_ep/draft_tp/v41_kv_record）变化 → 完整身份变化。"""
    sc = _dflash_sc(deployment={"target_tp": 8, "target_ep": 8})
    a = run_v2(_dflash_sc())
    b = run_v2(sc)
    assert b.manifest.run_id != a.manifest.run_id
    dt = _dflash_sc(deployment={"mode": "dt_serial", "draft_tp": 2})
    assert run_v2(dt).manifest.run_id != a.manifest.run_id
    # v41_kv_record：注册设备无 SM100 → sm100 记录族校验直接拒绝（不能作
    # 为可运行场景），该参数进 input_digest 的职责由 digest 直接验证
    v41 = ScenarioV2(model_id="deepseek_v41_flash", method_id="baseline",
                     draft_id=None, quant=V41_QUANT,
                     deployment={"mode": "baseline", "target_tp": 8})
    alt = ScenarioV2(model_id="deepseek_v41_flash", method_id="baseline",
                     draft_id=None, quant=V41_QUANT,
                     deployment={"mode": "baseline", "target_tp": 8,
                                 "v41_kv_record": "fp8_ds_mla_sm100"})
    with pytest.raises(ValueError, match="SM100|sm100"):
        alt.validate()
    assert v41.scenario_digest() != ScenarioV2(
        model_id="deepseek_v41_flash", method_id="baseline", draft_id=None,
        quant=V41_QUANT,
        deployment={"mode": "baseline", "target_tp": 8,
                    "engram_table_placement": "gpu"}).scenario_digest()
    assert run_v2(v41).manifest.input_digest == v41.scenario_digest()


def test_nonsemantic_key_order_does_not_change_digest():
    assert digest({"a": 1, "b": {"x": 2, "y": 3}}) == \
        digest({"b": {"y": 3, "x": 2}, "a": 1})


# ---------------------------------------------------------------------------
# 计数不可变：成本口径轮 before 证据钉死 work/memory
# ---------------------------------------------------------------------------

def test_v41_cost_route_fix_left_ledgers_untouched():
    """V4.1 的 R27 计数锚点在当前管线保持逐字段相等。"""
    folder = ROOT / "docs/evidence/cost-integrity-r27"
    scenario = ScenarioV2.from_json(json.loads(
        (folder / "scenario-v41-baseline.json").read_text(encoding="utf-8")))
    before = json.loads((folder / "before-v41-baseline.json").read_text(encoding="utf-8"))
    fresh = run_v2(scenario).to_json()
    assert fresh["work"] == before["work"]
    assert fresh["cost"]["memory"] == before["cost"]["memory"]
    assert fresh["cost"]["target_decode"]["t_ms"] > 0
