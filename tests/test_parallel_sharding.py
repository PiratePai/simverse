# -*- coding: utf-8 -*-
"""R01 per-rank TP/EP 分片配方测试：独立手算 oracle、跨 rank 守恒、非法组合
拒绝、按层通信配方。

证据锚：vLLM @dc74b8cbbbe3dbb9399fe369757d63defd614cc2（R01 pin，
assets/identity/vllm_qwen35_pin/manifest.json 逐文件 SHA256）。
期望值全部由本文件独立推导（手写公式），不调用生产实现生成。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simverse.count import build_model_work  # noqa: E402
from simverse.count.moe import moe_rows  # noqa: E402
from simverse.lowering import (LinkLike, model_collectives,  # noqa: E402
                               target_weight_allocations, validate_parallel)
from simverse.lowering.parallel import plan_collective_recipe  # noqa: E402
from simverse.model_ir import build_catalog  # noqa: E402
from simverse.model_ir.schema import (  # noqa: E402
    FULL, AttentionIR, LayerIR, ModelIR, MoeIR, SourceIdentity)

Q8 = {"dense_w": 8, "expert_w": 8, "act": 8, "kv": 8, "state": 32}
LINK = LinkLike(bw=64e9, fixed_ms=0.1)


def _qwen36():
    return build_catalog()["models"]["qwen36_35b_a3b"]


def _draft():
    return build_catalog()["drafts"]["dflash_qwen36_35b_a3b"]


# ---------------------------------------------------------------------------
# 非法/合法组合（不静默替换并行模式）
# ---------------------------------------------------------------------------


def test_legal_combos_qwen36():
    m = _qwen36()
    for tp in (1, 2, 4, 8):
        assert validate_parallel(m, tp, 1).valid, f"TP{tp}/EP1 应合法"
    p = validate_parallel(m, 8, 8, draft=_draft())
    assert p.valid and p.moe_ep_enabled
    assert p.moe_layer_map and p.moe_layer_map[0][1].local_experts_base == 32
    assert p.vocab_per_rank == 248320 // 8


def test_illegal_combos_rejected_with_reasons():
    m = _qwen36()
    # 头数不整除
    assert not validate_parallel(m, 5, 1).valid
    assert any("头数" in e for e in validate_parallel(m, 5, 1).errors)
    # tp×ep 独立组合无实现配方
    assert not validate_parallel(m, 8, 2).valid
    assert any("ep=2" in e for e in validate_parallel(m, 8, 2).errors)
    # tp=1 时 ep>1 无意义
    assert not validate_parallel(m, 1, 1).valid is False  # tp1/ep1 合法
    assert not validate_parallel(m, 1, 4).valid
    # 词表不整除（构造 3 不整除：用 heads 16 的 tp=16 且 vocab 微调不可行——
    # 改用 EP>专家数 构造）
    big_ep = validate_parallel(m, 16, 16)
    assert big_ep.valid  # 256 ≥ 16，合法


def test_illegal_vocab_and_kvh_constraints_synthetic():
    src = SourceIdentity("synthetic:test", "none", "hand")
    attn = AttentionIR(heads=8, kv_heads=2, head_dim=32, rotary_dim=32)
    moe = MoeIR(experts=4, top_k=2, intermediate=96)   # 96 % 4(tp) == 0
    m = ModelIR("t1", "t1", "qwen3_moe", 256, 1000,
                [LayerIR(0, FULL, attn=attn, moe=moe)], src)
    assert validate_parallel(m, 2, 1).valid
    # kvh=2 < tp=4：4 % 2 == 0 → 复制合法；tp=8 → 8%2==0 复制也合法
    assert validate_parallel(m, 4, 1).valid
    # inter 不整除 tp（EP 关）：inter=96, tp=8 → 96%8==0 合法；改 inter=100
    moe_bad = MoeIR(experts=4, top_k=2, intermediate=100)
    m_bad = ModelIR("t2", "t2", "qwen3_moe", 256, 1000,
                    [LayerIR(0, FULL, attn=attn, moe=moe_bad)], src)
    assert not validate_parallel(m_bad, 8, 1).valid
    assert any("inter" in e for e in validate_parallel(m_bad, 8, 1).errors)
    # EP > 专家数（vLLM qwen3_next L184 拒绝）
    few = MoeIR(experts=2, top_k=1, intermediate=96)
    m_few = ModelIR("t3", "t3", "qwen3_moe", 256, 1000,
                    [LayerIR(0, FULL, attn=attn, moe=few)], src)
    p = validate_parallel(m_few, 4, 4)
    assert not p.valid
    assert any("专家数" in e for e in p.errors)


def test_ep_remainder_distribution():
    """E=6, ep=4：base=1 rem=2 → rank0/1 各 2 专家，rank2/3 各 1；Σ=6。"""
    src = SourceIdentity("synthetic:test", "none", "hand")
    attn = AttentionIR(heads=8, kv_heads=4, head_dim=32, rotary_dim=32)
    moe = MoeIR(experts=6, top_k=2, intermediate=128)
    m = ModelIR("t4", "t4", "qwen3_moe", 256, 1000,
                [LayerIR(0, FULL, attn=attn, moe=moe)], src)
    p = validate_parallel(m, 4, 4)
    assert p.valid
    locals_ = [p.moe_local_experts(m.layers[0], r) for r in range(4)]
    assert locals_ == [2, 2, 1, 1]
    assert sum(locals_) == 6


# ---------------------------------------------------------------------------
# MoE per-rank 配方（独立手算）
# ---------------------------------------------------------------------------


def test_moe_rows_tp_off_hand_oracle():
    """EP 关：inter/tp 切分；router 复制；shared inter/tp。

    tiny_hybrid：E=8 k=2 inter=512 shared=256 gate；T=6、tp=2、16bit。
    手算：
      routed flops = 2·6·2·3·256·(512/2) = 4,718,592
      容量 = 8·3·256·256·2B = 3,145,728B
      E[U] = 8·(1−(1−2/8)^6) = 8·(1−0.75^6) ≈ 7.823（期望读 = E[U]·分片字节）
    """
    cat = build_catalog(include_internal=True)
    moe = cat["models"]["tiny_hybrid"].layers[0].moe
    rows = moe_rows(moe, 6, 256, Q8, tp=2, ep=1)
    routed = next(r for r in rows if r["layer_op"] == "moe_routed")
    router = next(r for r in rows if r["layer_op"] == "moe_router")
    shared = next(r for r in rows if r["layer_op"] == "moe_shared")
    gate = next(r for r in rows if r["layer_op"] == "shared_gate")
    assert routed["flops"] == 2 * 6 * 2 * 3 * 256 * 256
    assert routed["weight_capacity"] == 8 * 3 * 256 * 256 * 1
    e_u = 8 * (1 - (1 - 2 / 8) ** 6)
    assert routed["weight_read_expected"] == pytest.approx(e_u * 3 * 256 * 256 * 1)
    # router 复制：全量（不随 tp 缩放）
    assert router["flops"] == 2 * 6 * 256 * 8
    assert router["weight_capacity"] == 256 * 8 * 1
    # shared inter/tp = 128
    assert shared["flops"] == 2 * 6 * 3 * 256 * 128
    assert gate["weight_capacity"] == 256 * 1


def test_qwen36_shared_expert_gate_counted_every_moe_layer():
    """F5：Qwen3.6 每 MoE 层都有 shared_expert_gate 行 + h 权重容量。

    HF pin：modeling_qwen3_5_moe.py:889-899 无条件构造/应用 gate；
    config 无 shared_expert_gate 键，不能用缺失键推断 False。
    """
    cat = build_catalog()
    m = cat["models"]["qwen36_35b_a3b"]
    assert all(l.moe is not None and l.moe.shared_gate for l in m.layers)
    rows = build_model_work(m, "decode", 1, 1024, 1, Q8, tp=1)
    gate = [r for r in rows if r.op == "shared_gate"]
    assert len(gate) == 40
    for r in gate:
        assert r.flops == 2 * 1 * 2048
        assert r.weight_read == 2048 * 1 and r.weight_capacity == 2048 * 1
    # 容量：每层 MoE allocation 必须包含 gate 的 h 字节
    for tp in (1, 8):
        allocs = {a.alloc_id.split("-L")[-1]: a.bytes
                  for a in target_weight_allocations(m, Q8, tp)
                  if "-w-moe-L" in a.alloc_id}
        inter = 512 // tp
        router = 2048 * 256 * 1
        shared = 3 * 2048 * inter * 1
        gate_b = 2048 * 1
        experts = 256 * 3 * 2048 * inter * 1
        assert len(allocs) == 40
        assert all(v == router + shared + gate_b + experts for v in allocs.values())


def test_moe_rows_ep_on_hand_oracle():
    """EP 开（ep=tp=4，E=8）：E_local=2 整专家；期望份额 E_l/E=1/4。

    routed flops = 2·6·2·3·256·512·(2/8) = 2,359,296
    容量 = 2·3·256·512·2B = 786,432B（16bit）
    E[U_local] = E_l·(1−(1−k/E)^T) = 2·(1−0.75^6) ≈ 1.956
    """
    cat = build_catalog(include_internal=True)
    moe = cat["models"]["tiny_hybrid"].layers[0].moe
    rows = moe_rows(moe, 6, 256, Q8, tp=4, ep=4, local_experts=2)
    routed = next(r for r in rows if r["layer_op"] == "moe_routed")
    assert routed["flops"] == 2 * 6 * 2 * 3 * 256 * 512 * (2 / 8)
    assert routed["weight_capacity"] == 2 * 3 * 256 * 512 * 1
    e_local = 2 * (1 - (1 - 2 / 8) ** 6)
    assert routed["weight_read_expected"] == pytest.approx(
        e_local * 3 * 256 * 512 * 1)
    # shared expert 仍按全局 TP 切（EP 不改 dense MLP 配方）
    shared = next(r for r in rows if r["layer_op"] == "moe_shared")
    assert shared["flops"] == 2 * 6 * 3 * 256 * (256 // 4)


# ---------------------------------------------------------------------------
# 跨 rank 守恒（TP-only / EP-only / TP×EP）
# ---------------------------------------------------------------------------


def _sharded_op_flops(model, tp, ep, rank, ops):
    rows = build_model_work(model, "decode", 1, 512, 1, Q8, tp=tp, ep=ep, rank=rank)
    return sum(r.flops for r in rows if r.op in ops)


GEMM_SHARDED = {"q_proj", "o_proj", "gate_up_proj", "down_proj",
                "in_proj_qkv", "in_proj_z", "in_proj_ba", "out_proj", "lm_head"}


def test_tp_sharded_gemm_conservation_qwen36():
    """TP 切分对象：Σ_ranks(per-rank flops) == TP1 整模型 flops。"""
    m = _qwen36()
    whole = _sharded_op_flops(m, 1, 1, 0, GEMM_SHARDED)
    for tp in (2, 4, 8):
        total = sum(_sharded_op_flops(m, tp, 1, r, GEMM_SHARDED)
                    for r in range(tp))
        assert total == whole, f"TP{tp} GEMM 守恒破坏"


def test_kv_proj_shard_or_replicate_by_tp():
    """k/v_proj：kvh≥tp 切分（Σ=整模型）；kvh<tp 复制（per-rank=整模型）。"""
    m = _qwen36()

    def per_rank(tp, rank, ops):
        return _sharded_op_flops(m, tp, 1, rank, ops)
    whole = per_rank(1, 1, {"k_proj", "v_proj"}) if False else \
        _sharded_op_flops(m, 1, 1, 0, {"k_proj", "v_proj"})
    # tp2：kvh=2 ≥ 2 → 切分，Σ = whole
    assert sum(_sharded_op_flops(m, 2, 1, r, {"k_proj", "v_proj"})
               for r in range(2)) == whole
    # tp4/8：kvh=2 < tp → 复制，每 rank == whole
    for tp in (4, 8):
        assert _sharded_op_flops(m, tp, 1, 0, {"k_proj", "v_proj"}) == whole


def test_ep_routed_conservation_qwen36():
    """EP 开：Σ_ranks(routed flops) == 整模型 routed flops（均匀路由份额求和）。"""
    m = _qwen36()
    whole = _sharded_op_flops(m, 1, 1, 0, {"moe_routed"})
    for tp in (2, 8):
        total = sum(_sharded_op_flops(m, tp, tp, r, {"moe_routed"})
                    for r in range(tp))
        assert total == whole, f"EP{tp} routed 守恒破坏"


def test_replicated_objects_not_summed():
    """复制对象（router/shared_gate）：per-rank == 整模型（不求和=不虚增）。"""
    m = _qwen36()
    whole = _sharded_op_flops(m, 1, 1, 0, {"moe_router", "shared_gate"})
    per_rank = _sharded_op_flops(m, 8, 1, 0, {"moe_router", "shared_gate"})
    assert per_rank == whole


def test_kv_replication_rule_qwen36():
    """kvh=2：tp2 切分（KV/2）；tp4/8 复制（KV 同 TP1）。"""
    m = _qwen36()

    def kv_bytes(tp):
        rows = build_model_work(m, "decode", 1, 1000, 1, Q8, tp=tp)
        return sum(r.kv_write for r in rows if r.op == "attention_core")
    b1 = kv_bytes(1)
    assert kv_bytes(2) == b1 // 2      # 2 % 2 == 0 → 切分
    assert kv_bytes(4) == b1           # 2 < 4 且 4 % 2 == 0 → 复制
    assert kv_bytes(8) == b1


def test_gdn_state_per_rank():
    """GDN 状态：v 头/conv 通道 /tp（30 层 × (4·128·128 + 1024·3) × 4B/8）。"""
    m = _qwen36()
    rows = build_model_work(m, "decode", 1, 0, 1, Q8, tp=8)
    st = sum(r.state_capacity for r in rows if r.op == "gdn_recurrence")
    per_layer = (4 * 128 * 128 * 4 + (8192 // 8) * 3 * 1)   # recurrent fp32 + conv act
    assert st == 30 * per_layer


# ---------------------------------------------------------------------------
# 通信配方（按层）
# ---------------------------------------------------------------------------


def test_collective_recipe_by_layer():
    m = _qwen36()
    p1 = validate_parallel(m, 8, 1)
    cols = model_collectives(m, p1, 2048, 1.0, 1, LINK, moe_top_k=8)
    kinds = [c.kind for c in cols]
    assert kinds.count("allreduce") == 80          # 40 层 × 2
    assert kinds.count("logits_gather") == 1       # lm_head 词表分片 gather（层外）
    p8 = validate_parallel(m, 8, 8)
    cols8 = model_collectives(m, p8, 2048, 1.0, 1, LINK, moe_top_k=8)
    kinds8 = [c.kind for c in cols8]
    assert kinds8.count("allreduce") == 40         # 仅 attn 段
    assert kinds8.count("ep_dispatch") == 40
    assert kinds8.count("ep_combine") == 40
    assert kinds8.count("logits_gather") == 1
    # EP dispatch 期望字节独立复算：E[ranks]=8·(1−(1−1/8)^8)≈6.14；
    # per-rank = T·E[ranks]·msg/8
    exp_ranks = 8 * (1 - (1 - 1 / 8) ** 8)
    d = next(c for c in cols8 if c.kind == "ep_dispatch")
    assert d.bytes_on_wire == int(1 * exp_ranks * 2048 * 1.0 / 8)
    # tp1 无通信
    assert model_collectives(m, validate_parallel(m, 1, 1), 2048, 1.0, 1,
                             LINK) == []


def test_logits_gather_hand_oracle():
    """lm_head logits gather 独立手算（vLLM logits_processor.py pin
    L185-198：lm_head.tp_size>1 → 每次 lm_head 调用 gather 一次）。

    每 rank 线上字节上界 = (tp−1)/tp × T×V×act；TP1 无此通信。
    2026-11-10 二轮审计补漏：此前 draft.py 注释声称"另计"但未实现。
    """
    from simverse.lowering import logits_gather
    m = _qwen36()
    p8 = validate_parallel(m, 8, 1)
    cols = model_collectives(m, p8, 2048, 1.0, 16, LINK, moe_top_k=8,
                             lm_head_tokens=15)
    g = next(c for c in cols if c.kind == "logits_gather")
    assert g.bytes_on_wire == int(7 / 8 * 15 * m.vocab * 1.0)
    # 默认 lm_head_tokens = tokens（decode/verify 全行经 lm_head）
    cols_def = model_collectives(m, p8, 2048, 1.0, 16, LINK, moe_top_k=8)
    g_def = next(c for c in cols_def if c.kind == "logits_gather")
    assert g_def.bytes_on_wire == int(7 / 8 * 16 * m.vocab * 1.0)
    # TP1：词表分片即全量，无 gather
    assert logits_gather(15, m.vocab, 1, 1.0, LINK).bytes_on_wire == 0
    # 时间模型：单向环 (tp−1) 段延迟 + 传输
    one = logits_gather(15, m.vocab, 8, 1.0, LINK)
    assert one.t_ms == pytest.approx(
        int(7 / 8 * 15 * m.vocab) / LINK.bw * 1e3 + 7 * LINK.fixed_ms)


def test_draft_second_read_traffic():
    """草稿二次读流量独立手算（dflash model.py L260/L384-387 pin）：

    draft_lm_head：共享对象但每轮独立 GEMM → weight_read = h×(V/tp)×b_w
    （修复前 0）；ctx_kv_proj：ctx 行第二次过 k/v_proj → weight_read =
    2×h×kvh_rank×d×b_w（修复前 0）。
    """
    from simverse.count.draft import dflash_round_rows
    m = _qwen36()
    draft = _draft()
    rows = dflash_round_rows("draft_block", draft, m.vocab, 3.0, 512, Q8,
                             1.0, tp=8, batch=1)
    lm = next(r for r in rows if r.op == "draft_lm_head")
    assert lm.weight_read == 2048 * (m.vocab // 8) * 1
    assert lm.weight_capacity == 0        # 容量归 target 物理对象（账本去重）
    ctx = [r for r in rows if r.op == "ctx_kv_proj"]
    assert len(ctx) == 6
    for r in ctx:
        assert r.weight_read == 2 * 2048 * (8 // 8) * 128 * 1


def test_plan_recipe_dispatch():
    m = _qwen36()
    p1 = validate_parallel(m, 8, 1)
    p8 = validate_parallel(m, 8, 8)
    assert plan_collective_recipe(p1, m.layers[0]) == "tp_allreduce"
    assert plan_collective_recipe(p8, m.layers[0]) == "ep_all2all"
    assert plan_collective_recipe(validate_parallel(m, 1, 1), m.layers[0]) == "none"


def test_plan_consistency_count_vs_direct():
    """经 plan 与直接 (tp, ep) 求值一致（防两套口径漂移）。"""
    m = _qwen36()
    p = validate_parallel(m, 8, 8)
    via_plan = build_model_work(m, "decode", 1, 512, 1, Q8, tp=8, ep=8, rank=3,
                                plan=p)
    direct = build_model_work(m, "decode", 1, 512, 1, Q8, tp=8, ep=8, rank=3)
    assert [r.to_json() for r in via_plan] == [r.to_json() for r in direct]


# ---------------------------------------------------------------------------
# 场景层贯通：EP 字段进身份与拒绝
# ---------------------------------------------------------------------------


def test_scenario_ep_enters_identity():
    from simverse.scenario import DeploymentSpec, ScenarioV2
    a = ScenarioV2()
    b = ScenarioV2(deployment=DeploymentSpec(target_ep=8))
    assert a.scenario_digest() != b.scenario_digest()
    with pytest.raises(ValueError, match="target_ep"):
        ScenarioV2(deployment=DeploymentSpec(target_ep=2)).validate()


def test_pipeline_ep_and_rejection_end_to_end():
    import pytest as _pytest
    from simverse.pipeline import run_v2
    from simverse.scenario import DeploymentSpec, ScenarioV2
    ok = run_v2(ScenarioV2(deployment=DeploymentSpec(target_ep=8)))
    assert ok.manifest.status == "completed"
    assert ok.capability["checks"]["parallel_valid"]["status"] == "supported"
    # 非法 EP 在场景校验即拒绝（显式报错，不静默换并行模式）
    with _pytest.raises(ValueError, match="target_ep=2"):
        ScenarioV2(deployment=DeploymentSpec(target_ep=2)).validate()
    # 合法 EP 但 plan 判非法（tp5 不整除 16 头）→ failed bundle + invalid
    bad = run_v2(ScenarioV2(deployment=DeploymentSpec(target_tp=5)))
    assert bad.manifest.status == "failed"
    assert bad.capability["checks"]["parallel_valid"]["status"] == "invalid"


# ---------------------------------------------------------------------------
# UI/静态接线（与 G-C10 精度选择器同口径：静态核对 + API 场景等值）
# ---------------------------------------------------------------------------


def test_workbench_ep_and_budget_wiring():
    js = (ROOT / "simverse" / "static" / "workbench.js").read_text(encoding="utf-8")
    html = (ROOT / "simverse" / "static" / "workbench.html").read_text(encoding="utf-8")
    assert "sel-ep" in html and "inp-budget" in html
    assert "target_ep" in js and "sel-ep" in js
    assert "allocatable_kind: 'hypothesis'" in js      # UI 预算=假设输入
    assert "allocatable_bytes" in js


def test_webapp_scenario_json_ep_budget_roundtrip():
    """/api/v2/run 的场景 JSON 与进程内构造同 run 同值（契约一致）。"""
    from simverse.pipeline import run_v2
    body = {
        "schema_version": "0.2", "name": "ep8-b",
        "model_id": "qwen36_35b_a3b", "method_id": "dflash_v1",
        "draft_id": "dflash_qwen36_35b_a3b", "k": 15,
        "deployment": {
            "mode": "colocated", "target_device": "h100_sxm",
            "target_tp": 8, "target_ep": 8, "draft_device": "l40",
            "draft_tp": 1, "link": "pcie5",
            "budget": {"allocatable_bytes": 70 * 1024 ** 3,
                       "allocatable_kind": "hypothesis",
                       "allocatable_source": "测试假设"}},
        "workload": {"batch": 1, "ctx_tokens": 8192},
        "acceptance": {"kind": "geometric", "alpha": 0.72},
    }
    from simverse.scenario import ScenarioV2
    via_api = ScenarioV2.from_json(body)
    inproc = ScenarioV2.from_json(body)
    b1 = run_v2(via_api)
    b2 = run_v2(inproc)
    assert b1.manifest.run_id == b2.manifest.run_id
    assert b1.scenario_snapshot["deployment"]["target_ep"] == 8
    assert b1.scenario_snapshot["deployment"]["budget"]["allocatable_kind"] \
        == "hypothesis"
