# -*- coding: utf-8 -*-
"""R02a/R02b 同卡对象身份与生命周期峰值测试。

独立对象清单 + 守恒：期望字节由本文件手写公式推导（与生产实现不同路径），
共享对象去重、per-rank 差异、生命周期 max 相加规则逐项验证。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simverse.lowering.memory import (  # noqa: E402
    Allocation, DeviceBudget, RankMemory, colocated_rank_memory,
    draft_extra_allocations)
from simverse.model_ir import build_catalog  # noqa: E402
from simverse.model_ir.schema import (  # noqa: E402
    FULL, LINEAR, AttentionIR, LayerIR, LinearAttnIR, ModelIR, MoeIR,
    SourceIdentity)

Q8 = {"dense_w": 8, "expert_w": 8, "act": 8, "kv": 8, "state": 32}
Q_T = {"dense_w": 8, "expert_w": 4, "act": 8, "kv": 8, "state": 32}


def _budget(nominal=80_000_000_000):
    return DeviceBudget("h100_sxm", nominal)


def _qwen36():
    return build_catalog()["models"]["qwen36_35b_a3b"]


def _draft():
    return build_catalog()["drafts"]["dflash_qwen36_35b_a3b"]


def _colocated(tp=8, batch=1, ctx=32768, ep=1, rank=0, blk=16, en=3.5):
    return colocated_rank_memory(_qwen36(), _draft(), Q_T, Q8, tp, batch, ctx,
                                 _budget(), ep=ep, rank=rank, block_size=blk,
                                 e_produced=en)


# ---------------------------------------------------------------------------
# R02a：共享对象身份（embed/lm_head 物理去重）
# ---------------------------------------------------------------------------


def test_shared_embed_lmhead_single_physical_object():
    """草稿共享 target 的词表行切对象：一份物理字节、双 consumer、零字节引用。"""
    rm = _colocated(tp=8)
    inv = {a["alloc_id"]: a for a in rm.inventory()}
    embed = inv["target-w-embed"]
    assert embed["bytes"] == 2048 * (248320 // 8) * 1      # h·V/tp·1B
    assert sorted(embed["consumers"]) == ["draft", "target"]
    assert embed["shared_identity"] == "qwen36_35b_a3b:embed@tp8"
    lm = inv["target-w-lm_head"]
    assert sorted(lm["consumers"]) == ["draft", "target"]
    # 引用行零字节且不重复驻留（无 draft-w-embed/draft-w-lm_head 实体）
    assert "draft-w-embed" not in inv and "draft-w-lm_head" not in inv
    refs = [a for a in rm.allocations if a.kind == "reference"]
    assert len(refs) == 2 and all(a.bytes == 0 for a in refs)


def test_no_phantom_draft_head_weights():
    """TP1 草稿挂 TP>1 组的旧错误不再出现：草稿无整词表副本。

    （修复前：draft-w-embed/lm_head 各 h×V 全量 ≈1GB 幻影驻留 + 去重键
    不匹配双重计费。）
    """
    rm = _colocated(tp=8)
    for a in rm.inventory():
        assert "draft-w-embed" not in a["alloc_id"]
        assert "draft-w-lm_head" not in a["alloc_id"]
    draft_w = sum(a["bytes"] for a in rm.inventory() if a["role"] == "draft"
                  and a["kind"] == "weights")
    # 手算草稿权重 per-rank（TP8）：层权重 + fc(replicated) + norms
    h, V = 2048, 248320
    d = _draft()
    per_layer = 0
    a0 = d.layers[0].attn
    per_layer += h * (a0.heads // 8) * a0.head_dim          # q
    per_layer += 2 * h * (a0.kv_heads // 8) * a0.head_dim   # k/v（8≥8 切分）
    per_layer += (a0.heads // 8) * a0.head_dim * h          # o
    per_layer += 3 * h * (6144 // 8)                        # MLP inter/tp
    per_layer += 2 * a0.head_dim                           # q/k norm
    layers_b = 6 * per_layer * 1
    final_norm = h * 1
    expect = layers_b + draft_extra_allocations(d, Q8, 8)[0].bytes + final_norm
    assert draft_w == expect


def test_independent_hand_oracle_steady_bytes():
    """整 rank 稳态字节独立复算（全部手写公式；TP8、Q_T 口径）。

    口径：dense_w=8bit=1B、expert_w=4bit=0.5B、kv=8bit、state fp32。
    结构：40 层（30 GDN + 10 full attn，kvh=2；全层 MoE E=256 inter=512
    shared=512 无 gate）；草稿 6 层 GQA（heads=32 kvh=8 d=128 inter=6144）。
    """
    rm = _colocated(tp=8, batch=1, ctx=32768)
    h, V, L, E = 2048, 248320, 40, 256
    b_e = 0.5
    # ① GDN 层 per rank（qkvz/ba 列切、conv 通道切、out 行切、A/dt fp32）
    gdn = ((2 * 2048 + 2 * 4096) // 8 * h       # in_proj_qkvz
           + 2 * (32 // 8) * h                  # in_proj_ba
           + (8192 // 8) * 4                    # conv1d（kernel=4）
           + (4096 // 8) * h                    # out_proj
           + 2 * (32 // 8) * 4)                 # A_log/dt_bias fp32
    # ② full attn 层 per rank（q 门控 2×；kvh=2<tp → 复制；q/k norm）
    attn = ((16 * 256 * 2) // 8 * h             # q_proj（gated 2×，列切）
            + (2 * 2 * 256) * h                 # k/v（复制，每 rank 全量）
            + ((16 * 256) // 8) * h             # o_proj 行切
            + 2 * 256)                          # q/k norm 权重
    dense_per_rank = 30 * gdn + 10 * attn
    # ③ MoE 层 per rank（EP 关：router 复制、shared inter/tp、E 全量分片；
    #   config 无 shared_expert_gate → 无 gate 行）
    moe_per_layer = (h * E * 1                       # router（复制）
                     + 3 * h * (512 // 8) * 1        # shared inter/tp
                     + E * 3 * h * (512 // 8) * b_e)  # routed 分片
    moe_per_rank = L * moe_per_layer
    # ④ 词表行切 embed/lm_head + final norm
    head = 2 * h * (V // 8) + h
    target_w = dense_per_rank + moe_per_rank + head
    # ⑤ KV（kvh=2<tp 复制）+ GDN 状态（per-rank 切分）
    kv = 10 * 2 * 2 * 256 * 32768 * 1
    state = 30 * ((32 // 8) * 128 * 128 * 4 + (8192 // 8) * 3 * 1)
    # ⑥ 草稿 per rank（层权重切分 + fc(replicated)+2norm+mask + final norm）
    #    与草稿 KV（kvh=8≥tp → 切分为 1 头/rank）
    dl = (h * (32 // 8) * 128                      # q
          + 2 * h * (8 // 8) * 128                 # k/v
          + (32 // 8) * 128 * h                    # o
          + 3 * h * (6144 // 8)                    # MLP inter/tp
          + 2 * 128)                               # q/k norm
    draft_w = 6 * dl + (8 * h * h + 2 * h + h) + h
    dkv = 6 * 2 * 1 * 128 * 32768 * 1
    expect = target_w + kv + state + draft_w + dkv
    assert rm.steady_bytes() == expect


# ---------------------------------------------------------------------------
# R02a：per-rank 差异（EP 余数）与草稿批
# ---------------------------------------------------------------------------


def _synthetic_ep_model(E=6, experts_inter=128, heads=8, kvh=4, h=256):
    src = SourceIdentity("synthetic:test", "none", "hand")
    attn = AttentionIR(heads=heads, kv_heads=kvh, head_dim=32, rotary_dim=32)
    moe = MoeIR(experts=E, top_k=2, intermediate=experts_inter)
    layers = [LayerIR(i, FULL, attn=attn, moe=moe) for i in range(2)]
    return ModelIR("ep_t", "ep_t", "qwen3_moe", h, 1024, layers, src)


def test_ep_remainder_rank_difference():
    m = _synthetic_ep_model(E=6)
    from simverse.lowering.memory import target_weight_allocations
    base, rem = divmod(6, 4)
    b_expert = 3 * 256 * 128 * 1
    lo = target_weight_allocations(m, Q8, 4, ep=4, rank=2)
    hi = target_weight_allocations(m, Q8, 4, ep=4, rank=0)
    hi_moe = sum(a.bytes for a in hi if a.alloc_id.endswith("L0"))
    lo_moe = sum(a.bytes for a in lo if a.alloc_id.endswith("L0"))
    assert hi_moe - lo_moe == (base + 1 - base) * b_expert  # 恰差 1 个专家
    # colocated per-rank 账本差异显式呈现
    rm_hi = colocated_rank_memory(m, None, Q8, Q8, 4, 1, 128, _budget(),
                                  ep=4, rank=0)
    rm_lo = colocated_rank_memory(m, None, Q8, Q8, 4, 1, 128, _budget(),
                                  ep=4, rank=3)
    assert rm_hi.steady_bytes() > rm_lo.steady_bytes()


def test_draft_batch_matches_target_batch():
    """草稿批默认 = 目标批（逐请求草稿 KV；修复前硬编码 1）。"""
    rm_b4 = _colocated(batch=4)
    dkv = [a for a in rm_b4.inventory() if a["alloc_id"] == "draft-kv"][0]
    per_req = 6 * 2 * 1 * 128 * 32768 * 1
    assert dkv["bytes"] == 4 * per_req
    rm_b1 = _colocated(batch=1)
    dkv1 = [a for a in rm_b1.inventory() if a["alloc_id"] == "draft-kv"][0]
    assert dkv1["bytes"] == per_req


# ---------------------------------------------------------------------------
# R02b：生命周期与峰值
# ---------------------------------------------------------------------------


def test_lifecycle_categories_present():
    rm = _colocated()
    cats = {a.lifecycle for a in rm._dedup()}
    assert "resident" in cats and "per-request" in cats
    assert {"phase:draft", "phase:verify", "cross-round"} <= cats


def test_peak_takes_phase_max_not_sum():
    """串行阶段（draft/verify）峰值取 max，不无条件相加。"""
    rm = RankMemory("r", "dev", _budget())
    rm.add(Allocation("w", "weights", "target", 1000, lifecycle="resident"))
    rm.add(Allocation("d", "transient", "draft", 300, lifecycle="phase:draft"))
    rm.add(Allocation("v", "transient", "target", 200, lifecycle="phase:verify"))
    rm.add(Allocation("c", "transient", "shared", 50, lifecycle="cross-round"))
    assert rm.steady_bytes() == 1000
    assert rm.transient_peak_bytes() == 50 + 300      # cross + max(300,200)
    assert rm.known_peak_bytes() == 1350


def test_unknown_components_block_feasible():
    """峰值不完整（workspace/graph/NCCL 无证据）→ 即使有 measured 预算也 unknown。"""
    rm = RankMemory("r", "dev", DeviceBudget(
        "d", 100_000, allocatable_bytes=90_000, allocatable_kind="measured",
        allocatable_source="test"))
    rm.add(Allocation("w", "weights", "target", 1000, lifecycle="resident"))
    rm.unknown_components = ["workspace 未知"]
    rep = rm.peak_vs_budget()
    assert rep["status"] == "unknown"
    assert any("峰值不完整" in r for r in rep["reasons"])


def test_shared_object_double_add_single_count():
    """同一物理身份重复声明：字节只计一次、consumer 合并（重复释放守恒）。"""
    rm = RankMemory("r", "dev", _budget())
    ident = "m:embed@tp2"
    rm.add(Allocation("t-embed", "weights", "target", 500,
                      shared_identity=ident, consumers=["target"]))
    rm.add(Allocation("d-embed", "weights", "draft", 500,
                      shared_identity=ident, consumers=["draft"]))
    assert rm.steady_bytes() == 500
    inv = rm.inventory()
    assert len(inv) == 1 and sorted(inv[0]["consumers"]) == ["draft", "target"]


# ---------------------------------------------------------------------------
# colocated 全链：峰值入册与 per-rank 清单
# ---------------------------------------------------------------------------


def test_pipeline_colocated_memory_report():
    from simverse.pipeline import run_v2
    from simverse.scenario import ScenarioV2
    b = run_v2(ScenarioV2())
    mem = b.to_json()["cost"]["memory"]
    assert len(mem["colocated_ranks"]) == 8
    r0 = mem["colocated_r0"]
    assert r0["steady_bytes"] + r0["transient_peak_bytes"] == \
        r0["known_peak_bytes"]
    assert r0["peak_complete"] is False            # workspace/graph 缺口传播
    assert r0["status"] == "unknown"
    assert "colocated_worst" in mem
    # M06 指标携带同一峰值（UI/导出口径一致）
    m06 = b.metrics["M06_memory_colocated_r0"]
    assert m06.value == r0["known_peak_bytes"]
    assert m06.feasibility.value if hasattr(m06.feasibility, "value") \
        else m06.feasibility == "unknown"


def test_ep8_ranks_uniform_when_exact_division():
    rm0 = _colocated(ep=8, rank=0)
    rm7 = _colocated(ep=8, rank=7)
    assert rm0.steady_bytes() == rm7.steady_bytes()   # 256/8 整除 → 均匀
