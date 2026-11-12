# -*- coding: utf-8 -*-
"""DFlash ctx-KV 显式实现路径的独立手算 oracle（2026-11-10 修正轮）。

覆盖 pinned vLLM qwen3_dflash.py 的三条分支：
  * L108-113 `_reads_raw_weights` → L555-563 `_fused_kv_weight` cat 副本；
  * L529-531 `_context_kv_proj`（ModelOpt MXFP8 独立线性层）；
  * L688-697 逐层完整 `qkv_proj` 后切 Q 的回退。

关键口径（TP8/B1/E[N]=3/blk16→verify 跨度 span=16 行/h2048/4 Q heads
per rank/1 KV head per rank/d128；R07a 2026-11-12 起投影计算按**跨度**
16 行计——两服务栈（vLLM _precompute_context_kv / SGLang
project_target_hidden）均对整 verify 跨度投影、只写 committed 行）：
  * raw/MXFP8 的 ctx 投影只算 K/V 行：2·16·2048·(2·1·128)=
    16,777,216 FLOPs/层；kv_write 仍按 produced=3 行；
  * 回退执行完整 qkv：2·16·2048·(4·128 + 2·1·128)=50,331,648 FLOPs/
    层，权重量读含 Q 行；
  * `_k_norm_weights` 的 L×d 堆叠在三条路径都存在（L565-568），且按
    草稿 config dtype（bfloat16→2B/elem）计，不随 `dense_w` 变化；
  * 位宽只是计价档，不决定分支；路径缺失/未知必须显式拒绝。

数字全部由本文件手写公式推导，不调用生产实现求期望。
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simverse.count.draft import dflash_round_rows  # noqa: E402
from simverse.draft_quant import (FUSED_RAW, MXFP8_CONTEXT_PROJ,  # noqa: E402
                                  PER_LAYER_FALLBACK)
from simverse.lowering.memory import (DeviceBudget,  # noqa: E402
                                      colocated_rank_memory,
                                      draft_extra_allocations,
                                      draft_peak_lower_bound,
                                      transient_allocations)
from simverse.model_ir import build_catalog  # noqa: E402
from simverse.scenario import ScenarioV2  # noqa: E402

Q8 = {"dense_w": 8, "expert_w": 8, "act": 8, "kv": 8, "state": 32}
Q4 = {"dense_w": 4, "expert_w": 4, "act": 8, "kv": 8, "state": 32}

H = 2048
D = 128
N_LAYERS = 6
B = 1
E_N = 3.0
TP = 8
Q_HEADS_RANK = 4          # 32 / 8
KV_HEADS_RANK = 1         # 8 / 8
PRODUCED = int(B * E_N)   # 3（committed 行——只进 kv_write）
SPAN = 16                 # B×blk verify 跨度（R07a：投影计算基）
NORM_BYTES = 2            # config dtype=bfloat16；RMSNorm 参数不随 dense_w


def _models():
    cat = build_catalog()
    return cat["models"]["qwen36_35b_a3b"], cat["drafts"]["dflash_qwen36_35b_a3b"]


def _ctx_rows(path, quant=Q8):
    m, d = _models()
    rows = dflash_round_rows("draft_block", d, m.vocab, E_N, 512, quant, 1.0,
                             tp=TP, batch=B, ctx_kv_path=path)
    ctx = [r for r in rows if r.op == "ctx_kv_proj"]
    assert len(ctx) == N_LAYERS
    return ctx


def test_raw_fused_path_hand_oracle():
    """raw 融合路径：每层只投影 K/V 行（fused 副本一次大 GEMM 的逐层份额）。

    R07a：计算按 verify 跨度 SPAN=16 行（两栈全跨度投影）；kv_write 按行内
    committed produced=3（被拒行 PAD 不落缓存——vLLM 测试负例在册）。
    """
    for r in _ctx_rows(FUSED_RAW):
        assert r.flops == 2 * SPAN * H * (2 * KV_HEADS_RANK * D) == 16_777_216
        assert r.weight_read == 2 * H * KV_HEADS_RANK * D * 1 == 524_288
        assert r.act_read == SPAN * H * 1
        assert r.kv_write == PRODUCED * 2 * KV_HEADS_RANK * D * 1
        assert "fused_raw" in r.shape


def test_fallback_full_qkv_hand_oracle():
    """4bit/量化回退：逐层完整 qkv_proj 后切 Q —— FLOPs 与权重读都含 Q。"""
    for r in _ctx_rows(PER_LAYER_FALLBACK, Q4):
        proj_out = Q_HEADS_RANK * D + 2 * KV_HEADS_RANK * D   # 512 + 256
        assert r.flops == 2 * SPAN * H * proj_out == 50_331_648
        assert r.weight_read == proj_out * H * 0.5 == 786_432
        assert "Q512" in r.shape and "KV128" in r.shape
        assert "per_layer_fallback" in r.shape


def test_mxfp8_context_projection_is_distinct_from_fallback():
    """MXFP8 独立 _context_kv_proj 只装 K/V 行：FLOPs/权重读都不含 Q。

    打包 8bit 权重按 1B/elem 下界计；block-scale 未入 pin，不得冒充已计。
    """
    for r in _ctx_rows(MXFP8_CONTEXT_PROJ):
        proj_out = 2 * KV_HEADS_RANK * D
        assert r.flops == 2 * SPAN * H * proj_out == 16_777_216
        assert r.weight_read == proj_out * H * 1 == 524_288
        assert "mxfp8_context_proj" in r.shape
        assert "block-scale" in r.shape


def test_dense_w_does_not_select_branch():
    """同一 dense_w=8 可显式走回退；dense_w=4 也可显式走 raw（函数层按路径）。

    这不是推荐组合，而是证明实现分支只由 path 选择，不由位宽阈值选择。
    """
    fb = _ctx_rows(PER_LAYER_FALLBACK)
    assert all(r.flops == 50_331_648 for r in fb)      # Q8 仍是完整 qkv
    raw = _ctx_rows(FUSED_RAW)
    assert all(r.flops == 16_777_216 for r in raw)


def test_unknown_path_rejected_not_guessed():
    """未提供路径 → 显式拒绝（不默认 raw、不按 dense_w 猜）。"""
    m, d = _models()
    with pytest.raises(ValueError, match="路径未知"):
        dflash_round_rows("draft_block", d, m.vocab, E_N, 512, Q8, 1.0,
                          tp=TP, batch=B)
    with pytest.raises(ValueError, match="路径未知"):
        draft_extra_allocations(d, Q8, TP)
    with pytest.raises(ValueError, match="未知 draft_ctx_kv_path"):
        draft_extra_allocations(d, Q8, TP, ctx_kv_path="not-a-path")


def test_k_norm_stack_uses_draft_dtype_not_quant_width():
    """`_k_norm_weights` L×d stack 在 raw/MXFP8/回退都按**草稿 dtype** 计。

    pinned layernorm.py L42-58：RMSNorm 未传 dtype 时以
    torch.get_default_dtype() 建 weight；base_loader.py L68-85 在
    model_config.dtype 下初始化；dflash_utils.py L51-54 用草稿
    model_config 加载。因此活跃草稿 bfloat16 → 6·128·2=1536B，
    与 draft_quant.dense_w=4/8/16 无关（此前 Q8=768、Q4=384 均错）。
    """
    _, d = _models()
    assert d.dtype_hint == "bfloat16"
    for path in (FUSED_RAW, MXFP8_CONTEXT_PROJ, PER_LAYER_FALLBACK):
        for quant in (Q8, Q4):
            allocs = draft_extra_allocations(d, quant, TP, ctx_kv_path=path)
            kn = [a for a in allocs if "k-norm-stack" in a.alloc_id]
            assert len(kn) == 1, (path, quant)
            assert kn[0].bytes == N_LAYERS * D * NORM_BYTES, (path, quant)
            assert "dtype_hint=bfloat16" in kn[0].live_note
    # dtype 变化由源数据驱动；fp32 草稿 → 4B/elem
    d32 = replace(d, dtype_hint="float32")
    kn32 = [a for a in draft_extra_allocations(
        d32, Q8, TP, ctx_kv_path=FUSED_RAW) if "k-norm-stack" in a.alloc_id][0]
    assert kn32.bytes == N_LAYERS * D * 4
    # 常见 torch.<dtype> 写法同样解析为 2B，不改变证据口径
    dtorch = replace(d, dtype_hint="torch.bfloat16")
    kn_torch = [a for a in draft_extra_allocations(
        dtorch, Q4, TP, ctx_kv_path=FUSED_RAW) if "k-norm-stack" in a.alloc_id][0]
    assert kn_torch.bytes == N_LAYERS * D * NORM_BYTES
    # 无 dtype 不得回退到 dense_w：显式拒绝
    d_unknown = replace(d, dtype_hint="")
    with pytest.raises(ValueError, match="dtype 未知"):
        draft_extra_allocations(d_unknown, Q8, TP, ctx_kv_path=FUSED_RAW)
    # 回退没有融合/独立投影常驻副本
    fb = draft_extra_allocations(d, Q4, TP, ctx_kv_path=PER_LAYER_FALLBACK)
    assert not any("fused-ctx-kv" in a.alloc_id for a in fb)
    assert not any("mxfp8-ctx-kv" in a.alloc_id for a in fb)


def test_memory_physical_objects_per_path():
    """常驻对象身份：raw 融合副本 vs MXFP8 独立投影 vs 回退无副本。"""
    _, d = _models()
    fused_elems = N_LAYERS * 2 * KV_HEADS_RANK * D
    raw = {a.alloc_id: a for a in
           draft_extra_allocations(d, Q8, TP, ctx_kv_path=FUSED_RAW)}
    assert raw["draft-w-dflash-fused-ctx-kv"].bytes == fused_elems * H * 1
    assert raw["draft-w-dflash-k-norm-stack"].bytes == N_LAYERS * D * NORM_BYTES

    mx = {a.alloc_id: a for a in
          draft_extra_allocations(d, Q8, TP, ctx_kv_path=MXFP8_CONTEXT_PROJ)}
    assert "draft-w-dflash-fused-ctx-kv" not in mx
    assert mx["draft-w-dflash-mxfp8-ctx-kv"].bytes == fused_elems * H * 1

    fb = {a.alloc_id: a for a in
          draft_extra_allocations(d, Q4, TP, ctx_kv_path=PER_LAYER_FALLBACK)}
    assert "draft-w-dflash-fused-ctx-kv" not in fb
    assert "draft-w-dflash-mxfp8-ctx-kv" not in fb
    assert fb["draft-w-dflash-k-norm-stack"].bytes == N_LAYERS * D * NORM_BYTES


def test_transient_fallback_includes_q_staging_lifecycle():
    """回退路径的逐层完整 qkv（含 Q）在 stack 前存活；生命周期下界取
    已知并存组 max，而不是把所有缓冲相加。

    Q4 仅 dense_w=4，act 仍 8（b_a=1）；B=1/blk=16：
      normed=16·2048=32,768；flat=fused=16·6·2·128=24,576；
      k_norm=16·6·128=12,288；qkv_staging=16·6·(4+2)·128=73,728；
      logits=15·31040·4=1,862,400（块尾，与 ctx 不并活）。
    raw ctx 组 max = max(57,344, 49,152, 36,864)=57,344；
    fallback 额外 normed+qkv=106,496、qkv+flat=98,304 → ctx max=106,496。
    logits 主导块峰值，故 t-draft-buffers 两路径同值；差异必须由
    ctx group/live_note 体现，而不是硬加到总字节。
    """
    m, d = _models()
    kw = dict(model=m, draft_ir=d, quant=Q4, dquant=Q4, batch=B, ctx=32768,
              tp=TP, block_size=16, e_produced=E_N,
              target_vocab_per_rank=m.vocab // TP)
    raw_buf = next(a for a in transient_allocations(
        ctx_kv_path=FUSED_RAW, **kw) if a.alloc_id == "t-draft-buffers")
    fb_buf = next(a for a in transient_allocations(
        ctx_kv_path=PER_LAYER_FALLBACK, **kw) if a.alloc_id == "t-draft-buffers")
    normed = int(SPAN * H)
    flat = int(SPAN * N_LAYERS * 2 * KV_HEADS_RANK * D)
    fused = flat
    k_normed = int(SPAN * N_LAYERS * KV_HEADS_RANK * D)
    qkv_staging = int(SPAN * N_LAYERS
                      * (Q_HEADS_RANK + 2 * KV_HEADS_RANK) * D)
    logits = int(B * (SPAN - 1) * (m.vocab // TP) * 4)
    noise = int(B * SPAN * H)
    fc_in = int(SPAN * 8 * H)
    assert qkv_staging == 73_728
    raw_ctx = max(normed + flat, flat + fused, fused + k_normed)
    fb_ctx = max(raw_ctx, normed + qkv_staging, qkv_staging + flat)
    assert raw_ctx == 57_344 and fb_ctx == 106_496
    assert max(fc_in, noise, logits, raw_ctx) == logits
    assert max(fc_in, noise, logits, fb_ctx) == logits
    assert raw_buf.bytes == fb_buf.bytes == logits
    assert raw_buf.bytes < (noise + fc_in + normed + flat + fused
                            + k_normed + logits)
    assert "qkv_staging" not in raw_buf.live_note
    assert "qkv_staging" in fb_buf.live_note
    # 独立生命周期 helper：组逐项手算 + 峰值取 max 而非 sum。
    peak, ctx_peak, groups = draft_peak_lower_bound(
        fc_in=fc_in, noise=noise, logits=logits, normed_ctx=normed,
        qkv_staging=qkv_staging, ctx_kv_flat=flat, fused_kv_out=fused,
        ctx_k_normed=k_normed)
    assert peak == logits and ctx_peak == fb_ctx
    assert dict(groups)["qkv_staging+flat"] == 98_304
    assert dict(groups)["normed_ctx+qkv_staging"] == 106_496
    assert sum(v for _, v in groups) > ctx_peak    # 组内也不是求和


def test_colocated_raw_vs_fallback_single_resident_difference():
    """同输入仅切路径：稳态差恰为 raw 融合副本字节；k_norm 两边都在。"""
    m, d = _models()
    budget = DeviceBudget("h100_sxm", 80_000_000_000)
    raw = colocated_rank_memory(m, d, Q8, Q8, TP, B, 32768, budget,
                                block_size=16, e_produced=E_N,
                                draft_ctx_kv_path=FUSED_RAW)
    fb = colocated_rank_memory(m, d, Q8, Q8, TP, B, 32768, budget,
                               block_size=16, e_produced=E_N,
                               draft_ctx_kv_path=PER_LAYER_FALLBACK)
    fused_elems = N_LAYERS * 2 * KV_HEADS_RANK * D
    assert raw.steady_bytes() - fb.steady_bytes() == fused_elems * H * 1
    for rm in (raw, fb):
        inv = {a["alloc_id"]: a for a in rm.inventory()}
        assert inv["draft-w-dflash-k-norm-stack"]["bytes"] == \
            N_LAYERS * D * NORM_BYTES


def test_mxfp8_memory_gap_propagates():
    """MXFP8 block-scale 字节无 pin 证据 → 峰值完整性缺口，不能静默计 0。"""
    m, d = _models()
    rm = colocated_rank_memory(
        m, d, Q8, Q8, TP, B, 32768, DeviceBudget("h100_sxm", 80_000_000_000),
        block_size=16, e_produced=E_N,
        draft_ctx_kv_path=MXFP8_CONTEXT_PROJ)
    assert any("MXFP8" in c and "block-scale" in c for c in rm.unknown_components)
    assert rm.peak_complete() is False


def test_scenario_requires_explicit_path():
    """场景层显式配方：默认 raw 有 config 证据；4bit+raw 矛盾拒绝；
    unknown/非法值拒绝。"""
    ScenarioV2().validate()                       # 默认 raw（无 quant_config）
    q4 = dict(Q4)
    with pytest.raises(ValueError, match="fused_raw"):
        ScenarioV2(draft_quant=q4,
                   draft_ctx_kv_path=FUSED_RAW).validate()
    ScenarioV2(draft_quant=q4,
               draft_ctx_kv_path=PER_LAYER_FALLBACK).validate()
    ScenarioV2(draft_ctx_kv_path=MXFP8_CONTEXT_PROJ).validate()
    with pytest.raises(ValueError, match="draft_ctx_kv_path=unknown"):
        ScenarioV2(draft_ctx_kv_path="unknown").validate()
    with pytest.raises(ValueError, match="未知 draft_ctx_kv_path"):
        ScenarioV2(draft_ctx_kv_path="dense_w_guessed").validate()
