"""V4.1 mHC（mega hyper-connections）语义与计数（2026-11-10 补齐轮）。

证据链：vllm models/deepseek_v41/nvidia/model.py pin @f358f057
（source-checked；hc 参数几何/FP32/每层 2 边界/首层 hc_expand/Engram 层分立）。
测试分层：
  * 单元手算锚点（小尺寸独立推导，不 import 生产公式）；
  * 真实 V4.1 配置的整模型独立复算；
  * TP 不变性（hc 全宽流不切分）与权重复制守恒（count 行 = memory 账本）；
  * v41 层不再重复计通用 norm/add 行。
Engram 8 键已 mapped（2026-11-11 R04，common/engram.py pin blob ba4ceea6）：
engram 层（1/14）现在额外出现 engram_* 4/3 行（tests/test_v41_engram.py
独立 oracle 覆盖），hc 边界行构成不变（仍 6 行/层）。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simverse.count.work import (build_model_work, hc_boundary_rows,  # noqa: E402
                                 totals)
from simverse.lowering.memory import target_weight_allocations  # noqa: E402
from simverse.model_ir.catalog import build_catalog  # noqa: E402

Q16 = {"dense_w": 16, "expert_w": 16, "act": 16, "kv": 16, "state": 32}

_CAT = build_catalog(include_internal=True)


def _model():
    return _CAT["models"]["deepseek_v41_flash"]


# ---------------------------------------------------------------------------
# 单元：小尺寸手算锚点（独立推导）
# ---------------------------------------------------------------------------

def test_hc_boundary_rows_hand_anchor():
    """m=2, h=8, iters=3, T=5（mix=8, hc_dim=16）逐行手算。"""
    hc = {"mult": 2, "sinkhorn_iters": 3, "eps": 1e-6, "post_alpha": 2.0,
          "engram_layer_ids": []}
    rows = hc_boundary_rows("decode", 0, 5, 8, hc, Q16, first_layer=True)
    by_op = {r.op: r for r in rows}
    # 首层：hc_expand 广播 2 流
    e = by_op["hc_expand"]
    assert e.flops == 5 * 16 and e.act_write == int(5 * 16 * 2)
    # 首层 attn 侧系数 GEMM：IC=h=8（广播嵌入）→ 2·5·8·8=640；权重 FP32 8×16×4
    a = by_op["hc_attn_mix"]
    assert a.flops == 2 * 5 * 8 * 8
    assert a.weight_read == 8 * 16 * 4 and a.weight_capacity == 8 * 16 * 4
    # ffn 侧（及非首层 attn 侧）：IC=hc_dim=16 → 2·5·16·8=1280
    f = by_op["hc_ffn_mix"]
    assert f.flops == 2 * 5 * 16 * 8
    # sinkhorn：T·mix·iters·5 = 5·8·3·5；标 approximated
    s = by_op["hc_attn_sinkhorn"]
    assert s.flops == 5 * 8 * 3 * 5 and s.statistic == "approximated"
    # post+pre：4·T·hc_dim + T·(4h+2) = 320 + 170
    p = by_op["hc_attn_post_pre"]
    assert p.flops == 4 * 5 * 16 + 5 * (4 * 8 + 2)
    # 每层 2 边界 ×（mix+sinkhorn+post_pre）= 6 行 + 首层 expand = 7
    assert len(rows) == 7


def test_hc_rows_non_first_layer_no_expand():
    hc = {"mult": 2, "sinkhorn_iters": 3, "eps": 1e-6, "post_alpha": 2.0,
          "engram_layer_ids": [1]}
    rows = hc_boundary_rows("decode", 3, 4, 8, hc, Q16, first_layer=False)
    assert len(rows) == 6 and all(r.op != "hc_expand" for r in rows)
    by_op = {r.op: r for r in rows}
    assert by_op["hc_attn_mix"].flops == 2 * 4 * 16 * 8   # IC=hc_dim（非首层）


# ---------------------------------------------------------------------------
# 整模型独立复算（真实 V4.1：h=5120, m=4, 40 层, iters=20）
# ---------------------------------------------------------------------------

def test_v41_hc_totals_independently_recomputed():
    m = _model()
    rows = build_model_work(m, "decode", 1, 1024, 1, Q16, tp=1)
    hc_rows = [r for r in rows if r.op.startswith("hc_")]
    # 独立复算（不 import 生产公式）：
    h, mm, iters, L = 5120, 4, 20, 40
    mix, hc_dim = (2 + mm) * mm, mm * h          # 24, 20480
    T = 1
    flops = 0
    for li in range(L):
        ic_a = h if li == 0 else hc_dim          # 首层 attn 侧读广播嵌入
        flops += 2 * T * ic_a * mix              # attn mix
        flops += 2 * T * hc_dim * mix            # ffn mix
        flops += 2 * (T * mix * iters * 5)       # sinkhorn ×2
        flops += 2 * (4 * T * hc_dim + T * (4 * h + 2))   # post_pre ×2
    flops += T * hc_dim                          # 首层 expand
    assert sum(r.flops for r in hc_rows) == flops
    # 行数：40 层 ×6 + expand 1 = 241
    assert len(hc_rows) == 6 * L + 1
    # Engram 层（1/14）边界算子构成与常规层相同（分立执行不改变总量）。
    # 归因更新（2026-11-11 R04）：Engram 注入已建模（engram.py pin），engram
    # 层额外出现 engram_* 行——首 engram 层 L1 挂 4 行（hash/lookup/wkv/
    # post_wkv），L14 无批级哈希行只 3 行；hc 边界行仍 6 行/层不变。
    for li in (1, 14):
        hc_layer = [r for r in hc_rows if r.layer == li]
        eng_layer = [r for r in rows if r.layer == li and r.op.startswith("engram_")]
        assert len(hc_layer) == 6
        if li == 1:
            assert [r.op for r in eng_layer] == [
                "engram_hash", "engram_lookup", "engram_wkv", "engram_post_wkv"]
        else:
            assert [r.op for r in eng_layer] == [
                "engram_lookup", "engram_wkv", "engram_post_wkv"]
    # 非 engram 层无 engram 行
    assert not any(r.op.startswith("engram_") for r in rows
                   if r.layer not in (1, 14))


def test_v41_no_generic_norm_add_double_counting():
    """v41 层的 norm/残差已融合进 mHC 边界核（F_MHC_POST_PRE）→ 不再另计。"""
    m = _model()
    rows = build_model_work(m, "decode", 1, 1024, 1, Q16, tp=1)
    v41_layers = {l.index for l in m.layers}
    for r in rows:
        if r.layer in v41_layers:
            assert r.op not in ("attn_norm", "mlp_norm", "attn_add", "mlp_add"), r.op
    # 对照：GQA 家族仍有通用 norm/add 行（未误删）
    gqa = _CAT["models"]["tiny_gqa"]
    grows = build_model_work(gqa, "decode", 1, 64, 1, Q16, tp=1)
    assert any(r.op == "attn_norm" for r in grows)


# ---------------------------------------------------------------------------
# TP 与容量守恒
# ---------------------------------------------------------------------------

def test_hc_rows_tp_invariant_full_width_stream():
    """hc 全宽流不随 TP 切分（attention 内部分片；边界算子每卡全宽执行）。"""
    m = _model()
    r1 = [r for r in build_model_work(m, "decode", 1, 1024, 1, Q16, tp=1)
          if r.op.startswith("hc_")]
    r8 = [r for r in build_model_work(m, "decode", 1, 1024, 1, Q16, tp=8)
          if r.op.startswith("hc_")]
    assert len(r1) == len(r8)
    for a, b in zip(r1, r8):
        assert (a.op, a.layer, a.flops, a.weight_read) == \
               (b.op, b.layer, b.flops, b.weight_read)


def test_hc_weights_replicated_under_tp():
    """hc 权重 FP32 每卡复制：tp=1 与 tp=8 的 mhc 分配字节相等。"""
    m = _model()
    def hc_bytes(tp):
        return sum(a.bytes for a in target_weight_allocations(m, Q16, tp)
                   if a.alloc_id.endswith("-w-mhc-L0"))
    assert hc_bytes(1) == hc_bytes(8)
    # 手算：mix=24, hc_dim=20480 → 2×(24·20480·4 + 24·4 + 3·4) = 3,932,376 B
    assert hc_bytes(1) == 2 * (24 * 20480 * 4 + 24 * 4 + 3 * 4)


def test_v41_tp_must_divide_o_groups():
    from simverse.lowering import validate_parallel
    m = _model()
    assert validate_parallel(m, 8, 1).valid
    bad = validate_parallel(m, 16, 1)
    assert not bad.valid and any("o_groups" in e for e in bad.errors)


def test_v41_oa_proj_group_count_absolute_oracle_tp1_2_4_8():
    """F2 独立绝对值 oracle：o_a 每 rank ng=o_groups/tp 组，不许只算一组。

    逐 TP 手算：2·T·(heads·d/groups)·orank·ng FLOPs；
    权重/容量 = (heads·d/groups)·orank·ng·2B；激活读/写同样乘 ng。
    """
    m = _model()
    h, heads, d, groups, orank = 5120, 64, 512, 8, 1024
    group_in = heads * d // groups          # 4096
    b = 2                                  # Q16
    T = 1
    for tp in (1, 2, 4, 8):
        ng = groups // tp
        rows = [r for r in build_model_work(m, "decode", 1, 1024, 1, Q16, tp)
                if r.op == "o_a_proj"]
        assert len(rows) == 40
        for r in rows:
            assert r.flops == 2 * T * group_in * orank * ng
            assert r.weight_read == group_in * orank * ng * b
            assert r.act_read == T * group_in * ng * b
            assert r.act_write == T * orank * ng * b
            assert r.weight_capacity == group_in * orank * ng * b
        # 每层一组值；TP1 是 TP8 的 8 倍（同一层、同一 token）
        assert sum(r.flops for r in rows) == 40 * 2 * T * group_in * orank * ng


def test_v41_oa_proj_memory_absolute_oracle_tp1_2_4_8():
    """memory 与 work 同步乘 ng：独立复算每层 dense 权重字节。"""
    m = _model()
    kvs = m.extras["kv_spec"]
    h, heads, d, groups, orank = 5120, 64, 512, 8, 1024
    qrank, di, ih = 1280, 128, 32
    for tp in (1, 2, 4, 8):
        ng = groups // tp
        heads_tp = heads // tp
        expect = 0
        for layer in m.layers:
            a = layer.csa2
            mlt = 2 if a.ratio == 2 else 1
            terms = (h * qrank + h * d + qrank * heads_tp * d
                     + (heads * d // groups) * orank * ng
                     + ng * orank * h)
            if a.produces_kv:
                terms += h * d * mlt + d * di
            if a.produces_index:
                terms += h * ih + qrank * ih * di
            expect += terms * 2          # Q16 dense
        got = sum(a.bytes for a in target_weight_allocations(m, Q16, tp)
                  if "-w-dense-L" in a.alloc_id)
        assert got == expect, (tp, got, expect)
        # 如果 o_a 漏乘 ng，恰好少这一段；TP8 无差异
        assert got - expect == 0


def test_v41_attention_weight_conservation_count_vs_memory():
    """count 行 weight_capacity 与 memory dense 分配逐层相等（同 tp/精度口径）。

    用 ctx=32768（全层走全候选打分路径）；短上下文 eager 旁路层计数行
    index_q_proj 容量为 0（不产生该 GEMM），而权重常驻仍由 memory 计——
    两量口径不同（阶段用量 vs 常驻），旁路分支不参与守恒断言。
    """
    m = _model()
    for tp in (1, 8):
        rows = build_model_work(m, "decode", 1, 32768, 1, Q16, tp)
        assert any(r.op == "index_q_proj" and r.weight_capacity > 0
                   for r in rows)   # 前提：无旁路层
        allocs = {(a.alloc_id.split("-L")[-1]): a.bytes
                  for a in target_weight_allocations(m, Q16, tp)
                  if "-w-dense-L" in a.alloc_id}
        for li in range(40):
            cap = sum(r.weight_capacity for r in rows
                      if r.layer == li and r.op in (
                          "q_a_proj", "kv_proj", "q_b_proj", "o_a_proj",
                          "o_b_proj", "compress_proj", "index_k_proj",
                          "index_weights_proj", "index_q_proj"))
            assert cap == allocs[str(li)], (tp, li)
