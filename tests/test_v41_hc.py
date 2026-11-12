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
import hashlib
import json
import math
from pathlib import Path

import pytest

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
    assert e.flops == 0 and e.act_write == int(5 * 16 * 2)
    # GEMM 640 + sqrsum 80 + prenorm 55；读 broadcast，容量仍为全量 fn。
    a = by_op["hc_attn_mix"]
    assert a.flops == 775
    assert a.weight_read == 8 * 8 * 4 and a.weight_capacity == 8 * 16 * 4
    # ffn 侧（及非首层 attn 侧）：IC=hc_dim=16 → 2·5·16·8=1280
    f = by_op["hc_ffn_mix"]
    assert f.flops == 1495  # 1280+160+55
    # gates16 + init8 + softmax20 + col8 + 2*(3−1)*8 = 84/token
    s = by_op["hc_attn_sinkhorn"]
    assert s.flops == 420 and s.statistic == "exact"
    # 首层无 post：collapse160+norm170；ffn post400+collapse160+norm170
    p = by_op["hc_attn_post_pre"]
    assert p.flops == 330 and p.statistic == "approximated"
    assert by_op["hc_ffn_post_pre"].flops == 730
    # 每层 2 边界 ×（mix+sinkhorn+post_pre）= 6 行 + 首层 expand = 7
    assert len(rows) == 7


def test_hc_rows_non_first_layer_no_expand():
    hc = {"mult": 2, "sinkhorn_iters": 3, "eps": 1e-6, "post_alpha": 2.0,
          "engram_layer_ids": [1]}
    rows = hc_boundary_rows("decode", 3, 4, 8, hc, Q16, first_layer=False)
    assert len(rows) == 6 and all(r.op != "hc_expand" for r in rows)
    by_op = {r.op: r for r in rows}
    assert by_op["hc_attn_mix"].flops == 1196  # 4*(256+32+11)


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
        flops += T * (2 * ic_a * mix + 2 * ic_a + mix + 3)
        flops += T * (2 * hc_dim * mix + 2 * hc_dim + mix + 3)
        flops += 2 * T * 1400
        flops += (1 if li == 0 else 2) * T * (mm*h + 2*mm*mm*h)
        flops += 2 * T * (2 * hc_dim + 4*h + 2)
    flops += T * (mm*h + 2*mm*mm*h) + 2*T*mm*h  # exit post + Triton acc+=
    assert sum(r.flops for r in hc_rows) == flops
    assert sum(r.flops for r in hc_rows) == 100968080
    # 40 层 ×6 + expand 1 + exit 2 = 243
    assert len(hc_rows) == 6 * L + 3
    exits = [r for r in hc_rows if r.layer == -1]
    assert [r.op for r in exits] == ["hc_exit_post", "hc_exit_collapse"]
    assert [r.flops for r in exits] == [184320, 40960]
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
                # R20：compressor/wk/norm 为 quant_config=None 的 bf16 常驻
                #（Q16 下与量化档同 2B，数值不变但来源固定）；norm 参数
                #（compressor norm 512 + indexer k_norm 128）入册
                terms += h * d * mlt + d * di + (d + di)
            if a.produces_index:
                # R20：weights_proj 亦 bf16（quant_config=None）
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
                          "index_weights_proj", "index_q_proj",
                          # R20：norm/k_norm 常驻由这两行持有（bf16）
                          "compress_pool", "index_k_store"))
            assert cap == allocs[str(li)], (tp, li)


def test_mhc_pin_blob_sha1_matches_tree():
    base = ROOT / "docs/rebuild-20261106/references"
    pin = json.loads((base / "vllm/mhc-pin.json").read_text(encoding="utf-8"))
    tree = {x["path"]: x["sha"] for x in json.loads(
        (base / "vllm-tree.json").read_text(encoding="utf-8"))["tree"]}
    expected = {f"vllm/model_executor/kernels/mhc/{name}.py" for name in
                ("__init__", "tilelang", "tilelang_kernels", "torch", "triton",
                 "warmup", "cpu", "aiter")}
    expected |= {"vllm/model_executor/layers/mhc.py",
                 "vllm/models/deepseek_v41/amd/mhc.py"}
    expected |= {f"vllm/models/deepseek_v41/nvidia/ops/{p}" for p in
                 ("mhc.py", "mega_mhc.py", "cute_dsl/all_reduce_mhc.py",
                  "cute_dsl/__init__.py")}
    assert {x["path"] for x in pin["files"]} == expected
    assert pin["executed"] is False
    for item in pin["files"]:
        data = (base / "vllm" / item["path"]).read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        assert blob == tree[item["path"]] == item["blob_sha1"]
        assert hashlib.sha256(data).hexdigest() == item["sha256"]
        assert len(data) == item["size"]


def _sinkhorn_op_oracle(m, iters):
    """独立逐算术事件计数，镜像 L395-440；归约 n−1，eps 按轴复用。"""
    events = []
    for _ in range(m):
        events.extend(["mul", "add", "sigmoid", "add"])  # pre
        events.extend(["mul", "add", "sigmoid", "mul"])  # post
    for _ in range(m):
        for _ in range(m):
            events.extend(["mul", "add"])               # comb logits
    for _ in range(m):
        for _ in range(m - 1):
            events.append("max")
        for _ in range(m):
            events.extend(["sub", "exp"])
        for _ in range(m - 1):
            events.append("add")
        for _ in range(m):
            events.extend(["div", "add"])

    def normalize_axis():
        for _ in range(m):
            for _ in range(m - 1):
                events.append("add")
            events.append("add")  # denominator + eps shared along this axis
            for _ in range(m):
                events.append("div")

    normalize_axis()  # initial column normalization
    for _ in range(iters - 1):
        normalize_axis()  # row
        normalize_axis()  # column
    return len(events)


@pytest.mark.parametrize("m,iters", [(2, 1), (2, 3), (3, 2), (4, 1), (4, 20)])
def test_sinkhorn_op_oracle(m, iters):
    count = _sinkhorn_op_oracle(m, iters)
    # 独立化简式：(4*iters+6)m² + 6m。
    assert count == (4 * iters + 6) * m * m + 6 * m
    rows = hc_boundary_rows("decode", 1, 3, 8,
                            {"mult": m, "sinkhorn_iters": iters}, Q16)
    assert [r.flops for r in rows if r.op.endswith("sinkhorn")] == [3 * count] * 2
    if (m, iters) == (4, 20):
        assert count == 1400


def _sinkhorn_kernel_mirror(logits, eps, iters):
    """只执行 stdlib 数学，非 pinned kernel/torch/CPU forward。"""
    m = len(logits)
    cm = [list(row) for row in logits]
    for j in range(m):
        peak = max(cm[j])
        cm[j] = [math.exp(x - peak) for x in cm[j]]
        denom = sum(cm[j])
        cm[j] = [x / denom + eps for x in cm[j]]

    def axis(dim):
        sums = ([sum(row) for row in cm] if dim == 1 else
                [sum(cm[j][k] for j in range(m)) for k in range(m)])
        for j in range(m):
            for k in range(m):
                cm[j][k] /= sums[j if dim == 1 else k] + eps

    axis(0)
    for _ in range(iters - 1):
        axis(1)
        axis(0)
    return cm


@pytest.mark.parametrize("iters", [1, 2, 20])
def test_sinkhorn_numeric_order_stdlib(iters):
    # 非对称 logits + 可见 eps，避免行列更新颠倒仍碰巧通过。
    logits = [[1.0, -2.0, 0.5, 3.0], [0.2, 2.0, -1.0, 0.0],
              [-3.0, 1.0, 4.0, -0.5], [2.0, 0.0, -2.0, 1.0]]
    eps = 0.01
    softmax = []
    for row in logits:
        z = [math.exp(x) for x in row]
        softmax.append([x / sum(z) + eps for x in z])

    def rows_normalized(matrix):
        return [[x / (sum(row) + eps) for x in row] for row in matrix]

    def transpose(matrix):
        return [list(row) for row in zip(*matrix)]

    def columns_normalized(matrix):
        return transpose(rows_normalized(transpose(matrix)))

    expected = columns_normalized(softmax)
    wrong = rows_normalized(softmax)  # 故意把初始列更新误读为行
    for _ in range(iters - 1):
        expected = columns_normalized(rows_normalized(expected))
        wrong = rows_normalized(columns_normalized(wrong))
    actual = _sinkhorn_kernel_mirror(logits, eps, iters)
    for a, b in zip(actual, expected):
        assert a == pytest.approx(b, abs=1e-14)
    assert max(abs(actual[j][k] - wrong[j][k])
               for j in range(4) for k in range(4)) > 1e-5


def test_first_boundary_has_no_post_or_pre_mix_read():
    hc = {"mult": 2, "sinkhorn_iters": 3}
    first = hc_boundary_rows("decode", 0, 5, 8, hc, Q16, first_layer=True)
    other = hc_boundary_rows("decode", 1, 5, 8, hc, Q16)
    a = next(r for r in first if r.op == "hc_attn_post_pre")
    b = next(r for r in other if r.op == "hc_attn_post_pre")
    assert b.flops - a.flops == 400  # post 5*(2*8+2*4*8)
    assert a.act_read == 160 and a.act_write == 80
    assert b.act_read - a.act_read == 400  # post read360 + pre_mix40
    assert b.act_write - a.act_write == 160


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("first", [False, True])
def test_hc_unfused_byte_conservation_oracle(bits, first):
    T, m, h, mix, b = 5, 2, 8, 8, bits // 8
    quant = {**Q16, "act": bits}
    rows = hc_boundary_rows("decode", 0, T, h,
                            {"mult": m, "sinkhorn_iters": 3}, quant, first)
    for side in ("attn", "ffn"):
        entry = first and side == "attn"
        part = [r for r in rows if r.op.startswith(f"hc_{side}_")]
        # 从 kernel 张量清单独立求和（含 norm_weight，不含独立参数）。
        post_r = 0 if entry else (m*m*4 + m*4 + m*h*b + h*b)
        post_w = 0 if entry else m*h*b
        gemm_r, gemm_w = (h if entry else m*h)*b, mix*4 + 4
        epi_r = mix*4 + 4 + m*h*b + (0 if entry else m*4)
        epi_w = m*4 + m*m*4 + m*4 + h*b
        assert sum(r.act_read for r in part) == T*(post_r+gemm_r+epi_r)
        assert sum(r.act_write for r in part) == T*(post_w+gemm_w+epi_w)
        # fn/broadcast 整批读一次；epilogue scale/base 与 norm 逐 token 读。
        assert sum(r.weight_read for r in part) == (
            mix*(h if entry else m*h)*4 + T*((mix+3)*4 + h*b))


def test_hc_capacity_broadcast_is_derived_replicated_buffer():
    model = _model()
    for tp in (1, 2, 4, 8):
        allocs = target_weight_allocations(model, Q16, tp)
        buffers = [a for a in allocs if "-w-mhc-broadcast-" in a.alloc_id]
        assert len(buffers) == 1
        assert buffers[0].alloc_id == "target-w-mhc-broadcast-L0"
        assert buffers[0].bytes == 24*5120*4
        assert "hc_attn_fn" in buffers[0].live_note
        rows = build_model_work(model, "decode", 1, 32768, 1, Q16, tp)
        for li in range(40):
            cap = sum(r.weight_capacity for r in rows
                      if r.layer == li and r.op.startswith("hc_"))
            allocation = next(a for a in allocs if a.alloc_id == f"target-w-mhc-L{li}")
            assert cap == allocation.bytes  # fn + base + scale，broadcast 非参数
        first_mix = next(r for r in rows if r.layer == 0 and r.op == "hc_attn_mix")
        assert first_mix.weight_read == buffers[0].bytes


def test_replay_preserves_engram_and_non_hc_rows():
    from simverse.scenario import ScenarioV2
    from simverse.validation.registry import build_registry
    folder = ROOT / "docs/evidence/r05-mhc"
    before = json.loads((folder / "before.json").read_text(encoding="utf-8"))
    scenario = ScenarioV2.from_json(json.loads((folder / "scenario.json").read_text(encoding="utf-8")))
    current = [r.to_json() for r in build_model_work(
        _model(), "target_decode", 1, 32768, 1, scenario.quant, tp=8)]
    # 排除 stage 名差异；冻结 R04 行与非 hc/非 R20 审计行（含 final_norm/
    # head）。R20（2026-11-12）重写的 CSA2 索引/压缩行由
    # tests/test_v41_csa2.py 的 replay 归因单独冻结（docs/evidence/csa2）。
    AUDITED = {
        "index_weights_proj", "index_q_rope", "index_select", "index_score",
        "index_topk", "index_fill", "sparse_index_build", "candidate_publish",
        "candidate_select", "candidate_mask", "compress_proj", "compress_norm",
        "compress_pool", "compressed_rope", "compressed_cache_write",
        "compressed_insert", "index_k_proj", "index_k_norm", "index_k_rope",
        "index_cache_write", "index_k_store", "swa_cache_write",
        "sparse_attention"}
    def non_hc(rows):
        return [{k: v for k, v in r.items() if k != "stage"}
                for r in rows
                if not r["op"].startswith("hc_") and r["op"] not in AUDITED]
    assert non_hc(current) == non_hc(before["work"]["target_decode"]["rows"])
    record = build_registry()["ev-vllm-v41-mhc"]
    assert record.kind == "source-checked" and record.executed is False
    assert len(record.content_sha256) == 64
