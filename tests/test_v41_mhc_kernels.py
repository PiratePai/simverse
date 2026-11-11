"""V4.1 mHC kernel 计数独立 oracle（2026-11-12 R05）。

证据链：vllm @f358f0578a83f4e1fe6fc49a492d3ae9362de8e5 的 mHC kernel 源码
pin（model_executor/kernels/mhc/* + layers/mhc.py + nvidia/ops/{mhc,
mega_mhc}.py + cute_dsl/all_reduce_mhc.py + amd/mhc.py + utils/deep_gemm.py，
12 核心文件 + 同批 aiter.py/cpu.py/cute_dsl__init__.py；git blob sha1 逐项
对 docs/rebuild-20261106/references/vllm-tree.json 校验；source-checked，
未执行）。本文件**不 import 生产公式当 oracle**：sinkhorn/post_pre 计数
按 tilelang_kernels.py 源码逐式独立推导。

与主控（上级）审查推导的记录差异（以源码为准，见
docs/validation/v41-mhc-sinkhorn-20261112.md §3/§7）：
  * sinkhorn 的 rms 步 `T.rsqrt(rms / rms_numel + rms_eps)`
    （tilelang_kernels.py L385）按计数约定（加/减/乘/除各 1、超越函数 1）
    是 3 op（÷ + eps 加 + rsqrt），主控分解写 2 → 常数项 +2 应为 +3；
  * post_pre 的 RMSNorm rsqrt（L507 `rsqrt(sumsq / hidden_size + norm_eps)`）
    同理 3 op 而非 1 → RMSNorm 5h 应为 5h+2；
本测试锚点按源码计数（m=2,h=8,R=3：sinkhorn 95/token、post_pre 154/token，
主控值 94/152）。

测试分层：
  * pin 完整性（blob sha1 防漂移）；
  * 小尺寸手算锚点（mix/sinkhorn/post_pre/expand/exit 逐行绝对值）；
  * 负例/结构（R=1 循环 0 次、精确标注、首层 broadcast 权重读、
    非 v41 家族无 hc/exit 行、engram 层构成、TP 逐字段相等、守恒）；
  * 真实 V4.1 整模型独立复算（行数 243）。
"""
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simverse.count.work import (build_model_work, hc_boundary_rows,  # noqa: E402
                                 hc_exit_rows, totals)
from simverse.lowering.memory import target_weight_allocations  # noqa: E402
from simverse.model_ir.catalog import build_catalog  # noqa: E402

Q16 = {"dense_w": 16, "expert_w": 16, "act": 16, "kv": 16, "state": 32}

REF_ROOT = (ROOT / "docs" / "rebuild-20261106" / "references" / "vllm")
TREE_JSON = (ROOT / "docs" / "rebuild-20261106" / "references" /
             "vllm-tree.json")

# R05 pin 集：12 核心文件（主控审查指定）+ 同批 3 文件（aiter/cpu/
# cute_dsl__init__，证据 ev-vllm-v41-mhc-kernels 同条登记为 12+3）
PIN_FILES = (
    "vllm/model_executor/kernels/mhc/tilelang.py",
    "vllm/model_executor/kernels/mhc/tilelang_kernels.py",
    "vllm/model_executor/kernels/mhc/triton.py",
    "vllm/model_executor/kernels/mhc/torch.py",
    "vllm/model_executor/kernels/mhc/warmup.py",
    "vllm/model_executor/kernels/mhc/__init__.py",
    "vllm/model_executor/layers/mhc.py",
    "vllm/models/deepseek_v41/nvidia/ops/mhc.py",
    "vllm/models/deepseek_v41/nvidia/ops/mega_mhc.py",
    "vllm/models/deepseek_v41/nvidia/ops/cute_dsl/all_reduce_mhc.py",
    "vllm/models/deepseek_v41/amd/mhc.py",
    "vllm/utils/deep_gemm.py",
    "vllm/model_executor/kernels/mhc/aiter.py",
    "vllm/model_executor/kernels/mhc/cpu.py",
    "vllm/models/deepseek_v41/nvidia/ops/cute_dsl/__init__.py",
)

_CAT = build_catalog(include_internal=True)


def _model():
    return _CAT["models"]["deepseek_v41_flash"]


# ---------------------------------------------------------------------------
# 独立 oracle：按 tilelang_kernels.py 源码逐式推导（不 import 生产公式）
# ---------------------------------------------------------------------------

def _sinkhorn_ops(m: int, iters: int, mix: int) -> int:
    """系数 epilogue 每 token 操作数（mhc_pre_big_fuse_with_norm_tilelang
    L383-441）。逐式（初等运算各 1 op；exp/sigmoid/rsqrt 1；归约 n−1）：

      rms 3（÷rms_numel + 加 eps + rsqrt，L385）
      + mixes×rms：mix 次乘（L390）
      + pre 门 4m（乘 scale+加 base+sigmoid+加 pre_eps，L397-400）
      + post 门 4m（乘+加+sigmoid+乘 post_alpha，L401-406）
      + comb affine 2m²（乘 scale + 加 base，L407-411）
      + 首轮 softmax 6m²−2m（max m(m−1) 比较 L417 + 减 m² + exp m² L419
        + 行和 m(m−1) L420 + 除 m² + 加 eps m² L422）
      + 列归一 ×1 2m²（列和 m(m−1) L424 + eps 加 m + 除 m² L426）
      + 循环 (R−1) 次 ×(行归一 2m² + 列归一 2m²)（L433-440）
      → m²(4R+6) + 6m + mix + 3
    """
    return (3 + mix + 8 * m + 2 * m * m + (6 * m * m - 2 * m) + 2 * m * m
            + (iters - 1) * 4 * m * m)


def _post_pre_ops(m: int, h: int) -> int:
    """post+collapse+RMSNorm 每 token 操作数。

      post 核 h(2m²+m)（mhc_post_tilelang L856-861：每 h 元素 m 乘
      （post·x）+ m² 乘 + m² 加（comb·residual）；T≤32 融合路径折入
      prenorm GEMM，操作量相同 mhc_fused_tilelang L766-780）
      + collapse 2mh（L478-483：每 h 元素 m·(乘+加)）
      + 两遍 RMSNorm 5h+2（sumsq 累积 2h L495-501 + 归约 h−1 L505
        + rsqrt 3（÷h + 加 eps + rsqrt，L507）+ 缩放 2h L517-522）
    """
    return h * (2 * m * m + m) + 2 * m * h + (2 * h + (h - 1) + 3 + 2 * h)


# ---------------------------------------------------------------------------
# pin 完整性（防漂移；仿 test_v41_engram.py L114-122）
# ---------------------------------------------------------------------------

def test_mhc_kernel_pins_blob_sha1_match_tree():
    """15 个 pin 文件的 git blob sha1 逐一对 vllm-tree.json 校验。"""
    tree = json.loads(TREE_JSON.read_text(encoding="utf-8"))
    assert tree["sha"] == "f358f0578a83f4e1fe6fc49a492d3ae9362de8e5"
    entries = {e["path"]: e for e in tree["tree"]}
    for rel in PIN_FILES:
        data = (REF_ROOT / rel).read_bytes()
        blob = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
        e = entries[rel]
        assert blob == e["sha"], rel
        assert len(data) == e["size"], rel


# ---------------------------------------------------------------------------
# 小尺寸手算锚点（m=2, h=8, R=3, T=5, Q16 → b_a=2；mix=8, hc_dim=16）
# ---------------------------------------------------------------------------

_SMALL = {"mult": 2, "sinkhorn_iters": 3, "eps": 1e-6, "post_alpha": 2.0,
          "engram_layer_ids": []}


def test_hand_anchor_first_layer_rows():
    """首层逐行手算（独立推导，见模块 docstring 的源码差异记录）：

      expand:  flops=5·16=80；ar=5·8·2=80；aw=5·16·2=160
      attn mix: flops=5·(2·8·8+2·8)=720（GEMM 640 + 核内 sqrsum 80）；
                wr=8·8·4=256（broadcast 派生张量 [mix,h] fp32）；
                cap=8·16·4=512（fn 参数本体）
      ffn mix:  flops=5·(2·16·8+2·16)=1440；wr=cap=512
      sinkhorn: 5·95=475（m²(4R+6)+6m+mix+3 = 4·18+12+8+3）
      post_pre: 5·154=770（h(2m²+3m+5)+2 = 8·19+2）
    """
    rows = hc_boundary_rows("decode", 0, 5, 8, _SMALL, Q16, first_layer=True)
    by_op = {r.op: r for r in rows}
    e = by_op["hc_expand"]
    assert e.flops == 5 * 16 and e.act_read == 80 and e.act_write == 160
    a = by_op["hc_attn_mix"]
    assert a.flops == 5 * (2 * 8 * 8 + 2 * 8) == 720
    assert a.weight_read == 8 * 8 * 4 and a.weight_capacity == 8 * 16 * 4
    assert a.act_read == 5 * 8 * 2 and a.act_write == 5 * 8 * 4
    f = by_op["hc_ffn_mix"]
    assert f.flops == 5 * (2 * 16 * 8 + 2 * 16) == 1440
    assert f.weight_read == f.weight_capacity == 8 * 16 * 4
    for side in ("attn", "ffn"):
        s = by_op[f"hc_{side}_sinkhorn"]
        assert s.flops == 5 * _sinkhorn_ops(2, 3, 8) == 5 * 95 == 475
        assert s.statistic == "exact"
        p = by_op[f"hc_{side}_post_pre"]
        assert p.flops == 5 * _post_pre_ops(2, 8) == 5 * 154 == 770
        assert p.statistic == "exact"
    assert len(rows) == 7


def test_hand_anchor_exit_rows():
    """模型出口两行手算（T=5, h=8, m=2, hc_dim=16, b_a=2）：

      exit_post:     flops=5·8·(2·4+2)=400；ar=(5·16+5·8)·2=240；
                     aw=5·16·2=160
      exit_collapse: flops=5·2·2·8=160；ar=5·16·2+5·2·4=200；aw=5·8·2=80
    """
    rows = hc_exit_rows("decode", 5, 8, _SMALL, Q16)
    assert [r.op for r in rows] == ["hc_exit_post", "hc_exit_collapse"]
    assert all(r.layer == -1 for r in rows)
    post, col = rows
    assert post.flops == 5 * 8 * (2 * 2 * 2 + 2) == 400
    assert post.act_read == (5 * 16 + 5 * 8) * 2 == 240
    assert post.act_write == 5 * 16 * 2 == 160
    assert col.flops == 5 * 2 * 2 * 8 == 160
    assert col.act_read == 5 * 16 * 2 + 5 * 2 * 4 == 200
    assert col.act_write == 5 * 8 * 2 == 80


# ---------------------------------------------------------------------------
# 负例 / 结构
# ---------------------------------------------------------------------------

def test_sinkhorn_r1_loop_zero_and_step_exact_4m2():
    """R=1 → 循环 0 次：m²·10+6m+mix+3；R=2 与 R=1 之差恰 4m²·T。"""
    base = {**_SMALL, "sinkhorn_iters": 1}
    two = {**_SMALL, "sinkhorn_iters": 2}
    T, m, mix = 5, 2, 8
    r1 = hc_boundary_rows("decode", 3, T, 8, base, Q16)
    r2 = hc_boundary_rows("decode", 3, T, 8, two, Q16)
    s1 = next(r for r in r1 if r.op == "hc_attn_sinkhorn")
    s2 = next(r for r in r2 if r.op == "hc_attn_sinkhorn")
    assert s1.flops == T * (m * m * 10 + 6 * m + mix + 3) == 5 * 63
    assert s2.flops - s1.flops == 4 * m * m * T == 80


def test_sinkhorn_not_approximated_and_formula_registry_text():
    """statistic 不再是 approximated；公式注册表携带精确分解与计数约定。"""
    rows = hc_boundary_rows("decode", 3, 5, 8, _SMALL, Q16)
    for r in rows:
        assert r.statistic != "approximated"
    from simverse import formula_registry as FR
    sink = FR.get(FR.F_MHC_SINKHORN)
    assert "4R+6" in sink["expr"] and "R−1" in sink["expr"]
    assert "超越函数" in sink["source"] or "归约" in sink["source"]
    pp = FR.get(FR.F_MHC_POST_PRE)
    assert "2m²+3m+5" in pp["expr"] and "5h+2" in pp["expr"]
    mix = FR.get(FR.F_MHC_MIX)
    assert "sqrsum" in mix["expr"] and "2·IC" in mix["expr"]
    for fid in (FR.F_MHC_EXIT_POST, FR.F_MHC_EXIT_COLLAPSE):
        assert FR.get(fid)["expr"]


def test_first_layer_attn_mix_reads_broadcast_tensor():
    """首层 attn weight_read=mix·h·4（broadcast 派生张量）；非首层/ffn 侧
    =mix·hc_dim·4（fn 参数本体）；容量恒 mix·hc_dim·4。"""
    m, h = 4, 5120
    hc = {"mult": m, "sinkhorn_iters": 20, "eps": 1e-6, "post_alpha": 2.0,
          "engram_layer_ids": []}
    mix, hc_dim = (2 + m) * m, m * h
    first = {r.op: r for r in
             hc_boundary_rows("decode", 0, 1, h, hc, Q16, first_layer=True)}
    later = {r.op: r for r in
             hc_boundary_rows("decode", 7, 1, h, hc, Q16, first_layer=False)}
    assert first["hc_attn_mix"].weight_read == mix * h * 4      # 491,520
    assert first["hc_ffn_mix"].weight_read == mix * hc_dim * 4
    assert later["hc_attn_mix"].weight_read == mix * hc_dim * 4
    for r in list(first.values()) + list(later.values()):
        if r.op.endswith("_mix"):
            assert r.weight_capacity == mix * hc_dim * 4


def test_exit_rows_structure_and_non_v41_family_has_none():
    """exit 行恰 2 条、layer=-1、位于 final_norm 之前；tiny_gqa 无 hc/exit 行。"""
    m = _model()
    rows = build_model_work(m, "decode", 1, 1024, 1, Q16, tp=1)
    tail = [r for r in rows if r.layer == -1]
    exits = [r for r in tail if r.op.startswith("hc_exit_")]
    assert [r.op for r in exits] == ["hc_exit_post", "hc_exit_collapse"]
    # exit×2 紧邻 final_norm 之前（其后 final_norm → lm_head → embed）
    assert [r.op for r in tail[:3]] == ["hc_exit_post", "hc_exit_collapse",
                                        "final_norm"]
    gqa = _CAT["models"]["tiny_gqa"]
    grows = build_model_work(gqa, "decode", 1, 64, 1, Q16, tp=1)
    assert not any(r.op.startswith("hc_") for r in grows)


def test_engram_layers_hc_composition_unchanged():
    """engram 层（1/14）hc 行构成不变（每层 6 行；分立不改变计数）。"""
    m = _model()
    rows = build_model_work(m, "decode", 1, 1024, 1, Q16, tp=1)
    hc = [r for r in rows if r.op.startswith("hc_") and r.layer >= 0]
    for li in (1, 14):
        layer_rows = [r for r in hc if r.layer == li]
        assert len(layer_rows) == 6
        assert {r.op for r in layer_rows} == {
            "hc_attn_mix", "hc_attn_sinkhorn", "hc_attn_post_pre",
            "hc_ffn_mix", "hc_ffn_sinkhorn", "hc_ffn_post_pre"}
    # 与常规层逐字段相同（除 layer 号）
    ref = {r.op: r for r in hc if r.layer == 2}
    for li in (1, 14):
        for r in (x for x in hc if x.layer == li):
            ref_row = ref[r.op]
            assert (r.op, r.flops, r.weight_read, r.weight_capacity,
                    r.act_read, r.act_write) == (
                ref_row.op, ref_row.flops, ref_row.weight_read,
                ref_row.weight_capacity, ref_row.act_read, ref_row.act_write)


def test_hc_rows_tp_invariant_all_fields():
    """TP8 与 TP1 的 hc 行（含 exit 行）逐字段相等（全宽流不切分）。"""
    m = _model()
    r1 = [r for r in build_model_work(m, "decode", 1, 1024, 1, Q16, tp=1)
          if r.op.startswith("hc_")]
    r8 = [r for r in build_model_work(m, "decode", 1, 1024, 1, Q16, tp=8)
          if r.op.startswith("hc_")]
    assert len(r1) == len(r8) == 243
    for a, b in zip(r1, r8):
        assert a.to_json() == b.to_json()


def test_mhc_memory_broadcast_tp_invariant_and_conservation():
    """broadcast 分配每 rank 复制 = mix·h·4；count↔memory 守恒：
    mhc 层对象 − (base+scale) == 该层 mix 行 capacity 和；
    broadcast 对象 == 首层 attn mix 行 weight_read。"""
    m = _model()
    mix, hc_dim, h = 24, 20480, 5120
    base_scale = 2 * (mix * 4 + 3 * 4)     # 每侧 base[mix] fp32 + scale[3] fp32
    for tp in (1, 8):
        allocs = {a.alloc_id: a.bytes for a in target_weight_allocations(
            m, Q16, tp) if "-w-mhc" in a.alloc_id}
        assert allocs["target-w-mhc-broadcast-L0"] == mix * h * 4 == 491_520
        rows = build_model_work(m, "decode", 1, 1024, 1, Q16, tp)
        for li in range(40):
            cap = sum(r.weight_capacity for r in rows
                      if r.layer == li and r.op in ("hc_attn_mix", "hc_ffn_mix"))
            assert cap == allocs[f"target-w-mhc-L{li}"] - base_scale, (tp, li)
        l0 = next(r for r in rows if r.layer == 0 and r.op == "hc_attn_mix")
        assert l0.weight_read == allocs["target-w-mhc-broadcast-L0"]


# ---------------------------------------------------------------------------
# 真实 V4.1 整模型独立复算（h=5120, m=4, R=20, 40 层, T=1）
# ---------------------------------------------------------------------------

def test_v41_hc_totals_independently_recomputed_r05():
    """整模型 hc 总量逐层独立推导求和；行数 40×6 + expand 1 + exit 2 = 243。"""
    m = _model()
    rows = build_model_work(m, "decode", 1, 1024, 1, Q16, tp=1)
    hc_rows = [r for r in rows if r.op.startswith("hc_")]
    h, mm, R, L, T = 5120, 4, 20, 40, 1
    mix, hc_dim = (2 + mm) * mm, mm * h
    total = 0
    for li in range(L):
        ic_a = h if li == 0 else hc_dim           # 首层 attn 读广播嵌入
        total += T * (2 * ic_a * mix + 2 * ic_a)   # attn mix（GEMM+sqrsum）
        total += T * (2 * hc_dim * mix + 2 * hc_dim)   # ffn mix
        total += T * _sinkhorn_ops(mm, R, mix)     # attn sinkhorn
        total += T * _sinkhorn_ops(mm, R, mix)     # ffn sinkhorn
        total += T * _post_pre_ops(mm, h)          # attn post_pre
        total += T * _post_pre_ops(mm, h)          # ffn post_pre
    total += T * hc_dim                            # 首层 expand（写计数）
    total += T * h * (2 * mm * mm + mm)            # exit_post
    total += T * 2 * mm * h                        # exit_collapse
    assert sum(r.flops for r in hc_rows) == total == 101_582_480
    assert len(hc_rows) == 6 * L + 1 + 2 == 243
    # 全部 flops 为精确 int（无浮点/NaN 期望）
    assert all(isinstance(r.flops, int) for r in hc_rows)
    # V4.1 锚点：sinkhorn 1427 / post_pre 250882 / token（源码计数，见模块
    # docstring 与主控值 1426/250880 的差异记录）
    sink = next(r for r in hc_rows if r.op == "hc_attn_sinkhorn" and r.layer == 3)
    assert sink.flops == 1427
    pp = next(r for r in hc_rows if r.op == "hc_attn_post_pre" and r.layer == 3)
    assert pp.flops == 250_882
