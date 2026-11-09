"""逻辑工作量账本（Count 段 v0.2）：ModelIR × 负载 × 精度 → 每算子行。

口径（原理 §1/§2）：
  * FLOPs/五类字节中确定性量为精确整数；
  * MoE 期望并集权重读单列 weight_read_expected（float，expectation）；
  * GDN 线性层无 per-token KV：递推状态读写单列 state_read/state_write，
    每请求常数容量单列 state_capacity；
  * MLA（V3）按压缩布局计 KV（kv_lora+rope），展开式/吸收式 recipe 差异
    在 lowering 登记，本层给默认展开投影集；
  * V4.1 CSA2 沿用已对账的闭式（v0.1 移植 attention_structure，100% 对齐）；
    mHC 边界（hc_*，2026-11-10 按 vLLM pin 源码建模，替代该族通用 norm/add 行）
    与 Engram 注入缺口由 ModelIR.gaps 携带（Engram forward 未 pin，不猜）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .. import formula_registry as FR
from ..model_ir.schema import (FULL, LINEAR, MLA, MLA_CSA2, SLIDING, LayerIR,
                               ModelIR)
from .moe import moe_rows


@dataclass
class OpRow:
    stage: str
    layer: int                      # -1 = 层外（lm_head/embed/通信）
    op: str
    flops: int = 0                  # 精确
    weight_read: int = 0            # 精确字节
    weight_read_expected: Optional[float] = None  # 期望字节（MoE 并集）
    act_read: int = 0
    act_write: int = 0
    kv_read: int = 0
    kv_write: int = 0
    state_read: int = 0             # GDN 递推/卷积状态读写（字节）
    state_write: int = 0
    weight_capacity: int = 0        # 驻留（精确）
    state_capacity: int = 0         # 每请求常数状态（精确）
    formula: str = ""
    statistic: str = "exact"
    shape: str = ""
    exec_count: int = 1
    role: str = "target"

    def to_json(self) -> dict:
        return dict(self.__dict__)


def _sum_floor(n: int, r: int) -> int:
    q, m = divmod(n, r)
    return r * q * (q - 1) // 2 + q * (m + 1)


def causal_pairs(n: int, ratio: int, cap: int) -> int:
    """Σ_{t=1..n} min(floor(t/ratio), cap)（v0.1 闭式，已对账）。"""
    end = min(n, ratio * cap)
    return _sum_floor(end, ratio) + max(0, n - end) * cap


def _norm_row(toks: int, width: int, b_act: float):
    return (toks * (4 * width + 2), int(2 * width * b_act) if toks else 0,
            int(toks * width * b_act), int(toks * width * b_act))


def _rope_row(toks: int, heads: int, width: int, rotary: int, b_act: float):
    return (3 * toks * heads * rotary, 0,
            int(toks * (heads * width + rotary) * b_act),
            int(toks * heads * width * b_act))


# ---------------------------------------------------------------------------
# GQA / SLIDING 全注意力层
# ---------------------------------------------------------------------------


def gqa_layer_rows(stage: str, li: int, layer: LayerIR, B: int, S: int, Q: int,
                   h: int, quant: dict, tp: int = 1) -> List[OpRow]:
    attn = layer.attn
    b_w, b_a, b_kv = quant["dense_w"] / 8, quant["act"] / 8, quant["kv"] / 8
    heads_c = attn.heads // tp
    kv_heads_c = max(1, attn.kv_heads // tp) if tp <= attn.kv_heads else attn.kv_heads
    T = B * Q
    win = attn.sliding_window
    rows: List[OpRow] = []
    gemm = lambda op, ic, oc: OpRow(
        stage, li, op, flops=2 * T * ic * oc, weight_read=int(ic * oc * b_w),
        act_read=int(T * ic * b_a), act_write=int(T * oc * b_a),
        weight_capacity=int(ic * oc * b_w), formula=FR.F_GEMM,
        shape=f"[{T},{ic}]@[{ic},{oc}]")
    # q_proj（output_gated: 输出 2×，gate 与 query 同投影）
    rows.append(gemm("q_proj", h, attn.q_out // tp))
    rows.append(gemm("k_proj", h, attn.kv_out))
    rows.append(gemm("v_proj", h, attn.kv_out))
    # 因果（滑窗）可见对数
    if layer.kind == SLIDING and win:
        end, start = S + Q, S
        pairs = B * (causal_pairs(end, 1, win) - causal_pairs(start, 1, win))
        read_rows = min(end, win) - min(start, win)
        f_id = FR.F_SWA_PAIRS
    else:
        pairs = B * (S * Q + Q * (Q + 1) // 2)
        read_rows = S
        f_id = FR.F_CAUSAL_PAIRS
    kv_elems = 2 * attn.kv_heads * attn.head_dim
    rows.append(OpRow(
        stage, li, "attention_core",
        flops=pairs * heads_c * (4 * attn.head_dim + 5),
        act_read=int(T * heads_c * attn.head_dim * b_a),
        act_write=int(T * heads_c * attn.head_dim * b_a),
        kv_read=int(B * read_rows * kv_elems * b_kv),
        kv_write=int(T * kv_elems * b_kv),
        formula=FR.F_ATTN_CORE, shape=f"pairs={pairs} heads(card)={heads_c} {f_id}"))
    rows.append(gemm("o_proj", attn.heads * attn.head_dim // tp, h))
    # q/k head-norm + RoPE
    if attn.qk_head_norm:
        for name, hd in (("q_norm", attn.heads), ("k_norm", attn.kv_heads)):
            ops, w, ar, aw = _norm_row(T, attn.head_dim, b_a)
            rows.append(OpRow(stage, li, name, flops=ops * hd, weight_read=w,
                              act_read=ar, act_write=aw, formula=FR.F_RMSNORM,
                              shape=f"{hd} 头 × d={attn.head_dim}"))
    ops, _w, ar, aw = _rope_row(T, heads_c, attn.head_dim, attn.rotary_dim, b_a)
    rows.append(OpRow(stage, li, "rope", flops=ops, act_read=ar, act_write=aw,
                      formula=FR.F_ROPE, shape=f"rotary={attn.rotary_dim}"))
    return rows


# ---------------------------------------------------------------------------
# GDN 线性注意力层
# ---------------------------------------------------------------------------


def gdn_layer_rows(stage: str, li: int, layer: LayerIR, B: int, Q: int, h: int,
                   quant: dict, prefill: bool, tp: int = 1) -> List[OpRow]:
    g = layer.linear
    b_w = quant["dense_w"] / 8
    b_a = quant["act"] / 8
    b_state = quant.get("state", 32) / 8     # mamba_ssm_dtype 默认 fp32
    T = B * Q
    rows: List[OpRow] = []
    gemm = lambda op, ic, oc: OpRow(
        stage, li, op, flops=2 * T * ic * oc, weight_read=int(ic * oc * b_w),
        act_read=int(T * ic * b_a), act_write=int(T * oc * b_a),
        weight_capacity=int(ic * oc * b_w), formula=FR.F_GEMM,
        shape=f"[{T},{ic}]@[{ic},{oc}]")
    rows.append(gemm("in_proj_qkv", h, g.qkv_out))
    rows.append(gemm("in_proj_z", h, g.value_dim))
    rows.append(gemm("in_proj_ba", h, 2 * g.num_v_heads))
    # depthwise 因果卷积：每 token conv_dim×kernel 乘加（近似逐元素）
    rows.append(OpRow(
        stage, li, "causal_conv1d", flops=2 * T * g.qkv_out * g.conv_kernel,
        act_read=int(T * g.qkv_out * g.conv_kernel * b_a),
        act_write=int(T * g.qkv_out * b_a),
        formula=FR.F_GDN_STEP, statistic="exact",
        shape=f"depthwise k={g.conv_kernel} over {g.qkv_out}"))
    # 递推：每 token 4·state_elems OP；state 读+写（prefill 全序列逐步同样递推）
    rows.append(OpRow(
        stage, li, "gdn_recurrence", flops=4 * T * g.state_elems,
        state_read=int(T * g.state_elems * b_state),
        state_write=int(T * g.state_elems * b_state),
        state_capacity=int(g.state_elems * b_state + g.conv_state_elems * b_a),
        formula=FR.F_GDN_STEP, statistic="approximated",
        shape=f"N_v={g.num_v_heads} d_k={g.head_k_dim} d_v={g.head_v_dim} "
              f"state={g.state_elems} elem（每请求常数）"))
    # RMSNormGated(v) + out_proj
    ops, w, ar, aw = _norm_row(T, g.head_v_dim, b_a)
    rows.append(OpRow(stage, li, "v_norm_gated", flops=ops * g.num_v_heads,
                      weight_read=w, act_read=ar * g.num_v_heads,
                      act_write=aw * g.num_v_heads, formula=FR.F_RMSNORM,
                      shape=f"{g.num_v_heads} 头 × d_v={g.head_v_dim}"))
    rows.append(gemm("out_proj", g.value_dim, h))
    return rows


# ---------------------------------------------------------------------------
# MLA（V3 压缩布局，展开式投影集）
# ---------------------------------------------------------------------------


def mla_layer_rows(stage: str, li: int, layer: LayerIR, B: int, S: int, Q: int,
                   h: int, quant: dict, prefill: bool, tp: int = 1) -> List[OpRow]:
    m = layer.mla
    b_w, b_a, b_kv = quant["dense_w"] / 8, quant["act"] / 8, quant["kv"] / 8
    heads_c = m.heads // tp
    T = B * Q
    nope, rope, vdim = m.qk_nope_dim, m.qk_rope_dim, m.v_head_dim
    rows: List[OpRow] = []
    gemm = lambda op, ic, oc: OpRow(
        stage, li, op, flops=2 * T * ic * oc, weight_read=int(ic * oc * b_w),
        act_read=int(T * ic * b_a), act_write=int(T * oc * b_a),
        weight_capacity=int(ic * oc * b_w), formula=FR.F_GEMM,
        shape=f"[{T},{ic}]@[{ic},{oc}]")
    rows.append(gemm("q_a_proj", h, m.q_lora_rank))
    ops, w, ar, aw = _norm_row(T, m.q_lora_rank, b_a)
    rows.append(OpRow(stage, li, "q_a_norm", flops=ops, weight_read=w,
                      act_read=ar, act_write=aw, formula=FR.F_RMSNORM,
                      shape=f"width={m.q_lora_rank}"))
    rows.append(gemm("q_b_proj", m.q_lora_rank, heads_c * (nope + rope)))
    rows.append(gemm("kv_a_proj", h, m.kv_lora_rank + rope))
    kv_elems = m.kv_lora_rank + rope
    pairs = B * (S * Q + Q * (Q + 1) // 2)
    rows.append(OpRow(
        stage, li, "mla_absorbed_core",
        flops=pairs * heads_c * (2 * (m.kv_lora_rank + rope) + 2 * vdim + 5),
        act_read=int(T * heads_c * (nope + rope) * b_a),
        act_write=int(T * heads_c * vdim * b_a),
        kv_read=int(B * S * kv_elems * b_kv),   # 压缩 KV：每 token kv_lora+rope
        kv_write=int(T * kv_elems * b_kv),
        formula=FR.F_MLA_KV,
        shape=f"吸收式：q 与 W_UK/W_UV 在核内；压缩 KV {kv_elems} elem/token"))
    # 展开式 kv_b（吸收式将其折入核内 GEMM；此处给展开投影行供 recipe 对照）
    rows.append(gemm("kv_b_proj", m.kv_lora_rank, heads_c * (nope + vdim)))
    ops, _w, ar, aw = _rope_row(T, heads_c, nope + rope, rope, b_a)
    rows.append(OpRow(stage, li, "rope", flops=ops, act_read=ar, act_write=aw,
                      formula=FR.F_ROPE, shape=f"rope={rope} of {nope + rope}"))
    rows.append(gemm("o_proj", heads_c * vdim, h))
    return rows


# ---------------------------------------------------------------------------
# V4.1 CSA2 层（v0.1 对账闭式移植；Engram/hc 缺口由 ModelIR.gaps 携带）
# ---------------------------------------------------------------------------


def v41_layer_rows(stage: str, li: int, layer: LayerIR, B: int, S: int, Q: int,
                   h: int, kvs: dict, quant: dict, prefill: bool, tp: int = 1
                   ) -> List[OpRow]:
    m = layer.mla
    a = layer.csa2
    b_w, b_a, b_kv = quant["dense_w"] / 8, quant["act"] / 8, quant["kv"] / 8
    heads_full = kvs["heads"]
    heads = heads_full // tp
    qrank, orank, groups = kvs["q_lora_rank"], kvs["o_lora_rank"], kvs["o_groups"]
    d, rd = kvs["head_dim"], kvs["qk_rope_dim"]
    di, ih, topk = kvs["index_head_dim"], kvs["index_heads"], kvs["index_topk"]
    swa, ctb = kvs["sliding_window"], kvs["candidate_topk_blocks"]
    ng = groups // tp
    r = a.ratio
    T = B * Q
    start, length = (0, S + Q) if prefill else (S, Q)
    end = start + length
    rows: List[OpRow] = []
    gemm = lambda op, ic, oc, wmult=1.0: OpRow(
        stage, li, op, flops=2 * T * ic * oc, weight_read=int(ic * oc * wmult),
        act_read=int(T * ic * b_a), act_write=int(T * oc * b_a),
        weight_capacity=int(ic * oc * wmult), formula=FR.F_GEMM,
        shape=f"[{T},{ic}]@[{ic},{oc}]")
    rows.append(gemm("q_a_proj", h, qrank))
    rows.append(gemm("kv_proj", h, d))
    for name, width in (("q_norm", qrank), ("k_norm", d)):
        ops, w, ar, aw = _norm_row(T, width, b_a)
        rows.append(OpRow(stage, li, name, flops=ops, weight_read=w,
                          act_read=ar, act_write=aw, formula=FR.F_RMSNORM,
                          shape=f"width={width}"))
    rows.append(gemm("q_b_proj", qrank, heads * d))
    for name, hd in (("q_rope", heads), ("k_rope", 1)):
        ops, _w, ar, aw = _rope_row(T, hd, d, rd, b_a)
        rows.append(OpRow(stage, li, name, flops=ops, act_read=ar, act_write=aw,
                          formula=FR.F_ROPE, shape=f"rotary={rd} heads={hd}"))
    rows.append(OpRow(stage, li, "swa_cache_write", act_read=int(T * d * b_a),
                      kv_write=int(T * d * b_kv), formula=FR.F_KV_RW,
                      shape=f"每层本地 SWA 写 {T}×{d}"))
    new_rows = B * (end // r - start // r) if r else 0
    if a.produces_kv:
        mlt = 2 if r == 2 else 1
        rows.append(gemm("compress_proj", h, d * mlt))
        # compress_norm：pooling/state 估计（v0.1 同口径，标注近似）
        ops, w, ar, aw = _norm_row(new_rows, d, b_a)
        kv_state = B * d * 2 * 4 * length
        if r == 2:
            ops += new_rows * d * 12
            ar += int(T * 2 * d * b_a)
        rows.append(OpRow(stage, li, "compress_norm", flops=ops, weight_read=w,
                          act_read=int(ar), act_write=aw,
                          kv_read=kv_state if r == 2 else 0,
                          kv_write=kv_state if r == 2 else 0,
                          formula=FR.F_KV_RW, statistic="approximated",
                          shape="压缩 pooling 估计（非实测）；ratio2 含 fp32 state"))
        ops, _w, ar, aw = _rope_row(new_rows, 1, d, rd, b_a)
        rows.append(OpRow(stage, li, "compressed_rope", flops=ops, act_read=ar,
                          act_write=aw, formula=FR.F_ROPE, shape=f"new_rows={new_rows}"))
        rows.append(OpRow(stage, li, "compressed_cache_write",
                          act_read=int(new_rows * d * b_a),
                          kv_write=int(new_rows * d * b_kv), formula=FR.F_KV_RW,
                          shape=f"压缩 KV 写 {new_rows}×{d}"))
        rows.append(gemm("index_k_proj", d, di))
        ops, w, ar, aw = _norm_row(new_rows, di, b_a)
        rows.append(OpRow(stage, li, "index_k_norm", flops=ops, weight_read=w,
                          act_read=ar, act_write=aw, formula=FR.F_RMSNORM,
                          shape=f"width={di}"))
        ops, _w, ar, aw = _rope_row(new_rows, 1, di, rd, b_a)
        rows.append(OpRow(stage, li, "index_k_rope", flops=ops, act_read=ar,
                          act_write=aw, formula=FR.F_ROPE, shape=f"width={di}"))
        rows.append(OpRow(stage, li, "index_cache_write",
                          act_read=int(new_rows * di * b_a),
                          kv_write=int(new_rows * di * b_kv), formula=FR.F_KV_RW,
                          shape=f"索引 K 写 {new_rows}×{di}"))
    if a.produces_index:
        short = end // r <= topk
        rows.append(gemm("index_weights_proj", h, ih))
        if not short:
            rows.append(gemm("index_q_proj", qrank, ih * di))
            ops, _w, ar, aw = _rope_row(T, ih, di, rd, b_a)
            rows.append(OpRow(stage, li, "index_q_rope", flops=ops, act_read=ar,
                              act_write=aw, formula=FR.F_ROPE,
                              shape=f"{ih}×{di}"))
        else:
            rows.append(OpRow(stage, li, "index_q_proj",
                              shape="eager 短上下文旁路"))
        pairs = 0 if short else B * (_sum_floor(end, r) - _sum_floor(start, r))
        read_rows = 0 if short else (end // r if not prefill else pairs // max(B, 1))
        rows.append(OpRow(
            stage, li, "index_select",
            flops=2 * pairs * ih * di + T * topk,
            kv_read=int(B * read_rows * di * b_kv),
            act_write=int(T * topk * 4),
            formula=FR.F_GEMM, statistic="exact",
            shape=f"全候选打分 pairs={pairs}；top-{topk}（E19：索引读随 ctx 增长）"))
        if a.produces_candidates:
            rows.append(OpRow(stage, li, "candidate_publish",
                              act_write=0 if short else int(T * ctb * 4),
                              formula=FR.F_KV_RW,
                              shape=f"候选 block IDs ≤{ctb}"))
    local_pairs = causal_pairs(end, 1, swa) - causal_pairs(start, 1, swa)
    compressed_pairs = (causal_pairs(end, r, topk) - causal_pairs(start, r, topk)) if r else 0
    read_swa = min(end, swa) if not prefill else local_pairs
    read_comp = (min(end // r, topk) if (r and not prefill) else compressed_pairs)
    rows.append(OpRow(
        stage, li, "sparse_attention",
        flops=B * (local_pairs + compressed_pairs) * heads * (4 * d + 5),
        act_read=int(T * heads * d * b_a), act_write=int(T * heads * d * b_a),
        kv_read=int(B * (read_swa + read_comp) * d * b_kv),
        formula=FR.F_ATTN_CORE,
        shape=f"SWA {read_swa} + 压缩 {read_comp} 行（每消费者各读一次）"))
    ops, _w, ar, aw = _rope_row(T, heads, d, rd, b_a)
    rows.append(OpRow(stage, li, "o_rope", flops=ops, act_read=ar, act_write=aw,
                      formula=FR.F_ROPE, shape=f"rotary={rd}"))
    rows.append(gemm("o_a_proj", heads_full * d // groups, orank))
    rows[-1].shape = f"分组 {ng}×[{heads_full * d // groups},{orank}]"
    rows.append(gemm("o_b_proj", ng * orank, h))
    return rows


# ---------------------------------------------------------------------------
# 层公共件：FFN/MoE/norm/残差 + 整模型装配
# ---------------------------------------------------------------------------


def ffn_or_moe_rows(stage: str, li: int, layer: LayerIR, T: int, h: int,
                    quant: dict) -> List[OpRow]:
    b_w, b_a = quant["dense_w"] / 8, quant["act"] / 8
    rows: List[OpRow] = []
    if layer.moe is not None and layer.moe.experts:
        for r in moe_rows(layer.moe, T, h, quant):
            rows.append(OpRow(stage, li, r["layer_op"], flops=r["flops"],
                              weight_read=r["weight_read"],
                              weight_read_expected=r.get("weight_read_expected"),
                              act_read=r["act_read"], act_write=r["act_write"],
                              weight_capacity=r["weight_capacity"],
                              formula=r["formula"], statistic=r["statistic"],
                              shape=r["shape"]))
    elif layer.dense_inter:
        inter = layer.dense_inter
        for name, ic, oc in (("gate_up_proj", h, 2 * inter), ("down_proj", inter, h)):
            rows.append(OpRow(stage, li, name, flops=2 * T * ic * oc,
                              weight_read=int(ic * oc * b_w),
                              act_read=int(T * ic * b_a), act_write=int(T * oc * b_a),
                              weight_capacity=int(ic * oc * b_w),
                              formula=FR.F_GEMM, shape=f"[{T},{ic}]@[{ic},{oc}]"))
    return rows


def norm_add_rows(stage: str, li: int, T: int, h: int, quant: dict) -> List[OpRow]:
    b_a = quant["act"] / 8
    rows: List[OpRow] = []
    for name in ("attn_norm", "mlp_norm"):
        ops, w, ar, aw = _norm_row(T, h, b_a)
        rows.append(OpRow(stage, li, name, flops=ops, weight_read=w,
                          act_read=ar, act_write=aw, formula=FR.F_RMSNORM,
                          shape=f"width={h}"))
    for name in ("attn_add", "mlp_add"):
        rows.append(OpRow(stage, li, name, flops=T * h,
                          act_read=int(2 * T * h * b_a), act_write=int(T * h * b_a),
                          formula=FR.F_RESIDUAL, shape="残差逐元素加法（E09：不消失）"))
    return rows


def hc_boundary_rows(stage: str, li: int, T: int, h: int, hc: dict,
                     quant: dict, first_layer: bool = False) -> List[OpRow]:
    """mHC（mega hyper-connections）边界算子（v41_csa2 专用）。

    证据（vllm models/deepseek_v41/nvidia/model.py pin @f358f057；
    source-checked 2026-11-10）：
      * 每层 2 次边界（attn 侧 hc_attn_fn / ffn 侧 hc_ffn_fn），常规路径为
        mhc_shifted_post_pre 融合核 = 前一子层 post（流更新）+ 本侧系数 GEMM
        + sinkhorn(hc_sinkhorn_iters) + collapse + RMSNorm（norm_weight 即
        attn_norm/ffn_norm）→ v41 层**不再**另计通用 norm/add 行（防重复计费）；
      * hc_dim = hc_mult·h 为全宽流：mHC 边界不随 TP 切分（attention 内部自行
        分片；hc 权重每卡复制，见 lowering/memory），本函数不除 tp；
      * 首层输入为广播嵌入：attn 侧系数 GEMM IC=h（非 hc_dim），另有
        hc_expand 广播行（model.py `residual = hc_expand(x, hc_mult)`）；
      * Engram 层（extras.hc.engram_layer_ids）post/pre 分立执行 + Engram 注入
        ——注入算子未建模（common/engram.py 未 pin），由 ModelIR.gaps 携带，
        本函数不计猜测常数；边界算子构成与常规层相同（post+pre 总量不变）。
    """
    m = hc["mult"]
    iters = hc["sinkhorn_iters"]
    mix = (2 + m) * m                     # mix_hc（hc_attn_fn/hc_ffn_fn 行数）
    hc_dim = m * h
    b_a = quant["act"] / 8
    rows: List[OpRow] = []
    if first_layer:
        rows.append(OpRow(
            stage, li, "hc_expand", flops=T * hc_dim,
            act_read=int(T * h * b_a), act_write=int(T * hc_dim * b_a),
            formula=FR.F_MHC_EXPAND, shape=f"嵌入广播 {m} 流 × h"))
    for side, fn in (("attn", "hc_attn_fn"), ("ffn", "hc_ffn_fn")):
        ic = h if (first_layer and side == "attn") else hc_dim
        rows.append(OpRow(
            stage, li, f"hc_{side}_mix", flops=2 * T * ic * mix,
            weight_read=int(mix * hc_dim * 4),          # FP32（源码 dtype）
            weight_capacity=int(mix * hc_dim * 4),
            act_read=int(T * ic * b_a), act_write=int(T * mix * 4),
            formula=FR.F_MHC_MIX,
            shape=f"[{T},{ic}]@[{ic},{mix}]·{fn}（fp32，每卡复制）"))
        rows.append(OpRow(
            stage, li, f"hc_{side}_sinkhorn", flops=T * mix * iters * 5,
            act_read=int(T * mix * 4), act_write=int(T * mix * 4),
            formula=FR.F_MHC_SINKHORN, statistic="approximated",
            shape=f"{T}×{mix} 系数 × {iters} 迭代（FP32；每迭代 OP 数近似）"))
        norm_ops = T * (4 * h + 2)
        rows.append(OpRow(
            stage, li, f"hc_{side}_post_pre",
            flops=4 * T * hc_dim + norm_ops,
            weight_read=int(h * b_a),
            act_read=int((T * hc_dim + T * h) * b_a),
            act_write=int((T * hc_dim + T * h) * b_a),
            formula=FR.F_MHC_POST_PRE,
            shape=f"post 流更新+collapse（各 2·T·{hc_dim}）+ RMSNorm(h={h})；"
                  f"含残差（流上）"))
    return rows


def head_rows(stage: str, model: ModelIR, T: int, quant: dict, tp: int) -> List[OpRow]:
    h, v = model.hidden, model.vocab
    b_w, b_a = quant["dense_w"] / 8, quant["act"] / 8
    rows = [OpRow(stage, -1, "lm_head", flops=2 * T * h * (v // tp),
                  weight_read=int(h * (v // tp) * b_w),
                  act_read=int(T * h * b_a), act_write=int(T * (v // tp) * b_a),
                  weight_capacity=int(h * (v // tp) * b_w),
                  formula=FR.F_LM_HEAD, shape=f"h×V/tp（列切分）")]
    if not model.tie_embeddings:
        rows.append(OpRow(stage, -1, "embed_lookup",
                          act_read=int(T * h * b_a), act_write=int(T * h * b_a),
                          weight_capacity=int(h * v * b_w),
                          formula=FR.F_EMBED, shape="查表 1 行/token；容量 h×V"))
    return rows


def build_model_work(model: ModelIR, stage: str, B: int, ctx: int, queries: int,
                     quant: dict, tp: int = 1, prefill: bool = False,
                     role: str = "target") -> List[OpRow]:
    """整模型一个阶段的逻辑账本。

    decode/verify：ctx=查询前缓存长度 S；queries=新查询数 Q。
    prefill：S=0、queries=ctx（Q=整段）。
    """
    h = model.hidden
    S, Q = (0, ctx) if prefill else (ctx, queries)
    rows: List[OpRow] = []
    kvs = (model.extras or {}).get("kv_spec", {})
    hc_spec = (model.extras or {}).get("hc")
    first_layer_index = model.layers[0].index if model.layers else None
    for layer in model.layers:
        li = layer.index
        if layer.kind in (FULL, SLIDING) and layer.attn:
            rows += gqa_layer_rows(stage, li, layer, B, S, Q, h, quant, tp)
        elif layer.kind == LINEAR and layer.linear:
            rows += gdn_layer_rows(stage, li, layer, B, Q, h, quant, prefill, tp)
        elif layer.kind == MLA and layer.mla:
            rows += mla_layer_rows(stage, li, layer, B, S, Q, h, quant, prefill, tp)
        elif layer.kind == MLA_CSA2 and layer.mla and layer.csa2:
            rows += v41_layer_rows(stage, li, layer, B, S, Q, h, kvs, quant,
                                   prefill, tp)
        T = B * Q
        rows += ffn_or_moe_rows(stage, li, layer, T, h, quant)
        if layer.kind == MLA_CSA2 and hc_spec:
            # mHC 边界核融合 norm+残差（F_MHC_POST_PRE）：不再另计通用 norm/add 行
            rows += hc_boundary_rows(stage, li, T, h, hc_spec, quant,
                                     first_layer=(li == first_layer_index))
        else:
            rows += norm_add_rows(stage, li, T, h, quant)
    rows += head_rows(stage, model, B * Q, quant, tp)
    for r in rows:
        r.role = role
    return rows


def totals(rows: List[OpRow]) -> Dict[str, float]:
    out = {"flops": 0, "weight_read": 0, "weight_read_expected": 0.0,
           "act_read": 0, "act_write": 0, "kv_read": 0, "kv_write": 0,
           "state_read": 0, "state_write": 0, "rows": len(rows)}
    for r in rows:
        out["flops"] += r.flops
        out["weight_read"] += r.weight_read
        if r.weight_read_expected is not None:
            out["weight_read_expected"] += r.weight_read_expected
        out["act_read"] += r.act_read
        out["act_write"] += r.act_write
        out["kv_read"] += r.kv_read
        out["kv_write"] += r.kv_write
        out["state_read"] += r.state_read
        out["state_write"] += r.state_write
    return out
