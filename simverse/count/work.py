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
    与 Engram 注入（engram_*，2026-11-11 R04 按 common/engram.py pin
    @f358f057 blob ba4ceea6 建模：批级哈希/查表/wkv/门控四行）同源 extras。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .. import formula_registry as FR
from ..model_ir.families import engram_bucket_primes
from ..model_ir.schema import (FULL, LINEAR, MLA, MLA_CSA2, SLIDING, LayerIR,
                               ModelIR)
from .flashmla_sched import flashmla_decode_num_splits
from .moe import moe_rows, require_divisible_ep, v41_moe_route_rows


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
    # KV 头 per-rank（vLLM QKVParallelLinear：kvh≥tp → kvh/tp；kvh<tp → 复制）
    kv_heads_c = attn.kv_heads // tp if attn.kv_heads >= tp else attn.kv_heads
    kv_replicated = attn.kv_heads < tp
    T = B * Q
    win = attn.sliding_window
    rows: List[OpRow] = []
    gemm = lambda op, ic, oc, note="": OpRow(
        stage, li, op, flops=2 * T * ic * oc, weight_read=int(ic * oc * b_w),
        act_read=int(T * ic * b_a), act_write=int(T * oc * b_a),
        weight_capacity=int(ic * oc * b_w), formula=FR.F_GEMM,
        shape=f"[{T},{ic}]@[{ic},{oc}]{note}")
    # q_proj（output_gated: 输出 2×，gate 与 query 同投影）；列切 /tp
    rows.append(gemm("q_proj", h, attn.q_out // tp))
    # k/v_proj：kvh≥tp 列切 kvh/tp；kvh<tp 复制全量（每 rank 同字节）
    kv_note = "（KV 头复制）" if kv_replicated else ""
    rows.append(gemm("k_proj", h, kv_heads_c * attn.head_dim, kv_note))
    rows.append(gemm("v_proj", h, kv_heads_c * attn.head_dim, kv_note))
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
    kv_elems = 2 * kv_heads_c * attn.head_dim   # per-rank KV 元素
    rows.append(OpRow(
        stage, li, "attention_core",
        flops=pairs * heads_c * (4 * attn.head_dim + 5),
        act_read=int(T * heads_c * attn.head_dim * b_a),
        act_write=int(T * heads_c * attn.head_dim * b_a),
        kv_read=int(B * read_rows * kv_elems * b_kv),
        kv_write=int(T * kv_elems * b_kv),
        formula=FR.F_ATTN_CORE,
        shape=f"pairs={pairs} heads(card)={heads_c} kv(card)={kv_heads_c}"
              f"{'（复制）' if kv_replicated else ''} {f_id}"))
    # 输出门控逐元素行（R06 CPU fixture 对照补齐）：官方 L802
    # attn_output = attn_output * torch.sigmoid(gate)——q_proj 输出 2× 的
    # 门控分支消费由 q_proj GEMM 覆盖，但 sigmoid+乘的逐元素算量未计。
    if attn.output_gated:
        hd = heads_c * attn.head_dim
        rows.append(OpRow(stage, li, "attn_output_gate", flops=2 * T * hd,
                          act_read=int(2 * T * hd * b_a),
                          act_write=int(T * hd * b_a),
                          formula=FR.F_ATTN_OUT_GATE,
                          shape=f"sigmoid(gate)·attn_out 逐元素 {hd} elem"
                                f"（q_proj 2× 门控分支消费；R06 fixture 修正）"))
    rows.append(gemm("o_proj", attn.heads * attn.head_dim // tp, h))
    # q/k head-norm + RoPE
    if attn.qk_head_norm:
        for name, hd in (("q_norm", heads_c), ("k_norm", kv_heads_c)):
            ops, w, ar, aw = _norm_row(T, attn.head_dim, b_a)
            rows.append(OpRow(stage, li, name, flops=ops * hd, weight_read=w,
                              act_read=ar * hd, act_write=aw * hd,
                              formula=FR.F_RMSNORM,
                              shape=f"{hd} 头(卡上) × d={attn.head_dim}"
                                  f"（act 按 ×hd 计，2026-11-10 修正漏乘）"))
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
    # per-rank 几何（vLLM in_proj_qkvz/ba 列切 + out_proj 行切 + 状态切分）
    nv = g.num_v_heads // tp
    key_dim_r = g.head_k_dim * (g.num_k_heads // tp)
    value_dim_r = g.head_v_dim * nv
    qkv_out_r = g.qkv_out // tp
    state_elems_r = nv * g.head_k_dim * g.head_v_dim
    rows: List[OpRow] = []
    gemm = lambda op, ic, oc: OpRow(
        stage, li, op, flops=2 * T * ic * oc, weight_read=int(ic * oc * b_w),
        act_read=int(T * ic * b_a), act_write=int(T * oc * b_a),
        weight_capacity=int(ic * oc * b_w), formula=FR.F_GEMM,
        shape=f"[{T},{ic}]@[{ic},{oc}]")
    # in_proj_qkv + in_proj_z（vLLM 融合为 in_proj_qkvz，列切 /tp）
    rows.append(gemm("in_proj_qkv", h, (2 * key_dim_r + value_dim_r)))
    rows.append(gemm("in_proj_z", h, value_dim_r))
    rows.append(gemm("in_proj_ba", h, 2 * nv))
    # depthwise 因果卷积：每 token conv_dim(卡上)×kernel 乘加（近似逐元素）
    rows.append(OpRow(
        stage, li, "causal_conv1d", flops=2 * T * qkv_out_r * g.conv_kernel,
        act_read=int(T * qkv_out_r * g.conv_kernel * b_a),
        act_write=int(T * qkv_out_r * b_a),
        formula=FR.F_GDN_STEP, statistic="exact",
        shape=f"depthwise k={g.conv_kernel} over {qkv_out_r}(卡上)"))
    # 递推：每 token 4·state_elems(卡上) OP；state 读+写（prefill 同样逐步递推）
    rows.append(OpRow(
        stage, li, "gdn_recurrence", flops=4 * T * state_elems_r,
        state_read=int(T * state_elems_r * b_state),
        state_write=int(T * state_elems_r * b_state),
        state_capacity=int(state_elems_r * b_state
                           + qkv_out_r * (g.conv_kernel - 1) * b_a),
        formula=FR.F_GDN_STEP, statistic="approximated",
        shape=f"N_v={nv}(卡上) d_k={g.head_k_dim} d_v={g.head_v_dim} "
              f"state={state_elems_r} elem（每请求常数，per-rank）"))
    # RMSNormGated(v) + out_proj（行切：value_dim(卡上)→h）
    ops, w, ar, aw = _norm_row(T, g.head_v_dim, b_a)
    rows.append(OpRow(stage, li, "v_norm_gated", flops=ops * nv,
                      weight_read=w, act_read=ar * nv,
                      act_write=aw * nv, formula=FR.F_RMSNORM,
                      shape=f"{nv} 头(卡上) × d_v={g.head_v_dim}"))
    rows.append(gemm("out_proj", value_dim_r, h))
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
# V4.1 CSA2 层（R20，2026-11-12：索引打分/top-k/candidate block 与压缩器/
# 索引 K 按 csa2-pin.json 21 文件 pin（vllm @f358f057）source-checked 重写；
# R25，2026-11-16：注意力核本体按 csa2-core-pin.json 14 文件
# （vllm @f358f057 5 文件 + FlashMLA @847ef308 9 文件，cmake GIT_TAG 钉定）
# source-checked 重写：decode 头数按 padded_heads=64（fp8 kernel h_q∈{64,128}，
# Q 零填充）、SM90 恒 split-KV（o_accum/lse_accum/combine/调度器）、prefill
# 全量压缩前缀 gather + bf16 工作区口径、combine 仅 prefill 每 chunk 每层）
# ---------------------------------------------------------------------------


def v41_layer_rows(stage: str, li: int, layer: LayerIR, B: int, S: int, Q: int,
                   h: int, kvs: dict, quant: dict, prefill: bool, tp: int = 1,
                   v41_kv_record: str = "fp8_ds_mla_sm90",
                   quant_recipe: Optional[dict] = None,
                   device_sms: Optional[int] = None,
                   attn_type_first: bool = False,
                   attn_step_first: bool = False,
                   ) -> List[OpRow]:
    """V4.1 CSA2 层账本（per-rank）。

    证据（csa2-pin.json 21 文件，vllm @f358f057，blob sha1 对
    vllm-tree.json 逐项校验；source-checked 2026-11-12 R20）：
      * 层型：compress_ratios 0/1/2（sparse_mla.py L34-53）；kv source
        [2,8,14,20] 持有 compressor+压缩缓存+indexer K，index source
        [2,8,14,20,24,28,32,36] 持有打分器，非持有层复用下方最近源
        （attention.py L393-422/L502-553）；
      * 固定 bf16 权重（quant_config=None）：compressor fused_wkv_wgate、
        indexer weights_proj、indexer wk（attention.py L1484-1490/L1510-1516、
        compressor.py L321-332）——不随场景 dense_w 位宽缩放；
      * 打包记录：SWA/压缩 584|528 B/token、索引 K 132 B/行（显式
        v41_kv_record 配方；attention.py L583-589/L181-189）；
      * 短上下文旁路：batch max_seq_len//r ≤ topk 且非 graph 捕获 →
        跳过 wq_b/rope/打分/top-k/候选，只写全选索引（attention.py
        L1656-1680；fill_short_context_topk_indices 纯 store，0 浮点）；
      * candidate 两级：源层（20）发布 top-2048 个 8 压缩位块（块最大值
        NaN 传播、最新块钉 +inf）；消费层（24/28/32/36）打分后先 mask 到
        候选块再自有 top-512（sparse_attn_indexer.py L895-911/L1141-1153；
        candidate_blocks.py L168-215/L151-166）。
    """
    m = layer.mla
    a = layer.csa2
    b_w, b_a = quant["dense_w"] / 8, quant["act"] / 8
    # R22：V4.1 量化配方驱动字节（extras.quant；checkpoint 格式 + 注册设备
    # ≤SM90 Marlin W8A16 执行域）。量化线性 = MXFP8 1B + ue8m0 (1,32) scale；
    # 激活 BF16 2B。场景 quant 位宽对 V4.1 已被 validate 钉死为同值。
    if quant_recipe is not None:
        b_w = float(quant_recipe["dense_linear"]["value_itemsize"])
        b_a = float(quant_recipe["act_itemsize"])
    from ..scenario import V41_INDEX_K_BYTES, V41_KV_RECORDS
    rec = V41_KV_RECORDS[v41_kv_record]
    swa_rec, comp_rec = rec["swa_bytes"], rec["compressed_bytes"]
    heads_full = kvs["heads"]
    heads = heads_full // tp
    qrank, orank, groups = kvs["q_lora_rank"], kvs["o_lora_rank"], kvs["o_groups"]
    d, rd = kvs["head_dim"], kvs["qk_rope_dim"]
    di, ih, topk = kvs["index_head_dim"], kvs["index_heads"], kvs["index_topk"]
    swa, ctb = kvs["sliding_window"], kvs["candidate_topk_blocks"]
    cb = int(kvs.get("candidate_block_size", 0) or 0)
    cand_src = int(kvs.get("candidate_source_layer_id", -1))
    ng = groups // tp
    r = a.ratio
    T = B * Q
    start, length = (0, S + Q) if prefill else (S, Q)
    end = start + length
    rows: List[OpRow] = []

    def gemm(op, ic, oc, reps: int = 1, b: float = None):
        """reps>1：分组投影（V4.1 o_a 每 rank ng 组，每组独立权重/激活）。
        b：显式权重位宽（pinned bf16 模块覆盖场景 dense_w）。
        R22：b=None 且量化配方在场时，权重读/容量并入 ue8m0 scale 字节
        （oc×ic/32 每 rep，ic 须整除 32 否则按域拒绝；运行时展开后口径，
        quant_layout.mxfp8_linear_bytes）。"""
        bw = b_w if b is None else b
        scale_b = 0
        if b is None and quant_recipe is not None:
            from .quant_layout import mxfp8_linear_bytes
            _v, scale_b = mxfp8_linear_bytes(oc, ic)
            scale_b *= reps
        return OpRow(
            stage, li, op, flops=2 * T * ic * oc * reps,
            weight_read=int(ic * oc * reps * bw) + scale_b,
            act_read=int(T * ic * reps * b_a), act_write=int(T * oc * reps * b_a),
            weight_capacity=int(ic * oc * reps * bw) + scale_b,
            formula=FR.F_GEMM,
            shape=f"{reps}×[{T},{ic}]@[{ic},{oc}]" if reps > 1
            else f"[{T},{ic}]@[{ic},{oc}]")

    rows.append(gemm("q_a_proj", h, qrank))
    rows.append(gemm("kv_proj", h, d))
    # q_norm(q_lora_rank)/kv_norm(head_dim) 是普通 RMSNorm 参数，随模型
    # dtype 为 bf16（attention.py L440-450）；不随场景 act 位宽缩放。
    # weight_capacity 与 target_weight_allocations 的常驻项同口径。
    for name, width in (("q_norm", qrank), ("k_norm", d)):
        ops, _w, ar, aw = _norm_row(T, width, b_a)
        w = int(width * 2)
        rows.append(OpRow(stage, li, name, flops=ops, weight_read=w,
                          weight_capacity=w, act_read=ar, act_write=aw,
                          formula=FR.F_RMSNORM, shape=f"width={width}（bf16 权重）"))
    rows.append(gemm("q_b_proj", qrank, heads * d))
    for name, hd in (("q_rope", heads), ("k_rope", 1)):
        ops, _w, ar, aw = _rope_row(T, hd, d, rd, b_a)
        if name == "q_rope":
            # R25：fp8_ds_mla 布局下 fused insert 把 Q 零填充到 padded_heads
            # 再返回（attention.py L1216-1247 pad_to=padded_heads）；RoPE 只
            # 算真实头，但 q_padded 按填充宽度写（decode 核与 prefill 切片
            # 各自消费，见 sparse_attention 行）
            aw = int(T * (64 if heads_full <= 64 else 128) * d * 2)
            rows.append(OpRow(stage, li, name, flops=ops, act_read=ar,
                              act_write=aw, formula=FR.F_ROPE,
                              shape=f"rotary={rd} heads={hd}（写零填充 "
                                    "padded_heads 宽 q_padded，bf16）"))
        else:
            rows.append(OpRow(stage, li, name, flops=ops, act_read=ar,
                              act_write=aw, formula=FR.F_ROPE,
                              shape=f"rotary={rd} heads={hd}"))
    # SWA 写：fused 插入核读 bf16 kv 行、写打包记录（584/528 B/token；
    # R20 修正：旧式 d×b_kv 漏 scale/pad 尾部字节）
    rows.append(OpRow(stage, li, "swa_cache_write", act_read=int(T * d * 2),
                      kv_write=int(T * swa_rec), formula=FR.F_CSA2_RECORD,
                      shape=f"每层本地 SWA 写 {T}×{swa_rec}B"
                            f"（fp8_ds_mla 记录，{v41_kv_record}）"))
    new_rows = B * (end // r - start // r) if r else 0
    if a.produces_kv:
        mlt = 2 if r == 2 else 1
        # compressor fused_wkv_wgate：replicated bf16 权重、FP32 kv_score 输出
        #（compressor.py L321-332 quant_config=None + disable_tp；
        # aux_projections.py 文档"FP32 kv_score"；低批 ll_bf16 变体同口径）
        cg = gemm("compress_proj", h, d * mlt, b=2.0)
        # quant_config=None：输入 hidden 为 bf16、输出 kv_score 为 FP32
        #（aux_projections.py L40-74；均不随场景 act 位宽缩放）
        cg.act_read = int(T * h * 2)
        cg.act_write = int(T * d * mlt * 4)
        cg.shape = (f"[{T},{h}]@[{h},{d * mlt}]（replicated bf16 权重/输入，"
                    "FP32 kv_score 输出 4B/elem）")
        rows.append(cg)
        if r == 1:
            pool_ops = new_rows * (4 * d + 2)
            pool_ar = int(T * d * 4)
            pool_aw = int(new_rows * d * 2)
            st_rd = st_wr = 0
            pool_stat = "exact"
            pool_shape = ("ratio1 无门控无池化：逐 token RMSNorm"
                          "（fused_save_compress_norm ratio1 程序 L238-243）")
        else:
            # 每 closed group：softmax(2)=8 + 加权和 2d 乘 + (d−1) 加 +
            # RMSNorm 4d+2 → 7d+9（fused_compress_quant_cache.py L186-243）；
            # 环形 state：写 min(Q,cap) 行 ×4096B fp32/请求、读 ≤1 行/请求
            #（L168-215 请求程序；上界口径）
            pool_ops = new_rows * (7 * d + 9)
            pool_ar = int(T * 2 * d * 4)
            pool_aw = int(new_rows * d * 2)
            st_wr = int(B * min(Q, 8) * 2 * d * 4)
            st_rd = int(B * 2 * d * 4)
            pool_stat = "approximated"
            pool_shape = ("ratio2 softmax 门控池化 7d+9/group + RMSNorm；环形"
                          " state 写 min(Q,8)×4096B、读 ≤1 行/请求（上界）")
        rows.append(OpRow(stage, li, "compress_pool", flops=pool_ops,
                          weight_read=int(d * 2), weight_capacity=int(d * 2),
                          act_read=pool_ar, act_write=pool_aw,
                          state_read=st_rd, state_write=st_wr,
                          formula=FR.F_CSA2_COMPRESS_POOL,
                          statistic=pool_stat, shape=pool_shape))
        # rope_quant_insert：一条核做 RoPE+量化+分页写（R20 合并旧
        # compressed_rope + compressed_cache_write 两行；L295-374）
        q_ops = {"fp8_ds_mla_sm90": 3 * rd + (d // 64) * 65 + 7,
                 "fp8_ds_mla_sm100": 3 * rd + (d // 32) * 34 + 16,
                 "nvfp4_ds_mla_sm100": 3 * rd + (d // 16) * 18 + 32,
                 }[v41_kv_record]
        rows.append(OpRow(stage, li, "compressed_insert",
                          flops=new_rows * q_ops,
                          act_read=int(new_rows * d * 2),
                          kv_write=int(new_rows * comp_rec),
                          formula=FR.F_CSA2_COMPRESSED_INSERT,
                          statistic="exact",
                          shape=f"RoPE(3·{rd}) + 记录量化 + 分页写 "
                                f"{new_rows}×{comp_rec}B（{v41_kv_record}）"))
        # wk：latent(d)→di replicated bf16（attention.py L1508-1517
        # quant_config=None；latent 输入与 k_pre 输出均 bf16）
        kp = gemm("index_k_proj", d, di, b=2.0)
        kp.act_read = int(T * d * 2)
        kp.act_write = int(T * di * 2)
        rows.append(kp)
        # k_norm→RoPE→FP8 量化→分页写：一条核（indexer_k_store.py L126-266）
        rows.append(OpRow(stage, li, "index_k_store",
                          flops=new_rows * (5 * di + 3 * rd + 4),
                          weight_read=int(di * 2), weight_capacity=int(di * 2),
                          act_read=int(new_rows * di * 2),
                          kv_write=int(new_rows * V41_INDEX_K_BYTES),
                          formula=FR.F_CSA2_INDEX_K_STORE, statistic="exact",
                          shape=f"k_norm(4·{di}+2) + RoPE(3·{rd}) + FP8 量化"
                                f"(≈{di}+2) → {V41_INDEX_K_BYTES}B/行"))
    if a.produces_index:
        # 短上下文旁路：batch max_seq_len//r ≤ topk（attention.py L1656-1680；
        # 均匀批 = end//r）；weights_proj 仍执行，wq_b/rope/打分/top-k/候选
        # 全部跳过（graph 捕获中也不走，见 source_map uncovered）
        short = end // r <= topk
        # weights_proj：quant_config=None；hidden 输入与 weights 输出均 bf16，
        # 不随场景 act/dense_w 缩放（attention.py L1484-1490）
        wp = gemm("index_weights_proj", h, ih, b=2.0)
        wp.act_read = int(T * h * 2)
        wp.act_write = int(T * ih * 2)
        rows.append(wp)
        n_eff = end // r                                  # 每行有效压缩位数
        pairs = 0 if short else B * (_sum_floor(end, r) - _sum_floor(start, r))
        nblocks = (0 if short else -(-n_eff // cb)) if cb else 0
        logit_b = rec["logits_itemsize"]
        if short:
            rows.append(OpRow(stage, li, "index_q_proj",
                              shape="eager 短上下文旁路（attention.py L1656-1680："
                                    "wq_b/rope/打分/top-k/候选跳过）"))
            rows.append(OpRow(stage, li, "index_fill",
                              act_write=int(T * topk * 4),
                              formula=FR.F_CSA2_INDEX_FILL, statistic="exact",
                              shape="全选旁路纯 store 0..n−1 + (−1) 填充，"
                                    "0 浮点（fill_short_context_topk_indices "
                                    "L77-92；K 缓存仍由 produces_kv 行构建）"))
        else:
            # wq_b 走 quant_config（输入按场景 act 计），输出 q 为 bf16，
            # 与 index_q_rope 的 2B q 读取对齐（attention.py L1477-1482/
            # L1608-1637）
            qp = gemm("index_q_proj", qrank, ih * di)
            qp.act_write = int(T * ih * di * 2)
            rows.append(qp)
            ops = 3 * T * ih * rd + 2 * T * ih
            rows.append(OpRow(stage, li, "index_q_rope", flops=ops,
                              act_read=int(T * ih * di * 2 + T * ih * 2),
                              act_write=int(T * ih * di * 1 + T * ih * 4),
                              formula=FR.F_CSA2_INDEX_Q_ROPE, statistic="exact",
                              shape=f"{ih}×{di} RoPE(3·{rd}/头) + FP8 量化 + "
                                    "逐头权重缩放 2 op（fused_indexer_q_rope_"
                                    "quant；weights fp32 输出）"))
            # 打分：paged MQA logits（DeepGEMM），每 (行,压缩位) ih·di MAC +
            # ih 权重折算；decode 直读分页 K；prefill 先 gather 到稠密工作区
            #（gather 工作区按一次读/写计；logits 按 pairs 条有效对计，decode
            # Q=1 时 pairs=B·n_eff）
            rows.append(OpRow(
                stage, li, "index_score",
                flops=2 * pairs * ih * di + 2 * pairs * ih,
                kv_read=int(B * n_eff * V41_INDEX_K_BYTES),
                act_read=(int(B * n_eff * V41_INDEX_K_BYTES)
                          if prefill else 0),
                act_write=(int(B * n_eff * V41_INDEX_K_BYTES
                               + pairs * logit_b) if prefill
                           else int(pairs * logit_b)),
                formula=FR.F_CSA2_INDEX_SCORE, statistic="exact",
                shape=f"全候选打分 pairs={pairs}（E19：索引读随 ctx 增长）；"
                      f"logits 写 {pairs}×{logit_b}B（{v41_kv_record}）；"
                      "prefill gather 工作区为上界口径"))
            if a.produces_candidates:
                # 候选源：块得分（NaN 传播 max，最新块钉 +inf）→ top-ctb →
                # 候选缓冲（candidate_blocks.py L168-215 + L31-73）
                rows.append(OpRow(
                    stage, li, "candidate_select",
                    flops=T * (7 * nblocks + 2 * nblocks + ctb),
                    act_read=int(pairs * logit_b),
                    act_write=int(T * (nblocks * 4 + 4 + ctb * 4)),
                    formula=FR.F_CSA2_CANDIDATE_SELECT, statistic="exact",
                    shape=f"块最大值 7·nblocks + 选择 2·nblocks + 写 {ctb}；"
                          f"nblocks=⌈{n_eff}/{cb}⌉（-1 填充；最新块 +inf 钉扎）"))
            if 0 <= cand_src < li:
                # 候选消费层：打分后 mask 到候选块再自有 top-k
                #（sparse_attn_indexer.py L895-911/L1141-1153）
                rows.append(OpRow(
                    stage, li, "candidate_mask",
                    flops=T * (4 * n_eff + ctb + nblocks),
                    act_read=int(T * ctb * 4),
                    act_write=int(pairs * logit_b),
                    formula=FR.F_CSA2_CANDIDATE_MASK, statistic="exact",
                    shape=f"flags(⌈{n_eff}/{cb}⌉+1) + 逐列 block 索引/查掩码"
                          f"≈4·n + 候选读 {ctb}（有效前缀口径；CUDA-graph 整宽"
                          " -inf 尾写未建模见缺口）"))
            # top-k：cooperative(≤64 行)/persistent/per_row（indexer_topk.py
            # L264-282 auto 链）；选择 op 惯例 2·n 扫描 + k 写/行
            rows.append(OpRow(
                stage, li, "index_topk",
                flops=2 * pairs + T * topk,
                act_read=int(pairs * logit_b),
                act_write=int(T * topk * 4),
                formula=FR.F_CSA2_INDEX_TOPK, statistic="exact",
                shape=f"top-{topk} 选择惯例 2·n+k/行（n={n_eff}）；写共享 "
                      "topk_indices_buffer（R20 起与旧 index_select 的 "
                      "T·topk 口径区分扫描项）"))
    # ---- R25：注意力核本体（csa2-core-pin.json；vllm flashmla.py 消费链）----
    # 头数口径（flashmla.py L92-100 get_padded_num_q_heads：fp8 decode kernel
    # 仅支持 h_q∈{64,128}；fp8_ds_mla 布局下 fused insert 把 Q 零填充到
    # padded_heads=64，attention.py L1216-1247）：decode 按 padded 头计
    # FLOPs/激活；prefill 在 SM90 small-heads（8/16/32）用真实本地头切片
    # （flashmla.py L79-87 _get_prefill_num_heads + flash_mla_sparse_fwd
    # "SM90 supports 8, 16, 32, 64 or 128 heads"）。
    padded_heads = 64 if heads_full <= 64 else 128
    sm90 = rec["family"] == "sm90"
    kern_heads = (heads if (prefill and sm90 and heads in (8, 16, 32))
                  else padded_heads)
    local_pairs = causal_pairs(end, 1, swa) - causal_pairs(start, 1, swa)
    compressed_pairs = (causal_pairs(end, r, topk) - causal_pairs(start, r, topk)) if r else 0
    pairs = B * (local_pairs + compressed_pairs)
    if prefill:
        # prefill：每请求每 chunk 一次全量压缩前缀 gather（seq_lens//r 行，
        # flashmla.py L334-346 gather_lens=None → [0, seq_len) 全取）+ SWA
        # 尾窗 gather（gather_len = Q + min(S, swa−1)，ComputePrefillMetadata
        # Kernel L376-384）；记录读在 gather 行，注意力核读 bf16 工作区
        n_comp_rows = (end // r) if r else 0
        gather_len = Q + min(S, swa - 1)
        rows.append(OpRow(
            stage, li, "attn_kv_gather",
            flops=B * (n_comp_rows + gather_len) * 2 * d,
            kv_read=int(B * (n_comp_rows * comp_rec + gather_len * swa_rec)),
            act_write=int(B * (n_comp_rows + gather_len) * d * 2),
            formula=FR.F_CSA2_ATTN_GATHER, statistic="exact",
            shape=f"dequantize_and_gather：压缩前缀 {n_comp_rows}×{comp_rec}B"
                  f" + SWA 尾窗 {gather_len}×{swa_rec}B → bf16 工作区"
                  f"（{d}×2B/行）；dequant ≈2 op/elem 惯例"))
        rows.append(OpRow(
            stage, li, "sparse_attention",
            flops=pairs * kern_heads * (4 * d + 5),
            act_read=int(T * kern_heads * d * 2
                         + pairs * d * 2 + kern_heads * 4),
            act_write=int(T * kern_heads * d * 2 + T * kern_heads * 8),
            formula=FR.F_CSA2_ATTN_CORE, statistic="exact",
            shape=f"flash_mla_sparse_fwd：SWA {local_pairs} + 压缩 "
                  f"{compressed_pairs} 对/请求 × {kern_heads} 头 × (4·{d}+5)"
                  f"（QK 2d + PV 2d_v=2d + softmax 5 op 惯例；工作区逐对"
                  f" bf16 行读为逻辑口径不记跨行复用；out bf16 + "
                  "max_logits/lse fp32；sink 每头标量 4B）"))
        # combine_topk_swa_indices：prefill 每 chunk 每层（flashmla.py
        # L370-383）；R20 误记为 decode 运行首层——decode 路径不调用该
        # Triton 核（只有 specialized decode 变体的 SparseIndexTables 用
        # 等键复用），本轮归位到 prefill
        w_c = -(-((topk if r else 0) + swa) // 128) * 128
        items = pairs
        rows.append(OpRow(
            stage, li, "sparse_index_build",
            flops=items * 3,
            act_read=int(compressed_pairs * B * 4 + T * 16),
            act_write=int(T * w_c * 4 + items * 4 + T * 4),
            formula=FR.F_CSA2_SPARSE_INDEX_BUILD, statistic="approximated",
            shape=f"combine_topk_swa_indices：fill −1 整宽 {w_c}（128 对齐）"
                  f"+ 有效 {items} 项（topk+窗，无去重拼接，cache_utils.py "
                  "L970-1020）；3 op/项惯例"))
    else:
        # decode：flash_mla_with_kvcache 一次核联读 SWA 记录（indices=窗）
        # 与压缩记录（extra_indices=topk），逐行按各自窗/topk 直读分页记录
        rows.append(OpRow(
            stage, li, "sparse_attention",
            flops=pairs * kern_heads * (4 * d + 5),
            act_read=int(T * kern_heads * d * 2 + kern_heads * 4),
            act_write=int(T * kern_heads * d * 2),
            kv_read=int(B * (local_pairs * swa_rec
                             + compressed_pairs * comp_rec)),
            formula=FR.F_CSA2_ATTN_CORE, statistic="exact",
            shape=f"flash_mla_with_kvcache（SM90 v4 记录 584B）：SWA "
                  f"{local_pairs}×{swa_rec}B + 压缩 {compressed_pairs}×"
                  f"{comp_rec}B/请求；{kern_heads} 头 × (4·{d}+5)"
                  f"（Q 零填充 padded_heads；softmax 5 op 惯例；sink 4B/头）"))
        if r:
            # 每压缩层每步：共享 topk 缓冲 → 全局分页槽位（flashmla.py
            # L214-228 compute_global_topk_indices_and_lens；块表逐有效项）
            rows.append(OpRow(
                stage, li, "attn_index_global",
                flops=T * topk * 5,
                act_read=int(T * topk * 4 + compressed_pairs * B * 4),
                act_write=int(T * topk * 4 + T * 4),
                formula=FR.F_CSA2_ATTN_INDEX_GLOBAL, statistic="exact",
                shape="local topk → 全局槽位（整除/查表/取模/乘加 ≈5 op/项"
                      "惯例；topk 缓冲整宽读 + 块表按有效项读）"))
        # SM90 恒 split-KV（sparse_decode.cpp L488 must_split_kv）：逐层
        # o_accum/lse_accum fp32 写 + combine 归并；num_splits 由调度器
        # 移植精确给出（flashmla_sched）。SM100 记录族走 Decode_Sm100_*
        # 实例（有 unsplit 路径、fixed_overhead/block 不同）——本仓无注册
        # SM100 设备，域外显式拒绝，不静默套 SM90 口径
        if rec["family"] != "sm90":
            raise NotImplementedError(
                f"decode split-KV 账本仅按 SM90 实例 pin（sparse_decode.cpp "
                f"Decode_Sm90_Impl）；记录族 {v41_kv_record} 的 decode 行"
                "未建模（SM100 mega/FlashInfer 变体见 G-C16）")
        row_swa = [min(t, swa) for t in range(start + 1, end + 1)] * B
        row_extra = ([min(t // r, topk) for t in range(start + 1, end + 1)] * B
                     if r else [0] * (B * Q))
        splits = flashmla_decode_num_splits(row_swa, row_extra, device_sms or 0)
        n_split_rows = sum(1 for s in splits if s > 1)
        split_total = sum(s for s in splits if s > 1)
        accum = split_total * padded_heads * (d * 4 + 4)
        rows.append(OpRow(
            stage, li, "attn_splitkv",
            flops=split_total * padded_heads * (2 * d + 6),
            act_read=accum, act_write=accum + T * padded_heads * 4,
            formula=FR.F_CSA2_ATTN_SPLITKV, statistic="exact",
            shape=f"SM90 恒 split-KV：{n_split_rows}/{T} 行分片，Σsplits="
                  f"{split_total}；o_accum {d}×4B + lse_accum 4B/头/split，"
                  "combine 2·d_v+6 op/头/split + lse 每行一次"
                  f"（num_sm_parts={device_sms}，调度器移植见 "
                  "flashmla_sched）"))
        if attn_type_first:
            # 每层型每步一次调度器核（tile_sched 空实例 → C++ 分配+规划，
            # sparse_swa.py build_tile_scheduler L911-945 + sparse_decode.cpp
            # L633-656；DecodingSchedMeta=32B，params.h L104-113）
            rows.append(OpRow(
                stage, li, "attn_sched_meta",
                flops=T * 10,
                act_read=int(T * 4),
                act_write=int((device_sms or 0) * 32 + (T + 1) * 4),
                formula=FR.F_CSA2_ATTN_SCHED_META, statistic="approximated",
                shape="get_decoding_sched_meta 单 warp 贪心（每层型每步一次"
                      f"，记在各型首层）：{device_sms}×32B 元数据 + "
                      "num_splits T+1 项；≈10 op/行惯例"))
    if attn_step_first and not prefill:
        # SWA 窗口索引每步一次（全体层共享同一 spec 组的元数据构建；
        # sparse_swa.py L663-732 build() + kv_cache_interface.py 同 spec
        # 分组共享 builder）；记在第 0 层
        swa_entries = B * sum(min(t, swa) for t in range(start + 1, end + 1))
        rows.append(OpRow(
            stage, li, "attn_swa_index_build",
            flops=swa_entries * 5,
            act_read=int(swa_entries * 4),
            act_write=int(swa_entries * 4 + T * 4),
            formula=FR.F_CSA2_ATTN_SWA_INDEX_BUILD, statistic="exact",
            shape="_compute_swa_indices_and_lens（宽 window=128；≈5 op/项"
                  "惯例；每步一次全层共享，不乘层数）"))

    ops, _w, ar, aw = _rope_row(T, heads, d, rd, b_a)
    rows.append(OpRow(stage, li, "o_rope", flops=ops, act_read=ar, act_write=aw,
                      formula=FR.F_ROPE, shape=f"rotary={rd}"))
    # o_a_proj：每 rank 持有 ng=groups/tp 组，每组输入 heads_full·d/groups、
    # 输出 orank；count 必须乘 ng（2026-11-10 审计 F2：旧式只算一组，TP1 少 8×）
    rows.append(gemm("o_a_proj", heads_full * d // groups, orank, reps=ng))
    rows[-1].shape = (f"分组 {ng}×[{heads_full * d // groups},{orank}]"
                      f"（reps={ng}；TP{tp}）")
    rows.append(gemm("o_b_proj", ng * orank, h))
    return rows


# ---------------------------------------------------------------------------
# 层公共件：FFN/MoE/norm/残差 + 整模型装配
# ---------------------------------------------------------------------------


def ffn_or_moe_rows(stage: str, li: int, layer: LayerIR, T: int, h: int,
                    quant: dict, tp: int = 1, ep: int = 1,
                    local_experts: Optional[int] = None,
                    route: Optional[dict] = None,
                    quant_recipe: Optional[dict] = None,
                    act_recipe: Optional[dict] = None) -> List[OpRow]:
    b_w, b_a = quant["dense_w"] / 8, quant["act"] / 8
    rows: List[OpRow] = []
    if layer.moe is not None and layer.moe.experts:
        moe_dicts = moe_rows(layer.moe, T, h, quant, tp=tp, ep=ep,
                             local_experts=local_experts,
                             quant_recipe=quant_recipe,
                             act_recipe=act_recipe)
        if route is not None:
            # R21：V4.1 路由三行（gate/score/topk_renorm）替换通用 moe_router
            # 行（FP32 logits 出口 + sqrtsoftplus/noaux_tc 语义）；routed/
            # shared(+gate) 行不变；R24 的 moe_act/moe_shared_act 行随
            # moe_dicts[1:] 保留。
            moe_dicts = (v41_moe_route_rows(layer.moe, route, T, h)
                         + moe_dicts[1:])
        for r in moe_dicts:
            rows.append(OpRow(stage, li, r["layer_op"], flops=r["flops"],
                              weight_read=r["weight_read"],
                              weight_read_expected=r.get("weight_read_expected"),
                              act_read=r["act_read"], act_write=r["act_write"],
                              weight_capacity=r["weight_capacity"],
                              formula=r["formula"], statistic=r["statistic"],
                              shape=r["shape"]))
    elif layer.dense_inter:
        inter = layer.dense_inter // tp       # dense MLP 列切（同 shared expert）
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
                     quant: dict, first_layer: bool = False,
                     quant_recipe: Optional[dict] = None) -> List[OpRow]:
    """mHC 全宽复制边界：pinned TileLang 算术 op，未执行 kernel。

    字节为 unfused、n_splits=1 逻辑访问参考配方，非实测 HBM；映射见
    docs/validation/v41-mhc-kernels-20261112.md。mix 行拥有 GEMM/sqrsum
    及 prenorm 归并；sinkhorn 行拥有 gates/comb；post_pre 行拥有前一
    子层 post 和当前 collapse/norm。首层 attn 无 post、无 pre_mix_in 读。
    FMA=2，exp/sigmoid/rsqrt=1；norm 沿用 4h+2 近似惯例并单列标记。
      * Engram 层（extras.hc.engram_layer_ids）post/pre 分立执行 + Engram 注入
        ——注入算子已由 engram_rows 建模（2026-11-11 R04，common/engram.py
        pin @f358f057 blob ba4ceea6）；本函数边界算子构成与常规层相同
        （post+pre 总量不变，分立不改变计数）。
    """
    m = hc["mult"]
    iters = hc["sinkhorn_iters"]
    mix = (2 + m) * m                     # mix_hc（hc_attn_fn/hc_ffn_fn 行数）
    hc_dim = m * h
    b_a = quant["act"] / 8
    # R22：V4.1 隐状态沿 model dtype BF16（Marlin W8A16/W4A16 执行域激活
    # 不量化）；mHC 边界的通用 b_a 部分随配方改 2B，FP32 内部字节不动。
    if quant_recipe is not None:
        b_a = float(quant_recipe["act_itemsize"])
    rows: List[OpRow] = []
    if first_layer:
        rows.append(OpRow(
            stage, li, "hc_expand", flops=0,
            act_read=int(T * h * b_a), act_write=int(T * hc_dim * b_a),
            formula=FR.F_MHC_EXPAND, shape=f"嵌入广播 {m} 流 × h；拷贝无浮点运算"))
    for side, fn in (("attn", "hc_attn_fn"), ("ffn", "hc_ffn_fn")):
        entry = first_layer and side == "attn"
        ic = h if entry else hc_dim
        rows.append(OpRow(
            stage, li, f"hc_{side}_mix",
            flops=T * (2 * ic * mix + 2 * ic + mix + 3),
            weight_read=int(mix * ic * 4),             # L0 读派生 broadcast
            weight_capacity=int(mix * hc_dim * 4),
            act_read=int(T * ic * b_a), act_write=int(T * (mix + 1) * 4),
            formula=FR.F_MHC_MIX,
            shape=f"[{T},{ic}]@[{ic},{mix}]·{fn}；GEMM + sqrsum 2IC + "
                  f"prenorm {mix}+3 op/token；split=1 参考，FP32 每卡复制；"
                  f"L0 broadcast 派生缓冲容量另列"))
        axis_ops = m * (m - 1) + m + m * m
        gate_sinkhorn_ops = (8 * m + 2 * m * m
                             + 2 * m * (m - 1) + 4 * m * m
                             + (2 * iters - 1) * axis_ops)
        rows.append(OpRow(
            stage, li, f"hc_{side}_sinkhorn", flops=T * gate_sinkhorn_ops,
            weight_read=T * (mix + 3) * 4, weight_capacity=(mix + 3) * 4,
            act_read=int(T * (mix + 1) * 4), act_write=int(T * mix * 4),
            formula=FR.F_MHC_SINKHORN, statistic="exact",
            shape=f"gates + {m}×{m} comb；行 softmax+eps → 列归一 → "
                  f"{iters - 1}×(行,列)；{gate_sinkhorn_ops} op/token；"
                  f"算术 op 精确（exp/sigmoid=1，非指令数），有限输入"))
        norm_ops = T * (4 * h + 2)
        post_read = 0 if entry else T * ((m * m + m) * 4
                                         + (hc_dim + h) * b_a)
        post_write = 0 if entry else T * hc_dim * b_a
        pre_read = 0 if entry else T * m * 4
        rows.append(OpRow(
            stage, li, f"hc_{side}_post_pre",
            flops=(0 if entry else T * (m * h + 2 * m * m * h))
                  + 2 * T * hc_dim + norm_ops,
            weight_read=int(T * h * b_a),
            act_read=int(post_read + T * hc_dim * b_a + pre_read),
            act_write=int(post_write + T * h * b_a),
            formula=FR.F_MHC_POST_PRE, statistic="approximated",
            shape=f"{'首层无 post；one-hot collapse' if entry else '前一子层 post mh+2m²h；collapse'}"
                  f" 2mh + RMSNorm 4h+2（惯例近似，源码另有归约）；"
                  f"unfused/split=1 逻辑字节，b=act/8，非实测 HBM"))
    return rows


def hc_exit_rows(stage: str, T: int, h: int, hc: dict, quant: dict,
                 quant_recipe: Optional[dict] = None) -> List[OpRow]:
    """末层 post + carried pre_mix collapse；无 learned hc_head。"""
    m, b = hc["mult"], quant["act"] / 8
    if quant_recipe is not None:
        b = float(quant_recipe["act_itemsize"])
    return [
        OpRow(stage, -1, "hc_exit_post", flops=T * (m * h + 2 * m * m * h),
              act_read=int(T * ((m * m + m) * 4 + (m + 1) * h * b)),
              act_write=int(T * m * h * b), formula=FR.F_MHC_POST_PRE,
              shape="仅末层 FFN post；无 collapse/norm；mh+2m²h，unfused 逻辑字节"),
        OpRow(stage, -1, "hc_exit_collapse", flops=2 * T * m * h,
              act_read=int(T * (m * h * b + m * 4)),
              act_write=int(T * h * b), formula=FR.F_MHC_EXIT_COLLAPSE,
              shape="Triton 从零 acc+=pre*x，2mh op；数学最简归约 (2m−1)h；"
                    "final_norm 由既有行拥有"),
    ]


def engram_rows(stage: str, li: int, T: int, h: int, engram_spec: dict,
                quant: dict, tp: int = 1,
                first_engram_layer: bool = False, rank: int = 0,
                quant_recipe: Optional[dict] = None) -> List[OpRow]:
    """Engram 注入算子（v41_csa2 专用；per-rank 口径）。

    证据（vllm models/deepseek_v41/common/engram.py pin @f358f057，
    blob sha1 ba4ceea6；source-checked 2026-11-11 R04；注入点
    nvidia/model.py L583-600、批级哈希 L1030-1092）：
      * 常数不猜：264 B/行（FP8 256 + ue8m0 256/32，L1151-1161/L968-969）、
        24 哈希列 = (max_ngram−1)×n_heads（L243）、wkv 6144→25600 列切
        （L1719-1733）、q/k_weight [hc,d] bf16 每 rank 复制（L1707-1714）；
      * engram_hash：整批一次 kernel 算全部 engram 层（grid
        (cdiv(T,32), n_layers)，L565）→ 只挂首 engram 层行（T=B·Q 一次，
        G13 口径自动满足）；
      * engram_lookup：表行读为**数据依赖期望流量**（同 token 重复 n-gram
        不去重的保守上界，MoE 并集先例单列 weight_read_expected），
        weight_read=0；表容量 = 本 rank 分到的素数桶长之和 × 264（**不是**
        num_embeddings/tp；桶分配见 families.engram_bucket_primes）；
      * engram_wkv：ColumnParallelLinear 列切 /tp，gather_output=True →
        通信行的 all-gather（lowering/sharding）；
      * engram_post_wkv：每 (token,hc) 门控核，op 计数为解析近似口径
        （常数 10d+5 见公式 v2.engram_post_wkv）。
    """
    cols = int(engram_spec["n_hash_cols"])
    d_v = int(engram_spec["head_dim"])
    row_b = int(engram_spec["bytes_per_row"])
    e_max = int(engram_spec["max_ngram_size"])
    n_heads = int(engram_spec["n_heads"])
    layer_ids = list(engram_spec["layer_ids"])
    b_w = quant["dense_w"] / 8
    # R22：wkv 值字节由配方驱动（量化 ColumnParallelLinear；与 memory 同口径）
    if quant_recipe is not None:
        b_w = float(quant_recipe["dense_linear"]["value_itemsize"])
    # TP 合法性：镜像源码断言 (shards−1)·part < n_hash_cols（L1096-1099）。
    if tp < 1:
        raise ValueError("Engram TP 必须为正整数")
    part = -(-cols // tp)
    if (tp - 1) * part >= cols:
        legal = [t for t in range(1, cols + 2)
                 if (t - 1) * (-(-cols // t)) < cols]
        raise ValueError(
            f"Engram TP{tp} 非法：{cols} 个哈希头 ceil 切分 part={part}，"
            f"(tp−1)·part={(tp - 1) * part} ≥ {cols}（engram.py L1096 断言；"
            f"合法 tp = {legal}）")
    if not 0 <= rank < tp:
        raise ValueError(f"Engram rank {rank} 不在 TP{tp} 范围内")
    # 源码 lookup 只处理本 rank 的 part_n_hash_cols；gather 后才有全量列。
    head_start = rank * part
    local_cols = min(part, cols - head_start)
    # 本层素数桶容量与同一 rank 的查表流量对齐。
    primes = engram_bucket_primes(int(engram_spec["vocab_size"]), len(layer_ids),
                                  e_max, n_heads)
    rank_rows = sum(primes[layer_ids.index(li)][head_start:head_start + local_cols])
    rows: List[OpRow] = []
    if first_engram_layer:
        rows.append(OpRow(
            stage, li, "engram_hash",
            flops=T * len(layer_ids)
            * (2 * e_max + 2 * (e_max - 1) * n_heads),
            act_read=int(T * 4),
            act_write=int(T * len(layer_ids) * cols * 4),
            formula=FR.F_ENGRAM_HASH, statistic="exact",
            shape=f"批级一次 kernel 算全部 {len(layer_ids)} 个 engram 层"
                  f"（grid (cdiv(T,32),n_layers)，L565）；整数 op 计数："
                  f"每 (token,layer,shift) 1 mul+1 xor ×{e_max} + shift>0 "
                  f"每 head 1 mod+1 add ×{e_max - 1}×{n_heads}"))
    rows.append(OpRow(
        stage, li, "engram_lookup",
        flops=T * local_cols * d_v,
        weight_read=0,
        weight_read_expected=float(T * local_cols * row_b),
        act_read=int(T * local_cols * 4), act_write=int(T * local_cols * d_v * 2),
        weight_capacity=int(rank_rows * row_b),
        formula=FR.F_ENGRAM_LOOKUP, statistic="expectation",
        shape=f"[T,{cols}] int32 ids → 本 rank [T,{local_cols},{d_v}] bf16"
              f"（dequant 逐元素 1 乘，L726-731）；表行读 {T * local_cols}×{row_b}B 为数据依赖期望"
              f"流量（weight_read_expected；重复 n-gram 不去重保守上界），"
              f"weight_read=0；表容量另计（rank{rank} 分片 {rank_rows} 行×{row_b}B）"))
    wkv_in = cols * d_v                            # 6144（L1703-1705）
    wkv_out = h * (int(engram_spec["hc_mult"]) + 1)  # 25600（L1726）
    if wkv_out % tp:
        raise ValueError(
            f"Engram wkv 输出宽 {wkv_out} 不整除 TP{tp}（ColumnParallelLinear "
            "列切要求整除，L1724-1733；V4.1 部署配方 tp∈{{1,2,4,8}}）")
    oc_r = wkv_out // tp
    # R22：wkv 为量化 ColumnParallelLinear（quant_config 传入，L1724-1729）
    # → MXFP8 布局含 ue8m0 scale（oc_r×wkv_in/32；与 memory 同口径）。
    wkv_scale_b = 0
    if quant_recipe is not None:
        from .quant_layout import mxfp8_linear_bytes
        _v, wkv_scale_b = mxfp8_linear_bytes(oc_r, wkv_in)
    rows.append(OpRow(
        stage, li, "engram_wkv",
        flops=2 * T * wkv_in * oc_r,
        weight_read=int(wkv_in * oc_r * b_w) + wkv_scale_b,
        weight_capacity=int(wkv_in * oc_r * b_w) + wkv_scale_b,
        act_read=int(T * wkv_in * 2), act_write=int(T * oc_r * 2),
        formula=FR.F_ENGRAM_WKV, statistic="exact",
        shape=f"[{T},{wkv_in}]@[{wkv_in},{oc_r}]（列切 25600/tp={oc_r}，"
              f"gather_output=True → 全宽 [T,{wkv_out}] all-gather 见通信行）"))
    hc = int(engram_spec["hc_mult"])
    rows.append(OpRow(
        stage, li, "engram_post_wkv",
        flops=T * hc * (10 * h + 5),
        weight_read=int(2 * hc * h * 2), weight_capacity=int(2 * hc * h * 2),
        act_read=int(T * hc * h * 2 + T * (hc + 1) * h * 2 + T),
        act_write=int(T * hc * h * 2),
        formula=FR.F_ENGRAM_POST_WKV, statistic="exact",
        shape=f"grid (T×{hc},)（L1879）：ρ=(Σ hidden·q·k·key)·rsqrt(mean h²+eps)"
              f"·rsqrt(mean key²+eps)·rsqrt(d)，gate=sigmoid(sign(ρ)·"
              f"sqrt(clamp|ρ|,1e-6))，out=hidden+gate·value（L1334-1422）；"
              f"op 计数口径：乘 6d + 加 3(d−1)+d + 标量≈8 = 10d+5；"
              f"mask bool {T}B；q/k_weight [hc,d] bf16 每 rank 复制"))
    return rows


def head_rows(stage: str, model: ModelIR, T: int, quant: dict, tp: int,
              quant_recipe: Optional[dict] = None) -> List[OpRow]:
    h, v = model.hidden, model.vocab
    b_w, b_a = quant["dense_w"] / 8, quant["act"] / 8
    # R22：V4.1 embed/lm_head/RMSNorm 权重未被量化（get_quant_method 对非
    # LinearBase/RoutedExperts/Attention 返回 None → 默认方法 BF16；fp8.py
    # L174-203 + vocab_parallel_embedding.py L322-326），激活 BF16。
    if quant_recipe is not None:
        b_w = float(quant_recipe["unquantized_itemsize"])
        b_a = float(quant_recipe["act_itemsize"])
    rows: List[OpRow] = []
    # final norm（model.norm，每 rank 复制；2026-11-10 补齐——HF/vLLM forward
    # 均有 final RMSNorm，此前缺行）
    ops, w, ar, aw = _norm_row(T, h, b_a)
    if quant_recipe is not None:
        # RMSNorm 权重沿 model dtype BF16（quant_config 不覆盖 norm 参数）
        w = int(h * quant_recipe["unquantized_itemsize"])
    rows.append(OpRow(stage, -1, "final_norm", flops=ops, weight_read=w,
                      weight_capacity=w,
                      act_read=ar, act_write=aw, formula=FR.F_RMSNORM,
                      shape=f"width={h}（final，replicated）"))
    rows.append(OpRow(stage, -1, "lm_head", flops=2 * T * h * (v // tp),
                      weight_read=int(h * (v // tp) * b_w),
                      act_read=int(T * h * b_a), act_write=int(T * (v // tp) * b_a),
                      weight_capacity=int(h * (v // tp) * b_w),
                      formula=FR.F_LM_HEAD, shape=f"h×V/tp（列切分）"))
    if not model.tie_embeddings:
        rows.append(OpRow(stage, -1, "embed_lookup",
                          act_read=int(T * h * b_a), act_write=int(T * h * b_a),
                          weight_capacity=int(h * (v // tp) * b_w),
                          formula=FR.F_EMBED,
                          shape=f"查表 1 行/token；容量 h×V/tp（词表行切）"))
    return rows


def build_model_work(model: ModelIR, stage: str, B: int, ctx: int, queries: int,
                     quant: dict, tp: int = 1, prefill: bool = False,
                     role: str = "target", ep: int = 1, rank: int = 0,
                     plan=None, v41_kv_record: str = "fp8_ds_mla_sm90",
                     quant_recipe: Optional[dict] = None,
                     device_sms: Optional[int] = None,
                     ) -> List[OpRow]:
    """整模型一个阶段的逻辑账本（**per-rank 口径**，默认 rank=0）。

    decode/verify：ctx=查询前缓存长度 S；queries=新查询数 Q。
    prefill：S=0、queries=ctx（Q=整段）。
    ep/rank：EP 开时 MoE 行按该 rank 的本地专家数（余数 rank +1）；
    plan：可传入 lowering/parallel.ParallelPlan 校验一致性（缺省按
    (tp, ep) 直接求值，合法性与 plan 相同规则）。
    v41_kv_record：R20 显式 KV 记录配方（scenario.V41_KV_RECORDS；
    仅 v41_csa2 消费，pipeline 从场景 deployment 透传）。
    device_sms：R25 设备 SM 数（devices.sm_count；pipeline 透传）——V4.1
    decode 的 split-KV 调度器移植需要；缺失时 v41 decode 行显式拒绝
    （未知不得默认）。
    """
    h = model.hidden
    S, Q = (0, ctx) if prefill else (ctx, queries)
    rows: List[OpRow] = []
    if quant_recipe is None:
        quant_recipe = (model.extras or {}).get("quant")
    kvs = (model.extras or {}).get("kv_spec", {})
    hc_spec = (model.extras or {}).get("hc")
    engram_spec = (model.extras or {}).get("engram")
    moe_route = (model.extras or {}).get("moe_route")
    # R24：V4.1 SwiGLU clamp 激活配方（extras.swiglu；Qwen36 等其他家族
    # 无此键 → 无 act 行，其 silu 微操作维持 R06 冻结口径，见
    # tests/test_qwen36_forward_r06.py 的登记差异）
    act_recipe = (model.extras or {}).get("swiglu")
    first_layer_index = model.layers[0].index if model.layers else None
    if moe_route:
        # R21：V4.1 保守 EP 整除域（mega 路径 pinned 断言 L357-362；
        # fused 通用余数映射未入 pin/未建模，见 count/moe.require_divisible_ep）
        for layer in model.layers:
            if layer.moe is not None and layer.moe.experts:
                require_divisible_ep(layer.moe.experts, ep)

    def _local_experts(layer):
        if plan is not None:
            return plan.moe_local_experts(layer, rank) or None
        if layer.moe is None or not layer.moe.experts:
            return None
        if ep == 1:
            return layer.moe.experts
        base, rem = divmod(layer.moe.experts, ep)
        return base + (1 if rank < rem else 0)

    # R25：FlashMLA 注意力核的每步/每层型一次性核归层——SWA 窗口索引每步
    # 一次（全体层共享同 spec 组的元数据构建，记在第 0 层）；调度器元数据
    # 每层型（compress_ratio 去重值）每步一次（记在各型首层）
    prev_ratio = object()          # 哨兵：首个 CSA2 层必为其型首层
    first_csa2 = True
    for layer in model.layers:
        li = layer.index
        type_first = False
        if layer.kind == MLA_CSA2 and layer.csa2 is not None:
            if layer.csa2.ratio != prev_ratio:
                type_first = True
            prev_ratio = layer.csa2.ratio
        step_first = first_csa2 and layer.kind == MLA_CSA2
        if layer.kind == MLA_CSA2:
            first_csa2 = False
        if layer.kind in (FULL, SLIDING) and layer.attn:
            rows += gqa_layer_rows(stage, li, layer, B, S, Q, h, quant, tp)
        elif layer.kind == LINEAR and layer.linear:
            rows += gdn_layer_rows(stage, li, layer, B, Q, h, quant, prefill, tp)
        elif layer.kind == MLA and layer.mla:
            rows += mla_layer_rows(stage, li, layer, B, S, Q, h, quant, prefill, tp)
        elif layer.kind == MLA_CSA2 and layer.mla and layer.csa2:
            rows += v41_layer_rows(stage, li, layer, B, S, Q, h, kvs, quant,
                                   prefill, tp, v41_kv_record=v41_kv_record,
                                   quant_recipe=quant_recipe,
                                   device_sms=device_sms,
                                   attn_type_first=type_first,
                                   attn_step_first=step_first)
        T = B * Q
        rows += ffn_or_moe_rows(stage, li, layer, T, h, quant, tp=tp, ep=ep,
                                local_experts=_local_experts(layer),
                                route=moe_route, quant_recipe=quant_recipe,
                                act_recipe=act_recipe)
        if layer.kind == MLA_CSA2 and hc_spec:
            # mHC 边界核融合 norm+残差（F_MHC_POST_PRE）：不再另计通用 norm/add 行
            rows += hc_boundary_rows(stage, li, T, h, hc_spec, quant,
                                     first_layer=(li == first_layer_index),
                                     quant_recipe=quant_recipe)
        else:
            rows += norm_add_rows(stage, li, T, h, quant)
        # Engram 注入（R04；层位 extras.engram.layer_ids，与 hc_spec 同源 extras；
        # 哈希行只挂首个 engram 层——批级 kernel 一次算全部层）
        if (engram_spec and li in engram_spec.get("layer_ids", ())
                and layer.kind == MLA_CSA2):
            rows += engram_rows(stage, li, T, h, engram_spec, quant, tp,
                                first_engram_layer=(li == engram_spec["layer_ids"][0]),
                                rank=rank, quant_recipe=quant_recipe)
    if hc_spec and any(layer.kind == MLA_CSA2 for layer in model.layers):
        rows += hc_exit_rows(stage, B * Q, h, hc_spec, quant,
                             quant_recipe=quant_recipe)
    rows += head_rows(stage, model, B * Q, quant, tp,
                      quant_recipe=quant_recipe)
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
