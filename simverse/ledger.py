"""逐算子账本（Count 段）：结构 × 负载 → 每算子整数计数。

计数规则的两个来源（移植处逐条标注）：
  * V4.1 Flash 注意力子图 —— attention_structure.deepseek_attention（旧仓，已审阅）
    的忠实移植：逐层 producer/consumer DAG 计数、causal_pairs 精确闭式、
    SWA/压缩稀疏两段读、每消费者各读一次缓存（共享缓存≠共享读取）。
  * MoE / dense FFN / lm_head —— model_analyzer._analyze_moe_ffn 与 V41 适配器
    post_process 的分解展开（聚合行拆为 router/routed/shared 三行，差异报告 D-8）。

口径：所有行 per_card（TP 分片后）；每行绑定 layer/op/执行次数/公式 id/形状串。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from . import formulas as F
from .contracts import ModelSpec, OpRecord


# ---------------------------------------------------------------------------
# 旧仓精确闭式移植（attention_structure.py:119-128）
# ---------------------------------------------------------------------------

def _sum_floor(n: int, ratio: int) -> int:
    """Σ_{t=1..n} floor(t/ratio)，闭式精确。"""
    q, m = divmod(n, ratio)
    return ratio * q * (q - 1) // 2 + q * (m + 1)


def causal_pairs(n: int, ratio: int, cap: int) -> int:
    """Σ_{t=1..n} min(floor(t/ratio), cap)，闭式精确（压缩因果对数）。"""
    end = min(n, ratio * cap)
    return _sum_floor(end, ratio) + max(0, n - end) * cap


def _norm_row(tokens: int, width: int, ab: int) -> Tuple[int, int, int, int]:
    """RMSNorm：4D+2 OP/行；尺度权重按 BF16 2B；rsqrt 记 1 OP（旧仓 norm()）。"""
    ops = tokens * (4 * width + 2)
    weight = 2 * width if tokens else 0
    act = tokens * width * ab
    return ops, weight, act, act


def _rope_row(tokens: int, heads: int, width: int, rotary: int, ab: int) -> Tuple[int, int, int, int]:
    """RoPE：每旋转元素 3 OP；未旋转部分保留（旧仓 rope()）。"""
    ops = 3 * tokens * heads * rotary
    act_r = tokens * (heads * width + rotary) * ab
    act_w = tokens * heads * width * ab
    return ops, 0, act_r, act_w


class Ledger:
    """一个阶段（baseline_decode / prefill / verify / draft_step_j）的算子账本。"""

    def __init__(self, stage: str):
        self.stage = stage
        self.rows: List[OpRecord] = []

    def add(self, layer: int, op: str, *, flops: int = 0, weight_read: int = 0,
            act_read: int = 0, act_write: int = 0, kv_read: int = 0, kv_write: int = 0,
            weight_capacity: int = 0, formula_id: str = F.F_GEMM_FLOPS,
            shape_note: str = "", exec_count: int = 1) -> OpRecord:
        rec = OpRecord(stage=self.stage, layer=layer, op=op, exec_count=exec_count,
                       flops=flops, weight_read=weight_read, act_read=act_read,
                       act_write=act_write, kv_read=kv_read, kv_write=kv_write,
                       weight_capacity=weight_capacity, scope="per_card",
                       formula_id=formula_id, shape_note=shape_note)
        self.rows.append(rec)
        return rec

    def gemm(self, layer: int, op: str, ic: int, oc: int, tokens: int, w_bytes_per_param: float,
             groups: int = 1, note: str = "", ab: float = 1.0) -> OpRecord:
        """GEMM：FLOPs=2·T·ic·oc·g；权重=ic·oc·g·字节宽；激活读写=形状×ab(B/elem)。"""
        return self.add(
            layer, op,
            flops=2 * tokens * ic * oc * groups,
            weight_read=int(ic * oc * groups * w_bytes_per_param),
            act_read=int(tokens * ic * groups * ab), act_write=int(tokens * oc * groups * ab),
            weight_capacity=int(ic * oc * groups * w_bytes_per_param),
            shape_note=f"GEMM {groups}×[T={tokens},{ic}]@[{ic},{oc}]"
                       f" (w={w_bytes_per_param}B/param)" + (f"; {note}" if note else ""))

    def totals(self) -> Dict[str, int]:
        out = dict(flops=0, weight_read=0, act_read=0, act_write=0,
                   kv_read=0, kv_write=0, op_count=0)
        for r in self.rows:
            out["flops"] += r.flops
            out["weight_read"] += r.weight_read
            out["act_read"] += r.act_read
            out["act_write"] += r.act_write
            out["kv_read"] += r.kv_read
            out["kv_write"] += r.kv_write
            out["op_count"] += 1
        return out


# ---------------------------------------------------------------------------
# GQA dense 家族（tiny oracle）
# ---------------------------------------------------------------------------

def build_gqa_layer(led: Ledger, spec: ModelSpec, layer_idx: int, batch: int,
                    ctx: int, queries: int, tp: int, *, prefill: bool) -> None:
    """标准 GQA + dense FFN 一层（手算可复核）。

    decode/verify 约定：ctx 为查询前的缓存长度；Q 个查询位共享一次历史 KV 读
    （旧仓 spec_q 口径），写入 Q 行新 KV；attention 对数 = B·(S·Q + Q(Q+1)/2)。
    """
    h = spec.hidden
    attn = spec.layers[layer_idx].attn
    heads, kv_heads, d = attn["heads"], attn["kv_heads"], attn["head_dim"]
    qbits, abits, kbits = spec.quant["dense_w"], spec.quant["act"], spec.quant["kv"]
    wb, ab, kb = qbits / 8, abits / 8, kbits / 8
    assert heads % tp == 0
    heads_c = heads // tp
    T = batch * queries
    S = 0 if prefill else ctx

    # 注意力投影（TP 按 N 维切；顺序 = 数据依赖：q/k/v → core → out）
    for name, (ic, oc) in (("q_proj", (h, heads_c * d)), ("k_proj", (h, kv_heads * d)),
                           ("v_proj", (h, kv_heads * d))):
        led.gemm(layer_idx, name, ic, oc, T, wb, ab=ab)
    # attention 核心
    pairs = batch * (S * queries + queries * (queries + 1) // 2)
    kv_elems = 2 * kv_heads * d
    led.add(layer_idx, "attention_core",
            flops=pairs * heads_c * (4 * d + 5),
            act_read=T * heads_c * d * ab, act_write=T * heads_c * d * ab,
            kv_read=batch * S * kv_elems * kb,
            kv_write=T * kv_elems * kb,
            formula_id=F.F_GEMM_FLOPS,
            shape_note=f"pairs={pairs}, heads(card)={heads_c}, KV read {batch * S} 行×{kv_elems} elem")
    led.gemm(layer_idx, "out_proj", heads_c * d, h, T, wb, ab=ab)
    # FFN dense（SwiGLU 3 矩阵拆 gate_up + down）
    ffn = 512
    led.gemm(layer_idx, "gate_up_proj", h, 2 * ffn // tp, T, wb, ab=ab)
    led.gemm(layer_idx, "down_proj", ffn // tp, h, T, wb, ab=ab)
    # 两个 RMSNorm + 两个残差加
    for name in ("attn_norm", "mlp_norm"):
        ops, w, ar, aw = _norm_row(T, h, ab)
        led.add(layer_idx, name, flops=ops, weight_read=w, act_read=ar, act_write=aw,
                formula_id=F.F_ACT_BYTES, shape_note=f"RMSNorm T={T}, width={h}")
    for name in ("attn_add", "mlp_add"):
        led.add(layer_idx, name, act_read=2 * T * h * ab, act_write=T * h * ab,
                formula_id=F.F_ACT_BYTES, shape_note="残差加")


def build_gqa_model(led: Ledger, spec: ModelSpec, batch: int, ctx: int, queries: int,
                    tp: int, *, prefill: bool) -> None:
    h, v = spec.hidden, spec.vocab
    wb, ab = spec.quant["dense_w"] / 8, spec.quant["act"] / 8
    T = batch * queries
    for li in range(len(spec.layers)):
        build_gqa_layer(led, spec, li, batch, ctx, queries, tp, prefill=prefill)
    led.gemm(-1, "lm_head", h, v // tp, T, wb, note="列切分 per card", ab=ab)
    led.add(-1, "embed_lookup", act_read=T * h * ab, act_write=T * h * ab,
            weight_capacity=int(h * v * wb),  # untied：embed 驻留单列
            formula_id=F.F_ACT_BYTES, shape_note="查表读 1 行/token；容量 h×v")


# ---------------------------------------------------------------------------
# V4.1 Flash（MLA + CSA2 + 全层 MoE）
# ---------------------------------------------------------------------------

def _v41_attn_layer(led: Ledger, spec: ModelSpec, li: int, batch: int, ctx: int,
                    queries: int, tp: int, *, prefill: bool) -> None:
    """一层 MLA+CSA2 注意力子图（attention_structure.deepseek_attention 移植）。

    计数规则与旧仓逐条对应：q_a/kv/q_b/o_a/o_b 投影、SWA 写、压缩 KV 生产线
    (compress_proj FP16 权重、pooling 估计、index_k)、索引选择（短上下文 eager
    旁路、长上下文全候选打分）、sparse_attention 两段读（SWA 窗口 + 压缩 top-k）。
    """
    kvs = spec.kv_spec
    h = spec.hidden
    heads_full, d = kvs["heads"], kvs["head_dim"]
    qrank, orank, groups = kvs["q_lora_rank"], kvs["o_lora_rank"], kvs["o_groups"]
    rd, di, ih = kvs["qk_rope_dim"], kvs["index_head_dim"], kvs["index_heads"]
    topk, swa = kvs["index_topk"], kvs["sliding_window"]
    ctb = kvs["candidate_topk_blocks"]
    dense_b = spec.quant["dense_w"] / 8
    ab, kb = spec.quant["act"] / 8, spec.quant["kv"] / 8
    fp16_ops = set(kvs["fp16_weight_ops"])
    heads, ng = heads_full // tp, groups // tp

    layer = spec.layers[li]
    a = layer.attn
    r = a["ratio"]
    S, Q = (0, ctx) if prefill else (ctx, queries)
    start, length = (0, S) if prefill else (S, Q)
    end = start + length
    T = batch * length

    def gemm16(op, ic, oc, tokens, groups_n=1, fp16=False):
        led.gemm(li, op, ic, oc, tokens, 2.0 if fp16 else dense_b, groups=groups_n,
                 ab=ab,
                 note="权重 FP16（旧仓硬编码 weight_byte=2）" if fp16 else "")

    led.gemm(li, "q_a_proj", h, qrank, T, dense_b, ab=ab)
    led.gemm(li, "kv_proj", h, d, T, dense_b, ab=ab)
    ops, w, ar, aw = _norm_row(T, qrank, ab)
    led.add(li, "q_norm", flops=ops, weight_read=w, act_read=ar, act_write=aw,
            formula_id=F.F_ACT_BYTES, shape_note=f"RMSNorm T={T}, width={qrank}")
    ops, w, ar, aw = _norm_row(T, d, ab)
    led.add(li, "k_norm", flops=ops, weight_read=w, act_read=ar, act_write=aw,
            formula_id=F.F_ACT_BYTES, shape_note=f"RMSNorm T={T}, width={d}")
    led.gemm(li, "q_b_proj", qrank, heads * d, T, dense_b, ab=ab)
    ops, _rw, ar, aw = _rope_row(T, heads, d, rd, ab)
    led.add(li, "q_rope", flops=ops, act_read=ar, act_write=aw,
            formula_id=F.F_ACT_BYTES, shape_note=f"RoPE rotary={rd}, heads(card)={heads}")
    ops, _rw, ar, aw = _rope_row(T, 1, d, rd, ab)
    led.add(li, "k_rope", flops=ops, act_read=ar, act_write=aw,
            formula_id=F.F_ACT_BYTES, shape_note=f"RoPE rotary={rd}, 1 KV 头")
    led.add(li, "swa_cache_write", act_read=T * d * ab, kv_write=T * d * kb,
            formula_id=F.F_KV_RW, shape_note=f"每层本地 SWA 缓存写 {T} 行×{d} elem")

    new_rows = batch * (end // r - start // r) if r else 0
    if a["produces_kv"]:
        m = 2 if r == 2 else 1
        gemm16("compress_proj", h, d * m, T, fp16=True)
        # compress_norm：RMSNorm + ratio2 的 softmax pooling/state 读写估计（旧仓口径）
        ops, w, ar, aw = _norm_row(new_rows, d, ab)
        kv_state = batch * d * 2 * 4 * length
        if r == 2:
            ops += new_rows * d * 12
            ar += T * 2 * d * ab
        led.add(li, "compress_norm", flops=ops, weight_read=w, act_read=ar, act_write=aw,
                kv_read=kv_state if r == 2 else 0, kv_write=kv_state if r == 2 else 0,
                formula_id=F.F_ACT_BYTES,
                shape_note="压缩 pooling 估计（旧仓注明非实测）；ratio2 含 fp32 state 读写")
        ops, _rw, ar, aw = _rope_row(new_rows, 1, d, rd, ab)
        led.add(li, "compressed_rope", flops=ops, act_read=ar, act_write=aw,
                formula_id=F.F_ACT_BYTES, shape_note=f"new_rows={new_rows}")
        led.add(li, "compressed_cache_write", act_read=new_rows * d * ab,
                kv_write=new_rows * d * kb, formula_id=F.F_KV_RW,
                shape_note=f"压缩 KV 潜层写 {new_rows} 行")
        gemm16("index_k_proj", d, di, T, fp16=True)
        ops, w, ar, aw = _norm_row(new_rows, di, ab)
        led.add(li, "index_k_norm", flops=ops, weight_read=w, act_read=ar, act_write=aw,
                formula_id=F.F_ACT_BYTES, shape_note=f"RMSNorm width={di}")
        ops, _rw, ar, aw = _rope_row(new_rows, 1, di, rd, ab)
        led.add(li, "index_k_rope", flops=ops, act_read=ar, act_write=aw,
                formula_id=F.F_ACT_BYTES, shape_note=f"RoPE width={di}")
        led.add(li, "index_cache_write", act_read=new_rows * di * ab,
                kv_write=new_rows * di * kb, formula_id=F.F_KV_RW,
                shape_note=f"索引 K 缓存写 {new_rows} 行")

    if a["produces_index"]:
        short = end // r <= topk
        gemm16("index_weights_proj", h, ih, T, fp16=True)
        if not short:
            led.gemm(li, "index_q_proj", qrank, ih * di, T, dense_b, ab=ab)
            ops, _rw, ar, aw = _rope_row(T, ih, di, rd, ab)
            led.add(li, "index_q_rope", flops=ops, act_read=ar, act_write=aw,
                    formula_id=F.F_ACT_BYTES, shape_note=f"{ih} 索引头×{di}")
        else:
            led.add(li, "index_q_proj", shape_note="eager 短上下文旁路（旧仓 fast path）")
        pairs = 0 if short else batch * (_sum_floor(end, r) - _sum_floor(start, r))
        read_rows = 0 if short else (end // r if not prefill else pairs // batch)
        led.add(li, "index_select",
                flops=2 * pairs * ih * di + T * topk,
                kv_read=batch * read_rows * di * kb,
                act_write=T * topk * 4,
                formula_id=F.F_GEMM_FLOPS,
                shape_note=f"全候选打分 pairs={pairs}（不假定 sparse-logits 已启用）；top-{topk} 选择")
        if a["produces_candidates"]:
            led.add(li, "candidate_publish",
                    act_write=0 if short else T * ctb * 4,
                    formula_id=F.F_ACT_BYTES,
                    shape_note=f"发布候选 block IDs ≤{ctb}（与 indexer 同一融合接口）")

    local_pairs = causal_pairs(end, 1, swa) - causal_pairs(start, 1, swa)
    compressed_pairs = (causal_pairs(end, r, topk) - causal_pairs(start, r, topk)) if r else 0
    read_swa = min(end, swa) if not prefill else local_pairs
    read_comp = (min(end // r, topk) if (r and not prefill) else compressed_pairs)
    led.add(li, "sparse_attention",
            flops=batch * (local_pairs + compressed_pairs) * heads * (4 * d + 5),
            act_read=T * heads * d * ab, act_write=T * heads * d * ab,
            kv_read=batch * (read_swa + read_comp) * d * kb,
            formula_id=F.F_GEMM_FLOPS,
            shape_note=(f"SWA 读 {read_swa} 行 + 压缩读 {read_comp} 行（每消费者各读一次）；"
                        f"heads(card)={heads}"))
    ops, _rw, ar, aw = _rope_row(T, heads, d, rd, ab)
    led.add(li, "o_rope", flops=ops, act_read=ar, act_write=aw,
            formula_id=F.F_ACT_BYTES, shape_note=f"RoPE rotary={rd}")
    led.gemm(li, "o_a_proj", heads_full * d // groups, orank, T, dense_b, groups=ng, ab=ab,
             note=f"分组 {ng}×[{heads_full * d // groups},{orank}]")
    led.gemm(li, "o_b_proj", ng * orank, h, T, dense_b, ab=ab)


def _v41_moe_layer(led: Ledger, spec: ModelSpec, li: int, tokens: int, tp: int,
                   draws_override: Optional[int] = None) -> None:
    """一层 MoE：router / routed（含合并写）/ shared 三行（旧仓 moe_ffn 聚合的分解）。

    draws = T·top_k 次专家抽取 → 期望并集 U(draws) 个专家的权重读取（F_UNION_IID）。
    """
    h = spec.hidden
    moe = spec.layers[li].moe
    E, k, inter = moe["experts"], moe["top_k"], moe["inter"]
    si = moe["shared_inter"]
    wb, ewb, ab = spec.quant["dense_w"] / 8, spec.quant["expert_w"] / 8, spec.quant["act"] / 8
    mi, si_c = inter // tp, si // tp
    T = tokens
    draws = draws_override if draws_override is not None else T * k
    union = E * (1.0 - (1.0 - 1.0 / E) ** draws)

    led.add(li, "moe_router", flops=2 * T * h * E,
            weight_read=int(h * E * wb), weight_capacity=int(h * E * wb),
            formula_id=F.F_GEMM_FLOPS,
            shape_note=f"[{T},{h}]@[{h},{E}]；新仓补计 router 权重读（旧仓聚合未单列）")
    led.add(li, "moe_routed",
            flops=2 * T * k * 3 * h * mi,
            weight_read=int(union * 3 * h * mi * ewb),
            weight_capacity=int(E * 3 * h * mi * ewb),
            act_read=int(T * h * ab * k + 6 * k * T * mi * ab),
            act_write=int(T * h * ab),
            formula_id=F.F_UNION_IID,
            shape_note=(f"draws={draws} → E[U]={union:.2f}/{E} 专家（{ewb} B/param）；"
                        f"每卡 mi={mi}"))
    led.add(li, "moe_shared",
            flops=2 * T * 3 * h * si_c,
            weight_read=int(3 * h * si_c * wb), weight_capacity=int(3 * h * si_c * wb),
            act_read=int(T * h * ab + 6 * T * si_c * ab),
            formula_id=F.F_GEMM_FLOPS,
            shape_note=f"共享专家 ×1（每卡 si={si_c}）")


def _v41_norms(led: Ledger, spec: ModelSpec, li: int, tokens: int) -> None:
    h = spec.hidden
    ab = spec.quant["act"] / 8
    for name in ("attn_norm", "mlp_norm"):
        ops, w, ar, aw = _norm_row(tokens, h, ab)
        led.add(li, name, flops=ops, weight_read=w, act_read=ar, act_write=aw,
                formula_id=F.F_ACT_BYTES, shape_note=f"RMSNorm width={h}")
    for name in ("attn_add", "mlp_add"):
        led.add(li, name, act_read=2 * tokens * h * ab, act_write=tokens * h * ab,
                formula_id=F.F_ACT_BYTES, shape_note="残差加")


def build_v41_model(led: Ledger, spec: ModelSpec, batch: int, ctx: int, queries: int,
                    tp: int, *, prefill: bool) -> None:
    h, v = spec.hidden, spec.vocab
    wb, ab = spec.quant["dense_w"] / 8, spec.quant["act"] / 8
    S, Q = (0, ctx) if prefill else (ctx, queries)
    T = batch * (S + Q if prefill else Q)
    if spec.kv_spec["heads"] % tp or spec.kv_spec["o_groups"] % tp:
        raise ValueError("TP 必须整除 Q 头数与输出分组数（旧仓同校验）")
    for li in range(len(spec.layers)):
        _v41_attn_layer(led, spec, li, batch, ctx, queries, tp, prefill=prefill)
        _v41_moe_layer(led, spec, li, T, tp)
        _v41_norms(led, spec, li, T)
    led.gemm(-1, "lm_head", h, v // tp, T, wb, note="列切分 per card", ab=ab)
    led.add(-1, "embed_lookup", act_read=T * h * ab, act_write=T * h * ab,
            weight_capacity=int(h * v * wb),
            formula_id=F.F_ACT_BYTES, shape_note="查表读 1 行/token；untied 容量 h×v")


def build_model_ledger(spec: ModelSpec, stage: str, batch: int, ctx: int, queries: int,
                       tp: int, *, prefill: bool = False) -> Ledger:
    led = Ledger(stage)
    family = spec.kv_spec["family"]
    if family == "gqa":
        build_gqa_model(led, spec, batch, ctx, queries, tp, prefill=prefill)
    elif family == "v41_csa2":
        build_v41_model(led, spec, batch, ctx, queries, tp, prefill=prefill)
    else:
        raise ValueError(f"未知结构家族 {family}")
    return led


# ---------------------------------------------------------------------------
# 草稿步账本（链式：M=1，Catch-up 步 M=2）
# ---------------------------------------------------------------------------

def build_draft_ledger(stage: str, spec: ModelSpec, draft, tokens: int, ctx: int,
                       tp: int) -> Ledger:
    """一个草稿步（M=tokens 个 token 的草稿前向）的整数账本。

    口径（链式草稿 v0.1）：
      * dense 草稿：权重读 = params×weight_read_scale×字节宽（TP 下每卡 /tp）
        + KV 读 = 目标单步 KV 读 × kv_frac（旧仓 "ratio" 折算口径，标注近似）
      * MoE 草稿（dspark）：专家权重读 = U(M·k) 并集（F_UNION_IID）+ kv_frac 比例 KV
      * FLOPs = 2·params·M（聚合参数口径，未展开层结构 —— SpecMethod.approximations）
    """
    led = Ledger(stage)
    wb = draft.weight_bits / 8
    ab = spec.quant["act"] / 8
    M = tokens
    # 目标单步（B=该草稿批, Q=1）KV 读，用于 kv_frac 折算
    target_kv = _reference_kv_read(spec, M, ctx, tp)

    if draft.moe:
        moe = draft.moe
        E, k, inter, dh = moe["experts"], moe["top_k"], moe["inter"], moe["hidden"]
        draws = M * k
        union = E * (1.0 - (1.0 - 1.0 / E) ** draws)
        per_expert = 3 * dh * inter
        led.add(-1, "draft_moe_routed",
                flops=2 * M * k * per_expert,
                weight_read=int(union * per_expert * wb * draft.weight_read_scale),
                weight_capacity=int(E * per_expert * wb),
                act_read=int(M * dh * ab * k), act_write=int(M * dh * ab),
                formula_id=F.F_UNION_IID,
                shape_note=f"草稿 MoE: draws={draws} → E[U]={union:.2f}/{E}；config 公式口径 4.53B")
    else:
        p = draft.params_dense
        led.add(-1, "draft_dense_gemm",
                flops=2 * M * p,
                weight_read=int(p * wb * draft.weight_read_scale),
                weight_capacity=int(p * wb),
                act_read=int(M * spec.hidden * ab), act_write=int(M * spec.hidden * ab),
                formula_id=F.F_GEMM_FLOPS,
                shape_note=f"dense 草稿聚合口径 params={p}（×{draft.weight_read_scale} 读取倍率）")
    if draft.kv_frac > 0:
        led.add(-1, "draft_kv_scan", kv_read=int(target_kv * draft.kv_frac),
                formula_id=F.F_KV_RW,
                shape_note=f"草稿 KV 读 = 目标单步 KV 读 × {draft.kv_frac}（旧仓 kvFrac 口径）")
    return led


def _reference_kv_read(spec: ModelSpec, batch: int, ctx: int, tp: int) -> int:
    """目标模型 decode 单步（Q=1）的每卡 KV 读字节，供 kv_frac 折算。"""
    led = Ledger("_ref")
    family = spec.kv_spec["family"]
    if family == "gqa":
        build_gqa_model(led, spec, batch, ctx, 1, tp, prefill=False)
    else:
        build_v41_model(led, spec, batch, ctx, 1, tp, prefill=False)
    return sum(r.kv_read for r in led.rows)
