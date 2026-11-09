"""草稿前向计数（M2）：

  * DFlash v1 块前向：fc + 6 层草稿（GQA/SLIDING）+ 目标 lm_head + 噪声查表
    （z-lab/dflash pin 语义；块行写入后 crop 的浪费单列）；
  * 链式草稿：draft ModelIR（tiny/自定义）逐 token 前向 × M；
  * MTP：按目标层模板 × mtp_layers（approximated，结构按 config 声明）。
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .. import formula_registry as FR
from ..model_ir.schema import DraftIR, ModelIR
from .work import OpRow, build_model_work, causal_pairs, totals


def dflash_round_rows(stage: str, draft: DraftIR, target_vocab: int,
                      e_produced: float, start_ctx: int, quant: dict,
                      target_quant_act: float) -> List[OpRow]:
    """DFlash 每轮草稿前向账本。

    e_produced：上轮提交 token 数期望（fc/lm_head 的 token 维权重）；
    start_ctx：已提交前缀长度（草稿 KV 可见行数上限用 SWA 封顶）。
    """
    h = draft.hidden
    blk = draft.block_size
    b_w = quant["dense_w"] / 8
    b_a = quant["act"] / 8
    b_kv = quant["kv"] / 8
    rows: List[OpRow] = []
    # 1) fc 投影：E[produced]×(8h→h) + hidden_norm（target_hidden 拼接输入）
    rows.append(OpRow(
        stage, -1, "dflash_fc", flops=int(2 * e_produced * draft.fc_in * h),
        weight_read=int(draft.fc_in * h * b_w),
        act_read=int(e_produced * draft.fc_in * b_a),
        act_write=int(e_produced * h * b_a),
        weight_capacity=int(draft.fc_in * h * b_w), formula=FR.F_GEMM,
        statistic="expectation",
        shape=f"E[produced]={e_produced:.2f}×[{draft.fc_in}→{h}] fc"))
    # 2) 草稿 6 层：q_len=blk，KV 可见 = start_ctx+blk（SWA 层窗口封顶）
    for layer in draft.layers:
        li = layer.index
        a = layer.attn
        heads_c, kvh, d = a.heads, a.kv_heads, a.head_dim
        T = blk
        rows.append(OpRow(stage, li, "q_proj", flops=2 * T * h * heads_c * d,
                          weight_read=int(h * heads_c * d * b_w),
                          act_read=int(T * h * b_a),
                          act_write=int(T * heads_c * d * b_a),
                          weight_capacity=int(h * heads_c * d * b_w),
                          formula=FR.F_GEMM, shape=f"blk={blk}"))
        for name in ("k_proj", "v_proj"):
            rows.append(OpRow(stage, li, name, flops=2 * T * h * kvh * d,
                              weight_read=int(h * kvh * d * b_w),
                              act_read=int(T * h * b_a),
                              act_write=int(T * kvh * d * b_a),
                              weight_capacity=int(h * kvh * d * b_w),
                              formula=FR.F_GEMM))
        win = a.sliding_window
        end = start_ctx + blk
        if win:
            pairs = causal_pairs(end, 1, win) - causal_pairs(start_ctx, 1, win)
            read_rows = min(end, win)
        else:
            pairs = start_ctx * blk + blk * (blk + 1) // 2
            read_rows = start_ctx
        kv_elems = 2 * kvh * d
        rows.append(OpRow(
            stage, li, "draft_attn_core",
            flops=pairs * heads_c * (4 * d + 5),
            act_read=int(T * heads_c * d * b_a),
            act_write=int(T * heads_c * d * b_a),
            kv_read=int(read_rows * kv_elems * b_kv),
            kv_write=int(T * kv_elems * b_kv),
            formula=FR.F_ATTN_CORE if not win else FR.F_SWA_PAIRS,
            shape=f"pairs={pairs} rows={read_rows} win={win}"))
        # ctx K/V（target_hidden 投影，每轮 produced 行）
        rows.append(OpRow(stage, li, "ctx_kv_proj",
                          flops=int(2 * e_produced * h * 2 * kvh * d),
                          weight_read=0, statistic="expectation",
                          act_read=int(e_produced * h * b_a),
                          kv_write=int(e_produced * kv_elems * b_kv),
                          formula=FR.F_GEMM,
                          shape="k/v_ctx = k/v_proj(target_hidden) 每轮重取 produced 行"))
        rows.append(OpRow(stage, li, "o_proj", flops=2 * T * (heads_c * d) * h,
                          weight_read=int(heads_c * d * h * b_w),
                          act_read=int(T * heads_c * d * b_a),
                          act_write=int(T * h * b_a),
                          weight_capacity=int(heads_c * d * h * b_w),
                          formula=FR.F_GEMM))
        inter = layer.dense_inter or 0
        rows.append(OpRow(stage, li, "gate_up_proj",
                          flops=2 * T * h * 2 * inter,
                          weight_read=int(h * 2 * inter * b_w),
                          act_read=int(T * h * b_a),
                          act_write=int(T * 2 * inter * b_a),
                          weight_capacity=int(h * 2 * inter * b_w), formula=FR.F_GEMM))
        rows.append(OpRow(stage, li, "down_proj", flops=2 * T * inter * h,
                          weight_read=int(inter * h * b_w),
                          act_read=int(T * inter * b_a),
                          act_write=int(T * h * b_a),
                          weight_capacity=int(inter * h * b_w), formula=FR.F_GEMM))
        for name in ("attn_norm", "mlp_norm"):
            rows.append(OpRow(stage, li, name, flops=T * (4 * h + 2),
                              weight_read=int(2 * h * b_a),
                              act_read=int(T * h * b_a),
                              act_write=int(T * h * b_a), formula=FR.F_RMSNORM))
        for name in ("attn_add", "mlp_add"):
            rows.append(OpRow(stage, li, name, flops=T * h,
                              act_read=int(2 * T * h * b_a),
                              act_write=int(T * h * b_a), formula=FR.F_RESIDUAL))
    # 3) 目标 lm_head 读出 blk−1 个候选（uses_target_lm_head=True）
    rows.append(OpRow(
        stage, -1, "draft_lm_head", flops=2 * (blk - 1) * h * target_vocab,
        weight_read=0,                    # 权重属目标（同卡共享驻留；读流量另计）
        act_read=int((blk - 1) * h * b_a),
        act_write=int((blk - 1) * target_vocab * b_a),
        formula=FR.F_LM_HEAD,
        shape=f"(blk−1)×{h}×{target_vocab}（用目标 lm_head，权重读归目标侧）"))
    # 4) 噪声/首位 embedding 查表（用目标 embedding）
    rows.append(OpRow(stage, -1, "noise_embed", act_read=int(blk * h * b_a),
                      act_write=int(blk * h * b_a), formula=FR.F_EMBED,
                      shape=f"blk={blk} 行查表（目标 embedding 共享）"))
    # 5) KV crop 浪费：块行写入 blk−1 候选中 (blk−1−(E[N]−1)) 行被回退
    wasted = max(0.0, blk - e_produced)
    rows.append(OpRow(stage, -1, "kv_rewind_waste", kv_write=0,
                      statistic="expectation",
                      formula=FR.F_KV_RW,
                      shape=f"目标+草稿 KV 回退浪费期望 {wasted:.2f} 行/轮（E[N]={e_produced:.2f}）"))
    return rows


def chain_step_rows(stage: str, draft_model: ModelIR, tokens: int, ctx: int,
                    quant: dict, tp: int = 1) -> List[OpRow]:
    """链式草稿一步（M token）账本：直接复用目标家族计数器。"""
    return build_model_work(draft_model, stage, 1, ctx, tokens, quant, tp,
                            role="draft")


def mtp_step_rows(stage: str, target: ModelIR, tokens: int, ctx: int,
                  quant: dict, mtp_layers: int) -> List[OpRow]:
    """MTP 微头一步：按目标层模板 × mtp_layers（approximated：结构按 config）。

    共享主干 embedding/lm_head（读流量在此计，容量与主干去重）。
    """
    rows: List[OpRow] = []
    h = target.hidden
    b_a = quant["act"] / 8
    template = target.layers[0]
    for j in range(mtp_layers):
        sub = build_model_work(
            _single_layer_model(target, template), stage, 1, ctx, tokens,
            quant, tp=1, role="draft")
        for r in sub:
            r.layer = j
            rows.append(r)
    rows.append(OpRow(stage, -1, "mtp_embed_share", act_read=int(tokens * h * b_a),
                      act_write=int(tokens * h * b_a), formula=FR.F_EMBED,
                      shape="共享主干 embedding/lm_head（容量去重，读流量计）"))
    return rows


def _single_layer_model(target: ModelIR, layer) -> ModelIR:
    from ..model_ir.schema import SourceIdentity
    return ModelIR(id=f"{target.id}-layer{layer.index}", label="mtp-template",
                   family=target.family, hidden=target.hidden,
                   vocab=1,  # lm_head/embed 由共享行单独计
                   layers=[layer], source=SourceIdentity(
                       "derived", "layer-template", "", note="MTP 模板层"))
