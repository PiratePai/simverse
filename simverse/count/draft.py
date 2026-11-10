"""草稿前向计数（M2）：

  * DFlash v1 块前向：fc + 6 层草稿（GQA/SLIDING）+ 目标 lm_head + 噪声查表
    （z-lab/dflash pin 语义；块行写入后 crop 的浪费单列）；
  * 链式草稿：draft ModelIR（tiny/自定义）逐 token 前向 × M；
  * MTP：按目标层模板 × mtp_layers（approximated，结构按 config 声明）。
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .. import formula_registry as FR
from ..draft_quant import (MXFP8_CONTEXT_PROJ, PER_LAYER_FALLBACK,
                           path_note, require_ctx_kv_path)
from ..model_ir.schema import DraftIR, ModelIR
from .work import (OpRow, _norm_row, _rope_row, build_model_work, causal_pairs,
                   totals)


def dflash_round_rows(stage: str, draft: DraftIR, target_vocab: int,
                      e_produced: float, start_ctx: int, quant: dict,
                      target_quant_act: float, tp: int = 1,
                      batch: int = 1, ctx_kv_path: str = "unknown") -> List[OpRow]:
    """DFlash 每轮草稿前向账本（**per-rank、整批口径**）。

    e_produced：**每序列**上轮提交 token 数期望；batch=B 时所有 token 维
    （fc/hidden_norm/注意 KV/lm_head/noise）乘 B 一次。权重/容量与 B 无关。
    start_ctx：已提交前缀长度（草稿 KV 可见行数上限用 SWA 封顶）；
    tp：草稿与 target 同 TP 组运行（vLLM load_dflash_model pin，见
    lowering/parallel.py）；q/o 投影与 MLP 按 tp 切分，k/v 按 KV 头规则
    切分或复制，lm_head 读 target 的 V/tp 分片（logits 由集合通信聚合）。
    ctx_kv_path：ctx 行 K/V 的**实现路径**（draft_quant.py 枚举）；必须是
    显式配方，dense_w 仅是计价位宽，不据此选分支。
    """
    path = require_ctx_kv_path(ctx_kv_path)
    B = batch
    produced = B * e_produced
    h = draft.hidden
    blk = draft.block_size
    b_w = quant["dense_w"] / 8
    b_a = quant["act"] / 8
    b_kv = quant["kv"] / 8
    rows: List[OpRow] = []
    # 1) fc 投影：E[produced]×(8h→h) + hidden_norm；fc 为 ReplicatedLinear
    #    （每 rank 全宽复制，不随 tp 缩放）
    rows.append(OpRow(
        stage, -1, "dflash_fc", flops=int(2 * produced * draft.fc_in * h),
        weight_read=int(draft.fc_in * h * b_w),
        act_read=int(produced * draft.fc_in * b_a),
        act_write=int(produced * h * b_a),
        weight_capacity=int(draft.fc_in * h * b_w), formula=FR.F_GEMM,
        statistic="expectation",
        shape=f"B={B}·E[produced]={e_produced:.2f}→{produced:.2f}×"
              f"[{draft.fc_in}→{h}] fc（replicated，不随 TP{tp} 缩放）"))
    # hidden_norm（fc 输出归一，pinned dflash model.py L584；2026-11-10 补齐）
    ops_h, w_h, ar_h, aw_h = _norm_row(int(max(1.0, produced)), h, b_a)
    rows.append(OpRow(stage, -1, "dflash_hidden_norm", flops=int(ops_h),
                      weight_read=int(w_h), act_read=int(ar_h),
                      act_write=int(aw_h), statistic="expectation",
                      formula=FR.F_RMSNORM, shape=f"width={h}（fc 后归一）"))
    # 2) 草稿 6 层：q_len=blk，KV 可见 = start_ctx+blk（SWA 层窗口封顶）
    for layer in draft.layers:
        li = layer.index
        a = layer.attn
        heads_c = a.heads // tp
        # KV 头 per-rank（kvh≥tp 切分；kvh<tp 复制）
        kvh = a.kv_heads // tp if a.kv_heads >= tp else a.kv_heads
        d = a.head_dim
        T = B * blk
        rows.append(OpRow(stage, li, "q_proj",
                          flops=2 * T * h * heads_c * d,
                          weight_read=int(h * heads_c * d * b_w),
                          act_read=int(T * h * b_a),
                          act_write=int(T * heads_c * d * b_a),
                          weight_capacity=int(h * heads_c * d * b_w),
                          formula=FR.F_GEMM,
                          shape=f"blk={blk} heads(卡上)={heads_c}"))
        for name in ("k_proj", "v_proj"):
            rows.append(OpRow(stage, li, name, flops=2 * T * h * kvh * d,
                              weight_read=int(h * kvh * d * b_w),
                              act_read=int(T * h * b_a),
                              act_write=int(T * kvh * d * b_a),
                              weight_capacity=int(h * kvh * d * b_w),
                              formula=FR.F_GEMM,
                              shape=f"kv(卡上)={kvh}"
                                    f"{'（复制）' if a.kv_heads < tp else ''}"))
        win = a.sliding_window
        end = start_ctx + blk
        if win:
            pairs = B * (causal_pairs(end, 1, win)
                         - causal_pairs(start_ctx, 1, win))
            read_rows = B * min(end, win)
        else:
            pairs = B * (start_ctx * blk + blk * (blk + 1) // 2)
            read_rows = B * start_ctx
        kv_elems = 2 * kvh * d                # per-rank KV 元素
        rows.append(OpRow(
            stage, li, "draft_attn_core",
            flops=pairs * heads_c * (4 * d + 5),
            act_read=int(T * heads_c * d * b_a),
            act_write=int(T * heads_c * d * b_a),
            kv_read=int(read_rows * kv_elems * b_kv),
            kv_write=int(T * kv_elems * b_kv),
            formula=FR.F_ATTN_CORE if not win else FR.F_SWA_PAIRS,
            shape=f"B={B} pairs={pairs} rows={read_rows} win={win} "
                  f"heads(卡上)={heads_c}"))
        # ctx K/V（target_hidden 投影，每轮 produced 行；与草稿块 token 的
        # qkv 投影是**两次独立 GEMM**）。vLLM pinned 三分支（qwen3_dflash.py
        # L679-697）的执行口径不同，必须由显式 ctx_kv_path 选择：
        #   fused_raw      → 全层 K/V 行融合副本，一次 GEMM 只读 K/V 行；
        #   mxfp8_context  → 独立 ReplicatedLinear 只装 K/V 行（量化打包）；
        #   per_layer_fallback → 逐层执行**完整** qkv_proj（含 Q）再切 Q，
        #                        故每层 FLOPs/权重读都含 Q 行。
        # 2026-11-10 二轮审计补漏：原 weight_read=0；本轮修正回退路径只计
        # K/V 的错误（dense_w 不能决定分支）。
        q_size = heads_c * d
        kv_size = kvh * d
        if path == PER_LAYER_FALLBACK:
            proj_out = q_size + 2 * kv_size
            ctx_flops = int(2 * produced * h * proj_out)
            ctx_weight = int(proj_out * h * b_w)
            ctx_note = (f"逐层完整 qkv_proj 输出 {proj_out}=Q{q_size}+2×KV"
                        f"{kv_size} 后切 Q；权重读含 Q 行")
        elif path == MXFP8_CONTEXT_PROJ:
            proj_out = 2 * kv_size
            ctx_flops = int(2 * produced * h * proj_out)
            # 独立 MXFP8 投影只执行 K/V 行；打包权重按 1B/elem 下界计，
            # block-scale 字节无 pin 证据 → 由 memory/能力层传播 unknown。
            ctx_weight = int(proj_out * h * 1.0)
            ctx_note = ("MXFP8 独立 _context_kv_proj 仅 K/V 行；"
                        "权重读=packed 8bit 下界（block-scale 未入 pin）")
        else:  # FUSED_RAW
            proj_out = 2 * kv_size
            ctx_flops = int(2 * produced * h * proj_out)
            ctx_weight = int(proj_out * h * b_w)
            ctx_note = "全层 _fused_kv_weight 融合 GEMM，仅 K/V 行"
        rows.append(OpRow(stage, li, "ctx_kv_proj",
                          flops=ctx_flops,
                          weight_read=ctx_weight,
                          statistic="expectation",
                          act_read=int(produced * h * b_a),
                          kv_write=int(produced * kv_elems * b_kv),
                          formula=FR.F_GEMM,
                          shape=f"B={B}×produced={produced:.2f} 行 "
                                f"path={path}（{path_note(path)}）；{ctx_note}"))
        rows.append(OpRow(stage, li, "o_proj", flops=2 * T * (heads_c * d) * h,
                          weight_read=int(heads_c * d * h * b_w),
                          act_read=int(T * heads_c * d * b_a),
                          act_write=int(T * h * b_a),
                          weight_capacity=int(heads_c * d * h * b_w),
                          formula=FR.F_GEMM))
        inter = (layer.dense_inter or 0) // tp  # dense MLP 列切
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
        # q_norm/k_norm（head_dim 级，pinned dflash model.py Qwen3DFlashAttention
        # L361-362；vLLM DFlashQwen3Attention 同构；2026-11-10 补齐缺行）
        for name, hd_r in (("q_norm", heads_c), ("k_norm", kvh)):
            ops, w, ar, aw = _norm_row(T, d, b_a)
            rows.append(OpRow(stage, li, name, flops=ops * hd_r, weight_read=w,
                              act_read=ar * hd_r, act_write=aw * hd_r,
                              formula=FR.F_RMSNORM,
                              shape=f"{hd_r} 头(卡上) × d={d}"))
        # rope（pinned dflash model.py L394-397 rotary；2026-11-10 补齐缺行）
        ops_r, _w, ar_r, aw_r = _rope_row(T, heads_c, d, d, b_a)
        rows.append(OpRow(stage, li, "rope", flops=ops_r, act_read=ar_r,
                          act_write=aw_r, formula=FR.F_ROPE,
                          shape=f"rotary={d} heads(卡上)={heads_c}"))
        for name in ("attn_norm", "mlp_norm"):
            rows.append(OpRow(stage, li, name, flops=T * (4 * h + 2),
                              weight_read=int(2 * h * b_a),
                              act_read=int(T * h * b_a),
                              act_write=int(T * h * b_a), formula=FR.F_RMSNORM))
        for name in ("attn_add", "mlp_add"):
            rows.append(OpRow(stage, li, name, flops=T * h,
                              act_read=int(2 * T * h * b_a),
                              act_write=int(T * h * b_a), formula=FR.F_RESIDUAL))
    # 2.5) 草稿 final norm（DFlashDraftModel.norm，pinned L565；补齐）
    ops, w, ar, aw = _norm_row(T, h, b_a)
    rows.append(OpRow(stage, -1, "draft_final_norm", flops=ops, weight_read=w,
                      act_read=ar, act_write=aw, formula=FR.F_RMSNORM,
                      shape=f"B={B}×blk={blk} 行，width={h}（draft final，replicated）"))
    # 3) 目标 lm_head 读出 blk−1 个候选：草稿与 target 共享同一词表行切
    #    lm_head（V/tp）。容量归 target 物理对象（账本去重，见 memory），
    #    但**读流量按每次 GEMM 计**：草稿每轮对同一分片独立前向读一遍
    #    （dflash model.py L260 compute_logits → L599-601 output_head(hidden)；
    #    vLLM qwen3_dflash.py L920-935 走同一 LogitsProcessor）。
    #    2026-11-10 二轮审计补漏：原 weight_read=0 低估草稿侧读流量。
    #    logits 词表聚合通信（TP>1）在 lowering/sharding.logits_gather 计。
    cand = B * (blk - 1)
    rows.append(OpRow(
        stage, -1, "draft_lm_head",
        flops=2 * cand * h * (target_vocab // tp),
        weight_read=int(h * (target_vocab // tp) * b_w),
        act_read=int(cand * h * b_a),
        act_write=int(cand * (target_vocab // tp) * b_a),
        formula=FR.F_LM_HEAD,
        shape=f"B={B}×(blk−1)={cand}×{h}×{target_vocab}/tp{tp}"
              f"（目标 lm_head 分片共享，读流量入草稿；容量归目标对象；"
              f"logits 聚合通信在 lowering 计）"))
    # 4) 噪声/首位 embedding 查表（用目标 embedding 分片对象）
    rows.append(OpRow(stage, -1, "noise_embed", act_read=int(T * h * b_a),
                      act_write=int(T * h * b_a), formula=FR.F_EMBED,
                      shape=f"B={B}×blk={blk} 行查表（目标 embedding 共享，词表行切）"))
    # 5) KV crop 浪费：块行写入 blk−1 候选中 (blk−1−(E[N]−1)) 行被回退
    wasted = max(0.0, blk - e_produced)
    rows.append(OpRow(stage, -1, "kv_rewind_waste", kv_write=0,
                      statistic="expectation",
                      formula=FR.F_KV_RW,
                      shape=f"目标+草稿 KV 回退浪费期望 {wasted:.2f} 行/轮（E[N]={e_produced:.2f}）"))
    return rows


def chain_step_rows(stage: str, draft_model: ModelIR, tokens: int, ctx: int,
                    quant: dict, tp: int = 1, batch: int = 1) -> List[OpRow]:
    """链式草稿一步（每序列 M token × B 批）账本：复用目标家族计数器。"""
    return build_model_work(draft_model, stage, batch, ctx, tokens, quant, tp,
                            role="draft")


def mtp_step_rows(stage: str, target: ModelIR, tokens: int, ctx: int,
                  quant: dict, mtp_layers: int, batch: int = 1) -> List[OpRow]:
    """MTP 微头一步：按目标层模板 × mtp_layers（approximated：结构按 config）。

    共享主干 embedding/lm_head（读流量在此计，容量与主干去重）。
    batch>1 时 template 的全部 token 维乘 B。微头模板固定 TP1；原生 TP
    配方仍缺证据，method recipe 保持 unverified，不据此进入排名。
    """
    rows: List[OpRow] = []
    h = target.hidden
    b_a = quant["act"] / 8
    template = target.layers[0]
    for j in range(mtp_layers):
        sub = build_model_work(
            _single_layer_model(target, template), stage, batch, ctx, tokens,
            quant, tp=1, role="draft")
        for r in sub:
            r.layer = j
            rows.append(r)
    total_tokens = batch * tokens
    rows.append(OpRow(stage, -1, "mtp_embed_share",
                      act_read=int(total_tokens * h * b_a),
                      act_write=int(total_tokens * h * b_a), formula=FR.F_EMBED,
                      shape=f"B={batch}×tokens={tokens}；共享主干 embedding/"
                            "lm_head（容量去重，读流量计；微头 TP 配方未验证）"))
    return rows


def _single_layer_model(target: ModelIR, layer) -> ModelIR:
    from ..model_ir.schema import SourceIdentity
    return ModelIR(id=f"{target.id}-layer{layer.index}", label="mtp-template",
                   family=target.family, hidden=target.hidden,
                   vocab=1,  # lm_head/embed 由共享行单独计
                   layers=[layer], source=SourceIdentity(
                       "derived", "layer-template", "", note="MTP 模板层"))
