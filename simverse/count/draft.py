"""草稿前向计数（M2）：

  * DFlash v1 块前向：fc + 6 层草稿（GQA/SLIDING）+ 目标 lm_head + 噪声查表
    （z-lab/dflash pin 语义；块行写入后 crop 的浪费单列）；
  * 链式草稿：draft ModelIR（tiny/自定义）逐 token 前向 × M；
  * MTP：原生微头单步（R23，2026-11-13）——共享 embedding 查表 + 双预归一 +
    fc(2h→h) combine + full_attention 层（gated GQA+MoE）+ final norm +
    共享 lm_head；层选择 step % n_layers（pinned qwen3_5_mtp.py @10a854ed）。
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .. import formula_registry as FR
from ..draft_quant import (MXFP8_CONTEXT_PROJ, PER_LAYER_FALLBACK,
                           path_note, require_ctx_kv_path)
from ..model_ir.schema import DraftIR, FULL, LayerIR, ModelIR
from .work import (OpRow, _norm_row, _rope_row, build_model_work, causal_pairs,
                   ffn_or_moe_rows, gqa_layer_rows, norm_add_rows, totals)


def dflash_round_rows(stage: str, draft: DraftIR, target_vocab: int,
                      e_produced: float, start_ctx: int, quant: dict,
                      target_quant_act: float, tp: int = 1,
                      batch: int = 1, ctx_kv_path: str = "unknown",
                      block_size: Optional[int] = None) -> List[OpRow]:
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
    # 服务栈执行口径（R07a，2026-11-12）：fc/hidden_norm/ctx_kv_proj 每轮对
    # **整个 verify 跨度 B×blk 行**执行（vLLM speculator propose→
    # _precompute_context_kv(0, num_target_tokens) 覆盖全跨度、graph 路径
    # _num_graph_context_tokens=num_reqs×(steps+1)；SGLang v2 worker 以全跨度
    # target_hidden 调 project_target_hidden 后按 commit_lens 前缀写），
    # 只有**缓存写入**按 committed 行（kv_write=produced）。HF 参考实现按
    # [:produced] 前缀重投影是较少工作的口径——两服务栈均按跨度执行，
    # 本行按跨度计（此前按 produced 计属漏计，R07a 修正）。
    h = draft.hidden
    # block_size 显式传入（R07a 修正：场景 k 截断/覆盖时草稿前向按生效块计，
    # 不得用 checkpoint 原生 block_size——vLLM num_speculative_tokens 与
    # SGLang --speculative-dflash-block-size 均按生效块执行草稿前向）
    blk = int(block_size) if block_size else draft.block_size
    span = B * blk
    b_w = quant["dense_w"] / 8
    b_a = quant["act"] / 8
    b_kv = quant["kv"] / 8
    rows: List[OpRow] = []
    # 1) fc 投影：span×(8h→h) + hidden_norm；fc 为 ReplicatedLinear
    #    （每 rank 全宽复制，不随 tp 缩放）
    rows.append(OpRow(
        stage, -1, "dflash_fc", flops=int(2 * span * draft.fc_in * h),
        weight_read=int(draft.fc_in * h * b_w),
        act_read=int(span * draft.fc_in * b_a),
        act_write=int(span * h * b_a),
        weight_capacity=int(draft.fc_in * h * b_w), formula=FR.F_GEMM,
        statistic="expectation",
        shape=f"B={B}×blk={blk}（verify 跨度）×[{draft.fc_in}→{h}] fc"
              f"（replicated，不随 TP{tp} 缩放）"))
    # hidden_norm（fc 输出归一，pinned dflash model.py L584；2026-11-10 补齐；
    # R07a 改按 verify 跨度行执行）
    ops_h, w_h, ar_h, aw_h = _norm_row(span, h, b_a)
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
        # 层内因果性来自 pinned 实现（HF dflash/model.py L364-366；vLLM
        # qwen3_dflash.py L86-97；SGLang models/dflash.py L239-249）：
        # Qwen3.6 DFlash 的 5 个 sliding 层因果、1 个 full 层非因果。
        # 非因果 full 层的块内可见对数是 Q²（不是 Q(Q+1)/2）；非因果
        # sliding 层在窗口内还可见未来 Q 行。R07a 复核发现并修正。
        causal = bool(getattr(a, "causal", True))
        if win:
            causal_pairs_layer = (causal_pairs(end, 1, win)
                                  - causal_pairs(start_ctx, 1, win))
            if causal:
                pairs = B * causal_pairs_layer
            else:
                # 窗口内未来块键：query r（0..Q-1）可见 r+1..min(Q-1,r+W-1)
                future = sum(min(blk - 1 - r, win - 1) for r in range(blk))
                pairs = B * (causal_pairs_layer + future)
            read_rows = B * min(end, win)
        else:
            if causal:
                pairs = B * (start_ctx * blk + blk * (blk + 1) // 2)
            else:
                pairs = B * (start_ctx * blk + blk * blk)
            read_rows = B * start_ctx
        kv_elems = 2 * kvh * d                # per-rank KV 元素
        rows.append(OpRow(
            stage, li, "draft_attn_core",
            flops=pairs * heads_c * (4 * d + 5),
            act_read=int(T * heads_c * d * b_a),
            act_write=int(T * heads_c * d * b_a),
            kv_read=int(read_rows * kv_elems * b_kv),
            kv_write=int(T * kv_elems * b_kv),
            formula=FR.F_ATTN_CORE if (not win or not causal)
            else FR.F_SWA_PAIRS,
            shape=f"B={B} pairs={pairs} rows={read_rows} win={win} "
                  f"causal={causal} heads(卡上)={heads_c}"))
        # ctx K/V（target_hidden 投影；与草稿块 token 的 qkv 投影是**两次
        # 独立 GEMM**）。R07a 口径（2026-11-12）：计算/读按每轮 **verify
        # 跨度 B×blk 行**（vLLM _precompute_context_kv 对全跨度投影后被拒
        # 行 PAD 不落缓存；SGLang project_target_hidden/kv_proj_only 全跨度
        # 投影 + commit_lens 前缀写），**缓存写入按 committed 行 produced**
        # （两栈一致：被拒行不写 draft KV）。vLLM pinned 三分支
        # （qwen3_dflash.py L679-697）的执行口径不同，必须由显式
        # ctx_kv_path 选择：
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
            ctx_flops = int(2 * span * h * proj_out)
            ctx_weight = int(proj_out * h * b_w)
            ctx_note = (f"逐层完整 qkv_proj 输出 {proj_out}=Q{q_size}+2×KV"
                        f"{kv_size} 后切 Q；权重读含 Q 行")
        elif path == MXFP8_CONTEXT_PROJ:
            proj_out = 2 * kv_size
            ctx_flops = int(2 * span * h * proj_out)
            # 独立 MXFP8 投影只执行 K/V 行；打包权重按 1B/elem 下界计，
            # block-scale 字节无 pin 证据 → 由 memory/能力层传播 unknown。
            ctx_weight = int(proj_out * h * 1.0)
            ctx_note = ("MXFP8 独立 _context_kv_proj 仅 K/V 行；"
                        "权重读=packed 8bit 下界（block-scale 未入 pin）")
        else:  # FUSED_RAW
            proj_out = 2 * kv_size
            ctx_flops = int(2 * span * h * proj_out)
            ctx_weight = int(proj_out * h * b_w)
            ctx_note = "全层 _fused_kv_weight 融合 GEMM，仅 K/V 行"
        rows.append(OpRow(stage, li, "ctx_kv_proj",
                          flops=ctx_flops,
                          weight_read=ctx_weight,
                          statistic="expectation",
                          act_read=int(span * h * b_a),
                          act_write=int(span * proj_out * b_a),
                          kv_write=int(produced * kv_elems * b_kv),
                          formula=FR.F_GEMM,
                          shape=f"span=B={B}×blk={blk} 行投影/committed="
                                f"{produced:.2f} 行写 "
                                f"path={path}（{path_note(path)}）；{ctx_note}"))
        # ctx K 逐头 RMSNorm + RoPE：两栈都在全跨度投影后执行
        # （vLLM _normalize_context_k/_apply_context_rope；SGLang
        # apply_k_norm/apply_k_rope）。原账本只计块内 q/k norm 与 Q rope，
        # 漏了 ctx K；R07a 复核补入，按 span 行计。
        ops_ck, w_ck, ar_ck, aw_ck = _norm_row(span * kvh, d, b_a)
        rows.append(OpRow(stage, li, "ctx_k_norm", flops=int(ops_ck),
                          weight_read=int(w_ck), act_read=int(ar_ck),
                          act_write=int(aw_ck), formula=FR.F_RMSNORM,
                          shape=f"ctx K RMSNorm span={span}×kvh={kvh}×d={d}"))
        ops_cr, _w_cr, ar_cr, aw_cr = _rope_row(span, kvh, d, d, b_a)
        rows.append(OpRow(stage, li, "ctx_k_rope", flops=ops_cr,
                          act_read=ar_cr, act_write=aw_cr, formula=FR.F_ROPE,
                          shape=f"ctx K RoPE span={span}×kvh={kvh}×d={d}"))
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
    # 2.4) ctx K/V 布局拷贝：vLLM _project_context_kv 在逐层
    # [span,L,2,kv,d] 投影后做 permute(2,1,0,3,4).contiguous()（读+写全
    # span×L×2 元素）。SGLang 逐层 store_prefix 无此全局拷贝——本仓行按
    # vLLM implementation_id 计，差异登记在 source_map。
    if draft.layers:
        a0 = draft.layers[0].attn
        kvh0 = a0.kv_heads // tp if a0.kv_heads >= tp else a0.kv_heads
        elems = span * len(draft.layers) * 2 * kvh0 * a0.head_dim
        rows.append(OpRow(stage, -1, "ctx_kv_layout_copy",
                          act_read=int(elems * b_a), act_write=int(elems * b_a),
                          formula=FR.F_KV_LAYOUT,
                          shape=f"vLLM permute+contiguous span={span}×L="
                                f"{len(draft.layers)}×2×kv={kvh0}×d="
                                f"{a0.head_dim} 读+写（SGLang 无全局拷贝）"))
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


def mtp_step_rows(stage: str, target: ModelIR, step_idx: int, ctx: int,
                  quant: dict, tp: int = 1, ep: int = 1, batch: int = 1,
                  plan=None, rank: int = 0,
                  shared_quant: Optional[dict] = None) -> List[OpRow]:
    """MTP 微头**单步**账本（R23，2026-11-13；per-rank、整批口径）。

    结构按 pinned vLLM Qwen3_5MoeMTP（vllm_qwen36_mtp_pin @10a854ed，
    qwen3_5_mtp.py；source-checked，未执行）：
      1. 共享目标 embedding 查表（读流量；容量归目标对象）；
      2. pre_fc_norm_embedding / pre_fc_norm_hidden 两个 RMSNorm 分别归一
         （L137-142/L159-160）；
      3. cat → fc(2h→h, bias=False, ColumnParallel gather_output=True；
         L99-104/L161-162；TP>1 的 all-gather 在通信行）——无激活、无残差
         直加（区别于 DeepSeek V3 式 comb）；
      4. 层 spec_step_idx % n_layers 的 **full_attention** decoder 层
         （L120-127/L165-166；gated GQA + MoE，几何=目标 full 层）；
      5. final norm（L136/L175）；
      6. 共享目标 lm_head 读出 1 个候选/请求（L228-235 + compute_logits；
         容量归目标，读流量计每步一次）。

    step_idx：草稿步序号（0..K−1）；步内 Q=1 token/请求，draft 自有 KV
    可见行 = ctx + step_idx（本步之前已提交 + 本轮先前步）。K>n_predict 时
    同层循环复用（use_multi_module_mtp L1953-1967：n=1 单模块复用路径）。
    batch>1 时 token 维乘 B 一次（G13）；权重读/容量不随 B 变化。
    quant 是草稿私有层的精度档；共享 lm_head 的权重读取沿目标对象
    shared_quant（缺省与 quant 相同，供独立 oracle 使用）。
    """
    spec = (target.extras or {}).get("mtp")
    if not spec:
        raise ValueError(
            f"模型 {target.id} 无原生 MTP 结构 spec（extras['mtp']）："
            "mtp_step_rows 只服务 pinned qwen3_5_moe MTP1 头，不得套用模板")
    n_layers = int(spec["n_layers"])
    if not 0 <= step_idx:
        raise ValueError(f"step_idx 非负，得到 {step_idx}")
    h = target.hidden
    b_w = quant["dense_w"] / 8
    b_a = quant["act"] / 8
    shared_w = (shared_quant or quant)["dense_w"] / 8
    B = batch
    layer = _mtp_template_layer(target)
    li = step_idx % n_layers
    rows: List[OpRow] = []
    # 1) 共享 embedding 查表（B 行）
    rows.append(OpRow(stage, -1, "mtp_embed_share",
                      act_read=int(B * h * b_a), act_write=int(B * h * b_a),
                      formula=FR.F_EMBED,
                      shape=f"B={B} 行查表（目标 embedding 共享，词表行切；"
                            "容量归目标对象，读流量计）"))
    # 2) 双预归一（embedding/hidden 两支各自 RMSNorm）
    for name in ("mtp_pre_fc_norm_embedding", "mtp_pre_fc_norm_hidden"):
        ops, w, ar, aw = _norm_row(B, h, b_a)
        rows.append(OpRow(stage, -1, name, flops=ops, weight_read=w,
                          weight_capacity=w, act_read=ar, act_write=aw,
                          formula=FR.F_RMSNORM, shape=f"width={h}（replicated）"))
    # 3) fc combine：cat[B,2h] → [B,h/tp]（ColumnParallel gather_output=True）
    fc_out = h // tp
    rows.append(OpRow(
        stage, -1, "mtp_fc", flops=2 * B * (2 * h) * fc_out,
        weight_read=int(2 * h * fc_out * b_w),
        weight_capacity=int(2 * h * fc_out * b_w),
        act_read=int(B * 2 * h * b_a), act_write=int(B * fc_out * b_a),
        formula=FR.F_GEMM,
        shape=f"[{B},{2 * h}]@[{2 * h},{fc_out}]（列切 h/tp，gather_output="
              f"True → TP>1 all-gather 见通信行；无激活/无残差直加）"))
    # 4) full_attention 层（几何=目标 full 层；draft 自有 KV：S=ctx+step_idx）
    layer_rows = gqa_layer_rows(stage, li, layer, B, ctx + step_idx, 1, h,
                                quant, tp)
    rows += layer_rows
    T = B
    rows += ffn_or_moe_rows(stage, li, layer, T, h, quant, tp=tp, ep=ep,
                            local_experts=_mtp_local_experts(layer, plan, ep,
                                                             rank))
    rows += norm_add_rows(stage, li, T, h, quant)
    # 5) final norm（mtp.norm，replicated）
    ops, w, ar, aw = _norm_row(B, h, b_a)
    rows.append(OpRow(stage, -1, "mtp_final_norm", flops=ops, weight_read=w,
                      weight_capacity=w, act_read=ar, act_write=aw,
                      formula=FR.F_RMSNORM,
                      shape=f"width={h}（mtp.norm，replicated）"))
    # 6) 共享 lm_head：每步读出 B 个候选（读流量；容量归目标对象）
    rows.append(OpRow(
        stage, -1, "draft_lm_head",
        flops=2 * B * h * (target.vocab // tp),
        weight_read=int(h * (target.vocab // tp) * shared_w),
        act_read=int(B * h * b_a),
        act_write=int(B * (target.vocab // tp) * b_a),
        formula=FR.F_LM_HEAD,
        shape=f"B={B}×{h}×{target.vocab}/tp{tp}（目标 lm_head 分片共享，"
              "读流量入草稿；容量归目标；logits 聚合通信在 lowering 计）"))
    for r in rows:
        r.role = "draft"
    return rows


def mtp_kv_rewind_row(stage: str, k: int, e_produced: float, target: ModelIR,
                      quant: dict, tp: int) -> OpRow:
    """每轮草稿 KV 回退浪费期望（K 步行写入 vs E[N] 提交；DFlash 同式）。"""
    attn = _mtp_template_layer(target).attn
    kvh = attn.kv_heads // tp if attn.kv_heads >= tp else attn.kv_heads
    kv_elems = 2 * kvh * attn.head_dim
    wasted = max(0.0, k - e_produced)
    return OpRow(
        stage, -1, "mtp_kv_rewind_waste", kv_write=0, statistic="expectation",
        formula=FR.F_KV_RW,
        shape=f"草稿 KV 回退浪费期望 {wasted:.2f} 行/轮（K={k}，E[N]="
              f"{e_produced:.2f}；每行 {kv_elems} elem/层 × MTP1；"
              "被拒步不落 committed 缓存，下轮重写）")


def _mtp_template_layer(target: ModelIR) -> LayerIR:
    """MTP 层几何模板：目标 full_attention 层（qwen3_5_mtp.py L120-127
    layer_type="full_attention"；非目标首层——Qwen3.6 首层是 GDN linear）。"""
    for layer in target.layers:
        if layer.kind == FULL and layer.attn is not None:
            return layer
    raise ValueError(
        f"模型 {target.id} 无 full_attention 层可作 MTP 模板（pinned MTP 头"
        "固定 full_attention 层型，无对应层即结构失配）")


def _mtp_local_experts(layer: LayerIR, plan, ep: int, rank: int):
    """MTP 层 MoE 的本 rank 专家数（与目标同 ParallelPlan 配方；EP 开时
    余数 rank +1——vLLM 草稿同 TP/EP 组，QwenNextMixtureOfExperts 同构）。"""
    if plan is not None:
        return plan.moe_local_experts(layer, rank) or None
    if layer.moe is None or not layer.moe.experts:
        return None
    if ep == 1:
        return layer.moe.experts
    base, rem = divmod(layer.moe.experts, ep)
    return base + (1 if rank < rem else 0)


def mtp_layer_model(target: ModelIR) -> ModelIR:
    """MTP 头的单层伪模型（count 之外的消费者：memory/collectives 共用）。

    vocab/tie 沿目标（embed/lm_head 由 colocated 共享引用替身，容量归
    目标对象）；layers=[full 模板层]，index 归 0。
    """
    from ..model_ir.schema import SourceIdentity
    layer = _mtp_template_layer(target)
    return ModelIR(
        id=f"{target.id}-mtp1", label=f"{target.label} native MTP head",
        family=target.family, hidden=target.hidden, vocab=target.vocab,
        layers=[LayerIR(0, FULL, attn=layer.attn, moe=layer.moe,
                        dense_inter=layer.dense_inter)],
        source=SourceIdentity(
            "derived", "vllm@10a854ed6b3bbd7411c65ccbf87594e71c4d322c",
            "", note="qwen3_5_moe 原生 MTP 单层（pinned qwen3_5_mtp.py）"),
        tie_embeddings=False,
        coverage={"*": {"status": "mapped", "consumer": "model_ir",
                        "note": "MTP 单层伪模型（结构派生自目标 full 层）"}},
        mtp_layers=int((target.extras or {}).get("mtp", {}).get("n_layers", 0)),
        extras={"mtp": dict((target.extras or {}).get("mtp", {}))})
