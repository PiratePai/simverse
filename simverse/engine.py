"""L3 解析执行：run(scenario) → RunResult（Web / CLI / 导出唯一入口）。

组合方式（Execute 解析档，方案 §4.5）：
  baseline          TPOT = t_base
  colocated         TPOT = (Σ_j t_draft_j + t_verify) / E[n]        （草稿 100% 暴露）
  dt_serial         TPOT = (Σ_j t_draft_j + t_d2t + t_verify + t_t2d) / E[n]
  dt_pipeline_ideal TPOT = (max(Σ t_draft + t_d2t, t_verify) + t_t2d) / E[n]
                                              （跨微批无损交错稳态下界，乐观标注）

RunResult 同时携带全部部署的对比行（comparisons），scenario.deployment.mode
决定主视图/时间线取哪种；同一 RunResult 的数字全部来自同一 L1 计价。
"""
from __future__ import annotations

from typing import Dict, List

from . import acceptance as ACC
from . import formulas as F
from .cost import link_ms, roofline_ms
from .contracts import (DeviceSpec, RunResult, ScenarioSpec, StageTiming,
                        TimelineEvent, digest)
from .devices import get_device, get_link
from .ledger import build_draft_ledger, build_model_ledger
from .models import get_draft, get_model

# 算子 → 计价精度路由（vllm-cuda 隐式 recipe 的最小子集；FP4 专家在 H100 无 FP4 GEMM，
# 算力仍走 FP8 档 —— 旧仓 A6 口径）
_ELEMENTWISE = {"q_norm", "k_norm", "q_rope", "k_rope", "o_rope", "compressed_rope",
                "index_k_norm", "index_k_rope", "index_q_rope", "attn_norm", "mlp_norm",
                "attn_add", "mlp_add", "compress_norm", "swa_cache_write",
                "compressed_cache_write", "index_cache_write", "index_select",
                "candidate_publish", "embed_lookup", "attention_core",
                "sparse_attention", "draft_kv_scan"}


def _dtype_pair(quant: dict) -> tuple:
    """按模型精度选峰值档：gemm 看 dense_w 位宽（A6：H100 无 FP4 GEMM → ≤8bit 走 FP8 档），
    elementwise/attention 核看 act 位宽。"""
    gemm = "fp16" if quant["dense_w"] >= 16 else "fp8"
    ew = "fp16" if quant["act"] >= 16 else "fp8"
    return gemm, ew


def _price_ledger(led, device: DeviceSpec, gemm_dtype: str = "fp8",
                  ew_dtype: str = "fp8") -> float:
    """对账本逐行 T0 roofline 计价（每卡口径），返回阶段时延 ms。"""
    total = 0.0
    for r in led.rows:
        byte_count = r.weight_read + r.act_read + r.act_write + r.kv_read + r.kv_write
        dtype = ew_dtype if r.op in _ELEMENTWISE else gemm_dtype
        est = roofline_ms(r.flops, byte_count, device, dtype=dtype)
        r.t_comp_ms, r.t_mem_ms, r.t_ms, r.bound = (est["t_comp_ms"], est["t_mem_ms"],
                                                    est["t_ms"], est["bound"])
        total += r.t_ms
    return total


def _kv_retained_bytes(spec, ctx: int, batch: int) -> int:
    """KV 保留容量（驻留），与读写流量分列。V41 为 deepseek_capacity 移植。"""
    kb = spec.quant["kv"] / 8
    fam = spec.kv_spec["family"]
    if fam == "gqa":
        elems = spec.kv_spec["kv_elems_per_token_per_layer"]
        return int(len(spec.layers) * ctx * elems * kb) * batch
    # V41 CSA2：SWA(每层) + 压缩 KV(源层) + 索引 K(源层) + 压缩器 fp32 state
    d = spec.kv_spec["head_dim"]
    di = spec.kv_spec["index_head_dim"]
    swa = spec.kv_spec["sliding_window"]
    total = 0
    for layer in spec.layers:
        a = layer.attn
        total += min(ctx, swa) * d * kb          # 每层本地 SWA
        if a["produces_kv"]:
            r = a["ratio"]
            total += (ctx // r) * d * kb          # 压缩 KV 潜层
            total += (ctx // r) * di * kb         # 索引 K
            if r > 1:
                total += 2 * d * 4                # 压缩器 state（fp32，常数）
    return int(total) * batch


def _draft_step_costs(spec, draft, batch, ctx, alpha, k, device, tp,
                      suffix="") -> List[StageTiming]:
    """草稿 S=K 步的时延：step0 按 Catch-up 概率加权（整数账本 m1/m2 两条）。

    suffix 区分放置（""=同卡 TP 组，"@dt"=独立草稿设备 TP1）——账本阶段名
    带后缀，防两份放置的行在按 stage 求和时重复计入。
    """
    stages: List[StageTiming] = []
    p_full = ACC.catchup_prob_full(alpha, k)
    for j in range(k):
        if j == 0:
            led1 = build_draft_ledger(f"draft_step_0_m1{suffix}", spec, draft, batch * 1, ctx, tp)
            led2 = build_draft_ledger(f"draft_step_0_m2{suffix}", spec, draft, batch * 2, ctx, tp)
            t1 = _price_ledger(led1, device)
            t2 = _price_ledger(led2, device)
            t = (1 - p_full) * t1 + p_full * t2
            stages.append(StageTiming(
                stage="draft_step_0", label=f"草稿步 0（Catch-up 加权）", lane="draft",
                t_ms=t, formula_id=F.F_CATCHUP, tokens_in=1,
                detail={"t_m1_ms": t1, "t_m2_ms": t2, "p_full": p_full,
                        "E[M_draft,0]": 1 + p_full}))
            stages[-1]._ledgers = (led1, led2)  # type: ignore[attr-defined]
        else:
            led = build_draft_ledger(f"draft_step_{j}{suffix}", spec, draft, batch, ctx, tp)
            t = _price_ledger(led, device)
            stages.append(StageTiming(
                stage=f"draft_step_{j}", label=f"草稿步 {j}", lane="draft",
                t_ms=t, formula_id=F.F_GEMM_FLOPS, tokens_in=1))
            stages[-1]._ledger = led  # type: ignore[attr-defined]
    return stages


def _collect_ops(stage_list) -> List:
    """从 StageTiming 提取算子行（Catch-up 加权步展开为 m1/m2 两段整数账本）。"""
    rows = []
    for st in stage_list:
        if getattr(st, "_ledgers", None) is not None:
            led1, led2 = st._ledgers
            for r in led1.rows:
                rows.append(r)
            for r in led2.rows:
                rows.append(r)
        elif getattr(st, "_ledger", None) is not None:
            rows.extend(st._ledger.rows)
    return rows


def run(scenario: ScenarioSpec) -> RunResult:
    """一次计算：同一场景文件 → 同一 RunResult（可重放性的实现基础）。"""
    scenario.validate()
    model = get_model(scenario.model_id)
    try:
        draft = get_draft(scenario.draft_id)
    except ValueError as exc:
        if "unsupported_method" in str(exc):
            return RunResult(status="unsupported_method", status_detail=str(exc))
        raise
    if draft.method == "mtp" and scenario.k > 3:
        return RunResult(
            status="unsupported_combo",
            status_detail=f"MTP config 原生 {3} 层微头，K={scenario.k} 超出串行步数上限")

    target = get_device(scenario.target_device)
    draft_dev = get_device(scenario.draft_device)
    link = get_link(scenario.link)
    tp = scenario.tp
    B, ctx, K = scenario.batch, scenario.ctx_tokens, scenario.k
    acc = scenario.acceptance_spec()
    alpha = ACC.resolve_alpha(acc, K)
    en = ACC.expected_committed_length(acc, K)
    mode = scenario.mode()

    # ---- 显存容量检查（容量不足仍输出逻辑算量，结论标记 oom）----
    kv_ret = _kv_retained_bytes(model, ctx, B)
    led_ref = build_model_ledger(model, "baseline_decode", B, ctx, 1, tp)
    weight_cap = sum(r.weight_capacity for r in led_ref.rows)  # per card
    oom_target = (weight_cap + kv_ret) > target.mem_capacity
    draft_cap = (draft.moe and int(draft.moe["experts"] * 3 * draft.moe["hidden"]
                                   * draft.moe["inter"] * draft.weight_bits / 8)
                 or int(draft.params_dense * draft.weight_bits / 8))
    oom_draft = draft_cap > draft_dev.mem_capacity

    # ---- 各阶段账本与计价（每卡口径；Cost 全部 T0 公式估算）----
    gemm_dt, ew_dt = _dtype_pair(model.quant)
    base_led = build_model_ledger(model, "baseline_decode", B, ctx, 1, tp)
    t_base = _price_ledger(base_led, target, gemm_dt, ew_dt)
    prefill_led = build_model_ledger(model, "prefill", B, ctx, ctx, tp, prefill=True)
    t_prefill = _price_ledger(prefill_led, target, gemm_dt, ew_dt)
    verify_led = build_model_ledger(model, "verify", B, ctx, K + 1, tp)
    t_verify = _price_ledger(verify_led, target, gemm_dt, ew_dt)

    # 草稿两种放置：同卡（target TP 组）/ DT 分离（draft_dev TP1）
    draft_st_co = _draft_step_costs(model, draft, B, ctx, alpha, K, target, tp)
    draft_st_dt = _draft_step_costs(model, draft, B, ctx, alpha, K, draft_dev, 1,
                                    suffix="@dt")
    t_draft_co = sum(s.t_ms for s in draft_st_co)
    t_draft_dt = sum(s.t_ms for s in draft_st_dt)

    # DT 通信（F_COMM_BYTES；MTP 分离需回传 accepted hidden）
    bytes_d2t = B * (K + 1) * draft.d2t_bytes_per_token
    nbar = en
    bytes_t2d = B * (nbar * draft.t2d_bytes_per_token + draft.t2d_bytes_per_request)
    d2t = link_ms(link, bytes_d2t)
    t2d = link_ms(link, bytes_t2d)

    # ---- 部署对比（同一 RunResult 内并列；数字同源）----
    def _row(name: str, tpot: float, formula: str, extra: Dict | None = None) -> Dict:
        row = {"deployment": name, "tpot_ms": tpot, "speedup": t_base / tpot if tpot else 0.0,
               "tokens_per_round": en, "formula_id": formula,
               "source": "T0 公式估算"}
        if extra:
            row.update(extra)
        return row

    t_colocated = (t_draft_co + t_verify) / en
    t_dt_serial = (t_draft_dt + d2t["t_ms"] + t_verify + t2d["t_ms"]) / en
    t_dt_pipe = (max(t_draft_dt + d2t["t_ms"], t_verify) + t2d["t_ms"]) / en
    comparisons = [
        _row("baseline", t_base, F.F_TPOT_BASE, {"round_ms": t_base}),
        _row("colocated", t_colocated, F.F_TPOT_COLOCATED,
             {"round_ms": t_draft_co + t_verify,
              "detail": {"t_draft_ms": t_draft_co, "t_verify_ms": t_verify}}),
        _row("dt_serial", t_dt_serial, F.F_TPOT_DT_SERIAL,
             {"round_ms": t_draft_dt + d2t["t_ms"] + t_verify + t2d["t_ms"],
              "detail": {"t_draft_ms": t_draft_dt, "t_d2t_ms": d2t["t_ms"],
                         "t_verify_ms": t_verify, "t_t2d_ms": t2d["t_ms"],
                         "bytes_d2t": bytes_d2t, "bytes_t2d": bytes_t2d}}),
        _row("dt_pipeline_ideal", t_dt_pipe, F.F_TPOT_DT_PIPE_IDEAL,
             {"round_ms": max(t_draft_dt + d2t["t_ms"], t_verify) + t2d["t_ms"],
              "detail": {"note": "跨微批无损交错稳态下界（乐观，无命中率/回退建模）",
                         "t_draft_ms": t_draft_dt, "t_d2t_ms": d2t["t_ms"],
                         "t_verify_ms": t_verify, "t_t2d_ms": t2d["t_ms"]}}),
    ]
    if oom_draft:
        for row in comparisons:
            if row["deployment"].startswith("dt_"):
                row["status"] = "oom（草稿设备）"

    # 盈亏平衡 α*（speedup=1 反解；对 dt 部署含通信）。
    # 草稿时延随 α 变化仅经 Catch-up 步加权：t_draft(α) = Σ_{j>0} t_j + (1-α^K)·t_m1 + α^K·t_m2
    d0_co = draft_st_co[0].detail
    d0_dt = draft_st_dt[0].detail
    rest_co = sum(s.t_ms for s in draft_st_co[1:])
    rest_dt = sum(s.t_ms for s in draft_st_dt[1:])

    def _t_draft(a: float, which: str) -> float:
        p = ACC.catchup_prob_full(a, K)
        d0 = d0_co if which == "colocated" else d0_dt
        rest = rest_co if which == "colocated" else rest_dt
        return rest + (1 - p) * d0["t_m1_ms"] + p * d0["t_m2_ms"]

    def _speedup_at(a: float, which: str) -> float:
        en_a = ACC.expected_committed_length(acc_from_alpha(a, K), K)
        if which == "colocated":
            return t_base * en_a / (_t_draft(a, "colocated") + t_verify)
        t2d_a = link_ms(link, B * (en_a * draft.t2d_bytes_per_token
                                   + draft.t2d_bytes_per_request))["t_ms"]
        td = _t_draft(a, "dt")
        if which == "dt_serial":
            return t_base * en_a / (td + d2t["t_ms"] + t_verify + t2d_a)
        return t_base * en_a / (max(td + d2t["t_ms"], t_verify) + t2d_a)

    def acc_from_alpha(a: float, k: int):
        from .contracts import AcceptanceSpec
        return AcceptanceSpec(kind="chain_alpha", chain_alpha=a)

    breakevens = {w: ACC.breakeven_alpha(lambda a, w=w: _speedup_at(a, w))
                  for w in ("colocated", "dt_serial", "dt_pipeline_ideal")}

    # ---- 时间线（代表性一轮；mode 决定串行/理想流水排布）----
    timeline = _build_timeline(mode, draft_st_co if mode == "colocated" else draft_st_dt,
                               t_verify, d2t["t_ms"], t2d["t_ms"], en, K)

    # ---- 汇总指标 ----
    m = {"E_committed_len": en, "alpha": alpha, "P_full": ACC.catchup_prob_full(alpha, K),
         "E_draft_step0_tokens": ACC.expected_draft_step0_tokens(alpha, K),
         "verify_tokens_M": B * (K + 1), "K": K, "B": B, "ctx_tokens": ctx,
         "t_base_ms": t_base, "t_prefill_ms": t_prefill, "t_verify_ms": t_verify,
         "t_draft_colocated_ms": t_draft_co, "t_draft_dt_ms": t_draft_dt,
         "t_d2t_ms": d2t["t_ms"], "t_t2d_ms": t2d["t_ms"],
         "bytes_d2t": bytes_d2t, "bytes_t2d": bytes_t2d,
         "memory": {"weight_capacity_per_card": weight_cap, "kv_retained_per_card": kv_ret,
                    "target_mem_gb": target.mem_capacity / 1e9,
                    "oom_target": oom_target, "oom_draft": oom_draft,
                    "draft_weight_bytes": draft_cap},
         "baseline_tps_per_req": 1000.0 / t_base if t_base else 0.0,
         "breakeven_alpha": breakevens}

    # ---- 装配 RunResult ----
    res = RunResult(
        status="oom" if oom_target else "ok",
        status_detail="每卡权重驻留 + KV 保留超出设备容量（逻辑算量仍完整输出）" if oom_target else "",
        scenario_digest=digest(scenario.to_json()),
        model_digest=digest(model.to_json()),
        device_digest=digest({"target": target.to_json(), "draft": draft_dev.to_json(),
                              "link": link.to_json() if hasattr(link, "to_json") else link.__dict__}),
        scenario=scenario.to_json(),
        model=model.to_json(),
        draft=draft.to_json(),
        devices={"target": target.to_json(), "draft": draft_dev.to_json(),
                 "link": link.__dict__},
    )
    # 算子账本：基线 / prefill / verify / 草稿（同卡与 DT 各一份，catch-up 展开为 m1/m2）
    res.ops = list(base_led.rows) + list(prefill_led.rows) + list(verify_led.rows) \
        + _collect_ops(draft_st_co) + _collect_ops(draft_st_dt)
    res.stages = ([StageTiming("baseline_decode", "基线 decode 单步（B×1 token）", "target",
                               t_base, F.F_TPOT_BASE, tokens_in=B),
                   StageTiming("prefill", f"prefill（B×ctx={B * ctx} token）", "target",
                               t_prefill, F.F_TPOT_BASE, tokens_in=B * ctx),
                   StageTiming("verify", f"验证步 M=B×(K+1)={B * (K + 1)} token", "target",
                               t_verify, F.F_TPOT_COLOCATED, tokens_in=B * (K + 1))]
                  + [StageTiming(s.stage, s.label + "（同卡）", "draft", s.t_ms, s.formula_id,
                                 tokens_in=s.tokens_in) for s in draft_st_co]
                  + [StageTiming(s.stage + "@dt", s.label + "（DT 分离）", "draft", s.t_ms,
                                 s.formula_id, tokens_in=s.tokens_in) for s in draft_st_dt]
                  + [StageTiming("d2t", "候选 D→T 传输", "link", d2t["t_ms"], F.F_COMM_BYTES,
                                 detail={"bytes": bytes_d2t}),
                     StageTiming("t2d", "反馈 T→D 传输", "link", t2d["t_ms"], F.F_COMM_BYTES,
                                 detail={"bytes": bytes_t2d})])
    res.timeline = timeline
    res.metrics = m
    res.comparisons = comparisons
    res.assumptions = [
        "求值口径：解析档（单请求/固定批形状、无请求间干扰）；事件状态机与 serving 属 M3/M4",
        "全部时延为 T0 roofline 公式估算（决策 D-4：首发不采数；MFU 上限 0.75、"
        "访存聚合 HBM 带宽、15% 重叠尾部），无实测/拟合数字",
        "launch 开销未逐算子计入（eager 上界另行可查）；CUDA Graph 未建模",
        "链式草稿 S=K、步 M=1；全接受轮下一轮首步 M=2 按 P_full=α^K 概率加权（E[M₀]=1+α^K）",
        "草稿 KV 读按目标单步 KV 读 × kv_frac 折算（旧仓 ratio 口径，近似标注）",
        "通信为整批单消息/方向（链路 fixed + bytes/bw），链路竞争排队属 M3 事件核",
        "MoE 专家并集 U(M) 用 i.i.d. 近似（draws=T×top_k）；接受率 α 为外生输入，仿真不预测",
        "DT 分离草稿按 TP1 独立设备计价；同卡投机草稿按目标 TP 组分片计价",
    ]
    res.provenance = {
        "counts": "公式（Count 段整数算术；公式溯源见 ops[].formula_id）",
        "durations": "T0 公式估算（roofline；无实测/拟合）",
        "model_config": f"config:官方 config.json 固定快照 {model.config_digest}",
        "hardware": "spec:公开规格（devices[].sources 逐条列出）+ 假设值显式标注 assumed",
        "draft_calibers": "锚:旧仓 spec_gain_presets.py / dt_sim.py / spec_gain_model.js",
    }
    return res


def _build_timeline(mode, draft_stages, t_verify, t_d2t, t_t2d, en, K) -> List[TimelineEvent]:
    """代表性一轮投机的串行（或理想流水）时间线；供 Web Gantt 与 chrome-trace 导出。"""
    events: List[TimelineEvent] = []
    t = 0.0
    if mode == "dt_pipeline_ideal":
        # 草稿+d2t 与上一轮 verify 重叠：稳态周期 = max(draft+d2t, verify) + t2d
        for s in draft_stages:
            events.append(TimelineEvent(f"e{len(events)}", s.label, "draft", t, t + s.t_ms,
                                        stage=s.stage))
            t += s.t_ms
        events.append(TimelineEvent(f"e{len(events)}", "候选 D→T 传输", "link", t, t + t_d2t,
                                    depends_on=[events[-1].id]))
        t_link_end = t + t_d2t
        # verify 从 0 起并行（与草稿重叠），t2d 在 max 后
        events.append(TimelineEvent(f"e{len(events)}", "验证步 Verify", "target", 0.0,
                                    t_verify, stage="verify"))
        t2d_start = max(t_link_end, t_verify)
        events.append(TimelineEvent(f"e{len(events)}", "反馈 T→D 传输", "link", t2d_start,
                                    t2d_start + t_t2d,
                                    depends_on=[events[-1].id]))
        events.append(TimelineEvent(f"e{len(events)}",
                                    f"提交 E[n]={en:.2f} token（含 bonus）", "commit",
                                    t2d_start + t_t2d, t2d_start + t_t2d))
    else:
        for s in draft_stages:
            events.append(TimelineEvent(f"e{len(events)}", s.label, "draft", t, t + s.t_ms,
                                        stage=s.stage,
                                        depends_on=[events[-1].id] if events else []))
            t += s.t_ms
        if mode in ("dt_serial", "dt_pipeline_ideal"):
            events.append(TimelineEvent(f"e{len(events)}", "候选 D→T 传输", "link", t,
                                        t + t_d2t, depends_on=[events[-1].id]))
            t += t_d2t
        events.append(TimelineEvent(f"e{len(events)}", "验证步 Verify", "target", t,
                                    t + t_verify, stage="verify",
                                    depends_on=[events[-1].id] if events else []))
        t += t_verify
        if mode in ("dt_serial", "dt_pipeline_ideal"):
            events.append(TimelineEvent(f"e{len(events)}", "反馈 T→D 传输", "link", t,
                                        t + t_t2d, depends_on=[events[-1].id]))
            t += t_t2d
        events.append(TimelineEvent(f"e{len(events)}",
                                    f"提交 E[n]={en:.2f} token（含 bonus）", "commit", t, t))
    return events
