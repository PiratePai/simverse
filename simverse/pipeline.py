"""v0.2 计算管线：ScenarioV2 → ResultBundle（CLI/Web/导出唯一入口）。

数据流：Scenario + ModelIR → MethodPlan → Work(Count) → Lowering(TP/容量/通信)
→ Cost(T0) → Execute(串行轮事件) → ResultBundle（指标/能力/证据/公式）。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from . import formula_registry as FR
from .contracts import (CapabilityReport, Evidence, Feasibility, Metric,
                        ResultBundle, ResultKind, RunManifest, Scope, Statistic,
                        Support, TimeBasis, metric_from, digest)
from .count import build_model_work, totals
from .count.draft import chain_step_rows, dflash_round_rows, mtp_step_rows
from .devices import DEVICES, LINKS  # v0.1 设备注册表（v0.2 沿用数值+分层预算）
from .cost.t0 import dtype_route, price_rows, stage_total
from .cost.calibrated import CalibratedPricer
from .cost.profile_store import PROFILE_SETS, load_profile_set
from .execute.serial import build_serial_round
from .lowering import (DeviceBudget, LinkLike, colocated_rank_memory,
                       kv_shard_note, model_collectives, validate_parallel)
from .methods import build_method_plan, acceptance_summary
from .methods.plans import UNSUPPORTED
from .model_ir import build_catalog
from .validation import code_revision
from .scenario import ScenarioV2


def _link_like(link_id: str) -> LinkLike:
    lk = LINKS[link_id]
    return LinkLike(bw=lk.bw, fixed_ms=lk.fixed_ms, label=lk.label)


def _device_budget(device_id: str, budget_input=None) -> DeviceBudget:
    """设备预算：标称容量 + 场景预算输入（R03a；缺省 unknown 不套标称）。"""
    d = DEVICES[device_id]
    b = DeviceBudget(device_id, int(d.mem_capacity), None,
                     budget_note="标称容量；可分配预算未提供 → feasible 判定 "
                                 "unknown（§7.4；已知对象超标称仍判 OOM）")
    if budget_input is not None:
        b.allocatable_bytes = budget_input.allocatable_bytes
        b.allocatable_kind = budget_input.allocatable_kind
        b.allocatable_source = budget_input.allocatable_source
        b.extra_overhead_bytes = budget_input.extra_overhead_bytes
        b.overhead_source = budget_input.overhead_source
        if b.allocatable_bytes is not None and b.allocatable_bytes > b.nominal_bytes:
            raise ValueError(
                f"allocatable_bytes {b.allocatable_bytes} > 标称容量 "
                f"{b.nominal_bytes}（{device_id}）：非法输入")
    return b


def run_v2(scenario: ScenarioV2) -> ResultBundle:
    scenario.validate()
    cat = build_catalog()
    cap = CapabilityReport()

    # ---- 模型解析 + 能力检查 ----
    model = cat["models"].get(scenario.model_id)
    if model is None:
        raise ValueError(f"未知模型 {scenario.model_id}")
    cap.add("model_registered", "supported", model.id)
    if not model.structure_complete:
        cap.add("model_structure_complete", "unsupported",
                "存在 unsupported-semantic 字段（Engram/hc 等）",
                affected_metrics=["M06 每卡显存", "M02/M04 完整模型性能", "M05 MoE 完整读"])
    else:
        cap.add("model_structure_complete", "supported", "全部结构字段已映射")
    draft = cat["drafts"].get(scenario.draft_id) if scenario.draft_id else None

    # ---- 方法 ----
    if scenario.method_id in UNSUPPORTED:
        return _bundle_error(scenario, cap, "unsupported_method",
                             f"{scenario.method_id}: {UNSUPPORTED[scenario.method_id]}")
    if scenario.method_id == "baseline":
        plan = build_method_plan("baseline", {}, 1)
        draft_ir = None
    else:
        if draft is None and scenario.method_id == "dflash_v1":
            return _bundle_error(scenario, cap, "unsupported_method",
                                 "dflash_v1 需要 DraftIR（catalog.drafts）")
        try:
            plan = build_method_plan(
                scenario.method_id, scenario.acceptance, scenario.k,
                draft_ir=draft, mtp_layers=model.mtp_layers,
                block_size=scenario.block_size)
        except ValueError as exc:
            return _bundle_error(scenario, cap, "unsupported_combo", str(exc))
        draft_ir = draft
    if scenario.method_id == "dflash_v1" and model.id != "qwen36_35b_a3b":
        return _bundle_error(scenario, cap, "unsupported_combo",
                             "该 DFlash checkpoint 仅核对 Qwen3.6-35B-A3B 配对；不能用于其他主模型")
    acc = acceptance_summary(plan)

    # ---- 实测 profile 校准（可选；MERGE_CRITERIA §3：只改成本不改计数）----
    pricer = None
    if scenario.cost_profile:
        pset = load_profile_set(scenario.cost_profile)  # 完整性不符即抛错
        pricer = CalibratedPricer(pset, model, scenario)
        if pricer.applicable:
            cap.add("cost_profile_domain", "supported",
                    f"{scenario.cost_profile}：设备/精度/TP/形状域匹配")
        else:
            cap.add("cost_profile_domain", "approximated",
                    f"{scenario.cost_profile} 域不匹配：{'；'.join(pricer.domain_reasons)}",
                    affected_metrics=["M01_tpot_profile_calibrated"])

    # ---- 部署合法性（R01：统一 ParallelPlan；非法组合显式拒绝）----
    dep = scenario.deployment
    pplan = validate_parallel(model, dep.target_tp, dep.target_ep,
                              draft if (draft is not None
                                        and scenario.method_id == "dflash_v1"
                                        and dep.mode in ("colocated",
                                                         "dt_pipeline_sim"))
                              else None)
    if not pplan.valid:
        cap.add("parallel_valid", "invalid", "; ".join(pplan.errors[:2]),
                affected_metrics=["全部"])
        return _bundle_error(scenario, cap, "unsupported_combo",
                             f"TP{dep.target_tp}/EP{dep.target_ep} 非法："
                             f"{pplan.errors[0]}")
    cap.add("parallel_valid", "supported",
            f"TP{dep.target_tp}/EP{dep.target_ep}（{pplan.implementation_id}，"
            f"pin {pplan.source_pin[:22]}…）；KV 分片："
            f"{kv_shard_note(model, dep.target_tp)['strategy']}"
            + ("；EP 开：routed 整专家按 rank 分置" if pplan.moe_ep_enabled
               else "；EP 关：routed inter/tp 切分"))
    if scenario.method_id == "dflash_v1" and dep.mode == "dt_serial":
        cap.add("dflash_serving_integration", "unknown",
                "SGLang DFLASH/vLLM PR#40898 未逐行核对：模拟按 HF 语义，"
                "不冒充已部署验证", affected_metrics=["部署矩阵 served 标记"])

    # ---- 计数与计价 ----
    quant, dquant = scenario.quant, scenario.draft_quant
    gemm_dt, ew_dt = dtype_route(quant)
    target_dev = DEVICES[dep.target_device]
    draft_dev = DEVICES[dep.draft_device]
    B, ctx = scenario.workload.batch, scenario.workload.ctx_tokens
    work: Dict[str, Any] = {}
    cost: Dict[str, Any] = {}
    tp_link = _link_like(dep.link)
    calibration: Dict[str, Any] = {
        "profile_set": scenario.cost_profile,
        "stages": {},
        "note": "selected=域内实测+域外解析(差额)+kernel间隙；原始计数不变（G7）"
                "；草稿阶段无匹配域不校准",
    } if pricer else None

    base_rows = build_model_work(model, "target_decode", B, ctx, 1, quant,
                                 dep.target_tp, role="target", ep=dep.target_ep,
                                 plan=pplan)
    base_prices = price_rows(base_rows, target_dev, gemm_dt, ew_dt)
    t_base = stage_total(base_prices)
    roofline_points = _roofline_points(base_rows, base_prices, target_dev, gemm_dt)
    # 通信（R01 按层配方：attn allreduce 恒有；MLP 段 TP allreduce 或 EP all2all）
    col_base = _collectives_cost(model, pplan, B, quant, tp_link)
    t_base += col_base["t_ms"]
    if pricer:
        cal = pricer.price_stage("target_decode", base_rows, base_prices, B, ctx, 1)
        cal["collectives_ms"] = col_base["t_ms"]
        calibration["stages"]["target_decode"] = cal
    work["target_decode"] = _rows_json(base_rows)
    cost["target_decode"] = {"t_ms": t_base, "collectives": col_base["detail"]}
    cost["roofline_points"] = roofline_points

    stage_costs: Dict[str, float] = {}
    if scenario.method_id == "baseline":
        stage_costs["decode"] = t_base
        e_n = 1.0
        round_obj = build_serial_round(plan, stage_costs, 1.0)
        draft_rows: List = []
        msg_bytes = {"d2t": 0, "t2d": 0}
    else:
        e_n = acc["E_N"]
        if e_n is None:
            # 依赖污染按指标传播（plan §4）：保留结构/基线/容量已知部分
            cap.add("acceptance_known", "unsupported",
                    f"E[N] 未知：{acc['reason']}",
                    affected_metrics=["M01 TPOT", "M02 加速比", "M03 E[N]", "M04 轮分解"])
            memory = _memory_block(model, None, quant, dquant, dep, B, ctx,
                                   cap, plan=pplan)
            from .execute.serial import SerialRound
            empty = SerialRound("r0", [], 0.0, None, None, acc["reason"])
            metrics = _build_metrics_null_acceptance(acc, t_base, memory, cap, model)
            manifest = RunManifest(run_id="", status="completed",
                                   input_digest=scenario.scenario_digest(),
                                   model_ir_digest=digest(model.to_json()),
                                   method_digest=digest(plan.to_json()),
                                   code_revision=code_revision())
            manifest.compute_run_id()
            return ResultBundle(
                manifest=manifest, scenario_snapshot=scenario.to_json(),
                model_ref={"id": model.id, "structure_complete": model.structure_complete,
                           "gaps": model.gaps, "source": model.source.to_json()},
                method_plan=plan.to_json(),
                work={"target_decode": _rows_json(base_rows)},
                cost={"target_decode": {"t_ms": t_base}},
                execution={"rounds": [empty.to_json()], "mode": "serial-analytical"},
                metrics=metrics, capability=cap.to_json(),
                evidence_refs=["ev-qwen36-main-config"],
                formula_dict=FR.formulas_json()["formulas"],
                assumption_dict=_assumptions(scenario, plan))
        if scenario.method_id == "dflash_v1":
            # 草稿与 target 同 TP 组（vLLM load_dflash_model pin；dt_serial
            # 独立设备时草稿组 TP1 全量副本——无共享配方，见 memory）
            draft_tp = dep.target_tp if dep.mode in ("colocated",
                                                     "dt_pipeline_sim") else dep.draft_tp
            draft_rows = dflash_round_rows(
                "draft_block", draft, model.vocab, e_n, ctx, dquant,
                quant["act"] / 8, tp=draft_tp)
            d_gemm, d_ew = dtype_route(dquant)
            d_dev = target_dev if dep.mode == "colocated" else draft_dev
            d_prices = price_rows(draft_rows, d_dev, d_gemm, d_ew)
            t_draft = stage_total(d_prices)
            verify_rows = build_model_work(model, "verify_block", B, ctx,
                                           plan.block_size, quant, dep.target_tp,
                                           role="target", ep=dep.target_ep,
                                           plan=pplan)
            v_prices = price_rows(verify_rows, target_dev, gemm_dt, ew_dt)
            t_verify = stage_total(v_prices)
            col_v = _collectives_cost(model, pplan, B * plan.block_size,
                                      quant, tp_link)
            t_verify += col_v["t_ms"]
            if pricer:
                calibration["stages"]["verify_block"] = pricer.price_stage(
                    "verify_block", verify_rows, v_prices, B, ctx, plan.block_size)
                calibration["stages"]["draft_block"] = {
                    "stage": "draft_block", "domain_ok": False,
                    "reasons": ["草稿模型（DFlash 6L dense GQA）无匹配 profile 域"],
                    "selected_ms": None}
            blk = plan.block_size
            greedy = scenario.acceptance.get("kind") == "geometric"  # 概率随采样策略
            msg_bytes = {
                "d2t": int((blk - 1) * (4 + (2 if not greedy else 0))),
                "t2d": int(e_n * len(draft.target_layer_ids) * draft.hidden
                           * (quant["act"] / 8)),
            }
            stage_costs = {"draft_block": t_draft, "verify_block": t_verify}
            work["draft_block"] = _rows_json(draft_rows)
            work["verify_block"] = _rows_json(verify_rows)
            cost["draft_block"] = {"t_ms": t_draft, "device": d_dev.id}
            cost["verify_block"] = {"t_ms": t_verify,
                                    "collectives": col_v["detail"]}
        elif scenario.method_id == "mtp":
            draft_rows = mtp_step_rows("draft_step_0", model, 1, ctx, quant,
                                       model.mtp_layers or 1)
            d_prices = price_rows(draft_rows, target_dev, gemm_dt, ew_dt)
            t_draft = stage_total(d_prices)
            verify_rows = build_model_work(model, "verify", B, ctx,
                                           plan.k_effective + 1, quant,
                                           dep.target_tp, role="target",
                                           ep=dep.target_ep, plan=pplan)
            v_prices_m = price_rows(verify_rows, target_dev, gemm_dt, ew_dt)
            t_verify = stage_total(v_prices_m)
            if pricer:
                calibration["stages"]["verify"] = pricer.price_stage(
                    "verify", verify_rows, v_prices_m, B, ctx,
                    plan.k_effective + 1)
            msg_bytes = {"d2t": int(B * (plan.k_effective + 1) * 6),
                         "t2d": int(e_n * model.hidden * (quant["act"] / 8))}
            stage_costs = {"draft_step_0": t_draft, "verify": t_verify}
            work["draft_step_0"] = _rows_json(draft_rows)
            work["verify"] = _rows_json(verify_rows)
        else:  # chain_draft / chain_moe_proxy：草稿用 tiny_gqa 结构演示闭环
            draft_model = cat["models"]["tiny_gqa"]
            rows_all = []
            t_draft = 0.0
            p_full = acc["P_full"] or 0.0
            for j in range(plan.k_effective):
                m = 2 if j == 0 else 1
                weight = (1 - p_full if j == 0 else 1.0)
                rows = chain_step_rows(f"draft_step_{j}", draft_model, m, ctx,
                                       dquant, 1)
                t = stage_total(price_rows(rows, target_dev, gemm_dt, ew_dt))
                t_draft += t * (weight if j == 0 else 1.0) + (
                    p_full * t if j == 0 else 0.0)
                rows_all += rows
            verify_rows = build_model_work(model, "verify", B, ctx,
                                           plan.k_effective + 1, quant,
                                           dep.target_tp, role="target",
                                           ep=dep.target_ep, plan=pplan)
            v_prices_c = price_rows(verify_rows, target_dev, gemm_dt, ew_dt)
            t_verify = stage_total(v_prices_c)
            if pricer:
                calibration["stages"]["verify"] = pricer.price_stage(
                    "verify", verify_rows, v_prices_c, B, ctx,
                    plan.k_effective + 1)
            msg_bytes = {"d2t": int(B * (plan.k_effective + 1) * 6),
                         "t2d": int(B * 256)}
            stage_costs = {f"draft_step_{j}": t_draft / max(1, plan.k_effective)
                           for j in range(plan.k_effective)}
            stage_costs["verify"] = t_verify
            draft_rows = rows_all
            work["draft_steps"] = _rows_json(rows_all)
            work["verify"] = _rows_json(verify_rows)
        # 链路
        if dep.mode == "dt_serial":
            lk = LINKS[dep.link]
            t_d2t = lk.fixed_ms + msg_bytes["d2t"] / lk.bw * 1e3
            t_t2d = lk.fixed_ms + msg_bytes["t2d"] / lk.bw * 1e3
        else:
            t_d2t = t_t2d = 0.0
        step_map = {"d2t": t_d2t, "t2d": t_t2d}
        stage_costs.update({k: v for k, v in step_map.items() if v or dep.mode == "dt_serial"})
        cost["link"] = {"t_d2t_ms": t_d2t, "t_t2d_ms": t_t2d, "bytes": msg_bytes}
        if dep.mode == "dt_pipeline_sim":
            # M3 资源事件核：真实调度（互斥资源+半双工链路），周期≠请求延迟
            from .execute.resource_core import dt_pipeline_schedule
            sim = dt_pipeline_schedule(
                stage_costs.get("draft_block", 0.0) + stage_costs.get("draft_step_0", 0.0),
                t_d2t, stage_costs.get("verify_block", stage_costs.get("verify", 0.0)),
                t_t2d, rounds=dep.pipeline_rounds,
                microbatches=dep.pipeline_microbatches)
            if sim.violations:
                cap.add("pipeline_schedule_valid", "unsupported",
                        "; ".join(v[2] for v in sim.violations[:2]),
                        affected_metrics=["M18 周期/利用率"])
            else:
                cap.add("pipeline_schedule_valid", "supported",
                        "事件流通过 validate_trace（依赖/互斥/消息序）")
            round_obj = build_serial_round(plan, stage_costs, e_n)
            cost["pipeline_sim"] = sim.to_json()
        else:
            round_obj = build_serial_round(plan, stage_costs, e_n)
        # 串行时间线补 link 事件（计划步骤已含 d2t/t2d）

    # ---- 校准摘要与 selected 轮成本（不改 stage_costs 解析值与原始计数）----
    calibrated_tpot = None
    calibrated_baseline = None
    if pricer and calibration is not None:
        cost["calibration"] = calibration
        cov_rows = sum(s.get("coverage_rows", [0, 0])[0]
                       for s in calibration["stages"].values())
        tot_rows = sum(s.get("coverage_rows", [0, 0])[1]
                       for s in calibration["stages"].values())
        calibration["coverage"] = {
            "rows_covered": cov_rows, "rows_total": tot_rows,
            "holdout_error_report": pricer.pset.manifest["holdout"]["error_report"],
            "environment": pricer.pset.manifest["environment"],
            "source_head": pricer.pset.manifest["source"]["git_head"]}
        if scenario.method_id == "baseline":
            cal0 = calibration["stages"]["target_decode"]
            if cal0.get("selected_ms") is not None:
                sel = cal0["selected_ms"] + col_base["t_ms"]
                calibrated_tpot = sel
                calibrated_baseline = sel
        else:
            sel_costs = dict(stage_costs)
            repl = {}
            for stname, cal in calibration["stages"].items():
                if cal.get("selected_ms") is None or not cal.get("domain_ok"):
                    continue
                base_t = cal["analytic_ms"]
                extra = (cal.get("collectives_ms") or 0.0)
                # collectives 已在 stage_costs 内（t_verify 含 col_v）→ 差额替换
                repl[stname] = cal["selected_ms"] + extra
                sel_costs[stname] = repl[stname]
            if repl:
                round_sel = build_serial_round(plan, sel_costs, e_n)
                if round_sel.tpot_ms is not None:
                    calibrated_tpot = round_sel.tpot_ms
                cal0 = calibration["stages"].get("target_decode")
                if cal0 and cal0.get("selected_ms") is not None:
                    calibrated_baseline = cal0["selected_ms"] + col_base["t_ms"]

    # ---- 容量（R02a/b：per-rank 对象账本 + 生命周期峰值；R03：预算判定）----
    memory = _memory_block(
        model, draft if scenario.method_id == "dflash_v1" else None,
        quant, dquant, dep, B, ctx, cap, plan=pplan,
        block_size=(plan.block_size
                    if scenario.method_id == "dflash_v1" and e_n is not None else 0),
        e_produced=(e_n if e_n is not None else 0.0))

    cost["memory"] = memory       # R02a：per-rank 对象清单与判定入册（导出/UI 共用）

    # ---- 指标 ----
    metrics = _build_metrics(scenario, plan, acc, round_obj, t_base, msg_bytes,
                             base_rows, memory, cap, model, draft)
    if calibrated_tpot is not None:
        metrics["M01_tpot_profile_calibrated"] = metric_from(
            calibrated_tpot, "ms", "profile:" + scenario.cost_profile,
            statistic=Statistic.EXPECTATION,
            result_kind=ResultKind.MEASURED_COMPOSITE,
            support=Support.APPROXIMATED,
            evidence=(Evidence.MEASURED, Evidence.ASSUMED),
            scope=Scope(role="target", phase="steady-decode",
                        time_basis=TimeBasis.COMMITTED.value),
            assumptions=[
                "算子级实测 median 组合（域内查表/插值），非整步实测；"
                "组合含 eager kernel 间隙估计",
                "域外部分（MoE/router/lm_head/草稿/verify 注意力）仍为解析估算",
                f"留出误差见 cost.calibration.coverage（来源 "
                f"{scenario.cost_profile}）"])
    if calibrated_baseline is not None:
        metrics["M01_tpot_baseline_calibrated"] = metric_from(
            calibrated_baseline, "ms", "profile:" + scenario.cost_profile,
            statistic=Statistic.EXPECTATION,
            result_kind=ResultKind.MEASURED_COMPOSITE,
            support=Support.APPROXIMATED, evidence=(Evidence.MEASURED,),
            scope=Scope(role="target", phase="steady-decode",
                        time_basis=TimeBasis.COMMITTED.value),
            assumptions=["同 M01_tpot_profile_calibrated 口径"])
    # M18 周期/利用率/流水（dt_pipeline_sim；周期≠单请求 TPOT，分列）
    if cost.get("pipeline_sim"):
        ps = cost["pipeline_sim"]
        n_round_equiv = scenario.deployment.pipeline_rounds *             scenario.deployment.pipeline_microbatches
        metrics["M18_cohort_period_ms"] = {
            k: metric_from(v, "ms", "internal", statistic=Statistic.SAMPLE,
                           scope=Scope(role="system", phase="steady-decode"),
                           evidence=(Evidence.COUNTED,),
                           assumptions=["两微批交错实测调度；非 max 通则"])
            for k, v in ps["cohort_periods_ms"].items()}
        metrics["M18_per_round_throughput_ms"] = metric_from(
            ps["makespan_ms"] / n_round_equiv, "ms/round-equiv", "internal",
            statistic=Statistic.SAMPLE, scope=Scope(role="system"),
            evidence=(Evidence.COUNTED,),
            assumptions=["有限 cohort 含 warmup/drain"])
        metrics["M18_utilization"] = {
            k: metric_from(v, "ratio", "internal", statistic=Statistic.EXACT,
                           scope=Scope(role="system", rank=k),
                           evidence=(Evidence.COUNTED,))
            for k, v in ps["utilization"].items()}
        metrics["M01_tpot_analytical"] = Metric(
            value=metrics["M01_tpot_analytical"].value, unit="ms",
            scope=Scope(role="target", phase="steady-decode"),
            statistic=Statistic.EXPECTATION, result_kind=ResultKind.ANALYTICAL_ESTIMATE,
            support=Support.APPROXIMATED, evidence_refs=[], formula_ref=FR.F_ROOFLINE,
            assumption_refs=[],
            reason="dt_pipeline_sim：解析串行 TPOT 与流水周期分列（M18）；"
                   "单请求延迟需请求轨迹（M3 请求级事件）")

    manifest = RunManifest(run_id="", status="completed",
                           input_digest=scenario.scenario_digest(),
                           model_ir_digest=digest(model.to_json()),
                           method_digest=digest(plan.to_json()),
                           cost_digest=digest({"t0": FR.F_ROOFLINE,
                                               "profile": scenario.cost_profile}),
                           device_snapshot_digest=digest(
                               {"target": dep.target_device, "tp": dep.target_tp,
                                "draft": dep.draft_device, "link": dep.link}),
                           code_revision=code_revision())
    manifest.compute_run_id()
    bundle = ResultBundle(
        manifest=manifest,
        scenario_snapshot=scenario.to_json(),
        model_ref={"id": model.id, "label": model.label, "family": model.family,
                   "structure_complete": model.structure_complete,
                   "gaps": model.gaps, "kind_counts": model.kind_counts(),
                   "source": model.source.to_json(),
                   "coverage": model.coverage,
                   "digest": manifest.model_ir_digest},
        method_plan=plan.to_json(),
        work={k: v for k, v in work.items()},
        cost=cost,
        execution={"rounds": [round_obj.to_json()],
                   "mode": "serial-analytical",
                   "note": "串行解析档：阶段和=轮墙钟（无重叠）；M3 事件核接入后"
                           "本值作同假设对照"},
        metrics=metrics,
        capability=cap.to_json(),
        evidence_refs=[*(["ev-h200-tiny-distill"] if pricer else []),
                       "ev-qwen36-main-config", "ev-qwen36-dflash-config",
                       "ev-dflash-impl", "ev-dflash-readme", "ev-v41-config",
                       "ev-vllm-parallel-recipe"],
        formula_dict=FR.formulas_json()["formulas"],
        assumption_dict=_assumptions(scenario, plan),
    )
    return bundle


def _collectives_cost(model, plan, tokens: int, quant: dict, link: LinkLike):
    """按层通信配方（R01）：EP 开 MoE 层为 dispatch/combine，非 2×allreduce。"""
    if plan.tp < 2:
        return {"t_ms": 0.0, "detail": {"note": "TP1 无集合通信",
                                        "total_t_ms": 0.0}}
    top_k = next((l.moe.top_k for l in model.layers
                  if l.moe is not None and l.moe.experts), 0)
    cols = model_collectives(model, plan, model.hidden, quant["act"] / 8,
                             tokens, link, moe_top_k=top_k)
    t = sum(c.t_ms for c in cols)
    per_kind: Dict[str, int] = {}
    for c in cols:
        per_kind[c.kind] = per_kind.get(c.kind, 0) + 1
    return {"t_ms": t, "detail": {
        "per_layer": [c.to_json() for c in cols[:3]],
        "counts_by_kind": per_kind, "layers": len(model.layers),
        "recipe": ("EP all2all（MoE 层）+ attn allreduce" if plan.moe_ep_enabled
                   else "每层 attn+mlp 2 次 allreduce"),
        "total_t_ms": t}}


def _roofline_points(rows, prices, device, gemm_dtype: str, limit: int = 40):
    """算子 Roofline 坐标（后端算好：AI/attained/bound；前端只画点）。"""
    pts = []
    for row, price in zip(rows, prices):
        if row.flops <= 0:
            continue
        total_bytes = price.get("bytes_exact", 0) + price.get("bytes_expected", 0)
        if total_bytes <= 0:
            continue
        dtype = gemm_dtype if row.op not in __import__(
            "simverse.cost.t0", fromlist=["_ELEMENTWISE"])._ELEMENTWISE else gemm_dtype
        pts.append({
            "row_id": f"{row.stage}:{row.layer}:{row.op}",
            "layer": row.layer, "op": row.op, "stage": row.stage,
            "ai_flop_per_byte": row.flops / total_bytes,
            "attained_tflops": row.flops / (price["t_ms"] * 1e-3) / 1e12 if price["t_ms"] else 0.0,
            "bound": price["bound"], "t_ms": price["t_ms"],
        })
    pts.sort(key=lambda x: -x["t_ms"])
    return pts[:limit]


def _rows_json(rows):
    t = totals(rows)
    return {"totals": {k: v for k, v in t.items()},
            "rows": [r.to_json() for r in rows]}


def _memory_block(model, draft_ir, quant, dquant, dep, B, ctx, cap,
                  plan=None, block_size: int = 0, e_produced: float = 0.0):
    """容量块（R02a/b：per-rank 对象身份 + 生命周期峰值；R03a/b：预算判定）。

    colocated：草稿与 target 同 TP 组，全部 rank 各建一份账本（EP 余数
    rank 差异显式呈现），报告逐 rank + 最坏 rank；dt_serial：target/draft
    分设备。峰值含结构化临时缓冲；workspace/graph/NCCL 池为缺口组件
    （不填 0）→ 峰值不完整 → feasible 判定 unknown（已知下界超标称仍 OOM）。
    """
    from .lowering.memory import (RankMemory, colocated_rank_memory,
                                  kv_allocations, target_weight_allocations)
    pplan = plan
    blocks: Dict[str, Any] = {}
    if dep.mode == "colocated":
        tp = dep.target_tp if pplan is None else pplan.tp
        ep = dep.target_ep if pplan is None else pplan.ep
        ranks = []
        for r in range(tp):
            rank_mem = colocated_rank_memory(
                model, draft_ir, quant, dquant, tp, B, ctx,
                _device_budget(dep.target_device, dep.budget),
                ep=ep, rank=r, block_size=block_size, e_produced=e_produced)
            rep = rank_mem.peak_vs_budget()
            rep["inventory"] = rank_mem.inventory()
            ranks.append(rep)
        blocks["colocated_ranks"] = ranks
        # M06 键沿用 r0；ranks 非均匀（EP 余数）时逐 rank 单列
        blocks["colocated_r0"] = dict(ranks[0])
        if any(x["known_peak_bytes"] != ranks[0]["known_peak_bytes"] for x in ranks):
            for r, rep in enumerate(ranks):
                blocks[f"colocated_r{r}"] = dict(rep)
        worst = max(ranks, key=lambda x: x["known_peak_bytes"])
        blocks["colocated_worst"] = {"rank_id": worst["rank_id"],
                                     "known_peak_bytes": worst["known_peak_bytes"]}
    else:
        from .lowering.memory import RankMemory as RM
        tb = _device_budget(dep.target_device, dep.budget)
        ep = dep.target_ep if pplan is None else pplan.ep
        rm = RM("target-r0", dep.target_device, tb)
        for a in target_weight_allocations(model, quant, dep.target_tp,
                                           ep=ep, plan=pplan):
            a.consumers = ["target"]
            rm.add(a)
        for a in kv_allocations(model, quant, B, ctx, tp=dep.target_tp):
            a.consumers = ["target"]
            rm.add(a)
        rep = rm.peak_vs_budget()
        rep["inventory"] = rm.inventory()
        blocks["target_r0"] = rep
        if draft_ir is not None and dep.mode == "dt_serial":
            # 草稿独立设备 TP1：无共享对象配方（vLLM 配方为同组共享，
            # 独立组需全量 embed/lm_head 副本——服务栈未见证据，如实登记）
            db = _device_budget(dep.draft_device, dep.draft_budget)
            drm = RM("draft-r0", dep.draft_device, db)
            from .lowering.memory import _as_model, draft_extra_allocations
            dm = _as_model(draft_ir)
            dq = dict(quant)
            dq.update(dquant or {})
            for a in target_weight_allocations(dm, dq, dep.draft_tp,
                                               role="draft"):
                a.consumers = ["draft"]
                if "embed" in a.alloc_id or "lm_head" in a.alloc_id:
                    a.live_note += "；独立组无共享配方：按全量副本口径（保守）"
                drm.add(a)
            for a in draft_extra_allocations(draft_ir, dq, dep.draft_tp):
                a.consumers = ["draft"]
                drm.add(a)
            for a in kv_allocations(dm, dq, B, ctx, role="draft",
                                    tp=dep.draft_tp):
                a.consumers = ["draft"]
                drm.add(a)
            rep = drm.peak_vs_budget()
            rep["inventory"] = drm.inventory()
            blocks["draft_r0"] = rep
    status_key = {"oom": "unsupported", "unknown": "unknown",
                  "feasible": "supported", "feasible-conditional": "approximated"}
    for rk, blk in list(blocks.items()):
        if not isinstance(blk, dict) or "status" not in blk:
            continue
        st = blk["status"]
        detail = (f"{rk} known_peak={blk.get('known_peak_bytes', blk.get('live_bytes'))}B"
                  f"（steady={blk.get('steady_bytes', '?')}）；"
                  + "; ".join(blk.get("reasons", [])))
        cap.add(f"memory_{rk}", status_key.get(st, "unknown"), detail,
                affected_metrics=["M02 加速比", "M06 显存"])
    return blocks


def _draft_as_model(draft_ir):
    from .lowering.memory import _as_model
    return _as_model(draft_ir)


def _build_metrics(scenario, plan, acc, round_obj, t_base, msg_bytes,
                   base_rows, memory, cap, model, draft) -> Dict[str, Any]:
    m: Dict[str, Any] = {}
    common = dict(result_kind=ResultKind.ANALYTICAL_ESTIMATE,
                  evidence=[Evidence.COUNTED], formula=FR.F_ROOFLINE)
    sc = Scope(role="target", phase="steady-decode", aggregation="per-run",
               time_basis=TimeBasis.COMMITTED.value)
    # M01 TPOT（解析平稳档）
    if round_obj.tpot_ms is not None:
        m["M01_tpot_analytical"] = metric_from(
            round_obj.tpot_ms, "ms", FR.F_ROOFLINE, statistic=Statistic.EXPECTATION,
            scope=sc, evidence=(Evidence.COUNTED,),
            assumptions=["串行解析档 E[T_round]/E[N]；无排队/prefill（M01 口径分列）"])
    else:
        m["M01_tpot_analytical"] = Metric.unknown("ms", round_obj.reason or "unknown")
    m["M01_tpot_baseline"] = metric_from(
        t_base, "ms", FR.F_ROOFLINE, statistic=Statistic.EXPECTATION, scope=sc,
        evidence=(Evidence.COUNTED,), assumptions=["无投机基线同 B/ctx"])
    # M02 加速比（与自身基线同 run 配对；资格检查）
    elig = _eligibility(cap)
    if round_obj.tpot_ms and t_base:
        m["M02_speedup_vs_baseline"] = metric_from(
            t_base / round_obj.tpot_ms, "ratio", FR.F_ROOFLINE,
            statistic=Statistic.EXPECTATION, scope=sc,
            evidence=(Evidence.COUNTED,),
            assumptions=["同 B/ctx/精度/设备；分子分母同为解析估算"],
        ) if elig else Metric(
            value=t_base / round_obj.tpot_ms, unit="ratio", scope=sc,
            statistic=Statistic.EXPECTATION,
            support=Support.APPROXIMATED, feasibility=Feasibility.UNKNOWN,
            evidence_refs=[], formula_ref=FR.F_ROOFLINE,
            reason="不参与排名：" + "; ".join(
                str(chk["reason"]) for cid, chk in cap.checks.items()
                if (cid == "model_structure_complete" and chk["status"] != "supported")
                or (cid.startswith("memory_") and chk["status"] in ("unsupported", "unknown"))))
    else:
        m["M02_speedup_vs_baseline"] = Metric.unknown("ratio", "TPOT 缺失")
    # M03 每轮产出（baseline：恒 1 精确值）
    if plan.method_id == "baseline":
        m["M03_E_N"] = metric_from(1.0, "tokens/round", "internal",
                                   statistic=Statistic.EXACT, scope=sc,
                                   evidence=(Evidence.COUNTED,))
        m["M03_K"] = metric_from(0, "tokens", "internal",
                                 statistic=Statistic.EXACT, scope=sc,
                                 evidence=(Evidence.COUNTED,))
    elif acc["E_N"] is not None:
        m["M03_E_N"] = metric_from(acc["E_N"], "tokens/round", plan.en_formula,
                                   statistic=Statistic.EXPECTATION, scope=sc,
                                   evidence=(Evidence.ASSUMED,),
                                   assumptions=["Q8 hypothesis 接受率输入"])
        m["M03_K"] = metric_from(plan.k_effective or 0, "tokens", "internal",
                                 statistic=Statistic.EXACT, scope=sc,
                                 evidence=(Evidence.CONFIG,))
        if acc["P_full"] is not None:
            m["M03_P_full"] = metric_from(acc["P_full"], "probability",
                                          plan.en_formula,
                                          statistic=Statistic.EXPECTATION, scope=sc,
                                          evidence=(Evidence.ASSUMED,))
    else:
        m["M03_E_N"] = Metric.unknown("tokens/round", acc["reason"] or "")
    # M04 单轮分解
    m["M04_round_ms"] = metric_from(round_obj.t_round_ms, "ms", FR.F_ROOFLINE,
                                    statistic=Statistic.EXPECTATION, scope=sc,
                                    evidence=(Evidence.COUNTED,))
    m["M04_stage_breakdown_ms"] = {
        e.stage: metric_from(e.t_ms, "ms", FR.F_ROOFLINE,
                             statistic=Statistic.EXPECTATION, scope=sc,
                             evidence=(Evidence.COUNTED,))
        for e in round_obj.events}
    # M05 MoE 并集（从账本聚合）
    moe_u = _moe_union_summary(base_rows)
    if moe_u:
        m["M05_moe_union"] = moe_u
    # M06 显存：状态传播（oom→OOM；仅标称/峰值不完整→unknown；
    # 假设预算→feasible 但 support=approximated 且不进实际部署排名）
    for rk, blk in memory.items():
        if not isinstance(blk, dict) or "status" not in blk:
            continue
        feas = {"oom": Feasibility.OOM, "unknown": Feasibility.UNKNOWN,
                "feasible": Feasibility.FEASIBLE,
                "feasible-conditional": Feasibility.FEASIBLE}[blk["status"]]
        conditional = blk["status"] == "feasible-conditional"
        m[f"M06_memory_{rk}"] = Metric(
            value=blk["known_peak_bytes"], unit="B", statistic=Statistic.EXACT,
            support=(Support.APPROXIMATED if conditional else Support.SUPPORTED),
            feasibility=feas,
            formula_ref="v2.moe_routed",
            scope=Scope(role="target" if "target" in rk or "colocated" in rk else "draft",
                        phase="steady-decode", rank=rk),
            evidence_refs=["ev-qwen36-main-config"],
            reason=("; ".join(blk.get("reasons", [])) or None)
                if blk["status"] in ("unknown", "oom", "feasible-conditional")
                else None,
            assumption_refs=(["假设预算输入（hypothesis）：结论为条件可行，"
                              "非实测可部署"] if conditional else []))
        m[f"M06_memory_{rk}_detail"] = {
            "steady_bytes": blk.get("steady_bytes"),
            "transient_peak_bytes": blk.get("transient_peak_bytes"),
            "peak_complete": blk.get("peak_complete"),
            "unknown_components": blk.get("unknown_components", []),
            "allocatable_kind": blk.get("allocatable_kind", "unknown")}
    # M11 五类字节（target decode）
    t = totals(base_rows)
    m["M11_bytes_target_decode"] = {
        k: metric_from(v if not isinstance(v, float) else v,
                       "B" if k != "flops" else "FLOPs",
                       "v2.gemm" if k == "flops" else "v2.kv_rw",
                       statistic=Statistic.EXPECTATION if k == "weight_read_expected"
                       else Statistic.EXACT, scope=sc, evidence=(Evidence.COUNTED,))
        for k, v in t.items() if k != "rows"}
    # M19 通信
    if msg_bytes.get("d2t") or msg_bytes.get("t2d"):
        m["M19_msg_bytes"] = {
            "d2t": metric_from(msg_bytes["d2t"], "B/round", FR.F_DFLASH_MSG,
                               statistic=Statistic.EXPECTATION, scope=sc,
                               evidence=(Evidence.SOURCE_CHECKED,),
                               assumptions=["协议绑定 z-lab/dflash pin"]),
            "t2d": metric_from(msg_bytes["t2d"], "B/round", FR.F_DFLASH_MSG,
                               statistic=Statistic.EXPECTATION, scope=sc,
                               evidence=(Evidence.SOURCE_CHECKED,)),
        }
    # M07 状态
    m["M07_support_summary"] = {
        "capability_checks": {k: v["status"] for k, v in cap.checks.items()},
        "structure_complete": model.structure_complete,
        "result_kind": "analytical-estimate",
    }
    return m


def _eligibility(cap: CapabilityReport) -> bool:
    """比较资格：结构完整 + 无 OOM + 容量非 unknown/条件（§7.2；R03b：
    假设预算（approximated/feasible-conditional）不进实际部署排名）。"""
    for cid, chk in cap.checks.items():
        if cid == "model_structure_complete" and chk["status"] != "supported":
            return False
        if cid.startswith("memory_") and chk["status"] in (
                "unsupported", "unknown", "approximated"):
            return False
    return True


def _moe_union_summary(rows) -> Optional[Dict[str, Any]]:
    routed = [r for r in rows if r.op == "moe_routed"]
    if not routed:
        return None
    first = routed[0]
    return metric_from(first.weight_read_expected or 0.0, "B/layer",
                       FR.F_MOE_UNION, statistic=Statistic.EXPECTATION,
                       scope=Scope(role="target", phase="steady-decode"),
                       evidence=(Evidence.COUNTED, Evidence.ASSUMED),
                       assumptions=["均匀独立路由假设（无 trace）"])


def _assumptions(scenario, plan) -> Dict[str, str]:
    out = {
        "A1_performance": "全部时延为 T0 解析估算（roofline；Q2 首版无实测/拟合）",
        "A2_serial": "串行解析档：阶段和=轮墙钟；无重叠/流水（M3 事件核后扩展）",
        "A3_acceptance": f"接受率为显式假设输入（Q8 hypothesis）：{scenario.acceptance}",
        "A4_routing": "MoE 路由按均匀独立假设；期望并集非精确计数",
        "A5_workload": f"负载假设：B={scenario.workload.batch} ctx={scenario.workload.ctx_tokens}"
                       "（可编辑演示默认值，非用户确认）",
        "A6_budget": "设备容量为标称规格；可分配预算未提供 → feasible 判定 "
                    "unknown（已知对象下界超标称仍判 OOM）；假设预算只产生"
                    "条件结论（R03a/b）",
        "A7_parallel": f"TP{scenario.deployment.target_tp}/EP"
                       f"{scenario.deployment.target_ep} per-rank 配方 "
                       "(vLLM @dc74b8cb pin：EP 关 inter/tp 切分，EP 开整专家"
                       "分置；KV 头 <tp 复制；GDN 头/通道切分)",
    }
    if scenario.cost_profile:
        out["A1_performance"] = (
            f"混合口径：域内算子为实测 profile 校准（{scenario.cost_profile}，"
            "算子级 median 组合+kernel 间隙），域外部分仍为 T0 解析估算；"
            "见 cost.calibration 披露")
    for a in plan.approximations:
        out[f"plan:{a[:24]}"] = a
    return out


def _bundle_error(scenario: ScenarioV2, cap: CapabilityReport, status: str,
                  detail: str) -> ResultBundle:
    manifest = RunManifest(run_id="", status="failed",
                           input_digest=scenario.scenario_digest(),
                           code_revision=code_revision())
    manifest.compute_run_id()
    return ResultBundle(
        manifest=manifest, scenario_snapshot=scenario.to_json(),
        capability=cap.to_json(),
        metrics={"error": Metric.unsupported("none", detail)},
        assumption_dict={"error": status})


def _build_metrics_null_acceptance(acc, t_base, memory, cap, model) -> Dict[str, Any]:
    sc = Scope(role="target", phase="steady-decode", time_basis=TimeBasis.COMMITTED.value)
    reason = acc["reason"] or "接受分布未知"
    return {
        "M01_tpot_analytical": Metric.unknown("ms", reason),
        "M01_tpot_baseline": metric_from(
            t_base, "ms", FR.F_ROOFLINE, statistic=Statistic.EXPECTATION, scope=sc,
            evidence=(Evidence.COUNTED,)),
        "M02_speedup_vs_baseline": Metric.unknown("ratio", f"TPOT 未知：{reason}"),
        "M03_E_N": Metric.unknown("tokens/round", reason),
        "M03_K": metric_from(0, "tokens", "internal", statistic=Statistic.EXACT,
                             scope=sc, evidence=(Evidence.COUNTED,)),
        "M07_support_summary": {
            "capability_checks": {k: v["status"] for k, v in cap.checks.items()},
            "structure_complete": model.structure_complete,
            "result_kind": "analytical-estimate"},
    }
