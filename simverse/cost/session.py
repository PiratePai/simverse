"""One cost evaluation per counted stage; providers change costs, never ledgers.

Groups own disjoint terms. Analytical groups sum existing row costs (no implied
fusion or whole-model roofline). A measured fused group stays a group: its time
is never divided into invented measured row prices.
"""
from __future__ import annotations

from collections import defaultdict
from copy import copy
from dataclasses import replace

from ..contracts import Evidence, Feasibility, Metric, ResultKind, Support, digest
from ..execute.serial import evaluate_execution


def group_name(op):
    if op in {"lm_head", "draft_lm_head", "embed_lookup", "draft_embed"}:
        return "embedding_head"
    if op.startswith(("moe_", "shared_gate")):
        return "moe"
    if op.startswith(("in_proj", "gdn_", "causal_conv", "v_norm", "out_proj")):
        return "gdn"
    if op.startswith(("attn_", "attention", "sparse_attention", "mla_", "q_", "k_", "v_", "o_", "rope")):
        return "attention"
    if op.startswith(("engram_",)):
        return "engram"
    if op.startswith(("compress", "index_", "candidate_", "sparse_index", "swa_")):
        return "cache_index"
    if "norm" in op or op.endswith("_add"):
        return "norm_residual"
    return "runtime_other"


class CostSession:
    """Shared selection/ownership/execution path for analytic, records and profiles."""
    def __init__(self, scenario, model):
        self.scenario, self.model = scenario, model
        self.rows_by_stage = defaultdict(list)
        self.prices = defaultdict(list)
        self.selected_prices = defaultdict(list)
        self.communication = []
        self.links = []
        self.records = self.profile = None
        if scenario.cost_records:
            from .records import RecordPricer
            self.records = RecordPricer(scenario, model)
        if scenario.cost_profile:
            from .calibrated import CalibratedPricer
            from .profile_store import load_profile_set, profile_model_id
            from ..scenario import ScenarioInputError
            try:
                bound_model = profile_model_id(scenario.cost_profile)
            except ValueError as exc:
                raise ScenarioInputError("$.cost_profile", str(exc)) from exc
            if bound_model != model.id:
                raise ScenarioInputError(
                    "$.cost_profile",
                    f"profile {scenario.cost_profile} 绑定历史模型 {bound_model}，禁止用于 {model.id}（跨模型误用）")
            pset = load_profile_set(scenario.cost_profile)
            self.profile = CalibratedPricer(pset, model, scenario)

    def checks(self, cap):
        if self.profile:
            cap.add("cost_profile_domain", "supported" if self.profile.applicable else "approximated",
                    f"{self.scenario.cost_profile}：" + ("设备/精度/TP/形状域匹配" if self.profile.applicable
                    else "域不匹配：" + "；".join(self.profile.domain_reasons)), affected_metrics=["M01_tpot_selected"])

    def rows(self, rows, prices, device, quant):
        chosen = self.records.rows(rows, prices, device, quant) if self.records else prices
        for row, price, selected in zip(rows, prices, chosen):
            self.rows_by_stage[row.stage].append(row)
            self.prices[row.stage].append(price)
            self.selected_prices[row.stage].append(selected)
        return prices  # analytical path is retained; selection is a cost-only projection

    def collective(self, collective, ranks, dtype, device, stage):
        selected = copy(collective)
        if self.records:
            self.records.collective(selected, ranks, dtype, device, stage)
            selected.record_match = self.records.entries[-1]
        self.communication.append((stage, device.id, collective, selected))
        return collective

    def point_to_point(self, direction, link, payload_bytes, analytic_ms):
        self.links.append((direction, link, payload_bytes, analytic_ms))
        if self.records:
            self.records.point_to_point(direction, link, payload_bytes, analytic_ms)

    def _groups(self):
        groups = []
        for stage, rows in self.rows_by_stage.items():
            buckets = {}
            for row, analytic, selected in zip(rows, self.prices[stage], self.selected_prices[stage]):
                name = group_name(row.op)
                group = buckets.setdefault((row.role, name), dict(
                    group_id=f"{stage}:{row.role}:{name}", stage=stage, role=row.role, group=name,
                    analytic_ms=0.0, selected_ms=0.0, members=[], record_refs=[], fallback_reasons=[],
                    timing_boundary="analytical-owned-terms", source="analytic"))
                group["analytic_ms"] += analytic["t_ms"]
                group["selected_ms"] += selected["t_ms"]
                group["members"].append(f"{row.stage}:{row.layer}:{row.op}")
                group["record_refs"].extend(selected.get("record_refs", []))
                if selected.get("fallback_reason"):
                    group["fallback_reasons"].append(selected["fallback_reason"])
            groups.extend(buckets.values())
        for i, (stage, device, analytic, selected) in enumerate(self.communication):
            groups.append(dict(group_id=f"{stage}:communication:{i}", stage=stage, role="link",
                group="communication", analytic_ms=analytic.t_ms, selected_ms=selected.t_ms,
                members=[f"{stage}:communication:{i}"], device=device,
                primitive=analytic.kind, bytes_on_wire=analytic.bytes_on_wire,
                record_refs=getattr(selected, "record_match", {}).get("record_refs", []),
                fallback_reasons=[getattr(selected, "record_match", {}).get("fallback_reason")]
                    if getattr(selected, "record_match", {}).get("fallback_reason") else [], source="analytic",
                timing_boundary="whole-collective"))
        for direction, link, payload, ms in self.links:
            groups.append(dict(group_id=f"{direction}:point-to-point", stage=direction, role="link",
                group="point_to_point", analytic_ms=ms, selected_ms=ms, members=[f"{direction}:link"],
                source="analytic", record_refs=[], fallback_reasons=["点到点链路无匹配记录"] if self.records else [],
                link=link, payload_bytes=payload, timing_boundary="whole-message"))
        # Display and cost units aggregate equal stage/role/category, not kernel identities.
        combined = {}
        for item in groups:
            key = (item["stage"], item["role"], item["group"], item.get("primitive"))
            if key not in combined:
                combined[key] = item
            else:
                group = combined[key]
                group["analytic_ms"] += item["analytic_ms"]
                group["selected_ms"] += item["selected_ms"]
                if "bytes_on_wire" in group:
                    group["bytes_on_wire"] += item["bytes_on_wire"]
                for name in ("members", "record_refs", "fallback_reasons"):
                    group[name].extend(item[name])
        for group in combined.values():
            for key in ("record_refs", "fallback_reasons"):
                group[key] = list(dict.fromkeys(group[key]))
            if group["record_refs"] or group["selected_ms"] != group["analytic_ms"]:
                group["source"] = "conditional-record+analytic-fallback"
            group["basis"] = "sum-of-owned-terms; no fusion assumption or per-row measured allocation"
        return list(combined.values())

    def finish(self, bundle, plan, stage_costs, e_n, row_price_views, *, timing_gap=None):
        """Evaluate alternate costs using cached stages and the same execution function."""
        sc = self.scenario
        groups = self._groups()
        # Preserve original summation order and numerical anchors. Group displays
        # are projections of these terms, not a new roofline or reduction order.
        analytic_totals = {st: sum(p["t_ms"] for p in prices) + sum(
            a.t_ms for stage, _, a, _ in self.communication if stage == st)
            for st, prices in self.prices.items()}
        selected_totals = {st: sum(p["t_ms"] for p in prices) + sum(
            chosen.t_ms for stage, _, _, chosen in self.communication if stage == st)
            for st, prices in self.selected_prices.items()}
        calibration = None
        if self.profile:
            calibration = dict(profile_set=sc.cost_profile, stages={},
                note="组级选用成本；不向逻辑行分摊实测。原始计数/解析值保留")
            for stage, rows in self.rows_by_stage.items():
                if not rows or rows[0].role != "target" or stage.startswith("draft"):
                    calibration["stages"][stage] = dict(stage=stage, domain_ok=False, selected_ms=None,
                        reasons=["草稿阶段无匹配 profile 域"])
                    continue
                q = 1 if stage == "target_decode" else (plan.block_size if stage == "verify_block" else plan.k_effective + 1)
                cal = self.profile.price_stage(stage, rows, self.prices[stage], sc.workload.batch, sc.workload.ctx_tokens, q)
                comm = sum(a.t_ms for st, _, a, _ in self.communication if st == stage)
                cal["collectives_ms"] = comm
                calibration["stages"][stage] = cal
                if cal["selected_ms"] is not None:
                    owned = [g for g in groups if g["stage"] == stage and g["role"] != "link"]
                    groups = [g for g in groups if g not in owned]
                    provider = self.profile.stage_groups[stage]
                    covered = set()
                    for group in provider["groups"]:
                        if covered.intersection(group["members"]):
                            raise ValueError("同一计价项不得被两个 profile 组拥有")
                        covered.update(group["members"])
                        groups.append(dict(group, group_id=f"{stage}:profile:{group['group']}",
                            stage=stage, role="target", timing_boundary="profile-defined-regions",
                            basis="provider group total; no measured row allocation"))
                    fallback = [m for group in owned for m in group["members"] if m not in covered]
                    groups.append(dict(group_id=f"{stage}:analytic-fallback", stage=stage, role="target",
                        group="analytic_fallback", analytic_ms=provider["fallback_ms"], selected_ms=provider["fallback_ms"],
                        members=fallback, source="analytic", timing_boundary="analytical-owned-terms",
                        basis="uncovered owned terms; no extrapolation"))
                    groups.append(dict(group_id=f"{stage}:profile-boundary-gap", stage=stage, role="target",
                        group="profile_boundary_gap", analytic_ms=0.0, selected_ms=provider["gap_ms"],
                        members=[], source="assumed", timing_boundary="between-profile-regions",
                        basis=cal["gap_policy"]))
                    selected_totals[stage] = cal["selected_ms"] + comm
            stages = calibration["stages"].values()
            calibration["coverage"] = dict(
                rows_covered=sum(s.get("coverage_rows", [0, 0])[0] for s in stages),
                rows_total=sum(s.get("coverage_rows", [0, 0])[1] for s in stages),
                holdout_error_report=self.profile.pset.manifest["holdout"]["error_report"],
                environment=self.profile.pset.manifest["environment"], source_head=self.profile.pset.manifest["source"]["git_head"])
            bundle.cost["calibration"] = calibration
        chosen_steps = dict(stage_costs)
        for stage in chosen_steps:
            counted_stage = "target_decode" if stage == "decode" else stage
            if counted_stage in selected_totals and selected_totals[counted_stage] != analytic_totals[counted_stage]:
                chosen_steps[stage] = selected_totals[counted_stage]
        baseline = selected_totals.get("target_decode", bundle.cost["target_decode"]["t_ms"])
        if chosen_steps != stage_costs:
            selected_round, selected_sim = evaluate_execution(
                plan, chosen_steps, e_n, sc.deployment, timing_gap=timing_gap)
            execution = {**bundle.execution, "rounds": [selected_round.to_json()]}
            tpot, round_ms = selected_round.tpot_ms, selected_round.t_round_ms
        else:
            execution, selected_sim = bundle.execution, bundle.cost.get("pipeline_sim")
            old_tpot, old_round = bundle.metrics.get("M01_tpot_analytical"), bundle.metrics.get("M04_round_ms")
            tpot = old_tpot.value if isinstance(old_tpot, Metric) else None
            round_ms = old_round.value if isinstance(old_round, Metric) else None
        selection = dict(source="records" if self.records else "profile" if self.profile else "analytic",
            source_id=sc.cost_records or sc.cost_profile, granularity="cost-group",
            stage_costs_ms=chosen_steps, baseline_ms=baseline,
            row_price_semantics="diagnostic analytical terms or independently matched GEMM; not allocated fused timings")
        if execution != bundle.execution:
            selection["execution"] = execution
        if selected_sim != bundle.cost.get("pipeline_sim"):
            selection["pipeline_sim"] = selected_sim
        bundle.cost["selection"] = selection
        bundle.cost["groups"] = groups
        unknown = [{"id": k, "time_ms": None, "reason": v["reason"]}
                   for k, v in bundle.capability["checks"].items()
                   if k == "stage_cost_complete" and v["status"] != "supported"]
        for item in unknown:
            groups.append(dict(group_id="unknown:" + item["id"], stage="unpriced_runtime", role="system",
                group="unpriced_runtime", analytic_ms=None, selected_ms=None, members=[], source="unknown", reason=item["reason"]))
        record_report = None
        note = "成本组为已计价项之和；保留未知阶段，不等于完整部署实测"
        if self.records:
            record_report = self.records.report()
            record_report["unpriced_components"] = unknown
            if unknown:
                record_report["prerequisites"].append("未计价方法阶段仍为unknown，不在覆盖分母内：先确认Ascend实现，再采接受/回退/hidden提取，不按0ms忽略")
            base_rows = self.rows_by_stage.get("target_decode", [])
            base_prices = self.selected_prices.get("target_decode", [])
            selection["selected_row_prices"], selection["selected_roofline_points"] = (
                row_price_views(base_rows, base_prices))
            bundle.cost["record_cost"] = record_report
            note = (f"{record_report['label']}：{record_report['matched']}/{record_report['total']} 成本实例命中；"
                    "其余按原解析回退；记录含host/sync且身份不完备，仅作条件对照，不是生产校准或E2E实测")
            if unknown:
                note += "；仅包含已计价阶段，方法运行时阶段尚缺时长"
            bundle.capability["checks"]["archived_cost_records"] = dict(
                status="approximated" if record_report["matched"] else "unknown", reason=note,
                affected_metrics=["M01_tpot_selected", "M02_speedup_selected", "M04_round_selected"])
            bundle.assumption_dict["archived_cost_records"] = note
            bundle.formula_dict["record-cost:" + sc.cost_records] = dict(
                expr="selected = matching archived latency + unchanged analytical fallback; same execution plan",
                source="sha256:" + self.records.content_digest, applies="conditional archived cost experiment", version="1")
            bundle.manifest.cost_digest = digest({"base": bundle.manifest.cost_digest, "cost_records": self.records.content_digest})
            bundle.manifest.compute_run_id()
        # Chosen metrics use the same names for every provider.
        values = {"M01_tpot_selected": ("M01_tpot_analytical", tpot),
                  "M01_baseline_selected": ("M01_tpot_baseline", baseline),
                  "M02_speedup_selected": ("M02_speedup_vs_baseline", baseline / tpot if tpot else None),
                  "M04_round_selected": ("M04_round_ms", round_ms)}
        active = bool(record_report and record_report["matched"]) or bool(calibration and any(
            s.get("selected_ms") is not None for s in calibration["stages"].values()))
        selection["requested_source"] = selection["source"]
        selection["matched"] = record_report["matched"] if record_report else (
            calibration["coverage"]["rows_covered"] if calibration else 0)
        selection["applied"] = active
        selection["stage_aliases"] = {"decode":"target_decode"} if sc.method_id == "baseline" else {}
        selection["rounding_tolerance_ms"] = 1e-6 if calibration else 1e-12
        if not active:
            selection["source"] = "analytic"
        for key, (original, value) in values.items():
            old = bundle.metrics.get(original)
            if not isinstance(old, Metric):
                continue
            met = replace(old, value=value) if active else old
            if active:
                met.support, met.feasibility = (Support.APPROXIMATED if value is not None else Support.UNKNOWN), Feasibility.UNKNOWN
                met.reason = "；".join(filter(None, [old.reason, note]))
                met.evidence_refs = [Evidence.ASSUMED]
                if self.records:
                    met.result_kind = ResultKind.ANALYTICAL_ESTIMATE
                    met.formula_ref = "record-cost:" + sc.cost_records
                    met.input_refs = ["sha256:" + self.records.content_digest]
                else:
                    met.result_kind = ResultKind.MEASURED_COMPOSITE
                    met.evidence_refs = [Evidence.MEASURED, Evidence.ASSUMED]
                    met.formula_ref = "profile:" + sc.cost_profile
                    met.assumption_refs = ["组级实测 median 与解析回退组合，非整步实测；不向逻辑行分摊实测",
                                           "间隙仅计原采样区间边界，覆盖/留出见 cost.calibration"]
            bundle.metrics[key] = met
        if selected_sim:
            from ..contracts import Scope, Statistic, metric_from
            bundle.metrics["M18_selected_cohort_period_ms"] = {
                k: metric_from(v, "ms", "internal", statistic=Statistic.SAMPLE,
                    scope=Scope(role="system"), evidence=(Evidence.COUNTED,))
                for k,v in selected_sim["cohort_periods_ms"].items()}
            if selected_sim.get("timing_complete") is False:
                for metric in bundle.metrics["M18_selected_cohort_period_ms"].values():
                    metric.support = Support.APPROXIMATED
                    metric.feasibility = Feasibility.UNKNOWN
                    metric.reason = selected_sim["timing_reason"]
        summary = bundle.metrics.get("M07_support_summary")
        if summary:
            summary["capability_checks"] = {k:v["status"] for k,v in bundle.capability["checks"].items()}
        return bundle
