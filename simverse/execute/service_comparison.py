"""Pair already-reduced service traces and keep evidence gates explicit."""


def comparison_metrics(summary):
    goodput = summary["goodput"] or {}
    return {"completed_token_per_s": summary["completed_token_per_s"],
            "event_token_per_s": summary["event_token_per_s"],
            "ttft_p50_ms": summary["ttft"]["p50_ms"], "ttft_p95_ms": summary["ttft"]["p95_ms"],
            "goodput_requests_per_s": goodput.get("requests_per_s"), "goodput_token_per_s": goodput.get("token_per_s")}


def memory_coverage(ranks):
    observed = sum(
        rank["device_memory"]["complete"] and all(v["peak_bytes"] is not None for v in rank["memory"].values())
        and rank["requested_memory_bytes"]["value"] is not None and rank["kv_cache_budget_bytes"]["value"] is not None
        for rank in ranks
    )
    return {"expected_count": len(ranks), "observed_count": observed, "complete": bool(ranks) and observed == len(ranks)}


def compare_modes(plan, modes, outputs, missing_modes):
    baseline = modes.get("baseline")
    pairs, output_mismatches = {}, {}
    for method in (*modes, *missing_modes):
        if method == "baseline":
            continue
        candidate = modes.get(method)
        if baseline is None or candidate is None:
            missing = [name for name in ("baseline", method) if name in missing_modes]
            pairs[method] = {"max_arrival_delta_ns": None, "service_metrics_comparable": False,
                             "eligible": False, "hardware_eligible": False, "hardware_reason": "缺少配对采集记录",
                             "checks": {"mode_availability": "缺少配对采集记录"}, "missing_modes": missing,
                             "candidate_minus_baseline": None}
            output_mismatches[method] = None
            continue
        same_window = baseline["window"] == candidate["window"]
        mismatches = []
        pair_drift = 0 if same_window else None
        for rid, (arrival, result) in outputs[method].items():
            baseline_arrival, baseline_result = outputs["baseline"][rid]
            if pair_drift is not None:
                pair_drift = max(pair_drift, abs(arrival - baseline_arrival))
            if result != baseline_result:
                mismatches.append(rid)
        output_mismatches[method] = mismatches
        runs = (baseline, candidate)
        same_identity = baseline["source_kind"] == candidate["source_kind"] and baseline["runtime"] == candidate["runtime"]
        arrivals_match = all(row["arrivals_within_tolerance"] for row in runs) and (
            pair_drift is None or pair_drift <= plan["arrival_tolerance_ns"]
        )
        checks = {
            "source_and_runtime": None if same_identity else "采集来源或运行时身份不匹配",
            "arrival_alignment": None if arrivals_match else "到达偏差超过计划门槛",
            "window_and_clock": None if same_window else "观测窗口、时钟或时间基准不匹配",
            "output_consistency": None if not mismatches else "输出 token、结束状态或停止原因不匹配",
            "successful_samples": None if all(row["metrics"]["ttft_p95_ms"] is not None for row in runs)
            else "缺少窗口内成功请求的样本",
        }
        comparable = all(reason is None for reason in checks.values())
        residuals = [row["stage_reconciliation"]["attribution"]["unattributed_ns"] for row in runs]
        limit = plan.get("max_unattributed_ns")
        if limit is None:
            residual_reason = "计划未登记未归属时长门槛"
        elif None in residuals:
            residual_reason = "未归属时长未知"
        elif max(residuals) > limit:
            residual_reason = "未归属时长超过计划门槛"
        else:
            residual_reason = None
        checks.update({
            "stage_coverage": None if all(row["stage_reconciliation"]["phase_coverage_complete"] for row in runs)
            else "请求或运行时阶段覆盖不完整",
            "round_commits": candidate["round_commit_reason"],
            "memory_coverage": None if all(row["memory_coverage"]["complete"] for row in runs)
            else "逐 rank 显存或预算证据不完整",
            "unattributed_limit": residual_reason,
        })
        eligible = all(reason is None for reason in checks.values())
        if not all(row["source_kind"] == "captured" for row in runs):
            hardware_reason = "两路须均为硬件采集记录"
        elif not eligible:
            hardware_reason = "同负载对账资格未通过"
        elif not all(row["preflight_budget_met"] for row in runs):
            hardware_reason = "缺少通过前置显存门槛的证据"
        else:
            hardware_reason = None
        deltas = {}
        for key, value in candidate["metrics"].items():
            base = baseline["metrics"][key]
            deltas[key] = value - base if comparable and base is not None and value is not None else None
        pairs[method] = {"max_arrival_delta_ns": pair_drift, "service_metrics_comparable": comparable,
                         "eligible": eligible, "hardware_eligible": hardware_reason is None,
                         "hardware_reason": hardware_reason, "checks": checks, "candidate_minus_baseline": deltas}
    return {"eligible": all(pair["eligible"] for pair in pairs.values()),
            "hardware_eligible": all(pair["hardware_eligible"] for pair in pairs.values()),
            "max_unattributed_ns": plan.get("max_unattributed_ns"), "pairs": pairs}, output_mismatches
