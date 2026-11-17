"""Record-scoped latency replacement: independent values, boundaries and failures."""
import copy
import csv
import importlib.util
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from simverse.contracts import digest, js_restore
from simverse.cost.records import ROOT, RecordPricer, curve_lookup, load
from simverse.cost.profile_store import ProfileIntegrityError, PROFILE_SETS
from simverse.count.work import OpRow
from simverse.devices import DEVICES
from simverse.lowering.sharding import CollectiveCost
from simverse.model_ir import build_catalog
from simverse.pipeline import run_v2
from simverse.scenario import ScenarioV2
from simverse.webapp import app

Q16 = dict(dense_w=16, expert_w=16, act=16, kv=16, state=32)


def scenario(sid=None, tp=2, method="baseline", **over):
    data = dict(method_id=method, k=15, block_size=16, quant=Q16.copy(), draft_quant=Q16.copy(),
                workload=dict(batch=1, ctx_tokens=4096), cost_records=sid,
                deployment=dict(mode="baseline" if method == "baseline" else "colocated",
                                target_device="ascend_910b3", target_tp=tp))
    data.update(over)
    return ScenarioV2(**data)


def test_distillation_replays_all_sources_and_frozen_cost_hashes():
    spec = importlib.util.spec_from_file_location("distill_records", ROOT / "tools/distill_ascend_cost_records.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    values = module.distill()
    for sid in values:
        actual, _ = load(sid)
        assert actual == values[sid]
        assert actual["admission"] == "conditional-only"
        assert actual["measured_profile_admitted"] is False
        assert sid not in PROFILE_SETS
    assert load("ascend-r29-hccl")[1] == "db7b8278063dc03485afac51b598a2c6c3605a2a7cf6cbccfaa6d78e6d2d4def"
    assert load("ascend-r28-gemm")[1] == "5f5a35982ec47243158d2809ba727fbf0b94f1893282b5eb3826e6416c2688f2"


def test_curve_values_use_rank_samples_and_logical_input_not_bus_bandwidth():
    data, _ = load("ascend-r29-hccl")
    assert len(data["curves"]) == 5
    assert all(len(points) == 10 for points in data["curves"].values())
    expected = max(json.loads((ROOT / f"docs/evidence/ascend-910b3-r29/raw/collectives_rank{rank}.json")
                             .read_text(encoding="utf-8"))["results"][5]["median_ms"] for rank in (0, 1))
    ms, refs, mode = curve_lookup(data, "all_reduce", 4096, 2, "bfloat16", "ascend_910b3")
    assert ms == expected
    assert refs == ["r29:all_reduce:4096"] and mode == "exact"
    # AllGather key is INPUT 256 MiB, output 512 MiB. Do not infer latency from
    # the archived (incorrect) normalized busbw of 8.863 GB/s.
    ms, _, _ = curve_lookup(data, "all_gather", 2**28, 2, "bfloat16", "ascend_910b3")
    assert ms == pytest.approx(15.146026)
    assert data["curves"]["all_gather"][-1]["output_bytes"] == 2**29


def test_curve_interpolation_is_linear_in_bytes_with_refs_and_no_extrapolation():
    data, _ = load("ascend-r29-hccl")
    lo, hi = data["curves"]["all_reduce"][:2]
    midpoint = (lo["input_bytes"] + hi["input_bytes"]) // 2
    ms, refs, mode = curve_lookup(data, "all_reduce", midpoint, 2, "bfloat16", "ascend_910b3")
    assert ms == pytest.approx((lo["selected_ms"] + hi["selected_ms"]) / 2)
    assert refs == [lo["id"], hi["id"]] and mode == "interpolated"
    for n in (1, 2**28 + 1):
        assert curve_lookup(data, "all_reduce", n, 2, "bfloat16", "ascend_910b3")[2] == "bytes-out-of-range"
    assert curve_lookup(data, "all_reduce", 4096, 8, "bfloat16", "ascend_910b3")[2] == "rank-domain"
    assert curve_lookup(data, "all_reduce", 4096, 2, "fp8", "ascend_910b3")[2] == "dtype-mismatch"
    assert curve_lookup(data, "all_reduce", 4096, 2, "bfloat16", "h100_sxm")[2] == "device-mismatch"


def test_whole_collective_cost_is_replaced_once_and_ambiguous_primitives_fall_back():
    sc = scenario("ascend-r29-hccl")
    pricer = RecordPricer(sc, build_catalog()["models"][sc.model_id])
    c = CollectiveCost("allreduce", 4096, 99., "fixture")
    pricer.collective(c, 2, "bfloat16", DEVICES["ascend_910b3"], "target_decode")
    exact = curve_lookup(pricer.data, "all_reduce", 4096, 2, "bfloat16", "ascend_910b3")[0]
    assert c.t_ms == exact  # no +99, no extra fixed latency
    for kind in ("logits_gather", "ep_dispatch", "ep_combine"):
        c = CollectiveCost(kind, 4096, 99., "fixture")
        pricer.collective(c, 2, "bfloat16", DEVICES["ascend_910b3"], "verify")
        assert c.t_ms == 99.
        assert pricer.entries[-1]["fallback_code"] == "primitive-unresolved"


def test_pure_gemm_matching_is_op_shape_dtype_and_role_scoped():
    sc = scenario("ascend-r28-gemm", tp=8)
    pricer = RecordPricer(sc, build_catalog()["models"][sc.model_id])
    row = OpRow("target_decode", 0, "in_proj_qkv", flops=4194304,
                shape="[1,2048]@[2048,1024]")
    price = dict(t_ms=99., cost_basis="fixture")
    got = pricer.rows([row], [price], DEVICES["ascend_910b3"], Q16)[0]
    assert got["t_ms"] == pytest.approx(0.01874302513897419)
    assert got["analytic_ms"] == 99.
    assert price["t_ms"] == 99.  # original input not mutated
    row.stage = "draft_block"
    assert pricer.rows([row], [price], DEVICES["ascend_910b3"], Q16)[0]["t_ms"] == 99.
    assert pricer.entries[-1]["fallback_code"] == "model-role-mismatch"
    row.stage, row.shape, row.flops = "verify", "[16,2048]@[2048,1024]", 16 * 4194304
    assert pricer.rows([row], [price], DEVICES["ascend_910b3"], Q16)[0]["t_ms"] == 99.
    assert pricer.entries[-1]["fallback_code"] == "shape-missing"  # M16 not present; no invented interpolation


def test_proxy_and_different_stack_records_never_enter_selected_costs():
    data, _ = load("ascend-r28-gemm")
    assert len(data["operators"]) == 14
    assert not {"moe_shared", "moe_routed", "rope", "attention_core"} & {r["op"] for r in data["operators"]}
    sc = scenario("ascend-r29-hccl")
    pricer = RecordPricer(sc, build_catalog()["models"][sc.model_id])
    row = OpRow("target_decode", 0, "moe_router", flops=1048576,
                shape="[1,2048]@[2048,256]")
    assert pricer.rows([row], [dict(t_ms=4., cost_basis="fixture")], DEVICES["ascend_910b3"], Q16)[0]["t_ms"] == 4.
    assert pricer.entries[-1]["fallback_code"] == "operator-set-missing"
    with pytest.raises(ValueError, match="混用"):
        scenario("ascend-r29-hccl", cost_profile="h200-qwen3-30b-a3b-tiny-op-v1").validate()
    for invalid in ([], 0, "", "ascend-r28-gemm,ascend-r29-hccl"):
        with pytest.raises(ValueError, match="未知成本记录"):
            scenario(invalid).validate()


def test_source_or_derived_cost_tampering_fails_closed(tmp_path):
    sid = "ascend-r29-hccl"
    data, _ = load(sid)
    for source in data["sources"]:
        dest = tmp_path / source["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / source["path"], dest)
    shutil.copytree(ROOT / "assets/cost_records", tmp_path / "assets/cost_records")
    source = tmp_path / data["sources"][0]["path"]
    source.write_bytes(source.read_bytes() + b" ")
    with pytest.raises(ProfileIntegrityError, match="原始来源"):
        load(sid, tmp_path)
    shutil.copyfile(ROOT / data["sources"][0]["path"], source)
    path = tmp_path / "assets/cost_records/ascend_r29_hccl.json"
    data["curves"]["all_reduce"][0]["selected_ms"] = .000001
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ProfileIntegrityError, match="派生成本"):
        load(sid, tmp_path)
    path.unlink()
    with pytest.raises(ProfileIntegrityError, match="缺失/损坏"):
        load(sid, tmp_path)


def test_pipeline_preserves_analytical_counts_memory_and_event_causality():
    plain = run_v2(scenario(method="dflash_v1")).to_json()
    conditional = run_v2(scenario("ascend-r29-hccl", method="dflash_v1")).to_json()
    assert conditional["work"] == plain["work"]
    assert conditional["cost"]["memory"] == plain["cost"]["memory"]
    assert conditional["metrics"]["M01_tpot_analytical"] == plain["metrics"]["M01_tpot_analytical"]
    metric = conditional["metrics"]["M01_record_tpot"]
    assert metric["result_kind"] == "analytical-estimate"
    assert metric["support"] == "approximated" and metric["feasibility"] == "unknown"
    assert metric["value"] != plain["metrics"]["M01_tpot_analytical"]["value"]
    report = conditional["cost"]["record_cost"]
    assert report["matched"] == 172
    assert all(e["kind"] == "communication" for e in report["entries"] if e["record_refs"])
    stages = report["selected_execution"]["rounds"][0]
    assert conditional["metrics"]["M04_record_round_ms"]["value"] == pytest.approx(stages["t_round_ms"])
    for stage in report["stages"].values():
        assert stage["selected_ms"] == pytest.approx(stage["record_ms"] + stage["fallback_ms"])
    assert conditional["manifest"]["run_id"] != plain["manifest"]["run_id"]
    assert report["unpriced_components"] and report["unpriced_components"][0]["time_ms"] is None


def test_all_fallback_does_not_appear_as_record_calibration():
    bundle = run_v2(scenario("ascend-r29-hccl", tp=8)).to_json()
    report = bundle["cost"]["record_cost"]
    assert report["matched"] == 0
    assert "M01_record_tpot" not in bundle["metrics"]
    assert bundle["capability"]["checks"]["archived_cost_records"]["status"] == "unknown"
    assert "rank-domain" in {p["reason_code"] for p in report["measurement_plan"]}
    from simverse.cli import _ops_csv_v2
    exported = list(csv.DictReader(io.StringIO(_ops_csv_v2(bundle))))
    assert exported and all(row["fallback_reason"] for row in exported)
    assert all(row["cost_source"] == "analytic" for row in exported)


def test_plan_is_derived_from_unique_uncovered_cases_and_has_provenance_prerequisites():
    spec = importlib.util.spec_from_file_location("replay_records", ROOT / "tools/replay_ascend_record_costs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.replay() == json.loads((ROOT / "docs/evidence/ascend-record-costs/coverage-plan.json").read_text(encoding="utf-8"))
    report = run_v2(scenario("ascend-r28-gemm", tp=8)).cost["record_cost"]
    assert report["matched"] == 201
    assert report["deployment_ranking_eligible"] is False
    assert len(report["prerequisites"]) == 2
    for plan in report["measurement_plan"]:
        entries = [e for e in report["entries"] if e["fallback_code"] == plan["reason_code"] and e["analytic_ms"] > 0]
        assert plan["instances"] == len(entries)
        assert plan["fallback_analytic_ms"] == pytest.approx(sum(e["analytic_ms"] for e in entries))
        assert len(plan["cases"]) == plan["unique_cases"]
        assert sum(c["instances"] for c in plan["cases"]) == plan["instances"]
    assert all(p["reason_code"] != "device-mismatch" for p in report["measurement_plan"])


def test_api_cli_json_csv_trace_and_coverage_are_consistent(tmp_path):
    sc = scenario("ascend-r28-gemm", tp=8)
    client = app.test_client()
    assert len(client.get("/api/v2/catalog").get_json()["cost_record_sets"]) == 2
    response = client.post("/api/v2/run", json=sc.to_json())
    assert response.status_code == 200
    expected = js_restore(response.get_json())
    source = tmp_path / "scenario.json"
    source.write_text(json.dumps(sc.to_json()), encoding="utf-8")
    out, csv_file, trace, html = (tmp_path / name for name in ("bundle.json", "ops.csv", "trace.csv", "report.html"))
    proc = subprocess.run([sys.executable, "-m", "simverse.cli", "run-v2", str(source),
                           "-o", str(out), "--csv", str(csv_file), "--trace", str(trace), "--html", str(html)],
                          capture_output=True, text=True, encoding="utf-8", cwd=ROOT, timeout=30)
    assert proc.returncode == 0, proc.stderr
    actual = js_restore(json.loads(out.read_text(encoding="utf-8")))
    assert actual["metrics"] == expected["metrics"]
    assert actual["cost"]["record_cost"] == expected["cost"]["record_cost"]
    text = csv_file.read_text(encoding="utf-8")
    rows = list(csv.DictReader(io.StringIO(text)))
    hit = next(row for row in rows if row["op"] == "in_proj_qkv")
    assert float(hit["record_t_ms"]) == pytest.approx(.01874302513897419)
    assert float(hit["t_ms"]) == float(hit["record_t_ms"])
    assert hit["cost_source"] == "archived-observation-assumption"
    assert hit["fallback_reason"] == ""
    assert next(row for row in rows if row["op"] == "gdn_recurrence")["fallback_reason"]
    assert "M01_record_tpot" in html.read_text(encoding="utf-8")
    assert "记录成本覆盖" in html.read_text(encoding="utf-8")
    from simverse.experiments.sweep_v2 import sweep_1d
    point = sweep_1d(sc, "workload.batch", [1])["points"][0]
    assert point["metric_key"] == "M01_tpot_selected"  # canonical selection; legacy alias remains in the bundle
    assert point["tpot_ms"] == expected["metrics"]["M01_record_tpot"]["value"]


def test_ep_variable_splits_and_mtp_explicit_gather_are_different_domains():
    ep = scenario("ascend-r29-hccl", deployment=dict(mode="baseline", target_device="ascend_910b3",
                                                    target_tp=2, target_ep=2))
    entries = run_v2(ep).cost["record_cost"]["entries"]
    ep_rows = [e for e in entries if e["kind"] == "communication" and "dispatch" in e["name"]]
    assert ep_rows and all(e["fallback_code"] == "primitive-unresolved" for e in ep_rows)
    mtp = run_v2(scenario("ascend-r29-hccl", method="mtp")).cost["record_cost"]
    gathers = [e for e in mtp["entries"] if e["name"] == "mtp_fc_allgather"]
    assert gathers and all(e["record_refs"] and e["lookup"] == "interpolated" for e in gathers)
    dt = scenario("ascend-r29-hccl", method="dflash_v1",
                  deployment=dict(mode="dt_serial", target_device="ascend_910b3",
                                  target_tp=2, draft_device="ascend_910b3"))
    links = [e for e in run_v2(dt).cost["record_cost"]["entries"] if e["kind"] == "link"]
    assert len(links) == 2 and all(e["fallback_code"] == "point-to-point-uncovered" for e in links)
