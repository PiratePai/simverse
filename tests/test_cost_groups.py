"""Single-pass, ownership and execution regressions for canonical cost groups."""
import csv
import importlib.util
import io
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from simverse import pipeline
from simverse.contracts import js_restore
from simverse.cost.session import CostSession
from simverse.count.work import OpRow
from simverse.devices import DEVICES
from simverse.model_ir import build_catalog
from simverse.pipeline import run_v2
from simverse.scenario import ScenarioV2

ROOT = Path(__file__).resolve().parents[1]
Q16 = dict(dense_w=16, expert_w=16, act=16, kv=16, state=32)

def ascend(tp=2, method="dflash_v1", mode=None, sid="ascend-r29-hccl", **kwargs):
    return ScenarioV2(cost_records=sid, method_id=method, k=3, quant=Q16, draft_quant=Q16,
        deployment=dict(mode=mode or ("baseline" if method == "baseline" else "colocated"),
                        target_device="ascend_910b3", target_tp=tp, draft_device="ascend_910b3"),
        workload=dict(batch=1, ctx_tokens=4096), **kwargs)


@pytest.mark.parametrize("sid", [None, "ascend-r29-hccl"])
def test_one_count_capacity_and_price_pass(monkeypatch, sid):
    counts = Counter()
    work, memory, price = pipeline.build_model_work, pipeline._memory_block, pipeline.price_rows
    def counted_work(*args, **kwargs):
        counts["count:" + args[1]] += 1
        return work(*args, **kwargs)
    def counted_memory(*args, **kwargs):
        counts["memory"] += 1
        return memory(*args, **kwargs)
    def counted_price(rows, *args, **kwargs):
        counts["price:" + rows[0].stage] += 1
        return price(rows, *args, **kwargs)
    monkeypatch.setattr(pipeline, "build_model_work", counted_work)
    monkeypatch.setattr(pipeline, "_memory_block", counted_memory)
    monkeypatch.setattr(pipeline, "price_rows", counted_price)
    run_v2(ascend(sid=sid))
    assert counts == {"count:target_decode":1, "count:verify_block":1, "memory":1,
                      "price:target_decode":1, "price:verify_block":1, "price:draft_block":1}


@pytest.mark.parametrize("method", ["baseline", "dflash_v1", "mtp"])
def test_groups_partition_rows_and_preserve_stage_totals(method):
    bundle = run_v2(ascend(method=method)).to_json()
    groups = bundle["cost"]["groups"]
    rows = [r for stage in bundle["work"].values() if isinstance(stage,dict) for r in stage.get("rows", [])]
    expected = Counter(f"{r['stage']}:{r['layer']}:{r['op']}" for r in rows)
    actual = Counter(member for group in groups for member in group["members"] if member in expected)
    assert actual == expected and all(n == 1 for n in actual.values())
    totals = defaultdict(float)
    for group in groups:
        if group["selected_ms"] is not None: totals[group["stage"]] += group["selected_ms"]
    for stage, value in bundle["cost"]["selection"]["stage_costs_ms"].items():
        counted = "target_decode" if stage == "decode" else stage
        if value is None:
            assert counted not in totals
        else:
            assert totals[counted] == pytest.approx(value, abs=1e-10)
    assert "M01_record_tpot" not in bundle["metrics"]
    assert bundle["metrics"]["M01_tpot_selected"]["value"] > 0


def test_analytical_group_does_not_imply_fusion_or_new_roofline():
    session = CostSession(ScenarioV2(), build_catalog()["models"]["qwen36_35b_a3b"])
    rows = [OpRow("decode", 0, "attn_norm"), OpRow("decode", 0, "attn_add")]
    # Two opposing resource-bound terms, each max(1,9)+.15*min(1,9).
    session.rows(rows, [{"t_ms":9.15}, {"t_ms":9.15}], DEVICES["h100_sxm"], Q16)
    groups = session._groups()
    assert sum(g["selected_ms"] for g in groups) == pytest.approx(18.3)
    assert sum(g["selected_ms"] for g in groups) != pytest.approx(11.5)  # max(sumC,sumM)+.15*min(...)


def test_profile_fused_ownership_is_kept_without_measured_row_allocations():
    sc = ScenarioV2(model_id="qwen3_30b_a3b", draft_id=None, method_id="baseline",
        cost_profile="h200-qwen3-30b-a3b-tiny-op-v1", quant=Q16, draft_quant=Q16,
        workload=dict(batch=8, ctx_tokens=2048), deployment=dict(mode="baseline", target_device="h200_sxm", target_tp=1))
    b = run_v2(sc, catalog=build_catalog(include_internal=True)).to_json()
    groups = b["cost"]["groups"]
    fused = next(g for g in groups if g["group"] == "attn_pre_proj")
    assert fused["source"] == "measured" and len(fused["members"]) == 240
    assert fused["selected_ms"] == pytest.approx(5.6256)
    owners = [member for g in groups for member in g["members"]]
    assert len(owners) == len(set(owners)) == len(b["work"]["target_decode"]["rows"])
    assert sum(g["selected_ms"] for g in groups) == pytest.approx(b["metrics"]["M01_tpot_selected"]["value"], abs=1e-6)
    assert all("record_refs" not in p for p in b["cost"]["row_prices"])
    assert "selected_row_prices" not in b["cost"]["selection"]


def test_all_fallback_is_analytic_and_unknown_runtime_is_null():
    b = run_v2(ascend(tp=8)).to_json()
    selection = b["cost"]["selection"]
    assert selection["source"] == "analytic" and selection["requested_source"] == "records"
    assert selection["applied"] is False and selection["matched"] == 0
    assert "execution" not in selection and "pipeline_sim" not in selection
    assert "M01_record_tpot" not in b["metrics"]
    unknown = next(g for g in b["cost"]["groups"] if g["group"] == "unpriced_runtime")
    assert unknown["selected_ms"] is None and unknown["analytic_ms"] is None


def test_selected_pipeline_and_serial_boundaries_are_distinct():
    b = run_v2(ascend(mode="dt_pipeline_sim")).to_json()
    selection = b["cost"]["selection"]
    assert selection["execution"] != b["execution"]
    simulation = selection.get("pipeline_sim", b["cost"]["pipeline_sim"])
    serial = selection["execution"]["rounds"][0]
    priced = [event["t_ms"] for event in serial["events"] if event["t_ms"] is not None]
    assert sum(priced) == pytest.approx(serial["t_round_ms"])
    assert serial["timing_complete"] is False
    split = {event["stage"]: event for event in serial["events"]}
    assert split["d2t"]["t_ms"] > 0 and split["d2t"]["start_ms"] is not None
    assert split["commit_rewind"]["t_ms"] is None and split["commit_rewind"]["reason"]
    assert split["hidden_extract"]["t_ms"] is None and split["hidden_extract"]["reason"]
    assert split["t2d"]["t_ms"] == b["cost"]["link"]["t_t2d_ms"] > 0
    assert split["t2d"]["start_ms"] is None
    assert split["t2d"]["start_lower_bound_ms"] == pytest.approx(
        sum(split[k]["t_ms"] for k in ("draft_block", "d2t", "verify_block")))
    assert simulation["timing_complete"] is False
    assert simulation["excluded_stages"] == ["commit_rewind", "hidden_extract"]
    assert b["metrics"]["M04_round_selected"]["value"] == serial["t_round_ms"]
    for cohort, period in simulation["cohort_periods_ms"].items():
        metric = b["metrics"]["M18_selected_cohort_period_ms"][cohort]
        assert metric["value"] == period and metric["support"] == "approximated"
        assert metric["reason"] == simulation["timing_reason"]
    assert simulation["violations"] == []
    links = [g for g in b["cost"]["groups"] if g["group"] == "point_to_point"]
    assert {g["stage"] for g in links} == {"d2t", "t2d"}


def test_cost_group_and_historic_anchors():
    commands = [
        ("tools/replay_cost_groups.py",),
        ("tools/replay_qwen36_mtp.py", "--check"),
        ("tools/replay_cost_integrity_r27.py", "--check"),
    ]
    for script in commands:
        result = subprocess.run([sys.executable, "-X", "utf8", *script], cwd=ROOT,
                                capture_output=True, text=True, encoding="utf-8", timeout=60)
        assert result.returncode == 0, result.stderr or result.stdout


def test_group_csv_api_cli_and_trace_share_one_selection(tmp_path):
    from simverse.webapp import app
    from simverse.cli import _groups_csv_v2, _trace_csv_v2
    sc = ascend()
    bundle = run_v2(sc).to_json()
    web = js_restore(app.test_client().post("/api/v2/run", json=sc.to_json()).get_json())
    assert web["cost"]["groups"] == bundle["cost"]["groups"]
    exported = list(csv.DictReader(io.StringIO(_groups_csv_v2(bundle))))
    for row, group in zip(exported, bundle["cost"]["groups"]):
        assert row["group"] == group["group"]
        assert (float(row["selected_ms"]) if row["selected_ms"] else None) == group["selected_ms"]
    trace = list(csv.DictReader(io.StringIO(_trace_csv_v2(bundle))))
    assert sum(float(row["t_ms"]) for row in trace if row["t_ms"]) == pytest.approx(
        bundle["metrics"]["M04_round_selected"]["value"])
    assert {r["stage"] for r in trace if not r["t_ms"] and r["reason"]} == {
        "commit_rewind", "hidden_extract"}
    assert next(r for r in trace if r["stage"] == "t2d")["t_ms"] == "0.0"
    path, out = tmp_path / "scenario.json", tmp_path / "groups.csv"
    path.write_text(json.dumps(sc.to_json()), encoding="utf-8")
    proc = subprocess.run([sys.executable,"-m","simverse.cli","run-v2",str(path),"--groups-csv",str(out)],
                          cwd=ROOT,capture_output=True,text=True,encoding="utf-8",timeout=30)
    assert proc.returncode == 0, proc.stderr
    assert list(csv.DictReader(io.StringIO(out.read_text(encoding="utf-8")))) == exported


@pytest.mark.parametrize(("canonical", "legacy", "expected"), [
    (True, True, 1.0),
    (False, True, 2.0),
    (False, False, 3.0),
])
def test_export_v2_trace_reads_legacy_record_selection(tmp_path, canonical, legacy, expected):
    def execution(t_ms):
        return {"rounds": [{"events": [{"stage": "decode", "t_ms": t_ms}]}]}

    cost = {}
    if canonical:
        cost["selection"] = {"execution": execution(1.0)}
    if legacy:
        cost["record_cost"] = {"selected_execution": execution(2.0), "matched": 1}
    result = {"schema_version": "0.2", "cost": cost, "execution": execution(3.0)}
    path, out = tmp_path / "run.json", tmp_path / "trace.csv"
    path.write_text(json.dumps(result), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "simverse.cli", "export-v2", str(path), "--trace", str(out)],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert proc.returncode == 0, proc.stderr
    with out.open(encoding="utf-8") as stream:
        [event] = list(csv.DictReader(stream))
    assert float(event["t_ms"]) == expected


def test_analytic_pipeline_simulation_has_one_serialized_copy():
    bundle = run_v2(ScenarioV2(deployment={"mode": "dt_pipeline_sim"})).to_json()
    cost = bundle["cost"]
    assert cost["pipeline_sim"]["timing_complete"] is False
    assert "pipeline_sim" not in cost["selection"]
    assert "execution" not in cost["selection"]
    assert float(bundle["metrics"]["M04_round_selected"]["value"]) == (
        bundle["execution"]["rounds"][0]["t_round_ms"])
