"""Inherited device parameters are analytical inputs, never NPU measurements."""
import ast
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from simverse.contracts import CapabilityReport, js_restore
from simverse.cost.legacy import roofline_ms
from simverse.cost.t0 import price_row
from simverse.count.work import OpRow
from simverse.devices import DEVICES
from simverse.pipeline import _device_snapshot, _eligibility, run_v2
from simverse.scenario import DeploymentSpec, ScenarioV2
from simverse.webapp import app

ROOT = Path(__file__).resolve().parents[1]
Q16 = dict(dense_w=16, expert_w=16, act=16, kv=16, state=32)


def scenario(method="baseline", **over):
    data = dict(method_id=method, quant=Q16.copy(), draft_quant=Q16.copy(),
                deployment=dict(mode="baseline" if method == "baseline" else "colocated",
                                target_device="ascend_910b3", target_tp=2),
                workload=dict(batch=1, ctx_tokens=128))
    data.update(over)
    return ScenarioV2(**data)


def test_source_snapshot_replays_mapping_and_values_without_imports():
    folder = ROOT / "assets/identity/devices/ascend_910b3"
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["commit"] == "6a8acce825b19591a6331c20a3b16bf6b7d01661"
    assert [f["sha256"] for f in manifest["files"]] == [
        "cbc9b53b132025de794d52e80d53ba805f2005aa07b49cd6b00cfb3ed62e23ef",
        "a5d6bcef54caa10e0166f1e40376dc23f263dfe1574693266714a19e2a98325b"]
    for item in manifest["files"]:
        assert hashlib.sha256((folder / item["path"]).read_bytes()).hexdigest() == item["sha256"]
    tree = ast.parse((folder / "npu_params.py").read_text(encoding="utf-8"))
    table = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "npu_params" for t in n.targets))
    source, dev = table["ascend_910B_64G"], DEVICES["ascend_910b3"]
    assert dev.peak("fp16") == source["cube_ops"] == 249e12
    assert dev.peak("fp32") == source["vector_ops"] == 31e12
    assert dev.hbm_bw == source["bandwidth"] == 1.6e12
    assert dev.mem_capacity == source["memory_capacity"] == 64e9
    assert dev.mfu_cap == source["mfu"] == .2
    assert dev.mbu_cap == source["mbu"] == .2
    assert '"hardware": "ascend_910B_64G"' in (folder / "h3_pipeline_sim.py").read_text(encoding="utf-8")
    assert dev.sm_count is None
    with pytest.raises(ValueError, match="fp8"):
        dev.peak("fp8")


def test_independent_roofline_anchor_and_vector_route():
    dev = DEVICES["ascend_910b3"]
    # 49.8 GF / (249 TF/s * .2) = 1 ms; .32 GB / (.32 TB/s) = 1 ms.
    assert roofline_ms(49_800_000_000, 320_000_000, dev, "fp16")["t_ms"] == pytest.approx(1.15)
    row = OpRow(stage="test", layer=0, op="rope", flops=6_200_000_000)
    price = price_row(row, dev, "fp16", "fp16")
    assert price["compute_dtype"] == "fp32"
    assert price["t_comp_ms"] == pytest.approx(1.0)
    assert price["bandwidth_tb_s"] == pytest.approx(.32)
    assert "非实测" in price["cost_basis"]


def test_qwen_baseline_and_dflash_api_run_with_inherited_device():
    client = app.test_client()
    catalog = client.get("/api/v2/catalog").get_json()
    assert catalog["devices"]["ascend_910b3"]["backend"] == "ascend"
    assert list(DEVICES)[0] == "h100_sxm"  # do not change legacy default
    for method in ("baseline", "dflash_v1"):
        sc = scenario(method)
        response = client.post("/api/v2/run", json=sc.to_json())
        assert response.status_code == 200, response.data
        result = js_restore(response.get_json())
        assert result["metrics"]["M01_tpot_analytical"]["value"] > 0
        assert result["metrics"]["M01_tpot_analytical"]["support"] == "approximated"
        assert result["metrics"]["M01_tpot_analytical"]["feasibility"] == "unknown"
        assert result["metrics"]["M02_speedup_vs_baseline"]["feasibility"] == "unknown"
        assert result["capability"]["checks"]["ascend_device_recipe"]["status"] == "unknown"
        json.dumps(result, allow_nan=False)


def test_npu_precision_and_v41_are_explicitly_rejected():
    with pytest.raises(ValueError, match="BF16/FP16"):
        scenario(quant=dict(dense_w=8, expert_w=4, act=8, kv=8, state=32)).validate()
    with pytest.raises(ValueError, match="draft"):
        scenario("dflash_v1", draft_quant=dict(dense_w=8, expert_w=8, act=8, kv=8, state=32)).validate()
    with pytest.raises(ValueError, match="V4.1 Ascend"):
        scenario(model_id="deepseek_v41_flash").validate()


def test_draft_device_only_used_in_active_role():
    sc = scenario(deployment=dict(mode="baseline", target_device="h100_sxm", target_tp=2,
                                  draft_device="ascend_910b3"))
    sc.quant = dict(dense_w=8, expert_w=4, act=8, kv=8, state=32)
    sc.validate()  # inactive draft must not constrain the target
    sc = scenario("dflash_v1", deployment=dict(mode="dt_serial", target_device="h100_sxm",
                                              target_tp=2, draft_device="ascend_910b3"))
    bundle = run_v2(sc)
    assert bundle.capability["checks"]["ascend_device_recipe"]["status"] == "unknown"


def test_inherited_npu_recipe_alone_blocks_ranking_and_budget_stays_unknown():
    cap = CapabilityReport()
    cap.add("ascend_device_recipe", "unknown", "not calibrated")
    assert not _eligibility(cap)
    bundle = run_v2(scenario()).to_json()
    memory = bundle["capability"]["checks"]
    assert any(k.startswith("memory_") and v["status"] == "unknown" for k, v in memory.items())
    assert "ascend-910b3-r29" not in (bundle.get("evidence_refs") or [])


def test_mbu_change_is_in_run_identity_without_changing_gpu_defaults(monkeypatch):
    dep = DeploymentSpec(target_device="ascend_910b3")
    original = _device_snapshot(dep)
    monkeypatch.setitem(DEVICES, "ascend_910b3", replace(DEVICES["ascend_910b3"], mbu_cap=.3))
    assert original != _device_snapshot(dep)
    assert DEVICES["h100_sxm"].mbu_cap == 1.0
