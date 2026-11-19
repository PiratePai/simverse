"""Check unchanged cost-group anchors and the attributed CUDA KV migration."""
from __future__ import annotations
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simverse.contracts import digest
from simverse.model_ir import build_catalog
from simverse.pipeline import run_v2
from simverse.scenario import ScenarioV2

EVIDENCE = ROOT / "docs/evidence/cost-groups"
CUDA_KV_CASES = ("qwen-default", "qwen-baseline", "qwen-mtp", "dt-serial", "dt-pipeline")
CANONICAL_METRICS = {
    "/M01_record_tpot": "/M01_tpot_selected",
    "/M01_record_baseline": "/M01_baseline_selected",
    "/M02_record_speedup": "/M02_speedup_selected",
    "/M04_record_round_ms": "/M04_round_selected",
    "/M01_tpot_profile_calibrated": "/M01_tpot_selected",
    "/M01_tpot_baseline_calibrated": "/M01_baseline_selected",
}
UNPRICED_METRICS = {
    "/M04_stage_breakdown_ms/commit_rewind", "/M04_stage_breakdown_ms/hidden_extract",
}
Q16 = dict(dense_w=16, expert_w=16, act=16, kv=16, state=32)

def scenarios():
    cases = [("qwen-default", {}),
             ("qwen-baseline", dict(method_id="baseline", deployment=dict(mode="baseline"))),
             ("qwen-mtp", dict(method_id="mtp", k=3)),
             ("dt-serial", dict(deployment=dict(mode="dt_serial"))),
             ("dt-pipeline", dict(deployment=dict(mode="dt_pipeline_sim"))),
             ("unknown-acceptance", dict(acceptance=dict(kind="mean_only", mean=2))),
             ("v41", dict(model_id="deepseek_v41_flash", method_id="baseline",
                          deployment=dict(mode="baseline"), quant=dict(dense_w=8, expert_w=4, act=16)))]
    for sid, tp, method in [("ascend-r29-hccl", 2, "baseline"), ("ascend-r29-hccl", 8, "baseline"),
                            ("ascend-r29-hccl", 2, "dflash_v1"), ("ascend-r29-hccl", 2, "mtp"),
                            ("ascend-r28-gemm", 8, "baseline"), ("ascend-r28-gemm", 8, "dflash_v1")]:
        cases.append((f"{sid}-tp{tp}-{method}", dict(cost_records=sid, method_id=method, k=3,
                     quant=Q16, draft_quant=Q16, workload=dict(batch=1, ctx_tokens=4096),
                     deployment=dict(mode="baseline" if method == "baseline" else "colocated",
                                     target_device="ascend_910b3", target_tp=tp))))
    for model, sid, tp in [("qwen3_30b_a3b", "h200-qwen3-30b-a3b-tiny-op-v1", 1),
                           ("qwen3_235b_a22b", "h200-qwen3-235b-a22b-op-v1", 8)]:
        cases.append((sid, dict(model_id=model, draft_id=None, method_id="baseline", cost_profile=sid,
                     quant=Q16, draft_quant=Q16, workload=dict(batch=8, ctx_tokens=2048),
                     deployment=dict(mode="baseline", target_device="h200_sxm", target_tp=tp))))
    return cases

def snapshot():
    result = {}
    for name, kw in scenarios():
        sc = ScenarioV2(**kw)
        bundle = run_v2(sc, catalog=build_catalog(include_internal=True) if sc.cost_profile else None).to_json()
        metrics = {}
        def visit(obj, path=""):
            if isinstance(obj, dict):
                if "value" in obj and "unit" in obj:
                    metrics[path] = obj["value"]
                else:
                    for k, v in obj.items(): visit(v, path + "/" + k)
        visit(bundle["metrics"])
        result[name] = {"work": digest(bundle["work"]),
                        "memory": digest(bundle["cost"].get("memory", bundle.get("memory"))),
                        "metrics": metrics}
    return result

def assert_value(name, key, expected, actual):
    assert actual == expected or (expected is not None and actual is not None and
        abs(actual - expected) <= max(1e-10, abs(expected) * 1e-12)), (
            name, key, expected, actual)


def main():
    current = snapshot()
    old_cases = json.loads((EVIDENCE / "baseline.json").read_text(encoding="utf-8"))["cases"]
    cuda = json.loads((EVIDENCE / "cuda-kv-replication.json").read_text(encoding="utf-8"))["cases"]
    assert set(current) == set(old_cases)
    assert set(cuda) == set(CUDA_KV_CASES)
    for name, old in old_cases.items():
        new = current[name]
        if name in cuda:
            anchor = cuda[name]
            assert set(new["metrics"]) == set(anchor["metrics"]), name
            metrics = anchor["metrics"]
        else:
            anchor = old
            metrics = {
                CANONICAL_METRICS.get(key, key): None if key in UNPRICED_METRICS else value
                for key, value in old["metrics"].items()
            }
        assert anchor["work"] == new["work"], (name, "work")
        assert anchor["memory"] == new["memory"], (name, "memory")
        for key, value in metrics.items():
            assert_value(name, key, value, new["metrics"][key])
    print(f"{len(current)} scenarios: {len(cuda)} CUDA KV anchors, "
          f"{len(current) - len(cuda)} original work/memory anchors")


if __name__ == "__main__":
    main()
