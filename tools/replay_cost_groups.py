"""Fixed pre-refactor numerical/ledger anchors; offline only, no device execution."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simverse.contracts import digest
from simverse.model_ir import build_catalog
from simverse.pipeline import run_v2
from simverse.scenario import ScenarioV2

OUT = ROOT / "docs/evidence/cost-groups/baseline.json"
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
        # New canonical aliases are not pre-refactor anchors.
        metrics = {k:v for k,v in metrics.items() if not any(x in k for x in (
            "_selected", "M18_selected"))}
        result[name] = {"work": digest(bundle["work"]),
                        "memory": digest(bundle["cost"].get("memory", bundle.get("memory"))),
                        "metrics": metrics}
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", action="store_true", help="Capture only from the reviewed original revision")
    args = parser.parse_args()
    current = snapshot()
    if args.capture:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({"source_commit":"75c048990b51dc73b7bfb8e6a88989225b61fae4",
                                  "cases":current}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    else:
        expected = json.loads(OUT.read_text(encoding="utf-8"))["cases"]
        for name, old in expected.items():
            new = current[name]
            assert old["work"] == new["work"], (name, "work")
            assert old["memory"] == new["memory"], (name, "memory")
            for key, value in old["metrics"].items():
                actual = new["metrics"].get(key)
                assert actual == value or (value is not None and actual is not None and
                    abs(actual - value) <= max(1e-10, abs(value) * 1e-12)), (name, key, value, actual)
    print(f"{len(current)} scenarios: unchanged numerical/ledger anchors")

if __name__ == "__main__": main()
