"""Replay PR #26 review from committed audit artifacts; no external profiles/NPU."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "assets/profile_audit/ascend_910b_profile_audit_v1"
OUT = ROOT / "docs/evidence/ascend-profile-audit-r30-review/review.json"
MANIFEST_SHA = "165cfed78fc5400e5fed3ae401160357409f1ff2c2af1483754a443c98e33fe5"
REFERENCE_SHA = "a9a71634ed138f590fbc3ad0de277a5f56a06451522fd091b2657d6daf6084b0"


def review():
    def read(name):
        return json.loads((PACK / name).read_text(encoding="utf-8"))
    assert hashlib.sha256((PACK / "manifest.json").read_bytes()).hexdigest() == MANIFEST_SHA
    manifest = read("manifest.json")
    for name, sha in manifest["distill_files"].items():
        assert hashlib.sha256((PACK / name).read_bytes()).hexdigest() == sha, name
    reference = OUT.parent / "reference/patch_qwen3_5.py"
    assert hashlib.sha256(reference.read_bytes()).hexdigest() == REFERENCE_SHA
    spec = importlib.util.spec_from_file_location("audit_import_review", ROOT / "tools/import_ascend_profiles.py")
    importer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(importer)
    missing_hashes = {}
    consumed = ("kernel_details.csv", "op_statistic.csv", "step_trace_time.csv", "analysis.db")
    for run, files in manifest["source"]["files_sha256"].items():
        missing_hashes[run] = [name for name in consumed
                               if name not in files and "ASCEND_PROFILER_OUTPUT/" + name not in files]
    geometry = []
    for ident in read("identity.json")["runs"]:
        revised = importer._geometry_verify(ident["observed_geometry"],
                                            ident["candidate_model_id"], ident["tp_hint_from_groups"])
        geometry.append({"run_id": ident["run_id"],
                         "original_verification_passed": ident["geometry_verification"].get("verification_passed"),
                         "revised": revised})
    gaps = []
    ledger = {r["run_id"]: r for r in read("ledger_rows.json")["runs"]}
    for run in read("diff.json")["runs"]:
        rows = ledger[run["run_id"]]
        if rows["excluded_by_owner"] or not rows.get("rows"):
            continue
        listed = {g["group"] for g in rows["unallocated_groups"]}
        gaps.append({"run_id": run["run_id"], "groups_missing_from_unallocated": [
            {"group": g["group"], "observed_us": g["observed_us"]}
            for g in run["groups"] if g["group"] in ("gemm_grouped_moe", "attention_kv_write")
            and g["group"] not in listed],
            "allocation_warning": "K/N 按 OP Type dtype 并集聚合，M/逐调用角色未分离；"
                                  "宽度/调用数分摊不证明 target/draft 耗时相同，原数值只能是候选假设"})
    return {"source_pr": "https://github.com/PiratePai/simverse/pull/26",
            "source_commit": "89803d0a09c13bcd237049c2c2ccefe7ec381f4d",
            "source_manifest_sha256": MANIFEST_SHA,
            "verification": "committed derived artifacts + source reading + offline tests; no raw-profile replay or NPU run",
            "decision": "hold-merge: corrected importer must regenerate source-bound artifacts before approval",
            "missing_consumed_source_hashes": missing_hashes,
            "geometry_recheck": geometry,
            "unallocated_group_gaps": gaps,
            "reference": {"repo": "https://github.com/vllm-project/vllm-ascend",
                          "commit": "baa023ada41a5b889cac6249845a688aab130892", "sha256": REFERENCE_SHA,
                          "path": "vllm_ascend/patch/worker/patch_qwen3_5.py",
                          "finding": "qkv_proj runs before split_qkv_rmsnorm_mrope; the split kernel does not own projection GEMM",
                          "limit": "reference source only; collection engine commit remains unknown"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = review()
    if args.check:
        assert json.loads(OUT.read_text(encoding="utf-8")) == result, "Review replay drift"
    else:
        OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(result["decision"])
    print("unhashed source runs:", sum(bool(x) for x in result["missing_consumed_source_hashes"].values()))


if __name__ == "__main__":
    main()
