"""Replay the R30 rebuild review from committed audit artifacts; no external profiles/NPU.

输入：`assets/profile_audit/ascend_910b_profile_audit_v1/`（由修复后的
`tools/import_ascend_profiles.py` 从原件重建，含所有已消费源文件 SHA256）。
输出：`docs/evidence/ascend-profile-audit-r30-rebuild/review.json`。

本工具只验证 committed 派生产物，不读取约 400 MB 原始采集、不运行 NPU；
`--check` 可重放。历史审查（缺源哈希/旧包）保留在
`docs/evidence/ascend-profile-audit-r30-review/review.json`，本记录声明 supersedes。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "assets/profile_audit/ascend_910b_profile_audit_v1"
OLD_REVIEW = ROOT / "docs/evidence/ascend-profile-audit-r30-review/review.json"
OUT = ROOT / "docs/evidence/ascend-profile-audit-r30-rebuild/review.json"

CONSUMED = ("profiler_info_0.json", "profiler_metadata.json",
            "ascend_pytorch_profiler_0.db",
            "ASCEND_PROFILER_OUTPUT/kernel_details.csv",
            "ASCEND_PROFILER_OUTPUT/op_statistic.csv",
            "ASCEND_PROFILER_OUTPUT/step_trace_time.csv",
            "ASCEND_PROFILER_OUTPUT/analysis.db")
GROUPS_BY_CLASS = {"MatMul/QuantBatchMatmul": "gemm_dense_quant",
                   "FusedInferAttentionScore": "attention_core"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(name: str):
    return json.loads((PACK / name).read_text(encoding="utf-8"))


def review() -> dict:
    manifest = _read("manifest.json")
    pack_sha = _sha(PACK / "manifest.json")
    for name, sha in manifest["distill_files"].items():
        assert _sha(PACK / name) == sha, name
    # 1) 已消费源文件哈希必须逐 run 齐全
    missing = {}
    for run, files in manifest["source"]["files_sha256"].items():
        absent = [name for name in CONSUMED if name not in files]
        if absent:
            missing[run] = absent
    # 2) 总量对账（msprof 聚合 vs kernel_details 非 hcom 行）
    consistency = []
    for run in _read("census.json")["runs"]:
        item = run["op_statistic_consistency"]
        consistency.append({
            "run_id": run["run_id"],
            "count_match_non_hcom": item["count_match_non_hcom"],
            "total_us_rel_delta_non_hcom": item["total_us_rel_delta_non_hcom"],
            "kernel_rows": run["kernel_rows"],
            "hcom_kernel_rows": item["hcom_kernel_rows"],
        })
    # 3) 组级剩余量：每个 mapped/fused 组都必须有去向（已分摊或列 unallocated）
    diff = {r["run_id"]: r for r in _read("diff.json")["runs"]}
    ledger = {r["run_id"]: r for r in _read("ledger_rows.json")["runs"]}
    group_remaining = []
    for run_id, rows in sorted(ledger.items()):
        if rows.get("excluded_by_owner") or not rows.get("rows"):
            group_remaining.append({"run_id": run_id,
                                    "status": "excluded-or-no-row-match"})
            continue
        allocated = {}
        for row in rows["rows"]:
            group = GROUPS_BY_CLASS.get(row["kernel_class"])
            if group:
                allocated[group] = allocated.get(group, 0.0) + sum(
                    x["allocated_us"] for x in row.get("allocated_to", []))
        listed = {g["group"]: g for g in rows.get("unallocated_groups", [])}
        entries = []
        for g in diff[run_id]["groups"]:
            if g["coverage"] not in ("mapped", "fused_group"):
                continue
            remaining = round(max(0.0, g["observed_us"] - allocated.get(g["group"], 0.0)), 3)
            entry = {"group": g["group"], "observed_us": g["observed_us"],
                     "allocated_us": round(allocated.get(g["group"], 0.0), 3),
                     "remaining_us": remaining,
                     "listed_unallocated_us": (listed.get(g["group"]) or {}).get(
                         "unallocated_us")}
            if remaining > 0.001:
                assert entry["listed_unallocated_us"] == remaining, group
            entries.append(entry)
        group_remaining.append({"run_id": run_id, "groups": entries})
    # 4) ownership：split kernel 不认领投影 GEMM
    mapping = _read("mapping.json")
    split = next(r for r in mapping["rows"]
                 if r["group"] == "attn_qkv_norm_rope_fused")
    # 5) 几何：Qwen3.6 两个 run 因 TP 未知不能给完整通过；Qwen3-8B 可验证
    geometry = []
    for ident in _read("identity.json")["runs"]:
        gv = ident["geometry_verification"]
        geometry.append({"run_id": ident["run_id"],
                         "verification_passed": gv.get("verification_passed"),
                         "observed_compatible": gv.get("observed_compatible"),
                         "unknown": gv.get("unknown"),
                         "failures": gv.get("failures")})
    old = json.loads(OLD_REVIEW.read_text(encoding="utf-8"))
    return {
        "source_pr": "https://github.com/PiratePai/simverse/pull/26",
        "supersedes": {
            "review": "docs/evidence/ascend-profile-audit-r30-review/review.json",
            "source_commit": old["source_commit"],
            "source_manifest_sha256": old["source_manifest_sha256"]},
        "rebuilt_pack": {"path": PACK.relative_to(ROOT).as_posix(),
                         "manifest_sha256": pack_sha,
                         "distill_files": sorted(manifest["distill_files"])},
        "verification": "committed derived artifacts + manifest/source SHA256 + offline tests; "
                        "no raw-profile replay (raw tree absent) and no NPU run",
        "decision": "approve-audit-only: consumed-source hashes complete and group remaining "
                    "reconciled; production calibration still blocked (A-N4/ownership/阶段)",
        "missing_consumed_source_hashes": missing,
        "consumed_source_files_per_run": len(CONSUMED),
        "totals_consistency": consistency,
        "group_remaining": group_remaining,
        "ownership": {
            "attn_qkv_norm_rope_fused_ledger_rows": split["ledger_rows"],
            "note": "参考 vllm-ascend forward 中 qkv_proj 先于 split/norm/rope；"
                    "split kernel 不认领投影 GEMM。采集栈 commit 仍未知。"},
        "geometry_recheck": geometry,
        "residual_limits": [
            "M/阶段/角色/布局/权重未与调用关联；row-level 数值仅为 width/count-derived "
            "假设诊断，不标 measured",
            "设备 SKU/board、vLLM/vllm-ascend commit、权重 revision、warmup/step 边界仍缺",
            "融合组总时间可作 audit-only，正式 CalibratedPricer 接入另验（#28/#29）",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = review()
    if args.check:
        assert json.loads(OUT.read_text(encoding="utf-8")) == result, "Rebuild review drift"
        print("OK rebuild review replay", result["rebuilt_pack"]["manifest_sha256"][:16])
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8", newline="\n")
    print(result["decision"])
    print("missing consumed source hashes:", sum(bool(x) for x in
                                                 result["missing_consumed_source_hashes"].values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
