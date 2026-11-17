"""Review regressions: malformed manifests, provenance, uncertainty and dtype isolation."""
import csv
import hashlib
import importlib.util
import json
import shutil
import sqlite3
from pathlib import Path

import pytest

from simverse.cost import profile_audit as pa

ROOT = Path(__file__).resolve().parents[1]
SID = "ascend-910b-profile-audit-v1"
PACK = pa.AUDIT_ROOT / pa.AUDIT_PACKS[SID]


def importer():
    spec = importlib.util.spec_from_file_location("audit_import_review_test", ROOT / "tools/import_ascend_profiles.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("change", ["empty", "missing", "escape", "bad_sha"])
def test_manifest_cannot_skip_integrity(tmp_path, monkeypatch, change):
    dest = tmp_path / PACK.name
    shutil.copytree(PACK, dest)
    path = dest / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if change == "empty":
        manifest["distill_files"] = {}
    elif change == "missing":
        del manifest["distill_files"]["ledger_rows.json"]
    elif change == "escape":
        manifest["distill_files"]["../outside"] = "a" * 64
    else:
        manifest["distill_files"]["diff.json"] = "invalid"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(pa, "AUDIT_ROOT", tmp_path)
    with pytest.raises(pa.AuditIntegrityError):
        pa.load_audit_pack(SID)
    assert importer().main(["--check", "--out", str(dest)]) == 1


def test_payload_cannot_add_selected_cost_even_with_new_manifest(tmp_path):
    dest = tmp_path / "pkg"
    shutil.copytree(PACK, dest)
    path = dest / "diff.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["runs"][0]["groups"][0]["selected_ms"] = 1.0
    path.write_text(json.dumps(data), encoding="utf-8")
    man_path = dest / "manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    man["distill_files"]["diff.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
    man_path.write_text(json.dumps(man), encoding="utf-8")
    with pytest.raises(pa.AuditIntegrityError, match="selected_ms"):
        pa.validate_audit_directory(dest, SID)


def test_report_propagates_actual_identity_geometry_failure(monkeypatch):
    pack = pa.load_audit_pack(SID)
    ident = next(r for r in pack.identity["runs"] if r["run_id"].endswith("/mtp_4k_bs64"))
    ident["geometry_verification"]["failures"] = ["hidden mismatch sentinel"]
    monkeypatch.setattr(pa, "load_audit_pack", lambda _: pack)
    report = pa.audit_report_json(SID, model_id="qwen36_35b_a3b", tp=2)
    assert any("hidden mismatch sentinel" in s for s in report["admission"]["reasons"])
    assert any("TP 域未知" in s for s in report["admission"]["reasons"])


def test_incomplete_geometry_is_not_full_verification():
    mod = importer()
    pack = pa.load_audit_pack(SID)
    ident = next(r for r in pack.identity["runs"] if r["run_id"].endswith("/mtp_4k_bs64"))
    result = mod._geometry_verify(ident["observed_geometry"], ident["candidate_model_id"], None)
    assert result["verification_passed"] is False and result["unknown"]


def test_unknown_passes_do_not_crash_or_allocate():
    mod = importer()
    kernels = {"shape_stats": {}, "dtypes": {}, "op_stats": {}}
    owner = {"target_model_id": "qwen36_35b_a3b", "draft_method": "mtp"}
    result = mod._ledger_row_match("fixture", "mtp", owner, kernels, [])
    assert result["rows"] == [] and result["pass_structure"]["status"].startswith("unknown")
    groups = [{"group": name, "coverage": "mapped", "observed_us": value}
              for name, value in (("gemm_dense_quant", 100), ("gemm_grouped_moe", 50),
                                  ("attention_kv_write", 10))]
    partial = [{"kernel_class": "MatMul/QuantBatchMatmul", "allocated_to": [{"allocated_us": 40}]}]
    remaining = {g["group"]: g["unallocated_us"] for g in mod._unallocated_groups(groups, partial)}
    assert remaining == {"gemm_dense_quant": 60, "gemm_grouped_moe": 50, "attention_kv_write": 10}


def test_per_call_dtype_is_not_taken_from_operator_union(tmp_path):
    mod = importer()
    path = tmp_path / "kernel_details.csv"
    fields = ["Type", "Duration(us)", "Input Shapes", "Input Data Types"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for dtype, ms in (("DT_BF16;DT_BF16", 10), ("DT_INT8;DT_INT8", 20)):
            writer.writerow({"Type": "MatMulV3", "Duration(us)": ms,
                             "Input Shapes": "1,2048;256,2048", "Input Data Types": dtype})
    kernels = mod._parse_kernels(tmp_path)
    observed = mod._gemm_observed(kernels)
    assert observed[(2048, 256, "bf16")]["total_us"] == 10
    assert observed[(2048, 256, "int8")]["total_us"] == 20
    del kernels["gemm_samples"]
    assert mod._gemm_observed(kernels) == {}


def test_all_consumed_inputs_are_hashed(tmp_path):
    mod = importer()
    source = tmp_path / "src"
    shutil.copytree(ROOT / "tests/fixtures/ascend_profile_audit_tiny/src", source)
    db = source / "runA/ASCEND_PROFILER_OUTPUT/analysis.db"
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE StepTraceTime (stage REAL)")
    _, manifest = mod.build_pack([source])
    files = next(iter(manifest["source"]["files_sha256"].values()))
    for name in ("kernel_details.csv", "op_statistic.csv", "step_trace_time.csv", "analysis.db"):
        assert "ASCEND_PROFILER_OUTPUT/" + name in files


def test_step_comparison_rejects_missing_rows_and_unknown_is_null():
    mod = importer()
    result = mod._step_trace_consistency([{"Stage": 10, "Computing": 2}, {"Stage": 5}],
                                         [{"stage": 10, "computing": 2}])
    assert result["stage_values_match"] is False and result["computing_field_equal"] is False
    assert result["overlapped_communication_delta_us"] == [None]


def test_historical_review_recorded_and_superseded():
    """历史审查（旧包缺源哈希）保留为记录；不再对当前包重放。"""
    import subprocess
    import sys
    old_tool = ROOT / "tools/review_ascend_profile_audit_r30.py"
    result = subprocess.run([sys.executable, "-X", "utf8", str(old_tool),
                             "--check"], cwd=ROOT, capture_output=True,
                            text=True, encoding="utf-8", timeout=60)
    assert result.returncode == 0, result.stderr
    review = json.loads((ROOT / "docs/evidence/ascend-profile-audit-r30-review/"
                         "review.json").read_text(encoding="utf-8"))
    assert review["decision"].startswith("hold-merge")
    assert review["source_commit"] == "89803d0a09c13bcd237049c2c2ccefe7ec381f4d"
    assert len(review["missing_consumed_source_hashes"]) == 6
    assert all(len(files) == 4
               for files in review["missing_consumed_source_hashes"].values())


def test_rebuild_review_replays_and_clears_source_hashes():
    """重建包的审查：消费文件哈希齐全、组级剩余量守恒、几何 unknown 保留。"""
    spec = importlib.util.spec_from_file_location(
        "audit_rebuild_review",
        ROOT / "tools/review_ascend_profile_audit_r30_rebuild.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    result = mod.review()
    assert result["decision"].startswith("approve-audit-only")
    assert result["missing_consumed_source_hashes"] == {}
    assert result["consumed_source_files_per_run"] == 7
    assert result["rebuilt_pack"]["manifest_sha256"] == hashlib.sha256(
        (PACK / "manifest.json").read_bytes()).hexdigest()
    assert result == json.loads(mod.OUT.read_text(encoding="utf-8"))
    # 组级剩余量：每个 mapped/fused 组都有去向；GroupedMoE 与 KV write 必须列入
    mtp = next(r for r in result["group_remaining"]
               if r["run_id"].endswith("mtp_4k_bs64"))
    groups = {g["group"]: g for g in mtp["groups"]}
    assert groups["gemm_grouped_moe"]["remaining_us"] > 3_000_000
    assert groups["gemm_grouped_moe"]["listed_unallocated_us"] == \
        groups["gemm_grouped_moe"]["remaining_us"]
    assert groups["attention_kv_write"]["remaining_us"] > 0
    assert groups["attention_kv_write"]["listed_unallocated_us"] == \
        groups["attention_kv_write"]["remaining_us"]
    # ownership：split kernel 不认领投影 GEMM
    assert result["ownership"]["attn_qkv_norm_rope_fused_ledger_rows"] == [
        "q_norm", "k_norm", "rope"]
    # 几何：Qwen3.6 两个 run 因 TP 未知不得给完整通过
    q36 = [g for g in result["geometry_recheck"]
           if g["run_id"].startswith("qwen3_6_mtp_domino/")]
    assert q36 and all(g["verification_passed"] is False for g in q36)
    assert all("tp_for_global_heads/vocab" in g["unknown"] for g in q36)
