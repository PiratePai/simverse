"""R30 A-N5 离线阶段：Ascend profiling 审计包门禁/重放/隔离回归。

审计包是**只读、未准入**的原始 trace 审核 + 映射 + 差异表：
不产生成本（selected_ms 恒 None）、不进入 PROFILE_SETS、不改任何账本。
本文件不含 400MB 原始采集：hermetic 用 tests/fixtures/ascend_profile_audit_tiny
合成 profile 树重放导入器，真实 6-run 包走已入库的小 JSON 资产。
"""
import hashlib
import importlib.util
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simverse.cost.profile_audit import (  # noqa: E402
    AUDIT_EVIDENCE, AUDIT_PACKS, AuditAdmissionBlocked, AuditIntegrityError,
    audit_packs_json, audit_report_json, load_audit_pack, require_not_admitted)
from simverse.cost.profile_store import (  # noqa: E402
    PRODUCT_PROFILE_SETS, PROFILE_SETS, load_profile_set)

SET_ID = "ascend-910b-profile-audit-v1"
PACK = ROOT / "assets" / "profile_audit" / "ascend_910b_profile_audit_v1"
FIXTURE_SRC = ROOT / "tests" / "fixtures" / "ascend_profile_audit_tiny" / "src"
TOOL = ROOT / "tools" / "import_ascend_profiles.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_tool():
    spec = importlib.util.spec_from_file_location("import_ascend_profiles", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _finite(obj) -> bool:
    if isinstance(obj, float):
        return math.isfinite(obj)
    if isinstance(obj, dict):
        return all(_finite(k) and _finite(v) for k, v in obj.items())
    if isinstance(obj, (list, tuple)):
        return all(_finite(v) for v in obj)
    return True


# ---------------------------------------------------------------------------
# 1) 资产完整性 / 加载器
# ---------------------------------------------------------------------------

def test_pack_manifest_hashes_and_layout():
    man = json.loads((PACK / "manifest.json").read_text(encoding="utf-8"))
    assert man["profile_set_id"] == SET_ID
    assert man["schema_version"] == "ascend-profile-audit/1"
    assert man["kind"] == "ascend-profile-audit"
    assert man["calibration_admission"] == "not-admitted"
    assert man["admitted"] is False
    assert man["run_count"] == 6
    # 独立重算每个产物字节哈希（不经过加载器，防自证）
    assert set(man["distill_files"]) == {
        "README.md", "census.json", "diff.json", "identity.json",
        "ledger_rows.json", "mapping.json", "shape_costs.json"}
    for name, sha in man["distill_files"].items():
        assert _sha256(PACK / name) == sha, name
    # 加载器同样通过
    pack = load_audit_pack(SET_ID)
    assert pack.verified() and pack.admission == "not-admitted"
    assert pack.candidate_models == ["qwen36_35b_a3b", "qwen3_8b_dense"]


def test_loader_rejects_tampered_pack(tmp_path, monkeypatch):
    import simverse.cost.profile_audit as pa
    copy = tmp_path / PACK.name
    shutil.copytree(PACK, copy)
    target = copy / "diff.json"
    target.write_bytes(target.read_bytes() + b" ")
    monkeypatch.setattr(pa, "AUDIT_ROOT", tmp_path)
    with pytest.raises(AuditIntegrityError, match="哈希不符"):
        pa.load_audit_pack(SET_ID)


def test_unknown_audit_id_and_not_a_profile_set():
    with pytest.raises(ValueError, match="未知审计包"):
        load_audit_pack("ascend-910b3-r29-not-exist")
    # 审计包绝不进入可选择的 profile set 白名单（G8 边界）
    assert SET_ID not in PROFILE_SETS
    assert SET_ID not in PRODUCT_PROFILE_SETS
    with pytest.raises(ValueError, match="未知 profile"):
        load_profile_set(SET_ID)
    with pytest.raises(AuditAdmissionBlocked, match="未准入"):
        require_not_admitted(SET_ID)


# ---------------------------------------------------------------------------
# 2) 域门禁：跨模型 / 设备 / TP 显式回退（不是 0）
# ---------------------------------------------------------------------------

def test_cross_model_and_device_domain_rejected_with_reasons():
    rep = audit_report_json(SET_ID, model_id="deepseek_v41_flash")
    reasons = "；".join(rep["admission"]["reasons"])
    assert "跨模型拒绝" in reasons
    assert rep["admission"]["calibration_admission"] == "not-admitted"
    # 只按采集模型过滤，不跨模型借用
    assert rep["domain"]["runs_used"] == []
    assert rep["domain"]["merge_allowed"] is False
    # 设备域：本仓无 Ascend 设备，任何注册设备都不匹配
    rep2 = audit_report_json(SET_ID, device_id="h200_sxm")
    assert "设备不符" in "；".join(rep2["admission"]["reasons"])
    rep3 = audit_report_json(SET_ID, device_id=None)
    assert "无已证明匹配的注册设备" in "；".join(rep3["admission"]["reasons"])
    # 已注册的 Ascend 解析档（assumed 旧仓映射）也不能自动算域匹配：SKU 未证明
    rep5 = audit_report_json(SET_ID, device_id="ascend_910b3")
    r5 = "；".join(rep5["admission"]["reasons"])
    assert "SKU 未证明" in r5 and "assumed" in r5
    assert rep5["candidate_calibration"]["selected_ms"] is None
    rep4 = audit_report_json(SET_ID, tp=8)
    assert any("TP" in r for r in rep4["admission"]["reasons"])


def test_qwen36_candidate_runs_selected_and_8b_not_mixed():
    rep = audit_report_json(SET_ID, model_id="qwen36_35b_a3b")
    assert sorted(rep["domain"]["runs_used"]) == [
        "qwen3_6_mtp_domino/minidraft_4k_bs64",
        "qwen3_6_mtp_domino/mtp_4k_bs64"]
    assert all("qwen3-8b-0811/" not in r["run_id"] for r in rep["runs"])
    rep8 = audit_report_json(SET_ID, model_id="qwen3_8b_dense")
    assert len(rep8["domain"]["runs_used"]) == 4
    assert all("qwen3-8b-0811/" in r for r in rep8["domain"]["runs_used"])


# ---------------------------------------------------------------------------
# 3) 只审计不计价：selected_ms/analytic_base 恒 None
# ---------------------------------------------------------------------------

def test_candidate_calibration_is_null_and_not_measured():
    pack = load_audit_pack(SET_ID)
    for run in pack.diff["runs"]:
        assert run["mapped_ledger_time_share_of_compute"] is not None
        for g in run["groups"]:
            assert g["analytic_base_ms"] is None, g["group"]
            assert g["selected_ms"] is None, g["group"]
            assert g["fallback_reason"]
    rep = audit_report_json(SET_ID, model_id="qwen36_35b_a3b")
    cal = rep["candidate_calibration"]
    assert cal["selected_ms"] is None and cal["analytic_base_ms"] is None
    assert "not-admitted" in cal["fallback_reason"]
    # 不允许出现“measured / admitted”式声明
    assert rep["admission"]["admission_reasons_from_data"]
    assert "measured 校准" not in json.dumps(rep["candidate_calibration"])
    # 只读模块源码不得出现把解析值回填成 selected 的旧缺陷形态（G5 同型）
    src = (ROOT / "simverse" / "cost" / "profile_audit.py").read_text(
        encoding="utf-8")
    assert "selected_ms=estimate_ms" not in src


def test_report_is_js_safe_finite_and_json_roundtrip():
    rep = audit_report_json(SET_ID, model_id="qwen36_35b_a3b", tp=1)
    assert _finite(rep)
    text = json.dumps(rep, ensure_ascii=False, allow_nan=False)
    assert json.loads(text)["set_id"] == SET_ID
    from simverse.contracts import js_safe, js_restore
    assert js_restore(js_safe(rep))["admission"]["admitted"] is False


# ---------------------------------------------------------------------------
# 4) 覆盖/未建模传播 + 形状身份
# ---------------------------------------------------------------------------

def test_unmapped_groups_and_rows_coverage_propagate_not_zero():
    rep = audit_report_json(SET_ID, model_id="qwen36_35b_a3b")
    run = next(r for r in rep["runs"]
               if r["run_id"].endswith("mtp_4k_bs64"))
    unmapped = [g for g in run["groups"]
                if g["coverage"].startswith("unmapped")]
    assert unmapped, "mtp run 应存在未建模类（MoE permute/数据搬运等）"
    assert any(g["group"] == "moe_token_shuffle" for g in unmapped)
    # 无对应账本行的组不得列出 owned rows；实现分解不符的组默认不认领行
    # （如草稿运行时 kernel），如认领语义行必须在 note 中说明——此处只断言
    # 无账本行组为空，且实现不符组存在显式 note。
    assert all(g["owned_ledger_rows"] == [] for g in unmapped
               if g["coverage"] == "unmapped_no_ledger_row")
    for g in unmapped:
        if g["coverage"] == "unmapped_implementation_mismatch":
            assert g["note"]
    assert 0 < run["unmapped_time_share_of_compute"] < 1
    # 账本行静态覆盖：Qwen3.6 decode 27 行中 25 行有组级归属；
    # attn_output_gate（是否融合未知）与 embed_lookup（未观测）显式不覆盖
    cov = run["ledger_row_coverage"]
    assert cov["model_id"] == "qwen36_35b_a3b"
    assert cov["coverage"] == [25, 27]
    assert cov["rows"]["attn_output_gate"]["status"] == "fused_unknown"
    assert cov["rows"]["embed_lookup"]["status"] == "not_observed_in_device_kernels"
    # communication 类必须与账本覆盖分开统计
    assert run["communication_kernel_us"] > 0
    assert run["hcom_communication_kernel_us"] > 0


def test_geometry_identity_checked_and_draft_head_contamination_recorded():
    pack = load_audit_pack(SET_ID)
    by_run = {r["run_id"]: r for r in pack.identity["runs"]}
    mtp = by_run["qwen3_6_mtp_domino/mtp_4k_bs64"]["geometry_check"]
    assert mtp["shape_consistent"] is True and not mtp["failures"]
    assert mtp["q_heads_per_rank_observed"] == 16
    assert mtp["head_dim_observed"] == 256
    assert mtp["moe_experts_observed"] == [256]
    # minidraft_4k 同 OP Type 混有非 catalog 草稿头（hidden 1536 类）→ 记录污染组，
    # 但不让污染组把目标几何判为失败
    mini = by_run["qwen3_6_mtp_domino/minidraft_4k_bs64"]["geometry_check"]
    assert mini["shape_consistent"] is True
    assert mini["contamination"], "应记录同一 OP Type 的异构注意力几何"
    # 8B dense 家族：TP2 tp 组 + 16/4 头/卡 → 总 32/8
    b8 = by_run["qwen3-8b-0811/baseline-t=0.6-tp2-bs32-8k"]["geometry_check"]
    assert b8["shape_consistent"] is True
    assert b8["tp_hint_from_groups"] == 2
    assert b8["q_heads_total_inferred"] == 32
    assert b8["kv_heads_total_inferred"] == 8


def test_op_statistic_reconciles_and_step_trace_boundaries_recorded():
    pack = load_audit_pack(SET_ID)
    for run in pack.census["runs"]:
        c = run["op_statistic_consistency"]
        assert c["count_match_non_hcom"] is True, run["run_id"]
        assert c["total_us_rel_delta_non_hcom"] < 1e-6, run["run_id"]
        assert c["note"]
        st = run["step_trace_consistency"]
        assert "csv_computing_us" in st and "db_computing_us" in st
        # step 节点为空 → 无逐步窗口：CSV 与 analysis.db 的 stage 总数一致，
        # 但 computing 归类不同（原件并列记录、不换算）
        assert st["stage_values_match"] is True, run["run_id"]
        assert st["csv_computing_us"] and st["db_computing_us"]
        # kernel 时间/会话墙钟 = 0..1（含非 steady 区，不能当在线吞吐）
        assert run["kernel_total_over_session"] is None or \
            0 < run["kernel_total_over_session"] <= 1.0
        assert run["union_gap_stats"]["busy_union_us"] is not None
    # 设备内核间隙（device-kernel 边界）是实测值，不是 launch 归因
    mtp = next(r for r in pack.census["runs"]
               if r["run_id"].endswith("mtp_4k_bs64"))
    assert mtp["union_gap_stats"]["median_us"] is not None
    assert "device-kernel" in pack.census["note"]


# ---------------------------------------------------------------------------
# 5) 证据登记 + CLI 活性入口
# ---------------------------------------------------------------------------

def test_registry_evidence_entry_binds_manifest_bytes():
    from simverse.validation.registry import build_registry
    rec = build_registry()["ev-ascend-profile-audit-v1"]
    assert rec.evidence_id == AUDIT_EVIDENCE[SET_ID]
    assert rec.kind == "source-checked"
    assert rec.executed is False
    assert rec.content_sha256 == _sha256(PACK / "manifest.json")
    limits = "；".join(rec.limits)
    assert "不进入 PROFILE_SETS" in limits
    assert "跨 SKU、跨模型" in limits


def test_cli_ascend_audit_check_and_export(tmp_path):
    out = tmp_path / "report.json"
    r = subprocess.run(
        [sys.executable, "-m", "simverse.cli", "ascend-audit", "--check"],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8",
        timeout=180)
    assert r.returncode == 0, (r.stdout, r.stderr)
    assert "not-admitted" in r.stdout
    r2 = subprocess.run(
        [sys.executable, "-m", "simverse.cli", "ascend-audit",
         "--model", "qwen36_35b_a3b", "-o", str(out)],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8",
        timeout=180)
    assert r2.returncode == 0, (r2.stdout, r2.stderr)
    exported = json.loads(out.read_text(encoding="utf-8"))
    assert exported["set_id"] == SET_ID
    assert exported["candidate_calibration"]["selected_ms"] is None


# ---------------------------------------------------------------------------
# 6) 导入器 hermetic 重放（合成 fixture，不依赖 400MB 采集）
# ---------------------------------------------------------------------------

def test_importer_roundtrip_on_tiny_fixture(tmp_path):
    mod = _load_tool()
    out = tmp_path / "pkg"
    assert mod.main(["--src", str(FIXTURE_SRC), "--out", str(out)]) == 0
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert man["run_count"] == 1
    assert man["candidate_model_ids"] == ["qwen36_35b_a3b"]
    assert man["calibration_admission"] == "not-admitted"
    diff = json.loads((out / "diff.json").read_text(encoding="utf-8"))
    run = diff["runs"][0]
    # 手算：kernel 总 1400us，hcom 通信 400us → compute 1000us；
    # mapped=gemm300+attn200+norm250+router100=850（0.85），
    # unmapped=permute150（0.15）
    assert run["observed_kernel_us"] == 1400.0
    assert run["communication_kernel_us"] == 400.0
    assert run["compute_kernel_us"] == 1000.0
    assert run["mapped_ledger_time_share_of_compute"] == 0.85
    assert run["unmapped_time_share_of_compute"] == 0.15
    assert run["ledger_row_coverage"]["coverage"] == [25, 27]
    census = json.loads((out / "census.json").read_text(encoding="utf-8"))
    c = census["runs"][0]["op_statistic_consistency"]
    assert c["count_match_non_hcom"] is True
    assert c["total_us_rel_delta_non_hcom"] == 0.0
    ident = json.loads((out / "identity.json").read_text(encoding="utf-8"))
    geom = ident["runs"][0]["geometry_check"]
    assert geom["shape_consistent"] is True
    assert geom["head_dim_observed"] == 256
    # --check 可重放；篡改产物必须失败
    assert mod.main(["--check", "--out", str(out)]) == 0
    (out / "diff.json").write_bytes((out / "diff.json").read_bytes() + b"\n")
    assert mod.main(["--check", "--out", str(out)]) == 1


def test_manifest_exposes_identity_gaps_not_fabricated():
    pack = load_audit_pack(SET_ID)
    gaps = {g["field"]: g["reason"] for g in
            pack.identity["identity_gap_summary"]}
    for field in ("device_sku_board", "vllm_version/vllm_ascend_commit",
                  "model_repo/revision/config_sha256/checkpoint_sha256",
                  "warmup_active_window/repeat_runs/sync_policy",
                  "step_boundaries"):
        assert field in gaps and gaps[field]
    # 缺字段不得缺席或填 0；domain.sku/implementation_id 显式为 null（未知）
    dom = pack.manifest["domain"]
    assert dom["device_sku"] is None
    assert dom["implementation_id"] is None
    assert dom["tensor_parallel"] is None
    for run in pack.diff["runs"]:
        assert run["unmapped_time_us"] is not None
        assert run["coverage_note"]


# ---------------------------------------------------------------------------
# 7) 隔离性：审计不进入 run_v2 成本/账本
# ---------------------------------------------------------------------------

def test_geometry_verification_from_operator_shapes():
    """用具体算子输入/输出 shape 反推 hidden/vocab/heads，并核对声明模型。"""
    pack = load_audit_pack(SET_ID)
    by_run = {r["run_id"]: r for r in pack.identity["runs"]}
    mtp = by_run["qwen3_6_mtp_domino/mtp_4k_bs64"]["geometry_verification"]
    # Qwen3.6 run 未记录 tp 并行组：形状维度可核对，但 global heads/vocab 缩放
    # 无法证明 → 整体不得给"完整验证通过"（review 修复）；可用维度仍须给出证据。
    assert mtp["checked"] is True
    assert mtp["observed_compatible"] is True
    assert mtp["verification_passed"] is False
    assert "tp_for_global_heads/vocab" in mtp["unknown"]
    f = mtp["fields"]
    assert f["hidden_size"]["match"] is True
    assert 2048 in f["hidden_size"]["observed_norm_candidates"]
    assert 2048 in f["hidden_size"]["observed_gemm_k_candidates"]
    assert f["hidden_size"]["evidence"], "hidden 证据必须来自具体算子 shape"
    assert 248320 in f["vocab_size"]["observed_lm_head_N"]
    assert f["num_experts"]["match"] is True
    # Qwen3-8B（owner 声明）：tp=2 已知 → hidden 4096、vocab 151936 可完整核对
    b8 = by_run["qwen3-8b-0811/baseline-t=0.6-tp2-bs32-8k"][
        "geometry_verification"]
    assert b8["checked"] and b8["verification_passed"] is True
    assert b8["fields"]["hidden_size"]["match"] is True
    assert 4096 in b8["fields"]["hidden_size"]["observed_norm_candidates"]
    assert b8["fields"]["vocab_size"]["match"] is True
    assert b8["fields"]["vocab_size"]["expected"] == 151936
    # 反例：把 hidden 候选换成不匹配的值，验证器必须报失败（不静默通过）
    mod = _load_tool()
    bad = mod._geometry_verify(
        {"shape_candidates": {"hidden_candidates": {"1024": {"count": 1,
                                                             "op_types": {}}},
                              "gemm_k_candidates": {"1024": {"count": 1,
                                                             "op_types": {}}}},
         "moe_experts_observed": [256]}, "qwen36_35b_a3b", 1)
    assert bad["checked"] is True and bad["verification_passed"] is False
    assert any("hidden" in x for x in bad["failures"])


def test_owner_declared_identity_and_minidraft_excluded():
    """owner 声明的目标模型/草稿方法入册；minidraft 按声明排除（不参与匹配）。"""
    pack = load_audit_pack(SET_ID)
    by_run = {r["run_id"]: r for r in pack.identity["runs"]}
    assert by_run["qwen3_6_mtp_domino/mtp_4k_bs64"]["owner_declared"][
        "target_model_id"] == "qwen36_35b_a3b"
    assert by_run["qwen3_6_mtp_domino/mtp_4k_bs64"]["owner_declared"][
        "draft_method"] == "mtp"
    for rid, method in (
            ("qwen3-8b-0811/baseline-t=0.6-tp2-bs32-8k", "baseline"),
            ("qwen3-8b-0811/dflash-t=0.6-tp2-bs64-4k", "dflash"),
            ("qwen3-8b-0811/dspark-t=0.6-tp2-bs64-4k", "dspark")):
        assert by_run[rid]["owner_declared"]["draft_method"] == method
        assert by_run[rid]["owner_declared"]["target_model_id"] == "qwen3_8b_dense"
    excluded = [r for r in pack.identity["runs"]
                if r["owner_declared"]["excluded"]]
    assert sorted(r["owner_declared"]["draft_method"] for r in excluded) == \
        ["minidraft", "minidraft"]
    man = pack.manifest["owner_declared"]
    assert len(man["excluded_runs"]) == 2
    assert all("自研" in x["reason"] for x in man["excluded_runs"])
    # 声明不等于版本验证：身份缺口仍登记
    gaps = "；".join(g["reason"] for g in pack.identity["identity_gap_summary"])
    assert "owner 已声明目标模型名" in gaps and "仍未验证" in gaps


def test_ledger_row_match_extracts_rows_by_counts_not_guessing():
    """row-level 候选提取：passes/MTP K 由计数反解；融合按宽度、同形按计数分摊。"""
    pack = load_audit_pack(SET_ID)
    matches = pack.ledger_match_runs("qwen36_35b_a3b")
    assert len(matches) == 1, "只有未排除的 Qwen3.6 MTP run 参与行级匹配"
    mtp = matches[0]
    assert mtp["excluded_by_owner"] is False
    ps = mtp["pass_structure"]
    assert ps["passes"] == 96 and ps["mtp_k"] == 3
    assert "derived-from-counts" in ps["status"]
    rows = {tuple(sorted((r["signature"] or {}).items())): r
            for r in mtp["rows"] if r["signature"]}
    # GDN in_proj_qkvz 融合核：8192:4096 按输出宽度分摊，计数精确对上
    qkvz = next(r for r in mtp["rows"]
                if r["signature"] == {"K": 2048, "N": 12288, "dtype": "bf16"})
    assert qkvz["status"] == "matched"
    alloc = {a["op"]: a for a in qkvz["allocated_to"]}
    assert alloc["in_proj_qkv"]["allocated_us"] == pytest.approx(
        2 * alloc["in_proj_z"]["allocated_us"], rel=1e-6)
    assert all(a["split_basis"] == "width" for a in qkvz["allocated_to"])
    # 同形 o_proj/out_proj/mtp_fc：按计数方程 10:30:3:3 分摊
    o_same = next(r for r in mtp["rows"]
                  if r["signature"] == {"K": 4096, "N": 2048, "dtype": "bf16"})
    assert o_same["status"] == "matched" and o_same["count_residual"] == 0
    by_op = {(a["op"], a["role"]): a for a in o_same["allocated_to"]}
    assert by_op[("out_proj", "target")]["allocated_us"] == pytest.approx(
        3 * by_op[("o_proj", "target")]["allocated_us"], rel=1e-6)
    # lm_head 计数有残差 → 显式 ambiguous，不填 0/不硬摊
    lm = next(r for r in mtp["rows"]
              if r["signature"] == {"K": 2048, "N": 248320, "dtype": "bf16"})
    assert lm["status"] == "ambiguous_count_residual"
    assert lm["allocated_to"] == [] and lm["count_residual"] != 0
    # 注意力核按 10:3 计数分 target/draft
    attn = next(r for r in mtp["rows"]
                if r["kernel_class"] == "FusedInferAttentionScore")
    a_target = next(a for a in attn["allocated_to"] if a["role"] == "target")
    a_draft = next(a for a in attn["allocated_to"] if a["role"] == "draft")
    assert a_target["predicted_count"] == 960 and a_draft["predicted_count"] == 288
    cov = mtp["coverage"]
    assert cov["rows_scope"] and cov["rows_total"] >= 10
    assert 0 < cov["matched_time_us"] <= cov["candidate_time_us"]
    # 非 GEMM 融合组显式未分摊（不静默丢弃）
    unalloc = {g["group"] for g in mtp["unallocated_groups"]}
    assert {"norm_add_fused", "gdn_core", "attn_qkv_norm_rope_fused"} <= unalloc
    # 报告侧：row-level 匹配随 model_id 过滤，且仍是 candidate-only
    rep = audit_report_json(SET_ID, model_id="qwen36_35b_a3b")
    lrm = rep["ledger_row_match"]
    assert lrm["status"].startswith("candidate-only")
    assert [r["run_id"] for r in lrm["runs"]] == [mtp["run_id"]]
    assert lrm["runs"][0]["coverage"]["rows_total"] == cov["rows_total"]


def test_ledger_row_match_not_performed_for_cross_model_or_excluded():
    pack = load_audit_pack(SET_ID)
    # Qwen3-8B 为历史/离线模型：不做跨模型行级匹配
    b8 = pack.ledger_match_runs("qwen3_8b_dense")
    assert b8 and all("不在活跃 catalog" in (r.get("reason") or "") for r in b8)
    # minidraft 按 owner 声明排除（无论目标模型）
    excluded = [r for r in pack.ledger_rows["runs"] if r["excluded_by_owner"]]
    assert len(excluded) == 2
    assert all("自研" in (r.get("reason") or "") for r in excluded)


def test_audit_module_does_not_change_run_v2_ledgers():
    from simverse.pipeline import run_v2
    from simverse.scenario import ScenarioV2
    before = run_v2(ScenarioV2()).to_json()
    rep = audit_report_json(SET_ID, model_id="qwen36_35b_a3b")
    after = run_v2(ScenarioV2()).to_json()
    assert rep["candidate_calibration"]["selected_ms"] is None
    assert before["work"] == after["work"]
    assert before["memory"] == after["memory"]
    assert before["evidence_refs"] == after["evidence_refs"]
    assert "calibration" not in after["cost"]
    assert "calibration" not in before["cost"]
    # catalog 索引只读；profile_sets 仍为空（选择入口不变）
    items = audit_packs_json()
    assert [i["set_id"] for i in items] == [SET_ID]
    assert items[0]["admitted"] is False
    assert all(i["calibration_admission"] == "not-admitted"
               for i in items if "integrity_error" not in i)
