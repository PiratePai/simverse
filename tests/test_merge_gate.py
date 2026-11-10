"""合入准则自动门禁（MERGE_CRITERIA §5 G1–G7）。

这些测试是规范的机械化部分；它们失败 = 合入违反准则，不允许为通过而放宽。
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from simverse.devices import DEVICES, LINKS

ROOT = Path(__file__).resolve().parent.parent
STATUS_DOC = ROOT / "docs" / "IMPLEMENTATION_STATUS.md"


# ---------------------------------------------------------------------------
# G1 设备/链路来源完备与口径标注
# ---------------------------------------------------------------------------

def test_g1_device_sources_complete_and_dense_labeled():
    for dev in DEVICES.values():
        for key in list(dev.peak_by_dtype) + ["hbm_bw", "mem_capacity",
                                              "launch_overhead_ms", "mfu_cap"]:
            assert key in dev.sources or f"peak_{key}" in dev.sources, \
                f"{dev.id}: 数值 {key} 无来源"
        for dt, src_key in [("fp16", "peak_fp16"), ("fp8", "peak_fp8")]:
            src = dev.sources.get(src_key, "")
            assert "dense" in src.lower(), f"{dev.id}.{dt} 峰值来源未标注 dense 口径"


def test_g1_link_sources_declare_direction():
    for lk in LINKS.values():
        s = lk.source.lower()
        assert ("单向" in s or "双向" in s or "方向" in s), \
            f"链路 {lk.id} 来源未声明带宽方向（bw 为单向口径）"


def test_g1_no_sparse_value_in_peaks():
    """峰值表只允许 dense 数值；已知各卡 sparse=2×dense，防误录。"""
    for dev in DEVICES.values():
        for dt, v in dev.peak_by_dtype.items():
            src = dev.sources.get(f"peak_{dt}", "")
            m = re.search(r"sparse[^\d]*([\d,\.]+)", src.lower())
            if m:
                sparse = float(m.group(1).replace(",", ""))
                assert abs(v * 2 - sparse) / sparse > 0.01, \
                    f"{dev.id}.{dt} 数值 {v} 与来源声明的 sparse {sparse} 成 2× 关系" \
                    "——疑似把 sparse 值录成了 dense"


# ---------------------------------------------------------------------------
# G2 dense 数值抽查（官方数据表口径）
# ---------------------------------------------------------------------------

def test_g2_l40_dense_values_from_datasheet():
    l40 = DEVICES["l40"]
    # NVIDIA L40 Datasheet (Jan-2023): FP16/BF16 181.05 | 362.1 (dense|sparse)
    assert l40.peak_by_dtype["fp16"] == pytest.approx(181.05e12, rel=1e-3)
    src = l40.sources["peak_fp16"]
    assert "181.05" in src and "dense" in src.lower()
    assert "×2" not in src and "90.5 dense" not in src, \
        "旧错误注释（90.5 dense ×2）不得回归"


def test_g2_h200_registered_with_official_specs():
    h = DEVICES["h200_sxm"]
    assert h.peak_by_dtype["fp16"] == pytest.approx(989.5e12, rel=1e-3)
    assert h.hbm_bw == pytest.approx(4.8e12, rel=1e-6)     # HBM3e 4.8 TB/s
    assert h.mem_capacity == pytest.approx(141e9, rel=1e-6)  # 141 GB 标称
    assert "dense" in h.sources["peak_fp16"].lower()


def test_g2_capacity_is_nominal_and_budget_unknown_kept():
    """标称容量必须留在 DeviceBudget.nominal（allocatable=None→feasible=unknown）。"""
    from simverse.pipeline import _device_budget
    for dev_id in DEVICES:
        b = _device_budget(dev_id)
        assert b.allocatable_bytes is None, \
            f"{dev_id}: 无来源不得填 allocatable（feasible 须 unknown）"


# ---------------------------------------------------------------------------
# G3/G4 profile 产物完整性与 measured 准入
# ---------------------------------------------------------------------------

PROFILE_DIR = ROOT / "assets" / "profiles"


def test_g3_distill_files_hash_match_manifest():
    if not PROFILE_DIR.exists():
        pytest.skip("无 profile 资产")
    import hashlib
    for mf in PROFILE_DIR.rglob("manifest.json"):
        man = json.loads(mf.read_text(encoding="utf-8"))
        for name, sha in man["distill_files"].items():
            p = mf.parent / name
            assert p.exists(), f"{mf.parent}: 缺产物 {name}"
            assert hashlib.sha256(p.read_bytes()).hexdigest() == sha, \
                f"{name} 哈希与 manifest 不符"


def test_g4_measured_manifest_requires_environment_identity():
    if not PROFILE_DIR.exists():
        pytest.skip("无 profile 资产")
    for mf in PROFILE_DIR.rglob("manifest.json"):
        man = json.loads(mf.read_text(encoding="utf-8"))
        env = man.get("environment", {})
        for key in ("gpu_name", "driver_version", "torch_version", "cuda_version"):
            assert key in env and env[key], f"{mf}: measured 数据缺环境字段 {key}"
        src = man.get("source", {})
        assert src.get("git_head") and src.get("git_head") != "unknown", \
            f"{mf}: 缺来源提交身份"
        assert src.get("files_sha256"), f"{mf}: 缺源文件哈希清单"


def test_g4_measured_profile_record_requires_metadata():
    from simverse.cost.profile import ProfileRecord
    base = dict(
        profile_id="p", implementation_id="i", recipe_version="r",
        shape={"M": 1}, dtype="bf16", layout="row", device="d", software_stack="s",
        timing_boundary="device-kernel", warmup_runs=1, repeat_runs=2,
        sync_policy="cuda-sync", sample_unit="kernel-invocation",
        samples_ref="sha256:x", content_sha256="sha256:y", captured_at="2026-11-09")
    with pytest.raises(ValueError):
        ProfileRecord(**{**base, "warmup_runs": -1}, source_kind="measured")
    # synthetic 允许（接线 fixture），但 measured 元数据缺失必须拒绝
    ProfileRecord(**base, source_kind="synthetic")


# ---------------------------------------------------------------------------
# G5 无单一全局倍率 / measured 必须有独立预测值
# ---------------------------------------------------------------------------

def test_g5_measured_in_coverage_requires_independent_selected():
    from simverse.cost.profile import (CalibrationModel, ProfileRecord,
                                       apply_calibration, synthetic_profile_fixture)
    rec = synthetic_profile_fixture()
    measured = ProfileRecord(**{**rec.__dict__, "source_kind": "measured"})
    cal = CalibrationModel(
        calibration_id="c", version="1", content_sha256="x", train_refs=[],
        holdout_refs=[], coverage={"k": "in"}, method="lookup")
    est = apply_calibration(123.0, measured, cal, "k")  # 未提供 selected_ms
    assert est.selected_ms is None and est.cost_source == "analytic"
    assert est.fallback_reason and "预测值" in est.fallback_reason
    est2 = apply_calibration(123.0, measured, cal, "k", selected_ms=88.0)
    assert est2.selected_ms == 88.0 and est2.cost_source == "calibrated"
    assert est2.analytic_base_ms == 123.0  # 解析基线保留，不被覆盖


def test_g5_no_global_multiplier_in_profile_module():
    src = (ROOT / "simverse" / "cost" / "profile.py").read_text(encoding="utf-8")
    assert "selected_ms=estimate_ms" not in src, \
        "校准分支不得回填解析值冒充 selected（2026-11-09 已修复的缺陷）"


# ---------------------------------------------------------------------------
# G6 状态文档测试数与实际一致
# ---------------------------------------------------------------------------

def test_g6_status_doc_test_count_matches_reality():
    text = STATUS_DOC.read_text(encoding="utf-8")
    # 只认“当前收集数”标记；历史段落的 passed 数字不参与核对。
    m = re.search(r"pytest 收集数[：:]\s*(\d+)", text)
    assert m, "IMPLEMENTATION_STATUS.md 必须写明当前 'pytest 收集数：N'"
    claimed = int(m.group(1))
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "tests"],
        capture_output=True, text=True, cwd=str(ROOT), timeout=300)
    n = None
    mm = re.search(r"(\d+) tests? collected", out.stdout)
    if mm:
        n = int(mm.group(1))
    else:
        lines = [l for l in out.stdout.splitlines() if ":" in l and l.strip()]
        n = len(lines)
    assert n is not None and n > 0, f"pytest 收集失败：{out.stderr[-300:]}"
    assert claimed == n, \
        f"状态文档声称收集 {claimed} 项，实际收集 {n} 项——文档与实现不一致（G6）"


# ---------------------------------------------------------------------------
# G7 校准不改原始计数（固定重放）
# ---------------------------------------------------------------------------

def test_g7_calibration_leaves_ledgers_untouched():
    from simverse.model_ir import build_catalog
    from simverse.pipeline import run_v2
    from simverse.scenario import ScenarioV2
    scenario_path = ROOT / "assets" / "scenarios" / "h200_calib_loop1.json"
    data = json.loads(scenario_path.read_text(encoding="utf-8"))
    archive = build_catalog(include_internal=True)

    with_prof = run_v2(ScenarioV2.from_json(data), catalog=archive).to_json()
    no_prof = run_v2(ScenarioV2.from_json({**data, "cost_profile": None}),
                     catalog=archive).to_json()

    # 原始账本逐字段一致
    assert with_prof["work"] == no_prof["work"], "Work 账本被校准改变（G7 违规）"
    assert with_prof["model_ref"]["digest"] == no_prof["model_ref"]["digest"]
    # 解析基线指标不变
    assert with_prof["metrics"]["M01_tpot_analytical"]["value"] == \
        no_prof["metrics"]["M01_tpot_analytical"]["value"]
    # run 身份不同（成本身份变化 → 新 run_id，符合重放语义）
    assert with_prof["manifest"]["run_id"] != no_prof["manifest"]["run_id"]
    # 校准只在 cost 侧新增披露
    assert "calibration" in with_prof["cost"]
    assert "calibration" not in no_prof["cost"]


# ---------------------------------------------------------------------------
# G8 活跃目录恰两款主模型 + DFlash；历史/合成不进运行时选择
# ---------------------------------------------------------------------------

def test_g8_active_catalog_exactly_two_models_and_bound_draft():
    from simverse.model_ir import build_catalog, catalog_json
    cat = build_catalog()
    assert set(cat["models"]) == {"qwen36_35b_a3b", "deepseek_v41_flash"}
    assert set(cat["drafts"]) == {"dflash_qwen36_35b_a3b"}
    assert not any(k.startswith("tiny_") for k in cat["models"])
    assert set(catalog_json()["models"]) == {"qwen36_35b_a3b",
                                             "deepseek_v41_flash"}
    # CLI/API 不能运行被移除模型
    from simverse.pipeline import run_v2
    from simverse.scenario import ScenarioV2
    for mid in ("qwen3_30b_a3b", "qwen3_235b_a22b",
                "qwen3_next_80b_a3b", "deepseek_v3", "tiny_gqa"):
        with pytest.raises(ValueError, match="未知模型"):
            run_v2(ScenarioV2(model_id=mid))
    # 合成草稿 chain 方法默认拒绝（仅测试可显式 allow_internal_methods）
    b = run_v2(ScenarioV2(method_id="chain_draft", draft_id=None, k=4))
    assert b.manifest.status == "failed"
    assert "internal_research_method" in str(b.to_json())


def test_g8_api_catalog_hides_archive_and_synthetic_choices():
    from simverse.webapp import app
    client = app.test_client()
    data = client.get("/api/v2/catalog").get_json()
    assert set(data["catalog"]["models"]) == {"qwen36_35b_a3b",
                                              "deepseek_v41_flash"}
    assert set(data["catalog"]["drafts"]) == {"dflash_qwen36_35b_a3b"}
    assert data["profile_sets"] == []
    assert data["methods"]["supported"] == ["baseline", "dflash_v1"]
    assert "chain_draft" not in data["methods"]["supported"] + \
        data["methods"].get("research", [])
    assert all(p["status"] == "archived"
               for p in data["archived_profiles"])


# ---------------------------------------------------------------------------
# G9 完整 config 逐键覆盖；未知键不默认 ignored
# ---------------------------------------------------------------------------

def test_g9_complete_config_leaves_registered_no_default_ignore():
    from simverse.model_ir import build_catalog
    from simverse.model_ir.families import flatten_config, _coverage
    cat = build_catalog()
    pairs = [
        ("qwen36_35b_a3b_main__config.json", "qwen36_35b_a3b", "text_config"),
        ("deepseek_v41_config.json", "deepseek_v41_flash", "text_config"),
        ("qwen36_35b_a3b_dflash__config.json", "dflash_qwen36_35b_a3b", ""),
    ]
    for fname, mid, prefix in pairs:
        raw = json.loads((ROOT / "assets" / "config" / fname)
                         .read_text(encoding="utf-8"))
        item = (cat["models"].get(mid) or cat["drafts"].get(mid))
        leaves = {k for k, v in flatten_config(raw).items()
                  if not isinstance(v, dict)}
        assert leaves <= set(item.coverage), (mid, leaves - set(item.coverage))
        assert not any("默认不影响文本推理结构" in e.get("note", "")
                       for e in item.coverage.values()), mid
    # 未知键必须 unsupported + gap，不能默认 ignored
    cov, gaps = _coverage("qwen3_5_moe", {"num_experts": 1},
                          full_config={"num_experts": 1,
                                       "future_unknown_key": 1})
    assert cov["future_unknown_key"]["status"] == "unsupported-semantic"
    assert any("future_unknown_key" in g for g in gaps)
    # V4.1 缺口归因更新（2026-11-11 R04）：Engram 8 键已按 common/engram.py
    # pin（blob sha1 ba4ceea6 对 vllm-tree.json 校验）source-checked 建模
    # → **不再**是 gap（本断言曾断言 engram_vocab_size ∈ gap_keys；该底层
    # 事实变化是合法门禁更新，非放宽——剩余缺口不变，仍在阻断）。
    m = cat["models"]["deepseek_v41_flash"]
    gap_keys = {g.split(":")[0] for g in m.gaps}
    assert not gap_keys & {f"engram_{k}" for k in (
        "vocab_size", "num_embeddings", "layer_ids", "n_heads", "head_dim",
        "max_ngram_size", "compressed_vocab_size", "pad_token_id")}, \
        "Engram 8 键应已 mapped（R04）；若重新出现缺口须按准则登记原因"
    engram_cov = {k.split(".")[-1]: v["status"]
                  for k, v in m.coverage.items() if "engram_" in k}
    assert set(engram_cov.values()) == {"mapped"}
    # 剩余显式缺口原样保留（不因 Engram 解除而整体放行）
    assert {"quantization_config", "candidate_block_size",
            "swiglu_limit"} <= gap_keys
    assert any("compress_ratios[40:43]" in g for g in m.gaps)


# ---------------------------------------------------------------------------
# G10 证据分栏：config / forward 源码 / CPU 执行 / GPU 实测
# ---------------------------------------------------------------------------

def test_g10_evidence_levels_separated_and_not_overclaimed():
    from simverse.model_ir import build_catalog
    cat = build_catalog()
    q = cat["models"]["qwen36_35b_a3b"]
    v = cat["models"]["deepseek_v41_flash"]
    d = cat["drafts"]["dflash_qwen36_35b_a3b"]
    assert q.config_coverage_complete and not q.forward_evidence_complete
    assert not q.structure_complete
    assert q.evidence.forward_source == "partial-source"
    assert q.evidence.cpu_forward == "not-run"
    assert q.evidence.gpu_measurement == "none"
    assert q.evidence.method_status["mtp"] == "unverified"
    assert "sm-qwen35-gdn-decode-branch" in q.evidence.checked_paths
    assert not v.config_coverage_complete and not v.forward_evidence_complete
    # Engram 归因更新（2026-11-11 R04）：单机 Engram 算子已 pin + source-checked
    # → checked_paths/证据引用在册；未核范围收窄为 DP 交换/cpu_offload 运行时
    # 行为（本断言曾要求 engram_forward ∈ unverified_paths——底层事实变化，
    # 合法门禁更新而非放宽：csa2_moe_routing 仍须 unverified）。
    assert "sm-v41-engram-hc" in v.evidence.checked_paths
    assert "ev-vllm-v41-engram" in v.evidence_refs
    assert any("engram_dp_exchange" in p for p in v.evidence.unverified_paths)
    assert any("csa2_moe_routing" in p for p in v.evidence.unverified_paths)
    assert d.config_coverage_complete
    assert d.evidence.cpu_forward == "not-run"
    assert d.evidence.gpu_measurement == "none"
    assert any("serving_integration" in p for p in d.evidence.unverified_paths)
    # 未执行/未实测不得被写成 executed/measured
    for item in (q, v, d):
        assert item.evidence.cpu_forward != "cpu-executed"
        assert item.evidence.gpu_measurement == "none"


# ---------------------------------------------------------------------------
# G11 数值绝对 oracle（V4.1 o_a 分组 / Qwen3.6 shared gate）
# ---------------------------------------------------------------------------

def test_g11_v41_oa_group_flops_absolute_tp1_2_4_8():
    from simverse.count import build_model_work
    from simverse.model_ir import build_catalog
    m = build_catalog()["models"]["deepseek_v41_flash"]
    Q16 = {"dense_w": 16, "expert_w": 16, "act": 16, "kv": 16, "state": 32}
    for tp in (1, 2, 4, 8):
        ng = 8 // tp
        rows = [r for r in build_model_work(m, "decode", 1, 1024, 1, Q16, tp)
                if r.op == "o_a_proj"]
        assert len(rows) == 40
        for r in rows:
            assert r.flops == 2 * 1 * 4096 * 1024 * ng
            assert r.weight_capacity == 4096 * 1024 * ng * 2


def test_g12_qwen36_shared_gate_row_every_moe_layer():
    from simverse.count import build_model_work
    from simverse.model_ir import build_catalog
    m = build_catalog()["models"]["qwen36_35b_a3b"]
    Q8 = {"dense_w": 8, "expert_w": 8, "act": 8, "kv": 8, "state": 32}
    rows = build_model_work(m, "decode", 1, 1024, 1, Q8, 1)
    gate = [r for r in rows if r.op == "shared_gate"]
    assert len(gate) == 40
    assert all(r.weight_capacity == 2048 for r in gate)


# ---------------------------------------------------------------------------
# G13 草稿 B>1：token 维乘一次；权重/通信不重复乘
# ---------------------------------------------------------------------------

def test_g13_draft_batch_counts_scale_once_not_weights():
    from simverse.count import totals
    from simverse.count.draft import dflash_round_rows, mtp_step_rows
    from simverse.model_ir import build_catalog
    cat = build_catalog()
    m, d = cat["models"]["qwen36_35b_a3b"], cat["drafts"]["dflash_qwen36_35b_a3b"]
    Q8 = {"dense_w": 8, "expert_w": 8, "act": 8, "kv": 8, "state": 32}
    r1 = dflash_round_rows("d", d, m.vocab, 3.0, 512, Q8, 1.0, tp=8, batch=1,
                           ctx_kv_path="fused_raw")
    r2 = dflash_round_rows("d", d, m.vocab, 3.0, 512, Q8, 1.0, tp=8, batch=2,
                           ctx_kv_path="fused_raw")
    t1, t2 = totals(r1), totals(r2)
    assert t2["flops"] == 2 * t1["flops"]
    assert t2["act_read"] == 2 * t1["act_read"]
    assert t2["act_write"] == 2 * t1["act_write"]
    assert t2["kv_read"] == 2 * t1["kv_read"]
    assert t2["kv_write"] == 2 * t1["kv_write"]
    # 权重驻留/读取不随 B 变化
    assert t2["weight_read"] == t1["weight_read"]
    assert sum(r.weight_capacity for r in r1) == \
        sum(r.weight_capacity for r in r2)
    # fc/lm_head 的 token 维只乘一次（不出现 B²）
    for op in ("dflash_fc", "draft_lm_head", "noise_embed"):
        a = next(r for r in r1 if r.op == op)
        b = next(r for r in r2 if r.op == op)
        assert b.flops == 2 * a.flops, op
    # MTP 同样：batch 乘一次，权重不变
    u1 = mtp_step_rows("s", m, 1, 512, Q8, 1, batch=1)
    u2 = mtp_step_rows("s", m, 1, 512, Q8, 1, batch=2)
    v1, v2 = totals(u1), totals(u2)
    assert v2["flops"] == 2 * v1["flops"]
    assert v2["weight_read"] == v1["weight_read"]
    # 通信消息同样只乘一次 B（DFlash/MTP）
    from simverse.pipeline import run_v2
    from simverse.scenario import ScenarioV2
    for method, draft_id in (("dflash_v1", "dflash_qwen36_35b_a3b"),
                             ("mtp", None)):
        def bundle(batch):
            return run_v2(ScenarioV2(
                method_id=method, draft_id=draft_id, k=1,
                deployment={"mode": "dt_serial", "target_tp": 8,
                            "draft_tp": 1},
                workload={"batch": batch, "ctx_tokens": 512})).to_json()
        a, b = bundle(1), bundle(2)
        assert b["cost"]["link"]["bytes"]["d2t"] == \
            2 * a["cost"]["link"]["bytes"]["d2t"], method
        assert b["cost"]["link"]["bytes"]["t2d"] == \
            2 * a["cost"]["link"]["bytes"]["t2d"], method


# ---------------------------------------------------------------------------
# G14 证据引用按实际 model/draft/profile；不跨模型硬编码
# ---------------------------------------------------------------------------

def test_g14_evidence_refs_are_model_specific_and_registered():
    from simverse.model_ir import build_catalog
    from simverse.pipeline import run_v2
    from simverse.scenario import ScenarioV2
    from simverse.validation.registry import build_registry
    known = set(build_registry())
    cat = build_catalog()
    q = run_v2(ScenarioV2()).to_json()
    v = run_v2(ScenarioV2(model_id="deepseek_v41_flash",
                          method_id="baseline", draft_id=None,
                          deployment={"mode": "baseline",
                                      "target_tp": 8})).to_json()
    assert set(q["evidence_refs"]) <= known
    assert "ev-qwen36-main-config" in q["evidence_refs"]
    assert "ev-qwen36-dflash-config" in q["evidence_refs"]
    assert not any("v41" in r for r in q["evidence_refs"])
    assert set(v["evidence_refs"]) <= known
    assert "ev-v41-config" in v["evidence_refs"]
    assert not any("qwen36" in r or "dflash" in r for r in v["evidence_refs"])
    # 归档 profile 的证据只按实际 profile 加入（离线归档目录）
    from simverse.scenario import ScenarioV2 as S
    data = json.loads((ROOT / "assets" / "scenarios" /
                       "h200_calib_loop1.json").read_text(encoding="utf-8"))
    a = run_v2(S.from_json(data), catalog=build_catalog(include_internal=True)).to_json()
    assert "ev-h200-tiny-distill" in a["evidence_refs"]
    assert "ev-qwen3-30b-config" in a["evidence_refs"]


# ---------------------------------------------------------------------------
# G15 部署排名缺 forward/CPU/GPU/预算证据 → 阻断
# ---------------------------------------------------------------------------

def test_g15_deployment_ranking_blocked_without_forward_and_gpu_evidence():
    from simverse.experiments.compare import compare_runs, default_policy
    from simverse.pipeline import run_v2
    from simverse.scenario import ScenarioV2
    base = run_v2(ScenarioV2(method_id="baseline", draft_id=None,
                             deployment={"mode": "baseline", "target_tp": 8}))
    cand = run_v2(ScenarioV2())
    for b in (base, cand):
        m02 = b.metrics["M02_speedup_vs_baseline"]
        assert m02.reason and "不参与排名" in m02.reason
        assert "forward" in m02.reason or "CPU" in m02.reason
        checks = b.capability["checks"]
        assert checks["deployment_ranking_evidence"]["status"] != "supported"
        assert checks["model_gpu_measurement"]["status"] != "supported"
    row = compare_runs(base, [cand], default_policy())["rows"][0]
    assert row["eligible"] is False and row["ineligible_reason"]


def test_g15_forward_gap_alone_blocks_eligibility():
    """容量等其他条件都放行时，partial forward 仍必须单独拦住排名。"""
    from simverse.contracts.evidence import CapabilityReport
    from simverse.pipeline import _eligibility, _eligibility_issues

    cap = CapabilityReport()
    cap.add("model_forward_evidence", "approximated",
            "MTP forward 未核")
    assert _eligibility(cap) is False
    assert _eligibility_issues(cap) == ["MTP forward 未核"]


# ---------------------------------------------------------------------------
# G16 source_map 行号/状态修正，未核路径显式登记
# ---------------------------------------------------------------------------

def test_g16_source_map_line_and_status_corrections():
    from simverse.importers import source_map_json
    sm = {e["entry_id"]: e for e in source_map_json()["entries"]}
    gate = sm["sm-qwen35-attn-output-gate"]
    assert "731-805" in gate["source_location"] and "802" in gate["source_location"]
    shared = sm["sm-qwen35-moe-shared-gate"]
    assert shared["both_sides_covered"] is True
    assert "shared_gate=True" in shared["note"] or "修复" in shared["note"]
    mtp = sm["sm-qwen36-mtp-unverified"]
    assert mtp["both_sides_covered"] is False and mtp["uncovered_domain"]
    assert "Unverified" in mtp["entry_id"] or "unverified" in mtp["entry_id"]
    v41 = sm["sm-v41-csa2-moe-unverified"]
    assert "candidate_publish" in " ".join(v41["uncovered_domain"])


# ---------------------------------------------------------------------------
# G17 CI 工作流：三事件、不按路径过滤、Windows/Node/Python 3.12 明确
# ---------------------------------------------------------------------------

def test_g17_ci_workflow_triggers_and_toolchain_explicit():
    wf = (ROOT / ".github" / "workflows" / "merge-gate.yml").read_text(
        encoding="utf-8")
    for trigger in ("pull_request:", "push:", "merge_group:"):
        assert trigger in wf
    assert "paths:" not in wf and "paths-ignore:" not in wf
    assert "windows-latest" in wf
    assert 'python-version: "3.12"' in wf
    assert "actions/setup-node@v4" in wf and 'node-version: "20"' in wf
    assert "node --check simverse/static/workbench.js" in wf
    assert "python -m pytest tests -q" in wf
    criteria = (ROOT / "docs" / "MERGE_CRITERIA.md").read_text(encoding="utf-8")
    assert "required status check" in criteria or "required check" in criteria
    assert "pull request" in criteria.lower() or "PR review" in criteria
