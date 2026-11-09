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
    m = re.search(r"(\d+)\s*passed", text)
    assert m, "IMPLEMENTATION_STATUS.md 必须写明当前 'N passed' 测试数"
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
        f"状态文档声称 {claimed} passed，实际收集 {n} 项——文档与实现不一致（G6）"


# ---------------------------------------------------------------------------
# G7 校准不改原始计数（固定重放）
# ---------------------------------------------------------------------------

def test_g7_calibration_leaves_ledgers_untouched():
    from simverse.pipeline import run_v2
    from simverse.scenario import ScenarioV2
    scenario_path = ROOT / "assets" / "scenarios" / "h200_calib_loop1.json"
    data = json.loads(scenario_path.read_text(encoding="utf-8"))

    with_prof = run_v2(ScenarioV2.from_json(data)).to_json()
    no_prof = run_v2(ScenarioV2.from_json({**data, "cost_profile": None})).to_json()

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
