from simverse.pipeline import run_v2
from simverse.scenario import ScenarioV2, V41_QUANT


def test_default_pair_structure_source_checked_but_ranking_still_blocked():
    b = run_v2(ScenarioV2()).to_json()
    checks = b["metrics"]["M07_support_summary"]
    # R26（2026-11-16）文本 MRoPE 补齐后：config 全覆盖 + 文本域 forward
    # 源码证据完备（structure_complete=True，只表示源码核对，不冒充执行/
    # 实测）；排名资格仍被 CPU forward/GPU 实测/预算 unknown 阻断
    assert checks["config_coverage_complete"] is True
    assert checks["forward_source"] == "source-checked"
    assert checks["structure_complete"] is True
    cap = checks["capability_checks"]
    assert cap["model_structure_complete"] == "supported"
    assert cap["model_forward_evidence"] == "supported"
    assert cap["model_cpu_forward"] == "unknown"
    assert cap["model_gpu_measurement"] == "unknown"
    assert cap["memory_colocated_r0"] == "unknown"
    m02 = b["metrics"]["M02_speedup_vs_baseline"]
    reason = str(m02.get("reason") or "")
    assert "colocated_r0" in reason and "OOM" not in reason
    assert "mrope" not in reason          # R26 后不再是阻断项
    assert "CPU forward" in reason        # 执行/实测/预算仍阻断
    assert m02.get("support") == "approximated"


def test_dflash_rejects_unverified_target_pair():
    b = run_v2(ScenarioV2(model_id="deepseek_v41_flash",
                          quant=V41_QUANT)).to_json()
    assert "unsupported_combo" in str(b)
