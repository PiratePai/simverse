from simverse.model_ir.catalog import catalog_json
from simverse.pipeline import run_v2
from simverse.scenario import ScenarioV2, V41_QUANT


def test_product_catalog_only_focus_models():
    assert set(catalog_json()["models"]) == {"qwen36_35b_a3b", "deepseek_v41_flash"}


def test_default_pair_does_not_claim_forward_complete_and_budget_unknown():
    b = run_v2(ScenarioV2()).to_json()
    checks = b["metrics"]["M07_support_summary"]
    # config 全覆盖 ≠ forward 完备：R23 后 MTP 头已 source-checked，但
    # mrope 未核 + 无 CPU/GPU/预算 → 仍不能宣称结构完备
    assert checks["config_coverage_complete"] is True
    assert checks["forward_source"] == "partial-source"
    assert checks["structure_complete"] is False
    assert checks["capability_checks"]["memory_colocated_r0"] == "unknown"
    reason = b["metrics"]["M02_speedup_vs_baseline"]["reason"]
    assert "colocated_r0" in reason and "OOM" not in reason
    assert "mrope" in reason


def test_dflash_rejects_unverified_target_pair():
    b = run_v2(ScenarioV2(model_id="deepseek_v41_flash",
                          quant=V41_QUANT)).to_json()
    assert "unsupported_combo" in str(b)
