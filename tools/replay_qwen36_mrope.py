"""R26 same-scenario replay for the Qwen3.6 text-MRoPE semantic completion.

before.json was captured on pristine main ee70ce2 (2026-11-16, unmodified
worktree; MTP fixed scenario from docs/evidence/mtp-map/scenario.json plus
the DFlash default ScenarioV2). The R26 change is semantic/evidence-only:
rope rows keep the 3·T·heads·rotary_dim formula, so every work/traffic/
memory total and row count must be **identical**; the only allowed
differences are the evidence fields this round deliberately changed
(M07 structure_complete False→True, M02 ranking reason no longer cites
mrope).

`--check` recomputes the replay in memory against the committed artifacts
without writing anything.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simverse.pipeline import run_v2  # noqa: E402
from simverse.scenario import ScenarioV2  # noqa: E402

FOLDER = ROOT / "docs/evidence/mrope-map"
SCENARIOS = ("mtp_default", "dflash_default")


def _load_scenario(name: str) -> ScenarioV2:
    if name == "mtp_default":
        return ScenarioV2.from_json(json.loads(
            (ROOT / "docs/evidence/mtp-map/scenario.json").read_text(
                encoding="utf-8")))
    return ScenarioV2()


def _snapshot(bundle: dict) -> dict:
    return {
        "totals": {s: v.get("totals") for s, v in bundle.get("work", {}).items()},
        "n_rows": {s: len(v.get("rows", []))
                   for s, v in bundle.get("work", {}).items()},
        "memory_scalars": {k: v for k, v in (bundle.get("memory") or {}).items()
                           if isinstance(v, (int, float, str, bool))},
        "metric_values": {k: v.get("value") for k, v in
                          bundle.get("metrics", {}).items()
                          if isinstance(v, dict) and "value" in v
                          # Explicit additive views introduced after the R26
                          # freeze; every metric present in R26 still compares.
                          and k not in ("M02_speedup_vs_baseline", "M01_tpot_selected",
                                        "M01_baseline_selected", "M02_speedup_selected",
                                        "M04_round_selected")},
    }


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True,
                                     ensure_ascii=False).encode()).hexdigest()


def run(check: bool = False) -> None:
    before_all = json.loads((FOLDER / "before.json").read_text(encoding="utf-8"))
    result, digests = {}, {}
    for name in SCENARIOS:
        bundle = run_v2(_load_scenario(name)).to_json()
        assert bundle["manifest"]["status"] == "completed", name
        snap = _snapshot(bundle)
        base = before_all[name]
        # R26 断言 1：账本逐字段不变（totals/n_rows/memory/指标值）
        assert snap == base, f"{name}: 账本漂移（R26 语义补齐不得改计数）"
        digests[name] = _digest(snap)
        m07 = bundle["metrics"]["M07_support_summary"]
        # R26 断言 2：证据字段按预期翻转（且只有这两处语义变化）
        assert m07["structure_complete"] is True, name
        assert m07["forward_source"] == "source-checked", name
        checks = m07["capability_checks"]
        assert checks["model_structure_complete"] == "supported", name
        assert checks["model_forward_evidence"] == "supported", name
        # R26 断言 3：排名资格不因结构完备解除（cpu/gpu/预算仍阻断）
        m02 = bundle["metrics"]["M02_speedup_vs_baseline"]
        reason = str(m02.get("reason") or "")
        assert "不参与排名" in reason, name
        assert "mrope" not in reason, f"{name}: mrope 仍出现在排名阻断理由"
        cpu_ranked = all(v == "supported" for k, v in checks.items()
                         if k in ("model_cpu_forward", "model_gpu_measurement",
                                  "memory_colocated_r0"))
        assert not cpu_ranked, name
        result[name] = {
            "ledger_digest": digests[name],
            "structure_complete": m07["structure_complete"],
            "forward_source": m07["forward_source"],
            "m02_reason": reason,
            "m02_support": m02.get("support"),
        }
    if check:
        after = json.loads((FOLDER / "after.json").read_text(encoding="utf-8"))
        for name in SCENARIOS:
            assert after[name]["ledger_digest"] == digests[name], name
            assert after[name]["structure_complete"] is True, name
        print("check OK: ledger digests frozen; evidence flip as recorded")
        return
    (FOLDER / "after.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")
    provenance = {
        "captured": "2026-11-16",
        "before": "pristine main ee70ce2 (worktree D:/Github/new-viewer, clean)",
        "scenarios": {"mtp_default": "docs/evidence/mtp-map/scenario.json",
                      "dflash_default": "ScenarioV2() 默认"},
        "expectation": "R26 语义/证据级改动：账本零漂移；M07 structure_complete "
                       "False→True；M02 排名阻断理由不再含 mrope（cpu/gpu/预算"
                       "仍阻断）",
    }
    (FOLDER / "baseline-provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8")
    print("after.json + baseline-provenance.json written")
    for name in SCENARIOS:
        print(f"{name}: ledger_digest={digests[name][:16]}… "
              f"structure_complete=True")


if __name__ == "__main__":
    run(check="--check" in sys.argv)
