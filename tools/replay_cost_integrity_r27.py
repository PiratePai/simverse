"""R27 cost-integrity same-scenario replay and attribution（2026-11-17）.

审计面（docs/validation/cost-integrity-review-fixes.md）：
  * A 计算峰值路由：V4.1 GEMM 档按 extras.quant 执行配方选 bf16 档
    （Marlin W8A16/W4A16，R22 pin）；无配方模型沿用 dense_w 启发档；
  * B Roofline 元数据：compute_dtype/peak/effective_peak/bandwidth/cost_basis
    由 price_row 同一次计价输出；
  * C 期望通信字节：DFlash/MTP t2d 保留 E[N] 浮点期望（K15/α0.72 锚点
    58209.08759654315，旧代码 int() 截断成 58209）；
  * D 内部 chain 计价：草稿精度/设备与计数同源；
  * E pipeline 模式 Metric 增补不丢既有警告（Engram host + 流水分列并存）；
  * F 运行身份：device_snapshot_digest 覆盖硬件/链路数值与来源。

铁律：全部修复只改成本，work/cost.memory 逐字段与 before 相等（本工具
--check 强制断言）。before.json 于 pre-R27 工作区捕获（HEAD 1196510，
provenance 见 baseline-provenance.json），不可覆盖。
`--check` 对已提交 after/attribution 复算核对，不写任何文件。
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simverse.count.work import OpRow, totals  # noqa: E402
from simverse.cost.t0 import _ELEMENTWISE, dtype_route, price_row  # noqa: E402
from simverse.devices import DEVICES  # noqa: E402
from simverse.pipeline import run_v2  # noqa: E402
from simverse.scenario import ScenarioV2  # noqa: E402

FOLDER = ROOT / "docs/evidence/cost-integrity-r27"

# 场景 → 是否需要 allow_internal_methods（chain 为内部研究 fixture）
SCENARIOS = {
    "v41-baseline": False,        # V4.1 默认 TP8/B1/ctx32768（dtype 路由面）
    "qwen-dflash": False,         # Qwen3.6+DFlash CLI 默认 K15/α0.72（消息字节面）
    "v41-chain-pipeline": True,   # V4.1+chain+dt_pipeline_sim（警告保留/身份面）
}

# 期望字节锚点（E[N]=Σ_{i=0..15}0.72^i=3.552800756624948；8 层×h2048×1B）
T2D_ANCHOR = 58209.08759654315


def _load(name: str) -> ScenarioV2:
    return ScenarioV2.from_json(json.loads(
        (FOLDER / f"scenario-{name}.json").read_text(encoding="utf-8")))


def _run(name: str) -> dict:
    return run_v2(_load(name),
                  allow_internal_methods=SCENARIOS[name]).to_json()


def _old_dtype_route(quant: dict):
    """R27 前的路由（dense_w 启发，无执行配方入参）——用于 before 侧同账本
    复算归因；仅本工具使用，生产代码不再暴露该口径。"""
    return ("fp16" if quant["dense_w"] >= 16 else "fp8",
            "fp16" if quant["act"] >= 16 else "fp8")


def _v41_opclass_attribution(before: dict, after: dict, scenario) -> dict:
    """V4.1 baseline：按 gemm/elementwise 类聚合逐行 t_ms 归因。

    before/after 两侧都用各自时代的 dtype 路由计价（before=fp8 启发档，
    after=bf16 配方档），字节/算量取自账本（两侧逐字段相等，已由调用方
    断言）——t_ms 差只含计算档修正，不混入计数变化。
    """
    quant_recipe = (getattr(_load_model_ir(scenario), "extras", None)
                    or {}).get("quant")
    dev = DEVICES[scenario.deployment.target_device]
    out = {}
    for stage in ("target_decode",):
        old_rows = [OpRow(**r) for r in before["work"][stage]["rows"]]
        new_rows = [OpRow(**r) for r in after["work"][stage]["rows"]]
        g_old, e_old = _old_dtype_route(scenario.quant)
        g_new, e_new = dtype_route(scenario.quant, quant_recipe)
        # 修正轮 ew 档两侧相同（act=16）；断言防口径漂移
        assert e_old == e_new, "元素档两侧应一致（本修复只动 GEMM 档）"
        cls = {}
        for label, rows, gemm, ew in (
                ("before", old_rows, g_old, e_old),
                ("after", new_rows, g_new, e_new)):
            t_gemm = t_ew = 0.0
            for r in rows:
                p = price_row(r, dev, gemm, ew, engram_lookup_host=True)
                if r.op in _ELEMENTWISE:
                    t_ew += p["t_ms"]
                else:
                    t_gemm += p["t_ms"]
            cls[label] = {"gemm_t_ms": t_gemm, "elementwise_t_ms": t_ew}
        delta = (cls["after"]["gemm_t_ms"] - cls["before"]["gemm_t_ms"],
                 cls["after"]["elementwise_t_ms"] - cls["before"]["elementwise_t_ms"])
        out[stage] = {"classes": cls, "t_ms_delta": delta,
                      "gemm_tier": {"before": g_old, "after": g_new}}
    return out


def _load_model_ir(scenario):
    from simverse.model_ir import build_catalog
    return build_catalog()["models"][scenario.model_id]


def build_attribution() -> dict:
    att: dict = {}
    bundles = {name: _run(name) for name in SCENARIOS}
    befores = {name: json.loads((FOLDER / f"before-{name}.json").read_text(
        encoding="utf-8")) for name in SCENARIOS}
    # -- V4.1 baseline：dtype 路由面 --
    before, after = befores["v41-baseline"], bundles["v41-baseline"]
    att["v41_dtype_route"] = {
        "scenario": "V4.1 baseline TP8/B1/ctx32768/quant=V41_QUANT",
        "gemm_tier_before": "fp8（dense_w=8 启发档）",
        "gemm_tier_after": "fp16=bf16 档（Marlin W8A16/W4A16 执行配方）",
        "M01_before": before["metrics"]["M01_tpot_analytical"]["value"],
        "M01_after": after["metrics"]["M01_tpot_analytical"]["value"],
        "M04_before": before["metrics"]["M04_round_ms"]["value"],
        "M04_after": after["metrics"]["M04_round_ms"]["value"],
        "opclass": _v41_opclass_attribution(before, after, _load("v41-baseline")),
    }
    # -- Qwen3.6+DFlash：期望通信字节面 --
    before_q, after_q = befores["qwen-dflash"], bundles["qwen-dflash"]
    att["qwen_dflash_expected_bytes"] = {
        "scenario": "Qwen3.6+DFlash colocated TP8 K15/α0.72 blk16 ctx32768",
        "link_bytes_before": before_q["cost"]["link"]["bytes"],
        "link_bytes_after": after_q["cost"]["link"]["bytes"],
        "t2d_anchor": T2D_ANCHOR,
        "M19_t2d_statistic": after_q["metrics"]["M19_msg_bytes"]["t2d"]["statistic"],
        "M01_before": before_q["metrics"]["M01_tpot_analytical"]["value"],
        "M01_after": after_q["metrics"]["M01_tpot_analytical"]["value"],
    }
    # -- V4.1 chain+pipeline：警告保留/身份面 --
    before_e, after_e = (befores["v41-chain-pipeline"],
                         bundles["v41-chain-pipeline"])
    m01b, m01a = (before_e["metrics"]["M01_tpot_analytical"],
                  after_e["metrics"]["M01_tpot_analytical"])
    att["v41_chain_pipeline_metric_merge"] = {
        "scenario": "V4.1 chain_draft dt_pipeline_sim（内部研究组合）",
        "M01_reason_before": m01b.get("reason"),
        "M01_reason_after": m01a.get("reason"),
        "M01_support_after": m01a["support"],
        "formula_ref_after": m01a.get("formula_ref"),
        "host_note_preserved": ("Engram host" in (m01a.get("reason") or "")
                                and "dt_pipeline_sim" in (m01a.get("reason") or "")),
        "run_id_before": before_e["manifest"]["run_id"],
        "run_id_after": after_e["manifest"]["run_id"],
        "run_id_basis_change": ("R27 设备快照含硬件/链路数值+来源；"
                                "全部历史 run_id 基随之变更，证据包未回改"),
    }
    att["counts_invariant"] = {
        name: {"work_equal": befores[name]["work"] == bundles[name]["work"],
               "memory_equal": (befores[name]["cost"]["memory"]
                                == bundles[name]["cost"]["memory"])}
        for name in SCENARIOS}
    return att


def check() -> None:
    """无漂移复算：与已提交 after/attribution 核对 + 契约断言，不写文件。"""
    ok = True
    for name in SCENARIOS:
        before = json.loads((FOLDER / f"before-{name}.json").read_text(
            encoding="utf-8"))
        committed = json.loads((FOLDER / f"after-{name}.json").read_text(
            encoding="utf-8"))
        fresh = _run(name)
        # ① 成本修正不得改原始计数（work/memory 与 before 逐字段相等）
        assert fresh["work"] == before["work"], f"{name}: work 漂移"
        assert fresh["cost"]["memory"] == before["cost"]["memory"], \
            f"{name}: memory 漂移"
        # 集成审查新增的展示摘要不改变 R27 原始 after 证据；逐行覆盖
        # 单独断言，随后只对旧证据所含字段做无漂移核对。
        priced = fresh["cost"].get("row_prices", [])
        rows = fresh["work"]["target_decode"]["rows"]
        assert len(priced) == len(rows), f"{name}: 逐行计价覆盖不足"
        assert {p["row_id"] for p in priced} == {
            f"{r['stage']}:{r['layer']}:{r['op']}" for r in rows}
        comparable = dict(fresh)
        comparable["cost"] = {k: v for k, v in fresh["cost"].items()
                              if k not in ("row_prices", "groups", "selection")}
        # Additive cost-selection views (2026-11-18); keep comparing every old
        # metric and all old cost fields against the immutable R27 artifact.
        comparable["metrics"] = {k:v for k,v in fresh["metrics"].items() if k not in (
            "M01_tpot_selected", "M01_baseline_selected", "M02_speedup_selected", "M04_round_selected",
            "M18_selected_cohort_period_ms")}
        # HEAD necessarily changes after R27. Verify current identity, then
        # compare the immutable numerical artifact independently of code identity.
        from simverse.contracts import RunManifest
        current_manifest = RunManifest(**fresh["manifest"])
        assert current_manifest.compute_run_id() == fresh["manifest"]["run_id"]
        comparable["manifest"] = {k:v for k,v in fresh["manifest"].items() if k not in ("code_revision", "run_id")}
        committed["manifest"] = {k:v for k,v in committed["manifest"].items() if k not in ("code_revision", "run_id")}
        # ② 当前代码复现已提交 after（确定性）
        if comparable != committed:
            ok = False
            print(f"DRIFT: after-{name}.json differs from current replay")
    att = build_attribution()
    committed_att = json.loads((FOLDER / "attribution.json").read_text(
        encoding="utf-8"))
    for report in (att, committed_att):
        ids = report["v41_chain_pipeline_metric_merge"]
        assert ids["run_id_before"] != ids["run_id_after"], "执行模式变化必须改变运行身份"
        # Current run ID is checked above; frozen IDs remain untouched on disk.
        report["v41_chain_pipeline_metric_merge"].pop("run_id_before", None)
        report["v41_chain_pipeline_metric_merge"].pop("run_id_after", None)
    if json.dumps(att, sort_keys=True) != json.dumps(
            committed_att, sort_keys=True):
        ok = False
        print("ATTRIBUTION DRIFT")
    # ③ 契约锚点
    a = att["qwen_dflash_expected_bytes"]
    assert a["link_bytes_after"]["t2d"] == T2D_ANCHOR, "t2d 期望字节锚点漂移"
    assert isinstance(a["link_bytes_after"]["t2d"], float), "t2d 应为浮点期望"
    assert a["link_bytes_after"]["d2t"] == 60 and isinstance(
        a["link_bytes_after"]["d2t"], int), "d2t 应保持协议整数"
    assert a["M19_t2d_statistic"] == "expectation"
    e = att["v41_chain_pipeline_metric_merge"]
    assert e["host_note_preserved"], "Engram host 警告被流水模式覆盖"
    # ④ V4.1 GEMM 档修正的逐类归因一致性：M01 差 == gemm 类 t 差（elementwise 类 0）
    v = att["v41_dtype_route"]
    cls = v["opclass"]["target_decode"]
    m01_delta = v["M01_after"] - v["M01_before"]
    assert abs(m01_delta - cls["t_ms_delta"][0]) < 1e-9, "M01 差与 gemm 类 t 差不符"
    assert abs(cls["t_ms_delta"][1]) < 1e-12, "elementwise 类时延不应变化"
    assert ok, "R27 证据与当前复算不一致"
    print("check OK: counts invariant; anchors hold; attribution consistent")


def main() -> None:
    for name in SCENARIOS:
        bundle = _run(name)
        assert bundle["manifest"]["status"] == "completed", name
        (FOLDER / f"after-{name}.json").write_text(
            json.dumps(bundle, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8", newline="\n")
    att = build_attribution()
    (FOLDER / "attribution.json").write_text(
        json.dumps(att, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n")
    for f in sorted(FOLDER.glob("after-*.json")):
        print(f.name, hashlib.sha256(f.read_bytes()).hexdigest()[:16])
    print("M01 v41:", att["v41_dtype_route"]["M01_before"], "->",
          att["v41_dtype_route"]["M01_after"])
    print("t2d:", att["qwen_dflash_expected_bytes"]["link_bytes_before"]["t2d"],
          "->", att["qwen_dflash_expected_bytes"]["link_bytes_after"]["t2d"])


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    else:
        main()
