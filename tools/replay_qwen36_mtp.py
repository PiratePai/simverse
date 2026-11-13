"""R23 same-scenario replay and per-op attribution for the Qwen3.6 MTP mapping.

before.json was captured on unmodified main (`--capture-before`, scenario
method=mtp k=1 — the only legal K on main, where plan_chain rejects K>
mtp_layers). The default run replays the new ledger into after.json +
attribution.json and refuses to overwrite the baseline.

The audited change surface is the whole MTP draft subgraph (old rows:
GDN-template layer via build_model_work + mtp_embed_share, no draft KV/MTP
resident objects; new rows: pinned vLLM structure per step). Rows outside
the draft stage (target_decode / verify) must be field-for-field frozen.

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

FOLDER = ROOT / "docs/evidence/mtp-map"

# 审计面内的阶段（旧实现整段被替换）；target_decode/verify 必须逐字段冻结
DRAFT_STAGES = ("draft_step_0",)


def _load_scenario() -> ScenarioV2:
    return ScenarioV2.from_json(json.loads(
        (FOLDER / "scenario.json").read_text(encoding="utf-8")))


def capture_before() -> None:
    raise SystemExit(
        "before.json 已由 main 工作区一次性捕获（见 baseline-provenance.json）；"
        "禁止在分支上重写基线")


def _work_rows(bundle: dict, stage: str):
    return {(r["layer"], r["op"]): r for r in bundle["work"][stage]["rows"]}


def _diff_stage(before: dict, after: dict, stage: str) -> dict:
    rb, ra = _work_rows(before, stage), _work_rows(after, stage)
    by_op = {}
    for key in sorted(set(rb) | set(ra)):
        b, a = rb.get(key), ra.get(key)
        if b is None:
            row = {"status": "added"}
            row.update({f: a[f] for f in ("flops", "weight_read", "act_read",
                                          "act_write", "kv_read", "kv_write",
                                          "state_read", "state_write")})
            by_op[f"{key[0]}:{key[1]}"] = row
        elif a is None:
            by_op[f"{key[0]}:{key[1]}"] = {"status": "removed"}
        else:
            delta = {}
            for f in ("flops", "weight_read", "act_read", "act_write",
                      "kv_read", "kv_write", "state_read", "state_write"):
                if (b.get(f) or 0) != (a.get(f) or 0):
                    delta[f] = (b.get(f) or 0, a.get(f) or 0, a.get(f) - b.get(f))
            if delta or b.get("shape") != a.get("shape"):
                by_op[f"{key[0]}:{key[1]}"] = {"status": "changed", "delta": delta}
    return by_op


def run(check: bool = False) -> None:
    scenario = _load_scenario()
    after = run_v2(scenario).to_json()
    assert after["manifest"]["status"] == "completed"
    before = json.loads((FOLDER / "before.json").read_text(encoding="utf-8"))
    attribution = {
        "scenario": "method=mtp k=1 B1 ctx32768 TP8/EP1 W8A8+FP4E（与 before 同场景）",
        "changed_stages": {},
        "frozen_stages": {},
        "metrics": {
            "M01_before": before["metrics"]["M01_tpot_analytical"]["value"],
            "M01_after": after["metrics"]["M01_tpot_analytical"]["value"],
            "steady_before": before["cost"]["memory"]["colocated_r0"]["steady_bytes"],
            "steady_after": after["cost"]["memory"]["colocated_r0"]["steady_bytes"],
        },
        "k_domain_change": "main 上 K≤mtp_layers=1（其余拒绝）；R23 起 K≥1 循环"
                           "复用（pinned spec_step_idx % n；e2e K=3×MTP1 在册）",
    }
    for stage in DRAFT_STAGES:
        attribution["changed_stages"][stage] = _diff_stage(before, after, stage)
    for stage in ("target_decode", "verify"):
        same = _diff_stage(before, after, stage) == {}
        same = same and (
            before["work"][stage]["totals"] == after["work"][stage]["totals"])
        attribution["frozen_stages"][stage] = same
        if not same:
            attribution["frozen_stages"][f"{stage}_detail"] = _diff_stage(
                before, after, stage)
    if check:
        committed_after = json.loads(
            (FOLDER / "after.json").read_text(encoding="utf-8"))
        committed_attr = json.loads(
            (FOLDER / "attribution.json").read_text(encoding="utf-8"))
        ok = True
        for stage in ("target_decode", "verify", "draft_step_0"):
            if after["work"][stage]["totals"] != \
                    committed_after["work"][stage]["totals"]:
                ok = False
                print(f"DRIFT in {stage}")
        ok = ok and all(committed_attr["frozen_stages"].values())
        print("check:", "OK" if ok else "DRIFT")
        sys.exit(0 if ok else 1)
    (FOLDER / "after.json").write_text(
        json.dumps(after, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8", newline="\n")
    (FOLDER / "attribution.json").write_text(
        json.dumps(attribution, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8", newline="\n")
    sha = hashlib.sha256((FOLDER / "after.json").read_bytes()).hexdigest()
    print("after.json written, sha256", sha[:16])
    print("M01:", attribution["metrics"]["M01_before"], "->",
          attribution["metrics"]["M01_after"])
    print("steady:", attribution["metrics"]["steady_before"], "->",
          attribution["metrics"]["steady_after"])
    print("frozen ok:", attribution["frozen_stages"])


if __name__ == "__main__":
    if "--capture-before" in sys.argv:
        capture_before()
    run(check="--check" in sys.argv)
