"""R21 same-scenario replay and per-op attribution (no model/kernel execution).

The immutable before.json is captured before R21 edits (`--capture-before`).
The default run replays the new ledger into after.json + attribution.json and
refuses to overwrite the baseline. Changed op groups must all be members of
EXPECTED_CHANGED_OPS (the audited V4.1 MoE routing subgraph); any other row
changing is an attribution failure.

`--check` recomputes the replay in memory and verifies it against the
committed after.json/attribution.json without writing anything (the ledger
is deterministic; the plain run also refreshes manifest identity fields
code_revision/run_id, so re-running it after a commit legitimately dirties
after.json — use --check for no-drift verification).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simverse.count.work import OpRow, totals  # noqa: E402
from simverse.cost.t0 import dtype_route, price_row  # noqa: E402
from simverse.devices import DEVICES  # noqa: E402
from simverse.pipeline import run_v2  # noqa: E402
from simverse.scenario import ScenarioV2  # noqa: E402

FOLDER = ROOT / "docs/evidence/moe-route"

# 本轮审计面内的行组：旧通用 moe_router 行被拆为 v41 三行
# （gate/score/topk_renorm）；routed/shared 行字节语义未变，不在面内。
EXPECTED_CHANGED_OPS = {
    "moe_router",          # 旧合并行（before 侧）
    "moe_gate", "moe_score", "moe_topk_renorm",
}


def capture_baseline() -> None:
    scenario = ScenarioV2.from_json(json.loads(
        (FOLDER / "scenario.json").read_text(encoding="utf-8")))
    before = run_v2(scenario).to_json()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
        text=True, check=True).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
        text=True, check=True).stdout
    provenance = {
        "head": head,
        "captured_before_edits": True,
        "worktree": str(ROOT),
        "git_status_porcelain": status,
        "note": "R21 基线：HEAD 73738ad（main，R20 CSA2 合入后）+ 当时预置 "
                "MoE routing pin 源码（moe-pin.json 11 文件；gate 修复后 "
                "14 文件，blob 对 vllm-tree.json 校验）；simverse 生产代码"
                "未改动",
    }
    (FOLDER / "before.json").write_text(
        json.dumps(before, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8", newline="\n")
    provenance["before_sha256"] = hashlib.sha256(
        (FOLDER / "before.json").read_bytes()).hexdigest()
    (FOLDER / "baseline-provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n")
    print("baseline captured at", head)


def check() -> None:
    """无漂移复算：与已提交 after.json/attribution.json 核对，不写任何文件。"""
    scenario = ScenarioV2.from_json(json.loads(
        (FOLDER / "scenario.json").read_text(encoding="utf-8")))
    committed_after = json.loads(
        (FOLDER / "after.json").read_text(encoding="utf-8"))
    att = json.loads((FOLDER / "attribution.json").read_text(encoding="utf-8"))
    before = json.loads((FOLDER / "before.json").read_text(encoding="utf-8"))
    fresh = run_v2(scenario).to_json()
    # 账本本体必须逐字段复现（manifest 身份字段 code_revision/run_id 随
    # HEAD 变化，属登记性漂移；work/metrics 数值不得漂移）
    assert fresh["work"] == committed_after["work"], \
        "work 账本与已提交 after.json 不一致（确定性被破坏）"
    for key, val in committed_after["metrics"].items():
        if key.startswith("M01_"):
            assert fresh["metrics"][key] == val, key
    # 归因仍然成立：逐字段 delta = Σ by_op delta；审计面外行冻结
    old, new = before["work"]["target_decode"], fresh["work"]["target_decode"]
    for key in old["totals"]:
        delta = new["totals"][key] - old["totals"][key]
        assert delta == sum(g["delta"][key] for g in att["by_op"].values()), key
    AUDITED = set(att["expected_changed_ops"])
    frozen = lambda rows: [r for r in rows if r["op"] not in AUDITED]
    assert frozen(old["rows"]) == frozen(new["rows"])
    print("check OK: ledger reproduces committed after.json; attribution holds")


def main() -> None:
    scenario = ScenarioV2.from_json(json.loads(
        (FOLDER / "scenario.json").read_text(encoding="utf-8")))
    before_path = FOLDER / "before.json"
    before = json.loads(before_path.read_text(encoding="utf-8"))
    after = run_v2(scenario).to_json()
    (FOLDER / "after.json").write_text(
        json.dumps(after, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8", newline="\n")
    old = before["work"]["target_decode"]
    new = after["work"]["target_decode"]
    fields = list(old["totals"])
    groups = {}
    gemm, ew = dtype_route(scenario.quant)
    device = DEVICES[scenario.deployment.target_device]
    for op in sorted({r["op"] for r in old["rows"] + new["rows"]}):
        a = [r for r in old["rows"] if r["op"] == op]
        b = [r for r in new["rows"] if r["op"] == op]
        ar, br = [OpRow(**r) for r in a], [OpRow(**r) for r in b]
        ta, tb = totals(ar), totals(br)
        delta = {key: tb[key] - ta[key] for key in fields}
        def time(rows):
            return sum(price_row(r, device, gemm, ew, engram_lookup_host=True)["t_ms"]
                       for r in rows)
        if any(delta.values()):
            groups[op] = {"before": ta, "after": tb, "delta": delta,
                          "rows_before": len(a), "rows_after": len(b),
                          "t_ms_delta": time(br) - time(ar)}
    unexpected = set(groups) - EXPECTED_CHANGED_OPS
    assert not unexpected, f"审计面外的行发生变化: {sorted(unexpected)}"
    delta = {key: new["totals"][key] - old["totals"][key] for key in fields}
    for key in fields:
        assert sum(g["delta"][key] for g in groups.values()) == delta[key], key
    time_delta = (after["metrics"]["M01_tpot_analytical"]["value"]
                  - before["metrics"]["M01_tpot_analytical"]["value"])
    assert abs(sum(g["t_ms_delta"] for g in groups.values()) - time_delta) < 1e-10
    assert old["rows"][-3:] == new["rows"][-3:]  # final_norm/head/embed unchanged
    result = {
        "before_sha256": hashlib.sha256(before_path.read_bytes()).hexdigest(),
        "scenario_sha256": hashlib.sha256(
            (FOLDER / "scenario.json").read_bytes()).hexdigest(),
        "before": old["totals"], "after": new["totals"], "delta": delta,
        "by_op": groups, "t_ms_delta": time_delta,
        "before_memory": before["metrics"]["M06_memory_target_r0_detail"],
        "after_memory": after["metrics"]["M06_memory_target_r0_detail"],
        "before_known_peak": before["metrics"]["M06_memory_target_r0"]["value"],
        "after_known_peak": after["metrics"]["M06_memory_target_r0"]["value"],
        "expected_changed_ops": sorted(EXPECTED_CHANGED_OPS),
        "execution": "analytical replay only; no CPU forward or kernel execution",
    }
    (FOLDER / "attribution.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n")
    print(json.dumps({"delta": delta, "t_ms_delta": time_delta},
                     ensure_ascii=True))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--capture-before":
        capture_baseline()
    elif len(sys.argv) > 1 and sys.argv[1] == "--check":
        check()
    else:
        main()
