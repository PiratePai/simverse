"""R21 same-scenario replay and per-op attribution (no model/kernel execution).

The immutable before.json is captured before R21 edits (`--capture-before`).
The default run replays the new ledger into after.json + attribution.json and
refuses to overwrite the baseline. Changed op groups must all be members of
EXPECTED_CHANGED_OPS (the audited V4.1 MoE routing subgraph); any other row
changing is an attribution failure.
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
        "note": "R21 基线：HEAD 73738ad（main，R20 CSA2 合入后）+ 预置 MoE "
                "routing pin 源码（moe-pin.json 11 文件，blob 对 "
                "vllm-tree.json 校验）；simverse 生产代码未改动",
    }
    (FOLDER / "before.json").write_text(
        json.dumps(before, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8", newline="\n")
    (FOLDER / "baseline-provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n")
    print("baseline captured at", head)


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
    else:
        main()
