"""R24 V4.1 SwiGLU clamp + compress_ratios 尾部 same-scenario replay.

Same contract as tools/replay_v41_quant.py（R22）: the immutable before.json is
captured before R24 edits (`--capture-before`, run against the pre-R24
production code), the default run replays the new ledger into after.json +
attribution.json and refuses to overwrite the baseline. Changed op groups must
all be members of EXPECTED_CHANGED_OPS；任何面外行变化都是归因失败。

R24 审计面 = 新增 moe_act/moe_shared_act 两行（SwiGLU clamp 激活逐元素
6 算术 op；字节已归 moe_routed/moe_shared 单一所有者，故只动 flops/rows）。
compress_ratios 尾部与 swiglu_limit 的 config 域收紧只影响 model_ir 覆盖，
不改账本行——面外行必须逐字段冻结。

`--check` recomputes in memory and verifies against the committed
after.json/attribution.json without writing anything.
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

FOLDER = ROOT / "docs/evidence/swiglu-tail"

# R24 审计面：SwiGLU clamp 激活新行（routed + shared）。均为新增行
# （before 侧不存在），flops += rows·inter_rank·6；字节 0（防重复计费）。
EXPECTED_CHANGED_OPS = {"moe_act", "moe_shared_act"}


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
        "note": "R24 基线：生产代码 = main b8585a1（R22 合入后）于主工作树"
                "D:/Github/new-viewer 执行捕获（本工作树当时已含 R24 文档/"
                "pin 与 simverse 改动，不能自证基线）；场景/拓扑与 R22 "
                "quant-map 相同（quant=V41_QUANT 冻结档）",
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
    att = json.loads(
        (FOLDER / "attribution.json").read_text(encoding="utf-8"))
    before = json.loads(
        (FOLDER / "before.json").read_text(encoding="utf-8"))
    fresh = run_v2(scenario).to_json()
    assert fresh["work"] == committed_after["work"], \
        "work 账本与已提交 after.json 不一致（确定性被破坏）"
    for key, val in committed_after["metrics"].items():
        assert fresh["metrics"][key] == val, f"metric {key} 漂移"
    old, new = before["work"]["target_decode"], fresh["work"]["target_decode"]
    AUDITED = set(att["expected_changed_ops"])
    frozen = lambda rows: [r for r in rows if r["op"] not in AUDITED]
    assert frozen(old["rows"]) == frozen(new["rows"]), "面外行漂移"
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
