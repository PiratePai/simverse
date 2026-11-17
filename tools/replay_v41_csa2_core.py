"""R25 V4.1 稀疏 MLA 注意力核本体 same-scenario replay.

Same contract as tools/replay_v41_swiglu.py（R24）: the immutable before.json
is captured before R25 edits（`--capture-before`，于主工作树 D:/Github/
new-viewer 的干净 main 生产代码上执行），默认运行重放新账本到 after.json +
attribution.json，拒绝覆盖基线。变化的行组必须全部落在
EXPECTED_CHANGED_OPS；任何面外行变化都是归因失败。

R25 审计面 = csa2_attention_core（csa2-core-pin.json 14 文件）：
  * sparse_attention：decode 头数 heads//tp → padded_heads=64（fp8 kernel
    h_q∈{64,128}，fused insert 零填充 Q）；prefill kv_read 从"每对直读记录"
    改为 attn_kv_gather 全量压缩前缀+SWA 尾窗 gather，核改读 bf16 工作区；
  * q_rope：q_padded 写宽（padded_heads×d×2）；
  * 新增 attn_splitkv（SM90 恒 split-KV o_accum/combine）、
    attn_index_global（decode 每压缩层全局槽位转换）、
    attn_swa_index_build（每步一次窗口索引，记 L0）、
    attn_sched_meta（每层型首层调度器元数据）、attn_kv_gather（prefill）；
  * sparse_index_build：R20 的 decode 运行首归因修正为 prefill 每层每 chunk
    （decode 默认路径不调用该 Triton 核）。
`--check` 无漂移复算：对已提交 after.json/attribution.json 核对，不写文件。
本工具 --check 是当前的活性无漂移入口（承接 R24 的链条）。

R27 重基注记（2026-11-17）：after.json 已按 R27 成本口径重定价重基（工作
账本逐字段与 R25 版本一致；M01/M02/M04 等成本衍生指标随 GEMM 档
fp8→fp16(bf16) 修正变化，见 docs/evidence/csa2-core/rebase-r27-provenance.json
与 docs/evidence/cost-integrity-r27/）。attribution.json 保留 R25 历史记录；
main() 重放时 t_ms 列按当前成本档同价复算（字节 delta 不受影响）。
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

FOLDER = ROOT / "docs/evidence/csa2-core"

# R25 审计面：注意力核本体消费链（新增行 + 归因修正行 + 头数/写宽联动行）
EXPECTED_CHANGED_OPS = {
    "sparse_attention", "sparse_index_build", "q_rope",
    "attn_splitkv", "attn_index_global", "attn_swa_index_build",
    "attn_sched_meta", "attn_kv_gather",
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
        "note": "R25 基线：生产代码 = main ee70ce2（R23 合入后）于主工作树"
                "D:/Github/new-viewer 执行捕获（csa2mla 工作树已含 R25 改动，"
                "不能自证基线）；场景/拓扑与 R24 swiglu-tail 相同"
                "（TP8/B1/ctx32768/quant=V41_QUANT 冻结档）",
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
    # R27（2026-11-17）：GEMM 计算档按 extras.quant 执行配方改为 bf16 档后，
    # M01 差混入全局重定价，不再等于 R25 结构性逐行 t_ms delta。归因改用
    # **同价复算**：两侧行都用当前 dtype_route 计价，t_ms_delta 只反映
    # R25 账本结构变化（重定价本身见 docs/evidence/cost-integrity-r27/）。
    def stage_time(rows):
        return sum(price_row(r, device, gemm, ew, engram_lookup_host=True)["t_ms"]
                   for r in rows)
    time_delta = stage_time(
        [OpRow(**r) for r in new["rows"]]) - stage_time(
        [OpRow(**r) for r in old["rows"]])
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
