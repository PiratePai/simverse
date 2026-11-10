"""V12 一致性：fresh run → Web API / CLI / JSON/CSV/trace/HTML 逐字段对账。

不比对旧存档行数（E21 修复）：全部从当次 fresh run 派生。
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simverse.contracts import js_restore  # noqa: E402
from simverse.pipeline import run_v2  # noqa: E402
from simverse.scenario import ScenarioV2  # noqa: E402

SC = {
    "schema_version": "0.2", "name": "v12-check",
    "model_id": "qwen36_35b_a3b", "draft_id": "dflash_qwen36_35b_a3b",
    "method_id": "dflash_v1", "k": 15, "block_size": 16,
    "deployment": {"mode": "dt_serial", "target_device": "h100_sxm",
                   "target_tp": 8, "draft_device": "l40", "draft_tp": 1},
    "workload": {"batch": 1, "ctx_tokens": 8192},
    "acceptance": {"kind": "geometric", "alpha": 0.72},
}


def test_web_cli_json_csv_consistent(tmp_path):
    # 通道 1：进程内 API（与 /api/v2/run 同入口 run_v2）
    inproc = run_v2(ScenarioV2.from_json(SC)).to_json()

    # 通道 2：Flask test client（/api/v2/run）
    from simverse.webapp import app
    client = app.test_client()
    r = client.post("/api/v2/run", json=SC)
    assert r.status_code == 200, r.data
    web = js_restore(r.get_json())

    # 通道 3：CLI 子进程（四类导出）
    sc_file = tmp_path / "sc.json"
    sc_file.write_text(json.dumps(SC), encoding="utf-8")
    out = tmp_path / "run.json"
    csv_p = tmp_path / "ops.csv"
    trace_p = tmp_path / "trace.csv"
    html_p = tmp_path / "report.html"
    proc = subprocess.run(
        [sys.executable, "-m", "simverse.cli", "run-v2", str(sc_file),
         "-o", str(out), "--csv", str(csv_p), "--trace", str(trace_p),
         "--html", str(html_p)],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8")
    assert proc.returncode == 0, proc.stderr
    cli = json.loads(out.read_text(encoding="utf-8"))

    # 逐字段：manifest/metrics/capability 一致
    assert inproc["manifest"]["run_id"] == web["manifest"]["run_id"] == \
        cli["manifest"]["run_id"]
    for key in ("M01_tpot_analytical", "M01_tpot_baseline", "M02_speedup_vs_baseline",
                "M03_E_N", "M04_round_ms"):
        a, b, c = (inproc["metrics"][key], web["metrics"][key],
                   cli["metrics"][key])
        assert a["value"] == b["value"] == c["value"], key
        assert a["unit"] == b["unit"] == c["unit"], key
        assert a.get("reason") == b.get("reason") == c.get("reason"), key
    assert inproc["capability"] == web["capability"] == cli["capability"]

    # work 账本行数与逐行整数一致（CLI JSON vs 进程内）
    rows_a = inproc["work"]["target_decode"]["rows"]
    rows_c = cli["work"]["target_decode"]["rows"]
    assert len(rows_a) == len(rows_c) and len(rows_a) > 0
    for x, y in zip(rows_a, rows_c):
        for k2 in ("stage", "layer", "op", "flops", "weight_read", "kv_read",
                   "kv_write", "act_read", "act_write"):
            assert x[k2] == y[k2], (x["op"], k2)

    # CSV = bundle 行的忠实序列化（逐行字段，非只对行数）
    lines = csv_p.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].startswith("stage,layer,op")
    assert len(lines) - 1 == len(rows_c)
    for line, row in zip(lines[1:], rows_c):
        parts = line.split(",")
        assert parts[0] == row["stage"] and parts[2] == row["op"]
        assert int(parts[3]) == row["flops"]

    # trace CSV = 事件忠实序列化
    tlines = trace_p.read_text(encoding="utf-8").strip().splitlines()
    events = cli["execution"]["rounds"][0]["events"]
    assert len(tlines) - 1 == len(events) >= 3
    assert tlines[1].split(",")[0] == events[0]["event_id"]

    # HTML 导出携带 run 身份与 unknown 原因
    html = html_p.read_text(encoding="utf-8")
    assert cli["manifest"]["run_id"][:30] in html
    assert "T0 解析估算" in html


def test_stale_response_not_reused():
    """切换场景后 run_id 变化：晚到的旧响应不可覆盖新结果（UI 键含身份）。"""
    a = run_v2(ScenarioV2.from_json(SC))
    sc2 = dict(SC)
    sc2["acceptance"] = {"kind": "geometric", "alpha": 0.9}
    b = run_v2(ScenarioV2.from_json(sc2))
    assert a.manifest.run_id != b.manifest.run_id
    assert a.metrics["M03_E_N"].value != b.metrics["M03_E_N"].value
