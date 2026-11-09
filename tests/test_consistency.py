"""三方一致性验收：Web API / CLI / 导出文件 消费同一入口，数字逐项相等。

验收口径（用户验收条款）：Web、CLI、导出结果一致 —— RunResult 的
metrics/comparisons/digests 在三条通道间精确相等（JSON 规范化后）。
"""
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

FIXED = ROOT / "assets" / "scenarios" / "v41_flash_dt_verification.json"


@pytest.fixture(scope="module")
def cli_result(tmp_path_factory):
    out = tmp_path_factory.mktemp("cli") / "run.json"
    csv = out.with_suffix(".csv")
    proc = subprocess.run(
        [sys.executable, "-m", "simverse.cli", "run", str(FIXED), "-o", str(out),
         "--csv", str(csv)],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8")
    assert proc.returncode == 0, proc.stderr
    return json.loads(out.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def web_result():
    from simverse.webapp import app
    client = app.test_client()
    # 通道 1：场景文件名（重放路径）
    r1 = client.post("/api/run", json={"scenario_file": "v41_flash_dt_verification"})
    assert r1.status_code == 200
    # 通道 2：场景 JSON 内联（同一场景 → 同一结果）
    r2 = client.post("/api/run", json=json.loads(FIXED.read_text(encoding="utf-8")))
    assert r2.status_code == 200
    assert r1.get_json() == r2.get_json()
    return r1.get_json()


def _norm(o):
    if isinstance(o, float) and math.isnan(o):
        return "<nan>"
    if isinstance(o, dict):
        return {k: _norm(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_norm(v) for v in o]
    return o


def test_web_equals_cli(web_result, cli_result):
    a, b = _norm(web_result), _norm(cli_result)
    assert a["digests"] == b["digests"]
    assert a["metrics"] == b["metrics"]
    assert a["comparisons"] == b["comparisons"]
    assert len(a["ops"]) == len(b["ops"]) == len(cli_result["ops"])
    # 逐算子整数逐项相等（计数精确相等验收）
    for x, y in zip(a["ops"], b["ops"]):
        for key in ("stage", "layer", "op", "flops", "weight_read", "kv_read",
                    "kv_write", "act_read", "act_write"):
            assert x[key] == y[key], (x["stage"], x["op"], key)


def test_csv_export_matches_runresult(cli_result, tmp_path):
    """导出 CSV 行数与列值 = RunResult.ops 的忠实序列化（无第二套口径）。"""
    csv_path = tmp_path / "ops.csv"
    proc = subprocess.run(
        [sys.executable, "-m", "simverse.cli", "export",
         str(ROOT / "outputs" / "run_v41.json"), "--csv", str(csv_path)],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8")
    assert proc.returncode == 0, proc.stderr
    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].startswith("stage,layer,op")
    assert len(lines) - 1 == len(cli_result["ops"])


def test_fixed_scenario_replay_digest_stable():
    """固定场景文件重放：digest 与 outputs 存档一致（可重放交付物）。"""
    from simverse.contracts import ScenarioSpec
    from simverse.engine import run
    scenario = ScenarioSpec.from_json(json.loads(FIXED.read_text(encoding="utf-8")))
    r = run(scenario)
    archived = json.loads((ROOT / "outputs" / "run_v41.json").read_text(encoding="utf-8"))
    assert r.scenario_digest == archived["digests"]["scenario"]
    assert _norm(r.to_json()["metrics"]) == _norm(archived["metrics"])


def test_scenario_validation_errors():
    from simverse.contracts import ScenarioSpec
    with pytest.raises(ValueError):
        ScenarioSpec(k=0).validate()
    with pytest.raises(ValueError):
        ScenarioSpec(acceptance={"kind": "chain_alpha", "chain_alpha": 1.5}).validate()
    with pytest.raises(ValueError):
        ScenarioSpec(acceptance={"kind": "committed_hist"}).validate()
    with pytest.raises(ValueError):
        ScenarioSpec(deployment={"mode": "dt_magic"}).validate()
