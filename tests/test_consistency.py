"""三方一致性验收：Web API / CLI / 导出文件 消费同一入口，数字逐项相等。

验收口径（用户验收条款）：Web、CLI、导出结果一致 —— RunResult 的
metrics/comparisons/digests 在三条通道间精确相等（JSON 规范化后）。
"""
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

FIXED = ROOT / "assets" / "scenarios" / "v41_flash_dt_verification.json"
# 历史快照（2026-11-06 本地归档 outputs/run_v41.json 的 digests.scenario + metrics）。
# CI checkout 不含被 .gitignore 忽略的 outputs/；快照 provenance 见 fixture 文件。
FIXED_EXPECTED = (ROOT / "tests" / "fixtures"
                  / "v41_flash_dt_verification_expected.json")

OPS_CSV_COLUMNS = (
    "stage", "layer", "op", "exec_count", "flops", "weight_read",
    "act_read", "act_write", "kv_read", "kv_write", "scope",
    "formula_id", "t_ms", "bound", "shape_note",
)


def _cli_env(io_encoding="utf-8"):
    """显式设定子进程启动时的 stdio 编码，不继承 runner 的 locale 默认。"""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = io_encoding
    env.pop("PYTHONUTF8", None)  # 避免 UTF-8 mode 掩盖非 UTF-8 stdio 条件
    return env


def _assert_ops_csv(csv_path, ops):
    """CSV 表头/行数/逐列值必须等于 RunResult.ops 的忠实序列化。"""
    with Path(csv_path).open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    assert tuple(rows[0]) == OPS_CSV_COLUMNS
    data_rows = rows[1:]
    assert len(data_rows) == len(ops)
    for index, (op, row) in enumerate(zip(ops, data_rows)):
        for column, cell in zip(OPS_CSV_COLUMNS, row):
            expected = str(op.get(column, "")).replace(",", ";").replace("\n", " ")
            assert cell == expected, (index, column, cell, expected)


@pytest.fixture(scope="module")
def cli_run(tmp_path_factory):
    """一次 CLI run 的落盘结果：后续导出测试复用同一份 RunResult。"""
    out = tmp_path_factory.mktemp("cli") / "run.json"
    proc = subprocess.run(
        [sys.executable, "-m", "simverse.cli", "run", str(FIXED), "-o", str(out)],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8",
        env=_cli_env())
    assert proc.returncode == 0, proc.stderr
    return {"path": out, "result": json.loads(out.read_text(encoding="utf-8"))}


@pytest.fixture(scope="module")
def cli_result(cli_run):
    return cli_run["result"]


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


def test_csv_export_matches_runresult(cli_run, tmp_path):
    """导出 CSV 行数与列值 = RunResult.ops 的忠实序列化（无第二套口径）。

    输入是本次 `run` 新落盘的 RunResult（tmp_path），不是被 Git 忽略的
    outputs/ 存档；逐列对照 RunResult.ops，而不是用导出器自证导出器。
    子进程 stdio 显式 UTF-8，父进程按同一编码解码，不依赖 runner locale。
    """
    csv_path = tmp_path / "ops.csv"
    proc = subprocess.run(
        [sys.executable, "-m", "simverse.cli", "export",
         str(cli_run["path"]), "--csv", str(csv_path)],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8",
        env=_cli_env())
    assert proc.returncode == 0, proc.stderr
    _assert_ops_csv(csv_path, cli_run["result"]["ops"])


def test_export_survives_non_utf8_stdout(cli_run, tmp_path):
    """非 UTF-8 子进程 stdio 下，export 写文件后不得因状态文本崩溃。

    远端 Windows runner 的管道 stdio 使用系统 ANSI 代码页（失败 run 中为
    cp1252），而 CLI 成功提示含中文。此处显式把子进程 stdio 固定为 cp1252
    （并移除 PYTHONUTF8），要求 export 仍 exit 0，且 CSV 仍逐列忠实。
    """
    csv_path = tmp_path / "ops_cp1252.csv"
    proc = subprocess.run(
        [sys.executable, "-m", "simverse.cli", "export",
         str(cli_run["path"]), "--csv", str(csv_path)],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8",
        errors="replace", env=_cli_env("cp1252"))
    assert proc.returncode == 0, proc.stderr
    _assert_ops_csv(csv_path, cli_run["result"]["ops"])


def test_fixed_scenario_replay_digest_stable():
    """固定场景重放：digest/metrics 与纳入版本库的历史快照一致。

    快照只含 archived digests.scenario + metrics（无完整 ops/timeline），
    绑定原始归档 SHA256 与场景文件 LF 归一化 SHA256；期望值来自历史归档而非
    本次 run()，因此 CI（无 ignored outputs/）可独立复现同一断言。
    """
    from simverse.contracts import ScenarioSpec
    from simverse.engine import run
    expected = json.loads(FIXED_EXPECTED.read_text(encoding="utf-8"))
    provenance = expected["provenance"]
    # 按 LF 归一化哈希，避免 Git checkout 的 CRLF 转换把内容身份变成环境假阴性。
    scenario_lf = FIXED.read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(scenario_lf).hexdigest() == \
        provenance["scenario_file_sha256_lf"], \
        "固定场景已变更；须复核并显式重建历史快照 fixture"
    assert provenance["scenario_digest"] == expected["digests"]["scenario"]
    scenario = ScenarioSpec.from_json(json.loads(FIXED.read_text(encoding="utf-8")))
    r = run(scenario)
    assert r.scenario_digest == expected["digests"]["scenario"]
    metrics = r.to_json()["metrics"]
    # 独立算术期望（固定场景 alpha=0.75, K=4, B=1）：不从 run() 或快照反推。
    assert metrics["E_committed_len"] == pytest.approx(
        sum(0.75 ** i for i in range(5)))              # 3.05078125
    assert metrics["P_full"] == pytest.approx(0.75 ** 4)  # 0.31640625
    assert metrics["E_draft_step0_tokens"] == pytest.approx(1 + 0.75 ** 4)
    assert metrics["verify_tokens_M"] == 1 * (4 + 1)
    assert _norm(metrics) == _norm(expected["metrics"])


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
