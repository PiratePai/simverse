"""Ascend profiling 只读审计报告 → Flask API / 静态工作台接线测试。

覆盖三件事：
  * /api/v2/catalog 暴露 `profile_audit` 索引，同时 `profile_sets` 仍为空
    （审计包 ≠ cost profile，不得进入 #sel-profile 选择器）；
  * /api/v2/profile-audit/<set_id> 走 js_safe/js_restore 无损往返，恒为
    not-admitted、selected_ms=None（不产生成本/排名）；
  * 工作台静态文件只做展示/导出接线（前端零物理计算）。

审计产物由 tools/import_ascend_profiles.py 生成、哈希由 manifest 校验。
本测试无网络、无 NPU、无浏览器：只读仓库内的审计包与静态文本。
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simverse.contracts import js_restore  # noqa: E402

AUDIT_SET = "ascend-910b-profile-audit-v1"
STATIC = ROOT / "simverse" / "static"
PRODUCT_MODELS = {"qwen36_35b_a3b", "deepseek_v41_flash"}


def _client():
    from simverse.webapp import app
    app.config["TESTING"] = True
    return app.test_client()


def test_catalog_exposes_profile_audit_index_and_keeps_profile_sets_empty():
    """审计索引是独立只读键：profile_sets 仍为空、模型集合不变。"""
    c = _client()
    r = c.get("/api/v2/catalog")
    assert r.status_code == 200
    cat = js_restore(r.get_json())
    audit = cat["profile_audit"]
    assert isinstance(audit, list) and audit
    first = audit[0]
    assert first["set_id"] == AUDIT_SET
    assert first["calibration_admission"] == "not-admitted"
    assert first["admitted"] is False
    assert first["report_endpoint"] == f"/api/v2/profile-audit/{AUDIT_SET}"
    # 审计包不得混入校准/profile 选择
    assert cat["profile_sets"] == []
    assert set(cat["catalog"]["models"]) == PRODUCT_MODELS


def test_profile_audit_report_endpoint_js_safe_roundtrip():
    """完整报告：js_safe 大整数契约往返、6 个 run 并列观察、无成本结论。"""
    c = _client()
    r = c.get(f"/api/v2/profile-audit/{AUDIT_SET}")
    assert r.status_code == 200
    report = js_restore(r.get_json())
    assert report["set_id"] == AUDIT_SET
    assert report["admission"]["calibration_admission"] == "not-admitted"
    assert report["admission"]["admitted"] is False
    assert report["candidate_calibration"]["selected_ms"] is None
    runs = report["runs"]
    assert len(runs) == 6
    for run in runs:
        assert run["observed_kernel_us"] > 0
        assert run.get("analytic_base_ms") is None
    # 契约：入 JSON 的数值必须全有限（NaN/Infinity 禁止）
    json.dumps(report, allow_nan=False)


def test_profile_audit_report_domain_reasons():
    """域门禁原因由后端给出（跨模型/设备/TP），前端只展示。"""
    c = _client()
    url = f"/api/v2/profile-audit/{AUDIT_SET}"
    r = c.get(url, query_string={"model_id": "deepseek_v41_flash"})
    assert r.status_code == 200
    report = js_restore(r.get_json())
    assert any("跨模型拒绝" in x for x in report["admission"]["reasons"])
    assert report["candidate_calibration"]["selected_ms"] is None
    r = c.get(url, query_string={"device_id": "h200_sxm"})
    assert r.status_code == 200
    report = js_restore(r.get_json())
    reasons = "\n".join(report["admission"]["reasons"])
    assert "设备" in reasons and "h200_sxm" in reasons
    r = c.get(url, query_string={"tp": 8})
    assert r.status_code == 200
    report = js_restore(r.get_json())
    assert report["candidate_calibration"]["selected_ms"] is None
    json.dumps(report, allow_nan=False)


def test_profile_audit_unknown_set_404_and_bad_tp_400():
    """未知审计包/完整性失败 → 404；tp 非整数 → 400（均带 error 原因）。"""
    c = _client()
    r = c.get("/api/v2/profile-audit/ascend-910b-profile-audit-does-not-exist")
    assert r.status_code == 404
    assert r.get_json().get("error")
    r = c.get(f"/api/v2/profile-audit/{AUDIT_SET}", query_string={"tp": "abc"})
    assert r.status_code == 400
    assert r.get_json().get("error")


_SEL_PROFILE_RE = re.compile(r'<select id="sel-profile"[^>]*>(.*?)</select>', re.S)


def test_workbench_static_audit_wiring():
    """静态工作台：只读审计抽屉 + 详情/导出接线；审计包不进 profile 选择器。"""
    html = (STATIC / "workbench.html").read_text(encoding="utf-8")
    js = (STATIC / "workbench.js").read_text(encoding="utf-8")
    assert 'id="audit-body"' in html
    assert 'id="btn-export-audit"' in html
    assert "Ascend profiling 审计（未准入 · 只读）" in html
    assert "/api/v2/profile-audit/" in js
    assert "renderProfileAudit" in js
    assert "btn-export-audit" in js
    assert "esc(" in js
    assert "cat.profile_audit || []" in js
    # 审计 set_id 不得写死进 HTML/选择器（前端只消费后端索引）
    assert AUDIT_SET not in html
    match = _SEL_PROFILE_RE.search(html)
    assert match, "找不到 #sel-profile 选择器"
    select_block = match.group(0)
    assert "ascend-910b-profile-audit" not in select_block
    assert "profile_audit" not in select_block


def _find_node():
    """node 优先取 PATH；否则在 DSH 运行时布局中做一次相对发现
    （.../dependencies/python 与 .../dependencies/node/bin 并列），
    都没有则跳过（JS 语法另由 CI/浏览器验收覆盖）。"""
    exe = shutil.which("node")
    if exe:
        return exe
    sibling = Path(sys.executable).resolve().parent.parent / "node" / "bin"
    for name in ("node.exe", "node"):
        cand = sibling / name
        if cand.is_file():
            return str(cand)
    return None


def test_js_syntax_check():
    """workbench.js 语法可解析（node 可执行时；无 node 则跳过）。"""
    node = _find_node()
    if not node:
        pytest.skip("node 不可用（PATH 无 node，且未发现 DSH 运行时 node）")
    out = subprocess.run([node, "--check", "simverse/static/workbench.js"],
                         cwd=str(ROOT), capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
