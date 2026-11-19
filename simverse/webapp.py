"""Flask Web 工作台（v0.2 生产 + v0.1 legacy 双入口）。

铁律：前端零物理计算——所有数字来自 /api/v2/run 的 ResultBundle；本文件只做
入口转发。生产工作台 /（static/workbench.html）；v0.1 简表 /legacy；
原型演示仍在 docs/rebuild-20261106/prototype/（不进生产路径，经 /docs/ 只读）。
"""
from __future__ import annotations

import html
import re
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory

from .assets import docs_dir, packaged_data_dir, repo_root
from .contracts import js_restore, js_safe
from .exports import _groups_csv_v2, _ops_csv_v2
from .legacy.web import legacy

# 工作台顶栏文档链接的只读根（send_from_directory 自带穿越/绝对路径拒绝，
# 只开放仓库 docs/ 一层，不提供任意本地文件访问）
DOCS_DIR = docs_dir()

# 状态/验证文档指向的根目录文档白名单（R10b）：docs/ 内的相对链接
# ``[ROADMAP](../ROADMAP.md)`` 会落到站点根；只显式开放这两个被引用的
# 根文档，未列出的根路径仍返回 404。
ROOT_DOC_WHITELIST = ("ROADMAP.md", "AGENTS.md")


def _linked_markdown(text: str) -> str:
    """保留原文排版，并让文档内相对链接可点击。"""
    escaped = html.escape(text)

    def link(match):
        target = html.unescape(match.group(2))
        if not target or target.startswith(("/", "\\")) or ":" in target:
            return match.group(0)
        return (f'<a href="{html.escape(target, quote=True)}">'
                f'{match.group(1)}</a>')

    return re.sub(r"\[([^\]\n]+)\]\(([^)\s]+)\)", link, escaped)

app = Flask(__name__, static_folder=str(Path(__file__).parent / "static"),
            static_url_path="/static")
app.register_blueprint(legacy)


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "workbench.html")


@app.get("/docs/<path:fn>")
def docs_static(fn):
    """docs/ 只读静态路由（顶栏四个文档链接 + 原型存档的相对资源）。

    只开放仓库 docs/ 目录；Markdown 用最小 <pre> 包装（无渲染依赖、无 JS），
    其余文件按原样发送。docs/ 内文档的 ``../ROADMAP.md`` 等根文档相对
    链接由 root_doc 路由的白名单承接（R10b）。
    """
    path = (DOCS_DIR / fn).resolve()
    if not path.is_relative_to(DOCS_DIR.resolve()):
        return jsonify({"error": "只允许访问 docs 目录"}), 404
    if fn.lower().endswith(".md"):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return jsonify({"error": f"文档不存在或不可读：{fn}"}), 404
        return _markdown_page(fn, text)
    return send_from_directory(DOCS_DIR, fn)


@app.route("/<root_doc>")
def root_doc(root_doc):
    """根目录文档白名单（R10b）。

    状态/验证文档（FOCUS_STATUS、IMPLEMENTATION_STATUS、
    ascend-validation-plan 等）用 ``[ROADMAP](../ROADMAP.md)`` 引用仓库根
    文档，浏览器把它们解析到站点根 → 此前 404。只显式开放被引用的两个
    根文档（ROADMAP.md/AGENTS.md，README 未被引用不入白名单）；其余根
    路径仍返回 404。wheel 使用随包的文档。
    """
    if root_doc not in ROOT_DOC_WHITELIST:
        return jsonify({"error": f"根路径未开放：{root_doc}"
                                 f"（白名单 {list(ROOT_DOC_WHITELIST)}）"}), 404
    path = (repo_root() or packaged_data_dir()) / root_doc
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return jsonify({"error": f"文档不存在或不可读：{root_doc}"}), 404
    return _markdown_page(root_doc, text)


def _markdown_page(title: str, text: str):
    """Markdown 只读页（与 docs_static 同一渲染方式，根文档复用）。"""
    return (
        "<!DOCTYPE html><html lang=zh><head><meta charset=utf-8>"
        "<title>simverse docs · " + html.escape(title) + "</title>"
        "<body style=font-family:Consolas,monospace;margin=24px;"
        "white-space:pre-wrap;word-break:break-word;line-height:1.55>"
        + _linked_markdown(text) + "</body></html>"
    ), 200, {"Content-Type": "text/html; charset=utf-8"}


# ---------------------------------------------------------------------------
# v0.2 生产 API（ResultBundle 管线）
# ---------------------------------------------------------------------------


@app.get("/api/v2/catalog")
def api_v2_catalog():
    from .cost.records import catalog as record_catalog
    from .model_ir import catalog_json
    from .devices import DEVICES, LINKS
    from .methods.plans import UNSUPPORTED
    from .validation import registry_json
    from .importers import source_map_json
    from .cost.profile_store import (PRODUCT_PROFILE_SETS, archived_profiles_json,
                                     profile_model_id)
    from .presets import list_presets
    return jsonify(js_safe({
        "schema_version": "0.2",
        "catalog": catalog_json(),
        "devices": {k: d.to_json() for k, d in DEVICES.items()},
        "links": {k: dict(v.__dict__) for k, v in LINKS.items()},
        "methods": {"supported": ["baseline", "dflash_v1"],
                    "research": ["mtp"],
                    "unsupported": UNSUPPORTED},
        "profile_sets": sorted(PRODUCT_PROFILE_SETS),
        "cost_record_sets": record_catalog(),
        "archived_profiles": archived_profiles_json(),
        "profile_bindings": {sid: profile_model_id(sid)
                             for sid in sorted(PRODUCT_PROFILE_SETS)},
        "scenarios": list_presets(),
        "evidence": registry_json(),
        "source_map": source_map_json(),
    }))


@app.post("/api/v2/run")
def api_v2_run():
    """v0.2 单场景运行。

    输入校验与 /api/v2/sweep 共用 api_io.decode_scenario_v2（R10d）：
    结构/类型/取值非法 → 结构化 400（带字段路径）；合法但不受支持的
    场景照常进管线（bundle.status=unsupported_*，200）；未预期异常保持
    服务端 500 语义，不在入口吞掉。
    """
    from .api_io import ScenarioInputError, decode_scenario_v2, error_payload
    from .pipeline import run_v2
    body = request.get_json(force=True, silent=True)
    if body is None:
        return jsonify({"error": "请求体不是合法 JSON（或为空）", "field": "$"}), 400
    try:
        scenario = decode_scenario_v2(body)
        bundle = run_v2(scenario)
        return jsonify(js_safe(bundle.to_json()))
    except ScenarioInputError as exc:
        return jsonify(error_payload(exc)), 400


@app.post("/api/v2/sweep")
def api_v2_sweep():
    from .api_io import ScenarioInputError, decode_scenario_v2, error_payload
    from .experiments import sweep_suite
    body = request.get_json(force=True, silent=True)
    if body is None:
        return jsonify({"error": "请求体不是合法 JSON（或为空）", "field": "$"}), 400
    try:
        scenario = decode_scenario_v2(body)
        return jsonify(js_safe(sweep_suite(scenario)))
    except ScenarioInputError as exc:
        return jsonify(error_payload(exc)), 400


@app.get("/api/v2/export/<fmt>")
def api_v2_export_deprecated(fmt):
    """Retain the 410 from the former GET endpoint."""
    return jsonify({"error": "GET 导出端点已弃用。工作台通过 POST 提交已有结果，"
                             "CLI 可用 export-v2 从文件导出。"}), 410


@app.post("/api/v2/export/<fmt>")
def api_v2_export(fmt):
    """Serialize the supplied result without re-running or storing it."""
    render = {"ops": _ops_csv_v2, "groups": _groups_csv_v2}.get(fmt)
    if render is None:
        return jsonify({"error": "未知 CSV 导出类型"}), 404
    body = request.get_json(silent=True)
    if not isinstance(body, dict) or body.get("schema_version") != "0.2":
        return jsonify({"error": "需要 v0.2 ResultBundle"}), 400
    try:
        csv_text = render(js_restore(body))
    except (AttributeError, KeyError, TypeError, ValueError):
        return jsonify({"error": "结果中的 CSV 数据格式无效"}), 400
    return Response(csv_text, mimetype="text/csv")
