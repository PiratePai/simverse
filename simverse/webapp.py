"""Flask Web 工作台（v0.2 生产 + v0.1 legacy 双入口）。

铁律：前端零物理计算——所有数字来自 /api/v2/run 的 ResultBundle；本文件只做
入口转发。生产工作台 /（static/workbench.html）；v0.1 简表 /legacy；
原型演示仍在 docs/rebuild-20261106/prototype/（不进生产路径，经 /docs/ 只读）。
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from .contracts import ScenarioSpec, js_safe
from .engine import run
from .formulas import formulas_json
from .models import registry_json

ROOT = Path(__file__).resolve().parent.parent
SCENARIO_DIR = ROOT / "assets" / "scenarios"
# 工作台顶栏文档链接的只读根（send_from_directory 自带穿越/绝对路径拒绝，
# 只开放仓库 docs/ 一层，不提供任意本地文件访问）
DOCS_DIR = ROOT / "docs"


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


def _scenarios_index():
    """只列活跃场景；历史模型/标记 archived 的场景保留文件但不进选择。"""
    from .model_ir.catalog import PRODUCT_MODELS
    out = {}
    for p in sorted(SCENARIO_DIR.glob("*.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        if data.get("archived") or data.get("model_id") not in PRODUCT_MODELS:
            continue
        out[p.stem] = {"name": data.get("name", p.stem), "notes": data.get("notes", ""),
                       "scenario": data}
    return out


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "workbench.html")


@app.get("/legacy")
def legacy_index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/docs/<path:fn>")
def docs_static(fn):
    """docs/ 只读静态路由（顶栏四个文档链接 + 原型存档的相对资源）。

    只开放仓库 docs/ 目录；Markdown 用最小 <pre> 包装（无渲染依赖、无 JS），
    其余文件按原样发送。不注册任何 docs 之外的文件入口。
    """
    path = (DOCS_DIR / fn).resolve()
    if not path.is_relative_to(DOCS_DIR.resolve()):
        return jsonify({"error": "只允许访问 docs 目录"}), 404
    if fn.lower().endswith(".md"):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return jsonify({"error": f"文档不存在或不可读：{fn}"}), 404
        return (
            "<!DOCTYPE html><html lang=zh><head><meta charset=utf-8>"
            "<title>simverse docs · " + html.escape(fn) + "</title>"
            "<body style=font-family:Consolas,monospace;margin:24px;"
            "white-space:pre-wrap;word-break:break-word;line-height:1.55>"
            + _linked_markdown(text) + "</body></html>"
        ), 200, {"Content-Type": "text/html; charset=utf-8"}
    return send_from_directory(DOCS_DIR, fn)


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
        "evidence": registry_json(),
        "source_map": source_map_json(),
    }))


@app.post("/api/v2/run")
def api_v2_run():
    from .pipeline import run_v2
    from .scenario import ScenarioV2
    body = request.get_json(force=True)
    try:
        scenario = ScenarioV2.from_json(body)
        bundle = run_v2(scenario)
        return jsonify(js_safe(bundle.to_json()))
    except (ValueError, KeyError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/v2/sweep")
def api_v2_sweep():
    from .experiments import sweep_suite
    from .scenario import ScenarioV2
    body = request.get_json(force=True)
    try:
        scenario = ScenarioV2.from_json(body)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(js_safe(sweep_suite(scenario)))


@app.get("/api/v2/export/<fmt>")
def api_v2_export(fmt):
    """已移除（2026-11-17 工作台审查）：该端点从未有调用方且恒返 400，
    旧注释“前端导出前在此登记 manifest”与实际行为不符。导出统一从
    POST /api/v2/run 返回的同一 ResultBundle 派生（前端按钮 / CLI
    export-v2），服务端不维护“最近一次运行”会话状态。保留 410 明示弃用。
    """
    return jsonify({"error": "该端点已弃用（无调用方、恒不可用）。请 POST "
                             "/api/v2/run 并从返回的 bundle 导出（前端按钮或 "
                             "CLI export-v2）；服务端无状态导出。"}), 410


# ---------------------------------------------------------------------------
# v0.1 legacy API（保留，供旧 UI 与测试）
# ---------------------------------------------------------------------------


@app.get("/api/registry")
def api_registry():
    from .devices import DEVICES, LINKS
    from .model_ir.catalog import PRODUCT_MODELS
    reg = registry_json()
    reg["models"] = {k: v for k, v in reg["models"].items()
                     if k in PRODUCT_MODELS}
    reg["drafts"] = {k: v for k, v in reg["drafts"].items()
                     if k in PRODUCT_MODELS or k == "dspark"}
    return jsonify({
        "schema_version": "0.1",
        "registry": reg,
        "devices": {k: d.to_json() for k, v in DEVICES.items()},
        "links": {k: dict(v.__dict__) for k, v in LINKS.items()},
        "formulas": formulas_json()["formulas"],
    })


@app.get("/api/scenarios")
def api_scenarios():
    return jsonify({"schema_version": "0.1", "scenarios": _scenarios_index()})


@app.post("/api/run")
def api_run():
    body = request.get_json(force=True)
    try:
        if "scenario_file" in body:
            path = SCENARIO_DIR / (body["scenario_file"] + ".json")
            scenario = ScenarioSpec.from_json(json.loads(path.read_text(encoding="utf-8")))
        else:
            scenario = ScenarioSpec.from_json(body)
        from .model_ir.catalog import PRODUCT_MODELS
        if scenario.model_id not in PRODUCT_MODELS:
            return jsonify({"error": f"历史/合成模型 {scenario.model_id} 不在"
                                     "活跃 API 目录；仅作离线归档"}), 400
        result = run(scenario).to_json()
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/sweep")
def api_sweep():
    from .sweep import sweep_suite
    body = request.get_json(force=True)
    try:
        scenario = ScenarioSpec.from_json(body)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    from .model_ir.catalog import PRODUCT_MODELS
    if scenario.model_id not in PRODUCT_MODELS:
        return jsonify({"error": f"历史/合成模型 {scenario.model_id} 不在"
                                 "活跃 API 目录；仅作离线归档"}), 400
    return jsonify(sweep_suite(scenario))
