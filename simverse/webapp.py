"""Flask Web 工作台（v0.2 生产 + v0.1 legacy 双入口）。

铁律：前端零物理计算——所有数字来自 /api/v2/run 的 ResultBundle；本文件只做
入口转发。生产工作台 /（static/workbench.html）；v0.1 简表 /legacy；
原型演示仍在 docs/rebuild-20261106/prototype/（不进生产路径）。
"""
from __future__ import annotations

import json
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from .contracts import ScenarioSpec, js_safe
from .engine import run
from .formulas import formulas_json
from .models import registry_json

ROOT = Path(__file__).resolve().parent.parent
SCENARIO_DIR = ROOT / "assets" / "scenarios"

app = Flask(__name__, static_folder=str(Path(__file__).parent / "static"),
            static_url_path="/static")


def _scenarios_index():
    out = {}
    for p in sorted(SCENARIO_DIR.glob("*.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        out[p.stem] = {"name": data.get("name", p.stem), "notes": data.get("notes", ""),
                       "scenario": data}
    return out


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "workbench.html")


@app.get("/legacy")
def legacy_index():
    return send_from_directory(app.static_folder, "index.html")


# ---------------------------------------------------------------------------
# v0.2 生产 API（ResultBundle 管线）
# ---------------------------------------------------------------------------


@app.get("/api/v2/catalog")
def api_v2_catalog():
    from .model_ir import catalog_json
    from .devices import DEVICES, LINKS
    from .methods.plans import UNSUPPORTED
    from .validation import registry_json
    from .importers import source_map_json
    from .cost.profile_store import PROFILE_SETS
    return jsonify(js_safe({
        "schema_version": "0.2",
        "catalog": catalog_json(),
        "devices": {k: d.to_json() for k, d in DEVICES.items()},
        "links": {k: dict(v.__dict__) for k, v in LINKS.items()},
        "methods": {"supported": ["baseline", "dflash_v1", "mtp"],
                    "research": ["chain_draft", "chain_moe_proxy"],
                    "unsupported": UNSUPPORTED},
        "profile_sets": sorted(PROFILE_SETS),
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
    """从最近一次 run 的 body 重放导出（POST /api/v2/run 已含全部字段；
    CSV 由前端从同一 bundle 生成前调用此端点登记 manifest——保持单一来源）。"""
    return jsonify({"error": "请 POST /api/v2/run 并使用返回的 bundle 导出；"
                             "服务端不维护会话状态"}), 400


# ---------------------------------------------------------------------------
# v0.1 legacy API（保留，供旧 UI 与测试）
# ---------------------------------------------------------------------------


@app.get("/api/registry")
def api_registry():
    from .devices import DEVICES, LINKS
    return jsonify({
        "schema_version": "0.1",
        "registry": registry_json(),
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
    return jsonify(sweep_suite(scenario))
