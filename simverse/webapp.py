"""Flask Web 工作台（决策 D-2：Flask 延续两仓现状；flask-smorest 属 M0 后续，先显式路由）。

铁律（方案 §7 防冗余规则 2）：前端零物理计算——所有数字来自 /api/run 的
RunResult；本文件只做入口转发，不复制公式。
"""
from __future__ import annotations

import json
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from .contracts import ScenarioSpec
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
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/registry")
def api_registry():
    from .devices import DEVICES, LINKS
    return jsonify({
        "schema_version": "0.1",
        "registry": registry_json(),
        "devices": {k: d.to_json() for k, d in DEVICES.items()},
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
