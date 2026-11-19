"""Archived v0.1 workbench and API routes."""
import json

from flask import Blueprint, current_app, jsonify, request, send_from_directory

from ..assets import scenario_dir
from ..contracts.legacy import ScenarioSpec

legacy = Blueprint("legacy", __name__)


def _scenarios_index():
    """只列活跃 v0.1 场景；历史模型/标记 archived 的场景保留文件但不进
    选择。v0.2 预设（如 qwen36_dflash_default）走 /api/v2/catalog 的
    scenarios 段（R10d），不混入 legacy 目录（ScenarioSpec 读不了 v0.2）。"""
    from ..model_ir.catalog import PRODUCT_MODELS
    out = {}
    for p in sorted(scenario_dir().glob("*.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        if data.get("schema_version", "0.1") != "0.1":
            continue
        if data.get("archived") or data.get("model_id") not in PRODUCT_MODELS:
            continue
        out[p.stem] = {"name": data.get("name", p.stem), "notes": data.get("notes", ""),
                       "scenario": data}
    return out


@legacy.get("/legacy")
def legacy_index():
    return send_from_directory(current_app.static_folder, "index.html")


@legacy.get("/api/registry")
def api_registry():
    from ..formulas import formulas_json
    from ..models import registry_json
    from ..devices import DEVICES, LINKS
    from ..model_ir.catalog import PRODUCT_MODELS
    reg = registry_json()
    reg["models"] = {k: v for k, v in reg["models"].items()
                     if k in PRODUCT_MODELS}
    reg["drafts"] = {k: v for k, v in reg["drafts"].items()
                     if k in PRODUCT_MODELS or k == "dspark"}
    return jsonify({
        "schema_version": "0.1",
        "registry": reg,
        "devices": {k: d.to_json() for k, d in DEVICES.items()},
        "links": {k: dict(v.__dict__) for k, v in LINKS.items()},
        "formulas": formulas_json()["formulas"],
    })


@legacy.get("/api/scenarios")
def api_scenarios():
    return jsonify({"schema_version": "0.1", "scenarios": _scenarios_index()})


@legacy.post("/api/run")
def api_run():
    from ..engine import run

    body = request.get_json(force=True)
    try:
        if "scenario_file" in body:
            path = scenario_dir() / (body["scenario_file"] + ".json")
            scenario = ScenarioSpec.from_json(json.loads(path.read_text(encoding="utf-8")))
        else:
            scenario = ScenarioSpec.from_json(body)
        from ..model_ir.catalog import PRODUCT_MODELS
        if scenario.model_id not in PRODUCT_MODELS:
            return jsonify({"error": f"历史/合成模型 {scenario.model_id} 不在"
                                     "活跃 API 目录；仅作离线归档"}), 400
        result = run(scenario).to_json()
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@legacy.post("/api/sweep")
def api_sweep():
    from ..sweep import sweep_suite
    body = request.get_json(force=True)
    try:
        scenario = ScenarioSpec.from_json(body)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    from ..model_ir.catalog import PRODUCT_MODELS
    if scenario.model_id not in PRODUCT_MODELS:
        return jsonify({"error": f"历史/合成模型 {scenario.model_id} 不在"
                                 "活跃 API 目录；仅作离线归档"}), 400
    return jsonify(sweep_suite(scenario))
