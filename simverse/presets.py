"""共享场景预设（R10d）：CLI、场景目录 API 与工作台的同一份默认输入。

背景：默认 Qwen3.6+DFlash 场景此前硬编码在 CLI 入口（cli.V2_DEFAULT_SCENARIO），
活跃场景目录没有同一份预设，三处入口各说各话。本模块把预设固化为
``assets/scenarios/qwen36_dflash_default.json``（与原 CLI 字典逐键一致——
notes 等非默认键不入文件，保证场景 digest 与历史默认 run 可回放对齐）。

* v0.2 预设目录 = assets/scenarios/ 下 schema_version 为 "0.2" 且模型在
  活跃目录（PRODUCT_MODELS + 绑定 DFlash）的 JSON 文件；
* v0.1 的 V4.1 固定验证场景经 archive run 与 ScenarioSpec 重放，历史
  v0.2 成本场景由离线工具显式注入归档目录；
* 解码/校验统一走 decode_scenario_v2（本包 api_io 模块），run 与 sweep
  共用，非法输入携带可定位字段路径。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .assets import scenario_dir

DEFAULT_PRESET_ID = "qwen36_dflash_default"


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_preset(preset_id: str) -> Dict[str, Any]:
    """按 id（文件名去 .json）读取 v0.2 预设原始字典。"""
    if not preset_id or "/" in preset_id or "\\" in preset_id or ".." in preset_id:
        raise ValueError(f"非法预设 id：{preset_id!r}")
    path = scenario_dir() / f"{preset_id}.json"
    if not path.is_file():
        raise ValueError(f"未知预设 {preset_id}（目录：assets/scenarios/*.json）")
    data = _load_json(path)
    if data.get("schema_version") != "0.2":
        raise ValueError(f"{preset_id} 不是 v0.2 预设（schema_version="
                         f"{data.get('schema_version')!r}）；v0.1 场景请使用 archive run")
    return data


def default_preset() -> Dict[str, Any]:
    """共享默认预设（原 CLI V2_DEFAULT_SCENARIO 的固化副本）。"""
    return load_preset(DEFAULT_PRESET_ID)


def list_presets() -> List[Dict[str, Any]]:
    """活跃 v0.2 预设目录（目录 API/工作台选择器共用）。

    只列活跃模型；archived 标记或目录外模型不进选择（与 G8 边界一致）。
    每条带 scenario_digest，供三入口核对"同一份预设"的身份。
    """
    from .model_ir.catalog import PRODUCT_MODELS
    from .scenario import ScenarioV2

    out: List[Dict[str, Any]] = []
    root = scenario_dir()
    for p in sorted(root.glob("*.json")):
        try:
            data = _load_json(p)
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("schema_version") != "0.2":
            continue
        model_id = data.get("model_id")
        draft_id = data.get("draft_id")
        if data.get("archived") or model_id not in PRODUCT_MODELS:
            continue
        if draft_id is not None and draft_id != "dflash_qwen36_35b_a3b":
            continue
        try:
            digest = ScenarioV2.from_json(dict(data)).scenario_digest()
        except ValueError:
            continue
        out.append({"id": p.stem, "name": data.get("name", p.stem),
                    "schema_version": "0.2", "scenario_digest": digest,
                    "scenario": data})
    return out


def find_preset_by_digest(digest: str) -> Optional[str]:
    for entry in list_presets():
        if entry["scenario_digest"] == digest:
            return entry["id"]
    return None
