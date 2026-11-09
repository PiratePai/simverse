"""参数扫描与收益边界提取（方案 §4.7 对比页数据来源；矩阵 ≤200 格/次）。

扫描维度：acceptance α / K / ctx / batch / 链路带宽 / 草稿设备带宽。
每格 = 独立 run(scenario') 的对比行（不塞单一大对象）；输出带场景谱系
（base scenario digest + 覆盖字段），可由 CLI/Web 同入口复算。
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Dict, List

from .contracts import ScenarioSpec
from .engine import run


def _variant(base: ScenarioSpec, **over) -> ScenarioSpec:
    fields = {f: getattr(base, f) for f in ScenarioSpec.__dataclass_fields__}
    for key, value in over.items():
        if key == "acceptance":
            fields["acceptance"] = value
        else:
            fields[key] = value
    return ScenarioSpec(**fields).validate()


def sweep_1d(base: ScenarioSpec, field: str, values: List[Any],
             deployments=("baseline", "colocated", "dt_serial", "dt_pipeline_ideal")) -> dict:
    """一维扫描：每点输出各部署 TPOT/speedup + 关键中间量。"""
    points = []
    for v in values:
        if field == "link_bw_gb":
            sc = _variant(base, link="custom")
            # 链路带宽覆盖：直接改 devices 注册表会污染全局，改为单次注入
            from . import devices as DEV
            saved = DEV.LINKS["custom"]
            DEV.LINKS["custom"] = replace(saved, bw=float(v) * 1e9,
                                          label=f"自定义 {v} GB/s")
            try:
                res = run(sc)
            finally:
                DEV.LINKS["custom"] = saved
        elif field == "draft_bw_tbs":
            sc = _variant(base, draft_device="custom_draft")
            from . import devices as DEV
            if "custom_draft" not in DEV.DEVICES:
                DEV.DEVICES["custom_draft"] = replace(
                    DEV.DEVICES["l40"], id="custom_draft", label=f"草稿设备 {v} TB/s")
            saved = DEV.DEVICES["custom_draft"]
            DEV.DEVICES["custom_draft"] = replace(saved, hbm_bw=float(v) * 1e12,
                                                  label=f"草稿设备 {v} TB/s")
            try:
                res = run(sc)
            finally:
                DEV.DEVICES["custom_draft"] = saved
        elif field == "alpha":
            sc = _variant(base, acceptance={"kind": "chain_alpha", "chain_alpha": float(v)})
            res = run(sc)
        else:
            sc = _variant(base, **{field: int(v)})
            res = run(sc)
        point = {"value": v,
                 "E_committed_len": res.metrics.get("E_committed_len"),
                 "scenario_digest": res.scenario_digest,
                 "deployments": {c["deployment"]: {"tpot_ms": c["tpot_ms"],
                                                   "speedup": c["speedup"]}
                                 for c in res.comparisons if c["deployment"] in deployments}}
        points.append(point)
    return {"schema_version": "0.1", "base_scenario": base.to_json(), "field": field,
            "points": points}


def find_crossover(points: dict, deployment: str, other: str = "baseline") -> dict:
    """收益反转边界：speedup 曲线与 1（或与另一部署曲线）的交点（线性插值）。"""
    xs, ys, zs = [], [], []
    for p in points["points"]:
        d = p["deployments"].get(deployment)
        o = p["deployments"].get(other)
        if not d:
            continue
        xs.append(float(p["value"]))
        ys.append(d["speedup"])
        zs.append(o["speedup"] if o else 1.0)
    for i in range(1, len(xs)):
        f0, f1 = ys[i - 1] - zs[i - 1], ys[i] - zs[i]
        if f0 == 0:
            return {"x": xs[i - 1], "deployment": deployment, "vs": other}
        if f0 * f1 < 0:
            t = f0 / (f0 - f1)
            return {"x": xs[i - 1] + t * (xs[i] - xs[i - 1]), "deployment": deployment,
                    "vs": other}
    return {"x": None, "deployment": deployment, "vs": other,
            "note": "扫描区间内无交点（全程盈利或全程亏损）"}


def sweep_suite(base: ScenarioSpec) -> dict:
    """交付用扫描套件：α / K / ctx / B / 链路带宽 / 草稿带宽 + 边界提取。"""
    out: Dict[str, Any] = {"schema_version": "0.1", "base_scenario": base.to_json()}
    out["alpha"] = sweep_1d(base, "alpha", [round(0.30 + 0.05 * i, 2) for i in range(14)])
    out["k"] = sweep_1d(base, "k", list(range(1, 9)))
    out["ctx"] = sweep_1d(base, "ctx_tokens",
                          [2048, 4096, 8192, 16384, 32768, 65536, 131072])
    out["batch"] = sweep_1d(base, "batch", [1, 2, 4, 8, 16, 32, 64])
    out["link_bw"] = sweep_1d(base, "link_bw_gb", [2, 8, 32, 64, 128, 256, 512, 1024, 2048])
    out["draft_bw"] = sweep_1d(base, "draft_bw_tbs", [0.27, 0.5, 0.864, 2, 3.35, 8])
    out["boundaries"] = {
        "alpha_breakeven": {d: find_crossover(out["alpha"], d)["x"]
                            for d in ("colocated", "dt_serial", "dt_pipeline_ideal")},
        "k_optimal": _argmax_k(out["k"]),
        "dt_vs_colocated_link": find_crossover(out["link_bw"], "dt_pipeline_ideal",
                                               "colocated")["x"],
        "dt_serial_vs_colocated_link": find_crossover(out["link_bw"], "dt_serial",
                                                      "colocated")["x"],
        "batch_reversal": find_crossover(out["batch"], "colocated")["x"],
    }
    return out


def _argmax_k(scan: dict) -> dict:
    best = {}
    for p in scan["points"]:
        for dep, d in p["deployments"].items():
            if dep == "baseline":
                continue
            if dep not in best or d["speedup"] > best[dep]["speedup"]:
                best[dep] = {"K": p["value"], "speedup": d["speedup"]}
    return best
