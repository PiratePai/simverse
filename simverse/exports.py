"""Serialize saved results for the CLI and workbench."""
from __future__ import annotations

import json
from pathlib import Path


def dump_json(obj, path: Path | None):
    text = json.dumps(obj, ensure_ascii=False, indent=1)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"已写 {path}")
    else:
        print(text)


def _csv_field(v) -> str:
    """RFC 4180 字段转义：含逗号/引号/换行时引号包裹、内部引号翻倍。

    原样保留逗号与换行。CLI 和工作台使用同一序列化函数。
    """
    s = "" if v is None else str(v)
    if any(ch in s for ch in ',"\r\n'):
        return '"' + s.replace('"', '""') + '"'
    return s


def _csv(cols, records) -> str:
    lines = [",".join(cols)]
    for rec in records:
        lines.append(",".join(_csv_field(rec.get(c)) for c in cols))
    return chr(10).join(lines)


def _ops_csv_v2(bundle: dict) -> str:
    """按 row_id 关联选用行价，兼容只含解析行价或 Roofline 点的旧结果。"""
    cols = ["stage", "layer", "op", "flops", "weight_read", "weight_read_expected",
            "act_read", "act_write", "kv_read", "kv_write", "state_read",
            "state_write", "weight_capacity", "statistic", "formula", "t_ms",
            "bound", "shape"]
    rows = ((bundle.get("work") or {}).get("target_decode") or {}).get("rows") or []
    cost = bundle.get("cost") or {}
    record_cost = cost.get("record_cost") or {}
    use_records = bool(record_cost)
    if use_records:
        cols += ["analytic_t_ms", "record_t_ms", "cost_source", "record_refs", "fallback_reason"]
    selection = cost.get("selection") or {}
    pts = {p["row_id"]: p for p in
           (selection.get("selected_row_prices") or record_cost.get("selected_row_prices")
            or cost.get("row_prices") or cost.get("roofline_points") or [])}
    records = []
    for o in rows:
        p = pts.get(f"{o.get('stage')}:{o.get('layer')}:{o.get('op')}") or {}
        rec = {c: o.get(c) for c in cols}
        rec["t_ms"] = p.get("t_ms", "")
        rec["bound"] = p.get("bound", "")
        if use_records:
            rec.update(analytic_t_ms=p.get("analytic_ms", ""), record_t_ms=p.get("t_ms", ""),
                       cost_source=p.get("cost_source", ""),
                       record_refs=";".join(p.get("record_refs") or []),
                       fallback_reason=p.get("fallback_reason") or "")
        records.append(rec)
    return _csv(cols, records)


def _trace_csv_v2(bundle: dict) -> str:
    cols = ["event_id", "round_id", "role", "stage", "t_ms", "start_ms",
            "depends_on", "label", "start_lower_bound_ms", "reason"]
    execution = ((bundle.get("cost") or {}).get("selection") or {}).get("execution") or bundle.get("execution")
    events = [e for rnd in (execution or {}).get("rounds") or []
              for e in rnd.get("events") or []]
    return _csv(cols, events)


def _groups_csv_v2(bundle: dict) -> str:
    cols = ["stage", "role", "group", "primitive", "analytic_ms", "selected_ms", "source", "timing_boundary", "basis"]
    return _csv(cols, (bundle.get("cost") or {}).get("groups") or [])
