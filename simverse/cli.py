"""CLI：与 Web 完全同入口（engine.run / sweep），可重放验收的基础。

命令：
  simverse run [场景.json] [-o run.json] [--csv ops.csv]   单场景计算与导出
  simverse sweep [场景.json] [-o sweep.json] [--charts 目录]  扫描与边界图
  simverse export run.json [--csv ops.csv] [--events timeline.csv]  从 RunResult 导出
  simverse serve [--host --port]                              启动 Web 工作台
  simverse list [models|drafts|devices|links|formulas]        注册表

v0.2 命令（ResultBundle 管线；JSON/CSV/trace/HTML 四类导出）：
  simverse run-v2 [场景.json] [-o run.json] [--csv ops.csv] [--trace events.csv] [--html report.html]
  simverse sweep-v2 [场景.json] [-o sweep.json]
  simverse export-v2 run.json [--csv ...] [--trace ...] [--html ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .assets import default_output_dir, scenario_dir
from .contracts import ScenarioSpec, canonical_json
from .engine import run

DEFAULT_SCENARIO = scenario_dir() / "v41_flash_dt_verification.json"


def _configure_stdio():
    """让 CLI 状态文本在非 UTF-8 的管道/终端上也不会中断已完成的命令。

    Windows 上非控制台 stdio（管道、重定向文件）默认使用系统 ANSI 代码页
    （如 cp1252），而本 CLI 的提示含中文；不显式覆盖时，``print`` 会在文件
    已经写成功后抛 UnicodeEncodeError，把成功命令变成 exit 1。UTF-8 是跨
    平台稳定的输出编码；``backslashreplace`` 再兜底代理对等极端字符。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except (AttributeError, ValueError, OSError):
            pass


def _dump(obj, path: Path | None):
    text = json.dumps(obj, ensure_ascii=False, indent=1)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"已写 {path}")
    else:
        print(text)


def _csv_field(v) -> str:
    """RFC 4180 字段转义：含逗号/引号/换行时引号包裹、内部引号翻倍。

    与前端 static/csv.js 同语义；不做字符替换，保留原始字段内容
    （2026-11-17 工作台审查：旧实现把逗号替换成分号、丢换行，改变字段）。
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


def _ops_csv(result: dict) -> str:
    cols = ["stage", "layer", "op", "exec_count", "flops", "weight_read", "act_read",
            "act_write", "kv_read", "kv_write", "scope", "formula_id", "t_ms", "bound",
            "shape_note"]
    return _csv(cols, result["ops"])


def _events_csv(result: dict) -> str:
    cols = ["id", "stage", "lane", "label", "start_ms", "end_ms", "depends_on"]
    return _csv(cols, result["timeline"])


def _summary(result: dict):
    print(f"status={result['status']} digests={result['digests']}")
    if result["status"].startswith("unsupported"):
        print(result["status_detail"])
        return
    m = result["metrics"]
    print(f"E[n]={m['E_committed_len']:.3f}  alpha={m['alpha']}  P_full={m['P_full']:.4f}"
          f"  E[M_draft,0]={m['E_draft_step0_tokens']:.4f}")
    for c in result["comparisons"]:
        detail = c.get("detail") or {}
        print(f"  {c['deployment']:>18}  TPOT {c['tpot_ms']:8.4f} ms   speedup "
              f"{c['speedup']:6.3f}   {detail.get('note', '')}")
    be = m["breakeven_alpha"]
    print("  breakeven α*:", {k: (round(v, 4) if v == v else "none") for k, v in be.items()})


def _require_active_v01(scenario):
    from .model_ir.catalog import PRODUCT_MODELS
    if scenario.model_id not in PRODUCT_MODELS:
        raise SystemExit(
            f"历史/合成模型 {scenario.model_id} 不在活跃 CLI 目录；tiny 仅作测试 "
            "fixture，请使用 run-v2 的两款主模型")


def cmd_run(args):
    src = Path(args.scenario) if args.scenario else DEFAULT_SCENARIO
    scenario = ScenarioSpec.from_json(json.loads(src.read_text(encoding="utf-8")))
    _require_active_v01(scenario)
    result = run(scenario).to_json()
    result["replay"] = {"scenario_file": str(src)}
    _summary(result)
    if args.out:
        _dump(result, Path(args.out))
    if args.csv:
        p = Path(args.csv)
        p.write_text(_ops_csv(result), encoding="utf-8")
        print(f"已写 {p}")


def cmd_sweep(args):
    from .sweep import sweep_suite
    from .chartpng import line_chart
    src = Path(args.scenario) if args.scenario else DEFAULT_SCENARIO
    scenario = ScenarioSpec.from_json(json.loads(src.read_text(encoding="utf-8")))
    _require_active_v01(scenario)
    data = sweep_suite(scenario)
    print("boundaries:", json.dumps(data["boundaries"], ensure_ascii=False))
    out = Path(args.out) if args.out else default_output_dir() / "sweep.json"
    _dump(data, out)
    charts = Path(args.charts) if args.charts else out.parent
    charts.mkdir(parents=True, exist_ok=True)

    def series_of(scan, key="speedup", only=()) -> dict:
        s = {}
        for p in scan["points"]:
            for dep, d in p["deployments"].items():
                if only and dep not in only:
                    continue
                s.setdefault(dep, []).append((float(p["value"]), d[key]))
        return s

    (charts / "boundary_alpha.png").write_bytes(
        line_chart(series_of(data["alpha"]), "Speedup vs acceptance alpha",
                   "alpha", "speedup", hline=1.0))
    (charts / "boundary_k.png").write_bytes(
        line_chart(series_of(data["k"]), "Speedup vs draft tokens K", "K", "speedup",
                   hline=1.0))
    (charts / "boundary_ctx.png").write_bytes(
        line_chart(series_of(data["ctx"]), "Speedup vs context length", "ctx tokens",
                   "speedup", hline=1.0, log_x=True))
    (charts / "boundary_batch.png").write_bytes(
        line_chart(series_of(data["batch"]), "Speedup vs batch B", "batch", "speedup",
                   hline=1.0, log_x=True))
    (charts / "boundary_link.png").write_bytes(
        line_chart(series_of(data["link_bw"]), "Speedup vs D/T link bandwidth",
                   "link GB/s", "speedup", hline=1.0, log_x=True))
    (charts / "boundary_draftbw.png").write_bytes(
        line_chart(series_of(data["draft_bw"]), "Speedup vs draft device bandwidth",
                   "draft TB/s", "speedup", hline=1.0, log_x=True))
    print(f"图表已写 {charts}/boundary_*.png")


def cmd_export(args):
    result = json.loads(Path(args.run).read_text(encoding="utf-8"))
    if args.csv:
        p = Path(args.csv)
        p.write_text(_ops_csv(result), encoding="utf-8")
        print(f"已写 {p}")
    if args.events:
        p = Path(args.events)
        p.write_text(_events_csv(result), encoding="utf-8")
        print(f"已写 {p}")


def _v2_scenario(path):
    """v0.2 场景解析：显式文件按文件；缺省读共享默认预设（R10d）。

    预设源 = assets/scenarios/qwen36_dflash_default.json（原 CLI 硬编码
    V2_DEFAULT_SCENARIO 的逐键固化副本），与场景目录 API、工作台同源。
    """
    import json as _json
    from .api_io import decode_scenario_v2
    from .presets import default_preset
    data = (_json.loads(Path(path).read_text(encoding="utf-8")) if path
            else default_preset())
    return decode_scenario_v2(data)


def _ops_csv_v2(bundle: dict) -> str:
    """v0.2 算子 CSV：与前端导出同列；t_ms/bound 从 cost.roofline_points
    按 row_id（stage:layer:op）关联，无计价行留空（不伪造）。"""
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
    pts = {p["row_id"]: p for p in
           ((record_cost.get("selected_row_prices") if use_records else None)
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
            "depends_on", "label"]
    execution = ((bundle.get("cost") or {}).get("selection") or {}).get("execution") or bundle.get("execution")
    events = [e for rnd in (execution or {}).get("rounds") or []
              for e in rnd.get("events") or []]
    return _csv(cols, events)


def _groups_csv_v2(bundle: dict) -> str:
    cols = ["stage", "role", "group", "primitive", "analytic_ms", "selected_ms", "source", "timing_boundary", "basis"]
    return _csv(cols, (bundle.get("cost") or {}).get("groups") or [])


def _report_html_v2(bundle: dict) -> str:
    from html import escape as escape_html
    m = bundle.get("metrics") or {}

    def row(name):
        x = m.get(name) or {}
        v = x.get("value")
        reason = x.get("reason") or ""
        return ("<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>"
                % (escape_html(name), "UNKNOWN" if v is None else v,
                   escape_html(x.get("unit", "")), escape_html(reason[:80])))

    checks = (bundle.get("capability") or {}).get("checks") or {}
    cap_rows = "".join(
        "<tr><td>%s</td><td>%s</td><td>%s</td></tr>" % (escape_html(k), escape_html(v.get("status", "")),
                                                         escape_html(v.get("reason", "")))
        for k, v in sorted(checks.items()))
    html = ("<!DOCTYPE html><html lang=zh><meta charset=utf-8>"
            "<title>simverse run report</title>"
            "<body style=font-family:sans-serif;margin:24px>"
            "<h2>simverse ResultBundle 摘要</h2>"
            "<p>run_id <code>%s</code> · 状态 %s</p>"
            "<table border=1 cellpadding=6 style=border-collapse:collapse>"
            "<tr><th>指标</th><th>值</th><th>单位</th><th>缺口原因</th></tr>%s</table>"
            "<h3>能力检查</h3>"
            "<table border=1 cellpadding=6 style=border-collapse:collapse>"
            "<tr><th>检查</th><th>状态</th><th>原因</th></tr>%s</table>"
            "<p style=color:#64748b>静态导出：结果身份与假设随文件携带；"
            "非实时后端。性能为 T0 解析估算或显式选择的记录条件对照，非端到端实测。</p></body></html>"
            % (bundle["manifest"]["run_id"], bundle["manifest"]["status"],
               "".join(row(k) for k in sorted(m) if isinstance(m[k], dict) and "value" in m[k]),
               cap_rows))
    records = (bundle.get("cost") or {}).get("record_cost")
    if records:
        from html import escape
        summary = {key: records[key] for key in
                   ("set_id", "admission", "domain", "limits", "content_digest", "matched",
                    "total", "stages", "unpriced_components", "prerequisites", "measurement_plan")}
        details = "<h3>记录成本覆盖、回退与补采计划</h3><pre>" + escape(
            json.dumps(summary, ensure_ascii=False, indent=2)) + "</pre>"
        html = html.replace("</body>", details + "</body>")
    groups = (bundle.get("cost") or {}).get("groups")
    if groups:
        detail = "<h3>阶段与成本组</h3><p>解析组汇总原有计价项；融合组实测不向逻辑行分摊。未知阶段不计作零。</p><pre>" + escape_html(
            json.dumps(groups, ensure_ascii=False, indent=2)) + "</pre>"
        html = html.replace("</body>", detail + "</body>")
    return html


def cmd_run_v2(args):
    from .pipeline import run_v2
    scenario = _v2_scenario(args.scenario)
    if getattr(args, "cost_profile", None):
        scenario.cost_profile = args.cost_profile
    try:
        bundle = run_v2(scenario).to_json()
    except ValueError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(2)
    bundle["replay"] = {"scenario_file": str(args.scenario) if args.scenario
                        else "preset:qwen36_dflash_default"}
    _summary_v2(bundle)
    if args.out:
        _dump(bundle, Path(args.out))
    if args.csv:
        Path(args.csv).write_text(_ops_csv_v2(bundle), encoding="utf-8")
        print(f"已写 {args.csv}")
    if args.groups_csv:
        Path(args.groups_csv).write_text(_groups_csv_v2(bundle), encoding="utf-8")
        print(f"已写 {args.groups_csv}")
    if args.trace:
        Path(args.trace).write_text(_trace_csv_v2(bundle), encoding="utf-8")
        print(f"已写 {args.trace}")
    if args.html:
        Path(args.html).write_text(_report_html_v2(bundle), encoding="utf-8")
        print(f"已写 {args.html}")


def _summary_v2(bundle: dict):
    man = bundle["manifest"]
    print(f"status={man['status']} run_id={man['run_id'][:19]}... "
          f"code={man['code_revision'][:10]}")
    m = bundle.get("metrics") or {}
    for key in ("M01_tpot_selected", "M01_baseline_selected", "M02_speedup_selected",
                "M01_tpot_analytical", "M03_E_N", "M04_round_selected"):
        x = m.get(key)
        if not x:
            continue
        if x.get("value") is None:
            print(f"  {key:26} 未知 —— {(x.get('reason') or '')[:60]}")
        else:
            note = ("   ! 不参与排名；原因见下方能力检查与导出 JSON"
                    if key == "M02_speedup_vs_baseline" and x.get("reason")
                    else ("   ! " + x["reason"] if x.get("reason") else ""))
            print(f"  {key:26} {x['value']:.4f} {x.get('unit', '')}" + note)
    checks = (bundle.get("capability") or {}).get("checks") or {}
    bad = [k for k, v in checks.items() if v["status"] != "supported"]
    if bad:
        for key in bad:
            check = checks[key]
            if key.startswith("memory_"):
                print(f"  注意 {key} [{check['status']}]：已知显存下界仅供核对；"
                      "缺可分配预算或完整运行时峰值，不能判定可部署（明细见 JSON）")
            else:
                print(f"  注意 {key} [{check['status']}]：{check['reason']}")


def cmd_sweep_v2(args):
    from .experiments import sweep_suite
    scenario = _v2_scenario(args.scenario)
    data = sweep_suite(scenario)
    out = Path(args.out) if args.out else default_output_dir() / "sweep_v2.json"
    _dump(data, out)
    import json as _json
    print("breakevens:", _json.dumps(data["breakevens"], ensure_ascii=False))


def cmd_export_v2(args):
    import json as _json
    bundle = _json.loads(Path(args.run).read_text(encoding="utf-8"))
    if args.csv:
        Path(args.csv).write_text(_ops_csv_v2(bundle), encoding="utf-8")
        print(f"已写 {args.csv}")
    if args.groups_csv:
        Path(args.groups_csv).write_text(_groups_csv_v2(bundle), encoding="utf-8")
        print(f"已写 {args.groups_csv}")
    if args.trace:
        Path(args.trace).write_text(_trace_csv_v2(bundle), encoding="utf-8")
        print(f"已写 {args.trace}")
    if args.html:
        Path(args.html).write_text(_report_html_v2(bundle), encoding="utf-8")
        print(f"已写 {args.html}")


def cmd_list(args):
    if args.what == "formulas":
        from .formulas import formulas_json
        print(json.dumps(formulas_json(), ensure_ascii=False, indent=1))
        return
    if args.what in ("models", "drafts"):
        from .model_ir import catalog_json
        cj = catalog_json()
        payload = cj["models"] if args.what == "models" else cj["drafts"]
        print(json.dumps(payload, ensure_ascii=False, indent=1))
        return
    if args.what == "devices":
        from .devices import DEVICES, LINKS
        print(json.dumps({"devices": {k: d.to_json() for k, d in DEVICES.items()},
                          "links": {k: dict(v.__dict__) for k, v in LINKS.items()}},
                         ensure_ascii=False, indent=1))
        return
    print("用法: list [models|drafts|devices|formulas]")


def main(argv=None):
    _configure_stdio()
    ap = argparse.ArgumentParser(prog="simverse")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run", help="单场景计算（默认=固定验证场景）")
    p_run.add_argument("scenario", nargs="?", default=None)
    p_run.add_argument("-o", "--out", default=None)
    p_run.add_argument("--csv", default=None)
    p_run.set_defaults(fn=cmd_run)
    p_sw = sub.add_parser("sweep", help="扫描与边界")
    p_sw.add_argument("scenario", nargs="?", default=None)
    p_sw.add_argument("-o", "--out", default=None)
    p_sw.add_argument("--charts", default=None)
    p_sw.set_defaults(fn=cmd_sweep)
    p_ex = sub.add_parser("export", help="从 RunResult 导出 CSV")
    p_ex.add_argument("run")
    p_ex.add_argument("--csv", default=None)
    p_ex.add_argument("--events", default=None)
    p_ex.set_defaults(fn=cmd_export)
    p_ls = sub.add_parser("list", help="注册表")
    p_ls.add_argument("what", nargs="?", default="models")
    p_ls.set_defaults(fn=cmd_list)
    p_r2 = sub.add_parser("run-v2", help="v0.2 管线单场景（JSON/CSV/trace/HTML 导出）")
    p_r2.add_argument("scenario", nargs="?", default=None)
    p_r2.add_argument("-o", "--out", default=None)
    p_r2.add_argument("--csv", default=None)
    p_r2.add_argument("--groups-csv", default=None, help="默认耗时口径：成本组 CSV；--csv 保留细粒度诊断")
    p_r2.add_argument("--trace", default=None)
    p_r2.add_argument("--html", default=None)
    p_r2.add_argument("--cost-profile", default=None,
                      help="历史归档 profile 集 id（仅归档模型；跨模型直接拒绝）")
    p_r2.set_defaults(fn=cmd_run_v2)
    p_s2 = sub.add_parser("sweep-v2", help="v0.2 参数扫描（不可变快照）")
    p_s2.add_argument("scenario", nargs="?", default=None)
    p_s2.add_argument("-o", "--out", default=None)
    p_s2.set_defaults(fn=cmd_sweep_v2)
    p_e2 = sub.add_parser("export-v2", help="从 v0.2 ResultBundle 导出四类文件")
    p_e2.add_argument("run")
    p_e2.add_argument("--csv", default=None)
    p_e2.add_argument("--groups-csv", default=None)
    p_e2.add_argument("--trace", default=None)
    p_e2.add_argument("--html", default=None)
    p_e2.set_defaults(fn=cmd_export_v2)
    p_srv = sub.add_parser("serve", help="Web 工作台")
    p_srv.add_argument("--host", default="127.0.0.1")
    p_srv.add_argument("--port", type=int, default=8765)
    p_srv.set_defaults(fn=lambda a: _serve(a))
    args = ap.parse_args(argv)
    args.fn(args)


def _serve(args):
    from .webapp import app
    print(f"simverse web 工作台: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
