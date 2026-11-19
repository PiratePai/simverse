"""CLI：生产命令与工作台复用 v0.2 管线，旧命令用于归档重放。

命令：
  simverse archive {run,sweep,export}  v0.1 离线归档重放
  simverse serve [--host --port]                              启动 Web 工作台
  simverse list [models|drafts|devices|formulas]        注册表

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
from .assets import default_output_dir
from .contracts import js_restore
from .exports import _groups_csv_v2, _ops_csv_v2, _trace_csv_v2, dump_json as _dump
from .legacy.cli import register_commands as register_legacy_commands


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


def _write_v2_exports(bundle, args):
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
    _write_v2_exports(bundle, args)


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
    bundle = js_restore(json.loads(Path(args.run).read_text(encoding="utf-8")))
    _write_v2_exports(bundle, args)


def cmd_list(args):
    if args.what == "formulas":
        from .formula_registry import formulas_json
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
    archive = sub.add_parser("archive", help="v0.1 离线归档重放")
    register_legacy_commands(archive.add_subparsers(dest="archive_cmd", required=True))
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
