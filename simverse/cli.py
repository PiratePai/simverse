"""CLI：与 Web 完全同入口（engine.run / sweep），可重放验收的基础。

命令：
  simverse run [场景.json] [-o run.json] [--csv ops.csv]   单场景计算与导出
  simverse sweep [场景.json] [-o sweep.json] [--charts 目录]  扫描与边界图
  simverse export run.json [--csv ops.csv] [--events timeline.csv]  从 RunResult 导出
  simverse serve [--host --port]                              启动 Web 工作台
  simverse list [models|drafts|devices|links|formulas]        注册表
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .contracts import ScenarioSpec, canonical_json
from .engine import run
from .models import registry_json

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCENARIO = ROOT / "assets" / "scenarios" / "v41_flash_dt_verification.json"


def _dump(obj, path: Path | None):
    text = json.dumps(obj, ensure_ascii=False, indent=1)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"已写 {path}")
    else:
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
        print(text)


def _ops_csv(result: dict) -> str:
    cols = ["stage", "layer", "op", "exec_count", "flops", "weight_read", "act_read",
            "act_write", "kv_read", "kv_write", "scope", "formula_id", "t_ms", "bound",
            "shape_note"]
    lines = [",".join(cols)]
    for o in result["ops"]:
        vals = []
        for c in cols:
            v = o.get(c, "")
            v = str(v).replace(",", ";").replace("\n", " ")
            vals.append(v)
        lines.append(",".join(vals))
    return "\n".join(lines)


def _events_csv(result: dict) -> str:
    cols = ["id", "stage", "lane", "label", "start_ms", "end_ms", "depends_on"]
    lines = [",".join(cols)]
    for e in result["timeline"]:
        lines.append(",".join(str(e.get(c, "")).replace(",", ";") for c in cols))
    return "\n".join(lines)


def _summary(result: dict):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
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


def cmd_run(args):
    src = Path(args.scenario) if args.scenario else DEFAULT_SCENARIO
    scenario = ScenarioSpec.from_json(json.loads(src.read_text(encoding="utf-8")))
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
    data = sweep_suite(scenario)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print("boundaries:", json.dumps(data["boundaries"], ensure_ascii=False))
    out = Path(args.out) if args.out else ROOT / "outputs" / "sweep.json"
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


def cmd_list(args):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if args.what == "formulas":
        from .formulas import formulas_json
        print(json.dumps(formulas_json(), ensure_ascii=False, indent=1))
        return
    if args.what in ("models", "drafts"):
        print(json.dumps(registry_json(), ensure_ascii=False, indent=1))
        return
    if args.what == "devices":
        from .devices import DEVICES, LINKS
        print(json.dumps({"devices": {k: d.to_json() for k, d in DEVICES.items()},
                          "links": {k: dict(v.__dict__) for k, v in LINKS.items()}},
                         ensure_ascii=False, indent=1))
        return
    print("用法: list [models|drafts|devices|formulas]")


def main(argv=None):
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
