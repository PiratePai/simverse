"""Archived v0.1 CLI commands and file formats."""
import json
from pathlib import Path

from ..assets import default_output_dir, scenario_dir
from ..contracts.legacy import ScenarioSpec
from ..exports import _csv, dump_json

_DEFAULT_SCENARIO = "v41_flash_dt_verification.json"


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
    from ..model_ir.catalog import PRODUCT_MODELS
    if scenario.model_id not in PRODUCT_MODELS:
        raise SystemExit(
            f"历史/合成模型 {scenario.model_id} 不在活跃 CLI 目录；tiny 仅作测试 "
            "fixture，请使用 run-v2 的两款主模型")


def cmd_run(args):
    from ..engine import run

    src = Path(args.scenario) if args.scenario else scenario_dir() / _DEFAULT_SCENARIO
    scenario = ScenarioSpec.from_json(json.loads(src.read_text(encoding="utf-8")))
    _require_active_v01(scenario)
    result = run(scenario).to_json()
    result["replay"] = {"scenario_file": str(src)}
    _summary(result)
    if args.out:
        dump_json(result, Path(args.out))
    if args.csv:
        p = Path(args.csv)
        p.write_text(_ops_csv(result), encoding="utf-8")
        print(f"已写 {p}")


def cmd_sweep(args):
    from ..sweep import sweep_suite
    from ..chartpng import line_chart
    src = Path(args.scenario) if args.scenario else scenario_dir() / _DEFAULT_SCENARIO
    scenario = ScenarioSpec.from_json(json.loads(src.read_text(encoding="utf-8")))
    _require_active_v01(scenario)
    data = sweep_suite(scenario)
    print("boundaries:", json.dumps(data["boundaries"], ensure_ascii=False))
    out = Path(args.out) if args.out else default_output_dir() / "sweep.json"
    dump_json(data, out)
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


def register_commands(sub):
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
