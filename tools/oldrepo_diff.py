"""两旧仓对账工具：同一固定场景 → model-verse / pi-viewer / 新仓 并排，差异逐条归类。

用法：python tools/oldrepo_diff.py [--out outputs/oldrepo_diff.json]
差异分类（重建计划 M2 口径）：新仓错误 / 有据纠正旧值 / 口径差异(合法不可直对) /
新仓简化(标注影响) / 待解释。任何"待解释"都必须在报告中闭环。

对账通道：
  A. model-verse model_analyzer（逐算子结构口径，本仓移植的直接上游）
  B. model-verse spec_gain_model.js（前端量级解析口径，node 直接运行原 JS，不做移植）
  C. pi-viewer piviewer.roofline（同源五类字节账本，V4.1 Flash 支持度探测）
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MV = Path(r"D:/Github/model-verse/LLM-Viewer-main")
PV = Path(r"D:/Github/pi-viewer")
sys.path.insert(0, str(ROOT))

from simverse.contracts import ScenarioSpec  # noqa: E402
from simverse.engine import run  # noqa: E402

FIXED = ROOT / "assets" / "scenarios" / "v41_flash_dt_verification.json"


def new_repo_rows() -> dict:
    scenario = ScenarioSpec.from_json(json.loads(FIXED.read_text(encoding="utf-8")))
    res = run(scenario)
    assert res.status == "ok", res.status_detail

    def agg(stage):
        rows = {}
        for o in res.ops:
            if o.stage != stage:
                continue
            r = rows.setdefault(o.op, dict(flops=0, weight_read=0, act_read=0,
                                           act_write=0, kv_read=0, kv_write=0, n=0))
            r["flops"] += o.flops
            r["weight_read"] += o.weight_read
            r["act_read"] += o.act_read
            r["act_write"] += o.act_write
            r["kv_read"] += o.kv_read
            r["kv_write"] += o.kv_write
            r["n"] += 1
        return rows

    return {"baseline": agg("baseline_decode"), "verify": agg("verify"),
            "metrics": res.metrics, "comparisons": res.comparisons}


def old_analyzer_rows(spec_q: int) -> dict:
    """在旧仓目录下以子进程运行 model_analyzer（相对路径依赖其 CWD；同时隔离模块状态）。"""
    code = (
        "import sys, json; sys.path.insert(0, '.');\n"
        "from model_analyzer import ModelAnalyzer;\n"
        "an = ModelAnalyzer('DeepSeek_V41_Flash', 'nvidia_H100_SXM', None, "
        "source='DeepSeek_V41_Flash');\n"
        "res = an.analyze(batchsize=1, seqlen=32768, w_bit=8, a_bit=8, kv_bit=8, "
        "tp_size=8, spec_q=%d);\n"
        "print('@@JSON@@' + json.dumps({'decode': res['decode'], "
        "'totals': res['total_results']['decode']}))" % spec_q)
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(MV),
                          capture_output=True, text=True, encoding="utf-8", timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(f"旧仓 analyzer 运行失败: {proc.stderr[-1500:]}")
    payload = json.loads([ln for ln in proc.stdout.splitlines()
                          if ln.startswith("@@JSON@@")][0][len("@@JSON@@"):])
    L = 40
    # 旧仓 totals 将每行 ×L；post_process 追加的 lm_head 为整模型一条（×1），
    # 逐行对账时按此口径还原（否则旧值虚高 40×）
    SINGLE_ROWS = {"lm_head"}
    out = {}
    for name, row in payload["decode"].items():
        scale = 1 if name in SINGLE_ROWS else L
        out[name] = {k: row[k] * scale for k in
                     ("OPs", "load_weight", "load_act", "store_act",
                      "load_kv_cache", "store_kv_cache")}
    out["_totals"] = payload["totals"]
    return out


NODE_SCRIPT = """
import { evaluateModel, evaluateDisaggregated, resolveDraft, computeExpectedTau }
  from 'file:///D:/Github/model-verse/LLM-Viewer-main/frontend/src/spec_gain_model.js';
// 配置映射自新仓固定场景 v41_flash_dt_verification.json（映射口径见差异报告 §B）
const cfg = {
  batch: 1, gamma: 4, alpha: 0.75, prefixBonus: 0, tp: 8,
  ctxK: 32,                    // 32768 token
  hbmBwGBs: 3350, sramBwTBs: 12, tflops: 1979,   // H100 SXM FP8；带宽/卡 ×TP8=聚合
  hitTargetPct: 0, hitDraftPct: 0,               // 关闭分层缓存（新仓无此建模）
  commMs: 0, tileM: 1, infBw: false, bytesPerParam: 0.75,  // 官方混精 16B 激活 ≈ 8B FP8 + 8B×0.5
  modelId: 'DeepSeek_V41_Flash',
};
const model = { id: 'DeepSeek_V41_Flash', activeParamsB: 16, kvBytesPerToken: 890,
                moe: true, dAttn: 64 * 128 };
const methodDef = { weightScale: 1.0, flopsMult: 1.0, commFrac: 0.25, kvFrac: 0.15,
                    draftMoe: true, stages: 3 };   // 旧仓 DSpark 预设（3 阶段块近似）
const modelDraft = { paramsB: 4.53 };
const draft = resolveDraft('dspark', modelDraft, methodDef, cfg);
const co = evaluateModel(cfg, model, draft);
const dt = evaluateDisaggregated(cfg, model, draft,
  { mode: 'dt_serial', hw: { bwTBs: 0.864, tflops: 362, tileM: 1, hostMs: 0.18,
                             fallbackOverheadMs: 0.12 },
    link: { bwGBs: 64, fixedMs: 0.10 } }, null, null);
const out = {
  tau: computeExpectedTau(cfg.alpha, cfg.gamma),
  colocated: { tBase: co.tTargetBaseMs, tVerify: co.tTargetVerifyTotalMs,
               tDraft: co.tDraftTotalMs, tpot: co.tStepSpecMs, speedup: co.speedup,
               breakevenAlpha: co.breakevenAlpha },
  dt_serial: { tBase: dt.tTargetBaseMs, tDraft: dt.tDraftRawMs, dtCommMs: dt.dtCommMs,
               tpot: dt.tStepSpecMs, speedup: dt.speedup,
               breakevenAlpha: dt.breakevenAlpha, bytesD2T: dt.bytesD2T,
               bytesT2D: dt.bytesT2D },
};
console.log(JSON.stringify(out));
"""


def run_js_math() -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(NODE_SCRIPT)
        path = fh.name
    proc = subprocess.run(["node", path], capture_output=True, text=True,
                          encoding="utf-8", timeout=60)
    Path(path).unlink(missing_ok=True)
    if proc.returncode != 0:
        return {"error": proc.stderr[-2000:]}
    return json.loads(proc.stdout.strip().splitlines()[-1])


def pi_viewer_probe() -> dict:
    try:
        sys.path.insert(0, str(PV))
        sys.path.insert(0, str(MV))   # piviewer 依赖 mv_model_structure 共享包（LLM-Viewer-main 内）
        from piviewer import devices as pdev
        from piviewer.roofline import RooflineAnalyzer
        cfg = json.loads((ROOT / "assets" / "config" / "deepseek_v41_config.json")
                         .read_text(encoding="utf-8"))["text_config"]
        cfg["architectures"] = ["DeepseekV3ForCausalLM"]  # 尝试按 V3 路径解析
        dev = next(iter(pdev.DEVICES.values())) if hasattr(pdev, "DEVICES") else None
        an = RooflineAnalyzer(cfg, dev)
        return {"supported": True, "note": "按 DeepseekV3 路径解析成功（口径待人工核对）"}
    except Exception as exc:  # noqa: BLE001 —— 任何失败都作为"不支持原因"记录
        return {"supported": False, "reason": f"{type(exc).__name__}: {exc}"[:400]}


CLASSIFICATIONS = {
    # op 级差异的预归类说明（结合数值在报告中核对）
    "moe_router": "口径差异：新仓单列 router 权重读（旧仓聚合进 moe_ffn 未计权重），FLOPs 同式",
    "lm_head": "有据纠正：旧仓 post_process 未按 TP 切分（h×v 全量）；新仓列切分 h×v/TP per card",
    "embed_lookup": "口径差异：旧仓解码路径不单列 embed 查表行；新仓补计激活查表+驻留容量",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs" / "oldrepo_diff.json"))
    args = ap.parse_args()

    new = new_repo_rows()
    old_base = old_analyzer_rows(spec_q=1)
    old_verify = old_analyzer_rows(spec_q=5)
    js = run_js_math()
    pv = pi_viewer_probe()

    def op_table(new_stage, old_rows):
        rows = []
        for op, o in sorted(old_rows.items()):
            if op.startswith("_"):
                continue
            n = new_stage.get(op, dict(flops=0, weight_read=0, act_read=0,
                                       act_write=0, kv_read=0, kv_write=0))
            rows.append({
                "op": op,
                "old_flops": o["OPs"], "new_flops": n["flops"],
                "old_weight": o["load_weight"], "new_weight": n["weight_read"],
                "old_act_r": o["load_act"], "new_act_r": n["act_read"],
                "old_act_w": o["store_act"], "new_act_w": n["act_write"],
                "old_kv_r": o["load_kv_cache"], "new_kv_r": n["kv_read"],
                "old_kv_w": o["store_kv_cache"], "new_kv_w": n["kv_write"],
                "class": CLASSIFICATIONS.get(op, ""),
            })
        # 新仓有而旧仓没有的行
        for op, n in sorted(new_stage.items()):
            if op not in old_rows:
                rows.append({"op": op, "note": "仅新仓（旧仓无对应行）",
                             "new_flops": n["flops"], "new_weight": n["weight_read"],
                             "new_kv_r": n["kv_read"], "new_kv_w": n["kv_write"],
                             "old_flops": None, "old_weight": None,
                             "old_act_r": None, "new_act_r": n["act_read"],
                             "old_act_w": None, "new_act_w": n["act_write"],
                             "old_kv_r": None, "old_kv_w": None,
                             "class": CLASSIFICATIONS.get(op, "新仓补计（见报告）")})
        return rows

    report = {
        "schema_version": "0.1",
        "scenario": FIXED.name,
        "A_modelverse_analyzer": {
            "note": "逐算子结构口径；旧仓行×40 层 = decode 每卡总量（spec_q=1 基线 / spec_q=5 验证）",
            "baseline_ops": op_table(new["baseline"], old_base),
            "baseline_totals_old": old_base["_totals"],
            "verify_ops": op_table(new["verify"], old_verify),
            "verify_totals_old": old_verify["_totals"],
            "new_baseline_sums": {k: sum(r.get(f"new_{k}", 0) or 0
                                         for r in op_table(new["baseline"], old_base))
                                  for k in ("flops", "weight", "kv_r")},
        },
        "B_modelverse_specgain_js": js,
        "C_pi_viewer": pv,
        "new_repo": {"metrics": new["metrics"],
                     "comparisons": new["comparisons"]},
    }
    out = Path(args.out)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"已写 {out}")

    # 摘要打印（供报告引用）
    tot_old = old_base["_totals"]
    nb = new["baseline"]
    nf = sum(r["flops"] for r in nb.values())
    nw = sum(r["weight_read"] for r in nb.values())
    nkr = sum(r["kv_read"] for r in nb.values())
    print(f"基线 decode 每卡总量: FLOPs 旧={tot_old['OPs']:.4g} 新={nf:.6g} "
          f"({nf / tot_old['OPs'] * 100:.1f}%) | 权重读 旧={tot_old['load_weight']:.4g} "
          f"新={nw:.6g} ({nw / tot_old['load_weight'] * 100:.1f}%) | "
          f"KV读 旧={tot_old['load_kv_cache']:.4g} 新={nkr:.6g}")
    if "tau" in js and js.get("colocated", {}).get("tBase") is not None:
        m = new["metrics"]
        print(f"JS 口径: tau={js['tau']:.3f} (新仓 E[n]={m['E_committed_len']:.3f}) | "
              f"t_base={js['colocated']['tBase']:.3f}ms (新仓 {m['t_base_ms']:.3f}) | "
              f"colocated speedup={js['colocated']['speedup']:.3f} "
              f"(新仓 {new['comparisons'][1]['speedup']:.3f}) | "
              f"dt_serial speedup={js['dt_serial']['speedup']:.3f} "
              f"(新仓 {new['comparisons'][2]['speedup']:.3f})")
    elif "error" in js:
        print("JS 口径运行失败:", js["error"][:300])
    print("pi-viewer:", pv)


if __name__ == "__main__":
    main()
