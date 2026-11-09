"""H200 Qwen3-235B-A22B profile 蒸馏导入器（第二条实测校准闭环，可重放）。

与第一条（30B-tiny，TP1）不同：真模型官方身份已入册（2026-11-10），数据含
TP2/8 的 linear/attention decode-even 与 per-TP kernel_gap。MoE 数据在册但
**全部 defer**（准入裁决见 manifest）：
  * TP8/EP1（inter/tp 分片）：新仓 count 的 MoE 行为整模型口径、memory 按
    experts//tp——两者尚无统一 per-rank 分片账本（V04 工程缺口）；实测替换
    会与整模型解析混尺 → defer，待 V04 口径落地后升级。
  * TP1/EP4：DP4 部署（每卡 32 专家、token 复制），与单卡场景口径不符。
  * TP1 linear/attention 存在但 235B 无 TP1 kernel_gap 行 → gap 政策不完整，
    一并 defer（TP2/8 gap 行齐备才准入）。

用法：
  py -3.12 tools/distill_h200_235b.py --pi-viewer D:/Github/pi-viewer [--check]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from distill_h200_profiles import (  # noqa: E402
    LINEAR_OPS, ATTN_DECODE_OPS, ROUND_MS, _sha256, _holdout, _median_col,
    _count_col, _git_head)

MODEL_DIR = "Qwen3-235B-A22B"
PROFILE_SET_ID = "h200-qwen3-235b-a22b-op-v1"
OUT_DIR_NAME = "h200_qwen3_235b_a22b"
TPS = (2, 8)          # 准入域（TP1 defer：无 kernel_gap 行）
# replicated 算子（每卡全宽、与新仓未切分行同尺度）：235B 只在 TP1 campaign 采，
# 但其 per-rank 工作不随 TP 变 → 以 TP1 测量并入 TP2/8 域（manifest 显式登记）。
# 注意 emb 为全词表查表口径（新仓 embed_lookup 行同样不按 tp 切分）。
TP_INVARIANT_GROUPS = ("emb", "input_layernorm", "post_attention_layernorm")

SRC_FILES = [
    f"data/profiling/compute/h200/{MODEL_DIR}/linear_op.csv",
    f"data/profiling/compute/h200/{MODEL_DIR}/linear_op_kernel_only.csv",
    f"data/profiling/compute/h200/{MODEL_DIR}/attention.csv",
    f"data/profiling/compute/h200/{MODEL_DIR}/attention_kernel_only.csv",
    f"data/profiling/compute/h200/{MODEL_DIR}/moe.csv",
    f"data/profiling/compute/h200/{MODEL_DIR}/moe_kernel_only.csv",
    f"data/profiling/compute/h200/{MODEL_DIR}/kernel_gap.csv",
    "data/config/models/Qwen3-235B-A22B.json",
]


def load_linear_tp(pi: Path):
    """linear CUDA_EVENT medians + kernel counts，键 (tp, tokens)。

    sharded 组（attn_*）取 TP2/8 行（per-worker 形状）；replicated 组
    （emb/norms，235B 仅 TP1 campaign 采）取 TP1 行并复制到各域 TP 键
    （per-rank 全宽、TP 不变；manifest.tp_invariant_groups 登记）。
    """
    med: dict = {}
    with (pi / SRC_FILES[0]).open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            assert r["measurement_type"] == "CUDA_EVENT"
            assert r["profiling_precision"] == "BF16"
            tp = int(r["num_tensor_parallel_workers"])
            t = int(r["num_tokens"])
            for op in LINEAR_OPS:
                v = r.get(_median_col(op), "")
                if v == "":
                    continue
                val = round(float(v), ROUND_MS)
                if op in TP_INVARIANT_GROUPS:
                    if tp == 1:
                        for k in TPS:
                            med.setdefault(op, {})[(k, t)] = val
                elif tp in TPS:
                    med.setdefault(op, {})[(tp, t)] = val
    cnt: dict = {}
    with (pi / SRC_FILES[1]).open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            assert r["measurement_type"] == "KERNEL_ONLY"
            tp = int(r["num_tensor_parallel_workers"])
            t = int(r["num_tokens"])
            for op in LINEAR_OPS:
                v = r.get(_count_col(op), "")
                if v == "":
                    continue
                val = int(float(v))
                if op in TP_INVARIANT_GROUPS:
                    if tp == 1:
                        for k in TPS:
                            cnt.setdefault(op, {})[(k, t)] = val
                elif tp in TPS:
                    cnt.setdefault(op, {})[(tp, t)] = val
    return med, cnt


def load_attn_decode_tp(pi: Path):
    """decode-even per (tp, batch, kv) 4 算子 median + kernel_count。"""
    med: dict = {}
    with (pi / SRC_FILES[2]).open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["is_prefill"] != "False" or r["mode"] != "even":
                continue
            assert r["measurement_type"] == "CUDA_EVENT"
            assert r["profiling_precision"] == "BF16"
            assert r["attention_backend"] == "FLASHINFER"
            tp = int(r["num_tensor_parallel_workers"])
            if tp not in TPS:
                continue
            key = (tp, int(r["batch_size"]), int(r["kv_cache_size"]))
            med.setdefault(key, {})
            for op in ATTN_DECODE_OPS:
                v = r.get(_median_col(op), "")
                if v != "":
                    med[key][op] = round(float(v), ROUND_MS)
    cnt: dict = {}
    with (pi / SRC_FILES[3]).open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["is_prefill"] != "False" or r["mode"] != "even":
                continue
            tp = int(r["num_tensor_parallel_workers"])
            if tp not in TPS:
                continue
            key = (tp, int(r["batch_size"]), int(r["kv_cache_size"]))
            cnt.setdefault(key, {})
            for op in ATTN_DECODE_OPS:
                v = r.get(_count_col(op), "")
                if v != "":
                    cnt[key][op] = int(float(v))
    return med, cnt


def load_gaps(pi: Path):
    """per-TP kernel gap + 环境（TP2/8）。"""
    env, gaps = {}, []
    with (pi / SRC_FILES[6]).open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            tp = int(r["num_tensor_parallel_workers"])
            if tp not in TPS:
                continue
            gaps.append({
                "execution_mode": r["execution_mode"], "tensor_parallel": tp,
                "kernel_gap_us": round(float(r["kernel_gap_us"]), 4),
                "num_gaps": int(r["num_gaps"]), "chain": r["chain"]})
            env = {"gpu_name": r["gpu_name"], "driver_version": r["driver_version"],
                   "torch_version": r["torch_version"], "cuda_version": r["cuda_version"]}
    return env, gaps


def holdout_report_tp(linear_med, attn_med) -> dict:
    """留出误差（键含 tp；训练/留出按 (op, tp, shape) 哈希划分）。"""
    import math, bisect

    def err_stats(errs):
        if not errs:
            return None
        es = sorted(errs)
        n = len(es)

        def pct(p):
            return round(es[min(n - 1, int(p * n))], 4)
        return {"n": n, "median_rel_err": pct(0.5), "p90_rel_err": pct(0.9),
                "max_rel_err": round(es[-1], 4)}

    def interp_log(train: dict, x: float):
        xs = sorted(train)
        if x < xs[0] or x > xs[-1]:
            return None
        if x in train:
            return train[x]
        i = bisect.bisect_left(xs, x)
        x0, x1 = xs[i - 1], xs[i]
        lx, lx0, lx1 = math.log10(x), math.log10(x0), math.log10(x1)
        w = (lx - lx0) / (lx1 - lx0) if lx1 != lx0 else 0.0
        return train[x0] * (1 - w) + train[x1] * w

    def interp_kv(pts, kv):
        xs = sorted(pts)
        if kv < xs[0] or kv > xs[-1]:
            return None
        if kv in pts:
            return pts[kv]
        i = bisect.bisect_left(xs, kv)
        x0, x1 = xs[i - 1], xs[i]
        w = (kv - x0) / (x1 - x0) if x1 != x0 else 0.0
        return pts[x0] * (1 - w) + pts[x1] * w

    lin = {}
    for op, table in linear_med.items():
        errs = []
        # TP 不变组（TP1 测量复制到各 TP 键）：值域同源，取代表键评估一次
        tp_keys = (TPS[0],) if op in TP_INVARIANT_GROUPS else TPS
        for tp in tp_keys:
            sub = {t: v for (t2, t), v in table.items() if t2 == tp}
            if not sub:
                continue
            train = {t: v for t, v in sub.items()
                     if not _holdout(f"lin:{op}:{tp}:{t}")}
            for t, v in sub.items():
                if t in train:
                    continue
                p = interp_log(train, t)
                if p is not None and v > 0:
                    errs.append(abs(p - v) / v)
        rep = err_stats(errs)
        if rep:
            lin[op] = rep
    attn = {}
    for op in ATTN_DECODE_OPS:
        grid = {k: {op: v[op]} for k, v in attn_med.items() if op in v}
        errs = []
        for tp in TPS:
            sub = {k: v for k, v in grid.items() if k[0] == tp}
            train = {k: v for k, v in sub.items()
                     if not _holdout(f"attn:{op}:{tp}:{k[1]}:{k[2]}")}
            for key, v in sub.items():
                if key in train:
                    continue
                p = _bilinear_tp(train, key[1], key[2], op)
                if p is not None and v[op] > 0:
                    errs.append(abs(p - v[op]) / v[op])
        rep = err_stats(errs)
        if rep:
            attn[op] = rep
    return {"linear_ops": lin, "attn_decode": attn,
            "method": "键 md5 哈希 15% 留出（键含 tp）；log-token 线性 / "
                      "(log-batch, kv) 双线性插值于训练网格；相对误差"}


def _bilinear_tp(train, batch, kv, op):
    """同 TP 训练点上的 (log-batch, kv) 双线性插值；train 键为 (tp, b, k)。"""
    import math, bisect
    by_batch = {}
    for (_tp, b, k), v in train.items():
        by_batch.setdefault(b, {})[k] = v[op]
    bs = sorted(by_batch)
    if not bs or batch < bs[0] or batch > bs[-1]:
        return None

    def kv_i(b):
        row = by_batch[b]
        ks = sorted(row)
        if kv < ks[0] or kv > ks[-1]:
            return None
        if kv in row:
            return row[kv]
        i = bisect.bisect_left(ks, kv)
        k0, k1 = ks[i - 1], ks[i]
        w = (kv - k0) / (k1 - k0) if k1 != k0 else 0.0
        return row[k0] * (1 - w) + row[k1] * w

    if batch in by_batch:
        return kv_i(batch)
    i = bisect.bisect_left(bs, batch)
    b0, b1 = bs[i - 1], bs[i]
    v0, v1 = kv_i(b0), kv_i(b1)
    if v0 is None or v1 is None:
        return None
    lb, lb0, lb1 = math.log10(batch), math.log10(b0), math.log10(b1)
    w = (lb - lb0) / (lb1 - lb0) if lb1 != lb0 else 0.0
    return v0 * (1 - w) + v1 * w


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pi-viewer", required=True)
    ap.add_argument("--out", default="assets/profiles")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    pi = Path(args.pi_viewer)
    here = Path(__file__).resolve().parent.parent
    out = here / args.out / OUT_DIR_NAME
    manifest_path = out / "manifest.json"

    if args.check:
        man = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name, sha in man["distill_files"].items():
            if _sha256(out / name) != sha:
                print(f"FAIL 哈希不符 {name}")
                return 1
        print("OK 产物哈希与 manifest 一致")
        return 0

    missing = [s for s in SRC_FILES if not (pi / s).exists()]
    if missing:
        print("FAIL 源缺失:", missing)
        return 1

    src_hashes = {s: _sha256(pi / s) for s in SRC_FILES}
    linear_med, linear_cnt = load_linear_tp(pi)
    attn_med, attn_cnt = load_attn_decode_tp(pi)
    env, gaps = load_gaps(pi)
    cfg = json.loads((pi / SRC_FILES[7]).read_text(encoding="utf-8"))

    linear_ops_json = {
        "unit": "ms", "statistic": "median", "measurement_type": "CUDA_EVENT",
        "tp_keyed": True,
        "grid": {op: {f"{tp}:{t}": v for (tp, t), v in tab.items()}
                 for op, tab in linear_med.items()}}
    linear_kc_json = {
        "unit": "kernels", "measurement_type": "KERNEL_ONLY", "tp_keyed": True,
        "grid": {op: {f"{tp}:{t}": v for (tp, t), v in tab.items()}
                 for op, tab in linear_cnt.items()}}
    attn_json = {
        "unit": "ms", "statistic": "median", "measurement_type": "CUDA_EVENT",
        "phase": "decode", "mode": "even", "attention_backend": "FLASHINFER",
        "tp_keyed": True,
        "points": [
            {"tp": tp, "batch": b, "kv": k,
             "median_ms": {op: vals[op] for op in ATTN_DECODE_OPS if op in vals},
             "kernel_count": {op: attn_cnt[(tp, b, k)][op]
                              for op in ATTN_DECODE_OPS
                              if (tp, b, k) in attn_cnt and
                              op in attn_cnt[(tp, b, k)]}}
            for (tp, b, k), vals in sorted(attn_med.items())]}

    out.mkdir(parents=True, exist_ok=True)
    files = {"linear_ops.json": linear_ops_json,
             "linear_kernel_counts.json": linear_kc_json,
             "attn_decode.json": attn_json}
    distill_hashes = {}
    for name, payload in files.items():
        p = out / name
        p.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                     encoding="utf-8")
        distill_hashes[name] = _sha256(p)

    rep = holdout_report_tp(linear_med, attn_med)
    manifest = {
        "profile_set_id": PROFILE_SET_ID,
        "created_at": "2026-11-10",
        "extractor": "tools/distill_h200_235b.py",
        "rounding": f"median 四舍五入 {ROUND_MS} 位小数（ms）",
        "model_identity": {
            "repo_id": "Qwen/Qwen3-235B-A22B",
            "revision": "8efa61729e24bd65b1d152b5ab5409052aa80e65",
            "config_sha256": _sha256(here / "assets" / "config"
                                     / "qwen3_235b_a22b__config.json"),
            "note": "官方 main 快照（2026-11-10 HF resolve/main 获取）；"
                    "新仓 catalog 已登记 qwen3_235b_a22b（真模型，非复制品）"},
        "source": {
            "repo": "pi-viewer (本地路径 D:/Github/pi-viewer，只读)",
            "git_head": _git_head(pi),
            "files_sha256": src_hashes,
            "measurement_contract": "同 30B-tiny 闭环（CUDA_EVENT device "
                                    "wall-clock；kernel_only 记 kernel_count）",
            "measurement_caveats": [
                "同 30B-tiny 闭环 5 条 caveats（括号内 host 空闲/warm-L2/"
                "3+20 与 3+5 重复/host-ahead gap/even 均匀上下文），见其 manifest",
                "TP2/8 为 per-worker 形状测量（q 头 32/8、kv 复制 4 头@TP8）——"
                "与新仓 gqa 行 tp 切分口径一致（heads_c/kv_heads_c）",
            ],
        },
        "environment": env,
        "kernel_gap": gaps,
        "composition": {
            "selected": "域内实测 median 之和 + 域外解析（差额法） + eager 间隙补贴",
            "gap_policy": "按计时区间边界数计（同 30B-tiny 2026-11-10 修正；"
                          "区间内间隙已含于 CUDA_EVENT median）；eager gap 取"
                          "本 set 同 TP 行（TP2 1.5269µs / TP8 1.2384µs）",
        },
        "admission": {
            "admit": {
                "linear_op.csv:TP2/8:sharded": "387 token 网格 ×3 sharded 算子"
                                              "（attn_pre_proj/attn_rope/"
                                              "attn_post_proj）×2 TP 档；BF16 "
                                              "CUDA_EVENT per-worker 形状",
                "linear_op.csv:TP1:replicated": "emb/input_layernorm/"
                                               "post_attention_layernorm 仅 TP1 "
                                               "campaign 采（387 点）；per-rank "
                                               "全宽 TP 不变 → 并入 TP2/8 域"
                                               "（tp_invariant_groups 登记；emb 为"
                                               "全词表口径，与新仓未切分行同尺度）",
                "attention.csv:decode-even:TP2/8": "每 TP 5184 (B 1..64 × kv "
                                                  "0..4095) 点；FLASHINFER/BF16/"
                                                  "block16",
                "kernel_gap.csv:TP2/8": "per-TP eager/cuda_graph 间隙 + 环境元数据",
            },
            "reject": {},
            "defer": {
                "moe.csv:TP8/EP1": "真 128E vllm_fused 数据在册，但新仓 count 的 "
                                   "MoE 行为整模型口径、memory 按 experts//tp——"
                                   "per-rank 分片账本（V04）未统一前实测替换会"
                                   "混尺；待 V04 落地后升级为准入（首选增量路径）",
                "moe.csv:TP1/EP4": "DP4 部署（每卡 32 专家），与 TP1 单卡场景"
                                   "口径不符",
                "linear:TP1:sharded / attention:TP1": "sharded 组 TP1 行为空"
                                                     "（campaign 只在 TP2/8 采）"
                                                     "且无 TP1 kernel_gap 行；"
                                                     "TP1 场景用 30B-tiny set",
                "attention:prefill/true_mixed": "v1 域仅 decode（同 30B-tiny）",
                "cpu_overheads/llama2/其它": "未用（同 30B-tiny 裁决）",
            },
        },
        "tp_invariant_groups": list(TP_INVARIANT_GROUPS),
        "domain": {
            "device": "h200_sxm", "precision": "BF16",
            "tensor_parallel": list(TPS),
            "attention_backend": "FLASHINFER", "kv_block": 16,
            "linear_tokens": [1, 16384],
            "attn_decode": {"batch": [1, 64], "kv": [0, 4095]},
            "shape_identity": {
                "model_type": cfg.get("model_type"),
                "hidden_size": cfg["hidden_size"],
                "n_q_head": cfg["num_attention_heads"],
                "n_kv_head": cfg["num_key_value_heads"],
                "head_dim": cfg.get("head_dim"),
                "vocab_size": cfg["vocab_size"],
                "num_experts": cfg["num_experts"],
                "num_experts_per_tok": cfg["num_experts_per_tok"],
                "use_qk_norm": True,      # 官方 config 无键；HF Qwen3Moe 固定含 q/k norm
                "use_sliding_window": bool(cfg.get("use_sliding_window", False)),
                "rope_theta": cfg.get("rope_theta"),
                "full_rotary": True, "attention": "full",
            },
            "applicable_stages": "decode(Q=1)/verify 的线性算子（TP2/8）；"
                                 "decode 注意力块（TP2/8）；MoE 不在域内",
        },
        "holdout": {"rule": "md5(key 含 tp) % 100 < 15",
                    "error_report": rep},
        "distill_files": distill_hashes,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1),
                             encoding="utf-8")
    print(f"OK 蒸馏完成 → {out}")
    for grp in ("linear_ops", "attn_decode"):
        for op, r in rep.get(grp, {}).items():
            print(f"   holdout[{op}] n={r['n']} median={r['median_rel_err']}"
                  f" p90={r['p90_rel_err']} max={r['max_rel_err']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
