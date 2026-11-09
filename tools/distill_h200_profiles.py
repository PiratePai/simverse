"""H200 profile 蒸馏导入器（可重放；MERGE_CRITERIA §3）。

从 pi-viewer 固定提交读取 Qwen3-30B-A3B-tiny 的 H200 采集 CSV，蒸馏为 new-viewer
仓内自包含校准表（不运行时依赖旧仓路径），并产出含来源 SHA256/准入/适用域/留出
划分/插值误差报告的 manifest。

用法：
  py -3.12 tools/distill_h200_profiles.py --pi-viewer D:/Github/pi-viewer \
      [--out assets/profiles] [--check]

  --check 只校验已有产物与源哈希一致，不重写。

蒸馏内容（第一条实测校准闭环，v1 域）：
  * linear_ops.json      6 算子 × num_tokens 网格的 CUDA_EVENT median（ms）
  * linear_kernel_counts.json  同网格 KERNEL_ONLY kernel_count（gap 估算用）
  * attn_decode.json     (batch 1..64 × kv 0..4095) 逐点 5 算子 median + kernel_count
                         （decode、even、FLASHINFER、BF16、TP1）
准入裁决（写入 manifest.admission）：
  * admit  : linear_op / attention(decode-even) / kernel_gap(环境+间隙)
  * reject : moe（16 专家缩小复制品 ≠ 真模型 128E；形状域不符）
  * defer  : attention prefill/true_mixed（v1 未用，域内未蒸馏）、cpu_overheads、
             235B/llama2 其它 H200 数据（另立闭环）
铁律：只蒸馏不改数值；median 取自源 CSV time_stats.<op>.median 列，四舍五入到
1e-6 ms 精度（记录舍入政策）。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

MODEL_DIR = "Qwen3-30B-A3B-tiny"
PROFILE_SET_ID = "h200-qwen3-30b-a3b-tiny-op-v1"
OUT_DIR_NAME = "h200_qwen3_30b_a3b_tiny"
ROUND_MS = 6  # 小数位（1e-6 ms = 1 ns）

LINEAR_OPS = ["emb", "input_layernorm", "attn_pre_proj", "attn_rope",
              "attn_post_proj", "post_attention_layernorm"]
ATTN_DECODE_OPS = ["attn_input_reshape", "attn_kv_cache_save", "attn_decode",
                   "attn_output_reshape"]
# attn_kv_cache_extract 仅出现在 prefill/true_mixed 行（核对原始 CSV），decode-even
# 步为上列 4 算子；decode 块计价固定为这 4 项之和。
HOLDOUT_FRAC_NUM = 15   # 15/100 按键哈希留出
HOLDOUT_FRAC_DEN = 100

SRC_FILES = [
    "data/profiling/compute/h200/Qwen3-30B-A3B-tiny/linear_op.csv",
    "data/profiling/compute/h200/Qwen3-30B-A3B-tiny/linear_op_kernel_only.csv",
    "data/profiling/compute/h200/Qwen3-30B-A3B-tiny/attention.csv",
    "data/profiling/compute/h200/Qwen3-30B-A3B-tiny/attention_kernel_only.csv",
    "data/profiling/compute/h200/Qwen3-30B-A3B-tiny/moe.csv",
    "data/profiling/compute/h200/Qwen3-30B-A3B-tiny/kernel_gap.csv",
    "data/profiling/cpu_overhead/h200_dgx/Qwen3-30B-A3B-tiny/cpu_overheads.csv",
    "data/config/models/Qwen3-30B-A3B-tiny.json",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _holdout(key: str) -> bool:
    h = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16)
    return (h % HOLDOUT_FRAC_DEN) < HOLDOUT_FRAC_NUM


def _median_col(op: str) -> str:
    return f"time_stats.{op}.median"


def _count_col(op: str) -> str:
    return f"time_stats.{op}.kernel_count"


def _git_head(pi: Path) -> str:
    head = pi / ".git" / "HEAD"
    try:
        ref = head.read_text(encoding="utf-8").strip()
        if ref.startswith("ref: "):
            p = pi / ".git" / ref[5:].strip()
            return p.read_text(encoding="utf-8").strip()
        return ref
    except OSError:
        return "unknown"


def load_linear(pi: Path):
    """linear_op CUDA_EVENT medians + kernel counts，按 num_tokens。"""
    med, cnt = {}, {}
    with (pi / SRC_FILES[0]).open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            assert r["measurement_type"] == "CUDA_EVENT", "非 CUDA_EVENT 行混入"
            assert r["profiling_precision"] == "BF16"
            assert r["num_tensor_parallel_workers"] == "1"
            t = int(r["num_tokens"])
            for op in LINEAR_OPS:
                v = r.get(_median_col(op), "")
                if v != "":
                    med.setdefault(op, {})[t] = round(float(v), ROUND_MS)
    with (pi / SRC_FILES[1]).open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            assert r["measurement_type"] == "KERNEL_ONLY"
            t = int(r["num_tokens"])
            for op in LINEAR_OPS:
                v = r.get(_count_col(op), "")
                if v != "":
                    cnt.setdefault(op, {})[t] = int(float(v))
    return med, cnt


def load_attn_decode(pi: Path):
    """decode、even 行：per (batch,kv) 5 算子 median + kernel_count。"""
    med: dict = {}
    with (pi / SRC_FILES[2]).open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["is_prefill"] != "False" or r["mode"] != "even":
                continue
            assert r["measurement_type"] == "CUDA_EVENT"
            assert r["profiling_precision"] == "BF16"
            assert r["attention_backend"] == "FLASHINFER"
            assert r["num_tensor_parallel_workers"] == "1"
            key = (int(r["batch_size"]), int(r["kv_cache_size"]))
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
            key = (int(r["batch_size"]), int(r["kv_cache_size"]))
            cnt.setdefault(key, {})
            for op in ATTN_DECODE_OPS:
                v = r.get(_count_col(op), "")
                if v != "":
                    cnt[key][op] = int(float(v))
    return med, cnt


def load_env(pi: Path):
    env, gaps = {}, []
    with (pi / SRC_FILES[5]).open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            gaps.append({
                "execution_mode": r["execution_mode"],
                "kernel_gap_us": round(float(r["kernel_gap_us"]), 4),
                "num_gaps": int(r["num_gaps"]),
                "chain": r["chain"]})
            env = {"gpu_name": r["gpu_name"], "driver_version": r["driver_version"],
                   "torch_version": r["torch_version"], "cuda_version": r["cuda_version"]}
    return env, gaps


def interp_linear(table: dict, x: float) -> float | None:
    """训练网格上分段线性插值（log10 token）。域外返回 None。"""
    xs = sorted(table)
    if x < xs[0] or x > xs[-1]:
        return None
    if x in table:
        return table[x]
    import bisect
    import math
    i = bisect.bisect_left(xs, x)
    x0, x1 = xs[i - 1], xs[i]
    lx = math.log10(x)
    lx0, lx1 = math.log10(x0), math.log10(x1)
    w = (lx - lx0) / (lx1 - lx0) if lx1 != lx0 else 0.0
    return table[x0] * (1 - w) + table[x1] * w


def _interp_kv(train_by_batch: dict, batch: int, kv: int) -> float | None:
    """同 batch 内沿 kv 轴对**训练点**括号线性插值（评估留出点专用；
    禁止最近邻回退——那会把留出误差偷换成网格间距）。域外 None。"""
    import bisect
    pts = train_by_batch.get(batch)
    if not pts:
        return None
    xs = sorted(pts)
    if kv < xs[0] or kv > xs[-1]:
        return None
    if kv in pts:
        return pts[kv]
    i = bisect.bisect_left(xs, kv)
    x0, x1 = xs[i - 1], xs[i]
    w = (kv - x0) / (x1 - x0) if x1 != x0 else 0.0
    return pts[x0] * (1 - w) + pts[x1] * w


def _bilinear(grid: dict, batch: int, kv: int, op: str) -> float | None:
    """(batch, kv) 插值（留出评估用）：batch 轴取 log 括号、kv 轴线性括号，
    全部取训练点；任一轴括不出训练点 → None（不外推、不最近邻）。"""
    import bisect
    import math
    train_by_batch: dict = {}
    for (b, k), v in grid.items():
        val = v.get(op)
        if val is not None:
            train_by_batch.setdefault(b, {})[k] = val
    batches = sorted(train_by_batch)
    if not batches or batch < batches[0] or batch > batches[-1]:
        return None
    if batch in train_by_batch:
        return _interp_kv(train_by_batch, batch, kv)
    i = bisect.bisect_left(batches, batch)
    b0, b1 = batches[i - 1], batches[i]
    v0, v1 = _interp_kv(train_by_batch, b0, kv), _interp_kv(train_by_batch, b1, kv)
    if v0 is None or v1 is None:
        return None
    lb, lb0, lb1 = math.log10(batch), math.log10(b0), math.log10(b1)
    w = (lb - lb0) / (lb1 - lb0) if lb1 != lb0 else 0.0
    return v0 * (1 - w) + v1 * w


def holdout_report(linear_med: dict, attn_med: dict) -> dict:
    """留出误差：训练集=非留出键；对留出键插值并对比实测 median。"""
    def err_stats(errs):
        if not errs:
            return None
        es = sorted(errs)
        n = len(es)

        def pct(p):
            return round(es[min(n - 1, int(p * n))], 4)
        return {"n": n, "median_rel_err": pct(0.5), "p90_rel_err": pct(0.9),
                "max_rel_err": round(es[-1], 4)}

    lin = {}
    for op, table in linear_med.items():
        train = {t: v for t, v in table.items() if not _holdout(f"lin:{op}:{t}")}
        errs = []
        for t, v in table.items():
            if t in train:
                continue
            p = interp_linear(train, t)
            if p is not None and v > 0:
                errs.append(abs(p - v) / v)
        rep = err_stats(errs)
        if rep:
            lin[op] = rep
    attn = {}
    for op in ATTN_DECODE_OPS:
        grid = {k: {op: v[op]} for k, v in attn_med.items() if op in v}
        train = {k: v for k, v in grid.items() if not _holdout(f"attn:{op}:{k[0]}:{k[1]}")}
        errs = []
        for key, v in grid.items():
            if key in train:
                continue
            p = _bilinear(train, key[0], key[1], op)
            if p is not None and v[op] > 0:
                errs.append(abs(p - v[op]) / v[op])
        rep = err_stats(errs)
        if rep:
            attn[op] = rep
    return {"linear_ops": lin, "attn_decode": attn,
            "method": "键 md5 哈希 15% 留出；线性/双线性插值于训练网格；"
                      "相对误差 |pred-measured|/measured"}


def load_shape_identity(pi: Path) -> dict:
    """采集复制品 config 的逐算子形状+算子构成标识（生产端域门禁用）。

    2026-11-10 补强：QKNorm/滑窗/rope_theta 三项影响被测算子**构成**（块内
    是否含 QKNorm、rope 是否全 head_dim 旋转、是否全注意力），一并入册，
    生产端 _shape_mismatches 逐项核对——形状一致是必要条件，不是充分条件。
    """
    cfg = json.loads((pi / SRC_FILES[7]).read_text(encoding="utf-8"))
    return {
        "model_type": cfg.get("model_type"),
        "hidden_size": cfg["hidden_size"],
        "n_q_head": cfg["num_attention_heads"],
        "n_kv_head": cfg["num_key_value_heads"],
        "head_dim": cfg.get("head_dim"),
        "moe_intermediate_size": cfg["moe_intermediate_size"],
        "vocab_size": cfg["vocab_size"],
        "num_experts_replica": cfg["num_experts"],
        "num_experts_per_tok": cfg["num_experts_per_tok"],
        "torch_dtype": cfg.get("dtype") or cfg.get("torch_dtype"),
        # 算子构成语义（2026-11-10；采集实现证据 linear_op_impl.py CausalSelfAttention）
        "use_qk_norm": bool(cfg.get("use_qk_norm", False)),
        "use_sliding_window": bool(cfg.get("use_sliding_window", False)),
        "rope_theta": cfg.get("rope_theta"),
        "full_rotary": True,   # 采集 get_rope(rotary_dim=head_dim) 全旋转
        "attention": "full",   # 无滑窗全注意力（even decode 网格）
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pi-viewer", required=True)
    ap.add_argument("--out", default="assets/profiles")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    pi = Path(args.pi_viewer)
    out_root = Path(args.out)
    out = out_root / OUT_DIR_NAME
    here = Path(__file__).resolve().parent.parent

    manifest_path = here / out / "manifest.json"
    if args.check:
        man = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name, sha in man["distill_files"].items():
            p = here / out / name
            if not p.exists():
                print(f"FAIL 缺产物 {name}")
                return 1
            if _sha256(p) != sha:
                print(f"FAIL 哈希不符 {name}")
                return 1
        print("OK 产物哈希与 manifest 一致")
        return 0

    missing = [s for s in SRC_FILES if not (pi / s).exists()]
    if missing:
        print("FAIL 源缺失:", missing)
        return 1

    src_hashes = {s: _sha256(pi / s) for s in SRC_FILES}
    linear_med, linear_cnt = load_linear(pi)
    attn_med, attn_cnt = load_attn_decode(pi)
    env, gaps = load_env(pi)

    # 训练/留出键集（重放确定）
    linear_split = {
        "train_tokens": {op: sorted(t for t in tab if not _holdout(f"lin:{op}:{t}"))
                         for op, tab in linear_med.items()},
        "holdout_rule": f"md5(key) % {HOLDOUT_FRAC_DEN} < {HOLDOUT_FRAC_NUM}",
    }

    linear_ops_json = {
        "unit": "ms", "statistic": "median", "measurement_type": "CUDA_EVENT",
        "grid": {op: {str(t): v for t, v in tab.items()}
                 for op, tab in linear_med.items()}}
    linear_kc_json = {
        "unit": "kernels", "measurement_type": "KERNEL_ONLY",
        "grid": {op: {str(t): v for t, v in tab.items()}
                 for op, tab in linear_cnt.items()}}
    attn_json = {
        "unit": "ms", "statistic": "median", "measurement_type": "CUDA_EVENT",
        "phase": "decode", "mode": "even", "attention_backend": "FLASHINFER",
        "points": [
            {"batch": b, "kv": k,
             "median_ms": {op: vals[op] for op in ATTN_DECODE_OPS if op in vals},
             "kernel_count": {op: attn_cnt[(b, k)][op]
                              for op in ATTN_DECODE_OPS if (b, k) in attn_cnt
                              and op in attn_cnt[(b, k)]}}
            for (b, k), vals in sorted(attn_med.items())]}

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

    rep = holdout_report(linear_med, attn_med)
    n_attn = len(attn_med)
    manifest = {
        "profile_set_id": PROFILE_SET_ID,
        "created_at": "2026-11-09",
        "extractor": "tools/distill_h200_profiles.py",
        "rounding": f"median 四舍五入 {ROUND_MS} 位小数（ms）",
        "source": {
            "repo": "pi-viewer (本地路径 D:/Github/pi-viewer，只读)",
            "git_head": _git_head(pi),
            "files_sha256": src_hashes,
            "measurement_contract": "frontier/profiling/README.md: cuda_event→CUDA_EVENT"
                                    "（wall-clock device time between CUDA events）；"
                                    "kernel_only→KERNEL_ONLY 记 kernel_count",
            "measurement_caveats": [
                "CUDA_EVENT 事件括号内为 device wall-clock：launch-bound 时含 host "
                "launch 空闲，故逐 op median 含每 op 固定开销地板"
                "（device_timer.py CUDA_EVENT 分支；源码核对 2026-11-10）",
                "CUDA_EVENT 路径无 L2 刷新（仅 RECORD_FUNCTION 路径 _flush_l2）；"
                "WARMUP/ACTIVE 连续重复下权重/KV 可驻留 L2，weight-bound 小 GEMM "
                "实测值偏乐观（warm-L2 bias）",
                "重复次数：linear WARMUP=3/ACTIVE=20，attention WARMUP=3/ACTIVE=5"
                "（linear_op_wrapper.py / attention_wrapper.py 常量）",
                "kernel_gap eager 值在 host-ahead（spin kernel 占流、launch 预排队）"
                "条件下测得，为纯设备侧间隙（kernel_gap/main.py）；用作 launch-bound "
                "边界间隙补贴是低估",
                "decode-even：B 条序列×每条 1 新 query×均匀 kv_cache_size 上下文，"
                "分页 KV（block16，每序列独立连续 block，FLASHINFER）；连续组批的"
                "非均匀上下文为域外",
            ]},
        "environment": env,
        "kernel_gap": gaps,
        "composition": {
            "selected": "域内实测 median 之和 + 域外解析（差额法） + eager 间隙补贴",
            "gap_policy": "2026-11-10 修正：间隙补贴按计时区间边界数计（每个被测组"
                          "实例 1 个边界；注意力块 4 op 各自计时）。CUDA_EVENT median "
                          "的事件括号已含区间内 kernel 间隙（launch-bound 时含 host "
                          "空闲），按全部 kernel 数计费会重复计入区间内间隙（B8 全模型"
                          "673 kernel vs 433 区间，约 +0.32ms/步，~1.6%）。",
            "gap_bound": "eager 值为 host-ahead 纯设备间隙，用作 launch-bound 边界间隙"
                         "是低估；cuda_graph 档备选未采用",
        },
        "admission": {
            "admit": {
                "linear_op.csv": "387 token 网格 ×6 算子，TP1 BF16 CUDA_EVENT；"
                                 "形状与 Qwen3-30B-A3B 逐算子一致（hidden 2048/32q/4kv/"
                                 "head_dim 128/inter 768/vocab 151936）",
                "attention.csv": "decode+even 5184 (batch 1..64 × kv 0..4095) 点；"
                                 "FLASHINFER/BF16/TP1/block16",
                "kernel_gap.csv": "环境元数据 + eager/cuda_graph 每间隙 µs"},
            "reject": {
                "moe.csv": "16 专家缩小复制品（num_experts=16）≠ 真模型 128E；"
                           "grouped GEMM/router 形状域不符，不进入 v1 校准域"},
            "defer": {
                "attention.csv:prefill/true_mixed": "v1 闭环仅 decode；蒸馏未含 prefill",
                "cpu_overheads.csv": "v1 合成未用 CPU probe；已记录来源哈希",
                "Qwen3-235B-A22B/llama2_7b": "另立闭环（235B 需先在 catalog 登记）"},
            "replica_note": "采集模型为 pi-viewer data/config/models/Qwen3-30B-A3B-tiny.json"
                            "（官方 30B-A3B 的缩小复制品：层数 8、专家 16；逐算子形状"
                            "除专家数/router 输出维外与真模型一致）。校准域内算子不含"
                            "MoE/router，故专家数差异不影响已准入算子的形状一致性；"
                            "层数不进入逐算子计价。"},
        "domain": {
            "device": "h200_sxm", "precision": "BF16", "tensor_parallel": 1,
            "attention_backend": "FLASHINFER", "kv_block": 16,
            "linear_tokens": [1, 16384],
            "attn_decode": {"batch": [1, 64], "kv": [0, 4095]},
            "shape_identity": load_shape_identity(pi),
            "applicable_stages": "decode(Q=1)/verify 的线性算子；decode 的注意力块"},
        "holdout": {"rule": linear_split["holdout_rule"],
                    "train_keys_linear": {op: len(v) for op, v in
                                          linear_split["train_tokens"].items()},
                    "attn_points_total": n_attn,
                    "error_report": rep},
        "distill_files": distill_hashes,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1),
                             encoding="utf-8")
    print(f"OK 蒸馏完成 → {out}")
    print(f"   linear ops: {len(linear_med)} × {len(next(iter(linear_med.values())))} 点")
    print(f"   attn decode points: {n_attn}")
    for grp in ("linear_ops", "attn_decode"):
        for op, r in rep.get(grp, {}).items():
            print(f"   holdout[{op}] n={r['n']} median={r['median_rel_err']}"
                  f" p90={r['p90_rel_err']} max={r['max_rel_err']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
