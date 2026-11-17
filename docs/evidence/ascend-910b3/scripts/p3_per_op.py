import csv, json, re, statistics, sys, time
import torch, torch_npu
from torch.nn import functional as F
torch.npu.set_device(0)

GEMM = re.compile(r"\[(\d+)[,;](\d+)\]@\[(\d+)[,;](\d+)\]")

def bench(fn, est_ms):
    iters = int(min(400, max(10, 20.0 / max(est_ms, 1e-4))))
    for _ in range(20):
        fn()
    torch.npu.synchronize()
    per = []
    for _ in range(5):
        t = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.npu.synchronize()
        per.append((time.perf_counter() - t) * 1e3 / iters)
    per.sort()
    return iters, statistics.median(per), min(per), per[-1]

def build(op, shape, flops):
    m = GEMM.match(shape or "")
    if m:
        M, K, _, N = (int(x) for x in m.groups())
        a = torch.randn(M, K, dtype=torch.bfloat16, device="npu")
        w = torch.randn(K, N, dtype=torch.bfloat16, device="npu")
        return ("%d×%d×%d" % (M, K, N), 2 * M * K * N, "gemm",
                lambda: torch.matmul(a, w))
    if op == "lm_head":
        K = 2048; N = int(flops // (2 * K))
        a = torch.randn(1, K, dtype=torch.bfloat16, device="npu")
        w = torch.randn(K, N, dtype=torch.bfloat16, device="npu")
        return ("1×%d×%d" % (K, N), 2 * K * N, "gemm", lambda: torch.matmul(a, w))
    if op == "moe_shared":
        h, inter = 2048, 512
        x = torch.randn(1, h, dtype=torch.bfloat16, device="npu")
        w1 = torch.randn(h, inter, dtype=torch.bfloat16, device="npu")
        w2 = torch.randn(inter, h, dtype=torch.bfloat16, device="npu")
        return ("1×%d×%d + 1×%d×%d" % (h, inter, inter, h), 4 * h * inter, "gemm",
                lambda: torch.matmul(F.silu(torch.matmul(x, w1)), w2))
    if op == "moe_routed":
        h, inter, kk = 2048, 512, 8
        x = torch.randn(kk, 1, h, dtype=torch.bfloat16, device="npu")
        w1 = torch.randn(kk, h, inter, dtype=torch.bfloat16, device="npu")
        w2 = torch.randn(kk, inter, h, dtype=torch.bfloat16, device="npu")
        return ("grouped k=%d h=%d inter=%d" % (kk, h, inter), 4 * kk * h * inter, "moe",
                lambda: torch.bmm(F.silu(torch.bmm(x, w1)), w2))
    if op == "attention_core":
        h, d, S = 2, 256, 32769
        q = torch.randn(h, 1, d, dtype=torch.bfloat16, device="npu")
        k = torch.randn(h, S, d, dtype=torch.bfloat16, device="npu")
        v = torch.randn(h, S, d, dtype=torch.bfloat16, device="npu")
        return ("dense-SDPA h=%d S=%d d=%d" % (h, S, d), 4 * h * S * d, "attention-proxy",
                lambda: F.scaled_dot_product_attention(q, k, v))
    if op == "causal_conv1d":
        c, ksz = 1024, 4
        x = torch.randn(1, c, 8, dtype=torch.bfloat16, device="npu")
        w = torch.randn(c, 1, ksz, dtype=torch.bfloat16, device="npu")
        return ("depthwise c=%d k=%d" % (c, ksz), 2 * c * 8 * ksz, "elementwise",
                lambda: F.conv1d(x, w, groups=c))
    if op in ("attn_norm", "mlp_norm", "final_norm"):
        w = 2048
        x = torch.randn(1, w, dtype=torch.bfloat16, device="npu")
        g = torch.randn(w, dtype=torch.bfloat16, device="npu")
        return ("rmsnorm w=%d" % w, 2 * w, "elementwise",
                lambda: (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)) * g)
    if op in ("q_norm", "k_norm"):
        h, d = (16, 256) if op == "q_norm" else (2, 256)
        x = torch.randn(h, 1, d, dtype=torch.bfloat16, device="npu")
        g = torch.randn(d, dtype=torch.bfloat16, device="npu")
        return ("rmsnorm h=%d d=%d" % (h, d), 2 * h * d, "elementwise",
                lambda: (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)) * g)
    if op == "rope":
        x = torch.randn(2, 1, 64, dtype=torch.bfloat16, device="npu")
        c = torch.randn(1, 1, 64, dtype=torch.bfloat16, device="npu")
        s = torch.randn(1, 1, 64, dtype=torch.bfloat16, device="npu")
        rot = torch.cat([-x[..., 32:], x[..., :32]], dim=-1)
        return ("rotary d=64", 6 * 64, "elementwise", lambda: x * c + rot * s)
    if op in ("attn_add", "mlp_add"):
        w = 2048
        a = torch.randn(1, w, dtype=torch.bfloat16, device="npu")
        b = torch.randn(1, w, dtype=torch.bfloat16, device="npu")
        return ("add w=%d" % w, w, "elementwise", lambda: torch.add(a, b))
    if op == "embed_lookup":
        V, h = 248320, 2048
        tbl = torch.randn(V, h, dtype=torch.bfloat16, device="npu")
        idx = torch.zeros(1, dtype=torch.long, device="npu")
        return ("gather V=%d h=%d" % (V, h), 0, "elementwise",
                lambda: F.embedding(idx, tbl))
    return None

E = "/data/p00929506/TQ/npu-evidence/20261117"
rows = [r for r in csv.DictReader(open(E + "/raw/ops_qwen.csv")) if r["stage"] == "target_decode"]
bundle = json.load(open(E + "/raw/run_qwen.json"))
q = bundle["scenario_snapshot"].get("quant", {})
seen, recs, skipped = set(), [], set()
for r in rows:
    key = (r["op"], r["shape"])
    if key in seen:
        continue
    seen.add(key)
    built = build(r["op"], r["shape"], int(r["flops"]))
    if not built:
        skipped.add(r["op"]); continue
    label, flops, kind, fn = built
    est = flops / 100e12 * 1e3 if flops else 0.02
    analytic_us = float(r["t_ms"]) * 1e3
    try:
        iters, med, lo, hi = bench(fn, est)
        rec = {"op": r["op"], "shape_label": label, "kind": kind, "flops": flops,
               "bound": r.get("bound"), "iters_per_window": iters,
               "median_us": med * 1e3, "min_us": lo * 1e3, "max_us": hi * 1e3,
               "achieved_tflops": (flops / (med * 1e-3) / 1e12) if flops else 0.0,
               "analytic_us": analytic_us,
               "measured_over_analytic": (med / (analytic_us * 1e-3)) if analytic_us >= 1.0 else None}
    except Exception as e:
        rec = {"op": r["op"], "shape_label": label, "kind": kind, "error": "%s: %s" % (type(e).__name__, e)}
    recs.append(rec)
    torch.npu.empty_cache()

print("=== qwen36_35b_a3b target_decode | quant=%s | measured dtype=bf16 ===" % json.dumps(q))
print("%-14s %-20s %-16s %9s %9s %8s %10s %7s" % ("op", "shape", "kind", "med_us", "min_us", "TFLOPS", "analytic_us", "ratio"))
for r in recs:
    if "error" in r:
        print("%-14s %-20s %-16s ERR %s" % (r["op"], r["shape_label"], r["kind"], r["error"]))
    else:
        print("%-14s %-20s %-16s %9.2f %9.2f %8.2f %10.2f %7s" % (
            r["op"], r["shape_label"], r["kind"], r["median_us"], r["min_us"],
            r["achieved_tflops"], r["analytic_us"],
            ("%.2f" % r["measured_over_analytic"]) if r["measured_over_analytic"] else "-"))
print("no-equivalent:", sorted(skipped))

# M 扫描：同一 GEMM 在不同 M 下的达标算力（说明 decode M=1 是启动/访存主导）
K, N = 2048, 1024
scale = []
for M in (1, 8, 64, 256, 1024):
    a = torch.randn(M, K, dtype=torch.bfloat16, device="npu")
    w = torch.randn(K, N, dtype=torch.bfloat16, device="npu")
    fl = 2 * M * K * N
    iters, med, lo, _ = bench(lambda: torch.matmul(a, w), fl / 100e12 * 1e3)
    scale.append({"M": M, "median_us": med * 1e3, "achieved_tflops": fl / (med * 1e-3) / 1e12})
print("\n=== M 扫描 (in_proj_qkv K=2048 N=1024) ===")
for s in scale:
    print("M=%-5d %9.2f us %8.2f TFLOPS" % (s["M"], s["median_us"], s["achieved_tflops"]))
json.dump({"tag": "qwen36_35b_a3b target_decode", "quant": q, "measured_dtype": "bf16",
           "records": recs, "no_equivalent": sorted(skipped), "m_scale": scale},
          open(E + "/raw/p3_per_op_qwen.json", "w"), indent=2, ensure_ascii=False)