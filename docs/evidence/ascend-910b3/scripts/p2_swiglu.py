import json, math, os, time
import torch, torch_npu
torch.npu.set_device(0)
LIMIT = 10.0
ALPHA, BETA = 1.0, 0.0

def _sigmoid(x):
    if x != x: return x
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)

def _clamp_max(x, hi):
    if x != x: return x
    return min(x, hi)

def _clamp_minmax(x, lo, hi):
    if x != x: return x
    return max(lo, min(x, hi))

def silu_and_mul_with_clamp(gate_raw, up_raw, limit=LIMIT, alpha=ALPHA, beta=BETA):
    gate = [_clamp_max(g, limit) for g in gate_raw]
    up = [_clamp_minmax(u, -limit, limit) for u in up_raw]
    return [g * _sigmoid(alpha * g) * (u + beta) for g, u in zip(gate, up)]

cases = {
  "neg_gate_passes_through": [(-100.0, -100.0)],
  "pos_gate_clamped":        [(100.0, -100.0), (100.0, 100.0)],
  "at_limit":                [(LIMIT, LIMIT), (LIMIT, -LIMIT)],
  "beyond_limit_up":         [(0.0, LIMIT + 1e6), (0.0, -LIMIT - 1e6)],
  "eps_below_limit":         [(LIMIT - 1e-6, LIMIT - 1e-6)],
  "zero":                    [(0.0, 0.0)],
  "negative_both":           [(-0.5, -0.5), (-20.0, 20.0)],
  "nan_gate":                [(float("nan"), 1.0)],
  "nan_up":                  [(1.0, float("nan"))],
  "inf_gate":                [(float("inf"), 1.0)],
  "neg_inf_up":              [(1.0, float("-inf"))],
}
res = {"kind": "gc18_swiglu_clamp_device_semantics",
       "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "device": "Ascend910B3",
       "ascend_rt_visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES", ""),
       "dtype": "float32", "swiglu_limit": LIMIT, "alpha": ALPHA, "beta": BETA,
       "timing_boundary": "device-executed", "cases": []}
worst = 0.0
for name, pairs in cases.items():
    g = [p[0] for p in pairs]; u = [p[1] for p in pairs]
    ref = silu_and_mul_with_clamp(g, u)
    gt = torch.tensor(g, dtype=torch.float32, device="npu")
    ut = torch.tensor(u, dtype=torch.float32, device="npu")
    gcl = torch.clamp(gt, max=LIMIT)
    ucl = torch.clamp(ut, min=-LIMIT, max=LIMIT)
    got = (gcl * torch.sigmoid(ALPHA * gcl) * (ucl + BETA)).cpu().tolist()
    rows = []
    for i, (a, b) in enumerate(zip(got, ref)):
        if a != a and b != b:
            rows.append({"i": i, "npu": "nan", "oracle": "nan", "ok": True, "abs": None})
            continue
        d = abs(a - b); rel = d / max(abs(b), 1e-30)
        ok = (d <= 1e-5) or (rel <= 1e-4)
        if a == a and b == b: worst = max(worst, rel)
        rows.append({"i": i, "npu": a, "oracle": b, "abs": d, "rel": rel, "ok": ok})
    res["cases"].append({"case": name, "rows": rows, "all_ok": all(r["ok"] for r in rows)})
res["all_cases_ok"] = all(c["all_ok"] for c in res["cases"])
res["worst_rel_diff"] = worst
print(json.dumps(res, indent=2, ensure_ascii=False))