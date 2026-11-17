import json, math, os, time
import torch, torch_npu
torch.npu.set_device(0)
E, K, RSF, NEG = 256, 6, 1.5, -1e30

def _weight(x):
    if x != x: return float("nan")
    if x > 20.0: return math.sqrt(x)
    e = math.exp(x) if x > -87.3 else 0.0
    return math.sqrt(math.log1p(e))

def ref_route(logits, bias):
    n = len(logits); weights = [_weight(x) for x in logits]
    cur = [(weights[i] + bias[i]) if (weights[i] + bias[i]) == (weights[i] + bias[i]) else NEG
           for i in range(n)]
    chosen = []
    for _ in range(K):
        m = max(cur); eid = min(i for i in range(n) if cur[i] == m)
        chosen.append(eid); cur[eid] = -math.inf
    w = [weights[i] for i in chosen]
    s = sum(w); denom = s if s > 0.0 else 1.0
    return chosen, [x * RSF / denom for x in w]

def npu_route(logits, bias):
    lg = torch.tensor(logits, dtype=torch.float32, device="npu")
    bs = torch.tensor(bias, dtype=torch.float32, device="npu")
    w = torch.sqrt(torch.where(lg > 20.0, lg, torch.log1p(torch.exp(lg))))
    sel = w + bs
    sel = torch.where(sel != sel, torch.full_like(sel, NEG), sel)
    chosen = []
    work = sel.clone()
    for _ in range(K):
        idx = int(torch.argmax(work).item())
        chosen.append(idx)
        work[idx] = float("-inf")
    wv = w[torch.tensor(chosen, device="npu")]
    s = float(wv.sum().item())
    denom = s if s > 0.0 else 1.0
    return chosen, (wv * RSF / denom).cpu().tolist()

import random
random.seed(20261117)
scen = {}
logits = [random.uniform(-5, 5) for _ in range(E)]
bias = [0.002 * ((i * 37) % 11 - 5) for i in range(E)]
scen["random_normal"] = (logits, bias)
tie_l = [-1.0] * E; tie_b = [0.5] * E
scen["all_tie"] = (tie_l, tie_b)
nan_l = list(logits); nan_l[7] = float("nan"); nan_b = [0.1] * E; nan_b[7] = 100.0
scen["nan_logit_huge_bias"] = (nan_l, nan_b)
uf_l = [-1000.0] * E; uf_b = [((i * 29) % 13) * 0.01 for i in range(E)]
scen["underflow_all_zero"] = (uf_l, uf_b)
guard_l = [21.0] * E; guard_b = [0.0] * E
scen["guard_gt20"] = (guard_l, guard_b)

res = {"kind": "gc13_moe_route_device_semantics",
       "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "device": "Ascend910B3",
       "ascend_rt_visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES", ""),
       "dtype": "float32", "experts": E, "top_k": K, "routed_scaling_factor": RSF,
       "timing_boundary": "device-executed", "cases": []}
for name, (lg, bs) in scen.items():
    rc, rw = ref_route(lg, bs)
    nc, nw = npu_route(lg, bs)
    ids_ok = rc == nc
    wd = [abs(a - b) for a, b in zip(nw, rw)]
    w_rel = max([d / max(abs(b), 1e-30) for d, b in zip(wd, rw)] or [0.0])
    res["cases"].append({"case": name, "oracle_ids": rc, "npu_ids": nc, "ids_match": ids_ok,
                         "weights_sum": sum(nw), "oracle_weights_sum": sum(rw),
                         "max_abs_w_diff": max(wd) if wd else None,
                         "max_rel_w_diff": w_rel,
                         "weights_ok": w_rel <= 1e-5})
res["all_ids_match"] = all(c["ids_match"] for c in res["cases"])
res["all_weights_ok"] = all(c["weights_ok"] for c in res["cases"])
print(json.dumps(res, indent=2, ensure_ascii=False))