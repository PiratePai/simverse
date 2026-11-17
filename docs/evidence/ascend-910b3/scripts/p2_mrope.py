import json, os, time
import torch, torch_npu
torch.npu.set_device(0)
D, THETA = 128, 1_000_000.0
inv_freq_cpu = (1.0 / (THETA ** (torch.arange(0, D, 2, dtype=torch.float64) / D))).float()

def rope_ref(x, pos, dtype):
    invf = inv_freq_cpu.to(dtype)
    p = torch.tensor([float(pos)], dtype=dtype)
    freqs = p.unsqueeze(1) * invf.unsqueeze(0)
    emb = torch.cat([freqs, freqs], dim=-1)
    xr = torch.cat([-x[..., D // 2:], x[..., :D // 2]], dim=-1)
    return x * torch.cos(emb) + xr * torch.sin(emb)

def rope_npu(x, pos):
    invf = inv_freq_cpu.to("npu")
    p = torch.tensor([float(pos)], dtype=torch.float32, device="npu")
    freqs = p.unsqueeze(1) * invf.unsqueeze(0)
    emb = torch.cat([freqs, freqs], dim=-1)
    xr = torch.cat([-x[..., D // 2:], x[..., :D // 2]], dim=-1)
    return x * torch.cos(emb) + xr * torch.sin(emb)

torch.manual_seed(20261117)
base = torch.randn(2, 4, D, dtype=torch.float32)
xn = base.to("npu")
res = {"kind": "gc16_mrope_text_rope_device_semantics",
       "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "device": "Ascend910B3",
       "ascend_rt_visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES", ""),
       "head_dim": D, "rope_theta": THETA,
       "comparisons": {"device_equivalence": "NPU float32 vs CPU float32",
                       "algorithm_precision": "NPU float32 vs CPU float64"},
       "timing_boundary": "device-executed", "positions": [], "three_row_identity": None}
for pos in (0, 1, 127, 1000, 32767):
    got = rope_npu(xn, pos).cpu()
    d_f32 = (got - rope_ref(base, pos, torch.float32)).abs()
    d_f64 = (got.double() - rope_ref(base.double(), pos, torch.float64)).abs()
    res["positions"].append({"pos": pos,
                             "vs_cpu_fp32_max_abs": float(d_f32.max()),
                             "vs_cpu_fp64_max_abs": float(d_f64.max())})
half = D // 2
emb_h = torch.zeros(1, half, dtype=torch.float64)
k = 0
for n in (D // 4, D // 4, D // 2):
    emb_h[0, k:k + n // 2] = 1234.0 * inv_freq_cpu.double()[: n // 2]
    k += n // 2
emb = torch.cat([emb_h, emb_h], dim=-1)
x1 = base[0, 0:1].double()
xr = torch.cat([-x1[..., D // 2:], x1[..., :D // 2]], dim=-1)
three = x1 * torch.cos(emb) + xr * torch.sin(emb)
one = rope_ref(x1, 1234.0, torch.float64)
res["three_row_identity"] = {"max_abs_diff": float((three - one).abs().max()),
                             "bit_identical": bool(torch.equal(three, one))}
res["max_vs_cpu_fp32"] = max(p["vs_cpu_fp32_max_abs"] for p in res["positions"])
res["max_vs_cpu_fp64"] = max(p["vs_cpu_fp64_max_abs"] for p in res["positions"])
print(json.dumps(res, indent=2, ensure_ascii=False))