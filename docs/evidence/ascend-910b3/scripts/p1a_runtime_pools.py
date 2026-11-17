import json, os, time
import torch, torch_npu
GiB = 2**30
torch.npu.set_device(0)
def mib(x): return x / (1 << 20)
res = {"kind": "r02b_runtime_pools", "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
       "ascend_rt_visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES", ""),
       "torch": torch.__version__, "torch_npu": torch_npu.__version__,
       "timing_boundary": "runtime-api", "probes": {}}

# --- 1. allocator padding overhead: request vs actual ---
torch.npu.empty_cache()
free0, _ = torch.npu.mem_get_info()
pad = []
for gib in (1, 2, 4):
    n = gib * GiB
    t = torch.empty(n, dtype=torch.uint8, device="npu")
    a = torch.npu.memory_allocated()
    r = torch.npu.memory_reserved()
    pad.append({"requested_bytes": n, "allocated_bytes": a, "reserved_bytes": r,
                "padding_bytes": r - n})
    del t
    torch.npu.empty_cache()
res["probes"]["allocator_padding"] = pad

# --- 2. GEMM workspace: tensor bytes vs peak allocated during the op ---
a = torch.randn(4096, 4096, dtype=torch.bfloat16, device="npu")
b = torch.randn(4096, 4096, dtype=torch.bfloat16, device="npu")
base = torch.npu.memory_allocated()
torch.npu.reset_peak_memory_stats()
c = torch.matmul(a, b)
torch.npu.synchronize()
peak = torch.npu.max_memory_allocated()
res["probes"]["gemm_workspace"] = {
    "shape": [4096, 4096, 4096], "dtype": "bfloat16",
    "input_bytes": base, "output_bytes": c.numel() * 2,
    "peak_allocated_bytes": peak, "workspace_bytes": peak - base - c.numel() * 2}
del a, b, c
torch.npu.empty_cache()

# --- 3. NPU graph pool (capture) ---
g = {}
try:
    static_in = torch.randn(1024, 1024, dtype=torch.bfloat16, device="npu")
    static_w = torch.randn(1024, 1024, dtype=torch.bfloat16, device="npu")
    torch.npu.synchronize()
    before = torch.npu.mem_get_info()[0]
    graph = torch.npu.NPUGraph()
    s = torch.npu.Stream()
    with torch.npu.stream(s):
        for _ in range(3):
            torch.matmul(static_in, static_w)
        torch.npu.synchronize()
        baseline = torch.npu.mem_get_info()[0]
        with torch.npu.graph(graph):
            out = torch.matmul(static_in, static_w)
        torch.npu.synchronize()
    after = torch.npu.mem_get_info()[0]
    g = {"ok": True, "free_before_capture": before, "free_after_capture": after,
         "graph_pool_bytes": before - after, "baseline_delta": before - baseline}
except Exception as e:
    g = {"ok": False, "error": "%s: %s" % (type(e).__name__, e)}
res["probes"]["npu_graph_pool"] = g
print(json.dumps(res, indent=2, ensure_ascii=False))