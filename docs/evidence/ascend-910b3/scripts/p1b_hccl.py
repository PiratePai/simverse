import json, os, statistics, time
import torch, torch_npu
import torch.distributed as dist

rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
local = int(os.environ.get("LOCAL_RANK", rank))
torch.npu.set_device(local)
E = "/data/p00929506/TQ/npu-evidence/20261117/raw"
out = {"kind": "r17_hccl_collectives", "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
       "rank": rank, "world_size": world, "device_name": torch.npu.get_device_name(local),
       "ascend_rt_visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES", ""),
       "timing_boundary": "device-sync wallclock", "warmup": 5, "iters": 20, "results": []}
free_before, total = torch.npu.mem_get_info()
dist.init_process_group(backend="hccl")
free_after_init, _ = torch.npu.mem_get_info()
out["comm_pool_at_init_bytes"] = free_before - free_after_init

SZ = [1 << 10, 1 << 12, 1 << 14, 1 << 16, 1 << 18, 1 << 20, 1 << 22, 1 << 24, 1 << 26, 1 << 28]
OPS = ["all_reduce", "all_gather", "reduce_scatter", "all_to_all_single", "broadcast"]

def timed(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    s = []
    for _ in range(iters):
        torch.npu.synchronize(); t = time.perf_counter(); fn(); torch.npu.synchronize()
        s.append((time.perf_counter() - t) * 1e3)
    return statistics.median(s), min(s), max(s)

# warm one collective to force communicator buffer allocation, then re-measure
w = torch.ones(1024, dtype=torch.bfloat16, device="npu")
dist.all_reduce(w); torch.npu.synchronize()
free_after_warm, _ = torch.npu.mem_get_info()
out["comm_pool_after_warmup_bytes"] = free_before - free_after_warm

for nbytes in SZ:
    n = nbytes // 2
    x = torch.ones(n, dtype=torch.bfloat16, device="npu")
    for op in OPS:
        rec = {"op": op, "bytes_per_rank": nbytes}
        try:
            if op == "all_reduce":
                fn = lambda: dist.all_reduce(x)
            elif op == "all_gather":
                outs = [torch.empty(n, dtype=torch.bfloat16, device="npu") for _ in range(world)]
                fn = lambda: dist.all_gather(outs, x)
            elif op == "reduce_scatter":
                ins = [torch.ones(n, dtype=torch.bfloat16, device="npu") for _ in range(world)]
                o = torch.empty(n // world, dtype=torch.bfloat16, device="npu")
                fn = lambda: dist.reduce_scatter(o, ins)
            elif op == "all_to_all_single":
                o = torch.empty(n, dtype=torch.bfloat16, device="npu")
                fn = lambda: dist.all_to_all_single(o, x)
            else:
                fn = lambda: dist.broadcast(x, src=0)
            med, lo, hi = timed(fn)
            rec.update({"median_ms": round(med, 6), "min_ms": round(lo, 6),
                        "algbw_GBps": round((nbytes / (med * 1e-3)) / 1e9, 3)})
            if op == "all_reduce":
                rec["busbw_GBps"] = round(rec["algbw_GBps"] * (world - 1) / world, 3)
        except Exception as e:
            rec["error"] = "%s: %s" % (type(e).__name__, e)
        out["results"].append(rec)
    del x
    torch.npu.empty_cache()

dist.destroy_process_group()
with open(os.path.join(E, "p1b_hccl_rank%d.json" % rank), "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
if rank == 0:
    print("comm_pool_at_init=%d  comm_pool_after_warmup=%d" %
          (out["comm_pool_at_init_bytes"], out["comm_pool_after_warmup_bytes"]))
    print("%-16s %12s %12s %10s %12s" % ("op", "bytes", "median_ms", "algbw_GB/s", "busbw_GB/s"))
    for r in out["results"]:
        if "median_ms" in r:
            print("%-16s %12d %12.4f %10.2f %12s" % (r["op"], r["bytes_per_rank"], r["median_ms"],
                  r["algbw_GBps"], r.get("busbw_GBps", "-")))
        else:
            print("%-16s %12d ERROR %s" % (r["op"], r["bytes_per_rank"], r["error"]))