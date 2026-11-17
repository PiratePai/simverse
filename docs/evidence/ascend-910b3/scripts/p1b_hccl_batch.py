import json, os, statistics, time
import torch, torch_npu
import torch.distributed as dist

rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
local = int(os.environ.get("LOCAL_RANK", rank))
torch.npu.set_device(local)
E = "/data/p00929506/TQ/npu-evidence/20261117/raw"
out = {"kind": "r17_hccl_batch_timing", "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
       "rank": rank, "world_size": world, "device_name": torch.npu.get_device_name(local),
       "ascend_rt_visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES", ""),
       "transport": "HCCS (verified via HCCL plog: transporttype[HCCS])",
       "timing_boundary": "device-sync wallclock, batched: N collectives per timing window, per-op = window/N",
       "batches": 5, "results": []}
free_before, _ = torch.npu.mem_get_info()
dist.init_process_group(backend="hccl")
w = torch.ones(1024, dtype=torch.bfloat16, device="npu")
dist.all_reduce(w); torch.npu.synchronize()
free_after, _ = torch.npu.mem_get_info()
out["comm_pool_bytes"] = free_before - free_after

SZ = [1 << 10, 1 << 12, 1 << 14, 1 << 16, 1 << 18, 1 << 20, 1 << 22, 1 << 24, 1 << 26, 1 << 28]
OPS = ["all_reduce", "all_gather", "reduce_scatter", "all_to_all_single", "broadcast"]

def per_op(fn, iters):
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    per = []
    for _ in range(5):
        t = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.npu.synchronize()
        per.append((time.perf_counter() - t) * 1e3 / iters)
    return statistics.median(per), min(per)

for nbytes in SZ:
    n = nbytes // 2
    iters = 100 if nbytes <= (1 << 24) else 20
    x = torch.ones(n, dtype=torch.bfloat16, device="npu")
    for op in OPS:
        rec = {"op": op, "bytes_per_rank": nbytes, "iters_per_batch": iters}
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
            med, lo = per_op(fn, iters)
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
with open(os.path.join(E, "p1b_hccl_batch_rank%d.json" % rank), "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
if rank == 0:
    print("comm_pool_bytes=%d" % out["comm_pool_bytes"])
    print("%-16s %10s %12s %12s %10s" % ("op", "bytes", "median_ms", "algbw_GB/s", "busbw_GB/s"))
    for r in out["results"]:
        if "median_ms" in r:
            print("%-16s %10d %12.5f %12.2f %10s" % (r["op"], r["bytes_per_rank"], r["median_ms"],
                  r["algbw_GBps"], r.get("busbw_GBps", "-")))
        else:
            print("%-16s %10d ERR %s" % (r["op"], r["bytes_per_rank"], r["error"]))