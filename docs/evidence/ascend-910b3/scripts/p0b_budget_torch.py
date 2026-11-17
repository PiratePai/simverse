import json, os, time
import torch, torch_npu
GiB = 2**30
VIS = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
res = {"kind": "gd7_budget_torch", "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
       "ascend_rt_visible_devices": VIS, "torch": torch.__version__,
       "torch_npu": torch_npu.__version__, "timing_boundary": "runtime-api", "devices": []}
for dev in range(torch.npu.device_count()):
    torch.npu.set_device(dev)
    torch.npu.empty_cache()
    free0, total = torch.npu.mem_get_info()
    entry = {"logical_dev": dev, "name": torch.npu.get_device_name(dev),
             "total_bytes": total, "free_after_init": free0,
             "reserved_after_init": total - free0, "steps": []}
    for gib in (1, 4, 8):
        try:
            t = torch.empty(gib * GiB // 2, dtype=torch.bfloat16, device="npu")
            free, _ = torch.npu.mem_get_info()
            entry["steps"].append({"requested_gib": gib, "ok": True, "free_bytes": free,
                                   "allocated_bytes": torch.npu.memory_allocated(),
                                   "reserved_bytes": torch.npu.memory_reserved()})
            del t
            torch.npu.empty_cache()
        except Exception as e:
            entry["steps"].append({"requested_gib": gib, "ok": False, "error": str(e)})
            break
    free1, _ = torch.npu.mem_get_info()
    entry["free_after_release"] = free1
    res["devices"].append(entry)
print(json.dumps(res, indent=2, ensure_ascii=False))