import json, os, time, acl
GiB = 2**30
VIS = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
res = {"kind": "gd7_budget_acl", "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
       "ascend_rt_visible_devices": VIS, "timing_boundary": "runtime-api", "devices": []}
acl.init()
soc = acl.get_soc_name()
for dev in (0, 1):
    try:
        acl.rt.set_device(dev)
    except Exception:
        break
    acl.rt.create_context(dev)
    free0, total, r = acl.rt.get_mem_info(0)
    entry = {"logical_dev": dev, "soc": soc, "total_bytes": total, "free_after_init": free0,
             "reserved_bytes": total - free0, "alloc_steps": []}
    held = []
    for gib in (1, 4, 8):
        n = gib * GiB
        try:
            p, rr = acl.rt.malloc(n, 0)
            f, t, _ = acl.rt.get_mem_info(0)
            entry["alloc_steps"].append({"requested_gib": gib, "ok": True, "free_bytes": f,
                                         "delta_bytes": free0 - f})
            held.append(p)
        except Exception as e:
            entry["alloc_steps"].append({"requested_gib": gib, "ok": False, "error": str(e)})
            break
    for p in held:
        acl.rt.free(p)
    f2, t2, _ = acl.rt.get_mem_info(0)
    entry["free_after_release"] = f2
    try:
        acl.rt.destroy_context()
    except Exception:
        pass
    acl.rt.reset_device(dev)
    res["devices"].append(entry)
try:
    acl.finalize()
except Exception:
    pass
print(json.dumps(res, indent=2, ensure_ascii=False))