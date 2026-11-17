"""R29-A1：ACL 运行期显存预算（修正返回码与 context 生命周期）。

相对旧 p0a_budget_acl.py 的修正（见 docs/validation/ascend-910b3-20261117.md）：
- 每一次 ACL 调用都显式检查返回码，失败即记录并停止该设备；
- create_context 的句柄保存并在 destroy_context(ctx) 中显式回传；
- 每个阶段带 stage 标签，禁止把 after_init 与 after_release 混用；
- 不把 ACL 快照与 torch allocator 快照横向比较（那是另一个脚本、另一种负载）。
"""

from __future__ import annotations

import json
import sys

import acl

from r29_common import base_record, write_json

GiB = 2**30
STAGES = (1, 4, 8)


def rc(res) -> int:
    """pyacl 的多数调用返回整数返回码。"""
    return int(res) if isinstance(res, int) else 0


def main() -> int:
    rec = base_record("r29_budget_acl")
    rec["api"] = "pyacl (acl.rt.*)"
    rec["timing_boundary"] = "runtime-api-query, single process, sequential stages"
    rec["stages"] = []
    failures: list[dict] = []

    def fail(step: str, detail) -> None:
        failures.append({"step": step, "detail": detail})

    ret = rc(acl.init())
    rec["calls"] = {"acl.init": ret}
    if ret != 0:
        fail("acl.init", ret)
        rec["failures"] = failures
        write_json("budget_acl.json", rec)
        return 1

    try:
        soc = acl.get_soc_name()
        rec["soc_name"] = soc
    except Exception as exc:  # noqa: BLE001
        rec["soc_name"] = None
        fail("get_soc_name", f"{type(exc).__name__}: {exc}")

    for dev in (0, 1):
        entry: dict = {"logical_dev": dev, "calls": {}, "steps": []}
        try:
            ret = rc(acl.rt.set_device(dev))
            entry["calls"]["set_device"] = ret
            if ret != 0:
                fail(f"set_device({dev})", ret)
                rec["stages"].append(entry)
                continue

            ctx, ret = acl.rt.create_context(dev)
            entry["calls"]["create_context"] = ret
            if ret != 0:
                fail(f"create_context({dev})", ret)
                rec["stages"].append(entry)
                continue
            entry["context_handle"] = int(ctx) if ctx is not None else None

            free, total, ret = acl.rt.get_mem_info(0)
            entry["calls"]["get_mem_info@init"] = ret
            if ret != 0:
                fail(f"get_mem_info({dev})", ret)
            entry["after_init"] = {
                "stage": "after_init",
                "free_bytes": int(free),
                "total_bytes": int(total),
                "used_bytes": int(total) - int(free),
            }

            held = []
            for gib in STAGES:
                nbytes = gib * GiB
                ptr, ret = acl.rt.malloc(nbytes, 0)
                step = {
                    "stage": f"after_malloc_{gib}GiB_held",
                    "requested_bytes": nbytes,
                    "malloc_ret": ret,
                    "ptr": int(ptr) if ptr is not None else None,
                }
                if ret != 0 or ptr is None:
                    step["ok"] = False
                    fail(f"malloc({dev},{gib}GiB)", ret)
                    entry["steps"].append(step)
                    break
                step["ok"] = True
                held.append(ptr)
                free, total, ret2 = acl.rt.get_mem_info(0)
                step["get_mem_info_ret"] = ret2
                step["free_bytes"] = int(free)
                step["delta_from_init_bytes"] = entry["after_init"]["free_bytes"] - int(free)
                entry["steps"].append(step)

            entry["held_blocks"] = len(held)
            entry["held_bytes"] = sum(s["requested_bytes"] for s in entry["steps"] if s.get("ok"))

            for idx, ptr in enumerate(held):
                ret = rc(acl.rt.free(ptr))
                entry["calls"][f"free_{idx}"] = ret
                if ret != 0:
                    fail(f"free({dev},{idx})", ret)

            free, total, ret = acl.rt.get_mem_info(0)
            entry["calls"]["get_mem_info@release"] = ret
            entry["after_release"] = {
                "stage": "after_release_all_blocks",
                "free_bytes": int(free),
                "total_bytes": int(total),
                "delta_from_init_bytes": entry["after_init"]["free_bytes"] - int(free),
            }

            # pyacl 不同版本签名不同：优先显式回传 context 句柄，签名不兼容时退回无参形式并记录实际用法
            try:
                ret = rc(acl.rt.destroy_context(ctx))
                entry["calls"]["destroy_context"] = ret
                entry["destroy_context_form"] = "with_context_handle"
            except TypeError:
                ret = rc(acl.rt.destroy_context())
                entry["calls"]["destroy_context"] = ret
                entry["destroy_context_form"] = "no_argument_fallback"
            if ret != 0:
                fail(f"destroy_context({dev})", ret)
        except Exception as exc:  # noqa: BLE001
            entry["exception"] = f"{type(exc).__name__}: {exc}"
            fail(f"device_loop({dev})", entry["exception"])
        finally:
            try:
                ret = rc(acl.rt.reset_device(dev))
                entry["calls"]["reset_device"] = ret
                if ret != 0:
                    fail(f"reset_device({dev})", ret)
            except Exception as exc:  # noqa: BLE001
                fail(f"reset_device({dev})", f"{type(exc).__name__}: {exc}")
        rec["stages"].append(entry)

    try:
        ret = rc(acl.finalize())
        rec["calls"]["acl.finalize"] = ret
        if ret != 0:
            fail("acl.finalize", ret)
    except Exception as exc:  # noqa: BLE001
        fail("acl.finalize", f"{type(exc).__name__}: {exc}")

    rec["failures"] = failures
    rec["all_calls_ok"] = not failures
    rec["interpretation_boundary"] = (
        "These are free/used snapshots of the current device at each stage. "
        "They are not capacity guarantees, do not prove full-capacity allocation, "
        "and must not be compared to torch allocator snapshots from another process."
    )
    write_json("budget_acl.json", rec)
    print(json.dumps({"failures": failures, "devices": len(rec["stages"])}, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
