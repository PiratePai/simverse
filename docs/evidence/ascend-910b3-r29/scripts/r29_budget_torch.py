"""R29-A1b：torch_npu allocator 分阶段快照。

与 ACL 脚本分开运行、分开解读：两者是不同的分配器与负载，
同一阶段名不代表可比数值（旧报告 P1「预算阶段误读」的整改）。
"""

from __future__ import annotations

import json
import sys

import torch
import torch_npu  # noqa: F401 - 注册 NPU 后端

from r29_common import base_record, device_snapshot, write_json

GiB = 2**30


def meminfo(dev: int) -> dict:
    free, total = torch.npu.mem_get_info(dev)
    stats = torch.npu.memory_stats(dev) or {}
    return {
        "free_bytes": int(free),
        "total_bytes": int(total),
        "used_bytes": int(total) - int(free),
        "allocated_bytes": int(torch.npu.memory_allocated(dev)),
        "reserved_bytes": int(torch.npu.memory_reserved(dev)),
        "active_bytes": int(stats.get("active_bytes.all.current", -1)),
    }


def main() -> int:
    rec = base_record("r29_budget_torch")
    rec["api"] = "torch_npu（memory_allocated / memory_reserved / mem_get_info）"
    rec["timing_boundary"] = "allocator statistics, single process, sequential stages"
    rec["devices"] = device_snapshot(torch, torch_npu)
    stages = []

    for stage_name in ("startup", "after_first_tensor", "after_1GiB_held", "after_release", "after_empty_cache"):
        stages.append({"stage": stage_name})

    held = []
    for dev in range(torch.npu.device_count()):
        torch.npu.set_device(dev)
        entry: dict = {"logical_dev": dev, "snapshots": []}
        entry["snapshots"].append({"stage": "startup", **meminfo(dev)})

        t = torch.ones(1 << 20, dtype=torch.bfloat16, device=f"npu:{dev}")
        torch.npu.synchronize()
        entry["snapshots"].append({"stage": "after_first_tensor", **meminfo(dev)})

        block_bytes = 1 * GiB
        n = block_bytes // 2  # bf16
        blk = torch.ones(n, dtype=torch.bfloat16, device=f"npu:{dev}")
        torch.npu.synchronize()
        entry["snapshots"].append(
            {"stage": "after_1GiB_held", "block_bytes": block_bytes, **meminfo(dev)}
        )

        del blk
        torch.npu.synchronize()
        entry["snapshots"].append({"stage": "after_release", **meminfo(dev)})

        torch.npu.empty_cache()
        entry["snapshots"].append({"stage": "after_empty_cache", **meminfo(dev)})

        entry["note"] = (
            "after_release 只表示该进程的 allocator 统计；缓存区是否归还设备由 empty_cache 阶段单独观测。"
        )
        del t
        rec.setdefault("devices_detail", []).append(entry)

    rec["stage_names"] = stages
    rec["comparison_boundary"] = (
        "torch allocator numbers are not comparable with ACL pyacl snapshots; "
        "no cross-API subtraction is performed here."
    )
    write_json("budget_torch.json", rec)
    print(json.dumps({"devices": len(rec.get("devices_detail", []))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
