"""R29-A2：双卡集合通信重采（修正 shape 契约、数值校验与 busbw 定义）。

相对旧 p1b_hccl_batch.py 的修正：
- reduce_scatter 使用 list 契约：每个输入块与输出同形状；不再用 n 与 n/P 混配；
- 每次计时前重置输入，避免连续 in-place all_reduce 的累加溢出；
- 每个 (op,size) 先做一次与 CPU/NPU 参考结果的逐元素校验，再计时；
- busbw 采用 nccl-tests 显式归一化系数并注明来源（all_reduce 2(P-1)/P，
  all_gather/reduce_scatter/all_to_all (P-1)/P，broadcast 1）；
- 显式记录每 rank 的 send/recv 字节、collective 类型与最慢 rank（两 rank 各自落盘）；
- 不声称 HCCS 路由：plog/HCCL 传输类型未采集时 transport 保持 unknown。

启动：torchrun --standalone --nproc_per_node=2 r29_collectives.py
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

from r29_common import base_record, write_json

SIZES_BYTES = [
    1 << 10,
    1 << 12,
    1 << 14,
    1 << 16,
    1 << 18,
    1 << 20,
    1 << 22,
    1 << 24,
    1 << 26,
    1 << 28,
]
DTYPE = torch.bfloat16
ITEMSIZE = 2

def iters_for(nbytes: int) -> int:
    if nbytes <= (1 << 20):
        return 100
    if nbytes <= (1 << 24):
        return 30
    return 10


def busbw_factor(op: str, world: int) -> float:
    p = float(world)
    if op == "all_reduce":
        return 2.0 * (p - 1.0) / p
    if op in ("all_gather", "reduce_scatter", "all_to_all_single"):
        return (p - 1.0) / p
    if op == "broadcast":
        return 1.0
    raise KeyError(op)


def timed(fn, iters: int, world: int) -> dict:
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    batches = []
    for _ in range(5):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.npu.synchronize()
        batches.append((time.perf_counter() - t0) * 1e3 / iters)
    batches_sorted = sorted(batches)
    return {
        "iters_per_batch": iters,
        "batches_ms": [round(b, 6) for b in batches],
        "median_ms": round(statistics.median(batches), 6),
        "min_ms": round(min(batches), 6),
        "p90_ms": round(batches_sorted[min(len(batches_sorted) - 1, int(0.9 * len(batches_sorted)))], 6),
        "max_ms": round(max(batches), 6),
    }


def main() -> int:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.npu.set_device(local)

    rec = base_record("r29_collectives")
    rec.update(
        {
            "rank": rank,
            "world_size": world,
            "local_rank": local,
            "device_name": torch.npu.get_device_name(local),
            "dtype": str(DTYPE).replace("torch.", ""),
            "timing_boundary": (
                "batched device-sync wallclock: N collectives per window, per-op = window/N; "
                "includes host enqueue and sync amortisation, NOT pure device kernel time"
            ),
            "transport": "unknown (no HCCL plog / transporttype captured in this run)",
            "busbw_definition": (
                "algbw x factor; factors from nccl-tests conventions: all_reduce 2(P-1)/P, "
                "all_gather/reduce_scatter/all_to_all (P-1)/P, broadcast 1"
            ),
            "results": [],
            "errors": [],
        }
    )

    free_before = torch.npu.mem_get_info(local)[0]

    dist.init_process_group(backend="hccl")
    rec["init_method"] = os.environ.get("MASTER_ADDR", "") + ":" + os.environ.get("MASTER_PORT", "")

    warm = torch.ones(1024, dtype=DTYPE, device=f"npu:{local}")
    dist.all_reduce(warm)
    torch.npu.synchronize()
    free_after_first = torch.npu.mem_get_info(local)[0]
    rec["comm_init_free_delta_bytes"] = int(free_before - free_after_first)
    rec["comm_init_boundary"] = (
        "free delta from before init_process_group to after the first all_reduce; "
        "includes HCCL communicator pools and the warm-up tensor, NOT pure comm pool"
    )
    del warm

    try:
        for nbytes in SIZES_BYTES:
            n = nbytes // ITEMSIZE
            for op in ("all_reduce", "all_gather", "reduce_scatter", "broadcast", "all_to_all_single"):
                entry: dict = {
                    "op": op,
                    "bytes_per_rank_recv": nbytes,
                    "payload_bytes_per_rank": nbytes,
                    "numel_per_rank": n,
                    "dtype": str(DTYPE).replace("torch.", ""),
                }
                try:
                    if op == "all_reduce":
                        # 先用非零输入验证语义，再用全零缓冲计时：避免连续原位累加改变数值域
                        chk = torch.full((n,), float(rank + 1), dtype=DTYPE, device=f"npu:{local}")
                        dist.all_reduce(chk)
                        torch.npu.synchronize()
                        expected = float(world * (world + 1) / 2)
                        ok = bool(torch.all(chk.float() == expected).item())
                        del chk
                        x = torch.zeros((n,), dtype=DTYPE, device=f"npu:{local}")
                        entry.update(
                            {
                                "expected_value": expected,
                                "correctness": "exact" if ok else "mismatch",
                                "timing_input": "zeros (no in-place accumulation drift)",
                            }
                        )
                        entry["send_bytes_per_rank"] = nbytes
                        entry["recv_bytes_per_rank"] = nbytes

                        def fn() -> None:
                            dist.all_reduce(x)

                    elif op == "all_gather":
                        x = torch.full((n,), float(rank + 1), dtype=DTYPE, device=f"npu:{local}")
                        outs = [torch.empty(n, dtype=DTYPE, device=f"npu:{local}") for _ in range(world)]
                        dist.all_gather(outs, x)
                        torch.npu.synchronize()
                        ok = all(
                            bool(torch.all(outs[i].float() == float(i + 1)).item()) for i in range(world)
                        )
                        entry.update({"correctness": "exact" if ok else "mismatch"})
                        entry["send_bytes_per_rank"] = nbytes
                        entry["recv_bytes_per_rank"] = nbytes * world

                        def fn() -> None:
                            dist.all_gather(outs, x)

                    elif op == "reduce_scatter":
                        # list 契约：每个输入块的形状必须与输出一致
                        outs_shape = (n,)
                        inputs = [
                            torch.full(outs_shape, float(rank + 1), dtype=DTYPE, device=f"npu:{local}")
                            for _ in range(world)
                        ]
                        out = torch.empty(outs_shape, dtype=DTYPE, device=f"npu:{local}")
                        dist.reduce_scatter(out, inputs)
                        torch.npu.synchronize()
                        expected = float(world * (world + 1) / 2)
                        ok = bool(torch.all(out.float() == expected).item())
                        entry.update(
                            {
                                "input_blocks": world,
                                "input_block_numel": n,
                                "output_numel": n,
                                "contract": "each input block shape == output shape",
                                "expected_value": expected,
                                "correctness": "exact" if ok else "mismatch",
                            }
                        )
                        entry["send_bytes_per_rank"] = nbytes * world
                        entry["recv_bytes_per_rank"] = nbytes

                        def fn() -> None:
                            dist.reduce_scatter(out, inputs)

                    elif op == "broadcast":
                        x = torch.full((n,), 7.0 if rank == 0 else -1.0, dtype=DTYPE, device=f"npu:{local}")
                        dist.broadcast(x, src=0)
                        torch.npu.synchronize()
                        ok = bool(torch.all(x.float() == 7.0).item())
                        entry.update({"correctness": "exact" if ok else "mismatch"})
                        entry["send_bytes_per_rank"] = nbytes if rank == 0 else 0
                        entry["recv_bytes_per_rank"] = 0 if rank == 0 else nbytes

                        def fn() -> None:
                            dist.broadcast(x, src=0)

                    else:  # all_to_all_single
                        x = torch.empty((n,), dtype=DTYPE, device=f"npu:{local}")
                        chunk = n // world
                        for k in range(world):
                            x[k * chunk : (k + 1) * chunk] = float((k + 1) * 10 + (rank + 1))
                        out = torch.empty_like(x)
                        dist.all_to_all_single(out, x)
                        torch.npu.synchronize()
                        ok = True
                        for k in range(world):
                            expect = float((rank + 1) * 10 + (k + 1))
                            seg = out[k * chunk : (k + 1) * chunk].float()
                            ok = ok and bool(torch.all(seg == expect).item())
                        entry.update({"correctness": "exact" if ok else "mismatch", "chunk_numel": chunk})
                        entry["send_bytes_per_rank"] = nbytes
                        entry["recv_bytes_per_rank"] = nbytes

                        def fn() -> None:
                            dist.all_to_all_single(out, x)

                    entry.update(timed(fn, iters_for(nbytes), world))
                    if op == "reduce_scatter":
                        entry["payload_bytes_per_rank"] = nbytes * world
                    entry["algbw_GBps"] = round(
                        (entry["payload_bytes_per_rank"] / (entry["median_ms"] * 1e-3)) / 1e9, 3
                    )
                    factor = busbw_factor(op, world)
                    entry["busbw_factor"] = round(factor, 6)
                    entry["busbw_GBps"] = round(entry["algbw_GBps"] * factor, 3)
                    entry["algbw_definition"] = (
                        "payload_bytes_per_rank / median_ms; payload = per-rank input buffer "
                        "(reduce_scatter uses total input = block_bytes x world_size)"
                    )
                except Exception as exc:  # noqa: BLE001
                    entry["error"] = f"{type(exc).__name__}: {exc}"
                    rec["errors"].append({"op": op, "bytes": nbytes, "error": entry["error"]})
                rec["results"].append(entry)
                torch.npu.empty_cache()
    finally:
        try:
            dist.destroy_process_group()
        except Exception as exc:  # noqa: BLE001
            rec["errors"].append({"destroy_process_group": f"{type(exc).__name__}: {exc}"})

    rec["correctness_summary"] = {
        "checked": sum(1 for r in rec["results"] if r.get("correctness")),
        "exact": sum(1 for r in rec["results"] if r.get("correctness") == "exact"),
        "mismatch": sum(1 for r in rec["results"] if r.get("correctness") == "mismatch"),
        "errors": len(rec["errors"]),
    }
    path = write_json(f"collectives_rank{rank}.json", rec)
    print(
        json.dumps(
            {"rank": rank, "wrote": str(path), "summary": rec["correctness_summary"]},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
