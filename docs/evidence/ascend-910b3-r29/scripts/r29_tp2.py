"""R29-A5：A-N3 少卡正确性与容量（TP2 / EP2 切片，卡 5/6）。

对应计划 §2「2 卡」行：TP1 对 TP2 可重组数值、EP2 dispatch/combine token 数、
逆置换与空专家、每 rank 已知峰值。

做法：不加载整模，而用**保留真实维度**的算子切片做"分片—重组"对照：
- MLP 列并行（gate/up 按输出维切）与行并行（down 按输入维切），
  结果 all_gather / all_reduce 后与 fp64 CPU 稠密参考比较；
- GQA 注意力按 q 头切分（16 头 → 每 rank 8 头，kv 头 2 → 每 rank 1 头），
  拼接后与 fp64 稠密参考比较，并核对 kv 分片守恒；
- MoE EP2：token 按 rank 切分，按专家归属做 all_to_all 往返，
  校验 token 计数、逆置换恒等与空专家（本轮刻意留 2 个不参与路由的专家）；
- 每 rank 记录分阶段显存与峰值。

所有数值容差按 dtype 记录；不通过的不写"通过"。这是切片级证据，
不是 Qwen3.6 整模 forward，也不代表 TP>2 或跨机。
"""

from __future__ import annotations

import json
import os
import sys

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

from r29_common import base_record, write_json

DTYPE_MAP = {"float32": torch.float32, "bfloat16": torch.bfloat16}
TOL = {"float32": 1e-5, "bfloat16": 2e-2}  # 相对 ref 尺度的绝对容差，fixture 级判据

HIDDEN = 2048
INTER = 512
SEQ = 32
HEADS = 16
KV_HEADS = 2
HEAD_DIM = 256


def snap(dev: int, stage: str) -> dict:
    free, total = torch.npu.mem_get_info(dev)
    return {
        "stage": stage,
        "free_bytes": int(free),
        "total_bytes": int(total),
        "allocated_bytes": int(torch.npu.memory_allocated(dev)),
        "reserved_bytes": int(torch.npu.memory_reserved(dev)),
        "peak_allocated_bytes": int(torch.npu.max_memory_allocated(dev)),
    }


def cmp(name: str, got: torch.Tensor, ref: torch.Tensor, tol: float) -> dict:
    got64 = got.double()
    err = (got64 - ref).abs()
    denom = ref.abs().clamp_min(1e-9)
    scale = float(ref.abs().max()) or 1.0
    max_abs = float(err.max())
    return {
        "case": name,
        "max_abs_err": max_abs,
        "max_rel_err": float((err / denom).max()),
        "ref_scale": scale,
        "tol_abs": tol * max(scale, 1.0),
        "passed": bool(max_abs <= tol * max(scale, 1.0)),
    }


def main() -> int:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.npu.set_device(local)
    dev = local

    rec = base_record("r29_tp2_slice")
    rec.update(
        {
            "rank": rank,
            "world_size": world,
            "device_name": torch.npu.get_device_name(dev),
            "shapes": {
                "hidden": HIDDEN,
                "intermediate": INTER,
                "seq": SEQ,
                "heads": HEADS,
                "kv_heads": KV_HEADS,
                "head_dim": HEAD_DIM,
            },
            "tolerances": {
                k: {"abs_tol_scaled_by_ref_max": v, "note": "fixture level criterion, not an acceptance threshold"}
                for k, v in TOL.items()
            },
            "cases": [],
            "errors": [],
            "mem": [],
        }
    )
    rec["mem"].append(snap(dev, "startup"))

    dist.init_process_group(backend="hccl")
    torch.npu.set_device(dev)
    torch.manual_seed(20261117)  # 两侧同一权重：CPU 生成，各自搬上设备

    x = torch.randn(SEQ, HIDDEN) / 8.0

    # ---------- 1) MLP 列并行 ----------
    w_up = torch.randn(INTER, HIDDEN) / (HIDDEN ** 0.5)
    per = INTER // world
    rows = slice(rank * per, (rank + 1) * per)
    ref_up = x.double() @ w_up.double().T
    for dtype_name, dtype in DTYPE_MAP.items():
        w_local = w_up[rows].to(dtype).npu()
        x_dev = x.to(dtype).npu()
        y_local = x_dev @ w_local.T
        gathered = [torch.empty(SEQ, per, dtype=dtype, device=f"npu:{dev}") for _ in range(world)]
        dist.all_gather(gathered, y_local)
        torch.npu.synchronize()
        full = torch.cat(gathered, dim=-1).float().cpu()
        rec["cases"].append(cmp(f"mlp_column_parallel[{dtype_name}]", full, ref_up, TOL[dtype_name]))

    # ---------- 2) MLP 行并行 ----------
    # W_down 形状 (HIDDEN, INTER)，按输入维（INTER）切列，各 rank 只算一部分 k
    w_down = torch.randn(HIDDEN, INTER) / (INTER ** 0.5)
    h_in = torch.randn(SEQ, INTER) / 8.0
    ref_down = h_in.double() @ w_down.double().T
    for dtype_name, dtype in DTYPE_MAP.items():
        cols = slice(rank * per, (rank + 1) * per)
        w_local = w_down[:, cols].to(dtype).npu()
        h_local = h_in[:, cols].to(dtype).npu()
        y_partial = h_local @ w_local.T
        dist.all_reduce(y_partial)
        torch.npu.synchronize()
        rec["cases"].append(
            cmp(f"mlp_row_parallel[{dtype_name}]", y_partial.float().cpu(), ref_down, TOL[dtype_name])
        )

    rec["mem"].append(snap(dev, "after_mlp_tp2"))

    # ---------- 3) GQA 注意力头分片 ----------
    q_full = torch.randn(SEQ, HEADS, HEAD_DIM) / 8.0
    k_full = torch.randn(SEQ, KV_HEADS, HEAD_DIM) / 8.0
    v_full = torch.randn(SEQ, KV_HEADS, HEAD_DIM) / 8.0
    group = HEADS // KV_HEADS

    def dense_attention(q, k, v):
        outs = []
        for h in range(HEADS):
            kv = h // group
            scores = (q[:, h].double() @ k[:, kv].double().T) / (HEAD_DIM ** 0.5)
            probs = torch.softmax(scores, dim=-1)
            outs.append(probs @ v[:, kv].double())
        return torch.stack(outs, dim=1)

    ref_attn = dense_attention(q_full, k_full, v_full)
    heads_per_rank = HEADS // world
    kv_per_rank = KV_HEADS // world
    my_q = slice(rank * heads_per_rank, (rank + 1) * heads_per_rank)
    my_kv = slice(rank * kv_per_rank, (rank + 1) * kv_per_rank)

    for dtype_name, dtype in DTYPE_MAP.items():
        q_local = q_full[:, my_q].to(dtype).npu()
        k_local = k_full[:, my_kv].to(dtype).npu()
        v_local = v_full[:, my_kv].to(dtype).npu()
        scores = torch.einsum("thd,shd->hts", q_local.float(), k_local.float()) / (HEAD_DIM ** 0.5)
        probs = torch.softmax(scores, dim=-1).to(dtype)
        out_local = torch.einsum("hts,shd->thd", probs, v_local).float()
        buf = [
            torch.empty(SEQ, heads_per_rank, HEAD_DIM, dtype=torch.float32, device=f"npu:{dev}")
            for _ in range(world)
        ]
        dist.all_gather(buf, out_local)
        torch.npu.synchronize()
        merged = torch.cat(buf, dim=1).cpu()
        rec["cases"].append(cmp(f"gqa_attention_head_shard[{dtype_name}]", merged, ref_attn, TOL[dtype_name]))

    kv_counts = torch.tensor([kv_per_rank], dtype=torch.int64, device=f"npu:{dev}")
    gathered_counts = [torch.zeros(1, dtype=torch.int64, device=f"npu:{dev}") for _ in range(world)]
    dist.all_gather(gathered_counts, kv_counts)
    total_kv = int(sum(int(t.item()) for t in gathered_counts))
    rec["kv_shard_conservation"] = {
        "per_rank_kv_heads": kv_per_rank,
        "sum_over_ranks": total_kv,
        "expected": KV_HEADS,
        "passed": total_kv == KV_HEADS,
    }
    rec["mem"].append(snap(dev, "after_attention_tp2"))

    # ---------- 4) MoE EP2 dispatch / combine ----------
    T_TOK = 32
    E = 8
    TOPK = 2
    H_ROUTE = 64
    H_EXP = 32
    xr = torch.randn(T_TOK, H_ROUTE) / 8.0
    w1 = torch.randn(E, H_EXP, H_ROUTE) / (H_ROUTE ** 0.5)  # up: H_ROUTE -> H_EXP
    w2 = torch.randn(E, H_ROUTE, H_EXP) / (H_EXP ** 0.5)    # down: H_EXP -> H_ROUTE

    # 路由只落到 0..5，专家 6/7 故意留空（空专家分支）；归属：0-2 → rank0，3-5 → rank1
    route = torch.empty(T_TOK, TOPK, dtype=torch.int64)
    for t in range(T_TOK):
        route[t, 0] = (t % 2) * 3 + ((t // 8) % 3)
        route[t, 1] = ((t + 1) % 2) * 3 + (((t + 1) // 8) % 3)
    weights = torch.rand(T_TOK, TOPK).double()
    weights = weights / weights.sum(dim=-1, keepdim=True)

    def owner_of(expert: int) -> int:
        return 0 if expert < 3 else 1

    local_tokens = [t for t in range(T_TOK) if t % world == rank]
    local_entries = [(t, s, int(route[t, s])) for t in local_tokens for s in range(TOPK)]
    ordered = sorted(local_entries, key=lambda item: owner_of(item[2]))

    def counts_for(entries):
        counts = [0] * world
        for _, _, e in entries:
            counts[owner_of(e)] += 1
        return counts

    send_counts = counts_for(ordered)
    recv_counts = [0] * world
    for src in range(world):
        src_entries = [
            (t, s, int(route[t, s])) for t in range(T_TOK) if t % world == src for s in range(TOPK)
        ]
        recv_counts[src] = counts_for(src_entries)[rank]

    payload = torch.tensor([[e, t, s] for (t, s, e) in ordered], dtype=torch.float32).npu().contiguous()
    recv = torch.empty(sum(recv_counts), 3, dtype=torch.float32, device=f"npu:{dev}")
    dist.all_to_all_single(
        recv, payload, output_split_sizes=recv_counts, input_split_sizes=send_counts
    )
    torch.npu.synchronize()
    recv_cpu = recv.cpu().long()

    out_buf = torch.zeros(recv_cpu.shape[0], H_ROUTE, dtype=torch.float32)
    for idx in range(E):
        if owner_of(idx) != rank:
            continue
        mask = recv_cpu[:, 0] == idx
        if not bool(mask.any()):
            continue
        toks = recv_cpu[mask][:, 1]
        h = xr[toks].double()
        out_buf[mask] = (torch.nn.functional.gelu(h @ w1[idx].double().T) @ w2[idx].double().T).float()

    back = torch.empty(sum(send_counts), H_ROUTE, dtype=torch.float32, device=f"npu:{dev}")
    dist.all_to_all_single(
        back, out_buf.npu(), output_split_sizes=send_counts, input_split_sizes=recv_counts
    )
    torch.npu.synchronize()
    back_cpu = back.cpu()

    combined = torch.zeros(len(local_tokens), H_ROUTE, dtype=torch.float64)
    for row, t in enumerate(local_tokens):
        acc = torch.zeros(H_ROUTE, dtype=torch.float64)
        for s in range(TOPK):
            i = ordered.index((t, s, int(route[t, s])))
            acc += weights[t, s] * back_cpu[i].double()
        combined[row] = acc

    ref = torch.zeros(len(local_tokens), H_ROUTE, dtype=torch.float64)
    for row, t in enumerate(local_tokens):
        acc = torch.zeros(H_ROUTE, dtype=torch.float64)
        for s in range(TOPK):
            e = int(route[t, s])
            acc += weights[t, s] * (
                torch.nn.functional.gelu(xr[t].double() @ w1[e].double().T) @ w2[e].double().T
            )
        ref[row] = acc

    expert_counts = [
        sum(1 for t in range(T_TOK) for s in range(TOPK) if int(route[t, s]) == idx) for idx in range(E)
    ]
    expected_recv = sorted(
        [
            (int(route[t, s]), t, s)
            for src in range(world)
            for t in range(T_TOK)
            if t % world == src
            for s in range(TOPK)
            if owner_of(int(route[t, s])) == rank
        ]
    )
    got_recv = sorted([(int(r[0]), int(r[1]), int(r[2])) for r in recv_cpu])

    rec["moe_ep2"] = {
        "config": {
            "tokens": T_TOK,
            "experts": E,
            "routed_experts": 6,
            "top_k": TOPK,
            "hidden": H_ROUTE,
            "expert_hidden": H_EXP,
        },
        "expert_counts": expert_counts,
        "zero_count_experts": [idx for idx, c in enumerate(expert_counts) if c == 0],
        "entries_per_rank_sent": sum(send_counts),
        "entries_per_rank_received": sum(recv_counts),
        "send_counts_by_dest": send_counts,
        "recv_counts_by_src": recv_counts,
        "total_entries": T_TOK * TOPK,
        "roundtrip_permutation_identity": got_recv == expected_recv,
        "token_coverage": {
            "local_tokens": len(local_tokens),
            "combined_rows": int(combined.shape[0]),
            "covered": int(combined.shape[0]) == len(local_tokens),
        },
        "balance_boundary": (
            "路由为确定性构造（不含真实倾斜分布）；空专家 6/7 覆盖分片分支，"
            "但极端倾斜/丢 token 仍需单独注入测试"
        ),
    }
    rec["cases"].append(cmp("moe_ep2_combine", combined.float(), ref, TOL["float32"]))
    rec["mem"].append(snap(dev, "after_moe_ep2"))

    failures = [c for c in rec["cases"] if not c["passed"]]
    rec["summary"] = {
        "cases": len(rec["cases"]),
        "passed": len(rec["cases"]) - len(failures),
        "failed": len(failures),
        "failed_cases": [c["case"] for c in failures],
    }
    rec["boundary"] = (
        "算子切片级证据：随机权重下的分片/重组与路由一致性；"
        "不是 Qwen3.6 整模 forward，不覆盖跨层 cache、通信重叠、allocator 峰值，也不外推 TP>2。"
    )

    try:
        dist.destroy_process_group()
    except Exception as exc:  # noqa: BLE001
        rec["errors"].append({"destroy_process_group": f"{type(exc).__name__}: {exc}"})

    path = write_json(f"tp2_slice_rank{rank}.json", rec)
    print(json.dumps({"rank": rank, "wrote": str(path), "summary": rec["summary"]}, ensure_ascii=False))
    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())
