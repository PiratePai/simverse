"""FlashMLA SM90 sparse-decode split-KV 调度器（R25，2026-11-16）。

get_decoding_sched_meta.cu 单 warp 贪心分配的逐行 Python 移植（纯整数，
无浮点/无依赖）；输入为每行（decode 每行 = 一个 batch 条目，s_q=1）的
topk_length（SWA 窗宽）与 extra_topk_length（压缩 topk 有效数），输出
num_splits[b]（该行的 split 数）。

证据（csa2-core-pin.json；FlashMLA @847ef308）：
  * 每行块数：cur = max(topk_len,1) 向上取整到 block_size_topk(64)，再加
    extra_topk_len（不取整）；num_blocks = (cur-1)//64 + 1
    （get_decoding_sched_meta.cu L43-57）；
  * total = Σ (num_blocks + fixed_overhead_num_blocks)；SM90 h64 调度
    meta fixed_overhead=5、block=64（sparse_decode.cpp Decode_Sm90_Impl
    get_meta L167-174）；
  * payload = ceil(total/num_sm_parts) + fixed_overhead；num_sm_parts =
    max(sms/s_q/(h_q/64),1)（s_q=1、h_q=64 → sms）；
  * 逐 partition 贪心：payload 余量 ≥ 剩余块+overhead → 整行吞掉
    （splits = 已切分次数+1）；否则吞 remain-overhead 块计一次 split。
    num_splits[b] > 1 的行：主核写 o_accum/lse_accum（fp32），combine 核
    归并；= 1 的行走 no-split 直写 out（splitkv_mla.cuh L333-380、
    combine.cu L64-69）。

本移植只复现调度算法；o_accum 流量公式在 count/work.v41_layer_rows。
"""
from __future__ import annotations

from typing import List, Sequence

SM90_FIXED_OVERHEAD_BLOCKS = 5     # Decode_Sm90_Impl get_meta
SM90_BLOCK_SIZE_TOPK = 64          # 同上（block_size_topk）


def decode_num_blocks(topk_len: int, extra_topk_len: int,
                      block: int = SM90_BLOCK_SIZE_TOPK) -> int:
    """一行的 KV 块数（L43-57）。

    ku::ceil(cur, block) 把长度向上取整到 block 的**倍数**（128→128，
    65→128；返回同一单位），再原样加 extra；块数 = (cur−1)//block + 1。
    """
    cur = max(topk_len, 1)
    cur = ((cur + block - 1) // block) * block
    cur += extra_topk_len
    return (cur - 1) // block + 1


def flashmla_decode_num_splits(topk_lens: Sequence[int],
                               extra_topk_lens: Sequence[int],
                               num_sms: int,
                               fixed_overhead: int = SM90_FIXED_OVERHEAD_BLOCKS,
                               block: int = SM90_BLOCK_SIZE_TOPK,
                               ) -> List[int]:
    """逐行 num_splits（num_splits_ptr[1..b] 的差分）。

    num_sms：设备 SM 数（devices 注册表 sm_count）；h_q=64、s_q=1 时
    num_sm_parts = num_sms（sparse_decode.cpp L170）。
    """
    b = len(topk_lens)
    assert len(extra_topk_lens) == b
    if b == 0:
        return []
    if not num_sms or num_sms <= 0:
        raise ValueError(
            "flashmla_decode_num_splits 需要设备 SM 数（devices.sm_count）；"
            "未知不得默认（MERGE_CRITERIA §4.2）")
    blocks = [decode_num_blocks(t, e, block)
              for t, e in zip(topk_lens, extra_topk_lens)]
    total = sum(n + fixed_overhead for n in blocks)
    num_sm_parts = max(num_sms, 1)
    payload = -(-total // num_sm_parts) + fixed_overhead

    splits = [0] * b
    now_req, now_block, now_split = 0, 0, 0
    for _ in range(num_sm_parts):
        remain = payload
        while now_req < b:
            rem_blocks = blocks[now_req] - now_block
            if remain >= rem_blocks + fixed_overhead:
                splits[now_req] = now_split + 1
                remain -= rem_blocks + fixed_overhead
                now_req += 1
                now_block, now_split = 0, 0
            else:
                if remain - fixed_overhead > 0:
                    now_block += remain - fixed_overhead
                    now_split += 1
                    remain = 0
                break
        if now_req >= b:
            break
    if now_req < b or now_block != 0:
        raise AssertionError("调度器移植未能覆盖全部行（与贪心不变式不符）")
    return splits
