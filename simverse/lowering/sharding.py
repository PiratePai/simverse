"""TP/EP 降层（lowering v0.2）：合法组合校验、分片/复制、集合通信。

只登记 v0.2 串行闭环需要的最小集：
  * TP：GEMM 列/行切分计数（count 已按 per_card 口径接收 tp）、每层 2 次
    allreduce（attn 输出 + mlp 输出；MoE 层含 dispatch/combine 拓扑差异另记）；
  * KV：heads%tp 整除时每 rank Hkv/tp；tp>Hkv 时复制（显式登记，不静默除）；
  * EP：专家按 E/tp 分片（v0.2 登记容量分片；dispatch/combine 字节按
    路由表模型给出期望，routing trace 未接入前按均匀假设并标注）。
MLA/V4.1 的 KV 复制策略沿用 kv_spec 声明（kv_replicated_across_tp）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .. import formula_registry as FR
from ..model_ir.schema import ModelIR

import math


@dataclass(frozen=True)
class LinkLike:
    bw: float                 # B/s 单向
    fixed_ms: float           # 每消息启动
    label: str = "intra-node"


@dataclass
class CollectiveCost:
    kind: str                 # allreduce | allgather | reducescatter | dispatch/combine
    bytes_on_wire: int        # 每 rank 实发字节（ring: 2(N-1)/N·msg）
    t_ms: float
    formula: str
    note: str = ""

    def to_json(self) -> dict:
        return dict(self.__dict__)


def validate_tp(model: ModelIR, tp: int) -> List[str]:
    """TP 合法性：返回阻断原因列表（空=合法）。不盲测非法组合。"""
    errors = []
    for layer in model.layers:
        if layer.attn is not None:
            if layer.attn.heads % tp:
                errors.append(
                    f"TP{tp} 不整除 Q 头数 {layer.attn.heads}（layer {layer.index}）")
            # KV 头：tp>Hkv → 复制（登记为 replicated，不报错）
        if layer.moe is not None and layer.moe.experts and layer.moe.experts % tp:
            errors.append(
                f"TP/EP{tp} 不整除专家数 {layer.moe.experts}（layer {layer.index}）")
    return errors


def kv_shard_note(model: ModelIR, tp: int) -> Dict[str, str]:
    """KV 分片策略登记：per_family。"""
    fam = model.family
    if fam in ("deepseek_v3", "v41_csa2"):
        return {"strategy": "replicated", "reason":
                "MLA kv_heads=1：压缩 KV 每卡全量（kv_spec 声明口径）"}
    per_layer = []
    for layer in model.layers:
        if layer.attn is not None:
            kvh = layer.attn.kv_heads
            if tp <= kvh and kvh % tp == 0:
                per_layer.append(f"Hkv/tp={kvh // tp}")
            else:
                per_layer.append(f"replicated(Hkv={kvh}<tp)")
    return {"strategy": "mixed", "detail": "; ".join(sorted(set(per_layer)))}


def ring_allreduce(msg_bytes: int, ranks: int, link: LinkLike) -> CollectiveCost:
    """ring allreduce：每 rank 发送 2(N−1)/N·msg；时间含 2(N−1) 段传输+启动。"""
    if ranks < 2:
        return CollectiveCost("allreduce", 0, 0.0, FR.F_COLLECTIVE,
                              note="单卡无集合通信")
    per_rank = int(2 * (ranks - 1) / ranks * msg_bytes)
    t_transfer = per_rank / link.bw * 1e3
    t_lat = 2 * (ranks - 1) * link.fixed_ms
    return CollectiveCost("allreduce", per_rank, t_transfer + t_lat,
                          FR.F_COLLECTIVE,
                          note=f"ring N={ranks} msg={msg_bytes}B")


def tp_layer_collectives(model: ModelIR, tp: int, hidden: int, act_bytes: float,
                         tokens: int, link: LinkLike) -> List[CollectiveCost]:
    """每层 2 次 allreduce（attn/FFN 输出，TP 行切分后的规约）。

    msg = tokens·h·act_bytes（每层两次；MoE 层同口径——dispatch/combine
    的路由字节在 traffic 层单独按期望模型登记，不在此重复计）。
    """
    out = []
    if tp < 2:
        return out
    msg = int(tokens * hidden * act_bytes)
    for name in ("attn_allreduce", "mlp_allreduce"):
        c = ring_allreduce(msg, tp, link)
        c.note = f"{name} layer-wide"
        out.append(c)
    return out


def ep_dispatch_combine(model: ModelIR, tp: int, tokens: int, hidden: int,
                        act_bytes: float, top_k: int, link: LinkLike
                        ) -> Tuple[CollectiveCost, CollectiveCost, dict]:
    """EP dispatch/combine 期望字节（均匀路由假设，登记 approximated）。

    all-to-all：每 token 的 hidden 副本发给其 top_k 个专家所在 rank 的
    期望不同 rank 数 u_r = tp·(1−(1−k̄/tp)^{k̄=top_k 分布})；近似按
    E[distinct ranks] = tp·(1−(1−top_k/tp)^{1})… 精确式按每专家均匀落位：
    P(rank 收到某 token) = 1−C(tp−⌈E/tp⌉…) 复杂；v0.2 采用
    E[bytes/token] = top_k·hidden·act_bytes（每专家一份；同 rank 专家去重
    期望因子 1−(1−1/tp)^{top_k} 近似），显式标注假设。
    """
    msg_per_token = int(hidden * act_bytes)
    # 期望每 token 覆盖的 rank 数（每专家独立均匀落位）
    exp_ranks = tp * (1.0 - (1.0 - 1.0 / tp) ** top_k)
    dispatch_bytes = int(tokens * exp_ranks * msg_per_token)
    combine_bytes = dispatch_bytes  # combine 回传同量（假设无稀疏压缩）
    d = CollectiveCost("dispatch", dispatch_bytes,
                       dispatch_bytes / link.bw * 1e3 + link.fixed_ms,
                       FR.F_COLLECTIVE,
                       note=f"E[ranks/token]={exp_ranks:.2f}（均匀落位假设）")
    c = CollectiveCost("combine", combine_bytes,
                       combine_bytes / link.bw * 1e3 + link.fixed_ms,
                       FR.F_COLLECTIVE, note="回传同量假设")
    assumption = {"routing": "uniform-per-expert",
                  "approximated": "无 trace 时按均匀假设；期望字节非精确计数"}
    return d, c, assumption
