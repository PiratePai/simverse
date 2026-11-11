"""TP/EP 降层（lowering v0.2）：合法组合校验、分片/复制、集合通信。

R01（2026-11-10）起并行配方统一由 lowering/parallel.ParallelPlan 判定
（vLLM @dc74b8cb pin；count/memory/lowering 三处消费同一 plan，不再互相
矛盾）。本模块保留：
  * validate_tp/kv_shard_note：plan 的薄封装（旧调用方兼容）；
  * ring allreduce / EP all2all 的字节与时间模型；
  * 按层通信配方：attention allreduce（TP>1 恒有）+ MLP 段
    （tp_allreduce 或 ep_all2all，见 parallel.plan_collective_recipe）。
MLA/V4.1 的 KV 复制策略沿用 kv_spec 声明（kv_replicated_across_tp）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .. import formula_registry as FR
from ..model_ir.schema import ModelIR

import math

from .parallel import (ParallelPlan, VLLM_PARALLEL_PIN, plan_collective_recipe,
                       validate_parallel)


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


def validate_tp(model: ModelIR, tp: int, ep: int = 1, draft=None) -> ParallelPlan:
    """并行合法性（旧入口保留）：返回 ParallelPlan，errors 非空=非法。"""
    return validate_parallel(model, tp, ep, draft)


def kv_shard_note(model: ModelIR, tp: int) -> Dict[str, str]:
    """KV 分片策略登记：per_family（plan 同规则，消息口径供 UI/manifest）。"""
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
    return {"strategy": "mixed" if len(set(per_layer)) > 1
            else ("sharded" if per_layer and per_layer[0].startswith("Hkv")
                  else "replicated"),
            "detail": "; ".join(sorted(set(per_layer)))}


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


def all2all_msg(per_rank_bytes: int, ranks: int, link: LinkLike,
                kind: str, note: str) -> CollectiveCost:
    """all2all（EP dispatch/combine）：每 rank 发送/接收 per_rank_bytes。

    时间按发送侧带宽饱和近似（N−1 个对端并发，链路聚合带宽计一次），
    启动按一次消息（vLLM all2all 实现按 batch 合并）；近似已登记。
    """
    if ranks < 2:
        return CollectiveCost(kind, 0, 0.0, FR.F_COLLECTIVE, note="单卡无 EP")
    t = per_rank_bytes / link.bw * 1e3 + link.fixed_ms
    return CollectiveCost(kind, per_rank_bytes, t, FR.F_COLLECTIVE, note=note)


def logits_gather(tokens_with_logits: int, vocab: int, tp: int,
                  act_bytes: float, link: LinkLike) -> CollectiveCost:
    """lm_head 词表行切后的 logits gather（TP>1 每次 lm_head 调用一次）。

    证据（vLLM @dc74b8cb logits_processor.py L90-91/122-133/185-198；
    source_map sm-vllm-lmhead-logits-gather）：lm_head.tp_size>1 时
    _get_logits 对 head 输出（head dtype=act 精度，fp32 转换在 gather 后的
    sampler）执行 all-gather（平台开关）或 gather-to-rank0。两条路径每 rank
    线上最大字节同为 (tp−1)/tp × T×V×act；延迟按单向环 (tp−1) 段近似
    （ring allreduce 的减半；已登记近似）。
    """
    if tp < 2:
        return CollectiveCost("logits_gather", 0, 0.0, FR.F_COLLECTIVE,
                              note="TP1 词表分片即全量，无 gather")
    total = int(tokens_with_logits * vocab * act_bytes)
    per_rank = int((tp - 1) / tp * total)
    t = per_rank / link.bw * 1e3 + (tp - 1) * link.fixed_ms
    return CollectiveCost(
        "logits_gather", per_rank, t, FR.F_COLLECTIVE,
        note=f"T={tokens_with_logits} V={vocab} 全量 {total}B；每 rank 线上"
             f"上界 (tp−1)/tp×总量（all-gather/gather-to-rank0 同界）")


def engram_gather(tokens: int, width: int, tp: int, bytes_per_element: float,
                  link: LinkLike, kind: str, note: str) -> CollectiveCost:
    """Engram 层 all-gather（R04；每 engram 层各一次，TP>1）。

    证据（engram.py pin @f358f057）：wkv 为 ColumnParallelLinear
    gather_output=True（L1728/L1328-1329 tensor_model_parallel_all_gather
    dim=1）→ [T,25600/tp] 分片聚合为全宽 [T,25600]；查表后按 head 分片
    的 bf16 行 [T,part,256] 同样 all-gather（L1831-1832）。每 rank 线上
    接收 = (tp−1)/tp × T×width×2B（pinned bf16；延迟按单向环
    (tp−1) 段近似，已登记）。表行读是 host UVA/peer 行为（cpu_offload
    研究假设），不进通信行。
    """
    if tp < 2:
        return CollectiveCost(kind, 0, 0.0, FR.F_COLLECTIVE,
                              note="TP1 无 Engram 通信")
    total = int(tokens * width * bytes_per_element)
    per_rank = int((tp - 1) / tp * total)
    t = per_rank / link.bw * 1e3 + (tp - 1) * link.fixed_ms
    return CollectiveCost(kind, per_rank, t, FR.F_COLLECTIVE, note=note)


def tp_layer_collectives(model: ModelIR, tp: int, hidden: int, act_bytes: float,
                         tokens: int, link: LinkLike
                         ) -> List[CollectiveCost]:
    """【旧接口，仅 TP/EP=1 路径】每层 2 次 allreduce。

    EP 开的模型请用 model_collectives（按层配方：EP MoE 层 mlp 段为
    dispatch/combine 而非 allreduce）。
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


def model_collectives(model: ModelIR, plan: ParallelPlan, hidden: int,
                      act_bytes: float, tokens: int, link: LinkLike,
                      moe_top_k: int = 0,
                      lm_head_tokens: Optional[int] = None
                      ) -> List[CollectiveCost]:
    """按层通信配方（R01）：每层 attention allreduce；MLP 段——

      * tp_allreduce（dense FFN / EP 关 MoE）：inter 分片部分和规约；
      * ep_all2all（EP 开 MoE）：dispatch+combine，期望字节按均匀路由
        E[ranks/token] = ep·(1−(1−1/ep)^top_k)（显式近似假设）；
      * logits_gather（层外，TP>1）：lm_head 词表行切后每次调用 gather
        logits（logits_processor.py pin；2026-11-10 二轮审计补漏——此前
        draft.py 注释声称"另计"但此处未实现）。lm_head_tokens：经 lm_head
        的行数（默认=tokens；草稿阶段为 B×(blk−1) 候选行）。
      * engram gathers（R04，TP>1，每 engram 层 2 次）：查表 head 分片
        all-gather + wkv gather_output=True all-gather（engram.py
        L1328-1329/L1831-1832；表行读为 host UVA 行为，不进通信行）。
    """
    out: List[CollectiveCost] = []
    if plan.tp < 2:
        return out
    msg = int(tokens * hidden * act_bytes)
    engram_spec = (model.extras or {}).get("engram") or {}
    e_ids = list(engram_spec.get("layer_ids", ()))
    for layer in model.layers:
        c = ring_allreduce(msg, plan.tp, link)
        c.note = f"attn_allreduce L{layer.index}"
        out.append(c)
        if layer.index in e_ids:
            cols = int(engram_spec["n_hash_cols"])
            d_v = int(engram_spec["head_dim"])
            wkv_out = hidden * (int(engram_spec["hc_mult"]) + 1)
            out.append(engram_gather(
                tokens, cols * d_v, plan.tp, 2, link,
                "engram_head_gather",
                note=f"L{layer.index} 查表 [T,{cols},{d_v}] bf16 head 分片"
                     " 聚合（L1831-1832）"))
            out.append(engram_gather(
                tokens, wkv_out, plan.tp, 2, link, "engram_wkv_gather",
                note=f"L{layer.index} wkv 列切 [T,{wkv_out // plan.tp}]→"
                     f"[T,{wkv_out}]（gather_output=True，L1728）"))
        recipe = plan_collective_recipe(plan, layer)
        if recipe == "tp_allreduce":
            c2 = ring_allreduce(msg, plan.tp, link)
            c2.note = f"mlp_allreduce L{layer.index}"
            out.append(c2)
        elif recipe == "ep_all2all":
            ep = plan.ep
            # 每 token 期望覆盖 E[ranks]=ep·(1−(1−1/ep)^k) 个 rank；线上
            # 总字节 = tokens·E[ranks]·msg，对称假设下每 rank 收/发 = 总量/ep
            exp_ranks = ep * (1.0 - (1.0 - 1.0 / ep) ** moe_top_k)
            per_rank_bytes = int(tokens * exp_ranks * hidden * act_bytes / ep)
            d = all2all_msg(per_rank_bytes, ep, link, "ep_dispatch",
                            note=f"L{layer.index} E[ranks/token]={exp_ranks:.2f}"
                                 f"（均匀路由，对称收发假设）")
            c3 = all2all_msg(per_rank_bytes, ep, link, "ep_combine",
                             note=f"L{layer.index} 回传同量假设")
            out.extend([d, c3])
    out.append(logits_gather(
        lm_head_tokens if lm_head_tokens is not None else tokens,
        model.vocab, plan.tp, act_bytes, link))
    return out


def ep_dispatch_combine(model: ModelIR, tp: int, tokens: int, hidden: int,
                        act_bytes: float, top_k: int, link: LinkLike
                        ) -> Tuple[CollectiveCost, CollectiveCost, dict]:
    """EP dispatch/combine 期望字节（均匀路由假设，登记 approximated）。

    all-to-all：每 token 的 hidden 副本发给其 top_k 个专家所在 rank；
    E[distinct ranks] = ep·(1−(1−1/ep)^top_k)（每专家独立均匀落位），
    显式标注假设；ep=tp（单引擎副本 EP 配方）。
    """
    ranks = tp
    msg_per_token = int(hidden * act_bytes)
    exp_ranks = ranks * (1.0 - (1.0 - 1.0 / ranks) ** top_k)
    per_rank_bytes = int(tokens * exp_ranks * msg_per_token / ranks)
    d = all2all_msg(per_rank_bytes, ranks, link, "dispatch",
                    note=f"E[ranks/token]={exp_ranks:.2f}（均匀落位假设）")
    c = all2all_msg(per_rank_bytes, ranks, link, "combine",
                    note="回传同量假设")
    assumption = {"routing": "uniform-per-expert",
                  "approximated": "无 trace 时按均匀假设；期望字节非精确计数"}
    return d, c, assumption
