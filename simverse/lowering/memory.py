"""容量账本（memory v0.2）：唯一同时存活对象 × 可分配预算。

规则（§7.4 / principles §6）：
  * 按 allocation identity 去重（共享权重一份），各 consumer 的读流量另计；
  * 峰值 = Σ_unique_live_allocations（不是各对象独立峰值之和）；
  * 预算分层：nominal / allocatable / allocated / reserved——预留不双扣；
  * 仅标称容量时：超过可证上限才 OOM；低于标称不给 feasible（unknown）。
v0.2 串行闭环登记静态 + 每请求对象（KV/状态随 ctx/batch 增长）；
生命周期交错峰值（verify 槽位等）在 M3 事件核接入后补 timeline。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..model_ir.schema import ModelIR


@dataclass
class Allocation:
    alloc_id: str
    kind: str                    # weights | kv | state | activation | workspace | comm
    role: str                    # target | draft | shared
    bytes: int
    shared_identity: Optional[str] = None   # 跨 rank/角色共享的去重身份
    live_note: str = "run-lifetime"

    def to_json(self) -> dict:
        return dict(self.__dict__)


@dataclass
class DeviceBudget:
    device_id: str
    nominal_bytes: int
    allocatable_bytes: Optional[int] = None   # 缺真实可分配值 → None（unknown）
    source: str = "spec"
    budget_note: str = ""

    @property
    def effective_budget(self) -> Optional[int]:
        return self.allocatable_bytes if self.allocatable_bytes is not None \
            else self.nominal_bytes

    def to_json(self) -> dict:
        return dict(self.__dict__)


@dataclass
class RankMemory:
    rank_id: str
    device_id: str
    budget: DeviceBudget
    allocations: List[Allocation] = field(default_factory=list)

    def add(self, alloc: Allocation) -> None:
        self.allocations.append(alloc)

    def live_total(self) -> int:
        """唯一存活对象求和（shared_identity 去重；同 run 生命周期）。"""
        seen: Dict[str, Allocation] = {}
        total = 0
        for a in self.allocations:
            key = a.shared_identity or f"{self.rank_id}:{a.alloc_id}"
            if key in seen:
                continue
            seen[key] = a
            total += a.bytes
        return total

    def peak_vs_budget(self) -> Dict[str, object]:
        total = self.live_total()
        eff = self.budget.effective_budget
        if eff is None:
            status = "unknown"
        elif total > eff:
            status = "oom"
        elif self.budget.allocatable_bytes is None:
            status = "unknown"    # 低于标称不能证明 feasible（§7.4）
        else:
            status = "feasible"
        return {"rank_id": self.rank_id, "live_bytes": total,
                "nominal_bytes": self.budget.nominal_bytes,
                "allocatable_bytes": self.budget.allocatable_bytes,
                "status": status,
                "headroom_bytes": (eff - total) if eff is not None else None}


def target_weight_allocations(model: ModelIR, quant: dict, tp: int,
                              role: str = "target") -> List[Allocation]:
    """主模型每卡权重驻留（per-rank；按 tp 切分后口径）。"""
    b_dense = quant["dense_w"] / 8
    b_expert = quant["expert_w"] / 8
    allocs: List[Allocation] = []
    fam = model.family
    for layer in model.layers:
        li = layer.index
        dense_b = 0
        if layer.attn is not None:
            a = layer.attn
            dense_b += (a.heads * a.head_dim * (2 if a.output_gated else 1) // tp
                        + 2 * a.kv_out) * b_dense + (a.heads * a.head_dim // tp) * b_dense
        if layer.linear is not None:
            g = layer.linear
            dense_b += (g.qkv_out + g.value_dim + 2 * g.num_v_heads) * model.hidden * b_dense
            dense_b += g.value_dim * model.hidden * b_dense   # out_proj
            dense_b += 4 * g.num_v_heads * b_dense            # dt/A 参数
        if layer.mla is not None and fam == "deepseek_v3":
            m = layer.mla
            heads_c = m.heads // tp
            dense_b += (model.hidden * m.q_lora_rank
                        + m.q_lora_rank * heads_c * (m.qk_nope_dim + m.qk_rope_dim)
                        + model.hidden * (m.kv_lora_rank + m.qk_rope_dim)
                        + m.kv_lora_rank * heads_c * (m.qk_nope_dim + m.v_head_dim)
                        + heads_c * m.v_head_dim * model.hidden) * b_dense
        if layer.mla is not None and fam == "v41_csa2" and layer.csa2:
            # V4.1 注意力投影（2026-11-10 补齐；几何与 count/work.v41_layer_rows
            # 同口径——TP 表达式一致（heads/ng 切分，其余每卡），逐投影引擎分片
            # 证据未逐项 pin，登记为与计数行自洽的口径）
            kvs = model.extras["kv_spec"]
            heads = kvs["heads"] // tp
            d, di = kvs["head_dim"], kvs["index_head_dim"]
            ih, qrank, orank = kvs["index_heads"], kvs["q_lora_rank"], kvs["o_lora_rank"]
            ng = kvs["o_groups"] // tp
            a = layer.csa2
            mlt = 2 if a.ratio == 2 else 1
            dense_b += model.hidden * qrank                       # q_a_proj
            dense_b += model.hidden * d                           # kv_proj
            dense_b += qrank * heads * d                          # q_b_proj
            dense_b += (kvs["heads"] * d // kvs["o_groups"]) * orank   # o_a（组输入全宽）
            dense_b += ng * orank * model.hidden                  # o_b（组切分）
            if a.produces_kv:
                dense_b += model.hidden * d * mlt                 # compress_proj
                dense_b += d * di                                 # index_k_proj
            if a.produces_index:
                dense_b += model.hidden * ih + qrank * ih * di    # index 投影
        hc_spec = (model.extras or {}).get("hc") if fam == "v41_csa2" else None
        if hc_spec and layer.kind == "mla_csa2":
            # mHC 权重（FP32，每卡复制不随 TP 缩放：全宽流 + torch.empty 全宽参数，
            # model.py pin @f358f057）：2×(fn mix×hc_dim + base mix + scale 3)
            mix = (2 + hc_spec["mult"]) * hc_spec["mult"]
            hc_dim = hc_spec["mult"] * model.hidden
            hc_b = 2 * (mix * hc_dim * 4 + mix * 4 + 3 * 4)
            allocs.append(Allocation(
                f"{role}-w-mhc-L{li}", "weights", role, int(hc_b),
                live_note="常驻 fp32 每卡复制（hc_attn_fn/hc_ffn_fn）"))
        if layer.dense_inter:
            dense_b += 3 * model.hidden * (layer.dense_inter // tp) * b_dense
        if dense_b:
            allocs.append(Allocation(
                f"{role}-w-dense-L{li}", "weights", role, int(dense_b),
                live_note="常驻"))
        if layer.moe is not None and layer.moe.experts:
            moe = layer.moe
            router_b = int(model.hidden * moe.experts * b_dense)
            shared_b = int(3 * model.hidden * (moe.shared_intermediate // tp)
                           * b_dense) if moe.shared_intermediate else 0
            gate_b = int(model.hidden * b_dense) if moe.shared_gate else 0
            experts_b = int(layer.moe.experts // tp * 3 * model.hidden
                            * moe.intermediate * b_expert)
            allocs.append(Allocation(
                f"{role}-w-moe-L{li}", "weights", role,
                router_b + shared_b + gate_b + experts_b, live_note="常驻"))
    if not model.tie_embeddings:
        allocs.append(Allocation(
            f"{role}-w-embed", "weights", role,
            int(model.hidden * model.vocab * b_dense), live_note="常驻"))
        allocs.append(Allocation(
            f"{role}-w-lm_head", "weights", role,
            int(model.hidden * (model.vocab // tp) * b_dense), live_note="常驻"))
    else:
        allocs.append(Allocation(
            f"{role}-w-embed-tied", "weights", role,
            int(model.hidden * (model.vocab // tp) * b_dense), live_note="常驻"))
    return allocs


def kv_allocations(model: ModelIR, quant: dict, batch: int, ctx: int,
                   role: str = "target") -> List[Allocation]:
    """KV/常数状态驻留（每请求；按 batch×ctx 计）。"""
    b_kv = quant["kv"] / 8
    b_state = quant.get("state", 32) / 8
    b_act = quant["act"] / 8
    fam = model.family
    allocs: List[Allocation] = []
    if fam in ("deepseek_v3", "v41_csa2"):
        spec = model.extras["kv_spec"]
        per_token_elems = 0
        if fam == "deepseek_v3":
            per_token_elems = spec["kv_elems_per_token_per_layer"] * len(model.layers)
            allocs.append(Allocation(f"{role}-kv", "kv", role,
                                     int(per_token_elems * ctx * b_kv) * batch,
                                     live_note="随 ctx 增长"))
        else:
            d, di = spec["head_dim"], spec["index_head_dim"]
            swa, r_set = spec["sliding_window"], spec["compress_ratios"]
            total = 0
            for i, layer in enumerate(model.layers):
                r = r_set[i]
                total += min(ctx, swa) * d
                csa = layer.csa2
                if csa.produces_kv:
                    total += (ctx // r) * d + (ctx // r) * di
                    if r > 1:
                        total += 2 * d * 4
            allocs.append(Allocation(f"{role}-kv", "kv", role, int(total * b_kv) * batch,
                                     live_note="SWA 封顶+压缩潜层（Engram/hc 状态缺口）"))
    else:
        elems = sum(2 * l.attn.kv_heads * l.attn.head_dim
                    for l in model.layers if l.attn is not None)
        allocs.append(Allocation(f"{role}-kv", "kv", role,
                                 int(elems * ctx * b_kv) * batch,
                                 live_note="随 ctx 增长"))
    state_elems = model.per_request_state_elems()
    if state_elems:
        allocs.append(Allocation(
            f"{role}-gdn-state", "state", role,
            int(state_elems * (quant.get("state", 32) / 8)) * batch,
            live_note="每请求常数（recurrent fp32 + conv）"))
    return allocs


def colocated_rank_memory(model: ModelIR, draft_ir, quant, draft_quant, tp: int,
                          batch: int, ctx: int, budget: DeviceBudget,
                          draft_batch: int = 1) -> RankMemory:
    """同卡混部：target+draft 权重与 KV/状态同时驻留（对象身份去重）。"""
    rank = RankMemory("colocated-r0", budget.device_id, budget)
    for a in target_weight_allocations(model, quant, tp, role="target"):
        rank.add(a)
    for a in kv_allocations(model, quant, batch, ctx, role="target"):
        rank.add(a)
    if draft_ir is not None:
        # DFlash 草稿：结构权重 + 自身 KV；共享目标 embed/lm_head（去重身份）
        draft_q = dict(quant)
        draft_q.update(draft_quant or {})
        for a in target_weight_allocations(_as_model(draft_ir), draft_q, 1,
                                           role="draft"):
            if "embed" in a.alloc_id or "lm_head" in a.alloc_id:
                a.shared_identity = f"shared:{a.alloc_id}"  # 与目标共享（去重）
            rank.add(a)
        for a in kv_allocations(_as_model(draft_ir), draft_q, draft_batch, ctx,
                                role="draft"):
            rank.add(a)
    return rank


def _as_model(draft_ir) -> ModelIR:
    """把 DraftIR 包装成最小 ModelIR 以复用容量函数。"""
    return ModelIR(
        id=draft_ir.id, label=draft_ir.label, family="gqa", hidden=draft_ir.hidden,
        vocab=draft_ir.vocab, layers=list(draft_ir.layers),
        source=draft_ir.source, tie_embeddings=False)
