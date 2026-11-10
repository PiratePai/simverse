"""统一 TP/EP 并行分片配方（R01，2026-11-10）。

证据（vLLM @dc74b8cbbbe3dbb9399fe369757d63defd614cc2，main 2026-11-10
网络获取入册 assets/identity/vllm_qwen35_pin/，逐文件 SHA256 见
simverse/validation/registry.py 的 ev-vllm-parallel-recipe）：

  * MoE（fused_moe/config.py FusedMoEParallelConfig.make L1157-1288）：
      - EP 关（ep=1）：routed 专家按 **TP 切 intermediate**（config.py L1402
        `intermediate_size_per_partition = intermediate_size // tp_size`），
        每 rank 持全部 E 个专家的 inter/tp 分片；要求 inter % tp == 0。
      - EP 开（tp>1 且 enable_expert_parallel）：MoE 内部 tp→1、ep→world——
        每 rank 持 E//ep 个**整专家**（expert_map_manager.determine_expert_map
        L62-77：余数分给低 rank，base+1），无 TP 切分。
      - 单引擎副本内 ep ∈ {1} 或 ep == tp；tp×ep 独立组合（如 tp8/ep2）无
        实现配方，显式拒绝。
  * router gate（fused_moe/router/gate_linear.py L66 `GateLinear(
    ReplicatedLinear)`）：每 rank 复制全量 h×E。
  * shared expert（qwen3_next.py Qwen3NextSparseMoeBlock L219 Qwen3NextMLP）：
    dense MLP 配方 inter/tp（需 shared_inter % tp==0，否则 L127 raise）；
    shared_expert_gate = ReplicatedLinear(h,1)（L205）。
  * full attention（qwen3_next.py Qwen3NextAttention L302-315）：
    heads/tp；kvh ≥ tp → kvh/tp 切分（要求 kvh%tp==0），kvh < tp → 整组
    复制（要求 tp%kvh==0）；o_proj 行切。
  * GDN 线性注意力（mamba/gdn/qwen_gdn_linear_attn.py L753-838）：
    in_proj_qkvz 列切（(2·key_dim+2·value_dim)/tp）、in_proj_ba 列切
    （2·num_v_heads/tp）、conv1d conv_dim/tp、out_proj 行切；
    状态（mamba/mamba_utils.py gated_delta_net_state_shape L280-301）：
    temporal = num_v_heads/tp×hk×hv，conv = conv_dim/tp×(k−1)。
  * embedding/lm_head：VocabParallelEmbedding/ParallelLMHead → 词表行切
    V/tp（qwen3_next.py L62-63 导入，Qwen3_5MoeForCausalLM 沿用）。
  * DFlash 草稿（v1/worker/gpu/spec_decode/dflash/utils.py load_dflash_model
    L16-70 + eagle/utils.py maybe_share_target_embed/_should_share）：
    草稿与 target **同 TP 组**运行（DFlashQwen3Attention 用全局
    get_tensor_model_parallel_world_size()，qwen3_dflash.py L191-204），
    embed_tokens/lm_head 直接替换为 target 的**同一模块对象**（词表行切），
    不复制权重；fc 为 ReplicatedLinear（全宽复制）。

本模块只做"配方判定与每 rank 几何"，不做计数（count/）与容量（memory/）；
三处消费同一 ParallelPlan，防止互相矛盾（G-C11 的根因）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ..model_ir.schema import AttentionIR, DraftIR, LayerIR, LinearAttnIR, ModelIR

VLLM_PARALLEL_PIN = "vllm@dc74b8cbbbe3dbb9399fe369757d63defd614cc2"
RECIPE_ID = "vllm-fusedmoe-tp-ep-v1"


@dataclass(frozen=True)
class MoEShardRule:
    """一层 routed 专家的 per-rank 配方。"""
    ep_enabled: bool            # True=EP 整专家；False=TP 切 inter
    experts_total: int
    local_experts_base: int     # E // ep
    remainder: int              # E % ep：前 remainder 个 rank 各 +1
    inter_per_rank: int         # 每专家 intermediate（EP=inter；TP=inter/tp）

    def local_experts(self, rank: int) -> int:
        if not self.ep_enabled:
            return self.experts_total     # TP：每 rank 全部专家（inter 分片）
        return self.local_experts_base + (1 if rank < self.remainder else 0)


@dataclass(frozen=True)
class AttnShardRule:
    """GQA/SLIDING 层 per-rank 注意力几何。"""
    heads_per_rank: int
    kv_heads_per_rank: int
    kv_replicated: bool         # kvh < tp → 每 rank 全量 KV 头

    @property
    def kv_note(self) -> str:
        return ("replicated" if self.kv_replicated
                else f"kv_heads/tp={self.kv_heads_per_rank}")


@dataclass(frozen=True)
class GDNShardRule:
    """GDN 线性注意力层 per-rank 几何。"""
    num_v_heads: int
    num_k_heads: int
    head_k_dim: int
    head_v_dim: int
    conv_kernel: int
    key_dim: int                # per rank
    value_dim: int              # per rank
    conv_dim: int               # per rank

    @property
    def qkvz_out_per_rank(self) -> int:
        """in_proj_qkv + in_proj_z 融合投影输出宽（vLLM in_proj_qkvz）。"""
        return 2 * self.key_dim + 2 * self.value_dim

    @property
    def ba_out_per_rank(self) -> int:
        return 2 * self.num_v_heads

    @property
    def state_elems_per_rank(self) -> int:
        """recurrent state 元素数（per rank，per request 常数）。"""
        return self.num_v_heads * self.head_k_dim * self.head_v_dim

    @property
    def conv_state_elems_per_rank(self) -> int:
        return self.conv_dim * (self.conv_kernel - 1)


def _gdn_rule(g: LinearAttnIR, tp: int) -> GDNShardRule:
    if g.num_v_heads % tp or g.num_k_heads % tp:
        raise ValueError(
            f"GDN 头数不整除 TP{tp}（Nk={g.num_k_heads}, Nv={g.num_v_heads}）")
    if g.qkv_out % tp:
        raise ValueError(f"GDN conv_dim {g.qkv_out} 不整除 TP{tp}")
    return GDNShardRule(
        num_v_heads=g.num_v_heads // tp, num_k_heads=g.num_k_heads // tp,
        head_k_dim=g.head_k_dim, head_v_dim=g.head_v_dim,
        conv_kernel=g.conv_kernel,
        key_dim=g.head_k_dim * (g.num_k_heads // tp),
        value_dim=g.head_v_dim * (g.num_v_heads // tp),
        conv_dim=g.qkv_out // tp)


def _attn_rule(a: AttentionIR, tp: int) -> AttnShardRule:
    if a.heads % tp:
        raise ValueError(f"TP{tp} 不整除 Q 头数 {a.heads}")
    if a.kv_heads >= tp:
        if a.kv_heads % tp:
            raise ValueError(f"KV 头 {a.kv_heads} 不整除 TP{tp}")
        return AttnShardRule(a.heads // tp, a.kv_heads // tp, False)
    if tp % a.kv_heads:
        raise ValueError(f"TP{tp} 不整除 KV 头组 {a.kv_heads}（复制约束）")
    return AttnShardRule(a.heads // tp, a.kv_heads, True)


@dataclass(frozen=True)
class ParallelPlan:
    """一次 run 的并行分片判定（count/memory/lowering 共用）。

    字段口径：全为 **per-rank** 值（rank=0 或均匀；EP 余数差异经
    moe_local_experts(rank) 查询）。errors 非空 = 组合非法（不静默降级）。
    """
    tp: int
    ep: int
    model_id: str
    errors: Tuple[str, ...] = ()
    moe_ep_enabled: bool = False
    moe_layer_map: Tuple[Tuple[int, MoEShardRule], ...] = ()  # layer index → 规则
    vocab_per_rank: int = 0
    implementation_id: str = RECIPE_ID
    source_pin: str = VLLM_PARALLEL_PIN
    notes: Tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.errors

    def attn_rule(self, layer: LayerIR) -> Optional[AttnShardRule]:
        if layer.attn is None:
            return None
        return _attn_rule(layer.attn, self.tp)

    def gdn_rule(self, layer: LayerIR) -> Optional[GDNShardRule]:
        if layer.linear is None:
            return None
        return _gdn_rule(layer.linear, self.tp)

    def moe_rule(self, layer: LayerIR) -> Optional[MoEShardRule]:
        if layer.moe is None or not layer.moe.experts:
            return None
        for idx, r in self.moe_layer_map:
            if idx == layer.index:
                return r
        return None

    def moe_local_experts(self, layer: LayerIR, rank: int) -> int:
        r = self.moe_rule(layer)
        return r.local_experts(rank) if r else 0

    def to_json(self) -> dict:
        return {
            "recipe_id": self.implementation_id, "source_pin": self.source_pin,
            "tp": self.tp, "ep": self.ep, "moe_ep_enabled": self.moe_ep_enabled,
            "vocab_per_rank": self.vocab_per_rank,
            "valid": self.valid, "errors": list(self.errors),
            "notes": list(self.notes),
            "moe_layer_map": [
                {"layer": li, "ep_enabled": r.ep_enabled,
                 "experts_total": r.experts_total,
                 "local_base": r.local_experts_base, "remainder": r.remainder,
                 "inter_per_rank": r.inter_per_rank}
                for li, r in self.moe_layer_map],
        }


def validate_parallel(model: ModelIR, tp: int, ep: int = 1,
                      draft: Optional[DraftIR] = None) -> ParallelPlan:
    """判定 (tp, ep) 组合对 model（及可选 colocated draft）是否合法。

    返回 ParallelPlan；非法时不抛错（plan.errors 携带原因），由调用方
    显式拒绝（pipeline → capability invalid + unsupported_combo）。
    """
    errors: List[str] = []
    notes: List[str] = []
    if tp < 1 or ep < 1:
        return ParallelPlan(tp, ep, model.id, ("tp/ep 必须 ≥ 1",))
    if ep != 1 and ep != tp:
        errors.append(
            f"ep={ep} 无实现配方：单引擎副本内 EP 只能取 1（TP 切 inter）或 "
            f"等于 tp（整专家 EP）；tp×ep 独立组合需 DP 展平，未见 pin 源码")
    if tp == 1 and ep != 1:
        errors.append("tp=1 时 EP 无意义（单卡已持全部专家）")

    # 注意力（FULL/SLIDING；MLA 家族按 kv_spec 声明，不在此判定）
    for layer in model.layers:
        if layer.attn is not None:
            try:
                _attn_rule(layer.attn, tp)
            except ValueError as e:
                errors.append(f"{e}（layer {layer.index}）")
        if layer.linear is not None:
            try:
                _gdn_rule(layer.linear, tp)
            except ValueError as e:
                errors.append(f"{e}（layer {layer.index}）")

    # MoE
    moe_ep = ep != 1 and not errors
    moe_rules: List[Tuple[int, MoEShardRule]] = []
    for layer in model.layers:
        if layer.moe is None or not layer.moe.experts:
            continue
        moe = layer.moe
        if moe_ep:
            if tp > moe.experts:
                errors.append(
                    f"TP{tp} > 专家数 {moe.experts}（vLLM qwen3_next L184 拒绝）")
            base, rem = divmod(moe.experts, ep)
            moe_rules.append((layer.index, MoEShardRule(
                True, moe.experts, base, rem, moe.intermediate)))
        else:
            if moe.intermediate % tp:
                errors.append(
                    f"EP 关时 routed inter {moe.intermediate} 不整除 TP{tp}"
                    "（vLLM inter/tp 切分要求整除）")
            if moe.shared_intermediate and moe.shared_intermediate % tp:
                errors.append(
                    f"shared inter {moe.shared_intermediate} 不整除 TP{tp}"
                    "（qwen3_next L127 raise）")
            moe_rules.append((layer.index, MoEShardRule(
                False, moe.experts, moe.experts, 0, moe.intermediate // tp)))
    # dense FFN（mlp_only 层）同样 inter/tp
    for layer in model.layers:
        if layer.dense_inter and layer.moe is None and layer.dense_inter % tp:
            errors.append(f"dense FFN inter {layer.dense_inter} 不整除 TP{tp}"
                          f"（layer {layer.index}）")

    # 词表行切（embedding/lm_head 共用）
    vocab_per_rank = model.vocab
    if model.vocab % tp:
        errors.append(f"vocab {model.vocab} 不整除 TP{tp}（词表行切）")
    else:
        vocab_per_rank = model.vocab // tp

    # colocated draft：与 target 同 TP 组（vLLM DFlash 配方；TP1 草稿挂
    # TP>1 组无实现证据 → 显式拒绝）
    if draft is not None:
        notes.append(
            "draft 与 target 同 TP 组（vLLM load_dflash_model：embed/lm_head "
            "共享 target 模块对象，词表行切；fc ReplicatedLinear 复制）")
        for layer in draft.layers:
            if layer.attn is not None:
                try:
                    _attn_rule(layer.attn, tp)
                except ValueError as e:
                    errors.append(f"草稿 {e}（layer {layer.index}）")
            if layer.dense_inter and layer.dense_inter % tp:
                errors.append(
                    f"草稿 FFN inter {layer.dense_inter} 不整除 TP{tp}"
                    f"（layer {layer.index}）")
        if draft.vocab != model.vocab:
            # 词表不同时 vLLM 走 draft_id_to_target_id 映射（草稿自有全量
            # lm_head + 映射表），与本仓共享配方的假设不符 → 拒绝
            errors.append(
                f"草稿词表 {draft.vocab} ≠ target {model.vocab}：需 "
                "draft_id_to_target_id 映射路径，未建模（不支持）")
    return ParallelPlan(
        tp=tp, ep=ep, model_id=model.id, errors=tuple(errors),
        moe_ep_enabled=moe_ep and not errors,
        moe_layer_map=tuple(moe_rules), vocab_per_rank=vocab_per_rank,
        notes=tuple(notes))


def plan_collective_recipe(plan: ParallelPlan, layer: LayerIR) -> str:
    """一层 MLP 段的通信配方：tp_allreduce（TP 切分求和）或 ep_all2all。

    attention 段恒为 attn allreduce（TP>1 时）。EP 开的 MoE 层：dispatch/
    combine all2all 替代 mlp allreduce（vLLM EP moe 语义：combine 后每
    rank 已得全 token 求和，无需再 allreduce）。
    """
    if plan.tp < 2:
        return "none"
    if layer.moe is not None and layer.moe.experts and plan.moe_ep_enabled:
        return "ep_all2all"
    return "tp_allreduce"
