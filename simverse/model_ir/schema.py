"""ModelIR schema：共享模型结构子包的唯一结构表述（v0.3.0）。

职责边界（Q3/plan §3）：本包只描述模型结构与 config 证据——层序、注意力家族、
MoE 参数、跨层缓存关系、字段覆盖；不做计数（count/）、不做成本（cost/）、
不依赖仿真器/UI。消费方：simverse 各模块 + 两旧仓薄适配器（固定版本）。

结构完备性（structure_complete）：缺 unsupported-semantic 字段（如 V4.1 的
Engram/hc）时为 False，携带 gaps 列表；依赖完整结构的结论必须检查该标志。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

FULL = "full_attention"
LINEAR = "linear_attention"
SLIDING = "sliding_attention"
MLA = "mla"
MLA_CSA2 = "mla_csa2"


@dataclass(frozen=True)
class AttentionIR:
    """标准（GQA）注意力层的投影几何。

    output_gated: q_proj 输出含门控分支（qwen3_5/3.6 家族 q_proj 输出 2×heads·d，
    chunk 成 query 与 gate；证据 hf_transformers/modeling_qwen3_5_moe.py Qwen3_5MoeAttention）。
    qk_head_norm: q_norm/k_norm 作用在 head_dim 上（计入逐元素算量）。
    """
    heads: int
    kv_heads: int
    head_dim: int
    rotary_dim: int                    # 部分旋转（partial_rotary_factor·head_dim）
    output_gated: bool = False
    qk_head_norm: bool = True
    sliding_window: Optional[int] = None   # SLIDING 层的窗口（token 数）

    @property
    def q_out(self) -> int:
        return self.heads * self.head_dim * (2 if self.output_gated else 1)

    @property
    def kv_out(self) -> int:
        return self.kv_heads * self.head_dim


@dataclass(frozen=True)
class LinearAttnIR:
    """Gated DeltaNet 线性注意力层（qwen3_next / qwen3_5 家族）。

    投影（证据 modeling_qwen3_5_moe.py Qwen3_5MoeGatedDeltaNet）：
      in_proj_qkv: h → 2·key_dim + value_dim
      in_proj_z:   h → value_dim；in_proj_b/in_proj_a: h → num_v_heads
      conv1d: depthwise 因果卷积 kernel=conv_kernel over 2·key_dim+value_dim
      out_proj: value_dim → h；RMSNormGated 作用在 head_v_dim
    状态（每请求每层常数，不随 ctx 增长）：
      recurrent: num_v_heads·head_k_dim·head_v_dim
      conv: (2·key_dim+value_dim)·(conv_kernel−1)
    """
    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    conv_kernel: int

    @property
    def key_dim(self) -> int:
        return self.head_k_dim * self.num_k_heads

    @property
    def value_dim(self) -> int:
        return self.head_v_dim * self.num_v_heads

    @property
    def qkv_out(self) -> int:
        return 2 * self.key_dim + self.value_dim

    @property
    def state_elems(self) -> int:
        return self.num_v_heads * self.head_k_dim * self.head_v_dim

    @property
    def conv_state_elems(self) -> int:
        return self.qkv_out * (self.conv_kernel - 1)


@dataclass(frozen=True)
class MlaIR:
    """DeepSeek MLA 投影几何（V3 展开/吸收共用结构参数；V4.1 CSA2 另加字段）。"""
    heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_rope_dim: int
    qk_nope_dim: int
    v_head_dim: int


@dataclass(frozen=True)
class Csa2IR:
    """V4.1 CSA2 跨层缓存计划（compress_ratios/kv 源/索引源/候选源）。"""
    ratio: int                          # 0=纯 SWA；1/2=压缩比例
    kv_source: Optional[int]            # 压缩 KV 潜层来源（producer）
    index_source: Optional[int]
    produces_kv: bool
    produces_index: bool
    produces_candidates: bool


@dataclass(frozen=True)
class MoeIR:
    experts: int
    top_k: int
    intermediate: int                   # 每专家 SwiGLU inter（gate/up/down 各 inter）
    shared_intermediate: int = 0        # 0 = 无共享专家
    shared_gate: bool = False           # shared_expert_gate: Linear(h,1)
    routed_scaling: Optional[float] = None
    scoring: str = "softmax"            # softmax | sigmoid（V3 系）


@dataclass(frozen=True)
class LayerIR:
    index: int
    kind: str                           # FULL | LINEAR | SLIDING | MLA | MLA_CSA2
    attn: Optional[AttentionIR] = None  # FULL/SLIDING 层
    linear: Optional[LinearAttnIR] = None  # LINEAR 层
    mla: Optional[MlaIR] = None
    csa2: Optional[Csa2IR] = None
    moe: Optional[MoeIR] = None         # None = dense FFN
    dense_inter: Optional[int] = None   # dense FFN inter（moe=None 时）


@dataclass(frozen=True)
class SourceIdentity:
    """模型资产身份（repo/revision/内容 hash；PENDING_ASSET 如实标注）。"""
    repo_id: str
    revision: str
    config_sha256: str
    captured_at: str = ""
    note: str = ""

    def to_json(self) -> dict:
        return dict(self.__dict__)


@dataclass(frozen=True)
class DraftIR:
    """草稿模型结构（DFlash v1：6 层 dense GQA + 块扩散绑定参数）。

    方法协议参数（block_size/iterations/hidden 消费方式）在 methods/ 绑定；
    这里只登记结构与其证据。DFlash v1 一次前向并行提案（非迭代），
    使用目标 lm_head 与输入 embedding（不自带）。
    """
    id: str
    label: str
    layers: List[LayerIR]
    hidden: int
    vocab: int
    uses_target_embeddings: bool
    uses_target_lm_head: bool
    target_layer_ids: Tuple[int, ...]   # 消费的目标层 hidden（fc 输入拼接）
    fc_in: int                          # len(target_layer_ids)·hidden
    block_size: int
    mask_token_id: Optional[int]
    source: SourceIdentity
    attention_family: str = "gqa"


@dataclass(frozen=True)
class ModelIR:
    """主模型结构 + 身份 + 字段覆盖 + 结构完备性。"""
    id: str
    label: str
    family: str                         # gqa | qwen3_moe | qwen3_5_moe | qwen3_next | deepseek_v3 | v41_csa2
    hidden: int
    vocab: int
    layers: List[LayerIR]
    source: SourceIdentity
    max_position: int = 0
    tie_embeddings: bool = False
    dtype_hint: str = ""                # config 声明 dtype（默认精度线索，非 recipe）
    coverage: Dict[str, Dict[str, str]] = field(default_factory=dict)  # key → {status,consumer,note}
    gaps: List[str] = field(default_factory=list)   # structure_complete=False 的原因
    extras: Dict[str, Any] = field(default_factory=dict)  # 家族私有（kv_spec 等），消费者按 family 读
    mtp_layers: int = 0                 # config 原生 MTP 层数（方法层消费）

    @property
    def structure_complete(self) -> bool:
        return not self.gaps

    def layer_kinds(self) -> List[str]:
        return [l.kind for l in self.layers]

    def kind_counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for k in self.layer_kinds():
            out[k] = out.get(k, 0) + 1
        return out

    def kv_bytes_per_token(self, kv_bits: int) -> int:
        """每 token 全模型逻辑 KV 字节（GQA/SLIDING 层）。

        LINEAR 层无 per-token KV（常数状态见 LinearAttnIR.state_elems，容量另计）；
        MLA 家族的压缩 KV 由 extras.kv_spec 描述，此处显式拒绝 GQA 口径冒充。
        """
        if self.family in ("deepseek_v3", "v41_csa2"):
            raise ValueError("MLA 家族 KV 按 extras.kv_spec 计算，勿用 GQA 口径")
        elems = 0
        for layer in self.layers:
            if layer.attn is not None:
                elems += 2 * layer.attn.kv_heads * layer.attn.head_dim
        return int(elems * (kv_bits / 8))

    def per_request_state_elems(self) -> int:
        """LINEAR 层每请求常数状态元素总数（recurrent + conv）。"""
        return sum(l.linear.state_elems + l.linear.conv_state_elems
                   for l in self.layers if l.linear is not None)

    def to_json(self) -> dict:
        return {
            "schema_version": "0.3",
            "id": self.id, "label": self.label, "family": self.family,
            "hidden": self.hidden, "vocab": self.vocab,
            "layers": [
                {"index": l.index, "kind": l.kind,
                 **({"attn": dict(l.attn.__dict__)} if l.attn else {}),
                 **({"linear": dict(l.linear.__dict__)} if l.linear else {}),
                 **({"mla": dict(l.mla.__dict__)} if l.mla else {}),
                 **({"csa2": dict(l.csa2.__dict__)} if l.csa2 else {}),
                 **({"moe": dict(l.moe.__dict__)} if l.moe else {}),
                 **({"dense_inter": l.dense_inter} if l.dense_inter else {})}
                for l in self.layers],
            "source": self.source.to_json(),
            "max_position": self.max_position,
            "tie_embeddings": self.tie_embeddings,
            "dtype_hint": self.dtype_hint,
            "coverage": self.coverage,
            "gaps": list(self.gaps),
            "extras": self.extras,
            "mtp_layers": self.mtp_layers,
            "structure_complete": self.structure_complete,
            "kind_counts": self.kind_counts(),
        }
