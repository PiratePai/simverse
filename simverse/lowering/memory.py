"""容量账本（memory v0.2）：对象身份 × 生命周期 × 分层预算。

R02a/R02b/R03a/b（2026-11-10）规则：
  * 物理 allocation 身份：shared_identity 指向唯一物理对象（如 target 的
    词表行切 embedding/lm_head），consumer 列表登记引用者（target/draft）；
    同名不同对象不去重，跨角色共享仅当同一物理分片（vLLM
    load_dflash_model：draft 直接替换为 target 的模块对象）。
  * 生命周期分层（R02b）：
      - resident：常驻权重（load 后不释放）；
      - per-request：KV/GDN 状态（随 batch×ctx 增长，请求存活期）；
      - cross-round：跨轮存活的缓冲（target_hidden 上下文特征）；
      - phase:draft / phase:verify：单轮内串行阶段的临时缓冲，峰值取
        max(phase) 而非求和（同卡串行路径配方）；
      - unknown-runtime：workspace/CUDA graph 池/NCCL 缓冲等无源码尺寸
        证据的运行时保留内存——**不计 0**，作为缺口传播（峰值不完整）。
  * 预算分层（R03a）：nominal（规格）/ allocatable（可分配，分
    measured/hypothesis/unknown）/ extra_overhead（运行时额外扣除，带来源）；
    allocatable=None → feasible 判定 unknown（§7.4），但**已知对象下界超过
    标称容量仍判 OOM**（不依赖预算）。
  * 判定（R03b）：oom / feasible / feasible-conditional（假设预算，仅条件
    结论，不进实际部署排名）/ unknown（预算缺失或峰值不完整），各带原因。
分片口径（R01）：所有对象 per-rank（ParallelPlan 配方；EP 余数 rank 差异
经 rank 参数表达）。DFlash 草稿与 target 同 TP 组，共享 embed/lm_head
物理对象（零额外驻留），fc/norm 复制，层权重按 TP 切分。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..model_ir.schema import ModelIR

# R02b：无尺寸证据的运行时保留内存（不填 0；峰值完整性标志）
UNKNOWN_RUNTIME_COMPONENTS = (
    "kernel workspace 池（cuBLAS/FlashInfer 等，无源码尺寸证据）",
    "CUDA graph 捕获池（vLLM eager/graph 模式保留内存，未见配方）",
    "NCCL 通信缓冲（allreduce/all2all 内部缓冲，未见配方）",
)


@dataclass
class Allocation:
    alloc_id: str
    kind: str                    # weights | kv | state | transient | reference
    role: str                    # target | draft | shared
    bytes: int
    shared_identity: Optional[str] = None   # 物理对象身份（去重键）
    lifecycle: str = "resident"  # resident | per-request | cross-round | phase:draft | phase:verify
    consumers: List[str] = field(default_factory=list)
    live_note: str = ""

    def to_json(self) -> dict:
        return dict(self.__dict__)


@dataclass
class DeviceBudget:
    """设备预算分层（R03a）。

    nominal_bytes：规格标称容量（devices 注册表，source=spec）。
    allocatable_bytes：真实可分配给模型对象的容量；None=unknown。
    allocatable_kind：measured（有环境元数据来源）/ hypothesis（用户研究
    假设）/ unknown。hypothesis 只产生条件结论（R03b）。
    extra_overhead_bytes：预算内需额外扣除的运行时开销（带来源），
    语义为 gross（allocatable 是模型对象可用总额，开销从其中扣除，
    不与 KV/权重重复扣减）。
    """
    device_id: str
    nominal_bytes: int
    allocatable_bytes: Optional[int] = None
    allocatable_kind: str = "unknown"      # measured | hypothesis | unknown
    allocatable_source: str = ""
    extra_overhead_bytes: int = 0
    overhead_source: str = ""
    budget_note: str = ""

    @property
    def effective_budget(self) -> Optional[int]:
        """模型对象可用预算（gross 口径：扣除显式登记的额外开销）。"""
        if self.allocatable_bytes is None:
            return None
        return self.allocatable_bytes - self.extra_overhead_bytes

    def to_json(self) -> dict:
        return dict(self.__dict__)


@dataclass
class RankMemory:
    rank_id: str
    device_id: str
    budget: DeviceBudget
    allocations: List[Allocation] = field(default_factory=list)
    unknown_components: List[str] = field(default_factory=list)

    def add(self, alloc: Allocation) -> None:
        self.allocations.append(alloc)

    def _dedup(self) -> List[Allocation]:
        """按物理身份去重（shared_identity 优先；reference 并入物理对象）。"""
        seen: Dict[str, Allocation] = {}
        out: List[Allocation] = []
        for a in self.allocations:
            if a.kind == "reference":
                # 引用行：把 consumer 并入同身份物理对象（不产生字节）
                prev = seen.get(a.shared_identity or "")
                if prev is not None:
                    for c in a.consumers:
                        if c not in prev.consumers:
                            prev.consumers.append(c)
                continue
            key = a.shared_identity or f"{self.rank_id}:{a.alloc_id}"
            if key in seen:
                # 同一物理对象被再次声明：合并 consumer，不重复计字节
                prev = seen[key]
                for c in a.consumers:
                    if c not in prev.consumers:
                        prev.consumers.append(c)
                continue
            seen[key] = a
            out.append(a)
        return out

    def live_total(self, kinds: Optional[Tuple[str, ...]] = None) -> int:
        """存活对象求和（kinds 过滤；默认全部实体对象）。"""
        return sum(a.bytes for a in self._dedup()
                   if kinds is None or a.kind in kinds)

    def steady_bytes(self) -> int:
        """常驻 + 每请求（稳态占用，不含临时缓冲）。"""
        return self.live_total(("weights", "kv", "state"))

    def transient_peak_bytes(self) -> int:
        """临时对象峰值：串行阶段取 max，跨轮缓冲单独相加。

        colocated 串行路径配方（dflash_generate pin：draft 前向 → verify
        → 拒绝/回退，两阶段不重叠）；phase:* 各自内部按同阶段并存求和。
        """
        cross = sum(a.bytes for a in self._dedup()
                    if a.lifecycle == "cross-round")
        phases: Dict[str, int] = {}
        for a in self._dedup():
            if a.lifecycle.startswith("phase:"):
                phases[a.lifecycle] = phases.get(a.lifecycle, 0) + a.bytes
        return cross + (max(phases.values()) if phases else 0)

    def known_peak_bytes(self) -> int:
        return self.steady_bytes() + self.transient_peak_bytes()

    def peak_complete(self) -> bool:
        """峰值完整性：存在无证据运行时组件 → False（缺口传播）。"""
        return not self.unknown_components

    def peak_vs_budget(self) -> Dict[str, object]:
        known = self.known_peak_bytes()
        steady = self.steady_bytes()
        eff = self.budget.effective_budget
        nominal = self.budget.nominal_bytes
        complete = self.peak_complete()
        reasons: List[str] = []
        if known > nominal:
            # R03b：已知存储下界超过标称物理上限 → OOM（不依赖预算证据）
            status = "oom"
            reasons.append(
                f"已知对象 {known}B（稳态 {steady}B + 结构化临时 "
                f"{known - steady}B）> 标称容量 {nominal}B")
        elif eff is None:
            status = "unknown"
            reasons.append(
                "可分配预算未知（allocatable=None）：仅标称容量，"
                "低于标称不能证明 feasible（§7.4）")
        elif not complete:
            status = "unknown"
            reasons.append("峰值不完整：" + "；".join(self.unknown_components))
        elif known > eff:
            status = "oom"
            reasons.append(f"峰值 {known}B > 可用预算 {eff}B")
        elif self.budget.allocatable_kind == "hypothesis":
            status = "feasible-conditional"
            reasons.append(
                f"假设预算（{self.budget.allocatable_source or '用户输入'}）："
                "仅条件结论，不构成实测可部署")
        else:
            status = "feasible"
        return {
            "rank_id": self.rank_id, "steady_bytes": steady,
            "transient_peak_bytes": self.transient_peak_bytes(),
            "known_peak_bytes": known,
            "nominal_bytes": nominal,
            "allocatable_bytes": self.budget.allocatable_bytes,
            "allocatable_kind": self.budget.allocatable_kind,
            "extra_overhead_bytes": self.budget.extra_overhead_bytes,
            "effective_budget": eff,
            "peak_complete": complete,
            "unknown_components": list(self.unknown_components),
            "status": status,
            "reasons": reasons,
            "headroom_bytes": (eff - known) if eff is not None else None,
        }

    def inventory(self) -> List[dict]:
        return [a.to_json() for a in self._dedup()]


# ---------------------------------------------------------------------------
# 权重驻留（per-rank；plan 配方）
# ---------------------------------------------------------------------------


def target_weight_allocations(model: ModelIR, quant: dict, tp: int,
                              role: str = "target", ep: int = 1, rank: int = 0,
                              plan=None) -> List[Allocation]:
    """主模型每 rank 权重驻留（TP/EP 分片后口径；R01 统一配方）。

    embed/lm_head 词表行切 V/tp（VocabParallelEmbedding/ParallelLMHead）；
    attn q/o 切 heads/tp、k/v 按 KV 头规则；GDN qkvz/ba/conv 列切、
    out_proj 行切；MoE router/shared_gate 复制、shared inter/tp、
    routed 按 EP 关(inter/tp)/开(E_l 整专家)。
    """
    b_dense = quant["dense_w"] / 8
    b_expert = quant["expert_w"] / 8
    allocs: List[Allocation] = []
    fam = model.family
    for layer in model.layers:
        li = layer.index
        dense_b = 0
        if layer.attn is not None:
            a = layer.attn
            kv_rank = a.kv_heads // tp if a.kv_heads >= tp else a.kv_heads
            # q_proj（output_gated 输出 2×）列切；k/v 切或复制；o_proj 行切。
            # 权重字节 = 输出宽 × 输入宽 h（2026-11-10 修正：旧式漏乘 h，
            # attention 投影按输出宽单元素计，低估 h 倍）
            dense_b += (a.heads * a.head_dim * (2 if a.output_gated else 1) // tp
                        + 2 * kv_rank * a.head_dim) * model.hidden * b_dense
            dense_b += (a.heads * a.head_dim // tp) * model.hidden * b_dense
            if a.qk_head_norm:
                dense_b += 2 * a.head_dim * b_dense
        if layer.linear is not None:
            g = layer.linear
            # vLLM in_proj_qkvz（=HF in_proj_qkv+in_proj_z）列切 /tp
            qkvz_out = 2 * g.key_dim + 2 * g.value_dim
            dense_b += (qkvz_out // tp + 2 * g.num_v_heads // tp) * model.hidden * b_dense
            dense_b += (g.qkv_out // tp) * g.conv_kernel * b_dense   # conv1d
            dense_b += (g.value_dim // tp) * model.hidden * b_dense  # out_proj
            dense_b += 2 * (g.num_v_heads // tp) * 4     # A_log/dt_bias fp32
        if layer.mla is not None and fam == "deepseek_v3":
            m = layer.mla
            heads_c = m.heads // tp
            dense_b += (model.hidden * m.q_lora_rank
                        + m.q_lora_rank * heads_c * (m.qk_nope_dim + m.qk_rope_dim)
                        + model.hidden * (m.kv_lora_rank + m.qk_rope_dim)
                        + m.kv_lora_rank * heads_c * (m.qk_nope_dim + m.v_head_dim)
                        + heads_c * m.v_head_dim * model.hidden) * b_dense
        if layer.mla is not None and fam == "v41_csa2" and layer.csa2:
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
                live_note="常驻（attn/GDN/dense FFN 按层聚合，TP 分片口径）"))
        if layer.moe is not None and layer.moe.experts:
            moe = layer.moe
            router_b = int(model.hidden * moe.experts * b_dense)  # replicated
            shared_b = int(3 * model.hidden * (moe.shared_intermediate // tp)
                           * b_dense) if moe.shared_intermediate else 0
            gate_b = int(model.hidden * b_dense) if moe.shared_gate else 0
            if ep != 1:
                base, rem = divmod(moe.experts, ep)
                e_l = base + (1 if rank < rem else 0)
                experts_b = int(e_l * 3 * model.hidden * moe.intermediate
                                * b_expert)
                note = f"EP{ep} E_local={e_l}（整专家）"
            else:
                experts_b = int(moe.experts * 3 * model.hidden
                                * (moe.intermediate // tp) * b_expert)
                note = f"EP off inter/tp={moe.intermediate // tp}（每 rank 全 E）"
            allocs.append(Allocation(
                f"{role}-w-moe-L{li}", "weights", role,
                router_b + shared_b + gate_b + experts_b,
                live_note=f"常驻（router/shared_gate 复制；{note}）"))
    vocab_r = model.vocab // tp if model.vocab % tp == 0 else model.vocab
    if not model.tie_embeddings:
        allocs.append(Allocation(
            f"{role}-w-embed", "weights", role,
            int(model.hidden * vocab_r * b_dense),
            shared_identity=f"{model.id}:embed@tp{tp}",
            live_note=f"常驻（词表行切 V/tp={vocab_r}；DFlash 草稿共享同一对象）"))
        allocs.append(Allocation(
            f"{role}-w-lm_head", "weights", role,
            int(model.hidden * vocab_r * b_dense),
            shared_identity=f"{model.id}:lm_head@tp{tp}",
            live_note=f"常驻（词表行切 V/tp={vocab_r}；DFlash 草稿共享同一对象）"))
    else:
        allocs.append(Allocation(
            f"{role}-w-embed-tied", "weights", role,
            int(model.hidden * vocab_r * b_dense),
            shared_identity=f"{model.id}:embed@tp{tp}",
            live_note="常驻（tied，词表行切）"))
    # final norm（model.norm，h 复制）
    allocs.append(Allocation(
        f"{role}-w-final-norm", "weights", role, int(model.hidden * b_dense),
        live_note="常驻（final RMSNorm，每 rank 复制）"))
    return allocs


# ---------------------------------------------------------------------------
# KV / 状态（per-request；per-rank 口径）
# ---------------------------------------------------------------------------


def kv_allocations(model: ModelIR, quant: dict, batch: int, ctx: int,
                   role: str = "target", tp: int = 1) -> List[Allocation]:
    """KV/常数状态驻留（每请求；batch×ctx；per-rank 分片口径）。

    GQA/SLIDING：KV 头按 plan 切分（kvh<tp 时每 rank 复制全量 = kv_rank
    即 kvh，公式自然成立）。GDN 状态：v 头与 conv 通道按 tp 切分。
    """
    b_kv = quant["kv"] / 8
    b_state = quant.get("state", 32) / 8
    b_act = quant["act"] / 8
    fam = model.family
    allocs: List[Allocation] = []
    if fam in ("deepseek_v3", "v41_csa2"):
        spec = model.extras["kv_spec"]
        if fam == "deepseek_v3":
            per_token_elems = spec["kv_elems_per_token_per_layer"] * len(model.layers)
            allocs.append(Allocation(f"{role}-kv", "kv", role,
                                     int(per_token_elems * ctx * b_kv) * batch,
                                     lifecycle="per-request",
                                     live_note="随 ctx 增长（MLA 压缩 KV 复制）"))
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
                                     lifecycle="per-request",
                                     live_note="SWA 封顶+压缩潜层（Engram/hc 状态缺口）"))
    else:
        elems = 0
        for l in model.layers:
            if l.attn is not None:
                kv_rank = (l.attn.kv_heads // tp if l.attn.kv_heads >= tp
                           else l.attn.kv_heads)
                elems += 2 * kv_rank * l.attn.head_dim
        allocs.append(Allocation(f"{role}-kv", "kv", role,
                                 int(elems * ctx * b_kv) * batch,
                                 lifecycle="per-request",
                                 live_note=f"随 ctx 增长（per-rank KV 头口径，"
                                           f"tp={tp}）"))
    # GDN 常数状态（per-rank：v 头/conv 通道 /tp）
    state_elems_r = 0
    conv_elems_r = 0
    for l in model.layers:
        if l.linear is not None:
            g = l.linear
            state_elems_r += (g.num_v_heads // tp) * g.head_k_dim * g.head_v_dim
            conv_elems_r += g.qkv_out // tp * (g.conv_kernel - 1)
    if state_elems_r or conv_elems_r:
        allocs.append(Allocation(
            f"{role}-gdn-state", "state", role,
            int(state_elems_r * b_state + conv_elems_r * b_act) * batch,
            lifecycle="per-request",
            live_note="每请求常数（recurrent fp32 + conv，per-rank 切分）"))
    return allocs


# ---------------------------------------------------------------------------
# 同卡混部（R02a：对象身份 + per-rank；R02b：生命周期峰值）
# ---------------------------------------------------------------------------


def _as_model(draft_ir) -> ModelIR:
    """把 DraftIR 包装成最小 ModelIR 以复用容量函数。"""
    return ModelIR(
        id=draft_ir.id, label=draft_ir.label, family="gqa", hidden=draft_ir.hidden,
        vocab=draft_ir.vocab, layers=list(draft_ir.layers),
        source=draft_ir.source, tie_embeddings=False)


def draft_extra_allocations(draft_ir, quant: dict, tp: int,
                            role: str = "draft") -> List[Allocation]:
    """DFlash 草稿的块扩散专用对象（层权重另由 target_weight_allocations 计）。

    fc：ReplicatedLinear(len(layer_ids)·h → h)，每 rank 复制；
    hidden_norm/norm：h 复制；mask_embedding：h（vLLM 默认 0 张量占位，
    读取 mask_token_id 行时由共享 embedding 提供，此处计占位）。
    """
    b_w = quant["dense_w"] / 8
    h = draft_ir.hidden
    fc_b = int(draft_ir.fc_in * h * b_w)
    norms_b = int(2 * h * b_w)
    mask_b = int(h * b_w)
    return [Allocation(
        f"{role}-w-dflash-fc-norms", "weights", role, fc_b + norms_b + mask_b,
        live_note=f"fc(replicated {draft_ir.fc_in}→{h}) + hidden_norm/norm "
                  f"+ mask_embedding 占位（不随 TP{tp} 缩放）")]


def transient_allocations(model: ModelIR, draft_ir, quant: dict, dquant: dict,
                          batch: int, ctx: int, tp: int, block_size: int,
                          e_produced: float,
                          target_vocab_per_rank: int) -> List[Allocation]:
    """结构化临时缓冲（R02b；尺寸来自 pinned 形状，非运行时测量）。

    phase:draft（草稿前向，一轮内与 verify 串行）：
      * noise embedding 行 B·blk·h（act 精度）；
      * fc 输入拼接 B·E[N]·8h（act 精度）；
      * 草稿 logits B·blk·(V/tp)·4B（fp32，logits_processor 输出口径）。
    phase:verify（目标块验证）：
      * 目标 logits B·blk·(V/tp)·4B（fp32）。
    cross-round（跨轮存活）：
      * target_hidden 上下文特征 B·E[N]·len(target_layer_ids)·h（act 精度），
        每轮取 [:produced] 前缀保留到下轮（dflash_generate pin L308）。
    kernel 内部激活工作集/workspace/graph/NCCL 池：无尺寸证据 →
    RankMemory.unknown_components（不计 0，峰值不完整标志）。
    """
    b_a = quant["act"] / 8
    h = model.hidden
    blk = block_size
    allocs: List[Allocation] = []
    draft_noise = int(batch * blk * h * b_a)
    fc_in = int(batch * max(1.0, e_produced) * 8 * h * b_a)
    draft_logits = int(batch * blk * target_vocab_per_rank * 4)
    allocs.append(Allocation(
        "t-draft-buffers", "transient", "draft",
        draft_noise + fc_in + draft_logits, lifecycle="phase:draft",
        live_note=f"noise {draft_noise}B + fc输入 {fc_in}B + logits(fp32) "
                  f"{draft_logits}B（B={batch} blk={blk} E[N]={e_produced:.2f}）"))
    verify_logits = int(batch * blk * target_vocab_per_rank * 4)
    allocs.append(Allocation(
        "t-verify-buffers", "transient", "target",
        verify_logits, lifecycle="phase:verify",
        live_note=f"目标 logits(fp32) B·blk·(V/tp)·4（B={batch} blk={blk}）"))
    if draft_ir is not None:
        n_feat = len(draft_ir.target_layer_ids)
        hidden_buf = int(batch * max(1.0, e_produced) * n_feat * h * b_a)
        allocs.append(Allocation(
            "t-target-hidden-ctx", "transient", "shared", hidden_buf,
            lifecycle="cross-round",
            live_note=f"target_hidden 上下文特征 B·E[N]·{n_feat}·h"
                      f"（跨轮保留 [:produced] 前缀）"))
    return allocs


def colocated_rank_memory(model: ModelIR, draft_ir, quant, draft_quant, tp: int,
                          batch: int, ctx: int, budget: DeviceBudget,
                          draft_batch: Optional[int] = None, ep: int = 1,
                          rank: int = 0, block_size: int = 0,
                          e_produced: float = 0.0,
                          with_transients: bool = True) -> RankMemory:
    """同卡混部 per-rank 账本（R02a/R02b）。

    草稿与 target 同 TP 组（vLLM pin）；embed/lm_head 为 target 的物理
    分片对象，草稿以 reference 行引用（零字节，consumer 登记），不再
    重复驻留。draft_batch：草稿批（默认 = target batch B；逐请求草稿）。
    """
    if draft_batch is None:
        draft_batch = batch
    rank_mem = RankMemory(f"colocated-r{rank}", budget.device_id, budget)
    for a in target_weight_allocations(model, quant, tp, role="target",
                                       ep=ep, rank=rank):
        a.consumers = ["target"]
        rank_mem.add(a)
    for a in kv_allocations(model, quant, batch, ctx, role="target", tp=tp):
        a.consumers = ["target"]
        rank_mem.add(a)
    if draft_ir is not None:
        draft_q = dict(quant)
        draft_q.update(draft_quant or {})
        # 草稿层权重：同 tp 切分（gqa 家族路径）
        for a in target_weight_allocations(_as_model(draft_ir), draft_q, tp,
                                           role="draft"):
            if "embed" in a.alloc_id or "lm_head" in a.alloc_id:
                # 草稿共享 target 的词表行切对象：登记引用（零字节），
                # 并把 consumer 并入 target 物理对象（去重键一致）
                ident = (f"{model.id}:embed@tp{tp}" if "embed" in a.alloc_id
                         else f"{model.id}:lm_head@tp{tp}")
                rank_mem.add(Allocation(
                    f"draft-ref-{a.alloc_id}", "reference", "shared", 0,
                    shared_identity=ident, consumers=["draft"],
                    live_note="引用 target 分片对象（load_dflash_model："
                              "同模块对象，零额外驻留）"))
                continue
            a.consumers = ["draft"]
            rank_mem.add(a)
        for a in draft_extra_allocations(draft_ir, draft_q, tp):
            a.consumers = ["draft"]
            rank_mem.add(a)
        for a in kv_allocations(_as_model(draft_ir), draft_q, draft_batch, ctx,
                                role="draft", tp=tp):
            a.consumers = ["draft"]
            rank_mem.add(a)
        if with_transients and block_size:
            vocab_r = model.vocab // tp if model.vocab % tp == 0 else model.vocab
            for a in transient_allocations(model, draft_ir, quant, draft_quant,
                                           batch, ctx, tp, block_size,
                                           e_produced, vocab_r):
                rank_mem.add(a)
    # R02b：无证据运行时组件 → 峰值不完整（缺口传播，不填 0）
    if with_transients:
        rank_mem.unknown_components = list(UNKNOWN_RUNTIME_COMPONENTS)
    return rank_mem
