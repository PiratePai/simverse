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
from ..model_ir.families import engram_bucket_primes
from ..count.moe import require_divisible_ep
from ..draft_quant import (FUSED_RAW, MXFP8_CONTEXT_PROJ, PER_LAYER_FALLBACK,
                           dtype_bytes, path_unknown_components,
                           require_ctx_kv_path)

# R02b：无尺寸证据的运行时保留内存（不填 0；峰值完整性标志）
UNKNOWN_RUNTIME_COMPONENTS = (
    "kernel workspace 池（cuBLAS/FlashInfer 等，无源码尺寸证据）",
    "CUDA graph 捕获池（vLLM eager/graph 模式保留内存，未见配方）",
    "NCCL 通信缓冲（allreduce/all2all 内部缓冲，未见配方）",
)

# R04（2026-11-11）：Engram 表驻留位置（显式场景输入，缺失/非法拒绝，不猜）。
# 默认 cpu_offload 为**研究假设配方**：证据 = pinned 源码 cpu_offload 路径
# （engram.py L1037-1039/L1205-1221 pinned host memory + UVA 读取）+
# 101.4 GiB/层（≈94 GiB）客观不可驻留单卡 GPU；不证明实际部署以该配置运行。
ENGRAM_PLACEMENT_CPU_OFFLOAD = "cpu_offload"
ENGRAM_PLACEMENT_GPU = "gpu"
ENGRAM_TABLE_PLACEMENTS = (ENGRAM_PLACEMENT_CPU_OFFLOAD, ENGRAM_PLACEMENT_GPU)


def _require_engram_placement(value: str) -> str:
    """Engram 表位置输入校验（镜像 draft_quant 路径的显式拒绝模式）。"""
    if value not in ENGRAM_TABLE_PLACEMENTS:
        raise ValueError(
            f"engram_table_placement 必须是 {ENGRAM_TABLE_PLACEMENTS} 之一，"
            f"收到 {value!r}：表驻留位置是显式配方输入（默认 cpu_offload 为"
            "研究假设），不能由位宽/层数推断")
    return value


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
    placement: str = "gpu"       # gpu | cpu_offload（host 驻留不进 GPU 峰值）

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
        """存活对象求和（kinds 过滤；默认全部实体对象）。

        R04：placement != "gpu"（cpu_offload 的 Engram 表）是 host 驻留对象，
        不进 GPU 侧任何求和；host 字节经 host_resident_bytes() 单列披露。
        """
        return sum(a.bytes for a in self._dedup()
                   if (kinds is None or a.kind in kinds) and a.placement == "gpu")

    def host_resident_bytes(self) -> int:
        """host 驻留对象字节（cpu_offload Engram 表等；不进 GPU 峰值）。"""
        return sum(a.bytes for a in self._dedup() if a.placement != "gpu")

    def steady_bytes(self) -> int:
        """常驻 + 每请求（稳态占用，不含临时缓冲）。"""
        return self.live_total(("weights", "kv", "state"))

    def transient_peak_bytes(self) -> int:
        """临时对象峰值：串行阶段取 max，跨轮缓冲单独相加。

        colocated 串行路径配方（dflash_generate pin：draft 前向 → verify
        → 拒绝/回退，两阶段不重叠）；phase:* 各自内部按同阶段并存求和。
        host 驻留对象（cpu_offload 表）不进 GPU 峰值（R04）。
        """
        dedup = self._dedup()
        cross = sum(a.bytes for a in dedup
                    if a.lifecycle == "cross-round" and a.placement == "gpu")
        phases: Dict[str, int] = {}
        for a in dedup:
            if a.lifecycle.startswith("phase:") and a.placement == "gpu":
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
            "host_resident_bytes": self.host_resident_bytes(),
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


def engram_allocations(engram_spec: dict, hidden: int, quant: dict, tp: int,
                       layer_id: int, role: str = "target", rank: int = 0,
                       table_placement: str = ENGRAM_PLACEMENT_CPU_OFFLOAD
                       ) -> List[Allocation]:
    """一个 Engram 层的驻留对象（R04；common/engram.py pin @f358f057
    blob sha1 ba4ceea6，source-checked 2026-11-11）。

      * 表：per-rank 行数 = 该 rank 分到的素数桶长之和 × 264 B/行
        （FP8 [rows,256] + ue8m0 [rows,8]，L1151-1161/L968-969；桶分配
        engram.py L211-252 → families.engram_bucket_primes 移植；TP head
        ceil 切分，断言 (tp−1)·part < 24，L1094-1099）。**不是**
        num_embeddings/tp；
      * 表位置：显式 table_placement（默认 cpu_offload = 研究假设配方，
        证据 = 源码 cpu_offload/pinned-host 路径 L1037-1039/L1205-1221 +
        101.4 GiB/层不可驻留单卡；host 驻留不进 GPU 峰值，经
        host_resident_bytes 披露）；gpu 驻留需显式场景输入；
      * wkv：ColumnParallelLinear 6144→25600 列切 /tp（L1719-1733）；
      * qk：q_weight/k_weight [hc_mult,5120] bf16 每 rank 复制（L1707-1714）。
    """
    _require_engram_placement(table_placement)
    cols = int(engram_spec["n_hash_cols"])
    d_v = int(engram_spec["head_dim"])
    row_b = int(engram_spec["bytes_per_row"])
    layer_ids = list(engram_spec["layer_ids"])
    if layer_id not in layer_ids:
        raise ValueError(f"层 {layer_id} 不在 engram_layer_ids {layer_ids}")
    if tp < 1 or not 0 <= rank < tp:
        raise ValueError(f"Engram rank {rank} / TP{tp} 非法")
    idx = layer_ids.index(layer_id)
    primes = engram_bucket_primes(int(engram_spec["vocab_size"]),
                                  len(layer_ids),
                                  int(engram_spec["max_ngram_size"]),
                                  int(engram_spec["n_heads"]))
    part = -(-cols // tp)
    if (tp - 1) * part >= cols:
        raise ValueError(
            f"Engram TP{tp} 非法：(tp−1)·part={(tp - 1) * part} ≥ {cols} 个"
            "哈希头（engram.py L1096 断言；合法 tp 见 count/work.engram_rows）")
    b_w = quant["dense_w"] / 8
    wkv_out = hidden * (int(engram_spec["hc_mult"]) + 1)
    if wkv_out % tp:
        raise ValueError(
            f"Engram wkv 输出宽 {wkv_out} 不整除 TP{tp}（列切要求整除）")
    head_start = min(rank * part, cols)
    head_end = min(head_start + part, cols)
    rows_r = sum(primes[idx][head_start:head_end])
    return [
        Allocation(
            f"{role}-w-engram-table-L{layer_id}", "weights", role,
            int(rows_r * row_b), placement=table_placement,
            live_note=(f"素数桶 head 分片 {rows_r} 行×{row_b}B（TP{tp} "
                       f"part={part}，非 num_embeddings/tp；L211-252/"
                       "L1151-1161）；默认 cpu_offload（engram_config "
                       "cpu_offload 路径 + "
                       f"{rows_r * row_b / 2**30:.1f} GiB/层客观不可驻留 GPU；"
                       "研究假设配方），表在 host pinned memory（UVA 读取）；"
                       "gpu 驻留需显式场景输入 engram_table_placement")),
        Allocation(
            f"{role}-w-engram-wkv-L{layer_id}", "weights", role,
            int(cols * d_v * (wkv_out // tp) * b_w),
            live_note=f"常驻（ColumnParallelLinear {cols * d_v}→{wkv_out} 列切 "
                      f"/tp={tp}，gather_output=True；L1719-1733）"),
        Allocation(
            f"{role}-w-engram-qk-L{layer_id}", "weights", role,
            int(2 * int(engram_spec["hc_mult"]) * hidden * 2),
            live_note="q_weight/k_weight [hc_mult,h] bf16 每 rank 复制"
                      "（不随 TP 缩放；L1707-1714）"),
    ]


def engram_staged_allocations(engram_spec: dict, tp: int, tokens: int,
                              role: str = "target") -> List[Allocation]:
    """Engram staged_rows 查表暂存（下界；跨轮常驻缓冲）。

    源码按 max_num_batched_tokens×dp ×part_heads×256 bf16 一次性分配
    （engram.py L1749-1756，persistent storage 稳定 graph 重放地址）——
    运行时常数未 pin，此处按场景 tokens 给**下界**并注明；不计 0。
    """
    cols = int(engram_spec["n_hash_cols"])
    d_v = int(engram_spec["head_dim"])
    part = -(-cols // tp)
    allocs: List[Allocation] = []
    for li_e in engram_spec["layer_ids"]:
        staged = int(tokens * part * d_v * 2)     # bf16 下界
        allocs.append(Allocation(
            f"{role}-t-engram-staged-L{li_e}", "transient", role, staged,
            lifecycle="cross-round",
            live_note=f"staged_rows [tokens×part_heads×{d_v}] bf16 下界 "
                      f"（TP{tp} part={part}）；实际按 max_num_batched_tokens"
                      "×dp 分配（L1749-1756，运行时常数未 pin）"))
    return allocs


def v41_index_buffer_allocations(kv_spec: dict, tokens: int,
                                 role: str = "target") -> List[Allocation]:
    """V4.1 共享索引/候选缓冲下界（R20；跨轮常驻，模型级单例）。

    topk_indices_buffer [max_num_batched_tokens, index_topk] int32 与
    candidate_block_buffer [max_num_batched_tokens, candidate_topk_blocks]
    int32（nvidia/model.py L781-800）按运行时常数 max_num_batched_tokens
    一次性分配（graph 重放地址稳定）——常数未 pin，按场景 tokens 给下界。
    两个缓冲**全模型一份**（所有 index source 层复写/共享），不乘层数。
    """
    if tokens <= 0:
        return []
    topk = int(kv_spec["index_topk"])
    allocs = [Allocation(
        f"{role}-t-v41-topk-buffer", "transient", role, int(tokens * topk * 4),
        lifecycle="cross-round",
        live_note=f"topk_indices_buffer [tokens,{topk}] int32 下界"
                  "（模型级单例；实际按 max_num_batched_tokens 分配，"
                  "L781-785 运行时常数未 pin）")]
    ctb = int(kv_spec.get("candidate_topk_blocks", 0) or 0)
    cand_src = int(kv_spec.get("candidate_source_layer_id", -1))
    if cand_src >= 0 and ctb > 0:
        allocs.append(Allocation(
            f"{role}-t-v41-candidate-buffer", "transient", role,
            int(tokens * ctb * 4), lifecycle="cross-round",
            live_note=f"candidate_block_buffer [tokens,{ctb}] int32 下界"
                      "（仅 candidate source 写、>源层 index source 读；"
                      "L793-798 运行时常数未 pin）"))
    return allocs


def target_weight_allocations(model: ModelIR, quant: dict, tp: int,
                              role: str = "target", ep: int = 1, rank: int = 0,
                              plan=None,
                              engram_table_placement: str =
                              ENGRAM_PLACEMENT_CPU_OFFLOAD
                              ) -> List[Allocation]:
    """主模型每 rank 权重驻留（TP/EP 分片后口径；R01 统一配方）。

    embed/lm_head 词表行切 V/tp（VocabParallelEmbedding/ParallelLMHead）；
    attn q/o 切 heads/tp、k/v 按 KV 头规则；GDN qkvz/ba/conv 列切、
    out_proj 行切；MoE router/shared_gate 复制、shared inter/tp、
    routed 按 EP 关(inter/tp)/开(E_l 整专家)。
    V4.1 Engram 层另加表/wkv/qk 对象（R04，engram_allocations）；
    engram_table_placement 为显式配方输入（默认 cpu_offload 研究假设）。
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
            # 单位必须与 count 的 weight_capacity 一致（字节）：旧式漏 b_dense，
            # BF16 下内存少 2×（2026-11-10 审查 F2 同轮发现，独立 oracle 暴露）。
            dense_b += model.hidden * qrank * b_dense             # q_a_proj
            dense_b += model.hidden * d * b_dense                 # kv_proj
            dense_b += qrank * heads * d * b_dense                # q_b_proj
            # q_norm(q_lora_rank)/kv_norm(head_dim) 为普通 RMSNorm bf16 参数
            #（attention.py L440-450），每层常驻、不随 TP/场景 dense_w 缩放。
            dense_b += (qrank + d) * 2
            # o_a（每 rank ng 组；F2：旧式漏乘 ng，TP1 少 8×）
            dense_b += (kvs["heads"] * d // kvs["o_groups"]) * orank * ng * b_dense
            dense_b += ng * orank * model.hidden * b_dense        # o_b（组切分）
            if a.produces_kv:
                # R20：compressor fused_wkv_wgate 与 indexer wk 均为
                # quant_config=None 的 replicated bf16 权重（compressor.py
                # L321-332、attention.py L1508-1517），不随场景 dense_w 缩放
                dense_b += model.hidden * d * mlt * 2             # compress_proj
                dense_b += d * di * 2                             # index_k_proj
                # k_norm [128] / norm [512] bf16（RMSNorm 参数）
                dense_b += (d + di) * 2
            if a.produces_index:
                # weights_proj h→ih replicated bf16（quant_config=None，
                # attention.py L1484-1490）；wq_b q_lora→ih·di 走 quant_config
                dense_b += model.hidden * ih * 2 + qrank * ih * di * b_dense
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
            if li == model.layers[0].index:
                allocs.append(Allocation(
                    f"{role}-w-mhc-broadcast-L{li}", "weights", role,
                    int(mix * model.hidden * 4),
                    live_note="派生自 hc_attn_fn.view(mix,m,h).sum(1)，"
                              "finalize_mhc_broadcast_weights 时生成；"
                              "FP32 每卡复制，非独立参数（model.py L1530-1543）"))
        engram_spec = (model.extras or {}).get("engram") \
            if fam == "v41_csa2" else None
        if engram_spec and li in engram_spec.get("layer_ids", ()):
            allocs += engram_allocations(
                engram_spec, model.hidden, quant, tp, layer_id=li,
                role=role, rank=rank, table_placement=engram_table_placement)
        if layer.dense_inter:
            dense_b += 3 * model.hidden * (layer.dense_inter // tp) * b_dense
        if dense_b:
            allocs.append(Allocation(
                f"{role}-w-dense-L{li}", "weights", role, int(dense_b),
                live_note="常驻（attn/GDN/dense FFN 按层聚合，TP 分片口径）"))
        if layer.moe is not None and layer.moe.experts:
            moe = layer.moe
            moe_route = (model.extras or {}).get("moe_route") \
                if fam == "v41_csa2" else None
            if moe_route is not None:
                # R21：V4.1 pinned EP 整除断言（与 count/work 同规则镜像；
                # qwen35 pin 的余数规则不适用本族）
                require_divisible_ep(moe.experts, ep)
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
            if moe_route is not None:
                # R21：路由偏置向量（FP32 每卡复制，不随 tp/ep 缩放）——
                # e_score_correction_bias[E]（noaux_tc）+ bias_vl[E]（视觉
                # checkpoint 才创建；V4.1-Flash vision_n_layers=32>0）。
                # 深度 DeepseekV4MoE L303-320；moe-pin.json source-checked。
                bias_b = int(moe_route["bias_bytes"]
                             + moe_route.get("bias_vl_bytes", 0))
                vl_note = ("+bias_vl（视觉 checkpoint，HAS_VL 哨兵判定）"
                           if moe_route.get("bias_vl_bytes") else "")
                allocs.append(Allocation(
                    f"{role}-w-moe-route-L{li}", "weights", role, bias_b,
                    live_note=f"常驻 fp32 复制（e_score_correction_bias"
                              f"{vl_note}；sqrtsoftplus/noaux_tc 路由）"))
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
                   role: str = "target", tp: int = 1,
                   v41_kv_record: str = "fp8_ds_mla_sm90") -> List[Allocation]:
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
            # R20：按打包记录宽度计（显式 v41_kv_record 配方；pinned
            # attention.py L583-589/L181-189）——SWA/压缩 584|528 B/token、
            # 索引 K 132 B/行；ratio2 环形 state 为每请求 capacity(8) 行
            # ×1024 B fp32（compressor.py L250-252），非每 token 常数。
            from ..scenario import V41_INDEX_K_BYTES, V41_KV_RECORDS, \
                V41_RING_CAPACITY
            rec = V41_KV_RECORDS[v41_kv_record]
            swa_rec, comp_rec = rec["swa_bytes"], rec["compressed_bytes"]
            swa, r_set = spec["sliding_window"], spec["compress_ratios"]
            total = 0
            for i, layer in enumerate(model.layers):
                r = r_set[i]
                total += min(ctx, swa) * swa_rec
                csa = layer.csa2
                if csa.produces_kv:
                    total += (ctx // r) * comp_rec + (ctx // r) * V41_INDEX_K_BYTES
                    if r > 1:
                        total += V41_RING_CAPACITY * 2 * spec["head_dim"] * 4
            allocs.append(Allocation(f"{role}-kv", "kv", role, int(total) * batch,
                                     lifecycle="per-request",
                                     live_note=f"SWA 封顶 {swa_rec}B/token + "
                                     f"压缩 {comp_rec}/索引 {V41_INDEX_K_BYTES}B"
                                     f"（{v41_kv_record}）+ ratio2 环形 state "
                                     f"{V41_RING_CAPACITY}×1024B/请求/层"))
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
        source=draft_ir.source, tie_embeddings=False,
        dtype_hint=getattr(draft_ir, "dtype_hint", ""))


def draft_extra_allocations(draft_ir, quant: dict, tp: int,
                            role: str = "draft",
                            ctx_kv_path: str = "unknown") -> List[Allocation]:
    """DFlash 草稿的块扩散专用对象（层权重另由 target_weight_allocations 计）。

    fc：ReplicatedLinear(len(layer_ids)·h → h)，每 rank 复制；
    hidden_norm/norm：h 复制；mask_embedding：h（vLLM 默认 0 张量占位，
    读取 mask_token_id 行时由共享 embedding 提供，此处计占位）。

    ctx-KV 路径（显式 `ctx_kv_path`；dense_w 只是计价位宽，不能据此推断）：
      * fused_raw：`_fused_kv_weight` 全层 K/V 行 `torch.cat` **新驻留副本**
        （qwen3_dflash.py L541-568，非视图；仅 UnquantizedLinearMethod +
        非 CPU 的 `_reads_raw_weights` 路径建立）；
      * mxfp8_context_proj：独立 `_context_kv_proj` ReplicatedLinear，仅装
        K/V 行（L529-531/L570-597）。MXFP8 打包权重按 1B/elem 计下界；
        block-scale 存储布局未入 pin，缺口由调用方传播；
      * per_layer_fallback：逐层 `qkv_proj` 回退（L688-697），**无**融合/
        独立投影副本；
      * `_k_norm_weights` 的 L×d `torch.stack` 在**所有**路径建立
        （L565-568），单列为常驻对象。
    """
    path = require_ctx_kv_path(ctx_kv_path)
    b_w = quant["dense_w"] / 8
    h = draft_ir.hidden
    fc_b = int(draft_ir.fc_in * h * b_w)
    norms_b = int(2 * h * b_w)
    mask_b = int(h * b_w)
    allocs = [Allocation(
        f"{role}-w-dflash-fc-norms", "weights", role, fc_b + norms_b + mask_b,
        live_note=f"fc(replicated {draft_ir.fc_in}→{h}) + hidden_norm/norm "
                  f"+ mask_embedding 占位（不随 TP{tp} 缩放）")]
    fused_kv_elems = 0
    k_norm_elems = 0
    for layer in draft_ir.layers:
        a = layer.attn
        if a is None:
            continue
        kv_rank = a.kv_heads // tp if a.kv_heads >= tp else a.kv_heads
        fused_kv_elems += 2 * kv_rank * a.head_dim
        k_norm_elems += a.head_dim
    if k_norm_elems:
        # k_norm 是 RMSNorm 参数：weight 在模型初始化 dtype 下创建
        # （pinned layernorm.py L42-58 + base_loader.py L68-85），
        # torch.stack 继承该 dtype；不能用 quant.dense_w 计。
        b_norm = dtype_bytes(draft_ir.dtype_hint)
        allocs.append(Allocation(
            f"{role}-w-dflash-k-norm-stack", "weights", role,
            int(k_norm_elems * b_norm),
            live_note=f"_k_norm_weights L×d={k_norm_elems} elem torch.stack"
                      f"（所有路径建立；dtype_hint={draft_ir.dtype_hint}→"
                      f"{b_norm}B/elem，与 dense_w 计价位宽分离）"))
    if path == FUSED_RAW and fused_kv_elems:
        allocs.append(Allocation(
            f"{role}-w-dflash-fused-ctx-kv", "weights", role,
            int(fused_kv_elems * h * b_w),
            live_note=f"_fused_kv_weight 全层 K/V 行 cat 副本："
                      f"{fused_kv_elems}×{h} elem（TP{tp} per-rank；"
                      f"raw/Unquantized 路径，计价位宽 b_w={b_w:g}B/elem）"))
    elif path == MXFP8_CONTEXT_PROJ and fused_kv_elems:
        allocs.append(Allocation(
            f"{role}-w-dflash-mxfp8-ctx-kv", "weights", role,
            int(fused_kv_elems * h * 1.0),
            live_note=f"_context_kv_proj 独立 MXFP8 投影："
                      f"{fused_kv_elems}×{h} elem（TP{tp} per-rank；"
                      f"packed 8bit 权重下界，block-scale 字节未入 pin）"))
    # per_layer_fallback：逐层回退不产生额外常驻副本（读流量差异在
    # count/draft.ctx_kv_proj 计）；k_norm 堆叠已在上面单列。
    return allocs


def draft_peak_lower_bound(*, fc_in: int, noise: int, logits: int,
                           normed_ctx: int, qkv_staging: int,
                           ctx_kv_flat: int, fused_kv_out: int,
                           ctx_k_normed: int) -> Tuple[int, int, List[Tuple[str, int]]]:
    """草稿阶段已知并存组的下界（不把互斥生命周期相加）。

    pinned vLLM `_project_context_kv`/`precompute_and_store_context_kv`：
      * `normed_ctx + flat`（F.linear 输入/输出）同活；
      * `flat + contiguous`（permute+contiguous）同活；返回后 flat 无引用；
      * `contiguous + K norm 输出`（_normalize_context_k）同活；
      * per-layer fallback 额外有 `normed_ctx + qkv_staging` 与
        `qkv_staging + flat`（逐层 qkv 输出与 stack）。
    noise/logits 位于块首/块尾，不与 ctx 组并活；fc_in 单独取 max。
    返回 `(peak, ctx_peak, groups)`；groups 供生命周期 oracle 逐项核对。
    """
    groups: List[Tuple[str, int]] = [
        ("normed_ctx+flat", normed_ctx + ctx_kv_flat),
        ("flat+contiguous", ctx_kv_flat + fused_kv_out),
        ("contiguous+k_normed", fused_kv_out + ctx_k_normed),
    ]
    if qkv_staging:
        groups.extend([
            ("normed_ctx+qkv_staging", normed_ctx + qkv_staging),
            ("qkv_staging+flat", qkv_staging + ctx_kv_flat),
        ])
    ctx_peak = max(v for _, v in groups)
    peak = max(fc_in, noise, logits, ctx_peak)
    return peak, ctx_peak, groups


def transient_allocations(model: ModelIR, draft_ir, quant: dict, dquant: dict,
                          batch: int, ctx: int, tp: int, block_size: int,
                          e_produced: float,
                          target_vocab_per_rank: int,
                          ctx_kv_path: str = "unknown") -> List[Allocation]:
    """结构化临时缓冲（R02b；尺寸来自 pinned 形状，非运行时测量）。

    phase:draft（草稿前向，一轮内与 verify 串行；R07a 2026-11-12 起
    fc 输入/ctx 归一/ctx 投影临时按 **verify 跨度 B·blk** 计——两服务栈
    均对全跨度物化：vLLM _aux_staging[:num_graph_context_tokens] +
    _project_context_kv 的 all_k/all_v[L, span,…]；SGLang
    project_target_hidden/kv_proj_only 全跨度后被拒行不落缓存）：
      * noise embedding 行 B·blk·h（act 精度）；
      * fc 输入拼接 B·blk·8h（act 精度）；
      * ctx 归一化副本 B·blk·h；ctx 投影输出按显式路径：
          - fused_raw / mxfp8_context_proj：flat 输出
            B·blk·L·2·kvh_rank·d（GEMM）→ permute/contiguous 拷贝
            B·blk·L·2·kvh_rank·d（vLLM；SGLang 无全局拷贝）；
          - per_layer_fallback：逐层**完整** qkv 输出
            B·blk·L·(q_rank+2·kvh_rank)·d（Q 随后被切掉但仍形成缓冲）
            + stack 后 K/V B·blk·L·2·kvh_rank·d（L688-697）；
      * ctx K 逐头 RMSNorm 输出 B·blk·L·kvh_rank·d（RoPE 原地，无额外）；
      * 草稿 logits B·(blk−1)·(V/tp)·4B（fp32，仅候选位置；anchor 不过
        lm_head，与 count/draft.draft_lm_head 的 cand 同口径）。
    **生命周期口径**：`t-draft-buffers.bytes` 是上述缓冲中**已知并存组**的
    最大值（lower bound），不是所有字节的并集/和。pinned vLLM 顺序为
    `normed_ctx + flat`（GEMM）→ `flat + contiguous`（permute+contiguous）；
    flat 返回后释放，再做 `contiguous + K norm 输出`；noise 在块首、logits
    在块尾，均不与 ctx 组并活。未建模的并存缓冲只会使该值偏低，不会把
    上界冒充下界。
    phase:verify（目标块验证）：
      * 目标 logits B·blk·(V/tp)·4B（fp32）；
      * verify 辅助 hidden B·blk·n_feat·h（act；speculator
        _store_hidden_states 的 aux 行，[:produced] 截断前全块行）。
    cross-round（跨轮存活）：
      * target_hidden 上下文特征 B·E[N]·len(target_layer_ids)·h（act 精度），
        每轮取 [:produced] 前缀保留到下轮（dflash_generate pin L308）。
    kernel 内部激活工作集/workspace/graph/NCCL 池：无尺寸证据 →
    RankMemory.unknown_components（不计 0，峰值不完整标志）。
    """
    path = require_ctx_kv_path(ctx_kv_path)
    b_a = quant["act"] / 8
    h = model.hidden
    blk = block_size
    allocs: List[Allocation] = []
    # 草稿 KV/Q per-rank 头数（kvh≥tp 切分；kvh<tp 复制）
    kv_rank = 0
    q_rank = 0
    d_head = 0
    if draft_ir is not None:
        for layer in draft_ir.layers:
            if layer.attn is not None:
                a = layer.attn
                kv_rank = a.kv_heads // tp if a.kv_heads >= tp else a.kv_heads
                q_rank = a.heads // tp
                d_head = a.head_dim
                break
    # R07a：verify 跨度基（两栈全跨度物化；见 docstring）；E[N] 只保留在
    # cross-round target_hidden 前缀（HF [:produced] 口径）。
    span_rows = int(batch * blk)
    draft_noise = int(batch * blk * h * b_a)
    fc_in = int(span_rows * 8 * h * b_a)
    normed_ctx = int(span_rows * h * b_a)
    n_draft_layers = len(draft_ir.layers) if draft_ir is not None else 0
    # vLLM ctx 物化：F.linear 输出（flat）→ permute/contiguous 拷贝（fused_kv_out）
    # → K 逐头 RMSNorm 输出（ctx_k_normed）→ RoPE 原地。SGLang 无全局
    # permute/contiguous 拷贝，但同样有 K norm 输出；本账本按 vLLM 口径。
    ctx_kv_flat = int(span_rows * n_draft_layers
                      * 2 * kv_rank * d_head * b_a)
    fused_kv_out = int(span_rows * n_draft_layers
                       * 2 * kv_rank * d_head * b_a)
    ctx_k_normed = int(span_rows * n_draft_layers
                       * kv_rank * d_head * b_a)
    qkv_staging = 0
    if path == PER_LAYER_FALLBACK:
        # 逐层 qkv 输出（含 Q）在 stack 前全部存活；stack 后再产生 KV 副本。
        qkv_staging = int(span_rows * n_draft_layers
                          * (q_rank + 2 * kv_rank) * d_head * b_a)
    # 草稿只对 blk−1 个候选位置出 logits（块首 anchor 是已提交 token 的
    # embedding，不再过 lm_head）；与 count/draft.draft_lm_head 的 cand 同口径。
    draft_logits = int(batch * max(0, blk - 1) * target_vocab_per_rank * 4)
    # 生命周期下界（不是并集求和）：pinned vLLM 的 ctx 物化顺序为
    #   normed_ctx（hidden_norm 输出）+ flat（F.linear 输出）同时存活
    #   → flat + contiguous（permute+contiguous 拷贝期间）同时存活
    #   → contiguous + K norm 输出（_normalize_context_k 期间）同时存活；
    # flat 在 contiguous 返回后即无引用，K norm 时不再与 normed_ctx 同活。
    # 因此 t-draft-buffers 取“已知并存组”的最大值，而不是所有字节相加；
    # 这是 lower bound（可能漏掉未建模的并存缓冲），不冒充实际峰值上界。
    draft_peak, ctx_peak, ctx_groups = draft_peak_lower_bound(
        fc_in=fc_in, noise=draft_noise, logits=draft_logits,
        normed_ctx=normed_ctx, qkv_staging=qkv_staging,
        ctx_kv_flat=ctx_kv_flat, fused_kv_out=fused_kv_out,
        ctx_k_normed=ctx_k_normed)
    ctx_note = (f"ctx groups max={ctx_peak}B（" + " / ".join(
        f"{name} {value}B" for name, value in ctx_groups) + "）")
    allocs.append(Allocation(
        "t-draft-buffers", "transient", "draft", draft_peak,
        lifecycle="phase:draft",
        live_note=f"lower-bound peak {draft_peak}B = max(fc输入 {fc_in}B, "
                  f"noise {draft_noise}B, logits(fp32) {draft_logits}B, "
                  f"{ctx_note}）；不是并存和（P1 生命周期 oracle）"))
    verify_logits = int(batch * blk * target_vocab_per_rank * 4)
    allocs.append(Allocation(
        "t-verify-buffers", "transient", "target",
        verify_logits, lifecycle="phase:verify",
        live_note=f"目标 logits(fp32) B·blk·(V/tp)·4（B={batch} blk={blk}）"))
    if draft_ir is not None:
        n_feat = len(draft_ir.target_layer_ids)
        aux_hidden = int(batch * blk * n_feat * h * b_a)
        allocs.append(Allocation(
            "t-verify-aux-hidden", "transient", "target", aux_hidden,
            lifecycle="phase:verify",
            live_note=f"verify 辅助 hidden B·blk·{n_feat}·h（aux 行，"
                      f"[:produced] 截断前）"))
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
                          with_transients: bool = True,
                          draft_ctx_kv_path: str = "unknown",
                          engram_table_placement: str =
                          ENGRAM_PLACEMENT_CPU_OFFLOAD,
                          engram_tokens: int = 0,
                          v41_kv_record: str = "fp8_ds_mla_sm90") -> RankMemory:
    """同卡混部 per-rank 账本（R02a/R02b）。

    草稿与 target 同 TP 组（vLLM pin）；embed/lm_head 为 target 的物理
    分片对象，草稿以 reference 行引用（零字节，consumer 登记），不再
    重复驻留。draft_batch：草稿批（默认 = target batch B；逐请求草稿）。
    draft_ctx_kv_path：显式 ctx-KV 实现路径（draft_quant.py 枚举）；
    未知时拒绝，不按 dense_w 猜分支。
    R04：engram_table_placement 显式表位置（默认 cpu_offload 研究假设）；
    engram_tokens>0 时补 staged_rows 下界临时对象（V4.1 主模型）。
    """
    if draft_batch is None:
        draft_batch = batch
    path = None
    rank_mem = RankMemory(f"colocated-r{rank}", budget.device_id, budget)
    for a in target_weight_allocations(model, quant, tp, role="target",
                                       ep=ep, rank=rank,
                                       engram_table_placement=
                                       engram_table_placement):
        a.consumers = ["target"]
        rank_mem.add(a)
    engram_spec = (model.extras or {}).get("engram")
    if engram_spec and engram_tokens > 0:
        for a in engram_staged_allocations(engram_spec, tp, engram_tokens,
                                           role="target"):
            a.consumers = ["target"]
            rank_mem.add(a)
    # R20：V4.1 共享 topk/候选缓冲下界（模型级单例，仅 v41_csa2）
    kv_spec = (model.extras or {}).get("kv_spec")
    if kv_spec and kv_spec.get("family") == "v41_csa2" and engram_tokens > 0:
        for a in v41_index_buffer_allocations(kv_spec, engram_tokens,
                                              role="target"):
            a.consumers = ["target"]
            rank_mem.add(a)
    for a in kv_allocations(model, quant, batch, ctx, role="target", tp=tp,
                            v41_kv_record=v41_kv_record):
        a.consumers = ["target"]
        rank_mem.add(a)
    if draft_ir is not None:
        path = require_ctx_kv_path(draft_ctx_kv_path)
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
        for a in draft_extra_allocations(draft_ir, draft_q, tp,
                                         ctx_kv_path=path):
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
                                           e_produced, vocab_r,
                                           ctx_kv_path=path):
                rank_mem.add(a)
    # R02b：无证据运行时组件 → 峰值不完整（缺口传播，不填 0）
    if with_transients:
        rank_mem.unknown_components = list(UNKNOWN_RUNTIME_COMPONENTS)
        if path is not None:
            rank_mem.unknown_components.extend(path_unknown_components(path))
    return rank_mem
