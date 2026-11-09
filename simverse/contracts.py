"""版本化数据契约（方案 v2.2 §3.1 的 v0.1 子集）。

Web / CLI / 导出共用同一份 JSON 结构；所有顶层对象带 schema_version。
不引入 pydantic（环境无此依赖）：dataclass + 显式校验函数 + 规范 JSON 序列化。

契约对象：
  DeviceSpec / LinkSpec    L1 硬件与链路（含逐数字 source）
  ModelSpec / DraftSpec    L0 结构（含 config 溯源与 digest）
  ScenarioSpec             场景（可存盘、可重放）
  OpRecord / StageTiming   RunResult 载荷（逐算子账本 / 阶段时延）
  RunResult                一次计算的完整输出（带 digest、假设、逐数字来源）

口径语义（方案 §4.3）：
  * OpRecord.scope = per_card | group：计数归属每卡或整组（TP 下 GEMM 按分片形状计）；
  * 字节分五类：weight_read / act_read / act_write / kv_read / kv_write（整数，逻辑字节）；
  * 权重容量（显存驻留）与读取量（每步实际读）分列；KV 区分保留量与本步读写；
  * 每条记录带 formula_id（公式溯源）与 source（config/spec/公式估算）。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from . import SCHEMA_VERSION, __version__

# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------


def canonical_json(obj: Any) -> str:
    """规范 JSON：键排序 + 紧凑分隔，作为 digest 与一致性比较的基础。"""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(obj: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()[:16]


def _num(value: Any, name: str, lo: float, hi: float, integer: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须为数值")
    v = float(value)
    if not (lo <= v <= hi):
        raise ValueError(f"{name} 必须在 {lo} 与 {hi} 之间，收到 {v}")
    if integer and v != int(v):
        raise ValueError(f"{name} 必须为整数")
    return int(v) if integer else v


# ---------------------------------------------------------------------------
# L1 硬件契约
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeviceSpec:
    """统一设备口径（GPU；NPU 换算属 M3 vllm-ascend 包，本切片不含）。

    peak_by_dtype: 各精度峰值算力 (FLOP/s)；hbm_bw: 聚合有效带宽 (B/s)；
    mem_capacity: HBM 容量 (B)；launch_overhead_ms: 单 kernel launch 估计（T0）；
    mfu_cap: roofline 计算项的 MFU 上限系数（来源：旧仓 spec_gain 口径 0.75）。
    每个数值都带 source（"spec:..." 或 "assumed:..."）——数据身份原则。
    """

    id: str
    label: str
    peak_by_dtype: Dict[str, float]
    hbm_bw: float
    mem_capacity: float
    launch_overhead_ms: float
    mfu_cap: float
    sources: Dict[str, str]
    card_count: int = 1  # 0 表示由场景的 tp 决定

    def peak(self, dtype: str) -> float:
        if dtype not in self.peak_by_dtype:
            raise ValueError(f"设备 {self.id} 无 {dtype} 峰值算力，请在 DeviceSpec 注册")
        return self.peak_by_dtype[dtype]

    def to_json(self) -> dict:
        return {"schema_version": SCHEMA_VERSION, **asdict(self)}


@dataclass(frozen=True)
class LinkSpec:
    """角色间链路（资源，非免费）：每方向 fixed 延迟 + 字节/带宽。"""

    id: str
    label: str
    bw: float          # B/s 单向
    fixed_ms: float    # 每消息单向启动延迟
    duplex: str = "full"
    source: str = ""

    def to_json(self) -> dict:
        return {"schema_version": SCHEMA_VERSION, **asdict(self)}


# ---------------------------------------------------------------------------
# L0 结构契约
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LayerSpec:
    """一层的主干结构（数学结构，框架无关；MLA/CSA2 细节由模型适配器展开为算子）。"""

    index: int
    kind: str                      # full_attention | linear_attention | dense_ffn 等
    attn: Dict[str, Any]           # 适配器私有：投影形状、KV/索引归属等
    moe: Optional[Dict[str, Any]]  # None = dense FFN；否则 {experts, top_k, inter, shared_inter}


@dataclass(frozen=True)
class ModelSpec:
    """模型结构 + 精度策略 + 溯源。总参数量等汇总由结构推导，不作为输入。"""

    id: str
    label: str
    hidden: int
    layers: List[LayerSpec]
    vocab: int
    kv_spec: Dict[str, Any]        # 每 token KV 保留量的推导参数（适配器私有）
    quant: Dict[str, Any]          # {"dense_w": 8, "expert_w": 4, "act": 8, "kv": 1}（bit）
    active_params: Optional[int] = None  # 官方口径激活参数（仅对账用，不进计数）
    total_params: Optional[int] = None
    lm_head_mult: int = 1          # lm_head 重复计（V4.1 Flash 双头，来源 config 适配器）
    source_note: str = ""
    config_digest: str = ""        # 输入 config 的数据身份

    def to_json(self) -> dict:
        return {"schema_version": SCHEMA_VERSION, **asdict(self)}


@dataclass(frozen=True)
class DraftSpec:
    """链式草稿模型 + 方法语义（SpecMethod 注册表的 v0.1 子集，方案 §4.2）。

    method: mtp | dspark | custom_chain（链式）；块并行方法在注册表显式 unsupported。
    propose_steps S / draft_step_tokens M / verify_tokens K 三参数正交（方案 §4.1 增量①）。
    """

    id: str
    method: str
    label: str
    propose_steps_rule: str        # "S=K"（链式：每轮 K 步）
    params_dense: int = 0          # dense 权重参数量（个）
    moe: Optional[Dict[str, Any]] = None  # 草稿自身 MoE（dspark：{experts, top_k, inter}）
    weight_read_scale: float = 1.0       # 权重读取倍率（mtp 与主干共享驻留 ×0.1）
    kv_frac: float = 0.0                 # 草稿每步 KV 读 = 目标单步 KV 读 × 比例
    kv_own_bytes_per_token: Optional[int] = None  # 或直接给草稿自身每 token KV 字节
    d2t_bytes_per_token: int = 6         # D→T 每 token（token id 4B + q 概率 2B）
    t2d_bytes_per_token: int = 0         # T→D 每接受 token（MTP 分离需回传 hidden 时 >0）
    t2d_bytes_per_request: int = 256     # T→D 每请求控制面（锚: 旧仓 spec_gain_model.js）
    weight_bits: int = 8
    source_note: str = ""
    status: str = "exact"          # exact | approximated
    approximations: List[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {"schema_version": SCHEMA_VERSION, **asdict(self)}


# ---------------------------------------------------------------------------
# 场景契约
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AcceptanceSpec:
    """接受率三字段契约（方案 §4.1）：chain_alpha / committed_mean 互斥；hist v0.1 不支持。"""

    kind: str = "chain_alpha"      # chain_alpha | committed_mean
    chain_alpha: Optional[float] = None
    committed_mean: Optional[float] = None
    includes_bonus: bool = True

    def validate(self, k: int) -> "AcceptanceSpec":
        if self.kind == "chain_alpha":
            _num(self.chain_alpha, "chain_alpha", 0.0, 1.0)
        elif self.kind == "committed_mean":
            tau = _num(self.committed_mean, "committed_mean", 1.0, float(k + 1))
            if not self.includes_bonus:
                raise ValueError("v0.1 仅支持 includes_bonus=true（τ 含 bonus）")
            object.__setattr__(self, "committed_mean", tau)
        else:
            raise ValueError(f"v0.1 不支持接受率输入 {self.kind}（committed_hist 属 M3）")
        return self


@dataclass(frozen=True)
class ScenarioSpec:
    """可重放场景：固定 模型/硬件/精度/负载/方法/部署 + 随机种子。

    deployment.mode:
      baseline        普通 decode（对照基线）
      colocated       同卡投机（draft 与 target 同设备串行）
      dt_serial       DT 分离串行（独立草稿设备 + 链路双向计价）
      dt_pipeline_ideal  DT 流水线理想下界（跨微批无损交错的稳态近似，显式乐观标注）
    """

    schema_version: str = SCHEMA_VERSION
    name: str = "unnamed"
    model_id: str = "deepseek_v41_flash"
    draft_id: str = "dspark"
    k: int = 4                      # verify_tokens：每轮草稿候选数 K（验证输入 K+1）
    tp: int = 8
    target_device: str = "h100_sxm"
    draft_device: str = "l40"       # dt_serial 下草稿独立设备
    link: str = "pcie5"             # dt_* 部署的链路预设 id
    batch: int = 1
    ctx_tokens: int = 32768
    deployment: Dict[str, Any] = field(
        default_factory=lambda: {"mode": "colocated"})
    acceptance: Dict[str, Any] = field(
        default_factory=lambda: {"kind": "chain_alpha", "chain_alpha": 0.75})
    seed: int = 0
    notes: str = ""

    def acceptance_spec(self) -> AcceptanceSpec:
        return AcceptanceSpec(**self.acceptance).validate(self.k)

    def mode(self) -> str:
        return self.deployment.get("mode", "colocated")

    def validate(self) -> "ScenarioSpec":
        _num(self.k, "k", 1, 32, integer=True)
        _num(self.tp, "tp", 1, 64, integer=True)
        _num(self.batch, "batch", 1, 512, integer=True)
        _num(self.ctx_tokens, "ctx_tokens", 1, 2 ** 20, integer=True)
        if self.mode() not in ("baseline", "colocated", "dt_serial", "dt_pipeline_ideal"):
            raise ValueError(f"未知部署模式 {self.mode()}")
        self.acceptance_spec()
        return self

    def to_json(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_json(data: dict) -> "ScenarioSpec":
        known = {f: data[f] for f in ScenarioSpec.__dataclass_fields__ if f in data}
        if data.get("schema_version") not in (None, SCHEMA_VERSION):
            raise ValueError(f"不支持的场景 schema_version: {data.get('schema_version')}")
        return ScenarioSpec(**known).validate()


# ---------------------------------------------------------------------------
# RunResult 载荷
# ---------------------------------------------------------------------------


@dataclass
class OpRecord:
    """逐算子账本行（Count 段输出，全整数；Cost 段附 T0 估算）。

    stage: baseline_decode | prefill | draft_step_0..S-1 | verify | d2t | t2d
    scope: per_card | group（TP 分片后的归属；group 行 = 各卡求和）
    次数字段 exec_count = 该算子在本阶段内的执行次数（如 40 层各 1 次 → 40 行，每行 1）。
    """

    stage: str
    layer: int                      # -1 = 非层内算子（router 总表 / lm_head / 通信）
    op: str
    exec_count: int
    flops: int
    weight_read: int
    act_read: int
    act_write: int
    kv_read: int
    kv_write: int
    weight_capacity: int            # 驻留容量（与读取量分列）
    scope: str = "per_card"
    formula_id: str = ""            # formulas.py 注册表 id
    shape_note: str = ""            # 形状串（M,N,K / 说明）
    fused_into: Optional[str] = None  # 被融合覆盖时指向融合组（防重复计时）

    # Cost 段（T0 公式估算；保留原始计数，可被 T1 校准替换）
    t_mem_ms: float = 0.0
    t_comp_ms: float = 0.0
    t_ms: float = 0.0
    bound: str = ""                 # memory | compute | none

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class StageTiming:
    """一个阶段的汇总时延与事件（Execute 解析档）。"""

    stage: str
    label: str
    lane: str                       # target | draft | link | cpu
    t_ms: float
    formula_id: str
    bytes_d2t: int = 0
    bytes_t2d: int = 0
    tokens_in: int = 0              # 本阶段流入 token 数（M）
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class TimelineEvent:
    """代表性一轮投机的串行时间线（v0.1 为解析串行链；事件状态机属 M3）。"""

    id: str
    label: str
    lane: str
    start_ms: float
    end_ms: float
    depends_on: List[str] = field(default_factory=list)
    stage: str = ""

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class RunResult:
    """一次计算的完整输出。Web/CLI/导出消费同一结构（一致性验收基础）。"""

    schema_version: str = SCHEMA_VERSION
    simulator_version: str = __version__
    status: str = "ok"              # ok | oom | unsupported_method | unsupported_combo
    status_detail: str = ""
    scenario_digest: str = ""
    model_digest: str = ""
    device_digest: str = ""
    scenario: Dict[str, Any] = field(default_factory=dict)
    model: Dict[str, Any] = field(default_factory=dict)
    draft: Dict[str, Any] = field(default_factory=dict)
    devices: Dict[str, Any] = field(default_factory=dict)
    ops: List[OpRecord] = field(default_factory=list)
    stages: List[StageTiming] = field(default_factory=list)
    timeline: List[TimelineEvent] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    comparisons: List[Dict[str, Any]] = field(default_factory=list)  # 各部署对比行
    assumptions: List[str] = field(default_factory=list)
    provenance: Dict[str, str] = field(default_factory=dict)  # 数字 → 来源（公式/规格/config）

    def to_json(self) -> dict:
        out = {
            "schema_version": self.schema_version,
            "simulator_version": self.simulator_version,
            "status": self.status,
            "status_detail": self.status_detail,
            "digests": {
                "scenario": self.scenario_digest,
                "model": self.model_digest,
                "device": self.device_digest,
            },
            "scenario": self.scenario,
            "model": self.model,
            "draft": self.draft,
            "devices": self.devices,
            "ops": [o.to_json() for o in self.ops],
            "stages": [s.to_json() for s in self.stages],
            "timeline": [e.to_json() for e in self.timeline],
            "metrics": self.metrics,
            "comparisons": self.comparisons,
            "assumptions": self.assumptions,
            "provenance": self.provenance,
        }
        return out
