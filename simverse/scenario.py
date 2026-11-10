"""场景契约 v0.2：显式假设快照（Q8 hypothesis / Q10 单任务规模）。

与 v0.1 ScenarioSpec 的差异：
  * 负载/接受率作为显式假设对象（source_kind=hypothesis 登记来源）；
  * 部署显式角色化（target/draft 设备与 TP/EP、链路），不再用 mode 字符串隐含；
  * 精度配方独立（storage/compute 语义在 cost 路由，不在此冒充）；
  * R01/R03a（2026-11-10）：target_ep 显式字段（1=TP 切 inter；=tp=EP 整
    专家；其余组合无实现配方，validate 拒绝）；budget 预算输入契约
    （nominal/allocatable/额外开销分列，缺省 unknown，区分 measured 与
    hypothesis——假设预算只产生条件结论）。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

from .contracts import RngScheme, digest
from .draft_quant import FUSED_RAW, UNKNOWN, normalize_ctx_kv_path

DEPLOY_MODES = ("baseline", "colocated", "dt_serial", "dt_pipeline_sim")

BUDGET_KINDS = ("measured", "hypothesis", "unknown")


@dataclass
class WorkloadHypothesis:
    batch: int = 1
    ctx_tokens: int = 32768
    osl_tokens: Optional[int] = None       # 输出上限（尾轮截断用；未设=未知）
    arrival: str = "fixed-batch"           # v0.2：固定批；到达过程 M4
    source_kind: str = "hypothesis"
    notes: str = "Q8：首版显式假设扫描；具体数值为可编辑演示默认值，非用户确认"


@dataclass
class BudgetInput:
    """设备组预算输入（R03a 契约；缺省=unknown，不自动套标称容量）。

    allocatable_bytes：该设备组上模型对象（权重+KV+状态+临时缓冲）可用的
    总额（gross 语义：运行时额外占用从其中扣除，不与对象重复扣减）。
    allocatable_kind：measured 需非空 allocatable_source（环境/方法）；
    hypothesis 为用户研究假设（判定只到 feasible-conditional）。
    extra_overhead_bytes：需额外扣除的运行时开销（框架预留等），带来源。
    """
    allocatable_bytes: Optional[int] = None
    allocatable_kind: str = "unknown"
    allocatable_source: str = ""
    extra_overhead_bytes: int = 0
    overhead_source: str = ""
    note: str = ""

    def validate(self) -> "BudgetInput":
        if self.allocatable_kind not in BUDGET_KINDS:
            raise ValueError(
                f"allocatable_kind 必须 ∈ {BUDGET_KINDS}，非 {self.allocatable_kind}")
        if self.allocatable_bytes is not None:
            if not isinstance(self.allocatable_bytes, int) \
                    or isinstance(self.allocatable_bytes, bool) \
                    or self.allocatable_bytes <= 0:
                raise ValueError(
                    f"allocatable_bytes 必须为正整数（B），非 {self.allocatable_bytes}")
            if self.allocatable_kind == "unknown":
                raise ValueError(
                    "提供了 allocatable_bytes 必须声明 kind（measured 需来源；"
                    "假设输入用 hypothesis）")
            if self.allocatable_kind == "measured" and not self.allocatable_source:
                raise ValueError(
                    "measured 预算必须带 allocatable_source（采集方法/环境；"
                    "准则 §3.2 同型）")
        if self.allocatable_kind != "unknown" and self.allocatable_bytes is None:
            raise ValueError(f"kind={self.allocatable_kind} 但无 allocatable_bytes")
        if not isinstance(self.extra_overhead_bytes, int) \
                or isinstance(self.extra_overhead_bytes, bool) \
                or self.extra_overhead_bytes < 0:
            raise ValueError(f"extra_overhead_bytes 必须为非负整数（B），非 "
                             f"{self.extra_overhead_bytes}")
        if self.extra_overhead_bytes > 0 and not self.overhead_source:
            raise ValueError("extra_overhead_bytes>0 必须带 overhead_source")
        return self

    def to_json(self) -> dict:
        return dict(self.__dict__)


@dataclass
class DeploymentSpec:
    mode: str = "colocated"                # baseline | colocated | dt_serial
    target_device: str = "h100_sxm"
    target_tp: int = 8
    target_ep: int = 1                     # R01：1（TP 切 inter）或 =tp（EP 整专家）
    draft_device: str = "l40"              # dt_serial 下草稿独立设备
    draft_tp: int = 1                      # dt_serial：独立组 TP；colocated 草稿同 target 组
    link: str = "pcie5"
    draft_on_target_group: bool = True     # colocated：草稿在目标组（与 target 同 TP，vLLM 配方）
    pipeline_microbatches: int = 2         # dt_pipeline_sim：微批数
    pipeline_rounds: int = 6               # 有限 cohort（warmup/drain 可见）
    budget: Optional[BudgetInput] = None            # target/colocated 组预算（R03a）
    draft_budget: Optional[BudgetInput] = None      # dt_serial 草稿组预算


@dataclass
class ScenarioV2:
    schema_version: str = "0.2"
    name: str = "unnamed"
    model_id: str = "qwen36_35b_a3b"
    draft_id: Optional[str] = "dflash_qwen36_35b_a3b"
    method_id: str = "dflash_v1"           # baseline | chain_draft | mtp | dflash_v1 | chain_moe_proxy
    k: int = 4                             # 链式 K 上限（块方法取 min(blk−1,k)）
    block_size: Optional[int] = None       # dflash 覆盖（默认 config 16）
    deployment: DeploymentSpec = field(default_factory=DeploymentSpec)
    workload: WorkloadHypothesis = field(default_factory=WorkloadHypothesis)
    acceptance: Dict[str, Any] = field(
        default_factory=lambda: {"kind": "geometric", "alpha": 0.75})
    quant: Dict[str, int] = field(
        default_factory=lambda: {"dense_w": 8, "expert_w": 4, "act": 8, "kv": 8,
                                 "state": 32})
    draft_quant: Dict[str, int] = field(
        default_factory=lambda: {"dense_w": 8, "expert_w": 8, "act": 8, "kv": 8,
                                 "state": 32})
    # DFlash ctx-KV 的执行路径（显式配方；dense_w 只是计价位宽，不决定分支）。
    # 默认 fused_raw 对应本仓活跃草稿 config 无 quantization_config + GPU
    # UnquantizedLinearMethod（pinned qwen3_dflash.py L108-113），不是从
    # dense_w=8 推出来的；量化/回退场景必须显式覆盖。
    draft_ctx_kv_path: str = FUSED_RAW
    cost_profile: Optional[str] = None   # profile 集 id（如 h200-qwen3-30b-a3b-tiny-op-v1）
    seed: int = 0
    notes: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.deployment, dict):
            self.deployment = _deploy_from(self.deployment)
        if isinstance(self.workload, dict):
            self.workload = WorkloadHypothesis(**self.workload)

    def validate(self) -> "ScenarioV2":
        if self.deployment.mode not in DEPLOY_MODES:
            raise ValueError(f"未知部署模式 {self.deployment.mode}")
        if self.method_id == "baseline" and self.deployment.mode != "baseline":
            raise ValueError("baseline 方法必须配 baseline 部署")
        if self.method_id != "baseline" and self.deployment.mode == "baseline":
            raise ValueError("非 baseline 方法不能用 baseline 部署（对照组请显式建 run）")
        if not (1 <= self.k <= 32):
            raise ValueError("k ∈ [1,32]")
        if self.workload.batch < 1 or self.workload.ctx_tokens < 1:
            raise ValueError("batch/ctx 必须为正")
        dep = self.deployment
        if dep.target_tp < 1 or dep.target_ep < 1 or dep.draft_tp < 1:
            raise ValueError("tp/ep 必须为正整数")
        if dep.target_ep not in (1, dep.target_tp):
            raise ValueError(
                f"target_ep={dep.target_ep} 无实现配方：单引擎副本内只能取 1 或 "
                f"等于 target_tp={dep.target_tp}（vLLM FusedMoEParallelConfig pin）")
        if dep.mode == "colocated" and dep.draft_tp != 1 and not dep.draft_on_target_group:
            raise ValueError("colocated 草稿与 target 同 TP 组（vLLM 配方）；"
                             "draft_tp 仅在 dt_serial 独立组语义下有效")
        if dep.budget is not None:
            dep.budget.validate()
        if dep.draft_budget is not None:
            dep.draft_budget.validate()
        for key, bits in {**self.quant, **self.draft_quant}.items():
            if key != "state" and bits not in (4, 8, 16):
                raise ValueError(f"精度 {key}={bits} 不在 4/8/16")
        path = normalize_ctx_kv_path(self.draft_ctx_kv_path)
        if path == UNKNOWN:
            raise ValueError(
                "draft_ctx_kv_path=unknown：ctx-KV 实现路径必须有证据，"
                "不能由 dense_w 推断；请显式指定 fused_raw/"
                "mxfp8_context_proj/per_layer_fallback")
        # 一致性护栏（不是用位宽选分支）：raw 融合副本要求未量化权重；
        # 4bit 计价档与 fused_raw 矛盾，必须显式改为 per_layer_fallback。
        if path == FUSED_RAW and (self.draft_quant or {}).get("dense_w", 16) < 8:
            raise ValueError(
                f"draft_ctx_kv_path=fused_raw 与 draft_quant.dense_w="
                f"{(self.draft_quant or {}).get('dense_w')} 矛盾：raw 路径要求 "
                "UnquantizedLinearMethod；量化回退请显式设置 "
                "draft_ctx_kv_path=per_layer_fallback（或 MXFP8 用 "
                "mxfp8_context_proj）")
        self.draft_ctx_kv_path = path
        return self

    def rng(self) -> RngScheme:
        return RngScheme(self.seed, semantic_keys={
            "model": self.model_id, "method": self.method_id,
            "acceptance": str(sorted(self.acceptance.items()))})

    def to_json(self) -> dict:
        out = asdict(self)
        out["rng"] = self.rng().to_json()
        return out

    def scenario_digest(self) -> str:
        return digest(self.to_json())

    @staticmethod
    def from_json(data: dict) -> "ScenarioV2":
        if data.get("schema_version") not in (None, "0.2"):
            raise ValueError(f"不支持 schema_version {data.get('schema_version')}")
        known = {f: data[f] for f in ScenarioV2.__dataclass_fields__ if f in data}
        if "deployment" in known:
            known["deployment"] = (known["deployment"] if not isinstance(
                known["deployment"], dict) else _deploy_from(known["deployment"]))
        if "workload" in known and isinstance(known["workload"], dict):
            known["workload"] = WorkloadHypothesis(**known["workload"])
        sc = ScenarioV2(**known)
        return sc.validate()


def _deploy_from(d: dict) -> DeploymentSpec:
    if "budget" in d and isinstance(d["budget"], dict):
        d = {**d, "budget": BudgetInput(**d["budget"]).validate()}
    if "draft_budget" in d and isinstance(d["draft_budget"], dict):
        d = {**d, "draft_budget": BudgetInput(**d["draft_budget"]).validate()}
    return DeploymentSpec(**d)
