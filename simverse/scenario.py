"""场景契约 v0.2：显式假设快照（Q8 hypothesis / Q10 单任务规模）。

与 v0.1 ScenarioSpec 的差异：
  * 负载/接受率作为显式假设对象（source_kind=hypothesis 登记来源）；
  * 部署显式角色化（target/draft 设备与 TP、链路），不再用 mode 字符串隐含；
  * 精度配方独立（storage/compute 语义在 cost 路由，不在此冒充）。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

from .contracts import RngScheme, digest

DEPLOY_MODES = ("baseline", "colocated", "dt_serial", "dt_pipeline_sim")


@dataclass
class WorkloadHypothesis:
    batch: int = 1
    ctx_tokens: int = 32768
    osl_tokens: Optional[int] = None       # 输出上限（尾轮截断用；未设=未知）
    arrival: str = "fixed-batch"           # v0.2：固定批；到达过程 M4
    source_kind: str = "hypothesis"
    notes: str = "Q8：首版显式假设扫描；具体数值为可编辑演示默认值，非用户确认"


@dataclass
class DeploymentSpec:
    mode: str = "colocated"                # baseline | colocated | dt_serial
    target_device: str = "h100_sxm"
    target_tp: int = 8
    draft_device: str = "l40"              # dt_serial 下草稿独立设备
    draft_tp: int = 1
    link: str = "pcie5"
    draft_on_target_group: bool = True     # colocated：草稿占目标组 1 rank TP1
    pipeline_microbatches: int = 2         # dt_pipeline_sim：微批数
    pipeline_rounds: int = 6               # 有限 cohort（warmup/drain 可见）


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
    seed: int = 0
    notes: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.deployment, dict):
            self.deployment = DeploymentSpec(**self.deployment)
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
        for key, bits in {**self.quant, **self.draft_quant}.items():
            if key != "state" and bits not in (4, 8, 16):
                raise ValueError(f"精度 {key}={bits} 不在 4/8/16")
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
        if "deployment" in known and isinstance(known["deployment"], dict):
            known["deployment"] = DeploymentSpec(**known["deployment"])
        if "workload" in known and isinstance(known["workload"], dict):
            known["workload"] = WorkloadHypothesis(**known["workload"])
        sc = ScenarioV2(**known)
        return sc.validate()
