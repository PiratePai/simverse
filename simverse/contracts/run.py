"""Run 身份与 ResultBundle v0.2 骨架（ui-data-contract §3）。

run_id = 完整身份哈希（输入 + ModelIR + method + lowering/cost 版本 + 硬件快照
+ 代码 revision + 随机流），不是时间戳；相同 scenario_digest 不足以重放相同。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .identity import digest
from .metrics import metrics_to_json

STATUS = ("queued", "running", "completed", "failed", "cancelled")


@dataclass
class RunManifest:
    run_id: str
    status: str = "queued"
    input_digest: str = ""            # 场景（含假设与随机身份）
    model_ir_digest: str = ""
    method_digest: str = ""
    lowering_digest: str = ""
    cost_digest: str = ""
    device_snapshot_digest: str = ""
    code_revision: str = ""
    parent_run_id: Optional[str] = None
    created_at: str = ""
    progress: float = 0.0
    error: str = ""

    def __post_init__(self) -> None:
        if self.status not in STATUS:
            raise ValueError(f"未知 run 状态 {self.status}")

    def compute_run_id(self) -> str:
        basis = {k: v for k, v in self.__dict__.items()
                 if k not in ("run_id", "status", "created_at", "progress", "error",
                              "parent_run_id")}
        self.run_id = digest(basis)
        return self.run_id

    def to_json(self) -> dict:
        return dict(self.__dict__)


@dataclass
class ResultBundle:
    """一次计算的不可变快照：引用各模块产物 + 指标 + 能力 + 证据/公式字典。

    schema_version 0.2；v0.1 RunResult 由 legacy 模块继续承担，互不冒充。
    """
    manifest: RunManifest
    scenario_snapshot: Dict[str, Any] = field(default_factory=dict)
    model_ref: Dict[str, Any] = field(default_factory=dict)      # ModelIR 摘要 + digest
    method_plan: Dict[str, Any] = field(default_factory=dict)
    lowered_ref: Dict[str, Any] = field(default_factory=dict)
    work: Dict[str, Any] = field(default_factory=dict)           # 逻辑计数
    traffic: Dict[str, Any] = field(default_factory=dict)        # 期望/物理流量
    memory: Dict[str, Any] = field(default_factory=dict)         # 容量对象/峰值
    cost: Dict[str, Any] = field(default_factory=dict)
    execution: Dict[str, Any] = field(default_factory=dict)      # 事件/时间线
    metrics: Dict[str, Any] = field(default_factory=dict)        # Metric 注册表
    capability: Dict[str, Any] = field(default_factory=dict)
    evidence_refs: List[str] = field(default_factory=list)
    formula_dict: Dict[str, Any] = field(default_factory=dict)   # 去重公式字典
    assumption_dict: Dict[str, str] = field(default_factory=dict)
    comparisons: List[Dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "schema_version": "0.2",
            "manifest": self.manifest.to_json(),
            "scenario_snapshot": self.scenario_snapshot,
            "model_ref": self.model_ref,
            "method_plan": self.method_plan,
            "lowered_ref": self.lowered_ref,
            "work": self.work,
            "traffic": self.traffic,
            "memory": self.memory,
            "cost": self.cost,
            "execution": self.execution,
            "metrics": metrics_to_json(self.metrics),
            "capability": self.capability,
            "evidence_refs": list(self.evidence_refs),
            "formula_dict": self.formula_dict,
            "assumption_dict": self.assumption_dict,
            "comparisons": self.comparisons,
        }

    def content_digest(self) -> str:
        return digest(self.to_json())
