"""串行执行器（Execute v0.2）：MethodPlan → 带依赖 ID 的轮事件 + 解析指标。

串行口径（plan M2）：T_round = Σ 阶段（无重叠）；TPOT = E[T_round]/E[N]。
事件带 event_id/depends_on/stage/role，供三轨时间轴与 M3 事件核扩展
（M3 加入资源约束后本解析值作为同假设对照：串行事件与解析相等 V07）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .. import formula_registry as FR


@dataclass
class RoundEvent:
    event_id: str
    round_id: str
    role: str                 # target | draft | link | commit
    stage: str
    label: str
    t_ms: float
    depends_on: List[str] = field(default_factory=list)
    start_ms: float = 0.0
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict:
        return dict(self.__dict__)


@dataclass
class SerialRound:
    round_id: str
    events: List[RoundEvent]
    t_round_ms: float
    e_n: Optional[float]
    tpot_ms: Optional[float]           # None 当 E[N] unknown（动态 K 等缺律）
    reason: Optional[str] = None

    def to_json(self) -> dict:
        return {"round_id": self.round_id,
                "events": [e.to_json() for e in self.events],
                "t_round_ms": self.t_round_ms, "E_N": self.e_n,
                "tpot_ms": self.tpot_ms, "reason": self.reason}


def build_serial_round(plan, stage_costs: Dict[str, float], e_n: Optional[float],
                       reason: Optional[str] = None) -> SerialRound:
    """把 MethodPlan 步骤 + 各阶段计价串成一条因果链。

    stage_costs: step_id → ms（由 pipeline 计价提供；缺失步骤计 0 并记录）。
    """
    events: List[RoundEvent] = []
    t_acc = 0.0
    prev_id: Optional[str] = None
    for step in plan.steps:
        t = float(stage_costs.get(step.step_id, 0.0))
        ev = RoundEvent(
            event_id=f"r0.{step.step_id}", round_id="r0", role=step.role,
            stage=step.step_id, label=step.desc, t_ms=t,
            depends_on=[prev_id] if prev_id else [],
            start_ms=t_acc,
            detail={"tokens": step.tokens})
        events.append(ev)
        t_acc += t
        prev_id = ev.event_id
    tpot = (t_acc / e_n) if (e_n is not None and e_n > 0) else None
    return SerialRound("r0", events, t_acc, e_n, tpot,
                       reason or (None if e_n is not None else
                                  "E[N] unknown：见 acceptance reason"))
