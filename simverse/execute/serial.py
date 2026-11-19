"""串行执行器（Execute v0.2）：MethodPlan → 带依赖 ID 的轮事件 + 解析指标。

串行口径（plan M2）：T_round = Σ 阶段（无重叠）；TPOT = E[T_round]/E[N]。
事件带 event_id/depends_on/stage/role，供三轨时间轴与 M3 事件核扩展
（M3 加入资源约束后本解析值作为同假设对照：串行事件与解析相等 V07）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class RoundEvent:
    event_id: str
    round_id: str
    role: str                 # target | draft | link | commit
    stage: str
    label: str
    t_ms: Optional[float]
    depends_on: List[str] = field(default_factory=list)
    start_ms: Optional[float] = 0.0
    detail: Dict[str, Any] = field(default_factory=dict)
    reason: Optional[str] = None
    start_lower_bound_ms: Optional[float] = None

    def to_json(self) -> dict:
        event = dict(self.__dict__)
        for key in ("reason", "start_lower_bound_ms"):
            if event[key] is None:
                del event[key]
        return event


@dataclass
class SerialRound:
    round_id: str
    events: List[RoundEvent]
    t_round_ms: Optional[float]
    e_n: Optional[float]
    tpot_ms: Optional[float]           # None 当 E[N] unknown（动态 K 等缺律）
    reason: Optional[str] = None

    timing_gap: Optional[str] = None

    @property
    def timing_complete(self) -> bool:
        return self.t_round_ms is not None and not (self.unpriced_stages or self.timing_gap)

    @property
    def unpriced_stages(self) -> list[str]:
        return [event.stage for event in self.events if event.t_ms is None]

    def to_json(self) -> dict:
        result = {"round_id": self.round_id,
                  "events": [e.to_json() for e in self.events],
                  "t_round_ms": self.t_round_ms, "E_N": self.e_n,
                  "tpot_ms": self.tpot_ms, "reason": self.reason}
        if not self.timing_complete:
            result["timing_complete"] = False
        return result


def evaluate_execution(plan, stage_costs, e_n, deployment, *,
                       acceptance_reason=None, timing_gap=None):
    """One execution policy for analytical and selected costs; no model recount."""
    if e_n is None:
        return SerialRound("r0", [], None, None, None, acceptance_reason or "E[N] unknown"), None
    serial = build_serial_round(plan, stage_costs, e_n, timing_gap=timing_gap)
    simulation = None
    if deployment.mode == "dt_pipeline_sim":
        from .resource_core import dt_pipeline_schedule
        draft = sum(v for k,v in stage_costs.items() if k == "draft_block" or k.startswith("draft_step_"))
        simulation = dt_pipeline_schedule(draft, stage_costs.get("d2t", 0.0),
            stage_costs.get("verify_block", stage_costs.get("verify", 0.0)), stage_costs.get("t2d", 0.0),
            rounds=deployment.pipeline_rounds, microbatches=deployment.pipeline_microbatches).to_json()
        if not serial.timing_complete:
            simulation["timing_complete"] = False
            omitted = serial.unpriced_stages
            simulation["excluded_stages"] = omitted
            stage_reason = "仅含已计价阶段的理想流水模拟，省略未计价步骤" if omitted else None
            simulation["timing_reason"] = "；".join(filter(None, (stage_reason, timing_gap)))
    return serial, simulation


def build_serial_round(plan, stage_costs: Dict[str, Optional[float]], e_n: Optional[float],
                       reason: Optional[str] = None, *,
                       timing_gap: Optional[str] = None) -> SerialRound:
    """把 MethodPlan 步骤 + 各阶段计价串成一条因果链。

    stage_costs: step_id → ms；None 表示时长未知，0 表示已知为零，缺失报错。
    遇到未知时长后仅累加已计价的阶段，保留后续步骤的开始时间下界。
    """
    events: List[RoundEvent] = []
    t_acc = 0.0
    prefix_known = True
    prev_id: Optional[str] = None
    for step in plan.steps:
        if step.step_id not in stage_costs:
            raise ValueError(f"缺少阶段成本：{step.step_id}")
        price = stage_costs[step.step_id]
        t = float(price) if price is not None else None
        stage_gap = timing_gap if step.role == "target" else None
        ev = RoundEvent(
            event_id=f"r0.{step.step_id}", round_id="r0", role=step.role,
            stage=step.step_id, label=step.desc, t_ms=t,
            depends_on=[prev_id] if prev_id else [],
            start_ms=t_acc if prefix_known else None,
            detail={"tokens": step.tokens},
            reason="阶段未单独计价，时长未知" if t is None else stage_gap,
            start_lower_bound_ms=t_acc if not prefix_known else None)
        events.append(ev)
        if t is not None:
            t_acc += t
        if t is None or stage_gap:
            prefix_known = False
        prev_id = ev.event_id
    tpot = (t_acc / e_n) if (e_n is not None and e_n > 0) else None
    if reason is None:
        if e_n is None:
            reason = "E[N] unknown：见 acceptance reason"
        elif any(event.t_ms is None for event in events):
            reason = "存在未计价阶段，轮耗时与 TPOT 为已计价阶段下界"
    if timing_gap:
        reason = "；".join(filter(None, (reason, timing_gap)))
    return SerialRound("r0", events, t_acc, e_n, tpot, reason, timing_gap)
