"""资源约束事件核（M3）：两微批 DT 流水与单请求超前分开建模。

机制（principles §8）：
  * 事件带资源需求；同资源互斥（容量 1 默认）；链路为半双工资源；
  * 确定性最早开始调度：start = max(依赖完成, 资源空闲时刻)——非预知策略
    只读当前可见信息（依赖完成时刻与资源队列）；
  * 有限 cohort：warmup/drain 可见；稳态周期由完成时刻差实测（不套 max 式）；
  * 单请求 TPOT 与吞吐周期分列（M18）：不把周期当请求延迟。
输出事件流经 execute.validate.validate_trace 校验（同资源重叠/依赖序负例
由验证器捕获），指标经 execute.reduce 从事件独立重算。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .validate import TraceEvent, validate_trace


@dataclass
class TaskSpec:
    task_id: str
    resource: str
    duration_ms: float
    depends_on: List[str] = field(default_factory=list)
    consumes_messages: List[str] = field(default_factory=list)
    is_message_send: bool = False


@dataclass
class ScheduleResult:
    events: List[TraceEvent]
    violations: List[Tuple[str, str, str]]
    makespan_ms: float
    cohort_periods_ms: Dict[str, float]     # cohort → 完成间隔（相邻轮）
    resource_busy_ms: Dict[str, float]
    utilization: Dict[str, float]

    def to_json(self) -> dict:
        return {"events": [e.__dict__ for e in self.events],
                "violations": [list(v) for v in self.violations],
                "makespan_ms": self.makespan_ms,
                "cohort_periods_ms": self.cohort_periods_ms,
                "resource_busy_ms": self.resource_busy_ms,
                "utilization": self.utilization}


def schedule_earliest(tasks: List[TaskSpec],
                      resources: Optional[Dict[str, float]] = None) -> ScheduleResult:
    """确定性最早开始调度（拓扑序处理；资源容量 1 互斥）。

    resources: resource → capacity（默认全 1；本核不建容量>1 的份额模型，
    需要 share 时显式拆资源 id）。
    """
    capacity = dict(resources or {})
    by_id = {t.task_id: t for t in tasks}
    # 拓扑排序（Kahn；环→报错）
    indeg = {t.task_id: len(t.depends_on) for t in tasks}
    children: Dict[str, List[str]] = {}
    for t in tasks:
        for d in t.depends_on:
            if d not in by_id:
                raise ValueError(f"{t.task_id} 依赖不存在的 {d}")
            children.setdefault(d, []).append(t.task_id)
    queue = sorted([t.task_id for t in tasks if indeg[t.task_id] == 0])
    order: List[str] = []
    while queue:
        tid = queue.pop(0)
        order.append(tid)
        for c in children.get(tid, []):
            indeg[c] -= 1
            if indeg[c] == 0:
                queue.append(c)
        queue.sort()
    if len(order) != len(tasks):
        raise ValueError("任务依赖存在环")

    end_at: Dict[str, float] = {}
    res_free: Dict[str, float] = {}
    events: List[TraceEvent] = []
    for tid in order:
        t = by_id[tid]
        ready = max((end_at[d] for d in t.depends_on), default=0.0)
        start = max(ready, res_free.get(t.resource, 0.0))
        end = start + t.duration_ms
        end_at[tid] = end
        res_free[t.resource] = end
        events.append(TraceEvent(
            event_id=tid, role=_role_of(t), resource_id=t.resource,
            start_ms=start, end_ms=end, depends_on=list(t.depends_on),
            consumes_messages=list(t.consumes_messages),
            is_message_send=t.is_message_send))
    events.sort(key=lambda e: (e.start_ms, e.event_id))
    violations = [(v.check, v.event_id, v.detail) for v in validate_trace(events)]
    makespan = max((e.end_ms for e in events), default=0.0)
    busy: Dict[str, float] = {}
    for e in events:
        busy[e.resource_id] = busy.get(e.resource_id, 0.0) + (e.end_ms - e.start_ms)
    util = {r: (b / makespan if makespan > 0 else 0.0) for r, b in busy.items()}
    return ScheduleResult(events, violations, makespan, {}, busy, util)


def _role_of(t: TaskSpec) -> str:
    if "link" in t.resource or t.is_message_send:
        return "link"
    if "draft" in t.resource or "draft" in t.task_id:
        return "draft"
    return "target"


def dt_pipeline_schedule(t_draft_ms: float, t_d2t_ms: float, t_verify_ms: float,
                         t_t2d_ms: float, rounds: int, microbatches: int = 2
                         ) -> ScheduleResult:
    """两微批 DT 流水（M3 核心用例）。

    资源：draft-gpu（互斥）、target-gpu（互斥）、link（半双工：同一时刻只
    一个方向的消息）。轮 i 微批 b：draft(i,b) → d2t(i,b) → verify(i,b) →
    t2d(i,b) → 下一轮同微批 draft 依赖其 t2d 反馈。
    """
    tasks: List[TaskSpec] = []
    for i in range(rounds):
        for b in range(microbatches):
            d = f"r{i}b{b}.draft"
            m1 = f"r{i}b{b}.d2t"
            v = f"r{i}b{b}.verify"
            m2 = f"r{i}b{b}.t2d"
            prev = [f"r{i - 1}b{b}.t2d"] if i > 0 else []
            tasks.append(TaskSpec(d, "draft-gpu", t_draft_ms, depends_on=prev))
            tasks.append(TaskSpec(m1, "link", t_d2t_ms, depends_on=[d],
                                  is_message_send=True))
            tasks.append(TaskSpec(v, "target-gpu", t_verify_ms, depends_on=[m1],
                                  consumes_messages=[m1]))
            tasks.append(TaskSpec(m2, "link", t_t2d_ms, depends_on=[v],
                                  is_message_send=True))
    res = schedule_earliest(tasks)
    # cohort 周期：每微批相邻轮 verify 完成时刻差（稳态段取中位差）
    periods: Dict[str, float] = {}
    for b in range(microbatches):
        ends = [e.end_ms for e in res.events
                if e.event_id.endswith(f"b{b}.verify")]
        ends.sort()
        if len(ends) >= 2:
            diffs = [ends[i + 1] - ends[i] for i in range(len(ends) - 1)]
            periods[f"b{b}"] = diffs[len(diffs) // 2]  # 中位（避开 warmup/drain）
    res.cohort_periods_ms = periods
    return res


def single_request_lookahead(t_draft_ms: float, t_d2t_ms: float, t_verify_ms: float,
                             t_t2d_ms: float, rounds: int) -> ScheduleResult:
    """单请求串行（超前=0 基线）：与流水对照；TPOT=轮和/E[N] 由上层计。"""
    return dt_pipeline_schedule(t_draft_ms, t_d2t_ms, t_verify_ms, t_t2d_ms,
                                rounds, microbatches=1)
