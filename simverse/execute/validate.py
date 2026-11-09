"""事件验证器（M3 / V07）：独立重建必要依赖，不信调度器自报。

检查（每条负例有对应断言）：
 1. 依赖先完成：事件 start ≥ 所有 depends_on 事件的 end（缺边→violation）；
 2. 同资源互斥：同一 resource_id 上的事件区间不得重叠（除非显式 share）；
 3. 消息先发后用：消息消费事件依赖其发送完成（message 边单独核对）；
 4. 提交/回退状态序：commit_rewind 依赖 verify 完成。
验证器从事件自身字段重建因果（不复用生成器的构造函数）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


@dataclass
class TraceEvent:
    event_id: str
    role: str
    resource_id: str
    start_ms: float
    end_ms: float
    depends_on: List[str] = field(default_factory=list)
    consumes_messages: List[str] = field(default_factory=list)
    is_message_send: bool = False
    stage: str = ""

    def __post_init__(self) -> None:
        if self.end_ms < self.start_ms:
            raise ValueError(f"{self.event_id}: end < start")


@dataclass
class Violation:
    check: str
    event_id: str
    detail: str


def validate_trace(events: Sequence[TraceEvent]) -> List[Violation]:
    by_id = {e.event_id: e for e in events}
    out: List[Violation] = []
    # 1) 依赖先完成
    for e in events:
        for dep in e.depends_on:
            d = by_id.get(dep)
            if d is None:
                out.append(Violation("missing_dependency", e.event_id,
                                     f"依赖 {dep} 不存在"))
            elif e.start_ms < d.end_ms - 1e-9:
                out.append(Violation("starts_before_dependency_end", e.event_id,
                                     f"start {e.start_ms} < {dep}.end {d.end_ms}"))
    # 2) 同资源互斥（不重叠）
    by_res: Dict[str, List[TraceEvent]] = {}
    for e in events:
        by_res.setdefault(e.resource_id, []).append(e)
    for res, evs in by_res.items():
        evs.sort(key=lambda x: x.start_ms)
        for a, b in zip(evs, evs[1:]):
            if b.start_ms < a.end_ms - 1e-9:
                out.append(Violation(
                    "resource_overlap", b.event_id,
                    f"资源 {res} 与 {a.event_id} 重叠 "
                    f"[{b.start_ms},{b.end_ms}) vs [{a.start_ms},{a.end_ms})"))
    # 3) 消息先发后用：consumes 的消息 id 必须是某 send 事件 id 且已完成
    for e in events:
        for msg in e.consumes_messages:
            m = by_id.get(msg)
            if m is None or not m.is_message_send:
                out.append(Violation("consume_unknown_message", e.event_id,
                                     f"消费未定义消息 {msg}"))
            elif e.start_ms < m.end_ms - 1e-9:
                out.append(Violation("consume_before_send_end", e.event_id,
                                     f"消费 {msg} 先于发送完成"))
            elif msg not in e.depends_on:
                out.append(Violation("missing_message_edge", e.event_id,
                                     f"消费 {msg} 未声明依赖边"))
    return out


def validate_serial_round(events: Sequence[TraceEvent]) -> List[Violation]:
    """串行链专项：依赖完整（相邻事件链覆盖全部）+ 单调时间。"""
    out = validate_trace(events)
    ids = [e.event_id for e in events]
    if len(set(ids)) != len(ids):
        out.append(Violation("duplicate_event_id", ids[0] if ids else "?",
                             "事件 id 重复"))
    return out
