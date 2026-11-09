"""Metric 契约（ui-data-contract §3）：每个关键数字的最小完备元组。

规则：
  * 无值 → value=None + reason 必填；已确定不存在才是 0；
  * support/feasibility/result_kind/statistic 四维正交，不合成单一可信度；
  * 期望值 statistic=expectation，保留浮点；禁止把期望 int() 后当 exact；
  * scope 记录 phase/role/rank/request/时间窗等聚合边界；
  * 公式/输入/假设/证据用引用 ID（去重字典，行内只挂引用）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .enums import Evidence, Feasibility, ResultKind, Statistic, Support, TimeBasis
from .identity import finite


@dataclass
class Scope:
    """数值的作用域与聚合边界。"""

    phase: str = "steady-decode"       # prefill/first-round/steady-decode/tail-round/drain
    role: str = "target"               # target | draft | link | cpu
    rank: Optional[str] = None         # "r0"/"avg"/"max"…
    group: Optional[str] = None        # tp-group / ep-group / device-group id
    request_id: Optional[str] = None
    round_id: Optional[str] = None
    window: Optional[str] = None       # "[start,end)" 或 "full-run"
    aggregation: str = "per-run"       # per-run/per-rank/per-request/per-token
    time_basis: Optional[str] = None   # committed / server_emitted / ...

    def to_json(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class Metric:
    value: Optional[float]
    unit: str
    scope: Scope = field(default_factory=Scope)
    statistic: Statistic = Statistic.EXACT
    result_kind: ResultKind = ResultKind.ANALYTICAL_ESTIMATE
    support: Support = Support.SUPPORTED
    feasibility: Feasibility = Feasibility.FEASIBLE
    evidence_refs: List[str] = field(default_factory=list)
    formula_ref: Optional[str] = None
    input_refs: List[str] = field(default_factory=list)
    assumption_refs: List[str] = field(default_factory=list)
    reason: Optional[str] = None       # value=None 时必填；有值时可注缺口
    interval: Optional[Dict[str, Any]] = None  # {kind,method,level,bounds,assumptions}
    sample: Optional[Dict[str, Any]] = None    # sample_metadata（§7.7）

    def __post_init__(self) -> None:
        if self.value is None:
            if not self.reason:
                raise ValueError("Metric 无值时必须给 reason（未知/不支持/未配置 ≠ 0）")
        else:
            finite(self.value, "Metric.value")
            if self.statistic in (Statistic.EXACT,) and isinstance(self.value, float):
                if self.value != int(self.value):
                    raise ValueError(
                        f"statistic=exact 但值为非整浮点 {self.value}；期望量请用 expectation")
        # 大整数按 int 保存（identity.canonical_json 原样序列化）

    def to_json(self) -> dict:
        return {
            "value": self.value,
            "unit": self.unit,
            "scope": self.scope.to_json(),
            "statistic": self.statistic.value,
            "result_kind": self.result_kind.value,
            "support": self.support.value,
            "feasibility": self.feasibility.value,
            "evidence_refs": list(self.evidence_refs),
            "formula_ref": self.formula_ref,
            "input_refs": list(self.input_refs),
            "assumption_refs": list(self.assumption_refs),
            "reason": self.reason,
            "interval": self.interval,
            "sample": self.sample,
        }

    @staticmethod
    def unknown(unit: str, reason: str, **kw) -> "Metric":
        return Metric(value=None, unit=unit, support=Support.UNKNOWN,
                      feasibility=Feasibility.UNKNOWN, reason=reason, **kw)

    @staticmethod
    def unsupported(unit: str, reason: str, **kw) -> "Metric":
        return Metric(value=None, unit=unit, support=Support.UNSUPPORTED,
                      reason=reason, **kw)


@dataclass
class Quantity:
    """账本行上的量：数值 + 统计身份 + 公式引用（OpRecord 的字段载体）。

    exact 整数或 expectation 浮点；weight_read 等期望字节禁止静默取整。
    """
    value: float
    unit: str
    statistic: Statistic
    formula_ref: str
    input_refs: List[str] = field(default_factory=list)
    note: str = ""

    def __post_init__(self) -> None:
        finite(self.value, "Quantity.value")
        if self.statistic is Statistic.EXPECTATION and isinstance(self.value, int):
            raise TypeError("期望量必须是浮点（防 int 冒充精确；请写 float(x)）")

    def to_json(self) -> dict:
        return {"value": self.value, "unit": self.unit,
                "statistic": self.statistic.value, "formula_ref": self.formula_ref,
                "input_refs": self.input_refs, "note": self.note}


def metric_from(value, unit: str, formula: str, *, statistic: Statistic,
                result_kind: ResultKind = ResultKind.ANALYTICAL_ESTIMATE,
                support: Support = Support.SUPPORTED,
                evidence: tuple = (Evidence.COUNTED,),
                scope: Scope | None = None, inputs: List[str] | None = None,
                assumptions: List[str] | None = None) -> Metric:
    return Metric(value=value, unit=unit,
                  scope=scope or Scope(), statistic=statistic, result_kind=result_kind,
                  support=support, evidence_refs=list(evidence), formula_ref=formula,
                  input_refs=inputs or [], assumption_refs=assumptions or [])


def metrics_to_json(obj):
    """递归把 Metric/Scope 对象转为 JSON 结构（ResultBundle.to_json 用）。"""
    if isinstance(obj, Metric):
        return obj.to_json()
    if isinstance(obj, dict):
        return {k: metrics_to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [metrics_to_json(v) for v in obj]
    return obj
