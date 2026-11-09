"""正交状态枚举（计划 §4：四个维度不得揉成一个）。

support       该结论的实现/证据支持状态
evidence      证据类型（可叠加，逐字段记录）
feasibility   资源可行性判断
result_kind   数值的语义身份（解析估算 / 界 / 事件仿真 / 实测）
statistic     数值的统计身份（精确 / 期望 / 采样 / 分位 / 界）
coverage      配置字段的消费状态（M0 硬门槛）
time_basis    时间口径（Q12：committed 与 server_emitted 分列）
"""
from __future__ import annotations

from enum import Enum


class Support(str, Enum):
    SUPPORTED = "supported"
    APPROXIMATED = "approximated"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"          # 数据缺失（区别于已确定不存在）
    UNMEASURED = "unmeasured"    # 接口在、无实测数据（Q2 首版性能口径）


class Evidence(str, Enum):
    CONFIG = "config"                  # 官方 config 快照（身份已核）
    SOURCE_CHECKED = "source-checked"  # 源码阅读（未执行）
    SHAPE_CHECKED = "shape-checked"    # 小尺寸/meta shape 执行
    NUMERIC_CHECKED = "numeric-checked"  # 小尺寸数值 vs 独立参考
    COUNTED = "counted"                # 逻辑计数（可手算复核）
    MEASURED = "measured"              # 真机实测
    FITTED = "fitted"                  # 拟合
    SPEC = "spec"                      # 公开规格
    ASSUMED = "assumed"                # 显式假设


class Feasibility(str, Enum):
    FEASIBLE = "feasible"
    OOM = "oom"
    INVALID = "invalid"
    UNKNOWN = "unknown"


class ResultKind(str, Enum):
    ANALYTICAL_ESTIMATE = "analytical-estimate"
    OPTIMISTIC_BOUND = "optimistic-bound"
    EVENT_SIMULATION = "event-simulation"
    MEASUREMENT = "measurement"


class Statistic(str, Enum):
    EXACT = "exact"            # 精确整数/确定值
    EXPECTATION = "expectation"  # 期望（可为浮点；禁止 int() 截断冒充实数）
    SAMPLE = "sample"
    QUANTILE = "quantile"
    BOUND = "bound"


class Coverage(str, Enum):
    """DSH 字段覆盖门槛：每个配置键必须登记其一。"""

    MAPPED = "mapped"                        # 已进入结构/计数语义
    METADATA_ONLY = "metadata-only"          # 仅登记，不影响计数（写明为何安全）
    IGNORED_WITH_REASON = "ignored-with-reason"
    UNSUPPORTED_SEMANTIC = "unsupported-semantic"  # 缺失即降级相关结论
    PENDING_ASSET = "pending-asset"          # 资产身份待核实（如实标注）


class TimeBasis(str, Enum):
    COMPUTED = "computed"
    ACCEPTED = "accepted"
    COMMITTED = "committed"          # 服务端提交口径（Q12 首版）
    SERVER_EMITTED = "server_emitted"  # 服务端发出（无 flush 模型时缺）
    CLIENT_OBSERVED = "client_observed"  # 后续扩展


class Phase(str, Enum):
    PREFILL = "prefill"
    FIRST_ROUND = "first-round"
    STEADY_DECODE = "steady-decode"
    TAIL_ROUND = "tail-round"
    DRAIN = "drain"
