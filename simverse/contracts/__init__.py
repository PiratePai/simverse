"""contracts v0.2：版本化类型、单位、状态、来源、canonical serialization。

v0.1 的扁平 contracts.py（RunResult 等）保留给 legacy 切片；本包是新管线的
唯一契约入口。两者由 schema_version 区分（0.1 / 0.2），互不冒充。
"""
from .enums import (Coverage, Evidence, Feasibility, Phase, ResultKind, Statistic,
                    Support, TimeBasis)
from .identity import (RngScheme, canonical_json, digest, finite, js_restore,
                       js_safe, sha256_hex, short)
from .metrics import Metric, Quantity, Scope, metric_from
from .run import ResultBundle, RunManifest, STATUS
from .evidence import CapabilityReport, EvidenceRecord, FieldCoverageEntry

# v0.1 数据类继续可导入（legacy 切片与既有测试）；其 digest 是 16-hex 截断版，
# 与 v0.2 全长 digest 不同名共存：legacy 路径请显式 from .contracts.legacy import。
from .legacy import (AcceptanceSpec, DeviceSpec, DraftSpec, LayerSpec, LinkSpec,  # noqa: E402,F401
                     ModelSpec, OpRecord, RunResult, ScenarioSpec, StageTiming,
                     TimelineEvent)

SCHEMA_VERSION = "0.2"
LEGACY_SCHEMA_VERSION = "0.1"

__all__ = [
    "SCHEMA_VERSION", "LEGACY_SCHEMA_VERSION", "Coverage", "Evidence", "Feasibility",
    "Phase", "ResultKind", "Statistic", "Support", "TimeBasis", "RngScheme",
    "canonical_json", "digest", "finite", "js_restore", "js_safe", "sha256_hex",
    "short", "Metric", "Quantity", "Scope", "metric_from",
    "ResultBundle", "RunManifest", "STATUS",
    "CapabilityReport", "EvidenceRecord", "FieldCoverageEntry",
    "AcceptanceSpec", "DeviceSpec", "DraftSpec", "LayerSpec", "LinkSpec",
    "ModelSpec", "OpRecord", "RunResult", "ScenarioSpec", "StageTiming",
    "TimelineEvent",
]
