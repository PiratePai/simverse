"""证据与能力报告契约（M0）。

EvidenceRecord 按 claim/domain/version 绑定（§8.1），验证状态绑定内容 hash：
资产改动 → 旧证据不自动继承。CapabilityReport 逐条件 status/reason/affected。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .enums import Coverage, Evidence


@dataclass
class EvidenceRecord:
    evidence_id: str
    kind: Evidence
    claim: str                      # 证明什么（写明不能证明什么的边界放 limits）
    source: str                     # repo/path/URL + revision
    content_sha256: str             # 绑定的内容身份
    captured_at: str
    domain: str = ""                # 适用域（模型/shape/分支）
    limits: List[str] = field(default_factory=list)   # 不能证明的范围
    executed: bool = False          # 是否实际执行（source-checked vs 运行）
    command: str = ""               # 复核命令（可重放）

    def to_json(self) -> dict:
        return dict(self.__dict__)


@dataclass
class FieldCoverageEntry:
    """单个配置键的消费状态（M0 硬门槛）。"""

    key: str
    status: Coverage
    consumer: str = ""              # 哪个模块消费它
    evidence_id: str = ""
    note: str = ""

    def to_json(self) -> dict:
        return {"key": self.key, "status": self.status.value, "consumer": self.consumer,
                "evidence_id": self.evidence_id, "note": self.note}


@dataclass
class CapabilityReport:
    """特性组合的显式检查：不支持的写原因与受影响指标，不静默降级。"""

    checks: Dict[str, Dict[str, object]] = field(default_factory=dict)

    def add(self, check_id: str, status: str, reason: str,
            affected_metrics: Optional[List[str]] = None) -> None:
        if status not in ("supported", "approximated", "unsupported", "unknown"):
            raise ValueError(f"未知能力状态 {status}")
        self.checks[check_id] = {"status": status, "reason": reason,
                                 "affected_metrics": affected_metrics or []}

    def unsupported_ids(self) -> List[str]:
        return [k for k, v in self.checks.items() if v["status"] != "supported"]

    def to_json(self) -> dict:
        return {"schema_version": "0.2", "checks": self.checks}
