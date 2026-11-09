"""Profile 校准契约（Q2/§10）：接口先行，无数据时 unmeasured。

三对象（M2 契约 + 合成 fixture 验证接线；真实采集/拟合/holdout 属 M4）：
  ProfileRecord   一次原始测量：实现/recipe/版本、shape/dtype/设备栈、计时边界、
                  warmup/重复政策、样本统计身份
  CalibrationModel 版本化校准模型：训练/留出引用、覆盖域、误差报告、外推政策
  CostEstimate    单条成本：analytic_base 与 selected 分列、来源、fallback 原因

铁律：校准只改成本，不改 ModelIR/Work/Traffic/Memory 原始计数（§10）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..contracts import Evidence, ResultKind, Statistic, Support

TIMING_BOUNDARIES = ("device-kernel", "host-launch", "graph-piece", "end-to-end")


@dataclass(frozen=True)
class ProfileRecord:
    profile_id: str
    implementation_id: str          # 绑定 implementation/recipe/version
    recipe_version: str
    shape: Dict[str, int]
    dtype: str
    layout: str
    device: str                     # 设备/驱动/后端栈
    software_stack: str
    timing_boundary: str            # TIMING_BOUNDARIES 之一；不同边界不可互替
    warmup_runs: int
    repeat_runs: int
    sync_policy: str
    sample_unit: str
    samples_ref: str                # 原始样本产物引用（hash）
    content_sha256: str
    captured_at: str
    source_kind: str = "synthetic"  # synthetic（接线测试）| measured

    def __post_init__(self) -> None:
        if self.timing_boundary not in TIMING_BOUNDARIES:
            raise ValueError(f"计时边界 {self.timing_boundary} 不在 "
                             f"{TIMING_BOUNDARIES}")
        if self.source_kind == "measured" and self.warmup_runs < 0:
            raise ValueError("warmup 非法")

    def to_json(self) -> dict:
        return dict(self.__dict__)


@dataclass(frozen=True)
class CalibrationModel:
    calibration_id: str
    version: str
    content_sha256: str
    train_refs: List[str]
    holdout_refs: List[str]
    coverage: Dict[str, str]        # domain 描述 → in/out of coverage
    method: str                     # 拟合/插值方法（非统一乘法因子）
    error_report: Optional[Dict[str, float]] = None   # holdout 误差（M4 有数据才有）
    extrapolation_policy: str = "fallback-analytic"   # 域外回退解析，不外推
    fallback_reason_field: str = "out-of-coverage"

    def to_json(self) -> dict:
        return dict(self.__dict__)


@dataclass
class CostEstimate:
    """单条成本：analytic_base 与 selected 分列；校准不改原始计数。"""
    target_id: str                  # 行/kernel/消息 id
    analytic_base_ms: float
    selected_ms: Optional[float]
    cost_source: str                # analytic | profile | calibrated
    profile_ref: Optional[str] = None
    calibration_ref: Optional[str] = None
    coverage_status: str = "out-of-coverage"
    fallback_reason: Optional[str] = None
    unmeasured: bool = True         # Q2：首版全部 unmeasured

    def to_json(self) -> dict:
        return dict(self.__dict__)


def synthetic_profile_fixture() -> ProfileRecord:
    """合成 ProfileRecord（仅验证接线：字段/边界/身份；不是硬件测量）。"""
    return ProfileRecord(
        profile_id="synthetic-fixture-1", implementation_id="gemm-fp8-h100",
        recipe_version="r0", shape={"M": 4096, "N": 4096, "K": 4096},
        dtype="fp8", layout="row-major", device="h100-sxm@driver-5xx",
        software_stack="torch-2.x/cu12", timing_boundary="device-kernel",
        warmup_runs=3, repeat_runs=10, sync_policy="cuda-sync-each",
        sample_unit="kernel-invocation", samples_ref="sha256:synthetic",
        content_sha256="sha256:synthetic-fixture", captured_at="2026-11-09",
        source_kind="synthetic")


def apply_calibration(estimate_ms: float, record: ProfileRecord,
                      calibration: Optional[CalibrationModel],
                      domain_key: str) -> CostEstimate:
    """校准应用规则：域内且 measured 才 selected=calibrated；否则回退解析。"""
    if calibration is None:
        return CostEstimate(
            target_id="", analytic_base_ms=estimate_ms, selected_ms=None,
            cost_source="analytic", unmeasured=True,
            fallback_reason="无校准模型（Q2 首版解析估算）")
    in_cov = calibration.coverage.get(domain_key, "out")
    if in_cov != "in":
        return CostEstimate(
            target_id="", analytic_base_ms=estimate_ms, selected_ms=None,
            cost_source="analytic", calibration_ref=calibration.calibration_id,
            coverage_status=in_cov, unmeasured=True,
            fallback_reason=calibration.fallback_reason_field)
    if record.source_kind != "measured":
        return CostEstimate(
            target_id="", analytic_base_ms=estimate_ms, selected_ms=None,
            cost_source="analytic", profile_ref=record.profile_id,
            calibration_ref=calibration.calibration_id, coverage_status=in_cov,
            unmeasured=True, fallback_reason="记录为 synthetic，非测量")
    return CostEstimate(
        target_id="", analytic_base_ms=estimate_ms, selected_ms=estimate_ms,
        cost_source="calibrated", profile_ref=record.profile_id,
        calibration_ref=calibration.calibration_id, coverage_status="in",
        unmeasured=False)
