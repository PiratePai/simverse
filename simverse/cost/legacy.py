"""L1 统一成本核：全仓唯一的 时间 = f(计数, 硬件) 公式维护点（方案 §4.3）。

输入是 ledger.py 的整数计数（Count 段已完成），输出 T0 roofline 估算。
校准（T1 拟合/实测替换）接口预留：CalibrationCoefficients 允许按算子类替换时延模型，
但永远保留原始计数（"数字带来源"的机制基础）。
"""
from __future__ import annotations

from typing import Dict, Optional

from .. import formulas as F  # 根包公式表（v0.1）
from ..contracts import DeviceSpec, LinkSpec

# 每类算子的精度路由（决定取哪档峰值算力）——RecipeSpec 的最小子集，v0.1 隐式 vllm-cuda
DTYPE_ROUTE = {
    "gemm_fp8": "fp8", "gemm_fp4w": "fp8",   # H100 无 FP4 GEMM → 算力不 ×4（A6 口径）
    "gemm_fp16": "fp16", "elementwise": "fp16",
}


def expected_unique(experts: int, draws: int) -> float:
    """MoE 专家并集 U(M)：i.i.d. 均匀路由下 M 次抽取命中的期望不同专家数。

    与旧仓 piviewer/roofline.py:141 _expected_unique 同式（M = tokens×top_k 次抽取）；
    精确组合式 E·(1−C(E−k,M)/C(E,M)) 在 E≫M 时偏差 <1%（见差异报告 D-6）。
    """
    if experts <= 1:
        return float(min(draws, max(experts, 1)))
    return experts * (1.0 - (1.0 - 1.0 / experts) ** draws)


def roofline_ms(flops: int, bytes_total: int, device: DeviceSpec, dtype: str = "fp8",
                overlap_tail: float = 0.15) -> Dict[str, float]:
    """t = max(计算, 访存) + overlap_tail·min(...)。

    计算项乘 MFU 上限（0.75，旧仓 spec_gain 口径）；访存用聚合 HBM 带宽。
    返回分项以便账本展示（t_mem/t_comp/t/bound），公式 id = roofline_max。
    """
    if bytes_total < 0 or flops < 0:
        raise ValueError("roofline 输入必须非负")
    peak_eff = device.peak(dtype) * device.mfu_cap
    t_comp_ms = (flops / peak_eff * 1e3) if flops else 0.0
    t_mem_ms = (bytes_total / device.hbm_bw * 1e3) if bytes_total else 0.0
    t_ms = max(t_comp_ms, t_mem_ms) + overlap_tail * min(t_comp_ms, t_mem_ms)
    bound = "compute" if t_comp_ms >= t_mem_ms else ("memory" if t_mem_ms else "none")
    return {"t_ms": t_ms, "t_comp_ms": t_comp_ms, "t_mem_ms": t_mem_ms, "bound": bound,
            "formula_id": F.F_ROOFLINE}


def launch_ms(device: DeviceSpec, kernel_count: int) -> float:
    """eager 口径的 launch 上界估计（CUDA Graph 未建模；方案 §4.6 三态之 unsupported）。"""
    return device.launch_overhead_ms * kernel_count


def link_ms(link: LinkSpec, byte_count: int) -> Dict[str, float]:
    """单消息单向：fixed + bytes/bw（每方向各计一次启动延迟）。"""
    transfer_ms = byte_count / link.bw * 1e3 if byte_count else 0.0
    return {"t_ms": link.fixed_ms + transfer_ms, "transfer_ms": transfer_ms,
            "fixed_ms": link.fixed_ms, "formula_id": F.F_LINK}


class CalibrationCoefficients:
    """T1 校准接口占位（决策 D-4：首发不采数，全部 T0）。

    M5 起允许 per-op 替换 hw_eff / launch；本切片仅透传 T0。
    """

    def __init__(self, overrides: Optional[Dict[str, Dict[str, float]]] = None):
        self.overrides = overrides or {}

    def apply(self, op_key: str, t_ms: float) -> float:
        return t_ms  # T0: 无替换
