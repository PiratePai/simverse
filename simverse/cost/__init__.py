"""cost v0.2：T0 解析估算 + profile 校准契约。

v0.1 的 roofline/link 实现保留在 legacy.py（供 v0.1 切片）；新管线用本包
t0.py/profile.py，语义按 kernel/量绑定，ProfileRecord 不改原始计数。
"""
from .legacy import CalibrationCoefficients, link_ms, roofline_ms  # noqa: F401
from .t0 import price_row, price_rows
from .profile import CalibrationModel, CostEstimate, ProfileRecord

__all__ = ["CalibrationCoefficients", "link_ms", "roofline_ms",
           "price_row", "price_rows", "ProfileRecord", "CalibrationModel",
           "CostEstimate"]
