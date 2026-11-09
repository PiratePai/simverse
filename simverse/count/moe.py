"""MoE 路由账本：期望并集（正确式）与有放回近似（显式选项）。

正确式：每 token 无放回均匀选 k 个不同专家、M token 独立
    E[U(M)] = E·(1 − (1 − k/E)^M)
有放回近似仅作显式登记的近似（v2.moe_union_with_repl），禁止缺省使用。
期望权重读保持浮点 + Statistic.EXPECTATION；容量/精确计数保持整数。
"""
from __future__ import annotations

from ..contracts import Statistic
from .. import formula_registry as FR


def expected_union(E: int, k: int, M: int) -> float:
    """E[U(M)]，正确式（原理 §4）。M=1 时恰为 k。"""
    if E < 1 or k < 1 or k > E or M < 0:
        raise ValueError(f"非法路由参数 E={E} k={k} M={M}")
    if M == 0:
        return 0.0
    return E * (1.0 - (1.0 - k / E) ** M)


def expected_union_with_replacement(E: int, k: int, M: int) -> float:
    """有放回近似（显式近似选项；M=1 时 < k，露馅端点）。"""
    if E < 1 or k < 1 or M < 0:
        raise ValueError(f"非法路由参数 E={E} k={k} M={M}")
    if M == 0:
        return 0.0
    return E * (1.0 - (1.0 - 1.0 / E) ** (M * k))


def routed_weight_read_expected(E: int, k: int, M: int, per_expert_bytes: int) -> float:
    """一次执行内被用专家权重理想读一次的期望字节（principles §4）。"""
    return expected_union(E, k, M) * per_expert_bytes


def routed_flops_exact(M: int, k: int, hidden: int, inter: int) -> int:
    """逻辑 FLOPs（有效 assignment，SwiGLU 3 GEMM）——精确整数。"""
    return 2 * M * k * 3 * hidden * inter


def routed_capacity_bytes(E: int, hidden: int, inter: int, expert_bytes: int) -> int:
    """全部 routed 专家驻留容量（精确整数）。"""
    return E * 3 * hidden * inter * expert_bytes


def moe_rows(moe, tokens: int, hidden: int, quant: dict) -> list:
    """一层 MoE 的账本行（router/routed/shared(+gate)）。

    返回 list[dict]，字段与 count.work.OpRow 对齐；期望字段单列
    weight_read_expected（float），不与精确整数列混排。
    """
    E, k, inter = moe.experts, moe.top_k, moe.intermediate
    b_router = quant["dense_w"] / 8
    b_expert = quant["expert_w"] / 8
    b_act = quant["act"] / 8
    T = tokens
    rows = [{
        "layer_op": "moe_router", "flops": 2 * T * hidden * E,
        "weight_read": int(hidden * E * b_router),
        "weight_capacity": int(hidden * E * b_router),
        "act_read": int(T * hidden * b_act), "act_write": int(T * E * b_act),
        "formula": FR.F_MOE_ROUTER, "statistic": "exact",
        "shape": f"[{T},{hidden}]@[{hidden},{E}]",
    }, {
        "layer_op": "moe_routed", "flops": routed_flops_exact(T, k, hidden, inter),
        "weight_read": 0,
        "weight_read_expected": routed_weight_read_expected(
            E, k, T, int(3 * hidden * inter * b_expert)),
        "weight_capacity": routed_capacity_bytes(E, hidden, inter, b_expert),
        "act_read": int(T * hidden * b_act * k + 6 * k * T * inter * b_act),
        "act_write": int(T * hidden * b_act),
        "formula": FR.F_MOE_UNION, "statistic": "expectation",
        "shape": f"E={E} top-{k} M={T} → E[U]={expected_union(E, k, T):.3f}",
    }]
    if moe.shared_intermediate:
        si = moe.shared_intermediate
        rows.append({
            "layer_op": "moe_shared", "flops": 2 * T * 3 * hidden * si,
            "weight_read": int(3 * hidden * si * b_router),
            "weight_capacity": int(3 * hidden * si * b_router),
            "act_read": int(T * hidden * b_act + 6 * T * si * b_act),
            "act_write": int(T * hidden * b_act),
            "formula": FR.F_GEMM, "statistic": "exact",
            "shape": f"shared inter={si}",
        })
        if moe.shared_gate:
            rows.append({
                "layer_op": "shared_gate", "flops": 2 * T * hidden,
                "weight_read": int(hidden * b_router),
                "weight_capacity": int(hidden * b_router),
                "act_read": int(T * hidden * b_act), "act_write": int(T * b_act),
                "formula": FR.F_GEMM, "statistic": "exact",
                "shape": "sigmoid gate Linear(h,1)",
            })
    return rows
