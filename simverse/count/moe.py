"""MoE 路由账本：期望并集（正确式）与有放回近似（显式选项）。

正确式：每 token 无放回均匀选 k 个不同专家、M token 独立
    E[U(M)] = E·(1 − (1 − k/E)^M)
有放回近似仅作显式登记的近似（v2.moe_union_with_repl），禁止缺省使用。
期望权重读保持浮点 + Statistic.EXPECTATION；容量/精确计数保持整数。

per-rank 配方（R01，vLLM @dc74b8cb pin；lowering/parallel.py 证据）：
  * router gate：Replicated——每 rank 全量 h×E（不随 tp 缩放）；
  * EP 关（ep=1）：每 rank 全部 E 个专家、inter/tp 分片；
  * EP 开（ep=tp）：每 rank E//ep(+余数) 个整专家；期望流量按均匀路由
    的本地份额计算（E_l/E 比例），显式登记 expectation。
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


def moe_rows(moe, tokens: int, hidden: int, quant: dict, tp: int = 1, ep: int = 1,
             local_experts: int | None = None) -> list:
    """一层 MoE 的账本行（router/routed/shared(+gate)）——**per-rank 口径**。

    tp/ep/local_experts 来自 lowering/parallel.ParallelPlan：
      * EP 关：每 rank 全部 E 专家、inter/tp 分片（E_l = E, inter_rank =
        inter/tp）；期望并集在分片上按全 E 专家计（每专家只读本 rank 分片）。
      * EP 开：E_l = local_experts（余数 rank +1）、整专家 inter；
        期望 distinct 本地专家 = E_l·(1−(1−k/E)^M)；流量按本地 assignment
        份额 k·E_l/E（均匀路由期望，登记 expectation）。
    返回 list[dict]，字段与 count.work.OpRow 对齐；期望字段单列
    weight_read_expected（float），不与精确整数列混排。
    """
    E, k, inter = moe.experts, moe.top_k, moe.intermediate
    b_router = quant["dense_w"] / 8
    b_expert = quant["expert_w"] / 8
    b_act = quant["act"] / 8
    T = tokens
    ep_on = ep != 1
    E_l = local_experts if local_experts is not None else E
    inter_rank = inter if ep_on else inter // tp
    if ep_on:
        share = E_l / E                       # 本地 assignment 份额（期望）
        routed_flops = int(2 * T * k * 3 * hidden * inter * share)
        per_expert_bytes = int(3 * hidden * inter * b_expert)
        # 期望 distinct 本地专家命中数 × 整专家字节
        exp_union_local = E_l * (1.0 - (1.0 - k / E) ** T)
        w_expected = exp_union_local * per_expert_bytes
        cap = int(E_l * per_expert_bytes)
        act_assign = T * k * share            # 期望本地 assignment 行数
        shape = (f"E={E} top-{k} M={T} EP{ep} E_local={E_l} → "
                 f"E[U_local]={exp_union_local:.3f}（均匀路由假设）")
    else:
        routed_flops = 2 * T * k * 3 * hidden * inter_rank
        per_expert_shard_bytes = int(3 * hidden * inter_rank * b_expert)
        # 每 rank 持全部专家的 inter/tp 分片：期望并集仍在全 E 上取
        exp_union = expected_union(E, k, T)
        w_expected = exp_union * per_expert_shard_bytes
        cap = int(E * per_expert_shard_bytes)
        act_assign = T * k
        shape = (f"E={E} top-{k} M={T} TP{tp} inter/tp={inter_rank} → "
                 f"E[U]={exp_union:.3f}（分片字节口径）")
    rows = [{
        # router gate：Replicated（vLLM GateLinear），每 rank 全量
        "layer_op": "moe_router", "flops": 2 * T * hidden * E,
        "weight_read": int(hidden * E * b_router),
        "weight_capacity": int(hidden * E * b_router),
        "act_read": int(T * hidden * b_act), "act_write": int(T * E * b_act),
        "formula": FR.F_MOE_ROUTER, "statistic": "exact",
        "shape": f"[{T},{hidden}]@[{hidden},{E}] replicated（不随 tp 缩放）",
    }, {
        "layer_op": "moe_routed", "flops": routed_flops,
        "weight_read": 0,
        "weight_read_expected": w_expected,
        "weight_capacity": cap,
        "act_read": int(act_assign * hidden * b_act
                        + 6 * act_assign * inter_rank * b_act),
        "act_write": int(act_assign * hidden * b_act),
        "formula": FR.F_MOE_UNION,
        "statistic": "expectation" if ep_on else "expectation",
        "shape": shape,
    }]
    if moe.shared_intermediate:
        si = moe.shared_intermediate
        si_rank = si // tp                    # dense MLP 配方：inter/tp（EP 不改）
        rows.append({
            "layer_op": "moe_shared", "flops": 2 * T * 3 * hidden * si_rank,
            "weight_read": int(3 * hidden * si_rank * b_router),
            "weight_capacity": int(3 * hidden * si_rank * b_router),
            "act_read": int(T * hidden * b_act + 6 * T * si_rank * b_act),
            "act_write": int(T * hidden * b_act),
            "formula": FR.F_GEMM, "statistic": "exact",
            "shape": f"shared inter={si}/tp{tp}={si_rank}",
        })
        if moe.shared_gate:
            rows.append({
                # shared_expert_gate：ReplicatedLinear(h,1)，每 rank 复制
                "layer_op": "shared_gate", "flops": 2 * T * hidden,
                "weight_read": int(hidden * b_router),
                "weight_capacity": int(hidden * b_router),
                "act_read": int(T * hidden * b_act), "act_write": int(T * b_act),
                "formula": FR.F_GEMM, "statistic": "exact",
                "shape": "sigmoid gate Linear(h,1) replicated",
            })
    return rows
