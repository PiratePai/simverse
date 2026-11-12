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

V4.1 路由细化（R21，vllm @f358f057 moe-pin.json 11 文件 source-checked）：
  * v41_moe_route_rows：sqrtsoftplus 打分 + noaux_tc 偏置平坦 top-k（无分组）
    + sum→1 归一 ×routed_scaling 三行替换通用 moe_router 行，per-rank 复制；
  * V4.1 EP 为整除断言（require_divisible_ep），与 qwen35 pin 的余数规则不同。
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


def require_divisible_ep(E: int, ep: int) -> None:
    """V4.1 EP 整除约束（DeepseekV4MoE pinned 断言，非余数分布）。

    vllm @f358f057 deepseek_v4/nvidia/model.py L416-421（fused 路径）与
    L343-368（mega 路径）：n_physical_experts % ep_size == 0 断言（eplb
    冗余专家默认 0）；experts_start_idx = ep_rank * n_local。与 qwen35 pin
    的 generic FusedMoE 余数规则不同——V4.1 族非整除组合直接拒绝。
    """
    if ep != 1 and E % ep:
        raise ValueError(
            f"V4.1 MoE EP{ep} 不整除专家数 {E}（pinned DeepseekV4MoE 断言 "
            f"n_physical % ep == 0；eplb 冗余默认 0，无余数 rank 分布）")


def v41_moe_route_rows(moe, route: dict, tokens: int, hidden: int,
                       quant: dict) -> list:
    """V4.1 MoE 路由三行（gate/score/topk_renorm）——**per-rank 全量复制**。

    证据（moe-pin.json 11 文件，vllm @f358f057，blob sha1 对 vllm-tree.json
    逐项校验；source-checked 2026-11-12 R21，未执行 kernel/CPU/GPU）：
      * gate：GateLinear(h→E, bias=False, out_dtype=fp32) 为 ReplicatedLinear
        子类（deepseek_v4/nvidia/model.py L270-277；router/gate_linear.py
        L26）→ 每 rank 全量 GEMM，router logits **FP32**；
      * scoring sqrtsoftplus：weights = sqrt(softplus(logits))，logits>20 取
        sqrt(logits) 数值护栏（dsv4_topk.py L88-92）；选择列 = weights +
        e_score_correction_bias，NaN→−1e30（L93-95）；CPU 侧 biased_topk_cpu
        同语义（cpu_router.py L81-134 docstring："bias steers selection only"）；
      * 选择：k 轮平坦 argmax（**无分组**：V4.1 config 无 n_group/topk_group
        键，router_factory.py L168-205 仅 use_grouped_topk 时走
        GroupedTopKRouter），平手取最小专家 id（L98-109 min(candidate)），
        权重取**无偏** score（L105-108）；
      * 归一化：selected_weights *= routed_scaling/sum（sum≤0 除 1 护栏，
        L111-115）；输出 FP32 weights + int32 ids（L69-70/L123-128）；
      * VL 分支：V4.1-Flash 为视觉 checkpoint（vision_n_layers=32>0）→
        bias_vl 存在且 HAS_VL=True，文本 token 也逐行加载 input_ids +
        bias_vl 做 is_image 判定（L60-70，5 个哨兵 id 才走 bias_vl）。
    op 口径：解析算术 op（exp/log/sqrt/比较各 1），非 kernel 指令数；
    字节为 unfused 逻辑访问参考，非实测 HBM。
    返回 list[dict]，字段与 count.work.OpRow 对齐。
    """
    E, k = moe.experts, moe.top_k
    T = tokens
    b_router = quant["dense_w"] / 8
    b_act = quant["act"] / 8
    has_vl = bool(route.get("bias_vl_bytes"))
    vl_ops = (E + 2) if has_vl else 0            # where E + 哨兵 2 比较/token
    vl_ids_read = T * 8 if has_vl else 0         # input_ids int64 上界
    bias_cnt = 2 if has_vl else 1                # e_score_correction_bias(+bias_vl)
    scale = float(route.get("routed_scaling", 1.0))
    score_ops = T * (8 * E + vl_ops)             # 打分 5 + bias 加 1 + NaN 护栏 2
    topk_ops = T * (k * (5 * E + 3) + 2 * k + 4)  # k 轮选择 + 归一/缩放
    return [{
        # GateLinear：Replicated，每 rank 全量 h×E；FP32 logits 出口
        "layer_op": "moe_gate", "flops": 2 * T * hidden * E,
        "weight_read": int(hidden * E * b_router),
        "weight_capacity": int(hidden * E * b_router),
        "act_read": int(T * hidden * b_act), "act_write": int(T * E * 4),
        "formula": FR.F_MOE_ROUTE_GATE, "statistic": "exact",
        "shape": f"[{T},{hidden}]@[{hidden},{E}] replicated（GateLinear "
                 "out_dtype=fp32，不随 tp/ep 缩放）",
    }, {
        # sqrtsoftplus 打分 + 偏置加 + NaN 护栏（+ VL 哨兵判定）
        "layer_op": "moe_score", "flops": score_ops,
        "weight_read": int(bias_cnt * E * 4),
        "weight_capacity": int(bias_cnt * E * 4),
        "act_read": int(T * E * 4 + vl_ids_read), "act_write": 0,
        "formula": FR.F_MOE_ROUTE_SCORE, "statistic": "exact",
        "shape": f"sqrt(softplus(logits)) {T}×{E}（>20 护栏；+bias 选择列，"
                 "NaN→−1e30）"
                 + (f"；HAS_VL 哨兵判定（bias_vl {E}×4B 复制，input_ids "
                    "int64 上界）" if has_vl else ""),
    }, {
        # k 轮平坦偏置 argmax + 权重取无偏 score + sum→1 归一 ×routed_scaling
        "layer_op": "moe_topk_renorm", "flops": topk_ops,
        "weight_read": 0, "weight_capacity": 0,
        "act_read": 0, "act_write": int(T * k * 8),
        "formula": FR.F_MOE_ROUTE_TOPK, "statistic": "exact",
        "shape": f"top-{k}/{E} 平坦偏置 argmax（无分组，平手最小 id）；"
                 f"权重=无偏 score；sum 归一 ×{scale}（sum≤0 除 1）；"
                 f"写 FP32 weights+int32 ids {T}×{k}",
    }]


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
