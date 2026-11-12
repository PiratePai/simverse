# SPDX-License-Identifier: MIT
"""V4.1 checkpoint 量化布局的共享字节公式（R22，2026-11-13）。

证据（quant-pin.json 20 文件，vllm @f358f057，blob sha1 对
references/vllm-tree.json 逐项校验；source-checked，未执行）：
  * MXFP8 线性权重（dense/attention 投影/共享专家，DeepseekV41FP8Config
    get_quant_method L221-243 → ModelOptLinearMethod(kMxfp8Static,
    kMxfp8Dynamic)）：值 e4m3 1B/elem + ue8m0 uint8 scale 每 (1,32) 块
    （modelopt.py KMxfp8Static L2424-2479；quant_utils.py L246-248
    ScaleDesc(uint8, static, (1,32))；mxfp8_utils.py L10-13 常数）。
    checkpoint scale 为 32×32 块 (N/32, K/32)，装载时 repeat_interleave
    展开为运行时 (N, K/32)（modelopt.py L2430-2445）——本模块按**运行时
    展开后**字节数计容量/流量；K 每 rank 必须整除 32（L2450-2454）。
  * MXFP4 路由专家（同一 dispatch L244-261 → Mxfp4MoEMethod）：
    值 fp4 2 值/byte 打包 uint8 (E, 2I, H/2)/(E, H, I/2) + ue8m0 scale
    (E, 2I, H/32)/(E, H, I/32)（mxfp4.py create_weights L765-840）。
  * Marlin（SM80+，stock SM90 专家后端；oracle/mxfp4.py L461-486 优先级
    + L850-855 对齐）：分配前 intermediate 上取整 128、hidden 上取整 256
    （routed_experts.py L190-203 在 create_weights 前调用）。
  * 未量化类（get_quant_method 返回 None → 默认方法）：embed/lm_head/
    RMSNorm 权重沿 model dtype BF16（fp8.py L174-203 +
    vocab_parallel_embedding.py L322-326）；GateLinear BF16 由 R21 单列。

本模块只做整数算术；不含任何设备/带宽假设。执行域（激活字节、kernel
变体）见 families.parse_v41_csa2 的 extras["quant"] 与 capability-gaps。
"""

from __future__ import annotations

MX_BLOCK = 32                    # OCP MX 块宽（mxfp8_utils.py L13）
MARLIN_INTER_ALIGN = 128         # oracle/mxfp4.py L850-855
MARLIN_HIDDEN_ALIGN = 256


def require_mxfp8_input_divisible(k_per_rank: int, what: str) -> None:
    """MXFP8 kernel 要求输入维整除 32（modelopt.py L2450-2454；
    marlin.py L33-38 同约束）。不满足的组合按 config 域拒绝，不静默取整。"""
    if k_per_rank % MX_BLOCK:
        raise ValueError(
            f"{what}: 输入维 {k_per_rank} 不整除 {MX_BLOCK}，MXFP8 线性布局"
            "非法（pinned modelopt.py L2450-2454 K % 32 断言）")


def mxfp8_linear_bytes(n: int, k: int) -> tuple[int, int]:
    """量化线性权重 (N, K) → (值字节, scale 字节)。

    值 = N·K·1B（e4m3）；scale = N·(K/32)·1B（ue8m0，运行时展开后口径）。
    """
    require_mxfp8_input_divisible(k, "mxfp8_linear")
    return n * k, n * (k // MX_BLOCK)


def marlin_round_up_intermediate(inter_per_rank: int) -> int:
    """Marlin 专家路径分配前的 intermediate 对齐（oracle L850-855）。"""
    return -(-inter_per_rank // MARLIN_INTER_ALIGN) * MARLIN_INTER_ALIGN


def marlin_round_up_hidden(hidden: int) -> int:
    return -(-hidden // MARLIN_HIDDEN_ALIGN) * MARLIN_HIDDEN_ALIGN


def mxfp4_expert_bytes(hidden: int, inter_eff: int) -> tuple[int, int]:
    """一个（本地）专家 (H, I_eff) → (值字节, scale 字节)。

    值：w13 (2I, H/2) + w2 (H, I/2) = 1.5·H·I；
    scale：w13 (2I, H/32) + w2 (H, I/32) = 3·H·I/32（mxfp4.py L786-840）。
    """
    if hidden % (2 * MX_BLOCK) or inter_eff % (2 * MX_BLOCK):
        raise ValueError(
            f"mxfp4_expert: H={hidden}/I={inter_eff} 须整除 {2 * MX_BLOCK}"
            "（打包 2 值/byte 与 32 元素 scale 块）")
    w13 = 2 * inter_eff * (hidden // 2)
    w2 = hidden * (inter_eff // 2)
    s13 = 2 * inter_eff * (hidden // MX_BLOCK)
    s2 = hidden * (inter_eff // MX_BLOCK)
    return w13 + w2, s13 + s2
