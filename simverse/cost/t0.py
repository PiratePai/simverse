"""T0 解析计价（Q2：首版性能全部 analytical-estimate，无实测数字）。

按行计价：byte 五类合计 → roofline max+ρ·min；逐行携带 formula/t_mem/t_comp。
设备参数来自 devices 注册表（v0.2 扩展：标称容量/可分配预算分开）。

R27（2026-11-17，cost-integrity 轮）：
  * dtype_route 增加执行配方入参——注册了 extras["quant"]（R22 quant-pin
    @f358f057）的模型按 source-checked 配方选 GEMM 计算档（Marlin
    W8A16/W4A16 在 BF16 Tensor Core 上执行），不再只看 dense_w 存储位宽；
  * price_row 在同一次计价内附带行级元数据（compute_dtype/峰值/带宽/
    cost_basis），供 Roofline 绘图与导出直接消费（前端不得自行推导物理
    参数；单位：TFLOP/s 与十进制 TB/s）。
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .. import formula_registry as FR
from ..count.work import OpRow
from .legacy import DeviceSpec, roofline_ms

# 算子 → 计价精度路由（显式表；gemm 看 dense_w 位宽，元素/注意力核看 act）
_ELEMENTWISE = {
    "q_norm", "k_norm", "rope", "q_rope", "k_rope", "o_rope", "attn_norm",
    "mlp_norm", "attn_add", "mlp_add", "swa_cache_write",
    "embed_lookup", "attention_core", "sparse_attention",
    "mla_absorbed_core", "causal_conv1d", "gdn_recurrence", "v_norm_gated",
    "q_a_norm", "draft_kv_scan", "shared_gate",
    # Engram 非 GEMM 算子（R04）：哈希/查表 dequant/门控核按 act 位宽计价
    #（engram_wkv 是 GEMM，走 dense_w 路由）
    "engram_hash", "engram_lookup", "engram_post_wkv",
    # CSA2 索引/压缩非 GEMM 算子（R20；index_score 是 DeepGEMM MQA logits，
    # 走 dense_w 路由；index_q_proj/weights_proj/k_proj/compress_proj 同 GEMM）
    "compress_pool", "compressed_insert", "index_k_store", "index_q_rope",
    "index_topk", "index_fill", "candidate_select", "candidate_mask",
    "sparse_index_build",
    # R25：FlashMLA 注意力核本体消费链（split-KV 归并/整数核/gather 按
    # act 档计价；sparse_attention 原已在表）
    "attn_splitkv", "attn_index_global", "attn_swa_index_build",
    "attn_sched_meta", "attn_kv_gather",
}


def row_dtype(row: OpRow, gemm_dtype: str, ew_dtype: str) -> str:
    return ew_dtype if row.op in _ELEMENTWISE else gemm_dtype


# ---- 成本口径说明（Roofline/导出元数据；R27） ------------------------------
# 无执行配方的 GEMM：按 dense_w 存储位宽启发选档（乐观假设档，如实披露）。
_BASIS_GEMM_HEURISTIC = (
    "T0 roofline max+0.15·min；GEMM 计算峰值按 dense_w 存储位宽启发选档"
    "（模型无 pinned 执行配方，乐观假设：存储位宽=计算位宽）")
# 元素/注意力核：FLOPs 按路由档 Tensor Core 峰值近似计价——这些算子实际
# 在 CUDA core/SFU/专用核上执行且多为带宽主导，注册表无对应执行单元峰值，
# 不虚构；该近似使解析耗时偏乐观，登记为建模缺口（不逐算子加 launch）。
_BASIS_ELEMENTWISE = (
    "T0 roofline max+0.15·min；元素/注意力核 FLOPs 按路由档 Tensor Core 峰值"
    "近似计价（非 Tensor Core 执行单元/launch/融合未建模；带宽主导行影响小）")


def gemm_cost_basis(quant_recipe: Optional[dict]) -> str:
    """GEMM 行的 cost_basis 文案（区分配方档与启发档）。"""
    if quant_recipe is not None and quant_recipe.get("gemm_compute_dtype"):
        basis = str(quant_recipe.get("gemm_compute_basis")
                    or "extras.quant 执行配方 gemm_compute_dtype")
        return ("T0 roofline max+0.15·min；GEMM 计算峰值档由执行配方选择（"
                + basis + "）")
    return _BASIS_GEMM_HEURISTIC


def ew_cost_basis() -> str:
    """元素/注意力核行的 cost_basis 文案。"""
    return _BASIS_ELEMENTWISE


def price_row(row: OpRow, device: DeviceSpec, gemm_dtype: str = "fp8",
              ew_dtype: str = "fp8", engram_lookup_host: bool = False,
              gemm_basis: str = "", ew_basis: str = ""
              ) -> Dict[str, float]:
    """单行 T0 计价；host Engram 表读取不套用 GPU HBM 带宽。

    返回值在同一次计价内附带行级元数据（R27）：compute_dtype（本行计算档
    tier 键，"fp16" 档含 BF16——设备注册表 BF16/FP16 同档）、peak_tflops
    （该档设备规格峰值，dense 口径）、effective_peak_tflops（本次计价采用
    的峰值 = 规格 × mfu_cap）、bandwidth_tb_s（本次计价带宽，十进制
    TB/s）、cost_basis（口径短述）。峰值/带宽取自 roofline_ms 同一调用
    消费的 device 档位——不另建一套路由。
    """
    bytes_exact = (row.weight_read + row.act_read + row.act_write + row.kv_read
                   + row.kv_write + row.state_read + row.state_write)
    host_expected = ((row.weight_read_expected or 0.0)
                     if engram_lookup_host and row.op == "engram_lookup" else 0.0)
    bytes_expected = (row.weight_read_expected or 0.0) - host_expected
    dtype = row_dtype(row, gemm_dtype, ew_dtype)
    if device.backend == "ascend":
        # 仅纯向量行使用旧表 Vector 档；矩阵/混合 attention 行仍为 Cube
        # 解析近似，不把 CUDA kernel 配方冒充 Ascend 实现。
        vector_ops = {"q_norm", "k_norm", "rope", "q_rope", "k_rope", "o_rope",
                      "attn_norm", "mlp_norm", "attn_add", "mlp_add", "v_norm_gated",
                      "q_a_norm", "shared_gate", "moe_act", "moe_shared_act"}
        if dtype != "fp16":
            raise ValueError("910B3 旧仓解析档仅支持 BF16/FP16；无 FP8/量化执行配方")
        dtype = "fp32" if row.op in vector_ops else "fp16"
    est = roofline_ms(row.flops, bytes_exact + bytes_expected, device,
                      dtype=dtype)
    basis = ((ew_basis or _BASIS_ELEMENTWISE) if row.op in _ELEMENTWISE
             else (gemm_basis or _BASIS_GEMM_HEURISTIC))
    if device.backend == "ascend":
        basis = ("T0 旧仓解析假设；Cube 249T / Vector 31T，MFU=0.20，"
                 "HBM 1.6TB/s×MBU=0.20；矩阵/混合核未按 Ascend 实现 lowering，"
                 "launch/融合/运行时池未校准；非实测性能")
    if host_expected:
        basis += ("；Engram host 表期望读字节未计价（UVA 时延 unknown，"
                  "不套 GPU HBM 带宽）")
    peak = device.peak(dtype)
    est["bytes_exact"] = float(bytes_exact)
    est["bytes_expected"] = float(bytes_expected)
    est["bytes_host_expected"] = float(host_expected)
    est["formula"] = FR.F_ROOFLINE
    est["compute_dtype"] = dtype
    est["peak_tflops"] = peak / 1e12
    est["effective_peak_tflops"] = peak * device.mfu_cap / 1e12
    est["bandwidth_tb_s"] = device.hbm_bw * device.mbu_cap / 1e12
    est["cost_basis"] = basis
    return est


def price_rows(rows: List[OpRow], device: DeviceSpec, gemm_dtype: str = "fp8",
               ew_dtype: str = "fp8", engram_lookup_host: bool = False,
               gemm_basis: str = "", ew_basis: str = ""
               ) -> List[Dict[str, float]]:
    return [price_row(r, device, gemm_dtype, ew_dtype, engram_lookup_host,
                      gemm_basis, ew_basis)
            for r in rows]


def stage_total(prices: List[Dict[str, float]]) -> float:
    """串行无重叠口径的阶段时延（解析档；并行归因在 execute 事件层）。"""
    return sum(p["t_ms"] for p in prices)


def dtype_route(quant: dict, quant_recipe: Optional[dict] = None) -> tuple:
    """精度路由（计算档选择）。

    无执行配方（quant_recipe=None，默认）：沿用存储位宽启发——GEMM 看
    dense_w、元素/注意力核看 act。该口径把"存储位宽=计算位宽"作乐观假设
    （如 W8A8 场景按 FP8 Tensor 峰值计价），在 cost_basis 披露，不改历史
    行为。

    有执行配方（ModelIR.extras["quant"]，R22 quant-pin.json 20 文件
    @f358f057 source-checked）：按配方选 GEMM 计算档——注册设备（H100/
    H200=SM90、L40=SM89，均 ≤SM90）上 MXFP8 线性/MXFP4 专家走 Marlin
    W8A16/W4A16：8/4bit 只是存储格式，kernel 寄存器内反量化后用 BF16
    Tensor Core 执行、激活保持 BF16（marlin.py L14-26、oracle/mxfp4.py
    L461-486）→ GEMM 峰值取 bf16 档（与 fp16 同档，989.5 TFLOPS dense
    H100）。这是**理论峰值路由修正**，不是实测吞吐校准：Marlin 实际吞吐
    低于 bf16 理论峰值，无 H100 实测证据时不做进一步折减（校准缺口另列，
    见 docs/validation/cost-integrity-review-fixes.md）。FP4 权重不自动 ×4
    算力（E10：storage/compute 分离）。域外执行变体（Humming W4A8、
    SM100 W8A8 动态激活、emulation 反量化驻留）不进本路由（G-C14 登记）。

    元素/注意力核路由不变（看 act）：配方只钉 GEMM kernel 执行域；非
    GEMM 行的"按 Tensor Core 峰值近似"披露见 _BASIS_ELEMENTWISE。
    """
    gemm = "fp16" if quant["dense_w"] >= 16 else "fp8"
    ew = "fp16" if quant["act"] >= 16 else "fp8"
    if quant_recipe is not None:
        compute = quant_recipe.get("gemm_compute_dtype")
        if compute:
            gemm = str(compute)
    return gemm, ew
