"""T0 解析计价（Q2：首版性能全部 analytical-estimate，无实测数字）。

按行计价：byte 五类合计 → roofline max+ρ·min；逐行携带 formula/t_mem/t_comp。
设备参数来自 devices 注册表（v0.2 扩展：标称容量/可分配预算分开）。
"""
from __future__ import annotations

from typing import Dict, List

from .. import formula_registry as FR
from ..count.work import OpRow
from .legacy import DeviceSpec, roofline_ms

# 算子 → 计价精度路由（显式表；gemm 看 dense_w 位宽，元素/注意力核看 act）
_ELEMENTWISE = {
    "q_norm", "k_norm", "rope", "q_rope", "k_rope", "o_rope", "attn_norm",
    "mlp_norm", "attn_add", "mlp_add", "compress_norm", "compressed_rope",
    "index_k_norm", "index_k_rope", "index_q_rope", "swa_cache_write",
    "compressed_cache_write", "index_cache_write", "index_select",
    "candidate_publish", "embed_lookup", "attention_core", "sparse_attention",
    "mla_absorbed_core", "causal_conv1d", "gdn_recurrence", "v_norm_gated",
    "q_a_norm", "draft_kv_scan", "shared_gate",
    # Engram 非 GEMM 算子（R04）：哈希/查表 dequant/门控核按 act 位宽计价
    #（engram_wkv 是 GEMM，走 dense_w 路由）
    "engram_hash", "engram_lookup", "engram_post_wkv",
}


def row_dtype(row: OpRow, gemm_dtype: str, ew_dtype: str) -> str:
    return ew_dtype if row.op in _ELEMENTWISE else gemm_dtype


def price_row(row: OpRow, device: DeviceSpec, gemm_dtype: str = "fp8",
              ew_dtype: str = "fp8", engram_lookup_host: bool = False
              ) -> Dict[str, float]:
    """单行 T0 计价；host Engram 表读取不套用 GPU HBM 带宽。"""
    bytes_exact = (row.weight_read + row.act_read + row.act_write + row.kv_read
                   + row.kv_write + row.state_read + row.state_write)
    host_expected = ((row.weight_read_expected or 0.0)
                     if engram_lookup_host and row.op == "engram_lookup" else 0.0)
    bytes_expected = (row.weight_read_expected or 0.0) - host_expected
    est = roofline_ms(row.flops, bytes_exact + bytes_expected, device,
                      dtype=row_dtype(row, gemm_dtype, ew_dtype))
    est["bytes_exact"] = float(bytes_exact)
    est["bytes_expected"] = float(bytes_expected)
    est["bytes_host_expected"] = float(host_expected)
    est["formula"] = FR.F_ROOFLINE
    return est


def price_rows(rows: List[OpRow], device: DeviceSpec, gemm_dtype: str = "fp8",
               ew_dtype: str = "fp8", engram_lookup_host: bool = False
               ) -> List[Dict[str, float]]:
    return [price_row(r, device, gemm_dtype, ew_dtype, engram_lookup_host)
            for r in rows]


def stage_total(prices: List[Dict[str, float]]) -> float:
    """串行无重叠口径的阶段时延（解析档；并行归因在 execute 事件层）。"""
    return sum(p["t_ms"] for p in prices)


def dtype_route(quant: dict) -> tuple:
    """精度路由：gemm 看 dense_w；元素/注意力核看 act。FP4 权重不自动 ×4 算力
    （E10：storage/compute 分离，H100 无 FP4 GEMM 走 FP8 档为显式口径）。"""
    gemm = "fp16" if quant["dense_w"] >= 16 else "fp8"
    ew = "fp16" if quant["act"] >= 16 else "fp8"
    return gemm, ew
