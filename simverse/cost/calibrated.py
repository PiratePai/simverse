"""实测 profile 校准计价（第一条 H200 闭环；MERGE_CRITERIA §3）。

算子所有权映射（采集实现证据）：
  * attn_pre_proj      ← q_proj+k_proj+v_proj+q_norm+k_norm
    （pi-viewer frontier/profiling/linear_op/linear_op_impl.py CausalSelfAttention：
     fused qkv_proj 定时器内含 QKNorm；qwen3_moe ∈ QK_NORM_MODEL_TYPE_ALLOWLIST）
  * input_layernorm    ← attn_add+attn_norm（fused add+norm；
    qwen3_moe ∈ FUSED_ADD_NORM_MODEL_TYPE_ALLOWLIST）
  * post_attention_layernorm ← mlp_add+mlp_norm（同上）
  * attn_rope ← rope；attn_post_proj ← o_proj；emb ← embed_lookup
  * decode 注意力块（4 算子：input_reshape+kv_cache_save+attn_decode+output_reshape）
    ← attention_core（含 kv_write 字节；attn_kv_cache_extract 仅 prefill/mixed 行、
    decode 行的 attn_prefill 列为空转核 median ~7.5µs——两者均不属 decode 运行时
    路径，已排除，见蒸馏器 DECODE_BLOCK_OPS 注记）

域门禁（全部满足才 selected=measured，否则逐组回退解析并记原因）：
  设备/精度(BF16)/TP1/形状一致(hidden,q/kv头,head_dim,vocab/QKNorm/全旋转/无滑窗)/
  阶段形状在网格内。QKNorm/旋转/滑窗三项在采集实现中影响被测算子构成
  （attn_pre_proj 含 QKNorm；rope 为全 head_dim 旋转；attention 为全注意力），
  模型侧不一致即形状域不符。
kernel 间隙政策（2026-11-10 修正，修复重复计费）：CUDA_EVENT 逐 op median 的
  事件括号**已包含**该 op 内部 kernel 间的间隙（launch-bound 时还含 host 空闲），
  因此间隙补贴只按**计时区间边界**数计（每个被测组实例 1 个边界；注意力块 4 op
  各自成区间），不再按全部 kernel 数计——那会把已含在实测内的 op 内间隙再计一遍
  （B8 全模型 673 kernel vs 433 区间，重复 ~0.32ms/步）。kernel_gap.csv 的 eager
  值为 host-ahead 条件下的纯设备间隙，用作边界间隙是低估（launch-bound 边界更长），
  作为显式估计披露，不外推。
铁律：只产出 CostEstimate，不改任何 OpRow 原始计数（G7 重放保证）。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..count.work import OpRow
from .profile_store import ProfileSet

# 组名 → (拥有的新仓账本行 op 列表, 查表键)
LINEAR_GROUPS: Dict[str, List[str]] = {
    "attn_pre_proj": ["q_proj", "k_proj", "v_proj", "q_norm", "k_norm"],
    "attn_rope": ["rope"],
    "attn_post_proj": ["o_proj"],
    "input_layernorm": ["attn_add", "attn_norm"],
    "post_attention_layernorm": ["mlp_add", "mlp_norm"],
    "emb": ["embed_lookup"],
}
ATTN_BLOCK_ROWS = ["attention_core"]
# 注意力块在采集侧为 4 个独立 CUDA_EVENT 计时区间（reshape/save/decode/reshape），
# 区间边界数 = 4/层（profile_store.DECODE_BLOCK_OPS 同序）。
DECODE_BLOCK_REGIONS = 4


def _shape_mismatches(model, ident: dict) -> List[str]:
    """被校准算子涉及的形状必须与采集复制品逐项一致（专家数除外：不在域内算子）。

    除逐算子维度外还核对三项**算子构成**语义（形状一致的必要条件补强）：
    QKNorm（attn_pre_proj 实测块含 QKNorm；无 QKNorm 的模型用该测量会多计 norm）、
    全 head_dim 旋转（采集 rope 为全旋转；部分旋转的 rope 时间形状不同）、
    无滑窗（采集为全注意力；滑窗层的 kernel 路径不同）。
    """
    bad = []
    if model.hidden != ident["hidden_size"]:
        bad.append(f"hidden {model.hidden}≠{ident['hidden_size']}")
    attns = [l.attn for l in model.layers if l.attn is not None]
    if attns:
        a = attns[0]
        if a.heads != ident["n_q_head"]:
            bad.append(f"heads {a.heads}≠{ident['n_q_head']}")
        if a.kv_heads != ident["n_kv_head"]:
            bad.append(f"kv_heads {a.kv_heads}≠{ident['n_kv_head']}")
        if a.head_dim != ident["head_dim"]:
            bad.append(f"head_dim {a.head_dim}≠{ident['head_dim']}")
        # ---- 算子构成语义（shape 一致 ≠ 块构成一致）----
        if ident.get("use_qk_norm") and not all(x.qk_head_norm for x in attns):
            bad.append("QKNorm 构成不符：实测 attn_pre_proj 含 QKNorm，模型无")
        if any(x.rotary_dim != x.head_dim for x in attns):
            bad.append("旋转构成不符：实测 rope 为全 head_dim 旋转，模型存在部分旋转层")
        if any(x.sliding_window for x in attns):
            bad.append("滑窗构成不符：实测为全注意力（无滑窗层），模型存在滑窗层")
    if model.vocab != ident["vocab_size"]:
        bad.append(f"vocab {model.vocab}≠{ident['vocab_size']}")
    return bad


class CalibratedPricer:
    """对一个 stage 的 OpRow 列表给出 measured/analytic 分列计价。"""

    def __init__(self, pset: ProfileSet, model, scenario):
        self.pset = pset
        self.model = model
        self.scenario = scenario
        self.tp = scenario.deployment.target_tp
        self.domain_reasons: List[str] = []
        dep = scenario.deployment
        if dep.target_device != pset.device:
            self.domain_reasons.append(
                f"设备不符：场景 {dep.target_device} ≠ 域 {pset.device}")
        if not pset.tp_allowed(self.tp):
            self.domain_reasons.append(
                f"TP 不符：场景 TP{self.tp} ∉ 域 "
                f"{pset.tensor_parallel}")
        q = scenario.quant
        if not (q["dense_w"] == 16 and q["act"] == 16 and q["kv"] == 16):
            self.domain_reasons.append(
                f"精度不符：域 BF16（16bit），场景 dense_w={q['dense_w']} "
                f"act={q['act']} kv={q['kv']}")
        bad = _shape_mismatches(model, pset.shape_identity)
        if bad:
            self.domain_reasons.append("形状不一致：" + "; ".join(bad))

    @property
    def applicable(self) -> bool:
        return not self.domain_reasons

    def price_stage(self, stage: str, rows: List[OpRow],
                    prices: List[Dict[str, float]], B: int, S: int,
                    Q: int) -> dict:
        """返回该 stage 的校准摘要。selected = 域内实测 + 域外解析(差额) + kernel 间隙。"""
        analytic_total = sum(p["t_ms"] for p in prices)
        if not self.applicable:
            return {
                "stage": stage, "domain_ok": False,
                "reasons": list(self.domain_reasons),
                "analytic_ms": analytic_total, "selected_ms": None,
                "measured_ms": 0.0, "fallback_ms": analytic_total,
                "gap_allowance_ms": 0.0, "kernels": 0, "gap_regions": 0,
                "groups": [], "coverage_rows": [0, len(rows)],
                "coverage_time_frac": None,
            }
        row_price = {(r.layer, r.op): p["t_ms"] for r, p in zip(rows, prices)}
        by_layer: Dict[int, Dict[str, OpRow]] = {}
        for r in rows:
            by_layer.setdefault(r.layer, {})[r.op] = r
        T = B * Q
        measured_ms = 0.0
        kernels = 0
        regions = 0                    # 计时区间边界数（间隙补贴基数；见模块 docstring）
        covered: set = set()          # 被 measured 组占有的 (layer, op)
        groups_out: List[dict] = []

        for gname, ops in LINEAR_GROUPS.items():
            layers = [li for li, m in by_layer.items() if all(o in m for o in ops)]
            if not layers:
                continue
            per_layer_t = self.pset.lookup_linear(gname, T, tp=self.tp)
            analytic_g = sum(row_price[(li, o)] for li in layers for o in ops)
            if per_layer_t is None:
                has_table = any(t2 == self.tp
                                for (t2, _t) in self.pset.linear_ms.get(gname, {}))
                reason = (f"算子组 {gname} 无 TP{self.tp} 实测表"
                          if not has_table else
                          f"token 数 {T} 在 linear 网格外"
                          f"（域 {self.pset.manifest['domain']['linear_tokens']}）")
                groups_out.append({
                    "group": gname, "layers": 0, "source": "analytic",
                    "reason": reason,
                    "t_ms": round(analytic_g, 6), "rows": len(layers) * len(ops)})
                continue
            measured_ms += per_layer_t * len(layers)
            kc = self.pset.lookup_linear_kc(gname, T, tp=self.tp)
            kernels += kc * len(layers)
            regions += len(layers)     # 每组实例 = 1 个 CUDA_EVENT 计时区间
            covered.update((li, o) for li in layers for o in ops)
            groups_out.append({
                "group": gname, "layers": len(layers), "source": "measured",
                "profile_ref": f"{self.pset.set_id}#{gname}",
                "t_ms": round(per_layer_t * len(layers), 6),
                "rows": len(layers) * len(ops), "kernel_count_per_layer": kc,
                "analytic_replaced_ms": round(analytic_g, 6)})

        # decode 注意力块：仅 Q==1 decode 形状（verify 多 query 域外）
        attn_layers = [li for li, m in by_layer.items() if "attention_core" in m]
        if attn_layers:
            blk = self.pset.lookup_attn_block(B, S, tp=self.tp) if Q == 1 else None
            analytic_a = sum(row_price[(li, "attention_core")] for li in attn_layers)
            if blk is not None:
                kc = self.pset.lookup_attn_block_kc(B, S, tp=self.tp)
                measured_ms += blk * len(attn_layers)
                kernels += kc * len(attn_layers)
                regions += DECODE_BLOCK_REGIONS * len(attn_layers)  # 4 op 各自计时
                covered.update((li, "attention_core") for li in attn_layers)
                groups_out.append({
                    "group": "attn_decode_block", "layers": len(attn_layers),
                    "source": "measured",
                    "profile_ref": f"{self.pset.set_id}#attn_decode(B={B},kv={S})",
                    "t_ms": round(blk * len(attn_layers), 6),
                    "rows": len(attn_layers), "kernel_count_per_layer": kc,
                    "analytic_replaced_ms": round(analytic_a, 6)})
            else:
                reason = (f"Q={Q}>1 非 decode 形状（Q=1 域）" if Q != 1 else
                          f"(B={B}, kv={S}) 在 decode 注意力网格外"
                          f"（域 {self.pset.manifest['domain']['attn_decode']}）")
                groups_out.append({
                    "group": "attn_decode_block", "layers": 0, "source": "analytic",
                    "reason": reason, "t_ms": round(analytic_a, 6),
                    "rows": len(attn_layers)})

        # 差额法回退：analytic 总量 − 被 measured 组替换的 analytic 部分
        # （MoE/router/lm_head/GDN/域外组等全部自动落入回退，不重不漏）
        replaced_analytic = sum(row_price[k] for k in covered)
        fallback_ms = analytic_total - replaced_analytic
        # 间隙只补计时区间**边界**（op 内间隙已在 CUDA_EVENT median 内，不重复计费）
        gap_us = self.pset.eager_gap_us(tp=self.tp)
        gap_ms = regions * gap_us * 1e-3
        selected = measured_ms + fallback_ms + gap_ms
        return {
            "stage": stage, "domain_ok": True, "reasons": [],
            "analytic_ms": round(analytic_total, 6),
            "selected_ms": round(selected, 6),
            "measured_ms": round(measured_ms, 6),
            "fallback_ms": round(fallback_ms, 6),
            "gap_allowance_ms": round(gap_ms, 6),
            "gap_policy": f"eager {gap_us} µs × {regions} 计时区间边界"
                          f"（区间内间隙已含于实测 median；cuda_graph 档 "
                          f"{self.pset.graph_gap_us} µs 备选，未采用）",
            "kernels": kernels,
            "gap_regions": regions,
            "groups": groups_out,
            "coverage_rows": [len(covered), len(rows)],
            # 时间口径覆盖：依赖 selected（含解析回退与间隙估计），非真实性度量
            "coverage_time_frac": round(measured_ms / selected, 4) if selected > 0 else None,
            "coverage_time_note": "measured/selected；分母含解析回退与间隙估计",
        }
