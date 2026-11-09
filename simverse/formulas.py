"""公式注册表：全仓唯一公式维护点（方案 §2 防冗余规则 1）。

每个公式一个 id → {expr: 人类可读表达式, source: 出处}。
OpRecord/StageTiming 只引用 id；Web 前端与报告用 /api/formulas 渲染，不在前端重写公式。

来源标签约定（精度阶梯，原理 v1.1 §2.2）：
  T0   = roofline 公式估算（首发全部数字属此级，决策 D-4）
  锚   = 旧仓对账锚点（文件:行号）
  spec = 硬件/官方公开规格
  config = 模型 config.json（固定快照 + SHA256）
"""
from typing import Dict

FORMULAS: Dict[str, Dict[str, str]] = {}


def _f(fid: str, expr: str, source: str) -> str:
    if fid in FORMULAS:
        raise ValueError(f"公式 id 重复: {fid}")
    FORMULAS[fid] = {"expr": expr, "source": source}
    return fid


# ---- Count 段（整数算术） ----
F_GEMM_FLOPS = _f("gemm_flops", "FLOPs = 2·M·N·K", "T0:矩阵乘标准计数")
F_GEMM_WBYTES = _f("gemm_weight_bytes", "weight_read = N·K·bytes_per_param", "T0:权重按列主读取")
F_ACT_BYTES = _f("act_bytes", "act = tokens·dim·bytes_per_elem", "T0:激活按形状×精度")
F_KV_RW = _f("kv_rw", "kv_read/write = rows·per_row_bytes", "T0:KV 按行×每行字节")
F_UNION_IID = _f(
    "moe_union_iid",
    "E[U(M)] = E_routed·(1 − (1 − k_top/E_routed)^M)",
    "T0:i.i.d. 均匀路由近似; 精确组合式 E·(1−C(E−k,M)/C(E,M)); 锚 piviewer/roofline.py:141 _expected_unique",
)
F_EN_CHAIN = _f(
    "en_chain",
    "E[n] = Σ_{i=0..K} α^i = (1 − α^{K+1})/(1 − α)（含 bonus）",
    "T0:链式独立同分布接受; 锚 旧仓 spec_gain_model.js computeExpectedTau / 方案 §4.1",
)
F_CATCHUP = _f(
    "draft_catchup",
    "E[M_draft,0] = 1 + α^K（上轮全接受时下一轮首步须喂入 [d_K, bonus] 两 token）",
    "T0:方案 v2.2 §4.5 增量② RoundTransitionSpec; 旧仓未建模（差异报告条目 D-2）",
)
F_KV_VERIFY = _f(
    "kv_verify_factor",
    "KV 读放大 = (ctx + K)/ctx（验证步查询 K+1 个位置，读 ctx+K 长 KV）",
    "T0:锚 旧仓 spec_gain_model.js:79 kvVerifyBytesGB 系数 (1+γ/ctx)",
)
F_COMM_BYTES = _f(
    "dt_comm_bytes",
    "D→T = B·(K+1)·6B (token id 4B + q 概率 2B)；T→D = B·(n̄·per_token + 256B 控制)",
    "T0:锚 旧仓 dt_sim.py:185 / spec_gain_model.js computeDtTransfer",
)

# ---- Cost 段（roofline T0） ----
F_ROOFLINE = _f(
    "roofline_max",
    "t = max(FLOPs/peak_eff, bytes/bw) + 0.15·min(...) ; peak_eff = peak·MFU_cap",
    "T0:roofline 上界 + 15% 重叠尾部; MFU 0.75 锚 旧仓 spec_gain_model.js:43 clusterFlops",
)
F_LAUNCH = _f(
    "launch_overhead",
    "t_launch = launch_overhead_ms × 未融合算子数（CUDA Graph 未建模时的上界估计）",
    "T0:方案 §4.6 CUDA Graph 三态之 unsupported/eager 口径",
)
F_LINK = _f(
    "link_time",
    "t_msg = fixed_ms + bytes/bw（每方向各计一次启动延迟）",
    "T0:锚 旧仓 dt_sim.py transfer()",
)

# ---- Execute 段（解析组合器） ----
F_TPOT_BASE = _f("tpot_base", "TPOT_base = t_decode(B, ctx)", "T0:解析档")
F_TPOT_COLOCATED = _f(
    "tpot_colocated",
    "TPOT_spec = (Σ_j t_draft_step_j + t_verify) / E[n]",
    "T0:串行同卡; 锚 旧仓 spec_gain_model.js evaluateModel tStepSpecMs",
)
F_TPOT_DT_SERIAL = _f(
    "tpot_dt_serial",
    "TPOT_spec = (Σ_j t_draft_step_j + t_d2t + t_verify + t_t2d) / E[n]",
    "T0:DT 串行（无 foresight，草稿等反馈）; 锚 旧仓 spec_gain_model.js evaluateDisaggregated dt_serial",
)
F_TPOT_DT_PIPE_IDEAL = _f(
    "tpot_dt_pipe_ideal",
    "TPOT_spec = (max(Σ t_draft + t_d2t, t_verify) + t_t2d) / E[n]",
    "T0:dt_pipe_microbatch 稳态理想下界（无命中率/回退建模，乐观标注）; 方案 §4.4 增量③",
)
F_SPEEDUP = _f("speedup", "speedup = TPOT_base / TPOT_spec", "T0")
F_BREAK_EVEN = _f(
    "break_even_alpha",
    "α* : E[n](α*) = TPOT_spec(α*)/TPOT_base（speedup=1 反解，二分）",
    "T0:锚 旧仓 spec_gain_model.js findRequiredAlpha",
)


def formulas_json() -> dict:
    return {"schema_version": "0.1", "formulas": FORMULAS}
