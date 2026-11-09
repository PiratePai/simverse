"""独立 oracle（v0.2 管线）：全部由人工推导/枚举书写，不 import simverse。

覆盖（对应 V03/V05/V08 与 §7.7/§7.8 反例）：
  * MoE 并集：小集合穷举 + 组合比值式 + 端点（M=1 → k；有放回近似在 M=1 时 < k）
  * 接受链：逐位 α_i、端点 α=0/1、mean 不决定 Pfull 的双分布反例
  * 动态 K 联合律反例：同边际不同联合 → E[N] 2.06/1.98、E[T]/E[N]≈3.8835/4.0404
  * GEMM/因果对数/残差加法/RoPE 计数（小尺寸手算）
  * GQA/MLA 每 token KV 元素
  * 时间与指标人工轨迹：TPOT/吞吐/窗口 token 计数/goodput/busy 积分/分位数
编写规则：数值以表达式书写（表达式即演算过程），常量只用于不可再分的输入。
"""
from itertools import combinations
from math import comb, prod

# ===========================================================================
# 1. MoE 并集 U(M) —— E 专家、每 token 无放回均匀选 k 个不同专家、M token 独立
# ===========================================================================

def moe_union_exact_enumeration(E: int, k: int, M: int) -> float:
    """穷举所有 token-专家子集组合，求被命中专家数的期望（M 很小时可行）。"""
    subsets = list(combinations(range(E), k))
    total_pairs = len(subsets) ** M
    acc = 0
    stack = [((), 0)]  # (已选专家元组的集合 frozenset 逐层展开)
    # 逐 token 展开（M 小；用动态规划数集合出现次数）
    from collections import Counter
    counts = Counter({frozenset(): 1})
    for _ in range(M):
        nxt: Counter = Counter()
        for seen, c in counts.items():
            for s in subsets:
                nxt[seen | frozenset(s)] += c
        counts = nxt
    assert sum(counts.values()) == total_pairs
    return sum(len(seen) * c for seen, c in counts.items()) / total_pairs


def moe_union_ratio_formula(E: int, k: int, M: int) -> float:
    """独立推导路径：P(e 未被一个 token 选中) = C(E-1,k)/C(E,k) = (E-k)/E；
    M token 独立 → E[U] = E·(1-((E-k)/E)^M)。"""
    p_unused_one = comb(E - 1, k) / comb(E, k)
    return E * (1.0 - p_unused_one ** M)


def moe_union_with_replacement(E: int, k: int, M: int) -> float:
    """有放回近似（错误参照系）：把 M·k 次选择当独立抽签。M=1 时 < k。"""
    return E * (1.0 - (1.0 - 1.0 / E) ** (M * k))


# 小集合黄金值（穷举 = 真值）
MOE_E4_K2_M2_ENUM = moe_union_exact_enumeration(4, 2, 2)          # = 3.0
MOE_E4_K2_M2_RATIO = moe_union_ratio_formula(4, 2, 2)             # = 3.0（与穷举一致）
MOE_E4_K2_M2_WITH_REPL = moe_union_with_replacement(4, 2, 2)      # = 2.734375（≠真值）
MOE_E6_K3_M3_ENUM = moe_union_exact_enumeration(6, 3, 3)          # = 5.25
MOE_E6_K3_M3_RATIO = moe_union_ratio_formula(6, 3, 3)             # = 5.25
MOE_E6_K3_M3_WITH_REPL = moe_union_with_replacement(6, 3, 3)      # ≠ 5.25
MOE_E384_K6_M5_RATIO = moe_union_ratio_formula(384, 6, 5)         # ≈ 29.077034（评审值）
MOE_E384_K6_M5_WITH_REPL = moe_union_with_replacement(384, 6, 5)  # ≈ 28.894244（≠）
MOE_M1_GIVES_K = moe_union_ratio_formula(64, 8, 1)                # = 8.0（端点：单 token 恰 k）
MOE_M1_WITH_REPL_LT_K = moe_union_with_replacement(64, 8, 1)      # < 8.0（反例端点）


# ===========================================================================
# 2. 链式接受：逐位条件接受率 α_i → E[N] = 1 + Σ_{i=1..K} Π_{j≤i} α_j
# ===========================================================================

def en_from_position_alphas(alphas: list) -> float:
    """E[N] 含 bonus；逐位置可以不同（不要求几何）。"""
    e = 1.0
    p_prefix = 1.0
    for a in alphas:
        p_prefix *= a
        e += p_prefix
    return e


EN_ALPHA0 = en_from_position_alphas([0.0, 0.0, 0.0, 0.0])   # = 1（全拒绝只剩 bonus）
EN_ALPHA1_K4 = en_from_position_alphas([1.0, 1.0, 1.0, 1.0])  # = 5 = K+1
EN_GEOM_075_K4 = en_from_position_alphas([0.75] * 4)        # = 3.05078125
P_FULL_GEOM_075_K4 = prod([0.75] * 4)                        # = 0.31640625

# mean 不决定分布（E07）：K=2、E[N]=2 的两个世界
MEAN_WORLD_A = {"N_dist": {2: 1.0}, "mean": 2.0, "P_full": 0.0}
MEAN_WORLD_B = {"N_dist": {1: 0.5, 3: 0.5}, "mean": 2.0, "P_full": 0.5}


# ===========================================================================
# 3. 动态 K 联合律反例（§7.7 原例；同边际不同联合）
# ===========================================================================

def _dyn_world(alphas_by_k: dict, alpha2_given_first: float, k_dist: dict,
               t_by_k: dict) -> dict:
    """构造联合分布的 E[N]/E[T]/E[T]/E[N]。K=1/2；第 2 位接受以第 1 位接受为条件。"""
    en, et = 0.0, 0.0
    for k, pk in k_dist.items():
        a1 = alphas_by_k[k]
        if k == 1:
            n_mean = 1.0 + a1
        else:
            n_mean = 1.0 + a1 + a1 * alpha2_given_first if k == 2 else None
        en += pk * n_mean
        et += pk * t_by_k[k]
    return {"E_N": en, "E_T": et, "TPOT": et / en}


_K_DIST = {1: 0.5, 2: 0.5}
_T_BY_K = {1: 6.0, 2: 10.0}
DYN_WORLD_A = _dyn_world({1: 0.5, 2: 0.9}, 0.8, _K_DIST, _T_BY_K)  # α1 依赖 K
DYN_WORLD_B = _dyn_world({1: 0.7, 2: 0.7}, 0.8, _K_DIST, _T_BY_K)  # α1 独立
# 两世界 α1 边际同为 0.5*0.5+0.5*0.9 = 0.7 = 0.7
DYN_MARGINAL_A = 0.5 * 0.5 + 0.5 * 0.9
DYN_MARGINAL_B = 0.7
# 期望值（供断言）：A: E[N]=2.06, TPOT=8/2.06≈3.8835；B: E[N]=1.98, TPOT≈4.0404
assert abs(DYN_WORLD_A["E_N"] - 2.06) < 1e-12
assert abs(DYN_WORLD_B["E_N"] - 1.98) < 1e-12
assert abs(DYN_MARGINAL_A - DYN_MARGINAL_B) < 1e-12


# ===========================================================================
# 4. 小尺寸计数（V03）
# ===========================================================================

GEMM_2MNK = lambda M_, N, K: 2 * M_ * N * K
# 例：X[4,8]@W[8,16] → 1024 FLOPs；权重字节 = 8*16*2(B) = 256
GEMM_CASE = {"M": 4, "N": 16, "K": 8, "flops": GEMM_2MNK(4, 16, 8),
             "weight_b16": 8 * 16 * 2, "act_in_b16": 4 * 8 * 2,
             "act_out_b16": 4 * 16 * 2}

def causal_pairs(S: int, Q: int) -> int:
    """历史 S + 新查询 Q 的可见 q-k 对（因果，头内）；P = S·Q + Q(Q+1)/2。"""
    return S * Q + Q * (Q + 1) // 2

CAUSAL_S100_Q1 = causal_pairs(100, 1)   # 101
CAUSAL_S100_Q3 = causal_pairs(100, 3)   # 306
RESIDUAL_ADDS_M4_D256 = 4 * 256         # 1024（E09：残差加法不因融合消失）
GQA_KV_ELEMS = lambda kv_heads, d: 2 * kv_heads * d     # 每 token 每层
MLA_COMPRESSED_ELEMS = lambda kv_lora, rope: kv_lora + rope  # V3 布局每 token 每层


# ===========================================================================
# 5. 时间与指标人工轨迹（V08 oracle；M3 reducer 的独立期望）
# ===========================================================================
# 请求 r1：arrival=0，tokens emitted(server_emitted) at 10,20,30,40 → 4 输出 token
# 请求 r2：arrival=5，emitted at 25, 25（同批同刻，允许）→ 2 token
# 窗口 [0,50)：窗口内 token 事件 = 6；窗口内完成请求 = 2（都完成）
# r1 平均间隔（M01 请求口径）= (40-10)/(4-1) = 10；r2 间隔 = 0/1 → 不适用（N≤1）
# TPOT kind=server_emitted；吞吐（按完成请求 token/墙钟）= (4+2)/50 = 0.12 tok/ms
# goodput SLO（ITL≤15ms）：r1 满足（间隔均 10），r2 间隔 0 满足 → goodput=6/50
# busy 积分：资源 X busy [0,12) 与 [20,33) → busy=25；重叠 [30,33)∪[20,30) 不双计
TRACE_TOKENS = {"r1": [10, 20, 30, 40], "r2": [25, 25]}
R1_MEAN_ITL = (40 - 10) / (4 - 1)          # 10.0
WINDOW_TOKEN_COUNT = 6
WINDOW_COMPLETED_REQUESTS = 2
THROUGHPUT_COMPLETED = 6 / 50.0
GOODPUT_SLO15 = 6 / 50.0
BUSY_SEGMENTS = [(0, 12), (20, 33)]
BUSY_INTEGRAL = (12 - 0) + (33 - 20) - max(0, min(33, 12) - max(20, 0))  # 无重叠 → 25

# 分位数 estimator（最近秩/nearest-rank）：样本 [10,20,30,40] 的 p50=20, p75=30
QUANT_NN_P50 = 20.0
QUANT_NN_P75 = 30.0


# ===========================================================================
# 6. 通信/roofline 手算锚点（V04-cost §7.8）
# ===========================================================================
ROOFLINE_ANCHOR = {"F": 1000.0, "P": 100.0, "Q": 200.0, "BW": 50.0,
                   "t_comp": 1000 / 100, "t_mem": 200 / 50,
                   "t_roofline": max(10.0, 4.0)}   # = 10 s（独立手算）
LINK_ANCHOR = {"bytes": 60000, "bw": 64e9, "fixed_s": 100e-6,
               "t_transfer_s": 60000 / 64e9, "t_total_s": 100e-6 + 60000 / 64e9}
# 60KB @64GB/s ≈ 0.9375 µs ≪ 100 µs 固定延迟（评审纠正 ZCode 量级错误的锚点）
