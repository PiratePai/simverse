"""独立手算期望（手算 oracle）：与实现分离存放（重建计划 §"怎样判定算对了"）。

tiny_dense_oracle · B=1, ctx=100, K=2, α=0.5, 16-bit（2 B/elem），TP=1。
所有数值由人工按公式逐步推导（表达式即演算过程），不 import simverse。
模型：2 层 GQA，h=256，Q 头 8 × d=32，KV 头 2，FFN 512（SwiGLU 3 矩阵），
vocab 1000；每层每 token KV 元素 = 2(K+V)×2 头×32 = 128。
"""

# ---- 模型常数 ----
H, HEADS, KV_HEADS, D, FFN, VOCAB, LAYERS = 256, 8, 2, 32, 512, 1000, 2
B, CTX, K, ALPHA = 1, 100, 2, 0.5
WB = AB = KB = 2  # 16-bit → 2 B/elem
KV_ELEMS = 2 * KV_HEADS * D            # 128

# ================= 基线 decode 单步（Q=1, T=1） =================
PAIRS_BASE = B * (CTX * 1 + 1 * 2 // 2)            # = 100+1 = 101

Q_PROJ_FLOPS = 2 * 1 * H * (HEADS * D)             # 2·256·256 = 131072
Q_PROJ_WEIGHT = H * (HEADS * D) * WB               # 131072 B
KV_PROJ_FLOPS = 2 * 1 * H * (KV_HEADS * D)         # 32768
KV_PROJ_WEIGHT = H * (KV_HEADS * D) * WB           # 32768 B
ATTN_CORE_FLOPS = PAIRS_BASE * HEADS * (4 * D + 5) # 101·8·133 = 107464
KV_READ_BASE_LAYER = B * CTX * KV_ELEMS * KB       # 100·128·2 = 25600 B
KV_WRITE_BASE_LAYER = 1 * KV_ELEMS * KB            # 256 B
GATE_UP_FLOPS = 2 * 1 * H * (2 * FFN)              # 524288
GATE_UP_WEIGHT = H * (2 * FFN) * WB                # 524288 B
DOWN_FLOPS = 2 * 1 * FFN * H                       # 262144
DOWN_WEIGHT = FFN * H * WB                         # 262144 B
NORM_OPS = 1 * (4 * H + 2)                         # 1026
NORM_WEIGHT = 2 * H                                # 512 B

LAYER_FLOPS_BASE = (Q_PROJ_FLOPS + 2 * KV_PROJ_FLOPS + ATTN_CORE_FLOPS + Q_PROJ_FLOPS
                    + GATE_UP_FLOPS + DOWN_FLOPS + 2 * NORM_OPS)      # 1223628
LM_HEAD_FLOPS = 2 * 1 * H * VOCAB                  # 512000
LM_HEAD_WEIGHT = H * VOCAB * WB                    # 512000 B
EMBED_CAPACITY = H * VOCAB * WB                    # 512000 B

TOTAL_FLOPS_BASE = LAYERS * LAYER_FLOPS_BASE + LM_HEAD_FLOPS          # 2959256
TOTAL_WEIGHT_BASE = LAYERS * (2 * Q_PROJ_WEIGHT + 2 * KV_PROJ_WEIGHT
                              + GATE_UP_WEIGHT + DOWN_WEIGHT + 2 * NORM_WEIGHT) \
                    + LM_HEAD_WEIGHT                                    # 2742272
TOTAL_KV_READ_BASE = LAYERS * KV_READ_BASE_LAYER    # 51200
TOTAL_KV_WRITE_BASE = LAYERS * KV_WRITE_BASE_LAYER  # 512

# ================= 验证步（Q=K+1=3, T=3） =================
PAIRS_VERIFY = B * (CTX * 3 + 3 * 4 // 2)           # 300+6 = 306
GEMM3 = 3  # GEMM/激活按 T=3 放大
VERIFY_ATTN_FLOPS = PAIRS_VERIFY * HEADS * (4 * D + 5)               # 306·8·133 = 325536
VERIFY_LAYER_FLOPS = ((Q_PROJ_FLOPS + 2 * KV_PROJ_FLOPS + Q_PROJ_FLOPS
                       + GATE_UP_FLOPS + DOWN_FLOPS) * GEMM3
                      + VERIFY_ATTN_FLOPS + 2 * 3 * NORM_OPS)
TOTAL_FLOPS_VERIFY = LAYERS * VERIFY_LAYER_FLOPS + LM_HEAD_FLOPS * GEMM3
TOTAL_KV_READ_VERIFY = TOTAL_KV_READ_BASE          # 3 查询位共享一次历史 KV 读
TOTAL_KV_WRITE_VERIFY = LAYERS * 3 * KV_WRITE_BASE_LAYER

# ================= 投机数学（链式） =================
E_N = 1 + ALPHA + ALPHA ** 2                       # = 1.75
P_FULL = ALPHA ** K                                # = 0.25
E_M0 = 1 + P_FULL                                  # = 1.25

# ================= 通信（custom_chain） =================
BYTES_D2T = B * (K + 1) * 6                        # 18 B（token id 4B + q 2B）
BYTES_T2D = B * (E_N * 0 + 256)                    # 256 B（无 per-token 回传）

# ================= 草稿（custom_chain 1.8B, FP8） =================
DRAFT_PARAMS = 1_800_000_000
DRAFT_M1_FLOPS = 2 * 1 * DRAFT_PARAMS
DRAFT_M1_WEIGHT = DRAFT_PARAMS * 1                 # 8bit = 1 B/param
DRAFT_M2_FLOPS = 2 * 2 * DRAFT_PARAMS
DRAFT_KV_FRAC = 0.1125
DRAFT_KV_READ_M1 = int(TOTAL_KV_READ_BASE * DRAFT_KV_FRAC)   # 5760
DRAFT_KV_READ_M2 = int(TOTAL_KV_READ_BASE * 2 * DRAFT_KV_FRAC)  # 11520（参考步 batch=M）

# ================= roofline 抽查（q_proj 行，H100 SXM） =================
H100_FP16 = 989.5e12
H100_BW = 3.35e12
MFU = 0.75
QPROJ_BYTES = Q_PROJ_WEIGHT + 1 * H * AB + (HEADS * D) * AB   # 131072+512+512 = 132096
QPROJ_T_COMP = Q_PROJ_FLOPS / (H100_FP16 * MFU) * 1e3
QPROJ_T_MEM = QPROJ_BYTES / H100_BW * 1e3
QPROJ_T = max(QPROJ_T_COMP, QPROJ_T_MEM) + 0.15 * min(QPROJ_T_COMP, QPROJ_T_MEM)
