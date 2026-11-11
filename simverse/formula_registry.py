"""v0.2 公式注册表：全仓唯一公式维护点（去重字典，行内只引用 id）。

与 v0.1 formulas.py 的关系：v0.1 属 legacy 切片继续服务旧路径；本表服务
v0.2 管线（count/cost/execute），语义按"量"绑定——同一 Attention 条目不得
用 GEMM 的公式 id 解释 KV 读或时间。每条携带 expr + source + 适用条件。
"""
from __future__ import annotations

from typing import Any, Dict

REGISTRY: Dict[str, Dict[str, str]] = {}


def _f(fid: str, expr: str, source: str, applies: str = "") -> str:
    if fid in REGISTRY:
        raise ValueError(f"公式 id 重复: {fid}")
    REGISTRY[fid] = {"expr": expr, "source": source,
                     "applies": applies or "通用", "version": "0.2"}
    return fid


# ---- Count：逻辑工作量（整数精确） ----
F_GEMM = _f("v2.gemm", "FLOPs = 2·T·IC·OC（一次乘加 2 FLOPs）", "约定：原理 §2",
            "T=batch·查询数 的 GEMM 行")
F_CAUSAL_PAIRS = _f("v2.causal_pairs", "pairs = B·(S·Q + Q(Q+1)/2)",
                    "原理 §2：历史 S 与新查询 Q 不重复计数", "full attention 因果窗口")
F_SWA_PAIRS = _f("v2.swa_pairs", "pairs_layer = causal(n,1,win) − causal(start,1,win)",
                 "原理 §3：滑窗层可见对", "sliding attention 层")
F_ATTN_CORE = _f("v2.attn_core", "FLOPs = pairs·H_card·(4d+5)",
                 "QK 2d + AV 2d + scale/softmax ≈5（v0.1 同约定）", "逐头注意力核")
F_ATTN_OUT_GATE = _f("v2.attn_output_gate",
                     "flops = 2·T·(heads·d)（sigmoid+乘，逐元素）",
                     "hf modeling_qwen3_5_moe Qwen3_5MoeAttention L802"
                     "（R06 CPU fixture 对照发现）",
                     "qwen3_5_moe full-attn 层")
F_ROPE = _f("v2.rope", "ops = 3·T·heads·rotary_dim（每旋转元素 3 OP）",
            "v0.1 rope() 约定；部分旋转只转 rotary_dim", "RoPE 层")
F_RMSNORM = _f("v2.rmsnorm", "ops = T·(4·w+2)；rsqrt 记 1 OP",
               "v0.1 norm() 约定", "RMSNorm（w=宽度）")
F_RESIDUAL = _f("v2.residual", "FLOPs = T·h（残差逐元素加法不因融合消失）",
                "E09 纠正：逻辑算量保留", "残差分支")
F_KV_RW = _f("v2.kv_rw", "kv_read/write = rows·(2·H_kv·d)·B_kv（GQA 族）",
             "原理 §3：每 token 每层 KV 元素 2·H_kv·d", "GQA/SLIDING 层")
F_GDN_STATE = _f("v2.gdn_state",
                 "state_elems = N_v·d_k·d_v；conv = (2K+V)·(kernel−1)（每请求常数）",
                 "hf modeling_qwen3_5_moe Qwen3_5MoeGatedDeltaNet", "GDN linear 层")
F_GDN_STEP = _f("v2.gdn_step",
                "每 token：decay·S + k⊗v 外积 ≈ 4·state_elems OP；state 读+写各 state_elems 元素",
                "GatedDeltaNet 递推（解析计数约定，标注 approximated）", "GDN 递推步")
F_MLA_KV = _f("v2.mla_kv",
              "kv_elems/token/layer = kv_lora_rank + qk_rope_dim（V3 压缩布局）",
              "原理 §3；展开式 decode 每头 KV 另计", "DeepSeek-V3 MLA")
F_MOE_ROUTER = _f("v2.moe_router", "FLOPs = 2·T·h·E；weight_read = h·E·B_router",
                  "路由打分 GEMM", "MoE 层")
F_MOE_ROUTED = _f("v2.moe_routed", "FLOPs = 2·T·k·3·h·inter（精确，按 assignment）",
                  "原理 §4：逻辑 FLOPs 用有效执行 assignment", "routed 专家")
F_MOE_UNION = _f("v2.moe_union", "E[U(M)] = E·(1 − (1 − k/E)^M)（无放回 top-k，token 独立）",
                 "原理 §4 正式式；独立枚举黄金 E4/k2/M2=3", "期望权重读")
F_MOE_UNION_WITH_REPL = _f(
    "v2.moe_union_with_repl", "E·(1 − (1 − 1/E)^(M·k))（有放回近似，仅显式选用）",
    "原理 §4：M=1 时 < k，不可作缺省", "近似选项，须显式登记")
F_LM_HEAD = _f("v2.lm_head", "FLOPs = 2·T·h·V_card", "列切分 per card", "输出头")
F_EMBED = _f("v2.embed", "读 = T·h·B_e（查表）；容量 = h·V·B_e（untied）",
             "嵌入查表", "输入嵌入")
F_CAPACITY = _f(
    "v2.capacity",
    "per-rank 对象账本：steady=常驻+每请求；峰值=steady+max(阶段临时)+跨轮"
    "（串行配方）；共享对象零字节引用去重；未知运行时组件不填 0",
    "lowering/memory.py（R02a/R02b 生命周期分层）", "同卡显存账本/峰值")

# ---- mHC（mega hyper-connections，V4.1；证据 vllm deepseek_v41/nvidia/model.py
#       pin @f358f057，source-checked 2026-11-10）----
F_MHC_MIX = _f(
    "v2.mhc_mix",
    "ops = T·[2·IC·mix + 2·IC + mix+3]；IC=mh（首层 attn IC=h）；"
    "mix=2m+m²；GEMM MAC=IC·mix；split=s 时另加 (s−1)(mix+1)；"
    "weight_read=mix·IC·4，fn capacity=mix·mh·4，L0 broadcast=mix·h·4 另列",
    "vllm @f358f057 tilelang_kernels.py L904-912（GEMM/sqrsum）；"
    "L383-390（prenorm 合并与缩放）；model.py L1530-1543（派生 broadcast）",
    "unfused/split=1 参考；算术 op 精确，rsqrt=1；不含设备归约/冗余指令")
F_MHC_SINKHORN = _f(
    "v2.mhc_sinkhorn",
    "ops = T·{8m+2m²+[2m(m−1)+4m²]+(2·iters−1)[m(m−1)+m+m²]}；"
    "m=4,iters=20 → 1400/token；仅 comb m×m 做 Sinkhorn",
    "vllm @f358f057 tilelang_kernels.py L395-443（无 norm 变体 L215-265）；"
    "torch.py L108-123：gates、行 softmax+eps、列归一、iters−1×(行,列)",
    "有限输入/save_pre_mix=True；exact 算术 op 公式：max 比较/exp/sigmoid "
    "各计 1 op，归约 n−1 次加，分母 eps 按轴复用；非指令数/实测 FLOPs")
F_MHC_POST_PRE = _f(
    "v2.mhc_post_pre",
    "ops = T·{[mh+2m²h]post + 2mh collapse + (4h+2)norm}；"
    "首层 attn 无 post；hc_exit_post 仅取 post 项，末层无后继边界",
    "vllm @f358f057 tilelang_kernels.py L766-769/L856-859（post），"
    "L447-450/L478-481（首层 one-hot/collapse），L491-524（norm）；"
    "model.py L1273（exit post）",
    "post/collapse 算术精确；norm 沿用 4h+2 近似惯例（非 kernel 指令数）；"
    "unfused/split=1 逻辑访问参考，保留每层 2 边界三行，无通用 norm/add 重计")
F_MHC_EXIT_COLLAPSE = _f(
    "v2.mhc_exit_collapse", "ops = 2·T·m·h（零初始化累加）；数学最简 (2m−1)·T·h",
    "vllm @f358f057 triton.py L96-109（acc=0; acc+=pre*x），L144-145 "
    "enable_fp_fusion=False；model.py L1187-1192；无 learned hc_head",
    "exit post 归 F_MHC_POST_PRE 的 post 项；既有 final_norm 行不变")
F_MHC_EXPAND = _f("v2.mhc_expand", "ops=0；读=T·h·b，写=T·m·h·b（首层广播）",
                  "ops/mhc.py L326-355：一次 load，各流 store，无浮点算术", "首层流广播")

# ---- Engram（DeepSeek-V4.1-Flash；证据 vllm/models/deepseek_v41/common/
#       engram.py pin @f358f057 blob sha1 ba4ceea6，source-checked 2026-11-11
#       R04；注入点 nvidia/model.py L583-600，批级哈希 L1030-1092）----
F_ENGRAM_HASH = _f(
    "v2.engram_hash",
    "FLOPs = T·n_engram_layers·(2·max_ngram + 2·(max_ngram−1)·n_heads) "
    "（V4.1: T·2·56）；读 = T·4（int32 ids）；写 = T·n_layers·24·4（int32 哈希）",
    "engram.py L291-423 _hash_ids_kernel：每 (token,layer,shift) 滚动哈希 "
    "1 mul+1 xor ×MAX_NGRAM=4；shift>0 每 head 1 mod+1 add ×3×8；"
    "grid (cdiv(T,32), n_layers) 一次算全部 engram 层（L565）——整数 op "
    "计数口径，非 kernel 指令数",
    "Engram 批级 n-gram 哈希（整批一次，挂首 engram 层行）")
F_ENGRAM_LOOKUP = _f(
    "v2.engram_lookup",
    "per-rank FLOPs = T·part_n_hash_cols·256（FP8→bf16 dequant 逐元素 1 乘）；"
    "weight_read_expected = T·part_n_hash_cols·264（表行读期望流量，不去重保守）；"
    "输出 = T·part_n_hash_cols·256·2B（固定 bf16）；"
    "容量 = per-rank 行数·264（素数桶 head 分片，非 num_embeddings/tp）",
    "engram.py L642-732 _engram_lookup_kernel（ue8m0 byte<<23 即 fp32 指数，"
    "L726-731）；264 B/行 = FP8 256 + E8M0 256/32（L968-969/L1151-1161）；"
    "桶长/TP 分片见 model_ir.families.engram_bucket_primes（L211-252 移植）",
    "Engram 查表（每 engram 层；表行读为数据依赖期望流量，容量另计）")
F_ENGRAM_WKV = _f(
    "v2.engram_wkv",
    "FLOPs = 2·T·6144·(25600/tp)；weight/容量 = 6144·(25600/tp)·B_w（列切）；"
    "act_read = T·6144·2B；act_write = T·(25600/tp)·2B（固定 bf16）；"
    "gather_output=True → all-gather 见通信行",
    "engram.py L1719-1733 ColumnParallelLinear(n_hash_cols·head_dim=6144 → "
    "hidden·(hc_mult+1)=25600, bias=False, gather_output=True)；"
    "无 sequence parallel 时输出列切",
    "Engram wkv 投影（每 engram 层，per-rank 列切口径）")
F_ENGRAM_POST_WKV = _f(
    "v2.engram_post_wkv",
    "FLOPs = T·hc_mult·(10·d+5)（乘 6d + 加 3(d−1)+d + 标量≈8；d=hidden）；"
    "weight/容量 = 2·hc·d·2（q/k_weight bf16 每 rank 复制）；"
    "hidden/kv 输入与 output 按 pinned bf16 2B/元素",
    "engram.py L1334-1422 _fused_engram_post_wkv_kernel（每 (token,hc)："
    "ρ=(Σ hidden·q·k·key)·rsqrt(mean h²+eps)·rsqrt(mean key²+eps)·rsqrt(d)；"
    "gate=sigmoid(sign(ρ)·sqrt(clamp|ρ|,1e-6))；out=hidden+gate·value）；"
    "eps=rms_norm_eps=1e-20、clamp=1e-6（L1695-1696）——op 计数为解析近似口径",
    "Engram 门控注入核（每 engram 层，post_wkv 逐 (token,hc)）")

# ---- Cost（T0 解析估算；Q2：无实测数字） ----
F_ROOFLINE = _f("v2.roofline", "t = max(F/P_eff, Q/BW_eff) + ρ·min(...)；ρ=0.15 登记",
                "原理 §5：乐观约束 + 显式经验尾部", "kernel 级 T0")
F_LAUNCH = _f("v2.launch", "t_launch = n_kernels·launch_ms（eager 上界）",
              "未建模 CUDA Graph 时的上界", "执行配方")
F_LINK = _f("v2.link", "t = fixed + bytes/BW（单方向单消息）", "原理 §6", "点到点链路")
F_COLLECTIVE = _f("v2.collective",
                  "ring allreduce: bytes/rank = 2·(N−1)/N·msg；时间 = 2(N−1)/N·msg/BW + 2(N−1)·lat",
                  "ring 算法模型（N=ranks）", "TP/EP 集合通信")

# ---- 投机方法（链式/DFlash；M2 语义在 methods/ 绑定） ----
F_EN_CHAIN = _f("v2.en_chain", "E[N] = 1 + Σ_{i=1..K} Π_{j≤i} α_j",
                "原理 §7：逐位条件接受率（几何仅特例）", "链式验收")
F_EN_BLOCK = _f("v2.en_block",
                "DFlash 块验收 E[N] = 1 + Σ_{i=1..K_blk} Π_{j≤i} β_j（块内逐位同链式）",
                "z-lab/dflash _rejection_sample：逐位拒绝+残差 bonus", "块草稿验收")
F_DFLASH_MSG = _f("v2.dflash_msg",
                  "T→D hidden 消息 = produced·Σ_{l∈target_ids} h·B_act（fc 输入拼接）；"
                  "D→T = K_blk·B_token",
                  "z-lab/dflash dflash_generate：target_hidden 按层拼接、produced 截断",
                  "DFlash 协议消息")


def formulas_json() -> dict:
    return {"schema_version": "0.2", "formulas": REGISTRY}


def get(fid: str) -> Dict[str, Any]:
    if fid not in REGISTRY:
        raise KeyError(f"未注册公式 {fid}")
    return REGISTRY[fid]
