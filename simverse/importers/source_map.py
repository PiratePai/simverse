"""源码映射（V02）：模型语义分支 → 固定源码位置 + 谓词 + 两侧用例。

每条 SourceMapEntry：
  * branch_predicate：实际源码谓词（不猜测阈值）；
  * source_location：pin 仓库的 文件:行 或 类/函数名；
  * input_domain / thresholds：谓词的输入域与阈值；
  * covered：两侧是否都有结构/计数用例；
  * uncovered_domain：如实标注未执行/未覆盖的域（source-checked ≠ executed）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

VLLM_PIN = "vllm@f358f0578a83f4e1fe6fc49a492d3ae9362de8e5"
VLLM_PARALLEL_PIN = "vllm@dc74b8cbbbe3dbb9399fe369757d63defd614cc2"
HF_TF_PIN = "huggingface/transformers@b6401d6605d16620e0f95b54fb2906f92de7e33f"
DFLASH_PIN = "z-lab/dflash@07ebd93db9f472af339b644bb70221ad8428328a"


@dataclass
class SourceMapEntry:
    entry_id: str
    model_or_method: str
    semantic: str                       # 结构/分支语义
    source_pin: str
    source_location: str
    branch_predicate: str = ""
    thresholds: str = ""
    input_domain: str = ""
    both_sides_covered: bool = False
    evidence_kind: str = "source-checked"
    uncovered_domain: List[str] = field(default_factory=list)
    note: str = ""

    def to_json(self) -> dict:
        return dict(self.__dict__)


ENTRIES: List[SourceMapEntry] = [
    SourceMapEntry(
        entry_id="sm-qwen35-gdn-decode-branch",
        model_or_method="qwen3_5_moe/qwen3_next GDN",
        semantic="单 token 解码走缓存递推（fused kernel），多 token/prefill 走 chunk 路径",
        source_pin=HF_TF_PIN,
        source_location="modeling_qwen3_5_moe.py:558/607 "
                        "(Qwen3_5MoeGatedDeltaNet.forward)",
        branch_predicate="conv_state is not None and seq_len == 1 and not "
                         "cache_params.is_recording_past(layer_idx)",
        thresholds="seq_len==1；recording_past False",
        input_domain="decode(seq_len=1) vs prefill/verify(seq_len>1)",
        both_sides_covered=True,
        uncovered_domain=["fused kernel 内部 tile 行为未执行"],
        note="计数层两分支同代价（递推逐步）；核内融合差异属 lowering"),
    SourceMapEntry(
        entry_id="sm-qwen35-attn-output-gate",
        model_or_method="qwen3_5_moe full attention",
        semantic="q_proj 输出 2×（query+gate chunk）；gate 经 ACT2FN 后乘输出",
        source_pin=HF_TF_PIN,
        source_location="modeling_qwen3_5_moe.py:731-805 (Qwen3_5MoeAttention；"
                        "q_proj 2× at 743-745，chunk at 771-773，gate 乘 at 802)",
        branch_predicate="q_proj 输出 view(..., head_dim*2) 后 chunk(2)",
        input_domain="全部 full attention 层（attn_output_gate 家族默认）",
        both_sides_covered=True,
        note="ModelIR AttentionIR.output_gated=True → q_out=2·heads·d（已进计数）；"
             "原 source_map 747-772 未覆盖 L802 gate 乘，2026-11-10 修正"),
    SourceMapEntry(
        entry_id="sm-qwen35-moe-router",
        model_or_method="qwen3_5_moe MoE",
        semantic="router 线性 h→E，softmax(fp32)，top-k，top-k 权重按和重归一",
        source_pin=HF_TF_PIN,
        source_location="modeling_qwen3_5_moe.py:864-881 "
                        "(Qwen3_5MoeTopKRouter)",
        branch_predicate="topk(router_probs, num_experts_per_tok)；"
                         "router_top_value /= sum(top-k)",
        input_domain="Qwen3.6-35B-A3B（256E top-8，norm_topk_prob 语义）",
        both_sides_covered=True,
        note="router 行按 2·T·h·E 精确计；routed 专家并集用无放回期望，"
             "softmax/renorm 的数值未执行"),
    SourceMapEntry(
        entry_id="sm-qwen35-moe-shared-gate",
        model_or_method="qwen3_5_moe MoE",
        semantic="共享专家输出经 sigmoid(shared_expert_gate(x)) 门控后与 routed 相加",
        source_pin=HF_TF_PIN,
        source_location="modeling_qwen3_5_moe.py:884-901 (Qwen3_5MoeSparseMoeBlock；"
                        "shared_expert at 889，gate at 890，sigmoid 乘 at 899)",
        branch_predicate="shared_expert_intermediate_size > 0 → shared+gate 常驻"
                         "（config 无 shared_expert_gate 键）",
        input_domain="Qwen3.6-35B-A3B（shared 512 + gate Linear(h,1)）",
        both_sides_covered=True,
        note="2026-11-10 修复：shared_gate=True 由 shared_intermediate>0 推导；"
             "moe_rows 每 MoE 层计 gate 行，memory 计 h 权重"),
    SourceMapEntry(
        entry_id="sm-qwen36-mtp-unverified",
        model_or_method="qwen3_5_moe MTP",
        semantic="config 声明 mtp_num_hidden_layers=1；原生 MTP 微头 forward 未 pin",
        source_pin=HF_TF_PIN,
        source_location="modeling_qwen3_5_moe.py:994/1807 "
                        "(_keys_to_ignore_on_load_unexpected=[^mtp.*])；"
                        "vllm_qwen35_pin/qwen3_next.py:847-848、qwen3_5.py:326/476 "
                        "（主模型丢弃 mtp.）",
        branch_predicate="MTP 权重不在主模型加载路径；仅 config 字段存在",
        input_domain="Qwen3.6-35B-A3B MTP1",
        both_sides_covered=False,
        uncovered_domain=["原生 MTP 微头 forward 无 pin",
                          "MTP 仅按目标首层模板估算，未执行、未实测"],
        note="2026-11-10 审查：MTP 路径必须标 unverified，不能进入部署排名"),
    SourceMapEntry(
        entry_id="sm-v41-index-short-branch",
        model_or_method="deepseek_v41_flash CSA2",
        semantic="短上下文走 eager 全索引旁路（无 top-k 选择打分）",
        source_pin=VLLM_PIN,
        source_location="vllm/models/deepseek_v41/nvidia/model.py（index 路径；"
                        "v0.1 移植口径 fast path）",
        branch_predicate="end//r <= index_topk",
        thresholds="压缩块数 ≤ topk(512) → eager",
        input_domain="ctx/r ≤ 512（r=2 → ctx ≤ 1024）vs 长上下文",
        both_sides_covered=True,
        uncovered_domain=["vLLM sparse-logits 启用与否的运行时分支未执行"],
        note="计数按不假定 sparse-logits 的保守全候选打分（E19）"),
    SourceMapEntry(
        entry_id="sm-v41-engram-hc",
        model_or_method="deepseek_v41_flash Engram/hc",
        semantic="hc(mHC) 混合流已建模（每层 2 次边界：post+系数 GEMM+sinkhorn+"
                 "collapse+RMSNorm 融合核；首层 hc_expand 广播；FP32 权重每卡复制）；"
                 "Engram 查表注入（层 1/14 residual 流，post/pre 分立）已建模："
                 "素数桶 FP8+E8M0 表 264B/行、TP head ceil 切分、批级 n-gram "
                 "哈希一次算全部层、wkv 6144→25600 列切 gather、q/k 门控注入",
        source_pin=VLLM_PIN,
        source_location="vllm/models/deepseek_v41/common/engram.py:211-252"
                        "（EngramLayout 素数桶分配）、426-604（NgramHashState："
                        "token_map int32[vocab]、rolling hash kernel grid "
                        "(cdiv(T,32),n_layers)、V1 slot cache 挂首本地层 SWA）、"
                        "642-732（_engram_lookup_kernel：FP8 行 + ue8m0 scale "
                        "byte<<23 dequant → bf16）、960-981（can_share 全表字节 "
                        "Σrows×(head_dim+head_dim//32)）、1021-1330"
                        "（ParallelEngramEmbedding：264B/行、TP head ceil 切分 "
                        "断言 (tp−1)·part<24、all-gather dim=1）、1334-1422"
                        "（_fused_engram_post_wkv_kernel 门控公式）、1664-1908"
                        "（Engram：dim=hidden、hc_mult、eps=1e-20、clamp=1e-6、"
                        "wkv ColumnParallelLinear gather_output=True、q/k_weight "
                        "bf16 复制、staged_rows max_num_batched_tokens）；"
                        "nvidia/model.py:350-361（构造）、563-567 与 583-600"
                        "（注入点：前一子层 post 分立 → Engram → mhc_pre）、"
                        "858-877（engram 层前一层不得 defer_finalize；调度细节"
                        "不建模）、1030-1092（批级哈希每 forward 一次）、"
                        "919-948（NgramHashState 挂第一个本地层 SWA cache）",
        branch_predicate="engram_layer_ids 覆盖层启用 Engram；hc_mult/hc_eps/"
                         "hc_sinkhorn_iters 控制 mhc 边界（全层执行，非可选开关）；"
                         "cpu_offload 默认路径（表在 host pinned memory，UVA 读）"
                         "为研究假设配方（场景 engram_table_placement 显式输入）",
        thresholds="TP 合法性：(tp−1)·ceil(24/tp)<24（合法 {1,2,3,4,5,6,8}）；"
                   "wkv 列切要求 25600%tp==0",
        input_domain="V4.1 主干（2 个 engram 层：1/14；24 哈希列/层）",
        both_sides_covered=False,
        uncovered_domain=["DP 分片交换（all-to-all/EngramPeerExchange/symmetric "
                          "memory/DPSharedEngramStorage）未建模——本仓默认单副本"
                          "路径不启用（EngramBatch L1539-1661 仅 DP>1+无 TP+"
                          "dim%128==0 时启用）",
                          "EngramBatch MXFP8 打包路径（kMxfp8Dynamic）未建模",
                          "cpu_offload 运行时行为（预取流/TLB/36864·4096 排序"
                          "阈值，GB200 实测值不进模型）未建模",
                          "amd/engram.py 变体（ROCm 路径）未 pin 未建模",
                          "kernel 内部 op 数（哈希/门控）为解析近似计数，非指令数"],
        note="hc 三键 mapped（2026-11-10）；Engram 8 键 mapped（2026-11-11 R04，"
             "common/engram.py pin blob sha1 ba4ceea6 对 vllm-tree.json 校验）；"
             "count/work.engram_rows + lowering/memory engram 对象 + "
             "sharding engram gathers；CSA2/MoE/量化缺口仍阻断 "
             "structure_complete=False，不冒充"),
    SourceMapEntry(
        entry_id="sm-v41-mhc-kernels",
        model_or_method="deepseek_v41_flash mHC kernels",
        semantic="一次 mHC 边界 kernel 分支表与操作计数（R05，2026-11-12）："
                 "B5 默认路径 mhc_fused_post_pre_delayed_tilelang（T≤32 且 "
                 "h%128==0 时 _MHC_FUSED_TILELANG_KERNEL 融合核把 post 折入 "
                 "prenorm GEMM；T>32 时 _MHC_POST_TILELANG_KERNEL + prenorm "
                 "GEMM，DeepGEMM TF32 split-K 或 tilelang FP32，FLOPs 相同；"
                 "model 路径 norm_weight 恒非 None → MHC_PRE_NORM_KERNEL "
                 "epilogue）；系数 epilogue = 门+affine+softmax+sinkhorn(R−1 "
                 "次行/列交替)；首层 attn 侧 IC=h 读 hc_attn_fn_broadcast + "
                 "hc_expand；Engram 层（1/14）post/pre 分立不改变总量；"
                 "模型出口 = 末层 FFN post + hc_collapse_triton 终折叠 + "
                 "final RMSNorm（另计）。计数约定：初等运算各 1 op、超越函数 "
                 "1 op、归约 n−1（公式 v2.mhc_* 六条）",
        source_pin=VLLM_PIN,
        source_location="vllm/model_executor/kernels/mhc/tilelang.py:256-486"
                        "（mhc_fused_post_pre_delayed_tilelang 分支）、"
                        "778-798（mhc_post_tilelang）、685-775"
                        "（mhc_pre_broadcast_tilelang）；kernels/mhc/"
                        "tilelang_kernels.py:63-115（fused config：T≤32 且 "
                        "h%128==0；T<16→(2,8,128)、T∈[16,32]→(6,8,128)；"
                        "single-pass 需 SM100 且 h==4096）、319-527"
                        "（mhc_pre_big_fuse_with_norm_tilelang：sinkhorn "
                        "L383-441、迭代 (R−1) L433-440、NaN 守卫 L432、"
                        "collapse L478-502、两遍 RMSNorm L457-524）、"
                        "693-811（mhc_fused_tilelang：post 折入+核内 sqrsum "
                        "L772-775）、813-863（mhc_post_tilelang L856-861）、"
                        "867-948（hc_prenorm_gemm_tilelang：sqrsum L911-912）；"
                        "kernels/mhc/triton.py:75-147（hc_collapse_triton，"
                        "enable_fp_fusion=False）；models/deepseek_v41/nvidia/"
                        "ops/mhc.py:45-57（supports_mhc_overlap）、60-73"
                        "（supports_mhc_all_reduce）、192-323"
                        "（mhc_shifted_post_pre dispatch）、326-355"
                        "（hc_expand）；ops/mega_mhc.py:16-44（can_use_"
                        "mega_mhc：SM100+DeepGEMM+h%1024==0+m==4）；"
                        "nvidia/model.py:524-655（forward 分支）、543-560"
                        "（首层 hc_expand+fn_broadcast）、583-614（engram 层"
                        "分立）、1259-1279（_collapse 出口 post）、1187-1192"
                        "（终折叠+final norm）、1530-1544"
                        "（finalize_mhc_broadcast_weights 派生张量）、"
                        "388-391（hc_eps/hc_post_alpha=2.0）",
        branch_predicate="B1 融合 all-reduce：TP4 且非 EP 且 h==5120 且 m==4 "
                         "且 MNNVL multicast（ops/mhc.py L60-73）→ 当前设备"
                         "注册表（H100/H200/L40）无 MNNVL，TP8 场景不适用；"
                         "B2/B3 overlap 流：SM100 家族 + DeepGEMM + mix≤32 + "
                         "FULL graph + T≤16（supports_mhc_overlap L45-57 + "
                         "model.py L524-531）→ 不适用；B4 mega：SM100+"
                         "DeepGEMM+h%1024==0+m==4（mega_mhc.py L16-28）→ "
                         "不适用；B5 默认融合/分立路径（本仓主路径）；"
                         "B6 首层、B7 engram 层分立、B8 模型出口（均计）；"
                         "B9 write_kv/decoder_replay/specialized_decode、SP "
                         "gather、amd aiter → 域外未建模",
        thresholds="融合核 T≤32 且 h%128==0（tilelang_kernels.py L89-115）；"
                   "single-pass 变体 T≤4 且 h==4096 且 SM100 → V4.1 h=5120 "
                   "不适用；sinkhorn 循环 R−1 次（R=hc_sinkhorn_iters=20）",
        input_domain="V4.1（h=5120, m=4, R=20, eps=1e-6, 40 层, engram 层 "
                     "1/14）decode/prefill 的 mHC 边界与出口",
        both_sides_covered=False,
        uncovered_domain=["源码阅读（source-checked）非执行；未运行任何 kernel",
                          "SM100 分支（mega_mhc/overlap 流/single-pass）/"
                          "MNNVL TP4 融合 all-reduce/write_kv+decoder_replay/"
                          "specialized_decode/SP gather/amd aiter（kernels/"
                          "mhc/aiter.py + amd/mhc.py）域外未建模，只登记",
                          "split-K 跨片规约 (n_splits−1)·(mix+1)/token 依赖"
                          "设备 SM 数（compute_num_split），未入账",
                          "中间缓冲（mixes/sqrsum splits 维）HBM 往返与 T≤32 "
                          "融合路径寄存器喂入的流量节省未细分：act 流量字段"
                          "保留解析近似",
                          "计数为文档约定（超越函数 1 op、归约 n−1、NaN 守卫/"
                          "split-K 不计），非 kernel 指令数"],
        note="R05（2026-11-12）：mHC 从『仅 model.py source-checked + 解析"
             "近似』升级为 kernel 源码级 source-checked 精确整数计数；"
             "12+3 pin 文件 blob sha1 对 vllm-tree.json 校验"
             "（tests/test_v41_mhc_kernels.py）；count hc_* 行 sinkhorn/"
             "post_pre 改精确、新增 hc_exit_* 两行与 -w-mhc-broadcast-L0 "
             "派生张量驻留；sm-v41-engram-hc 保留（Engram 侧证据不变）",
    ),
    SourceMapEntry(
        entry_id="sm-v41-csa2-moe-unverified",
        model_or_method="deepseek_v41_flash CSA2/MoE",
        semantic="CSA2 注意力/投影为 v0.1 移植；MoE sqrtsoftplus/noaux_tc/"
                 "分组路由未建模；候选块发布/量化未映射",
        source_pin=VLLM_PIN,
        source_location="vllm/models/deepseek_v41/nvidia/model.py:131-200 "
                        "(DeepseekV4MoE)、204-327（sparse-MLA 选择）、"
                        "787-800（candidate block buffer）",
        branch_predicate="config scoring_func=sqrtsoftplus、topk_method=noaux_tc、"
                         "candidate_topk_blocks=2048",
        input_domain="V4.1-Flash 全 40 层",
        both_sides_covered=False,
        uncovered_domain=["CSA2 注意力/投影未独立逐行核（仅 v0.1 同源对账）",
                          "MoE scoring/top-k/分组路由未消费",
                          "candidate_publish 容量/时序未建模",
                          "quantization_config 未映射到计算精度"],
        note="2026-11-10 审查：这些路径不能因 config 通过而标完整或参与部署排名"),
    SourceMapEntry(
        entry_id="sm-dflash-round-protocol",
        model_or_method="dflash v1",
        semantic="每轮：1 次草稿前向（并行块提案）→ 1 次目标块验证 → 逐位拒绝 → "
                 "KV 回退 crop(start) → target_hidden 取 produced 前缀",
        source_pin=DFLASH_PIN,
        source_location="dflash/model.py dflash_generate L233-308 "
                        "(verify_size=min(block_size,…); _crop_to; [:produced])",
        branch_predicate="verify_size = min(block_size, max_length − start)；"
                         "temperature>0 → 拒绝采样，否则 argmax 前缀匹配",
        thresholds="block_size 16（config）/ 服务端可用 8 覆盖",
        input_domain="贪心与随机采样两验收分支",
        both_sides_covered=True,
        uncovered_domain=["SGLang/vLLM 服务端集成未逐行核对（pending asset）"],
        note="methods/dflash.py 依此生成 MethodPlan 与消息量"),
    SourceMapEntry(
        entry_id="sm-dflash-cross-attention",
        model_or_method="dflash v1",
        semantic="草稿注意力 K/V 拼接 target_hidden 投影（ctx）与 noise 投影（块内）",
        source_pin=DFLASH_PIN,
        source_location="dflash/model.py Qwen3DFlashAttention.forward L369-420",
        branch_predicate="k = cat([k_proj(target_hidden), k_proj(hidden)])",
        input_domain="全部草稿层（5 sliding + 1 full）",
        both_sides_covered=True,
        note="计数按 ctx+q_len 两段 KV；fc 投影 8·h→h 入账"),
    SourceMapEntry(
        entry_id="sm-vllm-moe-tp-ep-recipe",
        model_or_method="qwen3_5_moe/qwen3_moe MoE 并行",
        semantic="EP 关：routed 专家 inter/tp 切分（每 rank 全 E）；"
                 "EP 开（ep=tp）：每 rank E//ep(+余数) 整专家；router "
                 "GateLinear=ReplicatedLinear；shared expert 全局 TP 切分",
        source_pin=VLLM_PARALLEL_PIN,
        source_location="vllm/model_executor/layers/fused_moe/config.py:1402 "
                        "(intermediate_size_per_partition = inter // tp)、"
                        "config.py:1257-1288 (FusedMoEParallelConfig.make："
                        "use_ep → tp→1/ep→world)、router/gate_linear.py:66 "
                        "(GateLinear(ReplicatedLinear))、expert_map_manager.py:"
                        "62-77（余数 base+1）、qwen3_next.py:139-231 "
                        "(Qwen3NextSparseMoeBlock)",
        branch_predicate="enable_expert_parallel 且 world>1 → EP；否则 TP 切 inter",
        thresholds="ep ∈ {1} 或 ep == tp；inter % tp == 0（EP 关）；"
                   "shared_inter % tp == 0；tp <= E",
        input_domain="Qwen3.6-35B-A3B（256E inter512 shared512）TP{1,2,4,8,16}",
        both_sides_covered=True,
        uncovered_domain=["量化 Marlin ba_proj disable_tp 分支未入配方（BF16 路径）",
                          "EPLB 冗余专家（n_redundant_experts）未建模"],
        note="R01 统一 per-rank 配方：count/moe.py、lowering/memory.py、"
             "lowering/sharding.py 共用 ParallelPlan（G-C11 解除）"),
    SourceMapEntry(
        entry_id="sm-vllm-attn-gdn-shard",
        model_or_method="qwen3_5_moe attention/GDN 分片",
        semantic="full attn：heads/tp；kvh>=tp 切 kvh/tp、kvh<tp 复制全量；"
                 "o_proj 行切。GDN：in_proj_qkvz((2k+2v)/tp)、in_proj_ba"
                 "(2nv/tp)、conv1d(conv_dim/tp)、out_proj 行切；状态 "
                 "temporal=nv/tp×hk×hv、conv=conv_dim/tp×(kernel−1)",
        source_pin=VLLM_PARALLEL_PIN,
        source_location="vllm/model_executor/models/qwen3_next.py:302-315 "
                        "(Qwen3NextAttention 头切分/复制断言)、"
                        "layers/mamba/gdn/qwen_gdn_linear_attn.py:753-838 "
                        "(MergedColumnParallelLinear/RowParallelLinear)、"
                        "layers/mamba/mamba_utils.py:280-301 "
                        "(gated_delta_net_state_shape)",
        branch_predicate="kvh >= tp → assert kvh % tp == 0 切分；"
                         "else assert tp % kvh == 0 复制",
        input_domain="Qwen3.6（heads16/kvh2/GDN 16k32v）TP{1,2,4,8}",
        both_sides_covered=True,
        note="KV 复制=每 rank 同字节（不随 tp 缩减）；计数/容量同口径"),
    SourceMapEntry(
        entry_id="sm-vllm-dflash-colocated-sharing",
        model_or_method="dflash v1 同卡共享",
        semantic="草稿与 target 同 TP 组运行（DFlashQwen3Attention 用全局 "
                 "TP）；embed_tokens/lm_head 直接替换为 target 同一模块对象"
                 "（词表行切，零额外驻留）；fc=ReplicatedLinear 全宽复制",
        source_pin=VLLM_PARALLEL_PIN,
        source_location="vllm/v1/worker/gpu/spec_decode/dflash/utils.py:"
                        "load_dflash_model L16-70（maybe_share_target_embed +"
                        " dflash_model.lm_head = target_lm_head）、"
                        "eagle/utils.py:_should_share/maybe_share_target_embed、"
                        "models/qwen3_dflash.py:DFlashQwen3Attention L191-204",
        branch_predicate="draft 无 has_own_lm_head/has_own_embed_tokens 且与 "
                         "target 权重相等 → 共享对象；否则草稿自持副本",
        input_domain="z-lab/Qwen3.6-35B-A3B-DFlash（vocab=target 248320）",
        both_sides_covered=True,
        uncovered_domain=["TP1 草稿挂 TP>1 组的服务栈配方未见（拒绝不支持）",
                          "workspace/graph/NCCL 池尺寸未 pin（峰值缺口）"],
        note="R02a：colocated 账本草稿 embed/lm_head 零字节引用 + consumer "
             "登记；修复前幻影副本 ~1GB 与去重键失配双重计费"),
    SourceMapEntry(
        entry_id="sm-vllm-lmhead-logits-gather",
        model_or_method="lm_head 词表行切后的 logits 集合通信（TP>1）",
        semantic="ParallelLMHead 每 rank 只产出 T×(V/tp) logits；"
                 "LogitsProcessor._get_logits 在 lm_head.tp_size>1 时对每次 "
                 "lm_head 调用执行 gather（all-gather 或 gather-to-rank0，"
                 "由 current_platform.use_all_gather 决定）。每 rank 线上"
                 "最大字节均为 (tp−1)/tp × T×V×head_dtype 字节",
        source_pin=VLLM_PARALLEL_PIN,
        source_location="vllm/model_executor/layers/logits_processor.py:90-91"
                        "（use_all_gather 平台开关）、122-133（_gather_logits："
                        "tensor_model_parallel_all_gather / _gather）、185-198"
                        "（_get_logits：lm_head.tp_size>1 → _gather_logits）、"
                        "qwen3_5.py:345/428（ParallelLMHead + logits_processor）、"
                        "qwen3_dflash.py:920-935（草稿 compute_logits 走同一 "
                        "LogitsProcessor）",
        branch_predicate="lm_head.tp_size > 1 且非 skip_gather → gather",
        thresholds="tp>1 恒触发；decode/verify/草稿候选各一次",
        input_domain="Qwen3.6 V=248320 TP{2,4,8,16}；B×T 按阶段（decode B、"
                     "verify B×blk、草稿 B×(blk−1)）",
        both_sides_covered=True,
        uncovered_domain=["gather 的平台分支差异（all-gather vs rank0-gather）"
                          "按同一字节上界建模；NCCL 内部缓冲另计缺口"],
        note="R01 通信配方补漏（2026-11-10 二轮审计）：此前 draft.py 注释声称"
             " 'logits 聚合通信另计' 但 model_collectives 未实现；本轮按 pin "
             "补 logits_gather 计费（lowering/sharding.py）"),
    SourceMapEntry(
        entry_id="sm-vllm-dflash-ctx-kv-second-read",
        model_or_method="DFlash 草稿 ctx KV 投影的第二次权重读与融合驻留缓冲",
        semantic="草稿层 K/V 投影权重每轮被读两次：noise 块 token 走 qkv_proj，"
                 "target_hidden 上下文行单独再投影。vLLM 集成三分支："
                 "未量化 GPU 把全层 K/V 行 torch.cat 成 _fused_kv_weight"
                 "（新驻留副本，非视图）后一次融合 GEMM；MXFP8 走独立 "
                 "ReplicatedLinear _context_kv_proj（仅 K/V 行，量化打包）；"
                 "其余量化/CPU repack 逐层执行**完整** qkv_proj 后切 Q。"
                 "k_norm 权重堆叠在三条路径都存在",
        source_pin=VLLM_PARALLEL_PIN,
        source_location="dflash/model.py:384-391（k_ctx/k_noise 两次独立 "
                        "k_proj/v_proj 调用后 cat）、models/qwen3_dflash.py:"
                        "108-113（_reads_raw_weights）、529-531（MXFP8 条件 "
                        "_context_kv_proj）、541-568（torch.cat → "
                        "_fused_kv_weight；k_norm 恒 stack）、617-698"
                        "（_project_context_kv_token_major 三分支）",
        branch_predicate="显式 draft_ctx_kv_path：fused_raw ← GPU 且全层 "
                         "UnquantizedLinearMethod；mxfp8_context_proj ← 全层 "
                         "qkv 为 ModelOptMxFp8Config 且非未量化；"
                         "per_layer_fallback ← 其余（量化打包/CPU repack）",
        thresholds="不由 dense_w 推断：dense_w 是计价位宽，不决定 quant_method；"
                   "路径缺失/未知显式拒绝；fused_raw 与 dense_w<8 在场景校验拒绝",
        input_domain="Qwen3.6 DFlash 草稿（6 层，kvh=8）TP{1,2,4,8}；"
                     "默认场景草稿 config 无 quantization_config → fused_raw",
        both_sides_covered=True,
        uncovered_domain=["MXFP8 _context_kv_proj 的 block-scale 存储布局"
                          "（modelopt.py 未入 pin）：权重/驻留只计 packed "
                          "8bit 下界，scale 缺口显式传播",
                          "量化 qkv 的 scale/zero 元数据与 CPU repack 细节未建模"],
        note="R01/R02a 二轮补漏（ctx_kv_proj 二次读/fused 副本）；"
             "三轮修正：回退按完整 qkv（含 Q）计 FLOPs/权重读/临时缓冲，"
             "k_norm 堆叠三路径均计，路径改显式输入"),
    SourceMapEntry(
        entry_id="sm-vllm-dflash-norm-dtype",
        model_or_method="DFlash 草稿 RMSNorm 参数 dtype 与 k_norm 堆叠字节",
        semantic="`_k_norm_weights = torch.stack([a.k_norm.weight.data ...])`"
                 " 继承草稿模型 dtype；RMSNorm 未显式传 dtype 时以"
                 " torch.get_default_dtype() 建 weight，而 vLLM "
                 "BaseModelLoader.load_model 在 "
                 "set_default_torch_dtype(model_config.dtype) 内初始化草稿。"
                 "因此 k_norm stack 是 6×128×2=1,536 B（bfloat16），"
                 "不是按 draft_quant.dense_w 的 768/384 B",
        source_pin=VLLM_PARALLEL_PIN,
        source_location="vllm/model_executor/layers/layernorm.py L42-58"
                        "（weight_dtype = dtype or torch.get_default_dtype()）、"
                        "vllm/model_executor/model_loader/base_loader.py"
                        " L68-85（set_default_torch_dtype(model_config.dtype)"
                        " 包住 create_model/load_weights）、"
                        "assets/identity/vllm_qwen35_pin/dflash_utils.py L51-54"
                        "（get_model(..., model_config=draft_model_config)）、"
                        "qwen3_dflash.py L288-289（RMSNorm 未传 dtype）、"
                        "L565-568（torch.stack 继承 weight dtype）、"
                        "assets/config/qwen36_35b_a3b_dflash__config.json"
                        "（\"dtype\": \"bfloat16\"）",
        branch_predicate="RMSNorm(dtype=None) → torch.get_default_dtype()；"
                         "草稿 get_model 的参数初始化由 draft_model_config.dtype "
                         "决定，与 Linear quant_method/dense_w 无关",
        thresholds="bfloat16/float16→2B/elem；float32→4B；未知 dtype 显式拒绝，"
                   "不得回退到 dense_w",
        input_domain="活跃 DFlash 草稿 config dtype=bfloat16；"
                     "draft_quant.dense_w=4/8/16 均不改变 k_norm 字节",
        both_sides_covered=True,
        uncovered_domain=["layernorm.py/base_loader.py 未入本地 pin（固定 commit "
                          "与行号登记，未执行）；Python 默认 dtype 的运行期覆盖"
                          "（如调用方另设）未建模"],
        note="复核轮补登记（2026-11-10）：修正 k_norm 计数为 6×128×2=1,536 B；"
             "DraftIR.dtype_hint 由 config dtype/torch_dtype 解析"),
]


def source_map_json() -> dict:
    return {"schema_version": "0.2",
            "pins": {"vllm": VLLM_PIN, "vllm_parallel": VLLM_PARALLEL_PIN,
                     "hf_transformers": HF_TF_PIN, "dflash": DFLASH_PIN},
            "entries": [e.to_json() for e in ENTRIES]}
