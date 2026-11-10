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
        source_location="modeling_qwen3_5_moe.py:747-772 (Qwen3_5MoeAttention)",
        branch_predicate="q_proj 输出 view(..., head_dim*2) 后 chunk(2)",
        input_domain="全部 full attention 层（attn_output_gate 家族默认）",
        both_sides_covered=True,
        note="ModelIR AttentionIR.output_gated=True → q_out=2·heads·d（已进计数）"),
    SourceMapEntry(
        entry_id="sm-qwen35-moe-shared-gate",
        model_or_method="qwen3_5_moe MoE",
        semantic="共享专家输出经 sigmoid(shared_expert_gate(x)) 门控后与 routed 相加",
        source_pin=HF_TF_PIN,
        source_location="modeling_qwen3_5_moe.py:884-901 (Qwen3_5MoeSparseMoeBlock)",
        branch_predicate="shared_expert_intermediate_size 存在 → shared+gate 常驻",
        input_domain="Qwen3.6-35B-A3B（shared 512 + gate Linear(h,1)）",
        both_sides_covered=True,
        note="shared_gate 行已入 moe_rows"),
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
                 "Engram 查表注入（层 1/14 residual 流）参与 forward",
        source_pin=VLLM_PIN,
        source_location="vllm/models/deepseek_v41/nvidia/model.py:388-499（hc 参数 "
                        "mix_hc=(2+m)·m、fp32）、501-678（forward 边界分支：首层 "
                        "hc_expand/engram 层 post+pre 分立/常规 shifted 融合）、"
                        "103-119（imports；kernel 实现在 vllm.model_executor.kernels."
                        "mhc，未入评审包）",
        branch_predicate="engram_layer_ids 覆盖层启用 Engram；hc_mult/hc_eps/"
                         "hc_sinkhorn_iters 控制 mhc 边界（全层执行，非可选开关）",
        input_domain="V4.1 主干",
        both_sides_covered=False,
        uncovered_domain=["Engram.forward 内部算子未 pin（vllm/models/deepseek_v41/"
                          "common/engram.py 不在评审包）→ 注入算子不计猜测常数",
                          "Engram 表 dtype 未知 → 容量字节未知（默认 cpu_offload）",
                          "mhc tilelang kernel 每迭代 OP 数为解析近似（行标 "
                          "approximated）"],
        note="hc 三键 mapped（2026-11-10）；Engram 8 键 unsupported → "
             "structure_complete=False 仍阻断完整结论，不冒充"),
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
]


def source_map_json() -> dict:
    return {"schema_version": "0.2",
            "pins": {"vllm": VLLM_PIN, "vllm_parallel": VLLM_PARALLEL_PIN,
                     "hf_transformers": HF_TF_PIN, "dflash": DFLASH_PIN},
            "entries": [e.to_json() for e in ENTRIES]}
