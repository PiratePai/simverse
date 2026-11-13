"""证据登记（M0）：资产 → EvidenceRecord，内容 hash 绑定。

所有记录在运行时从磁盘实算 SHA256（不信文件名）；源码阅读与执行分栏
（executed=False → source-checked）。证据改动后 content_sha256 变化，
依赖它的结论随之失效（不自动继承）。
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from ..contracts import EvidenceRecord

ROOT = Path(__file__).resolve().parent.parent.parent
ASSETS = ROOT / "assets"
REFS = ROOT / "docs" / "rebuild-20261106" / "references"

# 固定身份（assets/identity/*.json 由抓取脚本落盘；此处登记 claim）
_IDENTITY_FILES = {
    "ev-identity-required-scenarios": "identity/required_scenarios_identity.json",
    "ev-identity-dflash-search": "identity/hf_dflash_search.json",
}
_PINNED_ASSETS = [
    # (evidence_id, kind, claim, path, source, domain, limits, executed)
    ("ev-v41-config", "config",
     "DeepSeek-V4.1-Flash 官方 config.json 固定快照的字段身份",
     "config/deepseek_v41_config.json",
     "HF deepseek-ai/DeepSeek-V4.1-Flash @314113f8a7dadb388215396ba5fbee2c610a04d6",
     "V4.1 结构字段（text_config/压缩层计划）",
     ["config 不能证明 kernel/通信/位宽布局行为；Engram/hc 语义另见 ev-vllm-v41"], False),
    ("ev-qwen36-main-config", "config",
     "Qwen3.6-35B-A3B 主模型官方 config.json（qwen3_5_moe，混合层序+MoE+MTP1）",
     "config/qwen36_35b_a3b_main__config.json",
     "HF Qwen/Qwen3.6-35B-A3B @345d467b117d1fd5eda9b879a5988de9ae5f47a7 (main)",
     "Qwen3.6-35B-A3B 文本结构；vision_config 登记 metadata-only",
     ["config 证明结构字段，不证明 forward 数值语义或 kernel 行为"], False),
    ("ev-qwen36-dflash-config", "config",
     "z-lab/Qwen3.6-35B-A3B-DFlash 草稿 checkpoint config（6 层 GQA 草稿+块扩散参数）",
     "config/qwen36_35b_a3b_dflash__config.json",
     "HF z-lab/Qwen3.6-35B-A3B-DFlash @f181eece646affea2c38b2765f1aaa01a9734ccd",
     "DFlash 草稿结构：block_size 16 / mask_token / target_layer_ids 8 层",
     ["auto_map dflash.DFlashDraftModel 不在 checkpoint 内，实现见 ev-dflash-impl"], False),
    ("ev-hf-qwen35-modeling", "source-checked",
     "Qwen3.5/3.6 qwen3_5_moe HF forward：GDN decode 分支、full-attn output "
     "gate、shared expert gate、router/top-k",
     "identity/hf_transformers/modeling_qwen3_5_moe.py",
     "huggingface/transformers @b6401d6605d16620e0f95b54fb2906f92de7e33f "
     "modeling_qwen3_5_moe.py",
     "Qwen3.6-35B-A3B 文本 forward 的关键分支（源码阅读，未执行）",
     ["源码阅读非执行；融合 kernel 内部未核；MTP 被 HF 丢弃，另列 unverified"],
     False),
    ("ev-hf-qwen35-config", "source-checked",
     "Qwen3.5/3.6 qwen3_5_moe HF configuration 字段语义",
     "identity/hf_transformers/configuration_qwen3_5_moe.py",
     "huggingface/transformers @b6401d6605d16620e0f95b54fb2906f92de7e33f "
     "configuration_qwen3_5_moe.py",
     "config 字段到 HF module 的构造路径（shared expert/gate 等）",
     ["配置构造 ≠ forward 执行；不证明数值"], False),
    ("ev-qwen3-30b-config", "config",
     "Qwen3-30B-A3B 历史归档 config 快照（已移出活跃 catalog）",
     "config/qwen3_30b_a3b__config.json",
     "HF Qwen/Qwen3-30B-A3B @ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
     "历史校准/回归资产；禁止作为产品模型或跨模型 profile 依据",
     ["历史归档；无 HF forward pin；不得进入活跃目录"], False),
    ("ev-qwen3-235b-config", "config",
     "Qwen3-235B-A22B 历史归档 config 快照（已移出活跃 catalog）",
     "config/qwen3_235b_a22b__config.json",
     "HF Qwen/Qwen3-235B-A22B @8efa61729e24bd65b1d152b5ab5409052aa80e65",
     "历史校准/回归资产；禁止作为产品模型或跨模型 profile 依据",
     ["历史归档；无 HF forward pin；不得进入活跃目录"], False),
    ("ev-qwen3-next-config", "config",
     "Qwen3-Next-80B-A3B 历史归档本地快照（官方身份 PENDING）",
     "config/qwen3_next_80b_a3b__config.json",
     "model-verse 本地快照；官方 HF resolve gated",
     "历史回归资产；revision 未核实",
     ["身份不完整；历史归档；不得进入活跃目录或部署排名"], False),
    ("ev-deepseek-v3-config", "config",
     "DeepSeek-V3 历史归档 config 快照（已移出活跃 catalog）",
     "config/deepseek_v3__config.json",
     "HF deepseek-ai/DeepSeek-V3 @e815299b0bcbac849fa540c768ef21845365c9eb",
     "历史回归资产；dense FFN 宽度/revision 冲突根因已记录",
     ["历史归档；无 forward pin；不得进入活跃目录"], False),
    ("ev-dflash-impl", "source-checked",
     "DFlash 方法实现（dflash_generate 协议/DFlashDraftModel 结构/拒绝采样）",
     "identity/dflash_repo/model.py",
     "github z-lab/dflash @07ebd93db9f472af339b644bb70221ad8428328a dflash/model.py",
     "DFlash v1 单次并行提案、逐位拒绝、KV 回退语义",
     ["源码阅读未执行；服务端集成路径 2026-11-12 R07a 已逐行核对"
      "（ev-vllm-dflash-serving / ev-sglang-dflash-serving）；"
      "HF 参考与两服务栈的 ctx 投影跨度差异见 R07a 报告"], False),
    ("ev-dflash-readme", "source-checked",
     "DFlash 发布说明：SGLang DFLASH 部署配方、block 8/16 建议、B200 基准身份",
     "identity/dflash__README.md",
     "HF z-lab/Qwen3.6-35B-A3B-DFlash README @f181eece",
     "部署参数来源（--speculative-dflash-block-size 等）",
     ["基准数字（3.61x 等）是厂商口径，未复核，不入生产结果"], False),
    ("ev-vllm-v41-model", "source-checked",
     "vLLM V4.1 专用实现：Engram 注入、hc 维度参与 forward 的源码证据",
     str(REFS / "vllm" / "vllm" / "models" / "deepseek_v41" / "nvidia" / "model.py"),
     "vllm @f358f0578a83f4e1fe6fc49a492d3ae9362de8e5（评审包 pin）",
     "V4.1 Engram/hc 结构存在性与 forward 参与",
     ["AST/阅读证据，非执行；完整 lowering/容量仍缺（capability 登记）"], False),
    ("ev-vllm-v41-engram", "source-checked",
     "vLLM V4.1 Engram 完整建模证据（R04，2026-11-11）：EngramLayout 素数桶"
     "分配（vocab_size−1 起连续抽取、seen 全局不复用；层 1/14 的 24 桶长之和"
     "与 config engram_num_embeddings 精确相等）、表值 FP8 e4m3 + ue8m0 "
     "block-scale（32）= 264 B/行、TP 按 hash head ceil 切分（断言 "
     "(tp−1)·part<24，每 rank 行数=桶长和而非 num_embeddings/tp）、批级 "
     "n-gram 哈希一次 kernel 算全部 engram 层（lookback 3，挂首本地层 SWA）、"
     "wkv ColumnParallelLinear 6144→25600 gather_output=True、q/k_weight "
     "[hc,5120] bf16 复制、post_wkv 门控（rsqrt 归一点积→sign-sqrt→sigmoid）"
     "注入 residual 流",
     str(REFS / "vllm" / "vllm" / "models" / "deepseek_v41" / "common" / "engram.py"),
     "vllm @f358f0578a83f4e1fe6fc49a492d3ae9362de8e5（engram.py blob sha1 "
     "ba4ceea624ef6513ed7e787d084883f42d9d8322 + nvidia ops "
     "engram_host_table/engram_peer/specialized_decode engram prefetch 三文件；"
     "blob sha1 已对 docs/rebuild-20261106/references/vllm-tree.json 逐项校验）",
     "R04 Engram 结构映射（model_ir extras.engram）、count/work.engram_rows "
     "四行、lowering/memory engram 表/wkv/qk/staged 对象与 engram gathers",
     ["源码阅读（source-checked）非执行；CPU/GPU 均未运行",
      "DP 分片交换（all-to-all/peer/symmetric memory）与 EngramBatch MXFP8 "
      "路径未建模（本仓默认单副本路径不启用）",
      "cpu_offload 运行时行为（预取流/TLB/排序阈值）未建模；表位置为显式"
      "研究假设配方（场景 engram_table_placement，默认 cpu_offload）",
      "kernel 内部 op 数（哈希/门控常数 10d+5 等）为解析近似计数"],
     False, "2026-11-11"),
    ("ev-vllm-v41-mhc", "source-checked",
     "R05 pinned TileLang mHC：GEMM/sqrsum/prenorm、gates/Sinkhorn m×m、"
     "post comb 流更新、首层无 post/broadcast 派生缓冲、出口 post/collapse",
     str(REFS / "vllm" / "mhc-pin.json"),
     "https://github.com/vllm-project/vllm/tree/"
     "f358f0578a83f4e1fe6fc49a492d3ae9362de8e5；14 文件 blob SHA1 + SHA256",
     "V4.1 mHC 有限输入；unfused/split=1 逻辑访问与算术 op 参考配方",
     ["源码阅读 source-checked；所有 kernel/CPU forward 未执行，非 GPU measured",
      "transcendental=1 op，RMSNorm 4h+2 惯例近似；非指令数/物理 HBM",
      "split-k 运行时值及 overlap/mega/fused-AR/ROCm/CPU/XPU 部署变体未建模；"
      "capture_aux 仅 speculative，流均值工作未建模"], False, "2026-11-12"),
    ("ev-vllm-v41-csa2", "source-checked",
     "R20 CSA2 索引打分/top-k/candidate block：21 文件 pin（attention.py/"
     "sparse_mla.py/compressor.py/common ops/specialized_decode 7 文件/"
     "sparse_attn_indexer/candidate_blocks/indexer_topk/backends indexer/"
     "deepseek_v4 attention/config attention + test_decode_topk）。覆盖：层型"
     "0/1/2 与 kv/index 源拓扑、compressor bf16 权重+FP32 kv_score+ratio2 "
     "softmax 门控池化+环形 state(cap 8)、rope_quant_insert 打包记录 584/528/"
     "288、indexer K 132B/行、短上下文旁路(max_seq_len//r≤topk，全选 fill 纯 "
     "store)、decode paged MQA logits 全候选打分+weights 折算、topk 选择"
     "（cooperative≤64/persistent/per_row）、candidate 源发布 top-2048×8 位块"
     "（NaN 传播 max、最新块 +inf 钉扎、-1 填充）与消费层 mask 到候选、"
     "SparseIndexTables 等键运行首层 combine_topk_swa_indices 复用、共享"
     " topk/candidate 缓冲（模型级单例）",
     str(REFS / "vllm" / "csa2-pin.json"),
     "vllm @f358f0578a83f4e1fe6fc49a492d3ae9362de8e5；21 文件 blob SHA1 对 "
     "vllm-tree.json 逐项校验 + SHA256 清单",
     "V4.1 CSA2 索引管线（count/work.v41_layer_rows、lowering/memory 记录/"
     "环形/缓冲、scenario.v41_kv_record、公式 v2.csa2_*）",
     ["源码阅读 source-checked；kernel/CPU forward/GPU 均未执行",
      "SM100 sparse-logits（消费层只打分候选块）、decode_topk CuTe、TP 行"
      "分片 all-gather(≥12 行)、PCP/DCP、fused_mqa_topk、FlashInfer "
      "plain-row bf16、MXFP4 索引缓存(68B) 变体未建模",
      "CUDA-graph 静态整宽 logits 缓冲/整宽 -inf 尾写、workspace 池未入账"
      "（峰值缺口传播不变）；cooperative 候选在值相等边界为近似选择",
      "logits/选择 op 为文档化解析口径（非 kernel 指令数）；prefill gather "
      "工作区字节为上界；环形 state 读为每请求 1 行上界",
      "稀疏 MLA 注意力核本体（4d+5）仍 v0.1 移植，不在本证据范围"],
     False, "2026-11-12"),
    ("ev-vllm-v41-moe", "source-checked",
     "R21 V4.1 MoE scoring/top-k/分组路由：14 文件 pin（deepseek_v4 基类 "
     "nvidia/model.py + eplb_util、fused_moe router 8 文件（dsv4_topk/"
     "fused_topk_bias/router_factory/base/fused_moe/grouped_topk/cpu/"
     "gate_linear）+ moe_output + linear.py/base_loader.py/fused_moe/"
     "layer.py）。覆盖：GateLinear(h→E, out_dtype=fp32) Replicated 路由 "
     "GEMM（model.py L270-277）。**权重沿 model loader BF16**（未传 "
     "quant_config/params_dtype；linear.py L287-295/L179-199；"
     "base_loader.py L86；gate_linear.py L306-341；每层 weight 3,932,160 B）；"
     "**输入由 pinned mHC layer_input_cur BF16 证明**（tilelang.py "
     "L900/L1096，R05 pin；model.py L731-816 直接 self.ffn(x)）——不是"
     "由任意 model dtype 泛化；有效域仅 official V4.1 config "
     "dtype=bfloat16，full_config 非 BF16 拒绝；sqrtsoftplus 打分 "
     "sqrt(softplus(logits))（>20 取 sqrt(logits) 护栏，dsv4_topk.py L85）；"
     "noaux_tc e_score_correction_bias[E] fp32 只导选择不进权重（选择列="
     "score+bias L85-87、权重列=无偏 score L96-98；cpu_router.py L90-95 "
     "同语义 docstring）；k=6 轮平坦 argmax（无分组——V4.1 config 无 "
     "n_group/topk_group 键，工厂 L166-173 仅 use_grouped_topk 走 "
     "GroupedTopKRouter；pinned model 调用未传该参数，默认 False 已随 "
     "fused_moe/layer.py 入 pin）、"
     "平手取最小专家 id（L94-95）、NaN→−1e30（L87）；"
     "sum→1 归一后 ×routed_scaling_factor=1.5（sum≤0/NaN 时分母取 1，"
     "L104-107）；输出 FP32 weights + int32 ids（L149-150）；共享专家"
     "（n=1, inter 2304, 无 gate）在 routed 归一后以权重 1.0 **独立相加**"
     "（moe_output.py L151-175；FusedMoEFactory 调用未传 n_shared_experts/"
     "fuse_shared_experts，model.py L426-449；fused_topk_bias_router.py "
     "L447-464 的共享槽未启用）；V4.1 主干 num_hash_layers=0（v41 model.py "
     "L155，无 hash MoE）；视觉 checkpoint（vision_n_layers=32>0）"
     "bias_vl[E] fp32 每 MoE 层创建（L314-320）、HAS_VL 哨兵逐行判定"
     "（dsv4 L63-76）；EP 归属：mega 路径断言 n_physical(=E+eplb 冗余 0) "
     "% ep == 0（L357-362）；fused 路径 L419-421 的 start_idx 只赋值、"
     "未被所有权消费，fused 通用余数映射未入 pin/未建模——本仓按**保守"
     "整除域**收窄（require_divisible_ep），不主张 fused 路径非法",
     str(REFS / "vllm" / "moe-pin.json"),
     "vllm @f358f0578a83f4e1fe6fc49a492d3ae9362de8e5；14 文件 blob SHA1 对 "
     "vllm-tree.json 逐项校验 + SHA256 清单",
     "V4.1 MoE 路由管线（count/moe.v41_moe_route_rows 三行，gate 权重/"
     "输入 BF16 2B；model_ir extras.moe_route；lowering/memory router_b + "
     "路由偏置对象；count/work+memory 保守 EP 整除域校验；公式 "
     "v2.moe_route_*；config 域校验）",
     ["源码阅读 source-checked；kernel/CPU forward/GPU 均未执行",
      "ExpertMapManager 未入 moe-pin：fused EP 余数分配是未入 pin 的"
      "读取点；本 PR 只按 pinned model 调用 + mega 断言建模，不据 fused "
      "余数路径声明",
      "部署变体未建模：native MegaMoE（bf16_mega_gate 大批路径，需 EP+fp4 "
      "专家+SP token 分片）；GateLinear 的 bf16 专核/cuBLAS 档未执行"
      "（fp32/bf16x3 分层只适用 fp32 router 权重）；topk_hash_"
      "softplus_sqrt CUDA 专核与 dsv4_topk Triton 的实现差异（语义一致，"
      "指令构成不同）",
      "dsv4_topk Triton 档约束 E∈{256,384}、k=6、fp32、renormalize（不满足"
      "时落 ops.topk_hash_softplus_sqrt/CPU biased_topk_cpu——语义相同）；"
      "E/k 其他值未独立核",
      "DSpark 草稿层 MoE（dspark_n_routed_experts=128/k=3，128 专家 gate_"
      "threshold=1）为方法层字段，本 PR 不建模",
      "quantization_config（fp8/fp4 专家）仍缺：gate 权重已核实为未量化"
      " BF16 2B（不是场景 dense_w），专家/其余 dense 权重的量化布局未映射",
      "padding skip（VLLM_MOE_SKIP_PADDING）与 CUDA-graph 整宽路由缓冲、"
      "defer_moe_finalize（需 routed_scaling==1.0，V4.1-Flash=1.5 不适用）"
      "未入账；VL 图像 token 的 bias_vl 选择分支只登记未选路",
      "op 计数为解析算术口径（exp/log/sqrt/比较各 1，非 kernel 指令数）；"
      "input_ids 读按 int64 上界；bias_vl/哈希表布局字节为权重驻留口径"],
     False, "2026-11-12"),
    ("ev-vllm-v41-quant", "source-checked",
     "R22 V4.1 量化映射：20 文件 pin（dispatch 链 config/model.py+"
     "quantization/__init__+deepseek_v41/quant_config+transformers_utils/"
     "configs/deepseek_v41；MXFP8 线性 modelopt.py KMxfp8Static+"
     "KDynamicNoParam+fp8.py Fp8Config；MXFP4 专家 mxfp4.py create_weights/"
     "maybe_roundup+routed_experts.py+oracle/mxfp4.py 优先级与对齐；kernel "
     "域 kernels/linear/__init__+mxfp8 marlin/emulation/flashinfer+"
     "humming/priority+utils/humming；未量化类 fp8.py get_quant_method+"
     "vocab_parallel_embedding.py）。覆盖：quant_method=fp8+model_type "
     "deepseek_v41* → override deepseek_v4_fp8 → DeepseekV41FP8Config"
     "（CUDA/ROCm；quantization/__init__.py L120-132）；expert_dtype 由 "
     "config 类提升到 hf_config 顶层（configs/deepseek_v41.py L46-50），"
     "expert_dtype=='fp4' 派生 is_scale_e8m0（ue8m0 1B scale；fp8 专家则 "
     "float32 scale）；weight_block_size==[32,32]+e8m0 → 线性层 "
     "ModelOptLinearMethod(kMxfp8Static, kMxfp8Dynamic)：值 e4m3 1B + "
     "ue8m0 uint8 (1,32) scale，checkpoint 32×32 块装载展开为 (N,K/32)"
     "（modelopt.py L2424-2479）；激活 kMxfp8Dynamic 无落盘 scale"
     "（L2497-2510）；路由专家 Mxfp4MoEMethod(activation_key="
     "kMxfp8Dynamic)：w13 (E,2I,H/2) uint8 + w13_scale (E,2I,H/32)、w2 "
     "(E,H,I/2)+scale (E,H,I/32)（mxfp4.py L765-840）；scale_fmt 键不被 "
     "Fp8Config.from_config 读取（键表 L155-172；有效 dtype 由 "
     "expert_dtype 派生）；embed/lm_head/RMSNorm 未量化（get_quant_method "
     "对非 LinearBase/RoutedExperts/Attention 返回 None → 默认方法 "
     "BF16；vocab_parallel_embedding.py L322-326）；执行域（注册设备 "
     "≤SM90）：线性 Marlin MXFP8 W8A16（SM80+，激活 BF16）、专家 Marlin "
     "W4A16（oracle 优先级 L461-486）+ 分配对齐 I→128/H→256"
     "（routed_experts.py L190-203 + oracle L850-855）；KV 缓存不受 "
     "quantization_config 影响（无缓存叶键；Fp8KVCacheMethod 仅显式 fp8 "
     "kv_cache_dtype 时加载 checkpoint scale）",
     str(REFS / "vllm" / "quant-pin.json"),
     "vllm @f358f0578a83f4e1fe6fc49a492d3ae9362de8e5；20 文件 blob SHA1 对 "
     "vllm-tree.json 逐项校验 + SHA256 清单",
     "V4.1 量化布局（count/quant_layout 字节公式；count/work.v41_layer_rows"
     "/head_rows/hc 边界与 count/moe.moe_rows 的 recipe 计价；lowering/"
     "memory 权重 scale/打包/对齐对象与 embed-lm_head-norm BF16；"
     "pipeline 集合通信载荷 BF16；model_ir extras.quant + quantization_"
     "config 5 叶键 mapped；scenario V41_QUANT 冻结校验；公式 v2.quant_*）",
     ["源码阅读 source-checked；kernel/CPU forward/GPU 均未执行",
      "部署变体未建模：Humming（cc=90 优先于 Marlin，priority.py L10-23，"
      "但依赖可选外部 humming 包——utils/humming.py L18-97 lazy facade；"
      "其对齐档 I→64 落在 oracle round-up else 分支，与 Marlin 128 不同；"
      "W4A8 SM90 kernel 把 checkpoint MXFP8 激活按 block FP8(128) 运行，"
      "oracle/mxfp4.py L545-556）、SM100 TRTLLM_MXFP4_MXFP8/DeepGEMM/"
      "FlashInfer 线性 W8A8（mxfp8 动态激活 1B+K/32/行；无注册设备）、"
      "emulation 反量化驻留（VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD 默认开"
      "→BF16 2B，emulation.py L38-50）、NVFP4 moe_quant_algo 专家、"
      "expert_dtype=fp8（Flash-Base Fp8MoEMethod 块量化专家）",
      "Marlin 重排/补齐后布局与 workspace、marlin_utils_fp8/"
      "marlin_moe is_supported_config 细节未入 pin：驻留按 create_weights "
      "分配口径（含 maybe_roundup 对齐）计，重排等字节变换与 workspace "
      "缓冲未入账（峰值完整性 unknown 不变）",
      "scale_fmt 键不被 loader 消费：非 ue8m0 值按声明矛盾拒绝（不静默取"
      "expert_dtype 派生值）；moe_quant_algo/ignored_layers/modules_to_"
      "not_convert/store_dtype 等 from_config 键在官方资产不出现，未建模",
      "共享专家权重 pad 到 TP×32 整数倍在官方维度（2304%256=0）无补齐；"
      "若未来资产维度改变需复核 _pad_shared_expert_weight 分支",
      "集合通信/激活字节按模型隐状态 BF16 计（逻辑访问口径）；Marlin "
      "W8A16/W4A16 的逐 kernel 实际 HBM 流量未执行核验"],
     False, "2026-11-13"),
    ("ev-vllm-v41-swiglu-tail", "source-checked",
     "R24 V4.1 SwiGLU clamp + compress_ratios 尾部：13 文件 pin（语义核 "
     "activation.py SiluAndMulWithClamp L208-250——gate=clamp(max=limit) "
     "单侧、up=clamp(min=-limit,max=limit) 双侧、out=gate·sigmoid(alpha·"
     "gate)·(up+beta)，V4.1 alpha=1.0/beta=0.0；fused MoE 消费链 "
     "FusedMoEConfig.swiglu_limit（config.py L1365）→ ApplyMoEActivation"
     "Config.from_configs（fused_moe/activation.py L182-185 clamp_limit="
     "gemm1_clamp_limit or moe_config.swiglu_limit）→ 注册执行域 Marlin "
     "gemm1/gemm2 之间 apply_moe_activation（marlin_moe.py L140-158）；"
     "silu_and_mul_with_clamp L293-312/swiglu_limit_func utils.py L819-835 "
     "（HAS_LIMIT=swiglu_limit>0 L812）同语义；共享专家 MLP.act_fn="
     "SiluAndMulWithClamp（deepseek_v4/nvidia/model.py L176-183/L252/L331，"
     "CPU 变体同构 cpu/model.py L101-102/L132）；尾部归属：get_compress_"
     "ratios 'MTP layers included' + 派生路径补 [0]*num_nextn_predict_"
     "layers（deepseek_v4_utils.py L6-18）、DSpark 草稿层以层号 L..L+mtp-1 "
     "构造（dspark.py L93-96/L128-136）、attention.py L375-386 按层号取 "
     "compress_ratios[layer_id]（else 分支 0=纯 SWA）、L399-401 is_backbone "
     "门控使草稿层永不为 kv/index 源、后端只按去重层型集合消费（sparse_"
     "swa.py L494-502）",
     str(REFS / "vllm" / "swiglu-tail-pin.json"),
     "vllm @f358f0578a83f4e1fe6fc49a492d3ae9362de8e5；13 文件 blob SHA1 对 "
     "vllm-tree.json 逐项校验 + SHA256 清单",
     "V4.1 SwiGLU clamp 激活（count/moe.moe_act/moe_shared_act 行 6 算术 "
     "op/元素；model_ir extras.swiglu + swiglu_limit 键 mapped）与 "
     "compress_ratios 尾部归属（extras.kv_spec.draft_compress_ratios；"
     "长度域 L 或 L+mtp、非零尾部域外拒绝）；公式 v2.swiglu_clamp_act",
     ["源码阅读 source-checked；kernel/CPU forward/GPU 均未执行",
      "clamp CUDA kernel 与 native 的数值等价未执行（vLLM 侧测试在册 "
      "tests/kernels/core/test_activation.py L62-66/L251-285，本仓未跑）",
      "masked 每 expert padded 激活路径（fused_moe/activation.py L246-260）"
      "与 pad-aware Triton 跳过未路由槽/非本地专家的运行时差异未入账"
      "（utils.py L801-818；本仓按逻辑 assignment 计费，与 moe_routed 同口径）",
      "SWIGLUOAI/SWIGLUSTEP 家族常数（limit=7.0/alpha=1.702，fused_moe/"
      "activation.py L268-275）为 V4.1 域外变体，未建模",
      "native MegaMoE activation clamp（deepseek_v4/nvidia/model.py "
      "L393-394）语义同源但 SM100 域外（注册执行域 Marlin），未执行",
      "DSpark 草稿层本体未建模：非零尾部值（草稿层消费压缩 KV）无 official"
      " 证据，按域外拒绝；草稿层 MoE（E=128/k=3）维持 R21 方法层登记",
      "op 计数为解析算术口径（clamp 比较/sigmoid/乘各 1，非 kernel 指令"
      "数）；Qwen36 家族无 extras.swiglu，其 silu 微操作维持 R06 冻结口径"
      "（跨模型计量差异已登记）"],
     False, "2026-11-13"),
    ("ev-vllm-v41-dspark", "source-checked",
     "vLLM V4.1 DSpark speculator：num_reqs 批处理、复用 V4 decoder 的事实",
     str(REFS / "vllm" / "vllm" / "v1" / "worker" / "gpu" / "spec_decode" / "dspark" / "speculator.py"),
     "vllm @f358f0578（评审包 pin）",
     "DSpark 方法语义参考（块 backbone/Markov/confidence）",
     ["DSpark 不在本仓必需场景内；链式代理不得冒名"], False),
    ("ev-oldrepo-principles", "source-checked",
     "旧仓计算原理固定原文（迁移对账的历史基线）",
     str(REFS / "model-verse-principles-pinned.md"),
     "model-verse 指定提交（git show 固定）",
     "旧口径数值对账（含已知错误 38.9 等）",
     ["原文含已归档错误（E14），只作历史对照"], False),
    ("ev-vllm-parallel-recipe", "source-checked",
     "vLLM TP/EP 并行分片配方（Qwen3.5/3.6 qwen3_5_moe + DFlash 集成）：MoE "
     "EP 关 inter/tp 切分（fused_moe/config.py L1402）、EP 开整专家分置"
     "（FusedMoEParallelConfig.make L1157-1288 + expert_map_manager 余数规则）、"
     "router GateLinear replicated、shared expert TP 切分、attention KV 头 "
     "kvh<tp 复制（qwen3_next.py L302-315）、GDN in_proj_qkvz/ba/conv 列切与"
     "状态切分（qwen_gdn_linear_attn.py + mamba_utils state_shape）、词表行切、"
     "DFlash 草稿同 TP 组并共享 target embed/lm_head 模块对象"
     "（spec_decode/dflash/utils.py load_dflash_model）",
     "identity/vllm_qwen35_pin/manifest.json",   # 逐文件 SHA256 在该 manifest
     "vllm-project/vllm @dc74b8cbbbe3dbb9399fe369757d63defd614cc2（main 2026-11-10 "
     "经 GitHub raw 网络获取固定；文件级 SHA256 清单见目录内 manifest.json）",
     "R01 per-rank 分片配方（count/memory/lowering 共用 ParallelPlan）；"
     "R02a 草稿共享对象身份",
     ["源码阅读（source-checked），非执行验证；量化/Marlin 特殊分支"
      "（ba_proj disable_tp 等）未入配方，BF16 路径为准",
      "tp×ep 独立组合（ep∉{1,tp}）无实现配方，显式拒绝"],
     False),
    ("ev-vllm-logits-ctxkv-billing", "source-checked",
     "vLLM lm_head logits 集合通信与 DFlash ctx KV 二次读/融合驻留缓冲（"
     "R01/R02a）：LogitsProcessor._get_logits 在 lm_head."
     "tp_size>1 时每次 lm_head 调用 gather logits（all-gather 或 gather-"
     "to-rank0，logits_processor.py L90-91/122-133/185-198；qwen3_5.py "
     "L345/428、qwen3_dflash.py L920-935 草稿同路径）；DFlash 草稿 ctx "
     "行经 k_proj/v_proj 第二次 GEMM（dflash model.py L384-391）。"
     "qwen3_dflash.py L108-113/529-531/541-568/617-698 三分支："
     "raw 融合 cat 副本 / MXFP8 独立 _context_kv_proj / 逐层完整 qkv "
     "回退后切 Q；k_norm 堆叠三路径均在",
     "identity/vllm_qwen35_pin/manifest.json",
     "vllm-project/vllm @dc74b8cbbbe3dbb9399fe369757d63defd614cc2（同 "
     "ev-vllm-parallel-recipe pin；logits_processor.py 二轮补入，SHA256 "
     "e1fac869aef80f6c398b92d61444186d1921740dab7df41b95a260b08b4c25d2）",
     "R01 logits_gather 通信计费（decode/verify/草稿候选各一次）；"
     "R02a 草稿 ctx-KV 路径的驻留对象与二次读流量（count/draft + "
     "lowering/memory 同一显式 draft_ctx_kv_path）",
     ["gather 平台分支（all-gather vs rank0-gather）按同一每 rank 字节"
      "上界建模，延迟按单向环 (tp−1) 段近似；ctx-KV 路径不由 dense_w "
      "推断；MXFP8 block-scale 布局未入 pin（只计 packed 下界并传播 "
      "unknown）；源码阅读非执行验证"],
     False),
    ("ev-vllm-dflash-norm-dtype", "source-checked",
     "DFlash 草稿 RMSNorm 参数 dtype（k_norm 堆叠字节）：RMSNorm 未传 "
     "dtype 时以 torch.get_default_dtype() 建 weight；vLLM "
     "BaseModelLoader 在 set_default_torch_dtype(model_config.dtype) 内"
     "初始化草稿，故 _k_norm_weights 继承草稿 config dtype=bfloat16，"
     "6×128×2=1,536 B，与 draft_quant.dense_w 无关",
     "identity/vllm_qwen35_pin/dflash_utils.py",
     "vllm-project/vllm @dc74b8cbbbe3dbb9399fe369757d63defd614cc2："
     "layers/layernorm.py L42-58；model_loader/base_loader.py L68-85"
     "（两文件未入本地 pin，按固定 commit + 行号登记；qwen3_dflash.py "
     "L288-289/565-568 已在 pin）",
     "R02a k_norm 堆叠常驻字节（lowering/memory.draft_extra_allocations）；"
     "DraftIR.dtype_hint 来源与未知 dtype 拒绝口径",
     ["layernorm.py/base_loader.py 未入本地 pin，未执行；Python 默认 dtype "
      "被调用方覆盖的运行期场景未建模；源码阅读非执行验证"],
     False),
    ("ev-vllm-dflash-serving", "source-checked",
     "DFlash v1 服务栈（vLLM PR#40898 路径）逐行核对（R07a）：colocated "
     "proposer 每轮 in-process propose()（1 bonus+N mask 查询块）；aux "
     "hidden=target_layer_ids(+1) 经 fc(concat) 投影；ctx K/V 每轮对整 "
     "verify 跨度 B×blk 投影、被拒后缀 PAD 不落缓存（测试负例在册）；"
     "拒绝采样=共享 RejectionSampler（greedy argmax 前缀/随机比值检验+"
     "目标分布 bonus）；draft_sample_method 默认 greedy 全量 logits "
     "gather（local_argmax_reduction 可选、默认关）；PIECEWISE CG 不支持",
     "identity/vllm_dflash_serving_pin/manifest.json",
     "vllm-project/vllm @dc74b8cbbbe3dbb9399fe369757d63defd614cc2（与 "
     "vllm_qwen35_pin 同 SHA；13 文件 SHA256 清单在 manifest：dflash/"
     "speculator+cudagraph+config、eagle3_utils、上级 speculator、"
     "rejection_sampler(+utils)、config/speculative、4 个 DFlash 测试）",
     "R07a G-C3：dflash_serving_integration(colocated)=supported 的源码"
     "依据；count/draft span 基修正（fc/hidden_norm/ctx_kv_proj + ctx K "
     "norm/rope + vLLM layout copy）",
     ["source-checked 源码阅读，非 CPU 执行/实测；DraftLogitsCache"
      "（probabilistic）未入 R02b 账本（场景无 draft_sample_method 键）；"
      "FULL graph 桶取整 padding 行未计"],
     False, "2026-11-12"),
    ("ev-sglang-dflash-serving", "source-checked",
     "DFLASH v1 服务栈（SGLang）逐行核对（R07a）：DFlashWorkerV2 同驱 "
     "overlap/非 overlap；轮=prepare 块→草稿前向（非因果）→借目标 "
     "lm_head 产候选（tp>1 局部 (max,id) all-gather，非全量 gather）→"
     "目标 verify（capture FULL）→Triton accept+bonus（new_seq_lens="
     "prefix+commit）→全跨度 fc+hidden_norm→逐层 ctx KV 只写 committed "
     "行→next_draft_input；mamba/GDN 态回退 last_steps=commit−1；非贪婪"
     "采样 kernel 不可用时降级 greedy（分布不保持）；disagg=target 侧 "
     "PD 非 DT",
     "identity/sglang_dflash_pin/manifest.json",
     "sgl-project/sglang @74cb973d78dfa4af85a90d2841891fec27e78db2"
     "（main 2026-11-12；12 文件 SHA256 清单在 manifest：dflash_worker_v2、"
     "dflash_info(_v2)、dflash_utils、dflash_dp、dflash_disaggregation、"
     "spec_info、models/dflash、kernels dflash+candidates、2 个 e2e 测试）",
     "R07a G-C3：dflash_serving_integration(colocated)=supported 的第二栈"
     "源码依据；两栈通信口径差异（greedy 局部 argmax vs 全量 gather）与 "
     "dt 模式无栈对应的判定依据",
     ["source-checked 源码阅读，非 CPU 执行/实测；采样验证 kernel 可用性"
      "（构建/设备相关）unknown；greedy (max,id) all-gather 字节未入本仓"
      "通信行（按 vLLM 全量口径计，差异披露在 R07a 报告）；GDN 态回退"
      "extra_buffer 拷贝成本未计价；vLLM permute+contiguous 全局 layout "
      "拷贝行不对应 SGLang（逐层 store_prefix）"],
     False, "2026-11-12"),
    ("ev-h200-tiny-distill", "measured",
     "H200 实测 profile 蒸馏集（Qwen3-30B-A3B-tiny 复制品采集：linear 6 算子 ×387 "
     "token 网格 + decode 注意力 5184 点，CUDA_EVENT median；环境 H200/driver 570.124.06/"
     "torch 2.8.0+cu128）",
     "profiles/h200_qwen3_30b_a3b_tiny/manifest.json",
     "pi-viewer @c5909a7699a1d2bb422d931bdca6cba399451f00 "
     "data/profiling/compute/h200/Qwen3-30B-A3B-tiny/（逐文件 SHA256 在 manifest；"
     "采集由 pi-viewer frontier/profiling 在 H200 完成，本仓只蒸馏导入）",
     "算子级成本校准（TP1/BF16/FLASHINFER；linear tokens 1..16384；"
     "decode B 1..64 × kv 0..4095）",
     ["16 专家缩小复制品：MoE/router/lm_head 不在域内（回退解析）；"
      "不是 Qwen3.6-35BA3B+DFlash 场景的真实性证据；无端到端实测对照；"
      "本会话未做新的硬件测量"],
     True),
    ("ev-qwen36-cpu-fixture-r06", "shape-checked",
     "Qwen3.6 GDN+MoE 小尺寸 CPU forward 对照（R06）：pin 官方 forward 随机权重"
     "执行（GDN prefill/decode + FullAttn + 256→4E MoE 缩比），逐 GEMM/conv/router "
     "计数与本仓 build_model_work(tp=1) 精确对账；关键张量形状/递推状态几何/"
     "因果对数/无放回 top-k assignment 核验",
     "tests/fixtures/qwen36_r06_cpu_fixture.json",
     "HF pin modeling/configuration_qwen3_5_moe @b6401d66（in-repo "
     "assets/identity/hf_transformers）+ torch 2.15.0+cpu / transformers 5.22.1 "
     "/ py3.12 本机执行",
     "fixture 维度 h=32,2L(GDN+Full),4E top-2, nv/nk=4/2, dk=dv=8, conv4；"
     "不外推真实 35B 尺寸",
     ["随机权重小尺寸 fixture，不覆盖真实 checkpoint/MTP/视觉塔/mrope 位置 id/"
      "融合 kernel（fla/causal-conv1d 未装，走 PyTorch 参考实现）",
      "gdn_recurrence 4·state_elems 是注册 approximated 口径（官方参考递推 "
      "≈7 ops/elem，报告披露），本轮未改",
      "CPU 执行为 fixture 级证据；模型级 cpu_forward 保持 not-run，"
      "不构成完整 forward 或部署验证"],
     True, "2026-11-12"),
    ("ev-qwen36-cpu-numeric-r06", "numeric-checked",
     "Qwen3.6 两层随机权重 CPU fixture：独立张量参考逐元素对照 pinned HF "
     "prefill 与带缓存 decode 的 GDN、MoE、full-attn 门控、层输出、"
     "conv/GDN/KV 状态和 logits",
     "tests/fixtures/qwen36_r06_numeric_oracle.json",
     "HF pin modeling/configuration @b6401d66；"
     "tools/probe_qwen36_numeric_oracle.py 独立 PyTorch 张量参考；"
     "torch 2.15.0+cpu / transformers 5.22.1 / Python 3.12 本机执行",
     "h=32、两层 GDN+FullAttn、4 expert top-2、T=5 prefill 与 T=1 "
     "带缓存 decode；atol=1e-5、rtol=1e-4",
     ["只验证随机权重小尺寸文本 fixture，不覆盖真实 35B checkpoint/MTP/"
      "视觉塔/融合 GPU kernel 或部署性能",
      "RoPE cos/sin 取 pinned forward 的输入作为对照边界，mrope 位置生成未验证",
      "模型级 cpu_forward 仍 not-run；部署排名资格不变"],
     True, "2026-11-12"),
    ("ev-h200-235b-distill", "measured",
     "H200 Qwen3-235B-A22B TP2/8 算子级 profile 蒸馏集（历史归档）",
     "profiles/h200_qwen3_235b_a22b/manifest.json",
     "pi-viewer @c5909a7699a1d2bb422d931bdca6cba399451f00 "
     "data/profiling/compute/h200/Qwen3-235B-A22B/（逐文件 SHA256 在 manifest；"
     "本仓只蒸馏导入）",
     "历史算子级成本校准（TP2/8、BF16、FLASHINFER；MoE defer）",
     ["模型已移出活跃 catalog；profile 绑定 235B，禁止跨模型使用；"
      "不是 Qwen3.6/V4.1 的真实性证据；无端到端实测对照"],
     True),
    ("ev-vllm-qwen36-mtp", "source-checked",
     "Qwen3.6-35B-A3B 原生 MTP 头（R23）：vLLM Qwen3_5MoeMTP 逐行核对——"
     "combine=双预归一+cat+fc(2h→h, ColumnParallel gather_output=True, 无激活"
     "/无残差直加)；层=full_attention decoder 层（gated GQA+MoE，几何=目标 "
     "full 层）×n_predict（Qwen3.6=1）；层选择 spec_step_idx%n（K>n 循环复用）；"
     "加载只消费 mtp.*/embed_tokens/lm_head（mtp.→model. 重映射）；colocated "
     "经 load_eagle_model 物理共享目标 embed/lm_head（零字节）；每轮 K 均匀步"
     "（无 catch-up）；final norm+共享 lm_head 每步读 1 候选",
     "identity/vllm_qwen36_mtp_pin/manifest.json",
     "vllm-project/vllm @10a854ed6b3bbd7411c65ccbf87594e71c4d322c"
     "（8 文件 SHA256/blob sha1 清单在 manifest：qwen3_5_mtp、qwen3_next_mtp、"
     "config/speculative、eagle/utils、mtp/target_dependent_ar speculator、"
     "2 个 config 测试 + e2e 测试）",
     "R23：count/draft.mtp_step_rows、methods/plans.plan_mtp、lowering/memory "
     "MTP 对象、pipeline mtp 分支（逐步阶段+草稿通信+dt 拒绝）；"
     "method_status['mtp']=source-checked 的依据",
     ["source-checked 源码阅读，非 CPU 执行/实测；真实 checkpoint mtp.* 权重"
      "未本机加载（名单由 e2e weight_map 复制逻辑+loader 重映射证明）",
      "n_predict>1 多模块（MultiModuleMTPSpeculator/index_share）、"
      "draft_lm_head_quantization、local_argmax_reduction、probabilistic "
      "采样档未建模",
      "草稿 prefill KV 同步（tdar _prefill 全量执行）未入稳态 TPOT 账本",
      "HF transformers 无 MTP forward（主模型忽略 mtp.*）——HF 栏与推理栈栏"
      "分立，不能互相冒充"],
     False, "2026-11-13"),
    ("ev-sglang-qwen36-mtp", "source-checked",
     "SGLang Qwen3_5ForCausalLMMTP 对照核对（R23）：fc nn.Linear(2h→h)+双预归一"
     "（GemmaRMSNorm）；mtp_config.full_attention_interval=1/num_hidden_layers=1；"
     "set_embed_and_head/set_lm_head_from_target 权重物理共享（del 后指目标权重）；"
     "forward cat/fc 同构——跨栈确认主干语义",
     "identity/sglang_qwen36_mtp_pin/manifest.json",
     "sgl-project/sglang @1b2870d33de7b5f31844fd74e97eea6d7896c53a"
     "（1 文件 SHA256/blob sha1 清单在 manifest）",
     "R23：MTP 头主干语义的第二栈依据（vLLM 主 pin 的跨栈确认）",
     ["source-checked 非执行；SGLang NPU/量化分支（ascend unquant override）"
      "未建模；服务循环细节（worker 调度）未逐行核（主干 forward 语义对照）"],
     False, "2026-11-13"),
    ("ev-qwen36-mtp-cpu-numeric", "numeric-checked",
     "Qwen3.6 原生 MTP 头小尺寸独立数学对照（R23）：pinned vLLM 结构的忠实"
     "torch 复刻（fc+双预归一+full-attn 门控 GQA+MoE 路由+共享专家+final norm）"
     "与独立逐 token 标量参考互相逐元素核对；中间张量形状与输出数值、以及"
     "本仓 mtp_step_rows 逐算子 FLOPs/字节对账",
     "tests/fixtures/qwen36_mtp_r23_numeric.json",
     "tools/probe_qwen36_mtp_numeric.py（torch 2.15.0+cpu / Python 3.12 "
     "本机执行；结构按 assets/identity/vllm_qwen36_mtp_pin @10a854ed 手工"
     "复刻，双实现互核）",
     "h=32、1 full-attn 层（4 头/kv2/d16）、4E top-2+shared、V=97、"
     "K=3 步循环（MTP1 同层复用）；atol=1e-5",
     ["只验证随机权重小尺寸 MTP 头数学，不覆盖真实 checkpoint 权重/融合 "
      "kernel/部署性能；vLLM 模块本身未在 CPU 执行（复刻实现，结构逐行对源）",
      "fixture 级证据：模型级 cpu_forward 保持 not-run，不构成完整 forward "
      "或部署验证；部署排名资格不变"],
     True, "2026-11-13"),
]


def _sha256_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_revision() -> str:
    """当前代码身份（git HEAD；不可用时如实返回 unknown）。"""
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def build_registry() -> Dict[str, EvidenceRecord]:
    records: Dict[str, EvidenceRecord] = {}
    for ev_id, rel in _IDENTITY_FILES.items():
        path = ASSETS / rel
        sha = _sha256_file(path)
        if not sha:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        repos, revisions = {}, {}
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, dict):
                    repos[k] = v.get("repo_id")
                    revisions[k] = v.get("revision_main_sha")
        elif isinstance(data, list):
            repos = {str(i): m.get("id") for i, m in enumerate(data[:50])
                     if isinstance(m, dict)}
        records[ev_id] = EvidenceRecord(
            evidence_id=ev_id, kind="source-checked",
            claim=f"必需场景资产身份抓取记录：{repos}",
            source=f"HF API + 文件 SHA256 落盘（{rel}）",
            content_sha256=sha, captured_at="2026-11-09",
            domain="模型/checkpoint 身份",
            limits=["身份核实 ≠ 语义验证；configs 另有单列证据"],
            executed=True, command=f"py -3.12 <identity fetch> → {rel}")
    for entry in _PINNED_ASSETS:
        (ev_id, kind, claim, rel, source, domain, limits, executed) = entry[:8]
        # 可选第 9 元素：登记日期（缺省 2026-11-09，与既有记录一致）
        captured = entry[8] if len(entry) > 8 else "2026-11-09"
        # docs/ 与 tests/ 相对仓根解析（ev-vllm-* 用 docs/ 前缀先例；
        # R06 fixture 证据绑定 tests/fixtures/ 下落盘产物）
        path = (ASSETS / rel) if not rel.startswith(("docs", "tests")) \
            else ROOT / rel
        sha = _sha256_file(path)
        if not sha:
            # 资产缺失：登记为缺口而不是跳过（如实标注原则）
            records[ev_id] = EvidenceRecord(
                evidence_id=ev_id, kind="source-checked", claim=claim,
                source=source, content_sha256="MISSING", captured_at=captured,
                domain=domain, limits=["资产文件缺失：" + rel], executed=False)
            continue
        records[ev_id] = EvidenceRecord(
            evidence_id=ev_id, kind=kind, claim=claim, source=source,
            content_sha256=sha, captured_at=captured, domain=domain,
            limits=limits, executed=executed,
            command=f"sha256sum {rel}")
    return records


def registry_json() -> dict:
    return {"schema_version": "0.2", "code_revision": code_revision(),
            "evidence": {k: v.to_json() for k, v in build_registry().items()}}


def evidence_index() -> List[str]:
    return sorted(build_registry())
