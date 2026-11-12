"""config → ModelIR 家族解析器 + 字段覆盖登记。

每个解析器：(config_dict, SourceIdentity) → ModelIR。所有家族对未知/未消费
键显式登记 Coverage（M0 硬门槛）；未消费但确认不影响文本推理结构的键给
metadata-only 或 ignored-with-reason；影响结构而未建模的键给
unsupported-semantic 并写入 gaps（阻断完整结构结论）。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .schema import (FULL, LINEAR, MLA, MLA_CSA2, SLIDING, AttentionIR, Csa2IR,
                     LayerIR, LinearAttnIR, MlaIR, ModelIR, MoeIR, SourceIdentity)

# 每家族的结构键白名单（映射进 IR 的键）；其余键按规则归类
_STRUCT_KEYS = {
    "gqa": {"hidden_size", "num_hidden_layers", "num_attention_heads",
            "num_key_value_heads", "head_dim", "vocab_size", "intermediate_size",
            "max_position_embeddings", "tie_word_embeddings", "rope_theta"},
    "qwen3_moe": {"hidden_size", "num_hidden_layers", "num_attention_heads",
                  "num_key_value_heads", "head_dim", "vocab_size", "num_experts",
                  "num_experts_per_tok", "moe_intermediate_size",
                  "intermediate_size", "decoder_sparse_step",
                  "norm_topk_prob", "max_position_embeddings", "tie_word_embeddings",
                  "rope_theta", "mlp_only_layers"},
    "qwen3_5_moe": {"layer_types", "full_attention_interval", "hidden_size",
                    "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
                    "head_dim", "vocab_size", "num_experts", "num_experts_per_tok",
                    "moe_intermediate_size", "intermediate_size",
                    "shared_expert_intermediate_size",
                    "linear_num_key_heads", "linear_num_value_heads",
                    "linear_key_head_dim", "linear_value_head_dim",
                    "linear_conv_kernel_dim", "partial_rotary_factor",
                    "rope_parameters", "max_position_embeddings",
                    "tie_word_embeddings", "mtp_num_hidden_layers",
                    "norm_topk_prob", "use_sliding_window",
                    "mtp_use_dedicated_embeddings", "attn_output_gate",
                    "decoder_sparse_step"},
    "qwen3_next": {"layer_types", "full_attention_interval", "hidden_size",
                   "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
                   "head_dim", "vocab_size", "num_experts", "num_experts_per_tok",
                   "moe_intermediate_size", "intermediate_size",
                   "shared_expert_intermediate_size",
                   "linear_num_key_heads", "linear_num_value_heads",
                   "linear_key_head_dim", "linear_value_head_dim",
                   "linear_conv_kernel_dim", "partial_rotary_factor",
                   "rope_parameters", "max_position_embeddings",
                   "tie_word_embeddings", "decoder_sparse_step", "mlp_only_layers",
                   "norm_topk_prob", "use_sliding_window"},
    "deepseek_v3": {"hidden_size", "num_hidden_layers", "num_attention_heads",
                    "num_key_value_heads", "q_lora_rank", "kv_lora_rank",
                    "qk_rope_head_dim", "qk_nope_head_dim", "v_head_dim",
                    "vocab_size", "n_routed_experts", "n_shared_experts",
                    "num_experts_per_tok", "moe_intermediate_size",
                    "intermediate_size", "num_nextn_predict_layers",
                    "first_k_dense_replace", "moe_layer_freq", "routed_scaling_factor",
                    "scoring_func", "topk_method", "max_position_embeddings",
                    "tie_word_embeddings", "norm_topk_prob"},
    "dflash": {"hidden_size", "num_hidden_layers", "num_attention_heads",
               "num_key_value_heads", "head_dim", "intermediate_size",
               "vocab_size", "layer_types", "sliding_window", "max_window_layers",
               "num_target_layers", "use_sliding_window", "max_position_embeddings",
               "tie_word_embeddings", "is_causal"},
    "v41_csa2": {"hidden_size", "num_hidden_layers", "num_attention_heads",
                 "num_key_value_heads", "head_dim", "q_lora_rank", "o_lora_rank",
                 "o_groups", "qk_rope_head_dim", "index_head_dim", "index_n_heads",
                 "index_topk", "candidate_topk_blocks", "candidate_source_layer_id",
                 "sliding_window", "compress_ratios", "kv_source_layer_ids",
                 "index_source_layer_ids", "vocab_size", "n_routed_experts",
                 "n_shared_experts", "num_experts_per_tok", "moe_intermediate_size",
                 "routed_scaling_factor", "scoring_func", "topk_method",
                 "max_position_embeddings", "tie_word_embeddings",
                 "num_nextn_predict_layers", "norm_topk_prob",
                 # mHC（2026-11-10 mapped：count/work.hc_boundary_rows +
                 # lowering/memory hc 权重；vLLM pin source-checked）
                 "hc_mult", "hc_eps", "hc_sinkhorn_iters",
                 # Engram（2026-11-11 R04 mapped：common/engram.py pin
                 # @f358f057 blob ba4ceea6 source-checked；count/work.engram_rows
                 # + lowering/memory engram 表/wkv/qk 对象；逐键覆盖注记见
                 # parse_v41_csa2 的 _ENGRAM_KEY_NOTES）
                 "engram_layer_ids", "engram_num_embeddings", "engram_vocab_size",
                 "engram_compressed_vocab_size", "engram_max_ngram_size",
                 "engram_n_heads", "engram_head_dim", "engram_pad_token_id"},
}

# Engram 8 键的 mapped 覆盖注记（engram.py = vllm/models/deepseek_v41/common/
# engram.py pin @f358f057，blob sha1 ba4ceea6；逐键给行号证据）
_ENGRAM_KEY_NOTES = {
    "engram_layer_ids":
        "mapped：注入层位 [1,14]（EngramLayout L221；model.py L350-361 构造、"
        "L583-600 注入 residual 流）",
    "engram_num_embeddings":
        "mapped：每层表行数 = 该层 24 个素数桶长之和（EngramLayout L230-242 "
        "素数分配；L1=384,006,168 / L14=384,016,682 与本键精确相等）",
    "engram_vocab_size":
        "mapped：语义 = 哈希空间 / 素数扫描起点（find_next_prime 从 "
        "vocab_size−1=15,999,999 起抽素数，L130-135/L235-239），**不是存储规模**；"
        "存储规模是 engram_num_embeddings",
    "engram_compressed_vocab_size":
        "mapped：tokenizer 归一化压缩词表 99092，断言一致（L457-464）；"
        "哈希乘子由它派生（compute_hash_multipliers L189-208）",
    "engram_max_ngram_size":
        "mapped：n-gram 最大尺寸 4（lookback_depth=3，L444）；哈希列数 "
        "(max−1)×n_heads=24（L243）",
    "engram_n_heads":
        "mapped：每 (ngram 尺寸, head) 独立素数桶，8 heads × 3 组 = 24 列/层"
        "（L232-241）",
    "engram_head_dim":
        "mapped：表值 head_dim=256；FP8 行 256B + E8M0 scale 8B = 264B/行"
        "（L1152-1161/L968-969）",
    "engram_pad_token_id":
        "mapped：blocked/dead 位置哈希输入取 pad（L411 token_map 归一），"
        "参与哈希语义",
}

# R20（2026-11-12）：CSA2 索引打分/top-k/candidate block 键的 mapped 注记
# （证据 csa2-pin.json 21 文件 + ev-vllm-v41-csa2；消费者
# count/work.v41_layer_rows、lowering/memory、scenario.v41_kv_record）。
_CSA2_KEY_NOTES = {
    "index_topk":
        "mapped：每 index source 层选 top-512 压缩 token（attention.py L190/"
        "L1463；topk_indices_buffer [max_num_batched_tokens,512] int32，"
        "nvidia/model.py L781-785）；短上下文 ctx//r≤512 走 eager 全选旁路"
        "（attention.py L1656-1680）",
    "index_n_heads":
        "mapped：indexer MQA 头数 32（attention.py L1464）；wq_b q_lora→"
        "4096 replicated、weights_proj h→32 replicated bf16（quant_config=None，"
        "L1477-1490）",
    "index_head_dim":
        "mapped：索引头宽 128；K 记录 FP8 128B+4B scale=132B/行"
        "（attention.py L181-189；indexer_kv_dtype auto→fp8）",
    "index_source_layer_ids":
        "mapped：[2,8,14,20,24,28,32,36] 拥有 indexer；非持有层复用下方最近"
        "源的 topk（attention.py L393-422/L502-553）；K 缓存仅 kv source 持有、"
        "其余共享（L504-530）",
    "kv_source_layer_ids":
        "mapped：[2,8,14,20] 持有 compressor+压缩 KV+indexer K（attention.py"
        " L404/L508-517/L654-664）；消费者经 forward_context 读源层缓存"
        "（L667-682）",
    "candidate_source_layer_id":
        "mapped：层 20 的 indexer 发布候选块（candidate_write，attention.py"
        " L531-553；select_candidate_blocks 每 8 压缩位一块、最新块钉 +inf，"
        "candidate_blocks.py L168-215）",
    "candidate_topk_blocks":
        "mapped：候选块缓冲 [max_num_batched_tokens,2048] int32（nvidia/"
        "model.py L793-798）；消费层（>20 的 index source）打分后先 mask 到"
        "候选块再自有 top-k（sparse_attn_indexer.py L895-911/L1141-1153）",
    "candidate_block_size":
        "mapped：块粒度 8 压缩位（config；select/apply_candidate_mask 的 "
        "BLOCK_SIZE，candidate_blocks.py L51/L160-163）",
    "sliding_window":
        "mapped：SWA 窗 128（attention.py L373；SWA 记录 584/528 B/token 见"
        " scenario.v41_kv_record）",
    "compress_ratios":
        "mapped（主干 40 项）：0=纯 SWA、1=全压缩、2=2:1 池化（sparse_mla.py"
        " L34-43；compressor.py L295-310 ratio1 无门控）；MTP 尾部另列缺口",
}

# R21（2026-11-12）：V4.1 MoE 路由键的 mapped 注记（证据 moe-pin.json
# 14 文件 + ev-vllm-v41-moe；消费者 count/moe.v41_moe_route_rows、
# lowering/memory 路由偏置对象）。
# 本 PR 的 source-checked dtype 域只覆盖官方 V4.1 固定资产（config
# dtype=bfloat16）。gate 输入不是由任意 model dtype 推出：pinned mHC
# tilelang.py 的 layer_input_cur 硬编码 bfloat16（R05 pin L900/L1096），
# DeepseekV4Layer.forward 把它直接传给 self.ffn(x)（model.py L731-816）。
_V41_DTYPE_DOMAIN = "official_v41_bfloat16"
_V41_BF16_ITEMSIZE = 2

# R22：official V4.1-Flash 固定资产的 quantization_config（无 full_config
# 的内部调用按此默认；与 model_dtype 默认同模式——不是任意量化回退）。
_V41_DEFAULT_QUANT_CFG = {
    "quant_method": "fp8",
    "activation_scheme": "dynamic",
    "weight_block_size": [32, 32],
    "scale_fmt": "ue8m0",
    "expert_dtype": "fp4",
}

# R22：quantization_config 叶键 mapped 注记（20 文件 pin quant-pin.json；
# consumer=count/moe+work、lowering/memory、count/quant_layout）。
_V41_QUANT_KEY_NOTES = {
    "quant_method":
        "mapped：fp8 → override deepseek_v4_fp8（model_type deepseek_v41* → "
        "DeepseekV41FP8Config；config/model.py L1429/L1444-1465 override 链、"
        "quant_config.py L163-192、quantization/__init__.py L120-132）；"
        "checkpoint 必须 fp8 序列化（fp8.py L103-109 拒绝在线量化）",
    "activation_scheme":
        "mapped：dynamic——块量化硬要求（fp8.py L125-130）；激活 scale 不落盘"
        "（KDynamicNoParam，modelopt.py L2497-2510），运行时按 kernel 域"
        "量化（SM90 Marlin W8A16/W4A16 实为 BF16 激活，见 extras.quant）",
    "weight_block_size":
        "mapped：[32,32]——线性层走 MXFP8 分支的必要条件（quant_config.py "
        "L221-243）；checkpoint scale 32×32 块，装载展开为 (1,32) 运行时块"
        "（modelopt.py L2424-2479/L2430-2445）；count/memory 按 N·K/32 计",
    "scale_fmt":
        "mapped：ue8m0——声明与派生一致；loader 实际不读该键（fp8.py "
        "L155-172 键表无 scale_fmt），有效 scale dtype 由 expert_dtype=='fp4'"
        "→is_scale_e8m0 派生（quant_config.py L89-93），非 ue8m0 值按声明"
        "矛盾拒绝（不静默取派生值）",
    "expert_dtype":
        "mapped：fp4——config 类把它提升到 hf_config 顶层（transformers_utils/"
        "configs/deepseek_v41.py L46-50）；驱动两条 dispatch：线性 is_scale_e8m0"
        "（ue8m0 1B scale）+ 路由专家 Mxfp4MoEMethod(activation_key="
        "kMxfp8Dynamic)（quant_config.py L244-261）；scale_fmt 键的 dtype 由此"
        "派生而非 loader 读取",
}

_MOE_ROUTE_KEY_NOTES = {
    "scoring_func":
        "mapped：sqrtsoftplus——weights=sqrt(softplus(logits))，>20 取 "
        "sqrt(logits) 护栏（dsv4_topk.py L85；cpu_router.py L90-95 同语义"
        " docstring）；仅此一档（mega/model.py L255-258 断言 sqrtsoftplus）",
    "topk_method":
        "mapped：noaux_tc——e_score_correction_bias[E] fp32 只导选择不进权重"
        "（deepseek_v4/nvidia/model.py L303-313 建 Parameter；dsv4_topk.py "
        "L85-87 选择列=weights+bias、L96-98 权重取无偏 score）",
    "norm_topk_prob":
        "mapped：true——选中权重 sum→1 归一（dsv4_topk.py L104-107；"
        "cpu_router.py L107 assert renormalize；can_use_dsv4_topk 要求 "
        "renormalize）",
    "routed_scaling_factor":
        "mapped：1.5——归一后乘 routed_scaling_factor/sum（sum≤0/NaN 分母"
        "取 1，dsv4_topk.py L104-107）；只作用于 routed。共享专家（无 gate）在"
        "routed 归一后以权重 1.0 独立相加（moe_output.py L151-175 的"
        " shared_output 契约）；fused_topk_bias_router.py L447-464 的"
        "共享槽是通用融合能力，本模型 FusedMoE 调用未启用（未传"
        " n_shared_experts/fuse_shared_experts，model.py L426-449）",
}


def _is_prime(n: int) -> bool:
    """确定性 Miller-Rabin（n < 2**32 有效；基 {2,7,61}）。

    common/engram.py L105-127 `_is_prime` 的忠实移植（含小素数试除短路）；
    测试用**独立实现**对账，不 import 本函数当 oracle。
    """
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 7, 61):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def _find_next_prime(start: int, seen: set) -> int:
    """大于 start 且未在 seen 中出现过的最小素数（engram.py L130-135）。"""
    candidate = start + 1
    while not _is_prime(candidate) or candidate in seen:
        candidate += 1
    return candidate


def engram_bucket_primes(vocab_size: int, n_layers: int, max_ngram_size: int,
                         n_heads: int) -> List[List[int]]:
    """EngramLayout 的素数桶分配（engram.py L211-252 的忠实移植）。

    每个 (ngram 尺寸, head) 对拥有独立 prime 桶长：每 ngram 组从
    vocab_size−1 重置起点顺序抽 n_heads 个素数，但 seen 集合**全局**不复用
    → 实际为 vocab_size−1 之上连续 n_layers×(max−1)×n_heads 个素数，
    层 i 取第 [i×24, (i+1)×24) 个。返回 [n_layers][n_hash_cols] 桶长；
    TP 分片时每 rank 行数 = 其分到的桶长之和（**不是** num_embeddings/tp）。
    """
    seen: set = set()
    out: List[List[int]] = []
    for _ in range(n_layers):
        flat: List[int] = []
        for _ in range(max_ngram_size - 1):
            current = vocab_size - 1
            for _ in range(n_heads):
                current = _find_next_prime(current, seen)
                seen.add(current)
                flat.append(current)
        out.append(flat)
    return out

# 已核不影响文本推理结构、也不进 IR 的键（metadata/ignored 分类）
_META_KEYS = {
    "attention_bias", "attention_dropout", "hidden_act", "initializer_range",
    "rms_norm_eps", "rope_theta", "use_cache", "output_router_logits",
    "router_aux_loss_coef", "dtype", "bos_token_id", "eos_token_id",
    "pad_token_id", "auto_map", "architectures", "model_type",
    "transformers_version", "mamba_ssm_dtype", "linear_conv_activation",
    "torch_dtype",
    "rope_scaling", "logit_scale", "tie_word_embeddings",
}
_TRAIN_ONLY_KEYS = {
    "attention_dropout", "router_aux_loss_coef", "output_router_logits",
}


def _rotary_dim(head_dim: int, config: Dict[str, Any], default_factor: float = 1.0) -> int:
    params = config.get("rope_parameters") or {}
    factor = params.get("partial_rotary_factor",
                        config.get("partial_rotary_factor", default_factor))
    r = int(head_dim * factor)
    if r <= 0 or r > head_dim or r % 2:
        raise ValueError(f"rotary_dim 非法: {r} (head_dim={head_dim}, factor={factor})")
    return r


def _layer_schedule(config: Dict[str, Any], default_interval: Optional[int]) -> List[str]:
    n = config["num_hidden_layers"]
    types = config.get("layer_types")
    if types is not None:
        if len(types) != n or any(t not in (FULL, LINEAR) for t in types):
            raise ValueError("layer_types 必须逐层给出 full/linear attention")
        return list(types)
    interval = config.get("full_attention_interval", default_interval)
    if interval is None:
        return [FULL] * n
    if not isinstance(interval, int) or interval < 1:
        raise ValueError("full_attention_interval 必须为正整数")
    return [FULL if (i + 1) % interval == 0 else LINEAR for i in range(n)]


def _moe_of(config: Dict[str, Any], shared_inter_key: str = "shared_expert_intermediate_size",
            v3_style: bool = False) -> Optional[MoeIR]:
    if v3_style:
        experts = config.get("n_routed_experts")
    else:
        experts = config.get("num_experts")
    if experts is None:
        return None
    shared = config.get(shared_inter_key)
    if v3_style and shared is None:
        shared = (config.get("n_shared_experts") or 0) * config["moe_intermediate_size"]
    shared_inter = int(shared or 0)
    # HF/vLLM Qwen3.5/3.6、Qwen3-Next：只要 shared expert 存在就无条件构造
    # shared_expert_gate = Linear(h,1)。config 中没有 shared_expert_gate 键，
    # 不能用缺失键推断 False（2026-11-10 审计 F5）。
    # DeepSeek-V3 的 shared expert 没有 sigmoid gate，保持 False。
    if v3_style:
        shared_gate = False
    elif "shared_expert_gate" in config:
        shared_gate = bool(config["shared_expert_gate"])
    else:
        shared_gate = shared_inter > 0
    return MoeIR(
        experts=experts,
        top_k=config["num_experts_per_tok"],
        intermediate=config["moe_intermediate_size"],
        shared_intermediate=shared_inter,
        shared_gate=shared_gate,
        routed_scaling=(float(config["routed_scaling_factor"])
                        if config.get("routed_scaling_factor") is not None else None),
        scoring=str(config.get("scoring_func", "softmax")),
    )


def flatten_config(config: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """把完整 config（含 text_config/vision_config/quantization_config 等嵌套）
    展平成 path → leaf 值；容器本身也保留（便于登记容器级状态）。"""
    out: Dict[str, Any] = {}
    for key, value in config.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out[path] = value
            out.update(flatten_config(value, path))
        else:
            out[path] = value
    return out


_CONTAINER_META = {"text_config", "vision_config", "rope_parameters", "rope_scaling",
                   "auto_map"}
_CONTAINER_MAPPED = {"dflash_config"}
_CONTAINER_UNSUPPORTED = {"quantization_config"}
_OUTER_META = set(_META_KEYS) | {
    "architectures", "model_type", "transformers_version", "image_token_id",
    "video_token_id", "vision_start_token_id", "vision_end_token_id",
}


def _coverage(family: str, config: Dict[str, Any], gaps_hint: Dict[str, str] = None,
              meta_note: str = "", full_config: Dict[str, Any] = None,
              text_prefix: str = "") -> Tuple[Dict[str, Dict[str, str]], List[str]]:
    """完整 config 逐字段/嵌套字段覆盖登记（2026-11-10 修复 F7）。

    * 叶键必须显式登记；**未知键 → unsupported-semantic + gap，不默认 ignored**；
    * 视觉塔不在本仓文本推理 scope → metadata-only（写明领域，不是“无影响”）；
    * 只有已核不影响计数且原因明确的键才允许 ignored-with-reason。
    """
    covered: Dict[str, Dict[str, str]] = {}
    gaps: List[str] = []
    struct = _STRUCT_KEYS[family]
    full = full_config if full_config is not None else config
    hints = gaps_hint or {}
    flat = flatten_config(full)
    for path in sorted(flat):
        if isinstance(flat[path], dict):
            # 容器级登记（叶键另行登记）
            if path in _CONTAINER_MAPPED:
                covered[path] = {"status": "mapped", "consumer": "model_ir",
                                 "note": "容器；关键叶字段进入结构"}
            elif path in _CONTAINER_UNSUPPORTED:
                covered[path] = {"status": "unsupported-semantic", "consumer": "-",
                                 "note": "量化配置未映射到本仓计算精度；场景精度为独立输入"}
                gaps.append(f"{path}: 量化配置未映射（场景精度独立输入，不冒充已消费）")
            else:
                covered[path] = {"status": "metadata-only", "consumer": "identity",
                                 "note": "容器；叶字段逐项登记"}
            continue
        under_text = bool(text_prefix and path.startswith(text_prefix + "."))
        suffix = path[len(text_prefix) + 1:] if under_text else path
        # 视觉塔：文本推理 scope 外，显式登记为 metadata-only
        if path.startswith("vision_config."):
            covered[path] = {"status": "metadata-only", "consumer": "scope:text-only",
                             "note": "视觉塔不在本仓文本推理 scope；未执行，不作为结构完备证据"}
            continue
        if path.startswith("auto_map."):
            covered[path] = {"status": "metadata-only", "consumer": "identity",
                             "note": "checkpoint auto_map 指向；实际实现另 pin"}
            continue
        if path.startswith("quantization_config."):
            covered[path] = {"status": "unsupported-semantic", "consumer": "-",
                             "note": "量化配置未映射到本仓计算精度；场景精度为独立输入"}
            continue
        if suffix.startswith("dspark_"):
            covered[path] = {"status": "metadata-only", "consumer": "methods",
                             "note": "方法层 DSpark 参数；不是模型结构已消费"}
            continue
        if suffix in ("compress_rope_theta",):
            covered[path] = {"status": "metadata-only", "consumer": "identity",
                             "note": "压缩 RoPE 角度参数；不改变本仓计数，未执行"}
            continue
        if suffix.startswith("rope_scaling."):
            covered[path] = {"status": "metadata-only", "consumer": "identity",
                             "note": "RoPE 缩放影响角度/位置；不改变本仓计数，未执行"}
            continue
        if suffix.startswith("rope_parameters."):
            leaf = suffix.split(".", 1)[1]
            if leaf == "partial_rotary_factor":
                covered[path] = {"status": "mapped", "consumer": "model_ir",
                                 "note": "部分旋转维度进入 AttentionIR.rotary_dim"}
            elif leaf in ("rope_theta", "rope_type"):
                covered[path] = {"status": "metadata-only", "consumer": "identity",
                                 "note": "角度参数；不改变本仓计数，未执行"}
            elif leaf in ("mrope_section", "mrope_interleaved"):
                covered[path] = {"status": "metadata-only", "consumer": "identity",
                                 "note": "多模位置 id 选择语义；文本算量不变，未执行"}
            else:
                covered[path] = {"status": "unsupported-semantic", "consumer": "-",
                                 "note": "未登记的 rope_parameters 叶键（不默认忽略）"}
                gaps.append(f"{path}: 未登记的 rope 参数（不默认忽略）")
            continue
        if path.startswith("dflash_config."):
            covered[path] = {"status": "mapped", "consumer": "model_ir",
                             "note": "DFlash 块协议绑定参数"}
            continue
        if path in hints:
            covered[path] = {"status": "unsupported-semantic", "consumer": "-",
                             "note": hints[path]}
            gaps.append(f"{path}: {hints[path]}")
            continue
        if suffix in hints:
            covered[path] = {"status": "unsupported-semantic", "consumer": "-",
                             "note": hints[suffix]}
            gaps.append(f"{suffix}: {hints[suffix]}")
            continue
        if suffix in struct:
            covered[path] = {"status": "mapped", "consumer": "model_ir",
                             "note": "进入层图/投影几何"}
        elif suffix in _TRAIN_ONLY_KEYS:
            covered[path] = {"status": "ignored-with-reason", "consumer": "-",
                             "note": "仅训练期字段，推理结构不受影响"}
        elif suffix in _META_KEYS or path in _OUTER_META:
            covered[path] = {"status": "metadata-only", "consumer": "identity",
                             "note": meta_note or "登记身份/兼容性，不进结构计数"}
        else:
            covered[path] = {"status": "unsupported-semantic", "consumer": "-",
                             "note": "未登记键：不默认忽略；需映射或显式缺口"}
            gaps.append(f"{path}: 未登记 config 键（不默认忽略）")
    return covered, gaps


# ---------------------------------------------------------------------------
# 家族解析器
# ---------------------------------------------------------------------------


def parse_gqa(config: Dict[str, Any], source: SourceIdentity, model_id: str,
              label: str, full_config: Dict[str, Any] = None,
              text_prefix: str = "") -> ModelIR:
    """tiny 手算样例/通用 GQA dense（全部层 full attention + dense FFN）。"""
    h, v = config["hidden_size"], config["vocab_size"]
    heads, kv = config["num_attention_heads"], config["num_key_value_heads"]
    d = config.get("head_dim") or h // heads
    if heads % kv:
        raise ValueError("Q 头必须整除 KV 头（GQA）")
    attn = AttentionIR(heads=heads, kv_heads=kv, head_dim=d,
                       rotary_dim=_rotary_dim(d, config))
    layers = [LayerIR(i, FULL, attn=attn, moe=None,
                      dense_inter=config.get("intermediate_size"))
              for i in range(config["num_hidden_layers"])]
    coverage, gaps = _coverage("gqa", config, full_config=full_config,
                                text_prefix=text_prefix)
    return ModelIR(id=model_id, label=label, family="gqa", hidden=h, vocab=v,
                   layers=layers, source=source,
                   max_position=config.get("max_position_embeddings", 0),
                   tie_embeddings=bool(config.get("tie_word_embeddings", False)),
                   coverage=coverage, gaps=gaps)


def parse_qwen3_moe(config: Dict[str, Any], source: SourceIdentity, model_id: str,
                    label: str, full_config: Dict[str, Any] = None,
                    text_prefix: str = "") -> ModelIR:
    """Qwen3 MoE（Qwen3-30B-A3B 等）：全层 GQA full attention + 全层 MoE。"""
    h, v = config["hidden_size"], config["vocab_size"]
    heads, kv = config["num_attention_heads"], config["num_key_value_heads"]
    d = config.get("head_dim") or h // heads
    attn = AttentionIR(heads=heads, kv_heads=kv, head_dim=d,
                       rotary_dim=_rotary_dim(d, config))
    moe = _moe_of(config)
    if moe is None:
        raise ValueError("qwen3_moe 家族必须有 num_experts")
    step = config.get("decoder_sparse_step", 1)
    dense_only = set(config.get("mlp_only_layers") or ())
    layers = []
    for i in range(config["num_hidden_layers"]):
        use_moe = (i + 1) % step == 0 and i not in dense_only
        layers.append(LayerIR(i, FULL, attn=attn, moe=moe if use_moe else None,
                              dense_inter=None if use_moe else
                              config.get("intermediate_size", moe.intermediate)))
    coverage, gaps = _coverage("qwen3_moe", config, full_config=full_config,
                                text_prefix=text_prefix)
    return ModelIR(id=model_id, label=label, family="qwen3_moe", hidden=h, vocab=v,
                   layers=layers, source=source,
                   max_position=config.get("max_position_embeddings", 0),
                   tie_embeddings=bool(config.get("tie_word_embeddings", False)),
                   coverage=coverage, gaps=gaps,
                   extras={"shared_expert_gate": moe.shared_gate})


def _parse_hybrid(config: Dict[str, Any], source: SourceIdentity, model_id: str,
                  label: str, family: str, default_interval: Optional[int],
                  full_config: Dict[str, Any] = None,
                  text_prefix: str = "") -> ModelIR:
    """qwen3_next / qwen3_5_moe 共用：GDN linear + GQA full 混合层序 + MoE。"""
    h, v = config["hidden_size"], config["vocab_size"]
    heads, kv = config["num_attention_heads"], config["num_key_value_heads"]
    d = config.get("head_dim") or h // heads
    gated = bool(config.get("attn_output_gate", family == "qwen3_5_moe"))
    attn = AttentionIR(heads=heads, kv_heads=kv, head_dim=d,
                       rotary_dim=_rotary_dim(d, config, 0.25 if family else 1.0),
                       output_gated=gated)
    linear = LinearAttnIR(
        num_k_heads=config["linear_num_key_heads"],
        num_v_heads=config["linear_num_value_heads"],
        head_k_dim=config["linear_key_head_dim"],
        head_v_dim=config["linear_value_head_dim"],
        conv_kernel=config["linear_conv_kernel_dim"],
    )
    moe = _moe_of(config) or MoeIR(experts=0, top_k=0, intermediate=0)
    step = config.get("decoder_sparse_step", 1)
    dense_only = set(config.get("mlp_only_layers") or ())
    kinds = _layer_schedule(config, default_interval)
    layers = []
    for i, kind in enumerate(kinds):
        use_moe = (i + 1) % step == 0 and i not in dense_only
        layers.append(LayerIR(
            i, kind,
            attn=attn if kind in (FULL, SLIDING) else None,
            linear=linear if kind == LINEAR else None,
            moe=moe if use_moe else None,
            dense_inter=None if use_moe else moe.intermediate))
    coverage, gaps = _coverage(family, config, full_config=full_config,
                                text_prefix=text_prefix)
    return ModelIR(id=model_id, label=label, family=family, hidden=h, vocab=v,
                   layers=layers, source=source,
                   max_position=config.get("max_position_embeddings", 0),
                   tie_embeddings=bool(config.get("tie_word_embeddings", False)),
                   dtype_hint=str(config.get("dtype", "")),
                   coverage=coverage, gaps=gaps,
                   mtp_layers=int(config.get("mtp_num_hidden_layers", 0) or 0),
                   extras={
                       "attn_output_gate": gated,
                       "mtp_use_dedicated_embeddings":
                           bool(config.get("mtp_use_dedicated_embeddings", False)),
                       "mamba_ssm_dtype": config.get("mamba_ssm_dtype", ""),
                   })


def parse_qwen3_5_moe(config: Dict[str, Any], source: SourceIdentity, model_id: str,
                      label: str, full_config: Dict[str, Any] = None,
                      text_prefix: str = "") -> ModelIR:
    """qwen3_5_moe 文本配置（Qwen3.5/Qwen3.6 家族；hybrid GDN + full + MoE）。

    多模态封装（vision_config/image_token_id 等）不在文本 config 内，由外层
    parse_multimodal_text 登记 metadata-only。
    """
    return _parse_hybrid(config, source, model_id, label, "qwen3_5_moe", 4,
                         full_config=full_config, text_prefix=text_prefix)


def parse_qwen3_next(config: Dict[str, Any], source: SourceIdentity, model_id: str,
                     label: str, full_config: Dict[str, Any] = None,
                     text_prefix: str = "") -> ModelIR:
    return _parse_hybrid(config, source, model_id, label, "qwen3_next", 4,
                         full_config=full_config, text_prefix=text_prefix)


def parse_deepseek_v3(config: Dict[str, Any], source: SourceIdentity, model_id: str,
                      label: str, full_config: Dict[str, Any] = None,
                      text_prefix: str = "") -> ModelIR:
    """DeepSeek-V3：全层 MLA + first_k_dense_replace/moe_layer_freq 分层。

    归档修复（2026-11-10 审查 F3）：dense 层 FFN 宽度取 intermediate_size
    （18432），不是 moe_intermediate_size（2048）；模型已移出活跃 catalog。
    """
    h, v = config["hidden_size"], config["vocab_size"]
    mla = MlaIR(heads=config["num_attention_heads"],
                q_lora_rank=config["q_lora_rank"],
                kv_lora_rank=config["kv_lora_rank"],
                qk_rope_dim=config["qk_rope_head_dim"],
                qk_nope_dim=config.get("qk_nope_head_dim",
                                       config["kv_lora_rank"]),
                v_head_dim=config["v_head_dim"])
    n = config["num_hidden_layers"]
    first = config.get("first_k_dense_replace", 0)
    freq = config.get("moe_layer_freq", 1)
    moe = MoeIR(experts=config["n_routed_experts"],
                top_k=config["num_experts_per_tok"],
                intermediate=config["moe_intermediate_size"],
                shared_intermediate=(config.get("n_shared_experts") or 0)
                * config["moe_intermediate_size"],
                routed_scaling=(float(config["routed_scaling_factor"])
                                if config.get("routed_scaling_factor") is not None else None),
                scoring=str(config.get("scoring_func", "softmax")))
    layers = []
    for i in range(n):
        is_moe = i >= first and i % freq == 0
        layers.append(LayerIR(i, MLA, mla=mla, moe=moe if is_moe else None,
                              dense_inter=None if is_moe else
                              config.get("intermediate_size", moe.intermediate)))
    coverage, gaps = _coverage("deepseek_v3", config, full_config=full_config,
                                text_prefix=text_prefix)
    return ModelIR(id=model_id, label=label, family="deepseek_v3", hidden=h, vocab=v,
                   layers=layers, source=source,
                   max_position=config.get("max_position_embeddings", 0),
                   tie_embeddings=bool(config.get("tie_word_embeddings", False)),
                   coverage=coverage, gaps=gaps,
                   mtp_layers=int(config.get("num_nextn_predict_layers", 0) or 0),
                   extras={"kv_spec": {
                       "family": "mla_v3",
                       "kv_lora_rank": mla.kv_lora_rank,
                       "qk_rope_dim": mla.qk_rope_dim,
                       "kv_elems_per_token_per_layer": mla.kv_lora_rank + mla.qk_rope_dim,
                   }})


def parse_v41_csa2(config: Dict[str, Any], source: SourceIdentity, model_id: str,
                   label: str, full_config: Dict[str, Any] = None,
                   text_prefix: str = "") -> ModelIR:
    """DeepSeek-V4.1-Flash：MLA+CSA2 压缩层计划 + 全层 MoE + mHC/Engram 建模。

    mHC（hc_mult/hc_eps/hc_sinkhorn_iters）：2026-11-10 按 vLLM pin 源码建模
    （count/work.hc_boundary_rows + lowering/memory hc 权重）→ mapped。
    Engram 8 键：2026-11-11 R04 按 common/engram.py pin（vllm @f358f057，
    blob sha1 ba4ceea6 对 vllm-tree.json 校验）source-checked 建模 → mapped
    （count/work.engram_rows + lowering/memory engram 表/wkv/qk 对象）；
    表布局 FP8+E8M0 264B/行、素数桶、TP head 切分、wkv/qk 投影与门控注入
    语义见 extras["engram"]。
    CSA2 索引打分/top-k/candidate block：2026-11-12 R20 按 21 文件 pin
    （csa2-pin.json，vllm @f358f057，blob sha1 对 vllm-tree.json 逐项校验）
    source-checked 建模 → mapped（count/work.v41_layer_rows + lowering/
    memory 记录/环形状态/缓冲对象；公式 v2.csa2_*；短上下文旁路、
    prefill/decode 分支、candidate 所有权见 source_map sm-v41-csa2-indexer）。
    MoE 路由（scoring/top-k/归一化/缩放/共享专家/EP 归属）：2026-11-12 R21
    按 14 文件 pin（moe-pin.json，vllm @f358f057，blob sha1 对
    vllm-tree.json 逐项校验）source-checked 建模 → mapped
    （count/moe.v41_moe_route_rows 三行 + lowering/memory 路由偏置对象；
    V4.1 EP 保守整除域（mega 断言；fused 通用余数映射未入 pin）；公式
    v2.moe_route_*；平坦偏置 top-k 无分组——V4.1 config 无
    n_group/topk_group 键，分组仅 V3 式 use_grouped_topk 路径）。
    量化映射（R22，2026-11-13）：quantization_config 5 叶键按 20 文件 pin
    （quant-pin.json 20 文件，vllm @f358f057，blob sha1 对 vllm-tree.json 逐项校验）
    source-checked 映射 → mapped（extras["quant"]；dense/shared 线性 MXFP8
    1B+ue8m0 (1,32) scale、路由专家 MXFP4 打包 0.5B+scale、embed/lm_head/
    norm/gate BF16 未量化；Marlin SM90 执行域 act=BF16）。执行变体
    （Humming/TRTLLM SM100/emulation 反量化）登记 capability-gaps，不进
    本 gaps。
    swiglu clamp、注意力核本体、尾部 compress_ratios 仍为显式
    缺口 → gaps 非空，structure_complete=False。
    dspark_* 为方法层字段（methods 消费），模型侧 metadata-only。
    """
    tc = config
    h, v = tc["hidden_size"], tc["vocab_size"]
    mla = MlaIR(heads=tc["num_attention_heads"], q_lora_rank=tc["q_lora_rank"],
                kv_lora_rank=0, qk_rope_dim=tc["qk_rope_head_dim"],
                qk_nope_dim=tc["head_dim"], v_head_dim=tc["head_dim"])
    L = tc["num_hidden_layers"]
    ratios = tc["compress_ratios"]
    kv_ids, ix_ids = tc["kv_source_layer_ids"], tc["index_source_layer_ids"]
    candidate = tc.get("candidate_source_layer_id", -1)
    if len(ratios) < L or any(r not in (0, 1, 2) for r in ratios[:L]):
        raise ValueError("compress_ratios 必须覆盖所有主干层且仅含 0/1/2")
    if kv_ids != sorted(set(kv_ids)) or ix_ids != sorted(set(ix_ids)):
        raise ValueError("缓存源层必须递增不重复")
    if not set(kv_ids).issubset(ix_ids) or any(ratios[i] == 0 for i in set(kv_ids) | set(ix_ids)):
        raise ValueError("压缩 KV 源层必须同时产生索引 K，源层不能是纯 SWA")
    # R20：candidate 两级过滤的 config 域不变式（pinned attention.py
    # L399-400/L531-553：candidate buffer 只发给 candidate source 与其后的
    # index source；候选 block id 是源层压缩位置空间的块号，消费层与源层
    # 压缩比不同则坐标失配——vLLM 未显式断言，按 config 域拒绝）。
    ctb = int(tc.get("candidate_topk_blocks", 0) or 0)
    cb = int(tc.get("candidate_block_size", 0) or 0)
    if candidate >= 0:
        if candidate not in ix_ids:
            raise ValueError("candidate_source_layer_id 必须是 index source 层")
        if ctb <= 0 or cb <= 0:
            raise ValueError("candidate_source_layer_id >= 0 要求 "
                             "candidate_topk_blocks/candidate_block_size > 0")
        for i in ix_ids:
            if candidate < i and ratios[i] != ratios[candidate]:
                raise ValueError(
                    f"candidate 消费层 {i} 压缩比 {ratios[i]} 与源层 "
                    f"{ratios[candidate]} 不同：候选块坐标空间失配")
    moe = MoeIR(experts=tc["n_routed_experts"], top_k=tc["num_experts_per_tok"],
                intermediate=tc["moe_intermediate_size"],
                shared_intermediate=(tc.get("n_shared_experts") or 0)
                * tc["moe_intermediate_size"],
                routed_scaling=float(tc["routed_scaling_factor"]),
                scoring=str(tc.get("scoring_func", "softmax")))
    # R21：V4.1 路由 config 域校验（pinned 源码只支持该组合；非法值按
    # config 域拒绝，不静默回退）——dsv4_topk.py can_use L21-38（renormalize/
    # fp32 logits/bias）、deepseek_v4/nvidia/model.py L253-263（mega 路径
    # 断言 sqrtsoftplus+renormalize）、cpu_router.py L107（assert renormalize）。
    if moe.scoring != "sqrtsoftplus":
        raise ValueError(f"V4.1 scoring_func={moe.scoring!r} 不支持：pinned "
                         "实现仅 sqrtsoftplus（mega 断言 L255-258）")
    if str(tc.get("topk_method", "noaux_tc")) != "noaux_tc":
        raise ValueError(f"V4.1 topk_method={tc.get('topk_method')!r} 不支持："
                         "noaux_tc 之外无 e_score_correction_bias 建模依据")
    if not bool(tc.get("norm_topk_prob", True)):
        raise ValueError("V4.1 norm_topk_prob=false 不支持：dsv4/cpu 路由均"
                         "断言 renormalize")
    # R22：量化 config 域校验（pinned dispatch 只覆盖 official V4.1-Flash
    # 的组合；偏离值按 config 域拒绝，不静默回退到别的量化路径）——
    # quant-pin.json：deepseek_v41/quant_config.py L163-192（override 只认
    # quant_method ∈ fp8/deepseek_v4_fp8 + deepseek_v4* 模型型）、L76-80
    # （expert_dtype 域 fp4|fp8）、L221-243（线性 MXFP8 分支要求
    # weight_block_size==[32,32] 且 is_scale_e8m0）、fp8.py L111-131
    # （块量化必须 dynamic activation）。
    if full_config is None:
        quant_cfg = dict(_V41_DEFAULT_QUANT_CFG)
    else:
        raw_qcfg = full_config.get("quantization_config")
        if not isinstance(raw_qcfg, dict):
            raise ValueError(
                "V4.1 quantization_config 缺失/非对象：official 资产固定携带"
                " 5 叶键（fp8 动态块量化 + fp4 专家）；无量化变体不在"
                " source-checked 域")
        quant_cfg = raw_qcfg
    qm = str(quant_cfg.get("quant_method", ""))
    if qm not in ("fp8", "deepseek_v4_fp8"):
        raise ValueError(
            f"V4.1 quantization_config.quant_method={qm!r} 不在 pinned dispatch"
            " 域（override 只认 fp8/deepseek_v4_fp8 → DeepseekV41FP8Config；"
            "quant_config.py L175-192）")
    if quant_cfg.get("activation_scheme") != "dynamic":
        raise ValueError(
            f"V4.1 activation_scheme={quant_cfg.get('activation_scheme')!r} "
            "不支持：块量化要求 dynamic（fp8.py L125-130）；static 激活"
            "路径未建模")
    if list(quant_cfg.get("weight_block_size") or []) != [32, 32]:
        raise ValueError(
            f"V4.1 weight_block_size={quant_cfg.get('weight_block_size')!r} "
            "不在域：MXFP8 线性 dispatch 要求精确 [32,32]（quant_config.py "
            "L222-243）；[128,128] FP8 块路径为 V3 式未建模变体")
    if quant_cfg.get("expert_dtype") != "fp4":
        raise ValueError(
            f"V4.1 expert_dtype={quant_cfg.get('expert_dtype')!r} 不在已建模"
            "域：fp4→Mxfp4MoEMethod 已入账；fp8（Flash-Base 块量化专家）"
            "路径未建模（quant_config.py L244-263）")
    if quant_cfg.get("scale_fmt") != "ue8m0":
        # 注意：loader 不读 scale_fmt（Fp8Config.from_config 键表无该键，
        # fp8.py L155-172）；有效 scale dtype 由 expert_dtype 派生
        # （is_scale_e8m0，quant_config.py L89-93）。值不一致 = checkpoint
        # 声明与派生结果矛盾，拒绝而非静默取派生值。
        raise ValueError(
            f"V4.1 scale_fmt={quant_cfg.get('scale_fmt')!r} 不在域：official"
            " 资产 ue8m0；该键不被 loader 消费（fp8.py L155-172），实际"
            " scale dtype 由 expert_dtype=='fp4' 派生，不一致按声明矛盾拒绝")
    layers = []
    r_hist: Dict[int, int] = {}
    for i in range(L):
        r = ratios[i]
        if r:
            k = max(x for x in kv_ids if x <= i)
            x = max(x for x in ix_ids if x <= i)
            if ratios[k] != r or ratios[x] != r:
                raise ValueError(f"第 {i} 层 KV/索引来源压缩比例不一致")
        else:
            k = x = None
        layers.append(LayerIR(
            i, MLA_CSA2, mla=mla,
            csa2=Csa2IR(ratio=r, kv_source=k, index_source=x,
                        produces_kv=i in kv_ids, produces_index=i in ix_ids,
                        produces_candidates=i == candidate),
            moe=moe))
    gaps_hint = {
        "swiglu_limit": "SwiGLU clamp（10.0）未进入本仓 MLP 计数/数值语义；"
                        "不冒充已消费",
    }
    # hc_* / engram_* 8 键 → mapped（hc：count/work.hc_boundary_rows +
    # lowering/memory hc 权重，vllm model.py pin @f358f057 source-checked
    # 2026-11-10；engram：count/work.engram_rows + lowering/memory engram
    # 对象，common/engram.py pin blob ba4ceea6 source-checked 2026-11-11）
    coverage, gaps = _coverage("v41_csa2", tc, gaps_hint,
                                full_config=full_config, text_prefix=text_prefix)
    # Engram 8 键的 mapped 注记逐键替换为行号证据（默认 struct 注记过于笼统）
    for key, note in _ENGRAM_KEY_NOTES.items():
        path = f"{text_prefix}.{key}" if text_prefix else key
        if path in coverage and coverage[path]["status"] == "mapped":
            coverage[path] = {"status": "mapped", "consumer": "model_ir+count+memory",
                              "note": note}
    # R20：CSA2 索引/candidate 键的 mapped 注记（同上替换；21 文件 pin
    # csa2-pin.json，source-checked 2026-11-12）
    for key, note in _CSA2_KEY_NOTES.items():
        path = f"{text_prefix}.{key}" if text_prefix else key
        if path in coverage and coverage[path]["status"] == "mapped":
            coverage[path] = {"status": "mapped", "consumer": "model_ir+count+memory",
                              "note": note}
    if len(ratios) > L:
        gaps.append(
            f"compress_ratios[{L}:{len(ratios)}]: 尾部 {len(ratios) - L} 项未进入"
            "主干层图（MTP/草稿层作用未建模；不能当作已消费）")
    # mHC 参数（消费方：count/work.hc_boundary_rows、lowering/memory）。
    # post_alpha=2.0 为源码常数（model.py self.hc_post_alpha），入册便于对账。
    hc_spec = {
        "mult": int(tc["hc_mult"]),
        "sinkhorn_iters": int(tc["hc_sinkhorn_iters"]),
        "eps": float(tc["hc_eps"]),
        "post_alpha": 2.0,
        "engram_layer_ids": list(tc.get("engram_layer_ids", [])),
    }
    # Engram 参数（消费方：count/work.engram_rows、lowering/memory engram 对象；
    # 常数全部来自 common/engram.py pin @f358f057 blob ba4ceea6）：
    #   * 表值 FP8 e4m3 [rows, head_dim] + ue8m0 scale [rows, head_dim/32]
    #     （L1151-1161 分配、L726-731 dequant：scale byte<<23 即 fp32 指数）；
    #   * block_size=32（L1037 默认）→ 264 B/行（L968-969 can_share 同式）；
    #   * 哈希源 = 第一个本地层 SWA cache lookback(3)（L444/L919-948）。
    e_heads, e_dim = int(tc["engram_n_heads"]), int(tc["engram_head_dim"])
    e_max = int(tc["engram_max_ngram_size"])
    engram_spec = {
        "layer_ids": list(tc.get("engram_layer_ids", [])),
        "num_embeddings": list(tc.get("engram_num_embeddings", [])),
        "max_ngram_size": e_max,
        "vocab_size": int(tc["engram_vocab_size"]),
        "n_heads": e_heads,
        "head_dim": e_dim,
        "pad_token_id": int(tc["engram_pad_token_id"]),
        "compressed_vocab_size": int(tc["engram_compressed_vocab_size"]),
        "n_hash_cols": (e_max - 1) * e_heads,       # 24 = 3 组 × 8 heads（L243）
        "block_size": 32,                            # L1037 默认
        "value_dtype": "fp8_e4m3",                   # L1157-1160
        "scale_encoding": "ue8m0",                   # L726-727
        "bytes_per_row": e_dim + e_dim // 32,        # 264（L968-969/L1139）
        # wkv 几何（Engram L1693-1733）：in = n_hash_cols·head_dim = 6144，
        # out = hidden·(hc_mult+1) = 5120×5 = 25600；hc_mult 取自 config
        # （Engram.__init__ L1694 self.hc_mult = config.hc_mult）
        "hc_mult": int(tc["hc_mult"]),
        "hash_source": "first_local_layer_swa_lookback(3)",
    }
    # dspark_* 归方法层消费（登记 metadata-only，消费者 methods）
    for key in list(coverage):
        if key.startswith("dspark_"):
            coverage[key] = {"status": "metadata-only", "consumer": "methods",
                             "note": "V4.1 原生 DSpark 方法参数（方法层绑定，非模型结构）"}
    # R21：V4.1 MoE 路由键的 mapped 注记（14 文件 pin moe-pin.json，
    # source-checked 2026-11-12）
    for key, note in _MOE_ROUTE_KEY_NOTES.items():
        path = f"{text_prefix}.{key}" if text_prefix else key
        if path in coverage and coverage[path]["status"] == "mapped":
            coverage[path] = {"status": "mapped",
                              "consumer": "model_ir+count+memory",
                              "note": note}
    # R22：quantization_config 容器+5 叶键 unsupported → mapped（20 文件 pin
    # quant-pin.json source-checked 2026-11-13）；对应 gap 条目移除。
    coverage["quantization_config"] = {
        "status": "mapped", "consumer": "model_ir+count+memory",
        "note": "容器；5 叶键逐项 mapped（R22 quant-pin.json 20 文件；dispatch → "
                "MXFP8 线性 + MXFP4 专家 + BF16 未量化类）"}
    for key, note in _V41_QUANT_KEY_NOTES.items():
        coverage[f"quantization_config.{key}"] = {
            "status": "mapped", "consumer": "model_ir+count+memory",
            "note": note}
    gaps = [g for g in gaps
            if not g.startswith("quantization_config")]
    # R21：路由 spec（消费方：count/moe.v41_moe_route_rows、
    # lowering/memory 路由偏置对象、count/work EP 整除域校验）。
    # 常数全部来自 moe-pin.json @f358f057：
    #   * gate = GateLinear(h→E, out_dtype=fp32) Replicated（L270-277）：
    #     权重沿 model loader BF16（linear.py L287-295/L179-199；
    #     base_loader.py L86）；输入由 mHC tilelang.py L900/L1096 的
    #     layer_input_cur BF16 证明（model.py L731-816）；均 2B；fp32
    #     只是 logits 出口；有效域 official V4.1 bf16，非 BF16 拒绝；
    #   * bias = e_score_correction_bias[E] fp32（noaux_tc，L303-313）；
    #   * bias_vl：vision_n_layers>0 时每 MoE 层创建（L314-320；V4.1-Flash
    #     视觉塔 32 层 → 存在，文本 token 也逐行做哨兵判定 dsv4 L63-76）；
    #   * V4.1 主干 num_hash_layers=0（v41/nvidia/model.py L155）——无
    #     hash MoE/tid2eid 表；
    #   * EP：mega 路径断言 n_physical(=E+eplb 冗余 0) % ep == 0（L357-362）；
    #     fused 路径 L419-421 的 start_idx 只赋值、未被所有权消费；fused
    #     通用余数映射未入 pin/未建模。本域只声明整除组合
    #     （count/moe.require_divisible_ep 保守规则），不主张 fused 非法。
    #   * 共享专家：FusedMoEFactory 调用未传 n_shared_experts/
    #     fuse_shared_experts（model.py L426-449）→ router 不追加共享槽，
    #     runner 独立相加（权重 1.0；moe_output.py L151-175）。
    vision_layers = int((full_config or {}).get("vision_config", {})
                        .get("num_hidden_layers", 0) or 0)
    # dtype 域只覆盖官方 V4.1 固定资产（dtype=bfloat16）：
    #   * gate 权重：GateLinear 未传 quant_config/params_dtype → LinearBase
    #     取 torch.get_default_dtype()，base_loader 在 model dtype 内构造；
    #     官方 config dtype=bfloat16 → BF16 2B；
    #   * gate 输入：pinned mHC tilelang.py 的 layer_input_cur 硬编码
    #     bfloat16（L900/L1096），model.py L731-816 直接交给 self.ffn(x)；
    #     不能把"输入"泛化成任意 model dtype。
    # full_config 显式给出非 BF16 dtype 时拒绝；无 full_config 的内部调用
    # 只对固定 V4.1 资产采用 BF16 默认（不是通用 dtype 回退）。
    if full_config is None:
        model_dtype = "bfloat16"
        model_dtype_source = "fixed_v41_asset_default"
    else:
        raw_dtype = full_config.get("dtype")
        model_dtype = str(raw_dtype).lower() if raw_dtype is not None else ""
        if model_dtype != "bfloat16":
            raise ValueError(
                f"V4.1 模型 dtype={raw_dtype!r} 不在 source-checked 域："
                "official config dtype=bfloat16；gate 输入由 pinned mHC "
                "layer_input_cur BF16 证明（tilelang.py L900/L1096），"
                "不能按 model dtype 泛化")
        model_dtype_source = "full_config_dtype"
    model_dtype_bytes = _V41_BF16_ITEMSIZE
    moe_route_spec = {
        "scoring": "sqrtsoftplus",
        "topk_method": "noaux_tc",
        "renormalize": True,
        "routed_scaling": float(tc["routed_scaling_factor"]),
        "gate_out_dtype": "fp32",
        "logits_itemsize": 4,
        # 权重沿 model loader BF16 路径；输入由 mHC layer_input BF16 证明。
        # 两者在官方资产上都 2B，但证据链不同，不能泛化成任意 model dtype。
        "model_dtype": model_dtype,
        "model_dtype_source": model_dtype_source,
        "dtype_domain": _V41_DTYPE_DOMAIN,
        "gate_weight_dtype": "bfloat16",
        "gate_weight_itemsize": model_dtype_bytes,
        "gate_act_dtype": "bfloat16",
        "gate_act_itemsize": model_dtype_bytes,
        "gate_act_source": "mhc_layer_input_cur_bf16",
        "bias_bytes": moe.experts * 4,
        "bias_vl_bytes": moe.experts * 4 if vision_layers > 0 else 0,
        "hash_layers": 0,
        "grouped": False,
        # 独立相加（非 router 融合槽）：FusedMoEFactory 调用未传
        # n_shared_experts/fuse_shared_experts；见 moe_output.shared_output。
        "shared_expert_weight": 1.0,
    }
    # R22：量化 spec（消费方：count/work.v41_layer_rows、count/moe.moe_rows、
    # count/work.head_rows、lowering/memory.target_weight_allocations、
    # count/quant_layout 字节公式、scenario V41 量化一致性校验）。
    # 证据 quant-pin.json 20 文件 @f358f057，逐条见 _V41_QUANT_KEY_NOTES
    # 与 docs/validation/v41-quant-map-20261113.md；执行域按注册设备
    # （H100/H200=SM90、L40=SM89，均 ≤SM90）固定为 Marlin W8A16/W4A16。
    quant_spec = {
        "method": "deepseek_v4_fp8_dispatch",
        "method_source": "override_quantization_method（quant_config.py "
                         "L163-192；config/model.py L1429/L1444-1465）",
        "linear_class": "ModelOptLinearMethod(kMxfp8Static, kMxfp8Dynamic)",
        "linear_source": "quant_config.py L221-243；modelopt.py KMxfp8Static "
                         "L2424-2479",
        "dense_linear": {"value_dtype": "fp8_e4m3", "value_itemsize": 1,
                         "scale_dtype": "ue8m0", "scale_itemsize": 1,
                         "ckpt_scale_block": [32, 32],
                         "runtime_scale_block": [1, 32]},
        # 未量化类：get_quant_method 对非 LinearBase/RoutedExperts/Attention
        # 返回 None → 默认方法（fp8.py L174-203；embedding 落
        # UnquantizedEmbeddingMethod，vocab_parallel_embedding.py L322-326）
        "unquantized_itemsize": model_dtype_bytes,
        "unquantized_classes": ["embed", "lm_head", "rmsnorm", "gate_linear",
                                "compressor_fused_wkv_wgate", "indexer_wk",
                                "indexer_weights_proj", "mhc", "engram_table"],
        "routed_experts": {"value_dtype": "fp4_e2m1_packed",
                           "value_itemsize": 0.5, "scale_dtype": "ue8m0",
                           "scale_itemsize": 1, "scale_block": 32,
                           "activation_key": "kMxfp8Dynamic",
                           "source": "quant_config.py L244-261 → "
                                     "Mxfp4MoEMethod（mxfp4.py L648-703/"
                                     "L765-840）"},
        # 执行域（checkpoint 格式之外）：注册设备全部 ≤SM90 →
        #   * 线性：FlashInfer MXFP8 kernel 均要求 SM100+（flashinfer.py
        #     L47-53/L126-139）→ Marlin MXFP8 W8A16（SM80+，marlin.py
        #     L14-26；激活保持 BF16，无运行时激活量化）；
        #   * 专家：TRTLLM/DeepGEMM 为 SM100+（oracle/mxfp4.py L461-486
        #     docstring）→ Marlin W4A16（BF16 激活）；Humming 在 cc=90 优先
        #     于 Marlin（priority.py L10-23）但依赖可选外部 humming 包
        #     （utils/humming.py L18-97 lazy facade）→ 部署变体未建模；
        #   * 共享专家：与 dense 线性同 dispatch（DeepseekV4MLP 传
        #     quant_config，deepseek_v4/nvidia/model.py L322-335）。
        "act_itemsize": 2,
        "act_dtype": "bfloat16",
        "act_domain": "registered_sm80_sm90_marlin_w8a16_w4a16",
        "act_source": "kernels/linear/__init__.py L569-589/L962-1014 + "
                      "marlin.py L14；oracle/mxfp4.py L461-486",
        # Marlin 分配对齐（routed_experts.py L190-203 于 create_weights 前
        # 调 maybe_roundup_sizes；oracle/mxfp4.py L850-858）
        "expert_backend": "marlin_w4a16",
        "expert_round_up": {"intermediate": 128, "hidden": 256},
        "expert_round_up_source": "oracle/mxfp4.py L850-855（MARLIN 分支）",
        # checkpoint 声明（scale_fmt 不被 loader 读取；见键注记）
        "checkpoint_quantization_config": dict(quant_cfg),
        "kv_cache": "quantization_config 无缓存叶键；记录宽度由 R20 "
                    "v41_kv_record 配方决定（attention.py L583-589），"
                    "Fp8KVCacheMethod 仅在显式 fp8 kv_cache_dtype 时加载"
                    "checkpoint scale（fp8.py L201-203/L845-849）",
    }
    return ModelIR(
        id=model_id, label=label, family="v41_csa2", hidden=h, vocab=v,
        layers=layers, source=source,
        max_position=tc.get("max_position_embeddings", 0),
        tie_embeddings=bool(tc.get("tie_word_embeddings", False)),
        coverage=coverage, gaps=gaps,
        mtp_layers=int(tc.get("num_nextn_predict_layers", 0) or 0),
        extras={"kv_spec": {
            "family": "v41_csa2",
            "head_dim": tc["head_dim"],
            "q_lora_rank": tc["q_lora_rank"],
            "o_lora_rank": tc["o_lora_rank"],
            "o_groups": tc["o_groups"],
            "heads": tc["num_attention_heads"],
            "qk_rope_dim": tc["qk_rope_head_dim"],
            "index_head_dim": tc["index_head_dim"],
            "index_heads": tc["index_n_heads"],
            "index_topk": tc["index_topk"],
            "candidate_topk_blocks": tc["candidate_topk_blocks"],
            "candidate_block_size": int(tc.get("candidate_block_size", 0) or 0),
            "candidate_source_layer_id": int(candidate),
            "sliding_window": tc["sliding_window"],
            "compress_ratios": ratios,
            "kv_source_layer_ids": kv_ids,
            "index_source_layer_ids": ix_ids,
        },
        "hc": hc_spec,
        "engram": engram_spec,
        "moe_route": moe_route_spec,
        "quant": quant_spec})


def parse_dflash_draft(config: Dict[str, Any], source: SourceIdentity, model_id: str,
                       label: str, full_config: Dict[str, Any] = None,
                       text_prefix: str = "") -> Any:
    """DFlash 草稿 → DraftIR（结构 + 块扩散绑定；协议在 methods/dflash.py）。

    DraftIR 也走完整 config 覆盖门禁；协议/服务端集成证据在 evidence 中单列，
    不能因 config/结构 pin 就宣称已执行或已部署。
    """
    from .schema import DraftIR
    h, v = config["hidden_size"], config["vocab_size"]
    heads, kv = config["num_attention_heads"], config["num_key_value_heads"]
    d = config.get("head_dim") or h // heads
    swa = config.get("sliding_window")
    types = config.get("layer_types") or []
    n = config["num_hidden_layers"]
    fc = config.get("dflash_config", {})
    target_layer_ids = tuple(fc.get("target_layer_ids", ()))
    # 因果性来源优先级与 pinned 实现一致（HF dflash/model.py L364-366、
    # vLLM qwen3_dflash.py L86-97、SGLang models/dflash.py L239-249）：
    # 顶层 is_causal → dflash_config.causal → layer_types 派生
    # （sliding_attention 因果；full_attention 非因果）。不能把所有 full 层
    # 一律当因果窗口——Qwen3.6 DFlash 的 1 个 full 层是 ENCODER_ONLY。
    explicit_causal = config.get("is_causal")
    causal_override = fc.get("causal")
    layers = []
    for i in range(n):
        layer_type = types[i] if i < len(types) else "full_attention"
        kind = SLIDING if layer_type == "sliding_attention" else FULL
        if explicit_causal is not None:
            layer_causal = bool(explicit_causal)
        elif causal_override is not None:
            layer_causal = bool(causal_override)
        else:
            layer_causal = layer_type == "sliding_attention"
        layers.append(LayerIR(
            i, kind,
            attn=AttentionIR(heads=heads, kv_heads=kv, head_dim=d,
                             rotary_dim=_rotary_dim(d, config),
                             sliding_window=swa if kind == SLIDING else None,
                             causal=layer_causal),
            moe=None, dense_inter=config.get("intermediate_size")))
    coverage, gaps = _coverage("dflash", config, full_config=full_config,
                                text_prefix=text_prefix)
    # dtype 仍是元数据（不进层图/结构计数），但 k_norm stack 的容量要用它：
    # RMSNorm 参数在模型初始化 dtype 下创建（pinned layernorm/base_loader），
    # 与 draft_quant.dense_w 分离。这里显式登记消费方，避免“metadata-only”
    # 被误读成完全无人消费。
    for key in ("dtype", "torch_dtype"):
        if key in coverage:
            coverage[key] = {
                "status": "metadata-only", "consumer": "identity+memory",
                "note": "模型参数 dtype 线索；DFlash k_norm stack 容量使用"
                        "（不进结构计数，不冒充已执行/已部署）"}
    return DraftIR(
        id=model_id, label=label, layers=layers, hidden=h, vocab=v,
        uses_target_embeddings=True, uses_target_lm_head=True,
        target_layer_ids=target_layer_ids, fc_in=len(target_layer_ids) * h,
        block_size=int(fc.get("block_size", 16)),
        mask_token_id=fc.get("mask_token_id"), source=source,
        dtype_hint=str(config.get("dtype") or config.get("torch_dtype") or ""),
        coverage=coverage, gaps=gaps)
