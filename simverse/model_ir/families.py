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
               "tie_word_embeddings"},
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
    CSA2/MoE scoring、量化映射、swiglu clamp、candidate block、尾部
    compress_ratios 仍为显式缺口 → gaps 非空，structure_complete=False。
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
    moe = MoeIR(experts=tc["n_routed_experts"], top_k=tc["num_experts_per_tok"],
                intermediate=tc["moe_intermediate_size"],
                shared_intermediate=(tc.get("n_shared_experts") or 0)
                * tc["moe_intermediate_size"],
                routed_scaling=float(tc["routed_scaling_factor"]),
                scoring=str(tc.get("scoring_func", "softmax")))
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
        "candidate_block_size": "候选 block 缓冲粒度未进入 candidate_publish/"
                                "容量建模；不冒充已消费",
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
            "sliding_window": tc["sliding_window"],
            "compress_ratios": ratios,
            "kv_source_layer_ids": kv_ids,
            "index_source_layer_ids": ix_ids,
        },
        "hc": hc_spec,
        "engram": engram_spec})


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
    layers = []
    for i in range(n):
        kind = SLIDING if (i < len(types) and types[i] == "sliding_attention") else FULL
        layers.append(LayerIR(
            i, kind,
            attn=AttentionIR(heads=heads, kv_heads=kv, head_dim=d,
                             rotary_dim=_rotary_dim(d, config),
                             sliding_window=swa if kind == SLIDING else None),
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
