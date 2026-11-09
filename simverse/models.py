"""L0 结构层：模型/草稿注册表 + SpecMethod 语义表（方案 §4.2 的 v0.1 子集）。

两个模型适配器：
  tiny_dense_oracle   2 层 GQA dense 小模型 —— 手算 oracle（golden/expectations 独立期望）
  deepseek_v41_flash  首发（决策 D-3）：40 层 MLA+CSA2 + 全层 MoE，结构字段全部来自
                      官方 config.json 固定快照（assets/config/deepseek_v41_config.json，
                      SHA256 8be45ce0…，与 model-verse 仓 pinned 副本逐字节一致）。

计数公式为 attention_structure.deepseek_attention（旧仓已审阅实现）的忠实移植，
移植处逐一标注；本文件只产结构（ModelSpec），不产计数（ledger.py）与时间（cost.py）。

SpecMethod 注册表（v0.1）：
  mtp / dspark / custom_chain   链式草稿，三参数契约 (S=K, M=1(+catch-up), K)
  dflash / dflash2 / jetspecpp  显式 unsupported（决策 D-6：块并行与 JetSpec++ 属后续版本）
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .contracts import DraftSpec, LayerSpec, ModelSpec

ASSETS = Path(__file__).resolve().parent.parent / "assets"

# ---------------------------------------------------------------------------
# 模型适配器
# ---------------------------------------------------------------------------


def _tiny_dense() -> ModelSpec:
    """手算 oracle 模型：2 层 GQA dense。所有计数可用整数手算复核。

    结构：hidden 256, 8 Q 头 × head_dim 32, 2 KV 头（GQA）, FFN gate/up/down
    intermediate 512（SwiGLU 3 矩阵），vocab 1000，无 MoE。
    每 token 每层 KV = 2(K+V) × 2 头 × 32 维 = 128 元素。
    """
    h, heads, kv_heads, d, ffn = 256, 8, 2, 32, 512
    layers = []
    for i in range(2):
        layers.append(LayerSpec(
            index=i, kind="full_attention",
            attn={"family": "gqa", "heads": heads, "kv_heads": kv_heads,
                  "head_dim": d,
                  "projs": {"q_proj": [h, heads * d], "k_proj": [h, kv_heads * d],
                            "v_proj": [h, kv_heads * d], "out_proj": [heads * d, h]}},
            moe=None,
        ))
    return ModelSpec(
        id="tiny_dense_oracle", label="Tiny Dense Oracle (2L GQA, d=256, ffn=512)",
        hidden=h, layers=layers, vocab=1000,
        kv_spec={"family": "gqa", "kv_heads": kv_heads, "head_dim": d,
                 "kv_elems_per_token_per_layer": 2 * kv_heads * d},
        quant={"dense_w": 16, "expert_w": 16, "act": 16, "kv": 16},
        source_note="手算样例模型（非真实模型）：为 oracle 精确整数对账而设",
    )


def _v41_plan(tc: Dict[str, Any]) -> List[LayerSpec]:
    """移植 attention_structure.deepseek_plan（旧仓）的层计划解析，规则逐字保持。"""
    L = tc["num_hidden_layers"]
    ratios = tc["compress_ratios"]
    kv, ix = tc["kv_source_layer_ids"], tc["index_source_layer_ids"]
    candidate = tc.get("candidate_source_layer_id", -1)
    if len(ratios) < L or any(r not in (0, 1, 2) for r in ratios[:L]):
        raise ValueError("compress_ratios 必须覆盖所有主干层且仅包含 0/1/2")
    if kv != sorted(set(kv)) or ix != sorted(set(ix)):
        raise ValueError("缓存源层必须为递增且不重复")
    if not set(kv).issubset(ix) or any(ratios[i] == 0 for i in set(kv) | set(ix)):
        raise ValueError("每个压缩 KV 源层必须同时产生索引 K，源层不能是纯 SWA")
    layers = []
    for i in range(L):
        r = ratios[i]
        if r:
            k = max(x for x in kv if x <= i)
            x = max(x for x in ix if x <= i)
            if ratios[k] != r or ratios[x] != r:
                raise ValueError(f"第 {i} 层的 KV/索引来源压缩比例不一致")
        else:
            k = x = None
        layers.append(LayerSpec(
            index=i, kind="mla_csa2",
            attn={"family": "v41_mla_csa2", "ratio": r, "kv_source": k,
                  "index_source": x, "produces_kv": i in kv, "produces_index": i in ix,
                  "produces_candidates": i == candidate,
                  "candidate_source": candidate if (i in ix and 0 <= candidate < i) else None},
            moe={"experts": tc["n_routed_experts"], "top_k": tc["num_experts_per_tok"],
                 "inter": tc["moe_intermediate_size"], "shared_inter": tc["moe_intermediate_size"]},
        ))
    return layers


def _deepseek_v41_flash() -> ModelSpec:
    path = ASSETS / "config" / "deepseek_v41_config.json"
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    tc = json.loads(raw.decode("utf-8"))["text_config"]
    h = tc["hidden_size"]
    return ModelSpec(
        id="deepseek_v41_flash",
        label="DeepSeek-V4.1-Flash (552B / 16B 激活, MLA+CSA2, 384E top-6+1 shared)",
        hidden=h, layers=_v41_plan(tc), vocab=tc["vocab_size"],
        kv_spec={
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
            "kv_replicated_across_tp": True,   # MLA kv_heads=1 → 每卡全量（重建计划 MLA 锚点同口径）
            # 权重精度例外（旧仓 deepseek_attention 中 weight_byte=2 硬编码的三个投影）
            "fp16_weight_ops": ["compress_proj", "index_k_proj", "index_weights_proj"],
        },
        quant={"dense_w": 8, "expert_w": 4, "act": 8, "kv": 8},   # FP8 KV = 1 B/elem
        active_params=int(16e9), total_params=int(552e9),  # 官方口径，仅对账展示
        lm_head_mult=1,
        source_note="官方 config.json 固定快照（2026-11-03，model-verse 仓 pinned 副本同源）",
        config_digest="sha256:" + sha[:16],
    )


MODELS = {
    "tiny_dense_oracle": _tiny_dense(),
    "deepseek_v41_flash": _deepseek_v41_flash(),
}

# mv_model_structure 0.2.0 对本仓模型的支持状态（唯一层图入口的覆盖边界，如实声明）
MV_MODEL_STRUCTURE_SUPPORT = {
    "deepseek_v41_flash": "unsupported: mv_model_structure 0.2.0 注册表为 Qwen3/Qwen3.5/"
                          "Qwen3-Next/DeepSeek-V3；V4.1 Flash 结构由本仓 config 快照适配器"
                          "产出（M2 接回共享包时切换）",
    "tiny_dense_oracle": "unsupported: 手算样例非四款支持架构",
}

# ---------------------------------------------------------------------------
# SpecMethod 注册表：链式方法给出完整语义；其余显式不支持（不静默压扁成 γ 循环）
# ---------------------------------------------------------------------------

DRAFTS: Dict[str, DraftSpec] = {
    "mtp": DraftSpec(
        id="mtp", method="mtp", label="MTP（config 原生 3 层 nextn 微头，串行 K≤3 步）",
        propose_steps_rule="S=K (K≤3)",
        params_dense=3 * 3 * 5120 * 5120,      # 3 层 × 3 矩阵 × h²（旧仓预设公式）
        weight_read_scale=0.10, kv_frac=0.0,
        t2d_bytes_per_token=10240,  # DT 分离时 T→D 需回传 accepted hidden（5120×2B fp16）
        t2d_bytes_per_request=256,
        weight_bits=8,
        source_note="锚: 旧仓 spec_gain_presets.py DRAFT_METHODS['mtp'] + configs/DeepSeek_V41_Flash.py"
                    " get_spec_config；微头与主干共享驻留（读取 ×0.1），无独立 KV",
        status="approximated",
        approximations=["微头按 3·h² 参数等效 GEMM 计（未展开内部 attention/FFN 结构）",
                        "串行步数上限 3（config num_nextn_predict_layers）"],
    ),
    "dspark": DraftSpec(
        id="dspark", method="dspark", label="DSpark（config 原生半自归草稿，MoE 128E top-3）",
        propose_steps_rule="S=K（v2.2 §4.2 链式口径；旧仓预设按 3 阶段块近似，见差异报告 D-4）",
        params_dense=0,
        moe={"experts": 128, "top_k": 3, "inter": 2304, "hidden": 5120},
        weight_read_scale=1.0, kv_frac=0.15,
        t2d_bytes_per_token=0, t2d_bytes_per_request=256,
        weight_bits=8,
        source_note="锚: config dspark_n_routed_experts×3×5120×2304 ≈ 4.53B（公式口径）"
                    " + 旧仓 spec_gain_presets DRAFT_METHODS['dspark']（kvFrac 0.15）",
        status="approximated",
        approximations=["草稿规模按 config 公式 4.53B（仅专家 FFN 参数；笔记口径 41.3B 另列对照）",
                        "链式 S=K 为 v2.2 语义表口径；真实 DSpark 为半自归块结构",
                        "草稿量化官方未公布，按 FP8 计（旧仓 spec 场景按 2B/param，差异报告条目）"],
    ),
    "custom_chain": DraftSpec(
        id="custom_chain", method="custom_chain", label="自定义 dense 链式草稿 1.8B",
        propose_steps_rule="S=K",
        params_dense=1_800_000_000,
        weight_read_scale=1.0,
        kv_frac=0.1125,  # = P_draft/P_active（旧仓 "ratio" 口径）
        t2d_bytes_per_token=0, t2d_bytes_per_request=256,
        weight_bits=8,
        source_note="锚: UI 规格 Tab3 默认（自定义草稿 1.8B, S=4, M=1, K=4）；kvFrac 按参数比折算",
        status="approximated",
        approximations=["dense 草稿按参数量聚合计（未展开层结构）",
                        "草稿 KV 读按目标单步 KV 读 × 参数比折算（旧仓口径），非草稿自身 KV 语义"],
    ),
}

# 显式不支持（决策 D-6）：注册但不实现，调用时返回 unsupported_method（不静默输出）
UNSUPPORTED_METHODS = {
    "dflash": "块并行草稿（两迭代去噪）属后续版本：验证批形状为块级而非 γ+1 线性，"
              "v0.1 三参数契约仅覆盖链式",
    "dflash2": "块并行草稿（block=8）同上",
    "jetspecpp": "JetSpec++ 需 foresight 预测缓存 + 命中率/回退建模（dt_pipe_speculative），属 M3",
}


def get_model(model_id: str) -> ModelSpec:
    if model_id not in MODELS:
        raise ValueError(f"未知模型 {model_id}；可用: {sorted(MODELS)}")
    return MODELS[model_id]


def get_draft(draft_id: str) -> DraftSpec:
    if draft_id in UNSUPPORTED_METHODS:
        raise ValueError(f"unsupported_method: {draft_id} —— {UNSUPPORTED_METHODS[draft_id]}")
    if draft_id not in DRAFTS:
        raise ValueError(f"未知草稿方法 {draft_id}；可用: {sorted(DRAFTS)}；"
                         f"显式不支持: {sorted(UNSUPPORTED_METHODS)}")
    return DRAFTS[draft_id]


def registry_json() -> dict:
    return {
        "schema_version": "0.1",
        "models": {k: v.to_json() for k, v in MODELS.items()},
        "drafts": {k: v.to_json() for k, v in DRAFTS.items()},
        "unsupported_methods": UNSUPPORTED_METHODS,
        "mv_model_structure_support": MV_MODEL_STRUCTURE_SUPPORT,
    }
