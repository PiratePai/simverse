"""模型目录：pinned config 快照 → ModelIR / DraftIR（身份与覆盖一并携带）。

登记项（M1 边界如实标注）：
  deepseek_v41_flash     必需场景① 主模型（草稿未指定；Engram/hc gaps → 不完整结构）
  qwen36_35b_a3b         必需场景② 主模型（结构完备；多模 vision 登记 metadata-only）
  dflash_qwen36_35b_a3b  必需场景② 草稿（结构与协议证据已 pin）
  tiny_*                 手算锚点（tiny_gqa / tiny_hybrid / tiny_mla / tiny_moe）
  qwen3_30b_a3b          历史回归（固定 revision）
  qwen3_next_80b_a3b     历史回归（本地快照；官方仓 gated → 身份注记）
  deepseek_v3            历史回归（固定 revision）
  本地原称 "Qwen3.8"     PENDING_ASSET：以实际本地模型 model_type/revision 辨认后再入册
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

from .families import (parse_deepseek_v3, parse_dflash_draft, parse_gqa,
                       parse_qwen3_5_moe, parse_qwen3_moe, parse_qwen3_next,
                       parse_v41_csa2)
from .schema import (FULL, LINEAR, AttentionIR, DraftIR, LayerIR, LinearAttnIR,
                     MlaIR, ModelIR, MoeIR, SourceIdentity)

ASSETS = Path(__file__).resolve().parent.parent.parent / "assets"
_VERSION = "0.3.0"


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _src(repo: str, revision: str, sha: str, note: str = "") -> SourceIdentity:
    return SourceIdentity(repo_id=repo, revision=revision, config_sha256=sha,
                          captured_at="2026-11-09", note=note)


# ---------------------------------------------------------------------------
# 手算锚点模型（tiny 家族；config 为内存构造，身份为 synthetic）
# ---------------------------------------------------------------------------

def _tiny_gqa() -> ModelIR:
    """2 层 GQA dense（v0.1 oracle 模型，账本整数可手算）。"""
    h, heads, kv, d, ffn = 256, 8, 2, 32, 512
    attn = AttentionIR(heads=heads, kv_heads=kv, head_dim=d, rotary_dim=d)
    src = SourceIdentity("synthetic:tiny", "none", "hand-constructed",
                         captured_at="2026-11-09", note="手算样例，非真实模型")
    layers = [LayerIR(i, FULL, attn=attn, moe=None, dense_inter=ffn) for i in range(2)]
    return ModelIR("tiny_gqa", "Tiny GQA Dense (2L, d=256, ffn=512)", "gqa",
                   h, 1000, layers, src,
                   coverage={"*": {"status": "mapped", "consumer": "model_ir",
                                   "note": "合成模型，全部字段手写"}}, extras={})


def _tiny_hybrid() -> ModelIR:
    """4 层混合（1 GDN linear + 3 GQA full）+ MoE 8E top-2 + shared：混合层序锚点。"""
    h, heads, kv, d = 256, 8, 2, 32
    attn = AttentionIR(heads=heads, kv_heads=kv, head_dim=d, rotary_dim=16,
                       output_gated=True)
    lin = LinearAttnIR(num_k_heads=4, num_v_heads=8, head_k_dim=32, head_v_dim=32,
                       conv_kernel=4)
    moe = MoeIR(experts=8, top_k=2, intermediate=512, shared_intermediate=256,
                shared_gate=True)
    src = SourceIdentity("synthetic:tiny", "none", "hand-constructed",
                         captured_at="2026-11-09", note="混合层序手算锚点")
    kinds = [LINEAR, FULL, FULL, FULL]
    layers = [LayerIR(i, k, attn=attn if k == FULL else None,
                      linear=lin if k == LINEAR else None, moe=moe)
              for i, k in enumerate(kinds)]
    return ModelIR("tiny_hybrid", "Tiny Hybrid GDN+GQA MoE (4L, 8E top-2+shared)",
                   "qwen3_5_moe", h, 1000, layers, src,
                   coverage={"*": {"status": "mapped", "consumer": "model_ir",
                                   "note": "合成模型"}},
                   extras={"attn_output_gate": True, "mamba_ssm_dtype": "float32"})


def _tiny_mla() -> ModelIR:
    """2 层 MLA（V3 布局压缩 KV）：kv_lora 128 + rope 32。"""
    mla = MlaIR(heads=8, q_lora_rank=192, kv_lora_rank=128, qk_rope_dim=32,
                qk_nope_dim=128, v_head_dim=64)
    src = SourceIdentity("synthetic:tiny", "none", "hand-constructed",
                         captured_at="2026-11-09", note="MLA 元素量锚点")
    layers = [LayerIR(i, "mla", mla=mla, moe=None, dense_inter=512) for i in range(2)]
    return ModelIR("tiny_mla", "Tiny MLA Dense (2L, kv_lora=128+rope32)", "deepseek_v3",
                   256, 1000, layers, src,
                   coverage={"*": {"status": "mapped", "consumer": "model_ir",
                                   "note": "合成模型"}},
                   extras={"kv_spec": {"family": "mla_v3", "kv_lora_rank": 128,
                                       "qk_rope_dim": 32,
                                       "kv_elems_per_token_per_layer": 128 + 32}})


def _tiny_moe() -> ModelIR:
    """1 层 GQA + MoE（无 shared）：MoE 并集/路由账本锚点。"""
    attn = AttentionIR(heads=8, kv_heads=2, head_dim=32, rotary_dim=32)
    moe = MoeIR(experts=4, top_k=2, intermediate=512)
    src = SourceIdentity("synthetic:tiny", "none", "hand-constructed",
                         captured_at="2026-11-09", note="MoE 账本锚点")
    return ModelIR("tiny_moe", "Tiny MoE (1L, 4E top-2)", "qwen3_moe", 256, 1000,
                   [LayerIR(0, FULL, attn=attn, moe=moe)], src,
                   coverage={"*": {"status": "mapped", "consumer": "model_ir",
                                   "note": "合成模型"}})


# ---------------------------------------------------------------------------
# 真实模型（pinned config 快照）
# ---------------------------------------------------------------------------

def _v41() -> ModelIR:
    path = ASSETS / "config" / "deepseek_v41_config.json"
    cfg = _load(path)["text_config"]
    return parse_v41_csa2(
        cfg, _src("deepseek-ai/DeepSeek-V4.1-Flash",
                  "314113f8a7dadb388215396ba5fbee2c610a04d6", _sha(path),
                  "官方 config 固定快照"),
        "deepseek_v41_flash", "DeepSeek-V4.1-Flash (MLA+CSA2, 全层 MoE, Engram/hc 缺口)")


def _qwen36() -> ModelIR:
    path = ASSETS / "config" / "qwen36_35b_a3b_main__config.json"
    cfg = _load(path)["text_config"]
    ir = parse_qwen3_5_moe(
        cfg, _src("Qwen/Qwen3.6-35B-A3B", "345d467b117d1fd5eda9b879a5988de9ae5f47a7",
                  _sha(path), "官方 main 固定快照（用户原名 Qwen3.6-35BA3B）"),
        "qwen36_35b_a3b", "Qwen3.6-35B-A3B (30 GDN + 10 FullAttn, 256E top-8+shared, MTP1)")
    # 多模封装键登记 metadata-only（文本推理范围；vision 进前置预算属后续）
    for key in ("image_token_id", "video_token_id", "vision_start_token_id",
                "vision_end_token_id"):
        ir.coverage[key] = {"status": "metadata-only", "consumer": "identity",
                            "note": "多模输入通道；本仓文本推理范围不消费（vision 预算另立 scope）"}
    return ir


def _dflash_draft() -> DraftIR:
    path = ASSETS / "config" / "qwen36_35b_a3b_dflash__config.json"
    cfg = _load(path)
    return parse_dflash_draft(
        cfg, _src("z-lab/Qwen3.6-35B-A3B-DFlash",
                  "f181eece646affea2c38b2765f1aaa01a9734ccd", _sha(path),
                  "方法实现另 pin z-lab/dflash@07ebd93"),
        "dflash_qwen36_35b_a3b", "DFlash v1 for Qwen3.6-35B-A3B (6L dense GQA, block diffusion)")


def _qwen3_30b() -> ModelIR:
    path = ASSETS / "config" / "qwen3_30b_a3b__config.json"
    cfg = _load(path)
    return parse_qwen3_moe(
        cfg, _src("Qwen/Qwen3-30B-A3B", "ad44e777bcd14f90c4fc516864591c19e339c933",
                  _sha(path), "官方 main 固定快照"),
        "qwen3_30b_a3b", "Qwen3-30B-A3B (48L GQA MoE 128E top-8)")


def _qwen3_235b() -> ModelIR:
    path = ASSETS / "config" / "qwen3_235b_a22b__config.json"
    cfg = _load(path)
    return parse_qwen3_moe(
        cfg, _src("Qwen/Qwen3-235B-A22B",
                  "8efa61729e24bd65b1d152b5ab5409052aa80e65", _sha(path),
                  "官方 main 快照（2026-11-10 经 HF resolve/main 获取并取 revision；"
                  "pi-viewer 二手副本逐字段核对一致，仅其多自加 profiling 键）"),
        "qwen3_235b_a22b",
        "Qwen3-235B-A22B (94L GQA MoE 128E top-8; 官方身份 2026-11-10 补，G-D3 解除)")


def _qwen3_next() -> ModelIR:
    path = ASSETS / "config" / "qwen3_next_80b_a3b__config.json"
    cfg = _load(path)
    return parse_qwen3_next(
        cfg, _src("Qwen/Qwen3-Next-80B-A3B-Instruct", "local-snapshot",
                  _sha(path), "model-verse 本地快照；官方 HF 仓 gated(401)，"
                  "官方 revision 身份待补（PENDING_ASSET）"),
        "qwen3_next_80b_a3b", "Qwen3-Next-80B-A3B (48L: 12 Full@idx3,7..47 + 36 GDN, 512E top-10)")


def _deepseek_v3() -> ModelIR:
    path = ASSETS / "config" / "deepseek_v3__config.json"
    cfg = _load(path)
    return parse_deepseek_v3(
        cfg, _src("deepseek-ai/DeepSeek-V3", "e815299b0bcbef1ef060f18c8ede89408bb1e93c",
                  _sha(path), "官方 main 固定快照"),
        "deepseek_v3", "DeepSeek-V3 (61L MLA, 3 Dense + 58 MoE 256E top-8+1shared)")


# ---------------------------------------------------------------------------
# 目录注册
# ---------------------------------------------------------------------------

PENDING_ASSETS = {
    "qwen3_8_local": {
        "status": "pending-asset",
        "reason": "本地原称 Qwen3.8 的模型待以实际本地文件 model_type/revision 辨认；"
                  "HF 上存在 Qwen/Qwen3.8-27B（model_type qwen3）但未确认即用户所指，不冒领",
        "affected": ["该模型的结构回归与计数对账"],
    },
    "qwen36_dflash_serving_integration": {
        "status": "pending-asset",
        "reason": "SGLang DFLASH 服务端集成与 vLLM PR#40898 未逐行核对（部署能力证据）",
        "affected": ["部署矩阵中 DFlash 的 served 标记（模拟不冒充已部署）"],
    },
}


def build_catalog() -> Dict[str, Any]:
    models: Dict[str, ModelIR] = {
        "tiny_gqa": _tiny_gqa(),
        "tiny_hybrid": _tiny_hybrid(),
        "tiny_mla": _tiny_mla(),
        "tiny_moe": _tiny_moe(),
        "deepseek_v41_flash": _v41(),
        "qwen36_35b_a3b": _qwen36(),
        "qwen3_30b_a3b": _qwen3_30b(),
        "qwen3_235b_a22b": _qwen3_235b(),
        "qwen3_next_80b_a3b": _qwen3_next(),
        "deepseek_v3": _deepseek_v3(),
    }
    drafts = {"dflash_qwen36_35b_a3b": _dflash_draft()}
    return {"models": models, "drafts": drafts,
            "pending_assets": PENDING_ASSETS, "version": _VERSION}


PRODUCT_MODELS = frozenset({"qwen36_35b_a3b", "deepseek_v41_flash"})


def catalog_json() -> dict:
    cat = build_catalog()
    return {
        "schema_version": "0.2",
        "model_ir_version": _VERSION,
        "models": {k: m.to_json() for k, m in cat["models"].items() if k in PRODUCT_MODELS},
        "drafts": {k: {
            "id": d.id, "label": d.label, "hidden": d.hidden, "vocab": d.vocab,
            "n_layers": len(d.layers),
            "layer_kinds": [l.kind for l in d.layers],
            "uses_target_embeddings": d.uses_target_embeddings,
            "uses_target_lm_head": d.uses_target_lm_head,
            "target_layer_ids": list(d.target_layer_ids),
            "fc_in": d.fc_in, "block_size": d.block_size,
            "mask_token_id": d.mask_token_id,
            "source": d.source.to_json(),
        } for k, d in cat["drafts"].items()},
        "pending_assets": cat["pending_assets"],
    }
