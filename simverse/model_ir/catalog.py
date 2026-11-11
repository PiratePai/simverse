"""模型目录：pinned config 快照 → ModelIR / DraftIR（身份与覆盖一并携带）。

登记项（M1 边界如实标注）：
  deepseek_v41_flash     必需场景① 主模型（草稿未指定；Engram/hc 已 source-checked
                         建模（R04）；CSA2/MoE/量化 gaps → 不完整结构）
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
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict

from .families import (parse_deepseek_v3, parse_dflash_draft, parse_gqa,
                       parse_qwen3_5_moe, parse_qwen3_moe, parse_qwen3_next,
                       parse_v41_csa2)
from .schema import (FULL, LINEAR, AttentionIR, DraftIR, LayerIR, LinearAttnIR,
                     MlaIR, ModelEvidence, ModelIR, MoeIR, SourceIdentity)

ASSETS = Path(__file__).resolve().parent.parent.parent / "assets"
_VERSION = "0.3.0"

_QWEN36_EVIDENCE = ModelEvidence(
    forward_source="partial-source",
    cpu_forward="not-run", gpu_measurement="none",
    checked_paths=("sm-qwen35-gdn-decode-branch", "sm-qwen35-attn-output-gate",
                   "sm-qwen35-moe-shared-gate", "sm-qwen35-moe-router",
                   "sm-vllm-attn-gdn-shard", "sm-vllm-moe-tp-ep-recipe"),
    unverified_paths=(
        "mtp_head: config 声明 mtp_num_hidden_layers=1；HF _keys_to_ignore_on_load"
        "_unexpected=[^mtp.*]，vLLM 主模型丢弃 mtp.；无原生 MTP forward pin",
        "text_mrope_position_ids: Qwen3.6 文本 RoPE 读 mrope_section；本仓未执行"
        "位置 id 选择，算量不受影响但语义未实测",
    ),
    method_status={"baseline": "source-checked", "dflash_v1": "source-checked",
                   "mtp": "unverified"},
    note="GDN/full-attn output gate/MoE router+shared gate/MoE TP-EP 分片已逐行"
         "核过；MTP 仅 config+目标首层模板，未执行、未实测。"
         "R06（2026-11-12）已补 fixture 级小尺寸 CPU forward 对照"
         "（ev-qwen36-cpu-fixture-r06：随机权重 2 层缩比，逐 GEMM/conv/router "
         "计数与 shape 精确对账）；模型级 cpu_forward 仍 not-run。")

_V41_EVIDENCE = ModelEvidence(
    forward_source="partial-source",
    cpu_forward="not-run", gpu_measurement="none",
    checked_paths=("sm-v41-engram-hc", "sm-v41-index-short-branch",
                   "sm-v41-mhc-kernels"),
    unverified_paths=(
        "engram_dp_exchange: DP 分片交换（all-to-all/peer/symmetric memory）与"
        "EngramBatch MXFP8 路径未建模（本仓默认单副本路径不启用）；"
        "cpu_offload 运行时行为（预取流/TLB/阈值）未建模——单机 Engram 算子"
        "已 source-checked（ev-vllm-v41-engram，R04）",
        "mhc_tilelang_kernel: TileLang kernel 源已 pin，算术 op 公式源码精确；"
        "RMSNorm 4h+2 为惯例近似；CPU/GPU 未执行，部署变体/split-k 运行时未建模",
        "csa2_attention_projection: CSA2 注意力/投影为 v0.1 移植，非独立 forward 证据",
        "csa2_moe_routing: sqrtsoftplus/noaux_tc/分组路由未建模",
        "candidate_publish: 候选块发布/消费语义未独立核验",
        "quantization_config: FP8/FP4 配置未映射到本仓计算精度",
    ),
    method_status={"baseline": "partial-source", "dflash_v1": "unsupported",
                   "mtp": "unverified"},
    note="Engram 已 source-checked 建模（R04，ev-vllm-v41-engram；DP 交换与 "
         "cpu_offload 运行时行为未覆盖）；CSA2/MoE scoring/量化仍缺口；"
         "hc 仅 source-checked，未执行。")

_DFLASH_EVIDENCE = ModelEvidence(
    forward_source="source-checked",
    cpu_forward="not-run", gpu_measurement="none",
    checked_paths=("sm-dflash-round-protocol", "sm-dflash-cross-attention",
                   "sm-vllm-dflash-colocated-sharing",
                   "sm-vllm-dflash-norm-dtype"),
    unverified_paths=(
        "serving_integration: SGLang DFLASH / vLLM PR#40898 未逐行核对（pending asset）",
        "rejection_sampling_cost: 拒绝采样/KV 回退/hidden 提取无独立耗时，"
        "当前显式 0 占位",
        "runtime_pool_sizes: workspace/CUDA graph/NCCL 池尺寸无证据",
    ),
    method_status={"dflash_v1": "source-checked"},
    note="草稿结构/轮协议/同卡共享已源码核对；未执行、未实测、未部署验证。")


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
    full = _load(path)
    ir = parse_v41_csa2(
        full["text_config"], _src("deepseek-ai/DeepSeek-V4.1-Flash",
                                  "314113f8a7dadb388215396ba5fbee2c610a04d6",
                                  _sha(path), "官方 config 固定快照"),
        "deepseek_v41_flash",
        "DeepSeek-V4.1-Flash (MLA+CSA2, 全层 MoE, Engram/hc 已建模/CSA2-MoE 缺口)",
        full_config=full, text_prefix="text_config")
    return replace(ir, evidence=_V41_EVIDENCE,
                   evidence_refs=("ev-v41-config", "ev-vllm-v41-mhc",
                                  "ev-vllm-v41-model",
                                  "ev-vllm-v41-engram"))


def _qwen36() -> ModelIR:
    path = ASSETS / "config" / "qwen36_35b_a3b_main__config.json"
    full = _load(path)
    ir = parse_qwen3_5_moe(
        full["text_config"],
        _src("Qwen/Qwen3.6-35B-A3B", "345d467b117d1fd5eda9b879a5988de9ae5f47a7",
             _sha(path), "官方 main 固定快照（用户原名 Qwen3.6-35BA3B）"),
        "qwen36_35b_a3b", "Qwen3.6-35B-A3B (30 GDN + 10 FullAttn, 256E top-8+shared, MTP1)",
        full_config=full, text_prefix="text_config")
    return replace(ir, evidence=_QWEN36_EVIDENCE,
                   evidence_refs=("ev-qwen36-main-config", "ev-hf-qwen35-modeling",
                                  "ev-hf-qwen35-config",
                                  "ev-qwen36-cpu-fixture-r06"))


def _dflash_draft() -> DraftIR:
    path = ASSETS / "config" / "qwen36_35b_a3b_dflash__config.json"
    cfg = _load(path)
    ir = parse_dflash_draft(
        cfg, _src("z-lab/Qwen3.6-35B-A3B-DFlash",
                  "f181eece646affea2c38b2765f1aaa01a9734ccd", _sha(path),
                  "方法实现另 pin z-lab/dflash@07ebd93"),
        "dflash_qwen36_35b_a3b",
        "DFlash v1 for Qwen3.6-35B-A3B (6L dense GQA, block diffusion)",
        full_config=cfg)
    return replace(ir, evidence=_DFLASH_EVIDENCE,
                   evidence_refs=("ev-qwen36-dflash-config", "ev-dflash-impl",
                                  "ev-dflash-readme",
                                  "ev-vllm-dflash-norm-dtype"))


def _qwen3_30b() -> ModelIR:
    path = ASSETS / "config" / "qwen3_30b_a3b__config.json"
    cfg = _load(path)
    ir = parse_qwen3_moe(
        cfg, _src("Qwen/Qwen3-30B-A3B",
                  "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
                  _sha(path), "历史归档；revision 与 identity 全等（2026-11-10 修复）"),
        "qwen3_30b_a3b", "Qwen3-30B-A3B (48L GQA MoE 128E top-8) [历史归档]")
    return replace(ir, evidence_refs=("ev-qwen3-30b-config",))


def _qwen3_235b() -> ModelIR:
    path = ASSETS / "config" / "qwen3_235b_a22b__config.json"
    cfg = _load(path)
    ir = parse_qwen3_moe(
        cfg, _src("Qwen/Qwen3-235B-A22B",
                  "8efa61729e24bd65b1d152b5ab5409052aa80e65", _sha(path),
                  "历史归档；官方 main 快照（2026-11-10 HF resolve）"),
        "qwen3_235b_a22b",
        "Qwen3-235B-A22B (94L GQA MoE 128E top-8) [历史归档]")
    return replace(ir, evidence_refs=("ev-qwen3-235b-config",
                                      "ev-h200-235b-distill"))


def _qwen3_next() -> ModelIR:
    path = ASSETS / "config" / "qwen3_next_80b_a3b__config.json"
    cfg = _load(path)
    ir = parse_qwen3_next(
        cfg, _src("Qwen/Qwen3-Next-80B-A3B-Instruct", "local-snapshot",
                  _sha(path), "历史归档；model-verse 本地快照，官方 revision 待补"),
        "qwen3_next_80b_a3b",
        "Qwen3-Next-80B-A3B (48L: 12 Full@idx3,7..47 + 36 GDN, 512E top-10) [历史归档]")
    return replace(ir, evidence_refs=("ev-qwen3-next-config",))


def _deepseek_v3() -> ModelIR:
    path = ASSETS / "config" / "deepseek_v3__config.json"
    cfg = _load(path)
    ir = parse_deepseek_v3(
        cfg, _src("deepseek-ai/DeepSeek-V3",
                  "e815299b0bcbac849fa540c768ef21845365c9eb",
                  _sha(path), "历史归档；revision 与 identity 全等（2026-11-10 修复）"),
        "deepseek_v3",
        "DeepSeek-V3 (61L MLA, 3 Dense + 58 MoE 256E top-8+1shared) [历史归档]")
    return replace(ir, evidence_refs=("ev-deepseek-v3-config",))


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


def build_catalog(include_internal: bool = False) -> Dict[str, Any]:
    """运行时目录（默认只含两款主模型 + 与 Qwen3.6 绑定的 DFlash DraftIR）。

    include_internal=True 仅用于离线回归/历史校准验证：加入 tiny 合成 fixture
    与四款历史归档模型。CLI/API/Web/部署排名路径**不得**传 True；它们使用默认
    目录，历史模型只能通过测试显式注入的 archive catalog 运行。
    """
    models: Dict[str, ModelIR] = {
        "deepseek_v41_flash": _v41(),
        "qwen36_35b_a3b": _qwen36(),
    }
    drafts = {"dflash_qwen36_35b_a3b": _dflash_draft()}
    if include_internal:
        models.update({
            # 合成 fixture：手算/机制测试专用，不是产品/部署选择
            "tiny_gqa": _tiny_gqa(),
            "tiny_hybrid": _tiny_hybrid(),
            "tiny_mla": _tiny_mla(),
            "tiny_moe": _tiny_moe(),
            # 历史归档：保留解析与校准回归，禁止进入运行时目录
            "qwen3_30b_a3b": _qwen3_30b(),
            "qwen3_235b_a22b": _qwen3_235b(),
            "qwen3_next_80b_a3b": _qwen3_next(),
            "deepseek_v3": _deepseek_v3(),
        })
    return {"models": models, "drafts": drafts,
            "pending_assets": PENDING_ASSETS, "version": _VERSION}


PRODUCT_MODELS = frozenset({"qwen36_35b_a3b", "deepseek_v41_flash"})
ARCHIVE_MODELS = frozenset({"qwen3_30b_a3b", "qwen3_235b_a22b",
                            "qwen3_next_80b_a3b", "deepseek_v3"})
FIXTURE_MODELS = frozenset({"tiny_gqa", "tiny_hybrid", "tiny_mla", "tiny_moe"})


def catalog_json() -> dict:
    cat = build_catalog()
    return {
        "schema_version": "0.2",
        "model_ir_version": _VERSION,
        "models": {k: m.to_json() for k, m in cat["models"].items() if k in PRODUCT_MODELS},
        "drafts": {k: d.to_json() for k, d in cat["drafts"].items()},
        "pending_assets": cat["pending_assets"],
    }
