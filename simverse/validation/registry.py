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
    ("ev-dflash-impl", "source-checked",
     "DFlash 方法实现（dflash_generate 协议/DFlashDraftModel 结构/拒绝采样）",
     "identity/dflash_repo/model.py",
     "github z-lab/dflash @07ebd93db9f472af339b644bb70221ad8428328a dflash/model.py",
     "DFlash v1 单次并行提案、逐位拒绝、KV 回退语义",
     ["源码阅读未执行；SGLang/vLLM 服务端集成路径未逐行核对"], False),
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
    for (ev_id, kind, claim, rel, source, domain, limits, executed) in _PINNED_ASSETS:
        path = (ASSETS / rel) if not rel.startswith("docs") else ROOT / rel
        sha = _sha256_file(path)
        if not sha:
            # 资产缺失：登记为缺口而不是跳过（如实标注原则）
            records[ev_id] = EvidenceRecord(
                evidence_id=ev_id, kind="source-checked", claim=claim,
                source=source, content_sha256="MISSING", captured_at="2026-11-09",
                domain=domain, limits=["资产文件缺失：" + rel], executed=False)
            continue
        records[ev_id] = EvidenceRecord(
            evidence_id=ev_id, kind=kind, claim=claim, source=source,
            content_sha256=sha, captured_at="2026-11-09", domain=domain,
            limits=limits, executed=executed,
            command=f"sha256sum {rel}")
    return records


def registry_json() -> dict:
    return {"schema_version": "0.2", "code_revision": code_revision(),
            "evidence": {k: v.to_json() for k, v in build_registry().items()}}


def evidence_index() -> List[str]:
    return sorted(build_registry())
