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
        path = (ASSETS / rel) if not rel.startswith("docs") else ROOT / rel
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
