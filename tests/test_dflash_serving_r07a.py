# -*- coding: utf-8 -*-
"""R07a：DFlash 服务栈配对/协议/计时边界审计的独立 oracle 与负例（2026-11-12）。

证据锚（逐文件 SHA256 见 assets/identity/*/manifest.json）：
  * vLLM @dc74b8cbbbe3dbb9399fe369757d63defd614cc2
    （vllm_dflash_serving_pin：dflash/speculator、cudagraph、config、
    eagle3_utils、上级 speculator、rejection_sampler(+utils)、
    config/speculative、4 个 DFlash 测试）
  * SGLang @74cb973d78dfa4af85a90d2841891fec27e78db2
    （sglang_dflash_pin：dflash_worker_v2、dflash_info(_v2)、dflash_utils、
    dflash_dp、dflash_disaggregation、spec_info、models/dflash、
    kernels dflash+candidates、2 个 e2e 测试）

全部为 source-checked 源码阅读核对，**不构成 CPU 执行或 GPU 实测**。
数字由本文件手写公式推导（不调用生产实现求期望）。
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simverse.count.draft import dflash_round_rows  # noqa: E402
from simverse.importers.source_map import source_map_json  # noqa: E402
from simverse.model_ir import build_catalog  # noqa: E402
from simverse.pipeline import run_v2  # noqa: E402
from simverse.scenario import ScenarioV2  # noqa: E402
from simverse.validation.registry import build_registry  # noqa: E402

Q8 = {"dense_w": 8, "expert_w": 8, "act": 8, "kv": 8, "state": 32}
H = 2048
D = 128
N_LAYERS = 6
TP = 8
KV_HEADS_RANK = 1
Q_HEADS_RANK = 4


# ---------------------------------------------------------------------------
# 1) pin 完整性：manifest 在册、逐文件 SHA256（LF 归一）可复算
# ---------------------------------------------------------------------------

def _sha_lf(path: Path) -> str:
    b = path.read_bytes()
    return hashlib.sha256(b.replace(b"\r\n", b"\n")).hexdigest()


def test_vllm_serving_pin_manifest_hashes_replay():
    """vLLM 服务栈 pin：13 文件逐一按 manifest SHA256 复算（LF 归一）。

    负例内含：清单里不存在的文件名必须使查找失败（不静默跳过）。
    """
    base = ROOT / "assets/identity/vllm_dflash_serving_pin"
    m = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    assert m["repo"] == "vllm-project/vllm"
    assert m["sha"] == "dc74b8cbbbe3dbb9399fe369757d63defd614cc2"
    assert len(m["files"]) == 13
    for name, sha in m["files"].items():
        assert _sha_lf(base / name) == sha, name
    # 与既有 vllm_qwen35_pin 同源文件一致性（speculator 单一事实）
    q = _sha_lf(ROOT / "assets/identity/vllm_qwen35_pin/dflash_speculator.py")
    assert q == m["files"]["dflash_speculator.py"]
    assert "not-in-manifest.py" not in m["files"]


def test_sglang_serving_pin_manifest_hashes_replay():
    """SGLang DFLASH pin：12 文件逐一按 manifest SHA256 复算（LF 归一）。"""
    base = ROOT / "assets/identity/sglang_dflash_pin"
    m = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    assert m["repo"] == "sgl-project/sglang"
    assert m["sha"] == "74cb973d78dfa4af85a90d2841891fec27e78db2"
    assert len(m["files"]) == 12
    for name, sha in m["files"].items():
        assert _sha_lf(base / name) == sha, name
    # v1 worker 在 pin commit 已不存在（仅 worker_v2）——清单如实反映
    assert "dflash_worker.py" not in m["files"]
    assert "dflash_worker_v2.py" in m["files"]


def test_pinned_sources_present_key_audit_symbols():
    """pin 内容包含审计断言的关键符号（防'文件在而内容非'的空 pin）。"""
    vdir = ROOT / "assets/identity/vllm_dflash_serving_pin"
    spec = (vdir / "dflash_speculator.py").read_text(encoding="utf-8")
    assert "num_query_per_req = 1 + self.num_speculative_steps" in spec
    assert "_precompute_context_kv" in spec
    assert "valid_ctx_end = ctx_end - num_rejected" in spec
    rs = (vdir / "rejection_sampler_utils.py").read_text(encoding="utf-8")
    assert "target_argmax == draft_sampled" in rs          # greedy 前缀匹配
    assert "target_logprob > tl.log(u) + draft_logprob" in rs  # 随机比值检验
    cfg = (vdir / "speculative_config.py").read_text(encoding="utf-8")
    assert 'draft_sample_method: DraftSampleMethod = "greedy"' in cfg
    sdir = ROOT / "assets/identity/sglang_dflash_pin"
    w2 = (sdir / "dflash_worker_v2.py").read_text(encoding="utf-8")
    assert "DFLASH's draft has no head of its own; it borrows the target `lm_head`" in w2
    assert "falling back to greedy argmax verification" in w2  # 采样降级警告
    kern = (sdir / "kernels_dflash.py").read_text(encoding="utf-8")
    assert "last_steps + row, commit - 1" in kern          # mamba 态回退
    assert "prefix_lens_ptr + row * prefix_lens_stride) + commit_len" in kern
    models = (sdir / "models_dflash.py").read_text(encoding="utf-8")
    assert "num_context_features * hidden_size" in models  # fc 8h→h


# ---------------------------------------------------------------------------
# 2) F1 独立 oracle：ctx 投影跨度基（span=B×blk 计算 / produced 只进写入）
# ---------------------------------------------------------------------------

def test_fc_hidden_norm_ctx_proj_span_basis_oracle():
    """R07a 修正的独立手算：fc/hidden_norm/ctx_kv_proj 按 verify 跨度计。

    依据：vLLM _precompute_context_kv(0, num_target_tokens) 覆盖全跨度、
    graph 路径 _num_graph_context_tokens=num_reqs×(N+1)；SGLang
    project_target_hidden(全跨度) + commit_lens 前缀写。被拒行不落缓存
    （kv_write 仍 produced 行）。
    手算（B=1, blk=16, h=2048, fc_in=8h, TP8, fused_raw）：
      fc FLOPs   = 2·16·16384·2048 = 1,073,741,824
      hidden_norm ops = 16·(4·2048+2) = 131,096（RMSNorm 惯例 4h+2）
      ctx FLOPs/层 = 2·16·2048·256 = 16,777,216；kv_write = 3·2·1·128
    """
    cat = build_catalog()
    m, d = cat["models"]["qwen36_35b_a3b"], cat["drafts"]["dflash_qwen36_35b_a3b"]
    B, e_n, blk = 1, 3.0, 16
    rows = dflash_round_rows("draft_block", d, m.vocab, e_n, 512, Q8, 1.0,
                             tp=TP, batch=B, ctx_kv_path="fused_raw")
    by_op = {r.op: r for r in rows if r.layer == -1}
    fc = by_op["dflash_fc"]
    assert fc.flops == 2 * (B * blk) * (8 * H) * H == 1_073_741_824
    assert fc.act_read == (B * blk) * 8 * H
    assert fc.act_write == (B * blk) * H
    hn = by_op["dflash_hidden_norm"]
    assert hn.flops == (B * blk) * (4 * H + 2)
    ctx = [r for r in rows if r.op == "ctx_kv_proj"]
    assert len(ctx) == N_LAYERS
    for r in ctx:
        assert r.flops == 2 * (B * blk) * H * (2 * KV_HEADS_RANK * D)
        assert r.act_read == (B * blk) * H          # 跨度读
        assert r.act_write == (B * blk) * 2 * KV_HEADS_RANK * D
        # committed 写不随跨度放大：E[N]=3 → produced=3 行
        assert r.kv_write == int(B * e_n * 2 * KV_HEADS_RANK * D)
    # 草稿层 token 行与 lm_head 候选行不随 E[N] 变（已有口径回归）
    assert by_op["noise_embed"].act_read == B * blk * H
    lm = by_op["draft_lm_head"]
    assert lm.flops == 2 * B * (blk - 1) * H * (m.vocab // TP)


def test_span_correction_is_strictly_upward_not_kv_double():
    """负例：跨度基不得放大 kv_write 或草稿层 token 行（防重复计费）。

    E[N]→blk 只作用于投影的 flops/act；缓存写入、noise/q/k/v/o/MLP 的
    token 行（本就 B×blk）、权重读均不变。
    """
    cat = build_catalog()
    m, d = cat["models"]["qwen36_35b_a3b"], cat["drafts"]["dflash_qwen36_35b_a3b"]
    common = dict(stage="draft_block", draft=d, target_vocab=m.vocab,
                  start_ctx=512, quant=Q8, target_quant_act=1.0, tp=TP,
                  batch=1, ctx_kv_path="fused_raw")
    partial = dflash_round_rows(e_produced=3.0, **common)
    full = dflash_round_rows(e_produced=16.0, **common)   # E[N]=blk 全接受上界
    g = lambda rows, op: next(r for r in rows if r.op == op and r.layer == -1)
    # 跨度基：投影计算与 E[N] 完全无关（两运行同值——不随接受率变化）
    for op in ("dflash_fc", "dflash_hidden_norm"):
        assert g(partial, op).flops == g(full, op).flops
    ctx_p = [r for r in partial if r.op == "ctx_kv_proj"][0]
    ctx_f = [r for r in full if r.op == "ctx_kv_proj"][0]
    assert ctx_p.flops == ctx_f.flops and ctx_p.act_read == ctx_f.act_read
    # kv_write 严格随 E[N] 线性（不被 blk 覆盖：16/3 倍）
    assert ctx_f.kv_write == ctx_p.kv_write * 16 / 3
    # 草稿层 q_proj 的 token 行与 E[N] 无关
    q_old = [r for r in partial if r.op == "q_proj"][0]
    q_full = [r for r in full if r.op == "q_proj"][0]
    assert q_old.flops == q_full.flops


# ---------------------------------------------------------------------------
# 3) dflash_serving_integration 分模式判定（G-C3 验收）
# ---------------------------------------------------------------------------

def _cap(b) -> dict:
    return b.capability["checks"]   # id → check 字典


def test_capability_colocated_supported_source_checked():
    """colocated：supported 且注记 source-checked ≠ 已部署实测。"""
    b = run_v2(ScenarioV2())
    cap = _cap(b)
    c = cap["dflash_serving_integration"]
    assert c["status"] == "supported"
    assert "source-checked" in c["reason"] and "已部署实测" in c["reason"]
    assert "R07b" in c["reason"]
    # 证据引用按场景实际参与（两栈证据都在册）
    assert "ev-vllm-dflash-serving" in b.evidence_refs
    assert "ev-sglang-dflash-serving" in b.evidence_refs


def test_capability_dt_modes_unsupported_no_stack_counterpart():
    """负例：dt_serial/dt_pipeline_sim 无服务栈对应 → unsupported（非 unknown）。

    vLLM proposer in-process；SGLang DraftParallelSpec SINGLE_GROUP、
    disagg 是 target 侧 PD——均无 draft/target 分离路径。dt 模式是本仓
    建模假设，不得借 capability 暗示已有部署形态。
    """
    for mode in ("dt_serial", "dt_pipeline_sim"):
        sc = ScenarioV2()
        sc.deployment.mode = mode
        b = run_v2(sc)
        c = _cap(b)["dflash_serving_integration"]
        assert c["status"] == "unsupported", mode
        assert "无 draft/target 分离" in c["reason"] or "均无" in c["reason"]


def test_stage_cost_complete_stays_unknown_with_kernel_note():
    """accept/KV 回退/hidden 提取仍 unknown（in-graph GPU kernel 未计价）。"""
    b = run_v2(ScenarioV2())
    c = _cap(b)["stage_cost_complete"]
    assert c["status"] == "unknown"
    assert "in-graph" in c["reason"]


def test_pending_asset_registry_reflects_source_checked_status():
    """catalog pending-asset 状态翻转为 source-checked，且不得宣称已部署。"""
    from simverse.model_ir.catalog import PENDING_ASSETS
    pa = PENDING_ASSETS["qwen36_dflash_serving_integration"]
    assert pa["status"].startswith("source-checked")
    assert "R07b" in pa["reason"] and "served" in pa["affected"][0]


# ---------------------------------------------------------------------------
# 4) source_map / 证据注册：新条目在册且不冒充执行
# ---------------------------------------------------------------------------

def test_source_map_serving_entries_registered():
    sm = {e["entry_id"]: e for e in source_map_json()["entries"]}
    v = sm["sm-vllm-dflash-serving"]
    s = sm["sm-sglang-dflash-serving"]
    assert v["source_pin"].startswith("vllm@dc74b8cb")
    assert s["source_pin"].startswith("sgl-project/sglang@74cb973d")
    for e in (v, s):
        assert e["evidence_kind"] == "source-checked"
        assert e["both_sides_covered"] is True
        assert any("实测" in u or "执行" in u or "未知" in u or "unknown" in u
                   for u in e["uncovered_domain"] + [e["note"]])
    # 两栈通信口径差异显式登记（SGLang 非 vLLM 全量 gather）
    assert "非全量 logits gather" in s["semantic"]
    assert "DraftLogitsCache" in " ".join(v["uncovered_domain"])
    # round-protocol 条目不再声称服务栈未核对
    rp = sm["sm-dflash-round-protocol"]
    assert not any("未逐行核对" in u for u in rp["uncovered_domain"])
    assert source_map_json()["pins"]["sglang_dflash"].startswith(
        "sgl-project/sglang@74cb973d")


def test_evidence_registry_entries_exist_not_executed():
    reg = build_registry()
    for ev in ("ev-vllm-dflash-serving", "ev-sglang-dflash-serving"):
        r = reg[ev]
        assert r.kind == "source-checked" and r.executed is False
        assert r.content_sha256 not in ("", "MISSING")
        assert any("source-checked" in l or "非" in l for l in r.limits)


# ---------------------------------------------------------------------------
# 5) 默认场景重放：F1 修正只改草稿投影行（差异归因可复算）
# ---------------------------------------------------------------------------

def test_default_scenario_delta_only_in_draft_projection_rows():
    """重放口径：fc/hidden_norm/ctx_kv_proj 变大（produced→span 基），其余
    工作行不变。用 E[N] 与 blk 的比例独立复算期望倍数。
    """
    b = run_v2(ScenarioV2())
    blk = b.method_plan["block_size"]
    # E[N] 独立复算（默认几何 α：E[N]=1+Σ_{i=1..K} α^i，K=blk−1 候选）
    alpha = b.scenario_snapshot["acceptance"]["alpha"]
    e_n = 1.0 + sum(alpha ** i for i in range(1, blk))
    assert e_n > 1.0
    draft = {r["op"]: r for r in b.work["draft_block"]["rows"]}
    fc = draft["dflash_fc"]
    # 独立复算：span 基 flops = 2·B·blk·fc_in·h（B=默认场景 batch）
    B = b.scenario_snapshot["workload"]["batch"]
    assert "verify 跨度" in fc["shape"]
    # ratio 交叉验证：span/produced = blk/E[N]（fc act_read 维度）
    ratio = (B * blk) / (B * e_n)
    hn = draft["dflash_hidden_norm"]
    ctx = draft["ctx_kv_proj"]
    # 跨度基下三行同基：fc/hn/ctx 的行数一致（act_read 比例可验）
    assert fc["act_read"] / (8 * H) == B * blk
    assert hn["act_read"] / H == B * blk
    assert ctx["act_read"] / H == B * blk
    # kv_write 行（committed 基）不出现在这三行的 act 维度里：
    assert ctx["kv_write"] == int(B * e_n * 2 * KV_HEADS_RANK * D)
    assert abs(ratio - blk / e_n) < 1e-9


# ---------------------------------------------------------------------------
# 6) F2 修正负例：草稿行按生效块（plan.block_size）而非 checkpoint 原生块
# ---------------------------------------------------------------------------

def test_draft_rows_follow_effective_block_size_not_checkpoint():
    """负例（R07a 对账发现）：k 截断/覆盖时草稿行不得按 draft.block_size=16 计。

    默认场景 k=4 → 生效块 5：草稿 q_len/候选数/lm_head 行必须与
    verify/msg 同块（vLLM num_speculative_tokens / SGLang block_size 覆盖
    都按生效块执行草稿前向）。修复前草稿行按 16 计（双口径重复计费）。
    """
    b = run_v2(ScenarioV2())                    # k=4 → plan blk=5
    blk = b.method_plan["block_size"]
    assert blk == 5 and blk != 16
    rows = {r["op"]: r for r in b.work["draft_block"]["rows"]}
    B = b.scenario_snapshot["workload"]["batch"]
    q = [r for r in b.work["draft_block"]["rows"] if r["op"] == "q_proj"][0]
    # q_proj flops = 2·T·h·heads_c·d，T 必须 B×blk（生效块）而非 B×16
    heads_c = 32 // 8
    assert q["flops"] == 2 * (B * blk) * H * heads_c * D
    assert rows["draft_lm_head"]["flops"] ==         2 * B * (blk - 1) * H * (248320 // 8)
    # 块覆盖（k=7 → blk=8）同口径
    b8 = run_v2(ScenarioV2(k=7))
    assert b8.method_plan["block_size"] == 8
    q8 = [r for r in b8.work["draft_block"]["rows"] if r["op"] == "q_proj"][0]
    assert q8["flops"] == 2 * (B * 8) * H * heads_c * D
