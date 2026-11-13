"""M1 验证：ModelIR 结构回归 + 计数独立手算对账（V02/V03）。

独立期望书写规则：期望值在本测试内以独立算式推导（或引 golden/oracles_v02.py
的枚举），不 import 生产计数函数。数字对不上即失败，不做容差（整数精确）。
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simverse.model_ir import build_catalog, catalog_json  # noqa: E402
from simverse.count import build_model_work, totals  # noqa: E402
from simverse.count.moe import expected_union  # noqa: E402
from simverse.importers import source_map_json  # noqa: E402

_spec = importlib.util.spec_from_file_location("golden_v02", ROOT / "golden" / "oracles_v02.py")
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)

Q16 = {"dense_w": 16, "expert_w": 16, "act": 16, "kv": 16, "state": 32}
Q8 = {"dense_w": 8, "expert_w": 8, "act": 8, "kv": 8, "state": 32}


@pytest.fixture(scope="module")
def cat():
    # 历史模型/tiny 只作离线回归 fixture；默认运行时 catalog 只有两款主模型。
    return build_catalog(include_internal=True)


# ---------------------------------------------------------------------------
# V02 结构回归：四个历史模型 + 两条必需场景的层计划
# ---------------------------------------------------------------------------

def test_qwen3_30b_structure(cat):
    m = cat["models"]["qwen3_30b_a3b"]
    assert m.family == "qwen3_moe" and len(m.layers) == 48
    assert all(l.kind == "full_attention" for l in m.layers)
    assert all(l.moe and l.moe.experts == 128 and l.moe.top_k == 8 for l in m.layers)
    assert m.layers[0].attn.heads == 32 and m.layers[0].attn.kv_heads == 4
    # 2026-11-10 审查 F4：catalog revision 必须与 identity 全等
    assert m.source.revision == "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"


def test_qwen3_next_hybrid_schedule(cat):
    """计划锚点：48 层，全注意力层 0 起始 3,7,…,47（共 12），其余 GDN。"""
    m = cat["models"]["qwen3_next_80b_a3b"]
    full_idx = [l.index for l in m.layers if l.kind == "full_attention"]
    assert len(m.layers) == 48
    assert full_idx == list(range(3, 48, 4))
    assert len(full_idx) == 12
    assert m.layers[0].linear is not None and m.layers[0].linear.num_v_heads == 32
    assert m.layers[3].moe.experts == 512 and m.layers[3].moe.top_k == 10
    assert m.source.revision == "local-snapshot"  # 官方 revision 待补，如实标注


def test_deepseek_v3_mla_and_dense_moe_split(cat):
    """V3：61 层全 MLA；前 3 层 dense FFN，后 58 层 MoE（256E top-8+1shared）。"""
    m = cat["models"]["deepseek_v3"]
    assert len(m.layers) == 61
    dense = [l for l in m.layers if l.moe is None]
    moe = [l for l in m.layers if l.moe is not None]
    assert [l.index for l in dense] == [0, 1, 2]
    assert len(moe) == 58
    assert moe[0].moe.experts == 256 and moe[0].moe.top_k == 8
    assert moe[0].moe.shared_intermediate == 1 * 2048  # n_shared=1 × inter 2048
    # F3 根因修复（仅归档回归）：dense FFN 宽度 18432，不是 moe_intermediate 2048
    assert all(l.dense_inter == 18432 for l in dense)
    kv = m.extras["kv_spec"]
    assert kv["kv_elems_per_token_per_layer"] == 512 + 64   # kv_lora + rope
    # 历史模型只有 config 映射，无 forward 证据；不得再标结构完备
    assert m.coverage["intermediate_size"]["status"] == "mapped"
    assert not m.forward_evidence_complete and not m.structure_complete
    assert m.source.revision == "e815299b0bcbac849fa540c768ef21845365c9eb"


def test_v41_structure_and_gaps(cat):
    """V4.1：40 层 MLA+CSA2；hc 3 键 + Engram 8 键已映射（R04）→ 剩余缺口仍阻断。

    归因更新（2026-11-11 R04）：Engram 8 键曾为缺口（common/engram.py 未
    pin）；现按 engram.py pin（blob sha1 ba4ceea6 对 vllm-tree.json 校验）
    source-checked 建模 → mapped，CSA2/MoE/量化等缺口不变，
    structure_complete 仍 False（不冒充）。
    """
    m = cat["models"]["deepseek_v41_flash"]
    assert len(m.layers) == 40
    assert m.kind_counts() == {"mla_csa2": 40}
    assert not m.structure_complete          # 注意力核本体仍阻断（不冒充）
    # R24：swiglu_limit/尾部缺口解除后 config 覆盖完备；结构资格仍由
    # forward 证据（核本体 unverified_paths）阻断——两栏不得互相冒充。
    assert m.config_coverage_complete and not m.forward_evidence_complete
    assert m.evidence.forward_source == "partial-source"
    gap_keys = {g.split(":")[0] for g in m.gaps}
    assert not gap_keys & {"hc_mult", "hc_eps", "hc_sinkhorn_iters"}  # hc 已解除
    assert not gap_keys & {f"engram_{k}" for k in (
        "vocab_size", "num_embeddings", "layer_ids", "n_heads", "head_dim",
        "max_ngram_size", "compressed_vocab_size", "pad_token_id")}  # R04 已解除
    # R22：quantization_config 5 叶键已 mapped（quant-pin.json 20 文件），
    # 不再是 gap。
    # R24（2026-11-13）：swiglu_limit 与 compress_ratios[40:43] 尾部已按
    # swiglu-tail-pin.json 13 文件 source-checked 解除（合法门禁更新，
    # 非放宽）；structure_complete 仍由注意力核本体 unverified_paths 阻断。
    # R20（2026-11-12）：CSA2 索引/candidate 键已 mapped（csa2-pin.json 21
    # 文件 source-checked）；R24 补 candidate_block_size 入 struct 集合
    # （R20 注记此前因键漏登记未生效）。
    assert "text_config.swiglu_limit" not in gap_keys
    assert "quantization_config" not in gap_keys
    assert not gap_keys & {"candidate_block_size", "candidate_source_layer_id",
                           "candidate_topk_blocks", "index_topk"}
    assert not any("compress_ratios[" in g for g in m.gaps)
    assert m.coverage["text_config.swiglu_limit"]["status"] == "mapped"
    assert m.extras["swiglu"]["limit"] == 10.0
    # R21（2026-11-12）：MoE 路由主干已 source-checked（ev-vllm-v41-moe），
    # csa2_moe_routing 缺口改列 moe_deploy_variants（MegaMoE/DSpark/padding
    # skip 等部署变体仍 unverified）。
    assert any("moe_deploy_variants" in p for p in m.evidence.unverified_paths)
    assert any("MegaMoE" in p for p in m.evidence.unverified_paths)
    assert "sm-v41-moe-route" in m.evidence.checked_paths
    # hc 参数入册 extras（消费方 count/memory）
    hc = m.extras["hc"]
    assert hc["mult"] == 4 and hc["sinkhorn_iters"] == 20
    assert hc["eps"] == 1e-6 and hc["post_alpha"] == 2.0
    assert hc["engram_layer_ids"] == [1, 14]
    # Engram spec 入册 extras（R04；消费方 count/work.engram_rows、
    # lowering/memory engram 对象；常数见 ev-vllm-v41-engram）
    eng = m.extras["engram"]
    assert eng["layer_ids"] == [1, 14]
    assert eng["num_embeddings"] == [384006168, 384016682]
    assert eng["n_hash_cols"] == 24 and eng["max_ngram_size"] == 4
    assert eng["n_heads"] == 8 and eng["head_dim"] == 256
    assert eng["block_size"] == 32 and eng["bytes_per_row"] == 264
    assert eng["value_dtype"] == "fp8_e4m3" and eng["scale_encoding"] == "ue8m0"
    assert eng["vocab_size"] == 16000000 and eng["compressed_vocab_size"] == 99092
    assert eng["pad_token_id"] == 2 and eng["hc_mult"] == 4
    # 前 40 层 r 分布 0:2 / 2:18 / 1:20（主控已独立确认）
    ratios = [l.csa2.ratio for l in m.layers]
    assert ratios.count(0) == 2 and ratios.count(2) == 18 and ratios.count(1) == 20


def test_qwen36_required_scenario(cat):
    """必需场景②主模型：Qwen3.6-35B-A3B 混合层序 + MTP1 + 官方身份。"""
    m = cat["models"]["qwen36_35b_a3b"]
    assert m.family == "qwen3_5_moe" and len(m.layers) == 40
    assert m.kind_counts() == {"linear_attention": 30, "full_attention": 10}
    full_attn = next(l for l in m.layers if l.kind == "full_attention").attn
    assert full_attn.output_gated and full_attn.q_out == 2 * 16 * 256
    assert full_attn.rotary_dim == 64                    # 0.25 × 256
    assert m.mtp_layers == 1
    assert m.source.repo_id == "Qwen/Qwen3.6-35B-A3B"
    assert m.source.revision.startswith("345d467b")
    # config 全覆盖；但 MTP forward 未 pin → 不得宣称结构/forward 完备
    assert m.config_coverage_complete
    assert not m.forward_evidence_complete and not m.structure_complete
    assert m.evidence.forward_source == "partial-source"
    assert any("mtp_head" in p for p in m.evidence.unverified_paths)
    assert m.evidence.method_status["mtp"] == "unverified"
    # 已核分支保持 source-checked，不降格为 config-only
    assert "sm-qwen35-gdn-decode-branch" in m.evidence.checked_paths
    assert m.evidence.method_status["baseline"] == "source-checked"


def test_dflash_draft_structure(cat):
    """必需场景②草稿：DFlash v1 6 层（5 sliding + 1 full）、block 16、8 目标层。"""
    d = cat["drafts"]["dflash_qwen36_35b_a3b"]
    kinds = [l.kind for l in d.layers]
    assert kinds.count("sliding_attention") == 5 and kinds.count("full_attention") == 1
    assert d.block_size == 16 and d.mask_token_id == 248077
    assert list(d.target_layer_ids) == [1, 6, 11, 16, 22, 27, 32, 37]
    assert d.fc_in == 8 * 2048
    assert d.uses_target_lm_head and d.uses_target_embeddings
    assert d.source.revision.startswith("f181eece")


def test_every_config_key_has_coverage(cat):
    """M0 硬门槛：每个 pinned config 的键都有覆盖登记且状态合法。"""
    valid = {"mapped", "metadata-only", "ignored-with-reason",
             "unsupported-semantic", "pending-asset"}
    for mid in ("deepseek_v41_flash", "qwen36_35b_a3b", "qwen3_30b_a3b",
                "qwen3_next_80b_a3b", "deepseek_v3"):
        cov = cat["models"][mid].coverage
        assert cov, mid
        for key, entry in cov.items():
            assert entry["status"] in valid, (mid, key, entry)
        if mid == "deepseek_v41_flash":
            # R04（2026-11-11）：Engram 8 键从 unsupported → mapped
            # （common/engram.py pin source-checked；曾断言 len(eng)==8 为
            # unsupported——底层事实变化，随证据更新，非放宽）
            eng = [k for k, v in cov.items()
                   if k.split(".")[-1].startswith("engram_")]
            assert len(eng) == 8
            assert all(cov[k]["status"] == "mapped" for k in eng)
            for hk in ("hc_mult", "hc_eps", "hc_sinkhorn_iters"):
                assert cov[f"text_config.{hk}"]["status"] == "mapped"
    # 完整 config 的每个叶键都必须有登记（2026-11-10 F7）
    from simverse.model_ir.families import flatten_config
    for fname, mid, prefix in (
            ("qwen36_35b_a3b_main__config.json", "qwen36_35b_a3b", "text_config"),
            ("deepseek_v41_config.json", "deepseek_v41_flash", "text_config")):
        raw = json.loads((ROOT / "assets" / "config" / fname).read_text(encoding="utf-8"))
        leaves = {k for k, v in flatten_config(raw).items() if not isinstance(v, dict)}
        assert leaves <= set(cat["models"][mid].coverage), (mid, leaves - set(cat["models"][mid].coverage))


def test_v41_dspark_fields_are_method_scoped(cat):
    m = cat["models"]["deepseek_v41_flash"]
    ds = [k for k in m.coverage if k.split(".")[-1].startswith("dspark_")]
    assert ds and all(m.coverage[k]["consumer"] == "methods" for k in ds)


# ---------------------------------------------------------------------------
# V03 计数独立对账（期望值在本测试内独立推导）
# ---------------------------------------------------------------------------

def test_tiny_gqa_decode_totals_hand_derived(cat):
    """tiny_gqa decode（B=1,ctx=100,Q=1,16bit）全部总量手算对账。

    独立推导（与 v0.1 golden 同法，另加 v0.2 新增行：q/k 头维 norm、RoPE、
    残差加法 FLOPs——E09 修复后逻辑算量超集 5160）。
    """
    m = cat["models"]["tiny_gqa"]
    rows = build_model_work(m, "decode", 1, 100, 1, Q16, tp=1)
    t = totals(rows)
    h, heads, kv, d, ffn, V, L = 256, 8, 2, 32, 512, 1000, 2
    per_layer = (2 * h * heads * d + 2 * (2 * h * kv * d)
                 + (100 * 1 + 1) * heads * (4 * d + 5) + 2 * heads * d * h
                 + (4 * d + 2) * heads + (4 * d + 2) * kv      # q/k 头维 norm
                 + 3 * heads * d                                # rope
                 + 2 * h * (2 * ffn) + 2 * ffn * h              # gate_up + down
                 + 2 * (4 * h + 2)                              # attn/mlp norm
                 + 2 * h)                                       # 残差加（E09）
    # final norm（model.norm，2026-11-10 补齐缺行）：T·(4h+2) 每模型一次
    expect_flops = L * per_layer + 2 * h * V + (4 * h + 2)
    assert t["flops"] == expect_flops
    # q/k head-norm 的 scale 权重按 head_dim 共享（HF Qwen3RMSNorm(head_dim)），每层各 1 份
    # final norm 权重 h（2026-11-10 补齐）；16bit → 2 B/param
    expect_wr = L * ((h * heads * d + 2 * h * kv * d + heads * d * h
                      + h * 2 * ffn + ffn * h) * 2      # 16bit → 2 B/param
                     + 2 * (2 * h * 2) + 2 * (2 * d * 2)) + h * V * 2 + 2 * h * 2
    assert t["weight_read"] == expect_wr
    assert t["kv_read"] == L * 100 * (2 * kv * d) * 2
    assert t["kv_write"] == L * 1 * (2 * kv * d) * 2
    assert t["weight_read_expected"] == 0.0


def test_tiny_gqa_verify_query_scaling(cat):
    """验证步 Q=K+1=3：GEMM ×3、attention 对数 S·Q+Q(Q+1)/2=306（golden）。"""
    m = cat["models"]["tiny_gqa"]
    rows = build_model_work(m, "verify", 1, 100, 3, Q16, tp=1)
    core = next(r for r in rows if r.op == "attention_core" and r.layer == 0)
    assert core.flops == G.CAUSAL_S100_Q3 * 8 * (4 * 32 + 5)
    q = next(r for r in rows if r.op == "q_proj" and r.layer == 0)
    assert q.flops == 3 * (2 * 1 * 256 * 256)  # GEMM 2·T·IC·OC ×3 token


def test_tiny_moe_union_expectation_exact_split(cat):
    """MoE 行：期望并集权重读为浮点单列；精确 FLOPs/容量为整数（E05 修复）。"""
    m = cat["models"]["tiny_moe"]
    rows = build_model_work(m, "decode", 1, 0, 1, Q8, tp=1)  # ctx=0 只看 MoE 行
    routed = next(r for r in rows if r.op == "moe_routed")
    assert routed.weight_read == 0
    assert routed.weight_read_expected == pytest.approx(2.0 * 3 * 256 * 512 * 1.0)
    assert routed.flops == 2 * 1 * 2 * 3 * 256 * 512        # 精确整数
    assert routed.weight_capacity == 4 * 3 * 256 * 512      # 全专家驻留
    assert routed.statistic == "expectation"
    # 独立公式检查：正确式在 M=1 恰为 k；有放回近似在 M=1 < k（golden 端点）
    assert expected_union(64, 8, 1) == G.MOE_M1_GIVES_K == 8.0
    assert G.MOE_M1_WITH_REPL_LT_K < 8.0
    assert expected_union(4, 2, 2) == G.MOE_E4_K2_M2_ENUM == 3.0
    assert expected_union(384, 6, 5) == pytest.approx(G.MOE_E384_K6_M5_RATIO, rel=1e-12)


def test_hybrid_gdn_state_is_constant_not_kv(cat):
    """GDN 层：无 per-token KV；递推状态读写/容量单列（结构证据 L558/607）。"""
    m = cat["models"]["tiny_hybrid"]
    rows = build_model_work(m, "decode", 1, 1000, 1, Q8, tp=1)
    lin = [r for r in rows if r.layer == 0 and r.op in
           ("gdn_recurrence", "causal_conv1d", "in_proj_qkv")]
    assert lin, "GDN 层行缺失"
    rec = next(r for r in rows if r.op == "gdn_recurrence" and r.layer == 0)
    g = m.layers[0].linear
    # 独立推导：state_elems = N_v·d_k·d_v = 8·32·32 = 8192；fp32 → 32768 B/步
    assert g.state_elems == 8 * 32 * 32
    assert rec.state_read == 1 * 8192 * 4 and rec.state_write == 1 * 8192 * 4
    assert rec.state_capacity == 8192 * 4 + (2 * 128 + 256) * 3 * 1  # + conv 状态
    kv_rows = [r for r in rows if r.layer == 0 and (r.kv_read or r.kv_write)]
    assert not kv_rows, "LINEAR 层不得出现 per-token KV 行"


def test_full_attention_layer_kv_only_on_full_layers(cat):
    qwen = cat["models"]["qwen36_35b_a3b"]
    rows = build_model_work(qwen, "decode", 1, 1024, 1, Q8, tp=1)
    kv_by_layer = {}
    for r in rows:
        if r.kv_read:
            kv_by_layer.setdefault(r.layer, 0)
            kv_by_layer[r.layer] += r.kv_read
    full_idx = {l.index for l in qwen.layers if l.kind == "full_attention"}
    assert set(kv_by_layer) == full_idx
    # 独立：每 full 层 kv_read = 1024 行 × 2·2·256 elem × 1B = 524288
    assert all(v == 1024 * 2 * 2 * 256 * 1 for v in kv_by_layer.values())


def test_mla_kv_compressed_layout(cat):
    """V3 MLA：KV 读按压缩布局 (kv_lora+rope)/token；tiny_mla 独立对账。"""
    m = cat["models"]["tiny_mla"]
    rows = build_model_work(m, "decode", 1, 100, 1, Q16, tp=1)
    core = next(r for r in rows if r.op == "mla_absorbed_core" and r.layer == 0)
    assert core.kv_read == 100 * (128 + 32) * 2          # golden MLA 元素式
    assert core.kv_write == 1 * (128 + 32) * 2


def test_sliding_attention_window_pairs(cat):
    """SLIDING 层可见对数受窗口封顶（causal_pairs 闭式独立核对）。"""
    from simverse.count.work import causal_pairs
    win = 4096  # DFlash 草稿 sliding_window（config pin）
    assert causal_pairs(100, 1, win) == 100 * 101 // 2        # 未触顶
    assert causal_pairs(10**6, 1, win) == _manual_causal(10**6, win)


def _manual_causal(n: int, win: int) -> int:
    """独立暴力式：Σ min(t, win)。"""
    return sum(min(t, win) for t in range(1, n + 1))


def test_v41_counting_matches_v01_transplant(cat):
    """V4.1 CSA2：v0.2 行集合覆盖已对账注意子图（R20 起 SWA 行按 pinned 记录宽度）。"""
    m = cat["models"]["deepseek_v41_flash"]
    rows = build_model_work(m, "decode", 1, 32768, 1,
                            {"dense_w": 8, "expert_w": 4, "act": 8, "kv": 8,
                             "state": 32}, tp=8)
    kvs = m.extras["kv_spec"]
    # 独立重算 layer 0 sparse_attention（全 SWA r=0）：heads(card)=128/8=16
    l0 = next(r for r in rows if r.op == "sparse_attention" and r.layer == 0)
    from simverse.count.work import causal_pairs
    swa = kvs["sliding_window"]
    local = causal_pairs(32769, 1, swa) - causal_pairs(32768, 1, swa)
    heads_card = kvs["heads"] // 8      # config num_attention_heads=64
    assert kvs["heads"] == 64
    # R20：SWA 行按打包记录 584 B/token（fp8_ds_mla_sm90，attention.py
    # L583-589），不再 d×kv 位宽
    assert l0.kv_read == 1 * min(32769, swa) * 584
    assert l0.flops == 1 * local * heads_card * (4 * kvs["head_dim"] + 5)


def test_qwen36_active_params_ballpark(cat):
    """量级合理性（非精确断言）：implied_active = decode FLOPs/2 落在官方
    A3B 口径（模型名，≈3.5B，假设登记）±25% 内；注意力项解释超出部分。
    """
    m = cat["models"]["qwen36_35b_a3b"]
    rows = build_model_work(m, "decode", 1, 1024, 1, Q8, tp=1)
    t = totals(rows)
    implied_active = t["flops"] / 2
    official_a3b = 3.5e9   # 假设：官方命名 A3B 的量级口径（非 config 字段）
    assert 0.75 * official_a3b < implied_active < 1.25 * official_a3b,         f"implied={implied_active}"


# ---------------------------------------------------------------------------
# V02 源码映射
# ---------------------------------------------------------------------------

def test_source_map_entries_present():
    sm = source_map_json()
    ids = {e["entry_id"] for e in sm["entries"]}
    assert {"sm-qwen35-gdn-decode-branch", "sm-v41-engram-hc",
            "sm-dflash-round-protocol"} <= ids
    for e in sm["entries"]:
        assert e["source_pin"] and e["source_location"]
        assert e["branch_predicate"], e["entry_id"]
    # 未覆盖域如实标注
    engram = next(e for e in sm["entries"] if e["entry_id"] == "sm-v41-engram-hc")
    assert engram["both_sides_covered"] is False and engram["uncovered_domain"]


def test_catalog_json_serializable_and_pending_assets():
    cj = catalog_json()
    assert cj["pending_assets"]["qwen3_8_local"]["status"] == "pending-asset"
    import json
    from simverse.contracts import js_safe, js_restore
    payload = js_safe(cj)
    json.dumps(payload)   # 可序列化
    assert js_restore(payload)["model_ir_version"] == "0.3.0"
