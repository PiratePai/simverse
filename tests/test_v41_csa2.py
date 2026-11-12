"""V4.1 CSA2 索引打分/top-k/candidate block（R20，2026-11-12）。

证据链：csa2-pin.json 21 文件 pin（vllm @f358f057，blob sha1 对
vllm-tree.json 逐项校验；含 sparse_attn_indexer 直接 import 的 v1 backend；
source-checked，kernel/CPU/GPU 均未执行）。
测试分层（不 import 生产公式复算的地方均独立推导）：
  * pin 完整性：21 文件固定路径集合 + blob sha1（防清单删项/漂移）；
  * 拓扑 oracle：从官方 config 独立推导 kv/index/candidate 源与
    SparseIndexTables 等键运行首层，对照 Csa2IR 与账本行；
  * 分支 oracle：短上下文旁路（r=1/r=2 阈值、混合 ctx）两侧行集合；
  * 打分/top-k/candidate 语义：stdlib 独立实现 select/mask/fill 语义
    （小尺寸），再对真实常数断言账本行公式与字节；
  * 打包记录/环形 state/缓冲下界与 memory 账本逐项对账；
  * replay 归因：docs/evidence/csa2 前后冻结（审计面外行逐字段不变）。
"""
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simverse.count.work import build_model_work, totals  # noqa: E402
from simverse.lowering.memory import (kv_allocations,  # noqa: E402
                                      target_weight_allocations,
                                      v41_index_buffer_allocations)
from simverse.model_ir.catalog import build_catalog  # noqa: E402
from simverse.scenario import (DeploymentSpec, ScenarioV2,  # noqa: E402
                               V41_INDEX_K_BYTES, V41_KV_RECORDS,
                               V41_QUANT, V41_RING_CAPACITY)

Q8 = {"dense_w": 8, "expert_w": 4, "act": 8, "kv": 8, "state": 32}
Q16 = {"dense_w": 16, "expert_w": 16, "act": 16, "kv": 16, "state": 32}
BASE = ROOT / "docs/rebuild-20261106/references"
EVID = ROOT / "docs/evidence/csa2"

_CAT = build_catalog(include_internal=True)

# 官方 config（identity 固定快照；与 families/catalog 同一来源）
_CFG = json.loads(
    (BASE / "official-v41-config.json").read_text(encoding="utf-8"))["text_config"]

_KV_SRC = _CFG["kv_source_layer_ids"]          # [2, 8, 14, 20]
_IX_SRC = _CFG["index_source_layer_ids"]       # [2, 8, 14, 20, 24, 28, 32, 36]
_CAND = _CFG["candidate_source_layer_id"]      # 20
_RATIOS = _CFG["compress_ratios"][:_CFG["num_hidden_layers"]]
_TOPK = _CFG["index_topk"]                     # 512
_CTB = _CFG["candidate_topk_blocks"]           # 2048
_CB = _CFG["candidate_block_size"]             # 8
_IH, _DI, _D, _RD = _CFG["index_n_heads"], _CFG["index_head_dim"], \
    _CFG["head_dim"], _CFG["qk_rope_head_dim"]
_SWA = _CFG["sliding_window"]


def _model():
    return _CAT["models"]["deepseek_v41_flash"]


def _rows(B, ctx, Q=1, prefill=False, tp=8, quant=None, rec="fp8_ds_mla_sm90"):
    m = _model()
    return build_model_work(m, "target_decode" if not prefill else "prefill",
                            B, ctx, Q, quant or Q8, tp, prefill=prefill,
                            v41_kv_record=rec)


def _layer_rows(rows, li):
    return [r for r in rows if r.layer == li]


# ---------------------------------------------------------------------------
# 1. pin 完整性
# ---------------------------------------------------------------------------

def test_csa2_pin_blob_sha1_matches_tree():
    pin = json.loads((BASE / "vllm/csa2-pin.json").read_text(encoding="utf-8"))
    tree = {x["path"]: x["sha"] for x in json.loads(
        (BASE / "vllm-tree.json").read_text(encoding="utf-8"))["tree"]}
    pinned = {f["path"] for f in pin["files"]}
    expected = {
        "vllm/config/attention.py",
        "vllm/model_executor/kernels/attention/dsa/candidate_blocks.py",
        "vllm/model_executor/layers/indexer_topk.py",
        "vllm/model_executor/layers/sparse_attn_indexer.py",
        "vllm/models/deepseek_v4/attention.py",
        "vllm/models/deepseek_v41/attention.py",
        "vllm/models/deepseek_v41/common/ops/__init__.py",
        "vllm/models/deepseek_v41/common/ops/cache_utils.py",
        "vllm/models/deepseek_v41/common/ops/fused_compress_quant_cache.py",
        "vllm/models/deepseek_v41/common/ops/indexer_k_store.py",
        "vllm/models/deepseek_v41/compressor.py",
        "vllm/models/deepseek_v41/nvidia/specialized_decode/attention/"
        "aux_projections.py",
        "vllm/models/deepseek_v41/nvidia/specialized_decode/attention/"
        "candidate_topk.py",
        "vllm/models/deepseek_v41/nvidia/specialized_decode/attention/"
        "decode_topk_cute.py",
        "vllm/models/deepseek_v41/nvidia/specialized_decode/attention/"
        "indexer.py",
        "vllm/models/deepseek_v41/nvidia/specialized_decode/attention/"
        "indexer_k_store.py",
        "vllm/models/deepseek_v41/nvidia/specialized_decode/attention/"
        "sparse_mla.py",
        "vllm/models/deepseek_v41/nvidia/specialized_decode/attention/step.py",
        "vllm/models/deepseek_v41/sparse_mla.py",
        "vllm/v1/attention/backends/mla/indexer.py",
        "tests/models/deepseek_v41/nvidia/specialized_decode/test_decode_topk.py",
    }
    assert pinned == expected, "pin 清单漂移"
    # 上游 tree 中不存在该路径；此前误入目录的重复副本不应留在 pin 目录
    assert not (BASE / "vllm/vllm/model_executor/layers/"
                "candidate_blocks.py").exists()
    for f in pin["files"]:
        data = (BASE / "vllm" / f["path"]).read_bytes()
        sha1 = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
        assert sha1 == f["blob_sha1"] and len(data) == f["size"]
        assert tree[f["path"]] == sha1          # 对上游 tree 逐项校验
        assert f["sha256"] == hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# 2. 拓扑 oracle（从 config 独立推导，不信生产解析）
# ---------------------------------------------------------------------------

def test_topology_oracle_from_config():
    m = _model()
    kv_spec = m.extras["kv_spec"]
    assert kv_spec["kv_source_layer_ids"] == _KV_SRC
    assert kv_spec["index_source_layer_ids"] == _IX_SRC
    assert kv_spec["candidate_source_layer_id"] == _CAND
    assert kv_spec["candidate_block_size"] == _CB
    for layer in m.layers:
        r = _RATIOS[layer.index]
        a = layer.csa2
        assert a.ratio == r
        assert a.produces_kv == (layer.index in _KV_SRC)
        assert a.produces_index == (layer.index in _IX_SRC)
        assert a.produces_candidates == (layer.index == _CAND)
        if r:
            # 源解析：下方最近源（attention.py L414-419）
            assert a.kv_source == max(x for x in _KV_SRC if x <= layer.index)
            assert a.index_source == max(x for x in _IX_SRC if x <= layer.index)
        else:
            assert a.kv_source is None and a.index_source is None


def test_sparse_index_runs_and_candidate_ownership():
    """SparseIndexTables 等键运行首 + candidate 所有权（源写、>源消费）。"""
    rows = _rows(B=1, ctx=32768)
    # 等键运行首：key(layer) != key(prev)；key = r==0→"swa"，否则
    # (kv_source, index_source, r)（specialized sparse_mla.py L159-250）
    keys = []
    for layer in _model().layers:
        a = layer.csa2
        keys.append("swa" if not a.ratio else (a.kv_source, a.index_source, a.ratio))
    run_first = [i for i, k in enumerate(keys) if i == 0 or k != keys[i - 1]]
    build_layers = sorted(r.layer for r in rows if r.op == "sparse_index_build")
    assert build_layers == run_first
    # V4.1 实际运行首：L0(swa)、L2/L8/L14(r2 三源)、L20/L24/L28/L32/L36
    assert build_layers == [0, 2, 8, 14, 20, 24, 28, 32, 36]
    # candidate 所有权：candidate_select 仅 L20；candidate_mask 仅 >20 的
    # index source（24/28/32/36）
    assert [r.layer for r in rows if r.op == "candidate_select"] == [_CAND]
    assert [r.layer for r in rows if r.op == "candidate_mask"] == \
        [x for x in _IX_SRC if x > _CAND]
    # 非源压缩层（如 L3/L21）无任何 indexer 行——只消费 topk（sparse_attention）
    for li in (3, 21):
        lr = _layer_rows(rows, li)
        assert not any(r.op.startswith(("index_", "candidate_",
                                        "sparse_index")) for r in lr)


# ---------------------------------------------------------------------------
# 3. 短上下文旁路（两侧分支）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ctx,r_short,r_full", [
    # decode end = ctx+1（含新 token）：r1 旁路需 end≤512、r2 需 end//2≤512
    (500, {1, 2}, set()),        # 两种比例都旁路（501//1=501、250 ≤ 512）
    (512, {2}, {1}),             # 513>512 全打分、513//2=256 旁路
    (800, {2}, {1}),             # 混合：r2 旁路（400≤512）、r1 全打分
    (32768, set(), {1, 2}),      # 都全打分
])
def test_short_context_bypass_branches(ctx, r_short, r_full):
    rows = _rows(B=1, ctx=ctx)
    for li in _IX_SRC:
        lr = {r.op for r in _layer_rows(rows, li)}
        r = _RATIOS[li]
        n = (ctx + 1) // r                    # end = ctx+1（含新 token）
        short = n <= _TOPK
        assert short == (r in r_short), (li, r, ctx)
        if short:
            assert "index_fill" in lr and "index_score" not in lr
            assert "index_q_rope" not in lr and "index_topk" not in lr
            assert "candidate_select" not in lr and "candidate_mask" not in lr
            fill = next(r_ for r_ in _layer_rows(rows, li)
                        if r_.op == "index_fill")
            assert fill.flops == 0            # 纯 store（L77-92）
            assert fill.act_write == 1 * _TOPK * 4
        else:
            assert "index_score" in lr and "index_topk" in lr
            assert "index_fill" not in lr
        # weights_proj 恒执行（aux 投影在旁路判定之前发起）
        assert "index_weights_proj" in lr
        # wq_b 行恒在（旁路时 0 容量占位，与 R05 冻结口径一致）
        qp = next(r_ for r_ in _layer_rows(rows, li) if r_.op == "index_q_proj")
        assert (qp.weight_capacity > 0) == (not short)


def test_short_context_fill_semantics_independent():
    """fill 语义独立复刻：indices 0..n−1 + (−1) 填充（n=(pos+1)//r）。"""
    def fill(pos, r, topk):
        n = (pos + 1) // r
        return list(range(min(n, topk))) + [-1] * max(0, topk - min(n, topk))
    assert fill(9, 2, 512)[:6] == [0, 1, 2, 3, 4, -1]
    assert fill(1023, 2, 512)[-1] == 511      # n=512 恰满
    assert fill(1024, 2, 512) == list(range(512))  # n=512（(1024+1)//2=512）


# ---------------------------------------------------------------------------
# 4. 打分/top-k/candidate 账本行独立复算（真实常数）
# ---------------------------------------------------------------------------

def test_decode_score_topk_rows_oracle():
    """L2（r2）/L20（r1, 候选源）逐行独立复算（不 import 生产函数）。"""
    B, S = 2, 3000          # r2: 3001//2=1500 > 512 全打分
    rows = _rows(B=B, ctx=S, Q=1)
    for li, r in ((2, 2), (20, 1)):
        n = (S + 1) // r                       # 每行有效压缩位（end//r）
        pairs = B * n                          # decode：每行扫全部历史
        score = next(x for x in _layer_rows(rows, li) if x.op == "index_score")
        assert score.flops == 2 * pairs * (_IH * _DI + _IH)
        assert score.kv_read == B * n * V41_INDEX_K_BYTES
        assert score.act_write == B * n * 4    # fp32 logits（sm90）
        topk = next(x for x in _layer_rows(rows, li) if x.op == "index_topk")
        assert topk.flops == 2 * pairs + B * _TOPK
        assert topk.act_read == B * n * 4
        assert topk.act_write == B * _TOPK * 4


def test_prefill_score_pairs_oracle():
    """prefill：pairs = Σ_t floor(t/r)（因果历史压缩位）；旁路阈值同谓词。"""
    ctx = 2048
    rows = _rows(B=1, ctx=ctx, Q=ctx, prefill=True)
    li, r = 20, 1
    n = ctx // r
    # _sum_floor(n, r) = Σ_{t=1..n} floor(t/r)（独立求和对照）
    manual = sum(t // r for t in range(1, ctx + 1))
    score = next(x for x in _layer_rows(rows, li) if x.op == "index_score")
    assert score.flops == 2 * manual * (_IH * _DI + _IH)
    # gather 工作区按一次读/写计（不按每个 query 行重复）
    assert score.kv_read == 1 * n * V41_INDEX_K_BYTES
    # prefill 的 logits 按有效 (query, candidate) 对 pairs 写，而非仅 B 行
    assert score.act_read == 1 * n * V41_INDEX_K_BYTES
    assert score.act_write == 1 * n * V41_INDEX_K_BYTES + manual * 4
    topk = next(x for x in _layer_rows(rows, li) if x.op == "index_topk")
    assert topk.act_read == manual * 4
    assert topk.act_write == ctx * _TOPK * 4


def test_prefill_candidate_rows_oracle():
    """prefill candidate select/mask 按 tokens 行计（不是 decode 的 B 行）。"""
    B, ctx = 1, 2048
    rows = _rows(B=B, ctx=ctx, Q=ctx, prefill=True)
    r = 1
    n = ctx // r
    manual = sum(t // r for t in range(1, ctx + 1))
    nblocks = math.ceil(n / _CB)
    sel = next(x for x in _layer_rows(rows, _CAND) if x.op == "candidate_select")
    assert sel.act_read == manual * 4
    assert sel.act_write == ctx * (nblocks * 4 + 4 + _CTB * 4)
    assert sel.flops == ctx * (7 * nblocks + 2 * nblocks + _CTB)
    msk = next(x for x in _layer_rows(rows, 24) if x.op == "candidate_mask")
    assert msk.act_read == ctx * _CTB * 4
    assert msk.act_write == manual * 4


# ---- candidate 语义：stdlib 独立实现（select/mask/fill 的小尺寸对照）----

def _select_candidate_blocks_ref(logits_row, end, block, k):
    """candidate_blocks.py L31-73/L168-215 语义独立复刻（fp32 行）。

    排序键：NaN 最高（_max_with_nan 传播 + torch.topk 把 NaN 排最前；
    decode_topk 的 order_key 同样把 NaN 键放在 +inf 之上、cooperative
    "pinned 除非 ≥k 个 NaN 块"一致），其次钉扎的 +inf，再按值。
    """
    nblocks = math.ceil(end / block)
    scores = []
    for b in range(nblocks):
        seg = logits_row[b * block:min((b + 1) * block, end)]
        v = max(seg) if seg else float("-inf")
        if any(x != x for x in seg):           # NaN 传播
            v = float("nan")
        if b == (end - 1) // block and end > 0:
            v = float("inf")                   # 最新块钉 +inf
        scores.append(v)
    def rank(i):
        v = scores[i]
        return (2 if v != v else 0) + (1 if v == float("inf") else 0),             (0.0 if v != v else v)
    order = sorted(range(nblocks), key=rank, reverse=True)
    out = []
    for i in order[:k]:
        out.append(i if scores[i] != float("-inf") else -1)
    return out + [-1] * (k - len(out))


def _apply_candidate_mask_ref(logits_row, end, cand, block):
    """candidate_blocks.py L75-136 语义：候选外/因果外 → -inf。"""
    flags = {c for c in cand if c >= 0}
    row = list(logits_row)
    for col in range(len(row)):
        in_causal = col < end
        keep = (col // block) in flags if in_causal else False
        if not (in_causal and keep):
            row[col] = float("-inf")
    return row


def test_candidate_select_mask_semantics_independent():
    """小尺寸随机行：select 与 mask 的独立语义（钉扎/NaN/-1/-inf）。"""
    import random
    rnd = random.Random(7)
    end, block, k = 40, 8, 4
    row = [rnd.uniform(-2, 2) for _ in range(end)] + [0.0] * 8
    row[3] = float("nan")                      # 块 0 NaN 传播 → 排最前
    row[end - 1] = -100.0                      # 最新块最低但被钉 +inf
    cand = _select_candidate_blocks_ref(row, end, block, k)
    # NaN 块（0）排最前入选；钉扎的最新块（(40-1)//8=4）非 -1 且入选；
    # 其余取幸存块最大值 top（k−2 个，值互异无并列）
    assert 0 in cand and 4 in cand
    scores = [max(row[b * block:(b + 1) * block]) for b in range(1, 4)]
    best2 = sorted(range(1, 4), key=lambda i: scores[i - 1], reverse=True)[:2]
    assert sorted(c for c in cand if c >= 0) == sorted({0, 4, *best2})
    # 全 -inf 行（end 之前整段 -inf）：仅钉扎块入选
    dead = [float("-inf")] * end
    assert _select_candidate_blocks_ref(dead, end, block, k) == [4, -1, -1, -1]
    # mask：候选块内保留、块外/因果外 -inf（NaN 值经 NaN==NaN 恒 False，
    # 用同值判定含 NaN）
    def same(a, b):
        return (a != a and b != b) or a == b
    masked = _apply_candidate_mask_ref(row, end, cand, block)
    kept = {c for c in cand if c >= 0}
    for col in range(len(row)):
        keep = (col // block) in kept and col < end
        assert same(masked[col], row[col]) == keep
    # 负例：候选含 -1（padding）不掩掉任何块
    m2 = _apply_candidate_mask_ref(row, end, [-1] * k, block)
    assert all(v == float("-inf") for v in m2[:end])


def test_candidate_rows_oracle():
    """candidate_select/mask 行的字节与 op 公式（真实常数）。"""
    B, S = 1, 32768
    rows = _rows(B=B, ctx=S, Q=1)
    n = (S + 1) // 1
    nblocks = math.ceil(n / _CB)
    sel = next(x for x in _layer_rows(rows, _CAND) if x.op == "candidate_select")
    assert sel.flops == B * (7 * nblocks + 2 * nblocks + _CTB)
    assert sel.act_read == B * n * 4
    assert sel.act_write == B * (nblocks * 4 + 4 + _CTB * 4)
    msk = next(x for x in _layer_rows(rows, 24) if x.op == "candidate_mask")
    assert msk.flops == B * (4 * n + _CTB + nblocks)
    assert msk.act_read == B * _CTB * 4
    assert msk.act_write == B * n * 4


# ---------------------------------------------------------------------------
# 5. 打包记录 / 环形 state / 权重 bf16 / memory 账本
# ---------------------------------------------------------------------------

def test_sparse_index_build_width_alignment_128():
    """combine_topk_swa_indices 按 128 对齐（不是 256）；SWA-only topk=0。"""
    rows = _rows(B=1, ctx=32768)
    build = {r.layer: r for r in rows if r.op == "sparse_index_build"}
    # L0 为 swa-only：topk=0，仅 min(ctx,swa)=128 项 → 128 对齐宽
    assert build[0].act_write == 1 * (128 * 4 + 4)
    # L2/L20 等压缩运行首：topk 512 + 窗 128 = 640（已对齐）
    assert build[2].act_write == 1 * (640 * 4 + 4)
    for li, row in build.items():
        r = _RATIOS[li]
        win = min(32769, _SWA)
        w_c = -(-((_TOPK if r else 0) + win) // 128) * 128
        assert row.act_write == 1 * (w_c * 4 + 4), li


def test_bf16_module_activation_bytes_producer_consumer():
    """quant_config=None 的 compressor/wk/weights_proj 输入输出均 bf16；
    wq_b 输出 bf16 与 index_q_rope 读取对齐（不随场景 act 缩放）。"""
    B, S = 2, 3000          # r2: 1500>512、r1: 3001>512，均非旁路
    rows = _rows(B=B, ctx=S)
    T = B
    h, d, di, ih, qrank = 5120, 512, 128, 32, 1280
    for li in _KV_SRC:
        r = _RATIOS[li]
        mlt = 2 if r == 2 else 1
        cp = next(x for x in _layer_rows(rows, li) if x.op == "compress_proj")
        assert cp.act_read == T * h * 2
        assert cp.act_write == T * d * mlt * 4
        kp = next(x for x in _layer_rows(rows, li) if x.op == "index_k_proj")
        assert kp.act_read == T * d * 2 and kp.act_write == T * di * 2
    for li in _IX_SRC:
        wp = next(x for x in _layer_rows(rows, li)
                  if x.op == "index_weights_proj")
        assert wp.act_read == T * h * 2 and wp.act_write == T * ih * 2
        qp = next(x for x in _layer_rows(rows, li) if x.op == "index_q_proj")
        assert qp.act_write == T * ih * di * 2
        qr = next(x for x in _layer_rows(rows, li) if x.op == "index_q_rope")
        assert qr.act_read == T * ih * di * 2 + T * ih * 2


def test_packed_record_widths_pinned():
    assert V41_KV_RECORDS["fp8_ds_mla_sm90"]["swa_bytes"] == 584
    assert V41_KV_RECORDS["fp8_ds_mla_sm90"]["compressed_bytes"] == 584
    assert V41_KV_RECORDS["fp8_ds_mla_sm90"]["family"] == "sm90"
    assert V41_KV_RECORDS["fp8_ds_mla_sm100"]["swa_bytes"] == 528
    assert V41_KV_RECORDS["fp8_ds_mla_sm100"]["family"] == "sm100"
    assert V41_KV_RECORDS["nvfp4_ds_mla_sm100"]["compressed_bytes"] == 288
    assert V41_KV_RECORDS["nvfp4_ds_mla_sm100"]["family"] == "sm100"
    assert V41_INDEX_K_BYTES == 132
    # 584 = 448 fp8 + 64 bf16 rope + 7×ue8m0 + pad（attention.py L196-199）
    assert 448 + 128 + 7 + 1 == 584 or 448 + 64 * 2 + 8 == 584
    # 528 = 512 + 16×ue8m0(32dim)；288 = 256 e2m1 + 32 e4m3(16dim)
    assert 512 + 16 == 528 and 256 + 32 == 288
    # 索引 K 132 = 128 fp8 + 4 scale（L189）
    assert 128 + 4 == 132


def test_kv_allocations_records_and_ring_oracle():
    m = _model()
    B, ctx = 2, 32768
    allocs = kv_allocations(m, Q8, B, ctx, tp=8)
    (kv,) = [a for a in allocs if a.kind == "kv"]
    total = 0
    for i, r in enumerate(_RATIOS):
        total += min(ctx, _SWA) * 584
        if i in _KV_SRC:
            total += (ctx // r) * 584 + (ctx // r) * 132
            if r > 1:
                total += V41_RING_CAPACITY * 2 * _D * 4   # ring cap8×4096B fp32
    assert kv.bytes == total * B
    # sm100 记录族只改字节不改结构；nvfp4 只改压缩缓存
    a100 = kv_allocations(m, Q8, B, ctx, tp=8,
                          v41_kv_record="nvfp4_ds_mla_sm100")
    (kv4,) = [a for a in a100 if a.kind == "kv"]
    delta = 0
    for i, r in enumerate(_RATIOS):
        delta += min(ctx, _SWA) * (528 - 584)
        if i in _KV_SRC:
            delta += (ctx // r) * (288 - 584)
    assert kv4.bytes - kv.bytes == delta * B


def test_compressor_bf16_weights_independent_of_quant():
    """R22：V4.1 全部 dense 字节由 extras.quant 配方驱动（quant 位宽 inert）。

    compressor/wk/weights_proj 为 quant_config=None bf16（R20 事实不变）；
    量化模块按 MXFP8 1B + ue8m0 scale。Q8/Q16 传入不再改变任何 V4.1 字节。
    """
    m = _model()
    tp = 8
    w8 = sum(a.bytes for a in target_weight_allocations(m, Q8, tp)
             if a.alloc_id.endswith("-w-dense-L2"))
    w16 = sum(a.bytes for a in target_weight_allocations(m, Q16, tp)
              if a.alloc_id.endswith("-w-dense-L2"))
    assert w16 == w8, "V4.1 计价已配方冻结，quant 位宽不得改变字节"
    h, d, di, ih, qrank, orank, groups = 5120, 512, 128, 32, 1280, 1024, 8
    heads_tp, ng = 64 // tp, groups // tp
    scale = lambda n, k: n * (k // 32)
    quantized = (h * qrank + scale(qrank, h)              # fused_wqa_wkv（q_a）
                 + h * d + scale(d, h)                    # fused_wqa_wkv（kv）
                 + qrank * heads_tp * d + scale(heads_tp * d, qrank)  # 主 wq_b
                 + (64 * d // groups) * orank * ng
                 + ng * scale(orank, 64 * d // groups)    # o_a
                 + ng * orank * h + scale(h, ng * orank)  # o_b
                 + qrank * ih * di + scale(ih * di, qrank))  # indexer wq_b
    bf16_fixed = (2 * h * d * 2                            # compressor（ratio2）
                  + 2 * d * di + 2 * (d + di)
                  + 2 * h * ih                             # weights_proj/wk/norm
                  + 2 * (qrank + d))                       # 主 q_norm/kv_norm
    assert w8 == quantized + bf16_fixed, (w8, quantized + bf16_fixed)


def test_main_attention_norm_weight_capacity_and_read():
    """q_norm(q_lora_rank)/kv_norm(head_dim) 为 bf16 常驻，独立对账。

    单层：q_norm 1280×2=2560 B、k_norm 512×2=1024 B（共 3584 B）；
    整模型 40 层 = 143,360 B。weight_read/weight_capacity 不随场景 act
    位宽变化；整模型 memory dense 用独立公式复算并包含该固定项。
    """
    m = _model()
    B, S = 2, 3000
    rows8 = _rows(B=B, ctx=S, quant=Q8)
    rows16 = _rows(B=B, ctx=S, quant=Q16)
    qn8 = [r for r in rows8 if r.op == "q_norm"]
    kn8 = [r for r in rows8 if r.op == "k_norm"]
    assert len(qn8) == 40 and len(kn8) == 40
    assert all(r.weight_read == r.weight_capacity == 1280 * 2 for r in qn8)
    assert all(r.weight_read == r.weight_capacity == 512 * 2 for r in kn8)
    total8 = sum(r.weight_read for r in qn8 + kn8)
    total16 = sum(r.weight_read for r in rows16
                  if r.op in ("q_norm", "k_norm"))
    assert total8 == total16 == 40 * (1280 * 2 + 512 * 2) == 143_360

    # 整模型 memory 独立复算（TP8/Q8）：所有 dense 项 + 主 norm/compressor/
    # indexer bf16 固定项；不调用生产 target_weight_allocations 的公式。
    h, heads, d, groups, orank = 5120, 64, 512, 8, 1024
    qrank, di, ih, tp = 1280, 128, 32, 8
    ng, heads_tp = groups // tp, heads // tp
    expect = 0
    for layer in m.layers:
        a = layer.csa2
        mlt = 2 if a.ratio == 2 else 1
        terms = (h * qrank + h * d + qrank * heads_tp * d
                 + (heads * d // groups) * orank * ng
                 + ng * orank * h
                 + 2 * (qrank + d))              # 主 q_norm + kv_norm bf16
        if a.produces_kv:
            terms += (2 * h * d * mlt            # compressor bf16
                      + 2 * d * di               # wk bf16
                      + 2 * (d + di))            # compressor norm + k_norm
        if a.produces_index:
            terms += (2 * h * ih                 # weights_proj bf16
                      + qrank * ih * di)         # indexer wq_b（MXFP8 值）
        # R22：量化项并入 ue8m0 scale（N×K/32；与 target_weight_allocations
        # 的 _q_scale 同式，此处为独立复算）
        scale = lambda n, k: n * (k // 32)
        terms += (scale(qrank, h) + scale(d, h) + scale(heads_tp * d, qrank)
                  + scale(orank, heads * d // groups) * ng
                  + scale(h, ng * orank))
        if a.produces_index:
            terms += scale(ih * di, qrank)
        expect += terms
    got = sum(a.bytes for a in target_weight_allocations(m, Q8, tp)
              if "-w-dense-L" in a.alloc_id)
    assert got == expect


def test_compress_pool_and_ring_oracle():
    """ratio2 池化 op=7d+9/group 独立枚举 + ring state 字节。"""
    d = 512
    # 独立 op 枚举：softmax(2)=max1+sub2+exp2+add1+div2=8；加权和 2d 乘 +
    # (d−1) 加；RMSNorm 4d+2
    manual = 8 + 2 * d + (d - 1) + 4 * d + 2
    assert manual == 7 * d + 9
    B, S = 3, 999
    rows = _rows(B=B, ctx=S, Q=1)
    pool = next(x for x in _layer_rows(rows, 2) if x.op == "compress_pool")
    groups = B * ((S + 1) // 2 - S // 2)
    assert pool.flops == groups * manual
    assert pool.state_write == B * min(1, 8) * 2 * d * 4
    assert pool.state_read == B * 2 * d * 4        # 每请求 ≤1 行上界
    assert pool.statistic == "approximated"
    # ratio1（L20）：纯 norm，exact
    pool1 = next(x for x in _layer_rows(rows, 20) if x.op == "compress_pool")
    assert pool1.flops == B * 1 * (4 * d + 2)
    assert pool1.state_read == pool1.state_write == 0
    assert pool1.statistic == "exact"


def test_index_buffers_transient_lower_bound():
    kv_spec = _model().extras["kv_spec"]
    allocs = v41_index_buffer_allocations(kv_spec, tokens=4)
    by_id = {a.alloc_id: a.bytes for a in allocs}
    assert by_id["target-t-v41-topk-buffer"] == 4 * _TOPK * 4
    assert by_id["target-t-v41-candidate-buffer"] == 4 * _CTB * 4
    assert all(a.lifecycle == "cross-round" for a in allocs)
    assert v41_index_buffer_allocations(kv_spec, 0) == []


def test_scenario_record_field_and_identity():
    base = ScenarioV2(model_id="deepseek_v41_flash", method_id="baseline",
                      draft_id=None, quant=V41_QUANT,
                      deployment={"mode": "baseline", "target_tp": 8})
    assert base.validate().deployment.v41_kv_record == "fp8_ds_mla_sm90"
    bad = ScenarioV2(model_id="deepseek_v41_flash", method_id="baseline",
                     draft_id=None, quant=V41_QUANT,
                     deployment={"mode": "baseline", "target_tp": 8,
                                 "v41_kv_record": "bf16"})
    with pytest.raises(ValueError, match="v41_kv_record"):
        bad.validate()
    # 非 V4.1 模型不受该字段影响（记录宽度只进 v41 行；但进场景身份）
    q = ScenarioV2(model_id="qwen36_35b_a3b", method_id="dflash_v1")
    q2 = ScenarioV2(model_id="qwen36_35b_a3b", method_id="dflash_v1",
                    deployment=DeploymentSpec(v41_kv_record="nvfp4_ds_mla_sm100"))
    assert q.validate().scenario_digest() != q2.validate().scenario_digest()


@pytest.mark.parametrize("device,rec", [
    ("h100_sxm", "fp8_ds_mla_sm100"),      # H100 是 SM90，不是 SM100
    ("h200_sxm", "nvfp4_ds_mla_sm100"),    # H200 是 SM90
    ("h100_sxm", "nvfp4_ds_mla_sm100"),
    ("l40", "fp8_ds_mla_sm90"),            # L40 是 Ada SM89
    ("l40", "fp8_ds_mla_sm100"),
    ("b200", "fp8_ds_mla_sm100"),          # 未注册设备 → 未登记 family
])
def test_v41_kv_record_rejects_incompatible_device(device, rec):
    sc = ScenarioV2(
        model_id="deepseek_v41_flash", method_id="baseline", draft_id=None,
        quant=V41_QUANT,
        deployment={"mode": "baseline", "target_tp": 8,
                    "target_device": device, "v41_kv_record": rec})
    with pytest.raises(ValueError, match="v41_kv_record"):
        sc.validate()


def test_v41_kv_record_compatible_registered_devices():
    # 注册的 SM90 设备只允许 sm90 记录；当前无 SM100 设备
    for device in ("h100_sxm", "h200_sxm"):
        sc = ScenarioV2(
            model_id="deepseek_v41_flash", method_id="baseline",
            draft_id=None, quant=V41_QUANT,
            deployment={"mode": "baseline", "target_tp": 8,
                        "target_device": device,
                        "v41_kv_record": "fp8_ds_mla_sm90"})
        assert sc.validate().deployment.v41_kv_record == "fp8_ds_mla_sm90"
    rows_q = build_model_work(_CAT["models"]["qwen36_35b_a3b"], "decode", 1, 1024,
                              1, Q8, 8, v41_kv_record="nvfp4_ds_mla_sm100")
    rows_q2 = build_model_work(_CAT["models"]["qwen36_35b_a3b"], "decode", 1,
                               1024, 1, Q8, 8)
    assert totals(rows_q) == totals(rows_q2)      # 非 v41 家族不消费


# ---------------------------------------------------------------------------
# 6. config 域负例（families 解析拒绝）
# ---------------------------------------------------------------------------

def _mutated_config(**over):
    cfg = copy.deepcopy(_CFG)
    cfg.update(over)
    return cfg


def _parse(cfg):
    from simverse.model_ir.families import parse_v41_csa2
    return parse_v41_csa2(cfg, _model().source, "deepseek_v41_flash",
                          "V4.1-Flash")


def test_config_domain_negatives():
    # candidate 源必须是 index source
    with pytest.raises(ValueError, match="candidate_source_layer_id"):
        _parse(_mutated_config(candidate_source_layer_id=19))
    # 源 >= 0 但 ctb/cb 缺 0 → 拒绝
    with pytest.raises(ValueError, match="candidate_topk_blocks"):
        _parse(_mutated_config(candidate_topk_blocks=0))
    # 消费层与源层压缩比失配（L24 改 r2，源 L20 为 r1）→ 坐标空间失配
    ratios = list(_RATIOS)
    ratios[24] = 2
    with pytest.raises(ValueError, match="压缩比"):
        _parse(_mutated_config(compress_ratios=ratios))
    # compress_ratio 3 仍拒绝（既有门禁保持）
    ratios = list(_RATIOS)
    ratios[24] = 3
    with pytest.raises(ValueError, match="0/1/2"):
        _parse(_mutated_config(compress_ratios=ratios))


# ---------------------------------------------------------------------------
# 7. replay 归因冻结（docs/evidence/csa2）
# ---------------------------------------------------------------------------

def test_replay_attribution_and_freeze():
    before = json.loads((EVID / "before.json").read_text(encoding="utf-8"))
    after = json.loads((EVID / "after.json").read_text(encoding="utf-8"))
    att = json.loads((EVID / "attribution.json").read_text(encoding="utf-8"))
    old, new = before["work"]["target_decode"], after["work"]["target_decode"]
    fields = list(old["totals"])
    # Σ by-op delta == 总 delta（逐字段）
    for key in fields:
        assert sum(g["delta"][key] for g in att["by_op"].values()) == \
            new["totals"][key] - old["totals"][key], key
    # 审计面外行逐字段冻结（含 final_norm/head/embed）
    AUDITED = set(att["expected_changed_ops"])
    def frozen(rows):
        return [{k: v for k, v in r.items() if k != "stage"}
                for r in rows if r["op"] not in AUDITED]
    assert frozen(old["rows"]) == frozen(new["rows"])
    # 审计面 = 预期集合（无面外行变化）
    assert set(att["by_op"]) <= AUDITED
    # 基线即 R05 后状态（918 行）
    assert old["totals"]["rows"] == 918 and old["totals"]["flops"] == 7161190222
    # M01 差 = 逐行 t_ms 差
    assert abs(sum(g["t_ms_delta"] for g in att["by_op"].values())
               - att["t_ms_delta"]) < 1e-12
