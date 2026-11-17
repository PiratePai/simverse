"""V4.1 稀疏 MLA 注意力核本体（R25，2026-11-16）。

证据链：csa2-core-pin.json 14 文件（vllm @f358f057 5 文件 blob 对
vllm-tree.json 逐项校验 + FlashMLA @847ef308 9 文件——vllm cmake GIT_TAG
钉定的外部项目，blob SHA1 对该提交 git 对象计算）；全部 source-checked，
kernel/CPU forward/GPU 均未执行。

测试分层（不 import 生产公式当 oracle 的地方均独立推导）：
  * pin 完整性：14 文件固定集合 + blob/SHA256/size（防清单删项/漂移）；
  * 调度器 oracle：get_decoding_sched_meta.cu 贪心的**独立重演**
    （按 partition 逐字面消耗模拟，与生产移植对照）+ 手算锚点 + 负例；
  * decode/prefill 账本行手算绝对值（真实常数，逐字段）；
  * 头数口径：decode padded 64 / prefill SM90 small-heads 真实头；
  * 生产者/消费者字节对齐 + 行数守恒；
  * device_sms 缺失显式拒绝（未知不得默认）；
  * replay 归因冻结（docs/evidence/csa2-core 前后 + 面外行冻结）。
"""
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simverse.count.flashmla_sched import (decode_num_blocks,  # noqa: E402
                                           flashmla_decode_num_splits)
from simverse.count.work import build_model_work, totals  # noqa: E402
from simverse.devices import DEVICES  # noqa: E402
from simverse.model_ir.catalog import build_catalog  # noqa: E402
from simverse.validation.registry import build_registry  # noqa: E402

BASE = ROOT / "docs/rebuild-20261106/references"
EVID = ROOT / "docs/evidence/csa2-core"

_CAT = build_catalog(include_internal=True)
Q8 = {"dense_w": 8, "expert_w": 4, "act": 8, "kv": 8, "state": 32}
_D, _SWA, _TOPK = 512, 128, 512
_SMS = 132          # H100 SXM（devices.sm_count，spec 来源）


def _model():
    return _CAT["models"]["deepseek_v41_flash"]


def _rows(B, ctx, Q=1, prefill=False, tp=8, sms=_SMS):
    return build_model_work(_model(), "target_decode" if not prefill else
                            "prefill", B, ctx, Q, Q8, tp, prefill=prefill,
                            device_sms=sms)


def _one(rows, li, op):
    return next(r for r in rows if r.layer == li and r.op == op)


# ---------------------------------------------------------------------------
# 1. pin 完整性（csa2-core-pin.json 14 文件固定集合）
# ---------------------------------------------------------------------------

def test_csa2_core_pin_files_and_hashes():
    pin = json.loads((BASE / "vllm/csa2-core-pin.json").read_text(
        encoding="utf-8"))
    paths = {f["path"] for f in pin["files"]}
    vllm_files = {
        "vllm/models/deepseek_v41/nvidia/flashmla.py",
        "vllm/v1/attention/ops/flashmla.py",
        "cmake/external_projects/flashmla.cmake",
        "vllm/models/deepseek_v41/nvidia/specialized_decode/__init__.py",
        "vllm/v1/kv_cache_interface.py",
    }
    flashmla_files = {
        "flash_mla/flash_mla_interface.py", "csrc/params.h",
        "csrc/api/sparse_decode.cpp", "csrc/api/sparse_prefill.cpp",
        "csrc/cuda_kernels/smxx/decode/combine/combine.cu",
        "csrc/cuda_kernels/smxx/decode/get_decoding_sched_meta/"
        "get_decoding_sched_meta.cu",
        "csrc/cuda_kernels/legacy/sm90/decode/sparse/splitkv_mla.cuh",
        "csrc/cuda_kernels/legacy/sm90/decode/sparse/splitkv_mla.h",
        "csrc/cuda_kernels/legacy/sm90/prefill/sparse/phase1.h",
    }
    assert paths == vllm_files | flashmla_files, "pin 清单漂移"
    assert pin["vllm_commit"].startswith("f358f057")
    assert pin["flashmla_commit"].startswith("847ef308")
    tree = {x["path"]: x["sha"] for x in json.loads(
        (BASE / "vllm-tree.json").read_text(encoding="utf-8"))["tree"]}
    for f in pin["files"]:
        data = (BASE / f["local"]).read_bytes()
        sha1 = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
        assert sha1 == f["blob_sha1"] and len(data) == f["size"], f["path"]
        assert f["sha256"] == hashlib.sha256(data).hexdigest()
        if f["path"] in vllm_files:
            assert tree[f["path"]] == sha1      # 对上游 tree 逐项校验
        else:
            assert "FlashMLA" in f["repo"]      # 外部项目 pin（cmake GIT_TAG）
    # 外部项目钉定来源本身在 tree 中：cmake GIT_TAG 与 manifest 一致
    cmake = (BASE / "vllm/cmake/external_projects/flashmla.cmake").read_text(
        encoding="utf-8")
    assert pin["flashmla_commit"] in cmake


# ---------------------------------------------------------------------------
# 2. 调度器 oracle（get_decoding_sched_meta.cu 贪心的独立重演）
# ---------------------------------------------------------------------------

def _sched_reference(topk_lens, extra_lens, sms, overhead=5, block=64):
    """独立重演：逐 partition 字面消耗（与生产移植不同实现风格）。

    直接按 CUDA kernel 语义构造每个 partition 的工作区间：
    blocks[i] = (ceil(max(t,1),64) + e − 1)//64 + 1；total = Σ(blocks+5)；
    payload = ceil(total/sms)+5；然后顺序切 partition，每个 partition 从
    当前 (请求, 块内偏移) 消费至 payload 用尽或全部完成。
    """
    b = len(topk_lens)
    blocks = []
    for t, e in zip(topk_lens, extra_lens):
        cur = max(t, 1)
        cur = -(-cur // block) * block          # ceil 到 64 的倍数（长度单位）
        cur += e
        blocks.append((cur - 1) // block + 1)
    total = sum(n + overhead for n in blocks)
    payload = -(-total // sms) + overhead
    splits = [0] * b
    req, off, partial = 0, 0, 0
    for _ in range(sms):
        budget = payload
        while req < b:
            remain = blocks[req] - off
            if budget >= remain + overhead:
                splits[req] = partial + 1
                budget -= remain + overhead
                req, off, partial = req + 1, 0, 0
            else:
                if budget - overhead > 0:
                    off += budget - overhead
                    partial += 1
                break
        if req >= b:
            break
    assert off == 0
    return splits


@pytest.mark.parametrize("topk,extra,sms,expect", [
    # 手算锚点（H100 132 SM、块 64、固定开销 5；ku::ceil 把长度取整到
    # 64 的倍数——128→128，不是块数）：
    # 长行（窗 128 + topk 512）：cur=128+512=640 → 10 块；
    # total=15、payload=ceil(15/132)+5=6 < 15 → 每 partition 1 块 → 10 splits
    ([128], [512], 132, [10]),
    # B=2 同长行：total=30、payload=6 → 每 partition 1 块 → 各 10 splits
    ([128, 128], [512, 512], 132, [10, 10]),
    # 窗 64/extra 0：cur=64 → 1 块；1+5=6 ≤ payload 6 → 单 partition 吞整
    ([64], [0], 132, [1]),
    # 窗 65/extra 0：ceil 到 128 → 2 块；7 > 6 → p0 消 1 块、p1 收尾 → 2
    ([65], [0], 132, [2]),
    # 窗 1/extra 0：ceil(1)=64 → 1 块 → 单 partition
    ([1], [0], 132, [1]),
    ([0], [0], 132, [1]),
    # B=88×10 块：total=1320、payload=10+5=15 ≥ 10+5=15 → 整行吞 → 各 1 split
    ([128] * 88, [512] * 88, 132, [1] * 88),
    # 小 SM 数（如 8）：total=15、payload=ceil(15/8)+5=7 → 逐 partition 2 块
    # （7−5）：p0 块 0-1、p1 块 2-3、p2 块 4-5、p3 块 6-7、p4 块 8-9 收尾 → 5
    ([128], [512], 8, [5]),
])
def test_scheduler_hand_anchors_and_reference(topk, extra, sms, expect):
    got = flashmla_decode_num_splits(topk, extra, sms)
    assert got == expect, (topk, extra, sms, got)
    assert got == _sched_reference(topk, extra, sms), "独立重演不一致"


def test_scheduler_randomized_against_reference():
    import random
    rnd = random.Random(25)
    for _ in range(200):
        b = rnd.randint(1, 40)
        topk = [rnd.randint(0, 200) for _ in range(b)]
        extra = [rnd.choice([0, 512, rnd.randint(0, 600)]) for _ in range(b)]
        sms = rnd.choice([8, 66, 100, 132, 142])
        assert flashmla_decode_num_splits(topk, extra, sms) == \
            _sched_reference(topk, extra, sms)


def test_scheduler_negative_missing_sms():
    with pytest.raises(ValueError, match="sm_count"):
        flashmla_decode_num_splits([128], [512], 0)
    with pytest.raises(ValueError, match="sm_count"):
        flashmla_decode_num_splits([128], [512], None)


def test_block_count_formula_edges():
    # cur = ceil_to_multiple(max(t,1),64) + extra；blocks = (cur-1)//64 + 1
    assert decode_num_blocks(128, 512) == 10   # 128+512=640 → 10 块
    assert decode_num_blocks(0, 0) == 1        # max(...,1)→ceil→64 → 1 块
    assert decode_num_blocks(64, 0) == 1
    assert decode_num_blocks(65, 0) == 2       # ceil(65)=128 → 2 块
    assert decode_num_blocks(1, 512) == 9      # ceil(1)=64 + 512 = 576 → 9
    assert decode_num_blocks(128, 0) == 2


# ---------------------------------------------------------------------------
# 3. decode 账本行手算绝对值（TP8/B1/ctx32768，真实常数）
# ---------------------------------------------------------------------------

def test_decode_core_row_absolute_tp8():
    """L2（r=2，压缩层）：pairs=128+512；decode 头数 padded 64（非 8）。"""
    rows = _rows(B=1, ctx=32768)
    core = _one(rows, 2, "sparse_attention")
    local, comp = 128, min(32769 // 2, 512)      # 512
    assert (local, comp) == (128, 512)
    assert core.flops == 1 * (local + comp) * 64 * (4 * _D + 5)
    assert core.flops == 640 * 64 * 2053         # = 84,090,880
    assert core.kv_read == 1 * (local * 584 + comp * 584)
    assert core.act_read == 1 * 64 * _D * 2 + 64 * 4     # q bf16 + sink
    assert core.act_write == 1 * 64 * _D * 2
    # 旧口径（heads//tp=8）在 TP8 少算 8×——差值即本轮主要修正
    assert core.flops // 8 == 640 * 8 * 2053


def test_decode_swa_only_row_and_sched_meta_layers():
    """L0（r=0 纯 SWA）：pairs=128、仍 64 头 + split-KV（2 splits 手算）。"""
    rows = _rows(B=1, ctx=32768)
    core = _one(rows, 0, "sparse_attention")
    assert core.flops == 128 * 64 * 2053
    assert core.kv_read == 128 * 584
    split = _one(rows, 0, "attn_splitkv")
    # 调度器：窗 128 → 2 块 + 5 = 7 > payload 6 → 2 splits（手算锚点）
    assert flashmla_decode_num_splits([128], [0], _SMS) == [2]
    accum = 2 * 64 * (_D * 4 + 4)
    assert split.act_read == accum
    assert split.act_write == accum + 1 * 64 * 4
    assert split.flops == 2 * 64 * (2 * _D + 6)
    # sched_meta：每层型首层（r 去重 0/2/1 → L0/L2/L20）
    assert [r.layer for r in rows if r.op == "attn_sched_meta"] == [0, 2, 20]
    meta = _one(rows, 2, "attn_sched_meta")
    assert meta.act_write == _SMS * 32 + (1 + 1) * 4   # DecodingSchedMeta=32B
    # swa 窗索引：每步一次（记 L0）
    swa_build = _one(rows, 0, "attn_swa_index_build")
    assert swa_build.flops == 128 * 5
    assert swa_build.act_read == 128 * 4
    assert swa_build.act_write == 128 * 4 + 4


def test_decode_splitkv_row_absolute():
    """L2：9 splits（手算）；o_accum/lse_accum 字节与 combine flops。"""
    rows = _rows(B=1, ctx=32768)
    split = _one(rows, 2, "attn_splitkv")
    assert flashmla_decode_num_splits([128], [512], _SMS) == [10]
    accum = 10 * 64 * (_D * 4 + 4)
    assert split.act_read == accum == 10 * 64 * 2052
    assert split.act_write == accum + 64 * 4
    assert split.flops == 10 * 64 * (2 * _D + 6)


def test_decode_index_global_row_absolute():
    rows = _rows(B=1, ctx=32768)
    g = _one(rows, 2, "attn_index_global")
    assert g.flops == 1 * _TOPK * 5
    assert g.act_read == 1 * _TOPK * 4 + 512 * 4   # 缓冲整宽 + 块表有效项
    assert g.act_write == 1 * _TOPK * 4 + 4
    # swa-only 层（L0/L1）无该行；压缩层每层一条
    assert [r.layer for r in rows if r.op == "attn_index_global"] == \
        [i for i in range(40) if i >= 2]


def test_decode_heads_tp_independent_and_batch_splits():
    """decode 核算子头数恒 padded（TP1/TP8 同 64）；B 变化走调度器。"""
    r8 = _one(_rows(B=1, ctx=32768, tp=8), 2, "sparse_attention")
    r1 = _one(_rows(B=1, ctx=32768, tp=1), 2, "sparse_attention")
    assert r8.flops == r1.flops            # decode 头数不随 TP 变
    # B=8（uniform）：调度器独立重演对照 attn_splitkv
    rows = _rows(B=8, ctx=32768)
    split = _one(rows, 2, "attn_splitkv")
    per = flashmla_decode_num_splits([128] * 8, [512] * 8, _SMS)
    total_splits = sum(per)
    assert split.act_read == total_splits * 64 * (_D * 4 + 4)
    # 每行都分片（splits>1）时 lse 仍每行一次
    assert all(s > 1 for s in per)
    assert split.act_write == total_splits * 64 * (_D * 4 + 4) + 8 * 64 * 4


# ---------------------------------------------------------------------------
# 4. prefill 账本行手算绝对值（gather 口径）
# ---------------------------------------------------------------------------

def test_prefill_gather_and_core_rows_absolute():
    """B1/ctx4096：L2 压缩 2048 行 + SWA 尾窗 4096 行 gather；核读工作区。"""
    ctx = 4096
    rows = _rows(B=1, ctx=ctx, Q=ctx, prefill=True)
    g = _one(rows, 2, "attn_kv_gather")
    n_comp, gather = ctx // 2, ctx          # S=0：gather_len = Q
    assert g.kv_read == n_comp * 584 + gather * 584
    assert g.act_write == (n_comp + gather) * _D * 2
    assert g.flops == (n_comp + gather) * 2 * _D
    core = _one(rows, 2, "sparse_attention")
    # pairs（独立求和）：Σ min(t,128) + Σ min(⌊t/2⌋,512)
    local = sum(min(t, _SWA) for t in range(1, ctx + 1))
    comp = sum(min(t // 2, _TOPK) for t in range(1, ctx + 1))
    assert core.flops == (local + comp) * 8 * (4 * _D + 5)   # prefill 8 头
    assert core.kv_read == 0                    # 记录读在 gather 行
    assert core.act_read == (ctx * 8 * _D * 2 + (local + comp) * _D * 2
                             + 8 * 4)           # q 切片 + 工作区逐对 + sink
    assert core.act_write == ctx * 8 * _D * 2 + ctx * 8 * 8
    # prefill 头数 = 真实本地头（SM90 small-heads），TP1 → 64（padded 同值）
    core1 = _one(_rows(B=1, ctx=ctx, Q=ctx, prefill=True, tp=1), 2,
                 "sparse_attention")
    assert core1.flops == (local + comp) * 64 * (4 * _D + 5)


def test_prefill_combine_row_every_layer():
    """sparse_index_build：prefill 每层（R20 的 decode 归因已修正）。"""
    ctx = 1024
    rows = _rows(B=1, ctx=ctx, Q=ctx, prefill=True)
    builds = [r for r in rows if r.op == "sparse_index_build"]
    assert [r.layer for r in builds] == list(range(40))
    from simverse.count.work import causal_pairs
    b0 = builds[0]                                # r=0：topk=0
    local = causal_pairs(ctx, 1, _SWA)
    assert b0.act_write == ctx * 128 * 4 + local * 4 + ctx * 4
    b2 = builds[2]                                # r=2
    comp = causal_pairs(ctx, 2, _TOPK)
    assert b2.act_write == ctx * 640 * 4 + (local + comp) * 4 + ctx * 4
    assert b2.act_read == comp * 4 + ctx * 16
    assert b2.flops == (local + comp) * 3
    # SWA 窗索引由 decode metadata builder 构建；prefill 只走 combine。
    assert not [r for r in rows if r.op == "attn_swa_index_build"]
    # decode 无 combine 行（默认 FlashMLA 路径不调用该 Triton 核）
    assert not [r for r in _rows(B=1, ctx=ctx) if r.op == "sparse_index_build"]


def test_q_rope_padded_write_width():
    """fused insert 写 q_padded [T,64,512] bf16（RoPE 只算真实头）。"""
    rows = _rows(B=1, ctx=32768)
    qr = _one(rows, 2, "q_rope")
    assert qr.act_write == 1 * 64 * _D * 2        # padded 写宽
    assert qr.flops == 3 * 1 * 8 * 64             # rope 仅 n_local=8 头
    # decode 核读 64 头（与生产者对齐）；prefill 核读 8 头切片（≤ 写宽）
    assert _one(rows, 2, "sparse_attention").act_read >= 1 * 8 * _D * 2


# ---------------------------------------------------------------------------
# 5. 守恒与对齐
# ---------------------------------------------------------------------------

def test_row_count_conservation_decode():
    """行数守恒：+40 splitkv +38 index_global +1 swa_build +3 sched_meta
    −9（decode combine 旧行）= +73（replay delta.rows）。"""
    before_rows = 1087        # docs/evidence/csa2-core before.json（main）
    rows = _rows(B=1, ctx=32768)
    assert totals(rows)["rows"] == before_rows + 73
    n_new = {op: sum(1 for r in rows if r.op == op)
             for op in ("attn_splitkv", "attn_index_global",
                        "attn_swa_index_build", "attn_sched_meta",
                        "sparse_index_build")}
    assert n_new == {"attn_splitkv": 40, "attn_index_global": 38,
                     "attn_swa_index_build": 1, "attn_sched_meta": 3,
                     "sparse_index_build": 0}


def test_transient_allocations_lower_bound():
    from simverse.lowering.memory import v41_attn_transient_allocations
    kv_spec = _model().extras["kv_spec"]
    allocs = {a.alloc_id: a.bytes for a in
              v41_attn_transient_allocations(kv_spec, 1, 32768, _SMS)}
    assert allocs["target-t-v41-attn-oaccum"] == \
        (1 + _SMS) * 64 * _D * 4 + (1 + _SMS) * 64 * 4
    # prefill 工作区最宽层型 r=1：⌊32768/1⌋ + 32768 行 × 1024B
    assert allocs["target-t-v41-attn-prefill-ws"] == 1 * (32768 + 32768) * _D * 2
    assert allocs["target-t-v41-attn-combined-idx"] == 32768 * 640 * 4 + 32768 * 4
    # SM 数未知 → o_accum 不入账（不默认），但 prefill 工作区仍在
    allocs0 = {a.alloc_id for a in
               v41_attn_transient_allocations(kv_spec, 1, 32768, None)}
    assert "target-t-v41-attn-oaccum" not in allocs0
    assert "target-t-v41-attn-prefill-ws" in allocs0


def test_devices_sm_count_registered_with_source():
    assert DEVICES["h100_sxm"].sm_count == 132
    assert DEVICES["h200_sxm"].sm_count == 132
    assert DEVICES["l40"].sm_count == 142
    for dev in DEVICES.values():
        if dev.backend == "cuda":
            assert dev.sources["sm_count"].startswith("spec:"), dev.id
        else:
            assert dev.sm_count is None, dev.id
            assert "不适用" in dev.sources["sm_count"], dev.id


# ---------------------------------------------------------------------------
# 6. device_sms 缺失显式拒绝（未知不得默认）
# ---------------------------------------------------------------------------

def test_decode_requires_device_sms_for_v41():
    with pytest.raises(ValueError, match="sm_count"):
        build_model_work(_model(), "target_decode", 1, 32768, 1, Q8, 8)
    # prefill 不消费调度器 → 不需要（gather 不依赖 SM 数）
    rows = build_model_work(_model(), "prefill", 1, 1024, 1024, Q8, 8,
                            prefill=True)
    assert any(r.op == "attn_kv_gather" for r in rows)
    # 非 V4.1 家族不受影响（Qwen36 行集不变）
    q = _CAT["models"]["qwen36_35b_a3b"]
    a = build_model_work(q, "decode", 1, 1024, 1, Q8, 8)
    b = build_model_work(q, "decode", 1, 1024, 1, Q8, 8, device_sms=_SMS)
    assert totals(a) == totals(b)


# ---------------------------------------------------------------------------
# 7. replay 归因冻结（docs/evidence/csa2-core）
# ---------------------------------------------------------------------------

def test_replay_attribution_and_freeze():
    before = json.loads((EVID / "before.json").read_text(encoding="utf-8"))
    after = json.loads((EVID / "after.json").read_text(encoding="utf-8"))
    att = json.loads((EVID / "attribution.json").read_text(encoding="utf-8"))
    old, new = before["work"]["target_decode"], after["work"]["target_decode"]
    fields = list(old["totals"])
    for key in fields:
        assert sum(g["delta"][key] for g in att["by_op"].values()) == \
            new["totals"][key] - old["totals"][key], key
    AUDITED = set(att["expected_changed_ops"])
    frozen = lambda rows: [{k: v for k, v in r.items() if k != "stage"}
                           for r in rows if r["op"] not in AUDITED]
    assert frozen(old["rows"]) == frozen(new["rows"])
    assert set(att["by_op"]) <= AUDITED
    # 基线即 main ee70ce2（R23 合入后；1087 行 / 7,176,937,062 flops）
    assert old["totals"]["rows"] == 1087
    assert old["totals"]["flops"] == 7176937062
    # 主要修正量：decode 核 flops +2.85G（≈ ×8 头 + splitkv/整数核/删旧行）
    assert att["delta"]["flops"] == 2850849054
    assert abs(sum(g["t_ms_delta"] for g in att["by_op"].values())
               - att["t_ms_delta"]) < 1e-12


# ---------------------------------------------------------------------------
# 8. 门禁事实：证据四栏 / 资格不冒充
# ---------------------------------------------------------------------------

def test_evidence_record_and_catalog_facts():
    rec = build_registry()["ev-vllm-v41-csa2-core"]
    assert rec.kind == "source-checked" and rec.executed is False
    assert len(rec.content_sha256) == 64
    m = _model()
    assert "ev-vllm-v41-csa2-core" in m.evidence_refs
    assert "sm-v41-csa2-attn-core" in m.evidence.checked_paths
    # 源码阅读 ≠ CPU forward 执行：四栏不冒充
    assert m.evidence.cpu_forward == "not-run"
    assert m.evidence.gpu_measurement == "none"
    # structure_complete 仍 False：剩余 unverified 为显式执行变体
    assert not m.structure_complete
    assert any("csa2_attn_execution_variants" in u
               for u in m.evidence.unverified_paths)
    assert not any("仍为 v0.1 移植" in u for u in m.evidence.unverified_paths)


def test_formula_registry_entries_exist():
    from simverse import formula_registry as FR
    for fid in (FR.F_CSA2_ATTN_CORE, FR.F_CSA2_ATTN_GATHER,
                FR.F_CSA2_ATTN_SPLITKV, FR.F_CSA2_ATTN_INDEX_GLOBAL,
                FR.F_CSA2_ATTN_SWA_INDEX_BUILD, FR.F_CSA2_ATTN_SCHED_META):
        entry = FR.get(fid)                  # 全部入册（G16 语义锚）
        assert entry["expr"] and entry["source"], fid
