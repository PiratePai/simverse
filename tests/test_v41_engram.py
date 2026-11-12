"""V4.1 Engram 建模独立 oracle（2026-11-11 R04）。

证据链：vllm/models/deepseek_v41/common/engram.py pin @f358f057
（blob sha1 ba4ceea624ef6513ed7e787d084883f42d9d8322，对
docs/rebuild-20261106/references/vllm-tree.json 校验；source-checked，
未执行）。本文件**不 import 生产素数逻辑当 oracle**：Miller-Rabin 与
素数扫描独立自写，层桶和/首 8 素数/表字节全部独立复算。

测试分层：
  * pin 完整性（blob sha1 防漂移）；
  * 素数桶独立复算（层和 == engram_num_embeddings 精确相等）；
  * 表字节 TP1/2/4/8 独立分片复算 + tp 非法负例；
  * 四算子计数小 fixture 手算绝对值；
  * 门控公式独立数学对拍（纯 stdlib 复刻 post_wkv 核语义）；
  * 守恒：开/关 Engram 的整模型账本差 = 精确新增；count↔memory 对账；
  * 表位置（cpu_offload/gpu）与场景输入校验、通信行字节。
"""
import hashlib
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simverse.count.work import build_model_work, engram_rows, totals  # noqa: E402
from simverse.cost.t0 import price_row  # noqa: E402
from simverse.devices import DEVICES  # noqa: E402
from simverse.lowering.memory import (DeviceBudget, RankMemory,  # noqa: E402
                                      engram_allocations,
                                      engram_staged_allocations,
                                      target_weight_allocations)
from simverse.lowering.sharding import model_collectives  # noqa: E402
from simverse.lowering.parallel import validate_parallel  # noqa: E402
from simverse.model_ir.catalog import build_catalog  # noqa: E402
from simverse.pipeline import run_v2  # noqa: E402
from simverse.scenario import ScenarioV2, V41_QUANT  # noqa: E402

Q16 = {"dense_w": 16, "expert_w": 16, "act": 16, "kv": 16, "state": 32}
Q8 = {"dense_w": 8, "expert_w": 8, "act": 8, "kv": 8, "state": 32}

REF_ENGRAM = (ROOT / "docs" / "rebuild-20261106" / "references" / "vllm" /
              "vllm" / "models" / "deepseek_v41" / "common" / "engram.py")
TREE_JSON = (ROOT / "docs" / "rebuild-20261106" / "references" /
             "vllm-tree.json")

_CAT = build_catalog(include_internal=True)


def _model():
    return _CAT["models"]["deepseek_v41_flash"]


def _spec():
    return _model().extras["engram"]


# ---------------------------------------------------------------------------
# 独立 oracle：Miller-Rabin + 素数扫描（不 import 生产 engram_bucket_primes）
# ---------------------------------------------------------------------------

def _mr_is_prime(n: int) -> bool:
    """独立确定性 Miller-Rabin（基 {2,7,61}，n<2**32 有效）。"""
    if n < 2:
        return False
    for p in (2, 3, 5, 7):
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
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


def _independent_primes(vocab_size: int, n_layers: int, max_ngram: int,
                        n_heads: int):
    """独立复刻 EngramLayout（engram.py L230-242）：每 ngram 组重置起点、
    seen 全局 → vocab_size−1 之上连续 n_layers×(max−1)×n_heads 个素数。"""
    seen = set()
    layers = []
    for _ in range(n_layers):
        flat = []
        for _ in range(max_ngram - 1):
            cur = vocab_size - 1
            for _ in range(n_heads):
                c = cur + 1
                while not _mr_is_prime(c) or c in seen:
                    c += 1
                seen.add(c)
                flat.append(c)
        layers.append(flat)
    return layers


# ---------------------------------------------------------------------------
# pin 完整性（防未来漂移）
# ---------------------------------------------------------------------------

def test_engram_pin_blob_sha1_matches_tree():
    """engram.py 的 git blob sha1 必须与 vllm-tree.json 记录一致（R04 pin）。"""
    tree = json.loads(TREE_JSON.read_text(encoding="utf-8"))
    assert tree["sha"] == "f358f0578a83f4e1fe6fc49a492d3ae9362de8e5"
    entry = next(e for e in tree["tree"]
                 if e["path"] == "vllm/models/deepseek_v41/common/engram.py")
    data = REF_ENGRAM.read_bytes()
    blob = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
    assert blob == entry["sha"] == \
        "ba4ceea624ef6513ed7e787d084883f42d9d8322"
    assert len(data) == entry["size"] == 75143


# ---------------------------------------------------------------------------
# 素数桶独立复算（R04 最强验证锚点）
# ---------------------------------------------------------------------------

def test_bucket_layer_sums_equal_num_embeddings_exactly():
    """独立扫描：层 1/14 的 24 桶长之和 == config engram_num_embeddings。"""
    layers = _independent_primes(16_000_000, 2, 4, 8)
    assert sum(layers[0]) == 384_006_168
    assert sum(layers[1]) == 384_016_682
    assert [sum(x) for x in layers] == _spec()["num_embeddings"]


def test_first_eight_primes_hardcoded():
    """首 8 素数硬编码对账（主控独立验证值；seed=engram_vocab_size−1）。"""
    layers = _independent_primes(16_000_000, 2, 4, 8)
    assert layers[0][:8] == [16000057, 16000079, 16000081, 16000097,
                             16000121, 16000129, 16000133, 16000183]


def test_production_primes_match_independent_and_unique():
    """生产 engram_bucket_primes 与独立扫描一致；素数全局不重复、
    桶 offset（cumsum 前缀和）单调递增。"""
    from simverse.model_ir.families import engram_bucket_primes
    prod = engram_bucket_primes(16_000_000, 2, 4, 8)
    indep = _independent_primes(16_000_000, 2, 4, 8)
    assert prod == indep
    flat = [p for layer in prod for p in layer]
    assert len(flat) == 48 and len(set(flat)) == 48      # 全局不复用
    for layer in prod:
        assert len(layer) == 24
        # offset = cumsum([0, *sizes[:-1]])（engram.py L244-246）严格递增
        offs, acc = [], 0
        for p in layer[:-1]:
            offs.append(acc)
            acc += p
        offs.append(acc)
        assert offs == sorted(offs) and len(set(offs)) == 24
        assert offs[-1] + layer[-1] == sum(layer)        # 末桶右端 = 行数


# ---------------------------------------------------------------------------
# 表字节与 TP 分片
# ---------------------------------------------------------------------------

BYTES_TP1 = 202_758_032_400     # 2 层桶和 × 264（主控独立验证值）


def test_table_bytes_tp1_exact():
    layers = _independent_primes(16_000_000, 2, 4, 8)
    total = sum(sum(x) for x in layers) * 264
    assert total == BYTES_TP1 == \
        (384_006_168 + 384_016_682) * 264


@pytest.mark.parametrize("tp,part", [(2, 12), (4, 6), (8, 3)])
def test_table_bytes_tp_shard_independent(tp, part):
    """tp=2/4/8：每 rank 行数 = 独立素数分片（连续 part 桶）之和 ×264；
    全 rank 表字节之和守恒（== TP1 总量）。"""
    layers = _independent_primes(16_000_000, 2, 4, 8)
    allocs = engram_allocations(_spec(), 5120, Q16, tp, layer_id=1, rank=0)
    table = next(a for a in allocs if a.alloc_id.endswith("table-L1"))
    assert table.placement == "cpu_offload"
    assert table.bytes == sum(layers[0][:part]) * 264
    # 全 rank 复算（tp=8 每 rank 3 桶；不同 rank 行数可不同，如实呈现）
    for idx, li in enumerate((1, 14)):
        per_rank = [sum(layers[idx][r * part:(r + 1) * part])
                    for r in range(tp)]
        assert sum(per_rank) == sum(layers[idx])        # 行数守恒
    # tp=5/6 合法（ceil 分片余数：part=5/4，(tp−1)·part < 24）
    for tp56 in (5, 6):
        part56 = -(-24 // tp56)
        assert (tp56 - 1) * part56 < 24
        assert sum(sum(layers[0][r * part56:min((r + 1) * part56, 24)])
                   for r in range(tp56)) == sum(layers[0])


@pytest.mark.parametrize("tp", [7, 9, 16])
def test_illegal_tp_rejected(tp):
    """负例：镜像源码断言 (tp−1)·ceil(24/tp) < 24（engram.py L1096）。"""
    with pytest.raises(ValueError, match="Engram TP%d 非法" % tp):
        engram_rows("decode", 1, 1, 5120, _spec(), Q16, tp)
    with pytest.raises(ValueError):
        engram_allocations(_spec(), 5120, Q16, tp, layer_id=1)


def test_tp6_head_legal_but_wkv_divisibility_rejects():
    """tp=6：head 分片合法但 wkv 25600 不整除 6 → 列切拒绝（如实分层）。"""
    part = -(-24 // 6)
    assert (6 - 1) * part < 24                     # head 合法
    with pytest.raises(ValueError, match="不整除"):
        engram_rows("decode", 1, 1, 5120, _spec(), Q16, 6)
    # V4.1 部署配方另受 o_groups 整除约束（tp∈{1,2,4,8}）
    assert validate_parallel(_model(), 8, 1).valid
    assert not validate_parallel(_model(), 3, 1).valid


# ---------------------------------------------------------------------------
# 四算子计数：小 fixture 手算绝对值（独立推导，不 import 生产公式）
# ---------------------------------------------------------------------------

_SMALL = {
    "layer_ids": [1], "num_embeddings": [0],  # 行数由素数分片独立算
    "max_ngram_size": 4, "vocab_size": 1000, "n_heads": 8, "head_dim": 256,
    "pad_token_id": 2, "compressed_vocab_size": 99092,
    "n_hash_cols": 24, "block_size": 32, "value_dtype": "fp8_e4m3",
    "scale_encoding": "ue8m0", "bytes_per_row": 264, "hc_mult": 2,
    "hash_source": "first_local_layer_swa_lookback(3)",
}


def test_engram_rows_hand_anchor_t3():
    """T=3、h=64、hc=2、tp=1、Q16（b_w=b_a=2）逐行手算。

    手算（行宽 264=256+8；cols=24；d_v=256；wkv_in=24·256=6144；
    wkv_out=64·(2+1)=192）：
      hash:  flops=3×1×(2·4+2·3·8)=168；ar=3×4=12；aw=3×1×24×4=288
      lookup: flops=3×24×256=18,432；wre=3×24×264=19,008.0；
              ar=3×24×4=288；aw=3×24×256×2=36,864；cap=Σ首24素数(>999)×264
      wkv:   flops=2×3×6144×192=7,077,888；wr=cap=6144×192×2=2,359,296；
              ar=3×6144×2=36,864；aw=3×192×2=1,152
      post:  flops=3×2×(10×64+5)=3,870；wr=cap=2×2×64×2=512；
              ar=3×2×64×2+3×3×64×2+3=1,923；aw=3×2×64×2=768
    """
    rows = engram_rows("decode", 1, 3, 64, _SMALL, Q16, tp=1,
                       first_engram_layer=True)
    by_op = {r.op: r for r in rows}
    assert list(by_op) == ["engram_hash", "engram_lookup", "engram_wkv",
                           "engram_post_wkv"]
    h = by_op["engram_hash"]
    assert h.flops == 168 and h.act_read == 12 and h.act_write == 288
    assert h.statistic == "exact" and h.weight_capacity == 0
    lk = by_op["engram_lookup"]
    indep_rows = sum(_independent_primes(1000, 1, 4, 8)[0])
    assert lk.flops == 3 * 24 * 256 == 18432
    assert lk.weight_read == 0
    assert lk.weight_read_expected == pytest.approx(19008.0)
    assert lk.act_read == 288 and lk.act_write == 36864
    assert lk.weight_capacity == indep_rows * 264
    assert lk.statistic == "expectation"
    wv = by_op["engram_wkv"]
    assert wv.flops == 2 * 3 * 6144 * 192 == 7077888
    assert wv.weight_read == wv.weight_capacity == 6144 * 192 * 2 == 2359296
    assert wv.act_read == 36864 and wv.act_write == 1152
    pw = by_op["engram_post_wkv"]
    assert pw.flops == 3 * 2 * (10 * 64 + 5) == 3870
    assert pw.weight_read == pw.weight_capacity == 2 * 2 * 64 * 2 == 512
    assert pw.act_read == 3 * 2 * 64 * 2 + 3 * 3 * 64 * 2 + 3 == 1923
    assert pw.act_write == 3 * 2 * 64 * 2 == 768


def test_engram_rows_token_scaling_g13():
    """T 翻倍：token 维量 ×2、权重读/容量不变（G13 口径）；期望流量随 T。"""
    a = totals(engram_rows("decode", 1, 3, 5120, _SMALL, Q16, tp=1,
                           first_engram_layer=True))
    b = totals(engram_rows("decode", 1, 6, 5120, _SMALL, Q16, tp=1,
                           first_engram_layer=True))
    assert b["flops"] == 2 * a["flops"]
    assert b["act_read"] == 2 * a["act_read"]
    assert b["act_write"] == 2 * a["act_write"]
    assert b["weight_read"] == a["weight_read"]
    assert b["weight_read_expected"] == 2 * a["weight_read_expected"]
    rows_a = engram_rows("decode", 1, 3, 5120, _SMALL, Q16, tp=1,
                         first_engram_layer=True)
    rows_b = engram_rows("decode", 1, 6, 5120, _SMALL, Q16, tp=1,
                         first_engram_layer=True)
    assert ([r.weight_capacity for r in rows_a] ==
            [r.weight_capacity for r in rows_b])


def test_lookup_rank_shard_and_bf16_bytes_independent_of_act_bits():
    """源码 lookup 仅处理本 rank 的 3 heads，输出与 wkv 输入固定 bf16。"""
    for rank in (0, 3, 7):
        rows = engram_rows("decode", 1, 2, 5120, _spec(), Q8, tp=8,
                           rank=rank)
        lookup = next(r for r in rows if r.op == "engram_lookup")
        wkv = next(r for r in rows if r.op == "engram_wkv")
        assert lookup.flops == 2 * 3 * 256
        assert lookup.weight_read_expected == 2 * 3 * 264.0
        assert lookup.act_read == 2 * 3 * 4
        assert lookup.act_write == 2 * 3 * 256 * 2
        assert wkv.act_read == 2 * 6144 * 2
        assert wkv.act_write == 2 * 3200 * 2
        post = next(r for r in rows if r.op == "engram_post_wkv")
        assert post.act_read == 2 * (4 + 5) * 5120 * 2 + 2
        assert post.act_write == 2 * 4 * 5120 * 2
        table = engram_allocations(_spec(), 5120, Q8, 8, 1,
                                   rank=rank)[0]
        assert lookup.weight_capacity == table.bytes


def test_cpu_offload_lookup_is_not_priced_as_gpu_hbm():
    row = next(r for r in engram_rows("decode", 1, 1, 5120, _spec(), Q8,
                                      tp=8) if r.op == "engram_lookup")
    gpu = price_row(row, DEVICES["h100_sxm"], engram_lookup_host=False)
    host = price_row(row, DEVICES["h100_sxm"], engram_lookup_host=True)
    assert gpu["bytes_expected"] == 3 * 264.0
    assert gpu["bytes_host_expected"] == 0.0
    assert host["bytes_expected"] == 0.0
    assert host["bytes_host_expected"] == 3 * 264.0
    assert host["t_ms"] <= gpu["t_ms"]


def test_real_model_engram_rows_only_on_engram_layers():
    """真实 V4.1：非 engram 层无 engram 行；layer_ids=[] → 无任何 engram 行。"""
    m = _model()
    rows = build_model_work(m, "decode", 1, 1024, 1, Q16, tp=1)
    eng_layers = {r.layer for r in rows if r.op.startswith("engram_")}
    assert eng_layers == {1, 14}
    # layer_ids=[]（显式去掉 engram 层）→ 完全无行（不猜默认）
    m_off = replace(m, extras={**m.extras, "engram":
                               {**m.extras["engram"], "layer_ids": []}})
    rows_off = build_model_work(m_off, "decode", 1, 1024, 1, Q16, tp=1)
    assert not any(r.op.startswith("engram_") for r in rows_off)


# ---------------------------------------------------------------------------
# 门控公式：纯 stdlib 独立实现 post_wkv 数学（源码 L1395-1421）
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _gate(hidden, q, k, key, eps: float, clamp: float) -> float:
    """独立复刻：ρ=(Σ h·q·k·key)·rsqrt(mean h²+eps)·rsqrt(mean key²+eps)·
    rsqrt(d)；gate=sigmoid(sign(ρ)·sqrt(max(|ρ|,clamp)))（L1395-1401）。"""
    d = len(hidden)
    dot = sum(h * qq * kk * kk2 for h, qq, kk, kk2 in zip(hidden, q, k, key))
    dot *= (1.0 / math.sqrt(sum(h * h for h in hidden) / d + eps))
    dot *= (1.0 / math.sqrt(sum(x * x for x in key) / d + eps))
    dot *= 1.0 / math.sqrt(d)
    g = math.sqrt(max(abs(dot), clamp))
    if dot < 0:
        g = -g
    return _sigmoid(g)


def test_gate_formula_matches_source_math():
    """小尺寸 d=8/hc=2 独立门控数学；公式常数 (10d+5) 为文档约定口径。

    独立实现按源码逐项重算（含 |ρ|<clamp 的钳位分支与负号分支）；
    计数口径（乘 6d + 加 3(d−1)+d + 标量≈8 = 10d+5）是文档约定而非
    指令数——测试断言公式注册表文本与数值一致。
    """
    hidden = [0.3 * (i + 1) - 1.0 for i in range(8)]
    q = [0.11 * (i + 1) for i in range(8)]
    k = [-0.07 * (i + 1) + 0.5 for i in range(8)]
    key = [0.05 * (i + 1) - 0.2 for i in range(8)]
    g = _gate(hidden, q, k, key, eps=1e-20, clamp=1e-6)
    assert 0.0 < g < 1.0
    # 负 ρ 分支：翻转 hidden 符号 → gate < 0.5（sigmoid(负)）
    g_neg = _gate([-h for h in hidden], q, k, key, 1e-20, 1e-6)
    assert g_neg < 0.5 < g
    # 钳位分支：全零输入 → ρ=0 → gate=sigmoid(sqrt(1e-6))（正号分支）
    g_zero = _gate([0.0] * 8, [0.0] * 8, [0.0] * 8, [0.0] * 8, 1e-20, 1e-6)
    assert g_zero == pytest.approx(_sigmoid(math.sqrt(1e-6)))
    # 公式注册表：口径文本与数值（T·hc·(10d+5)）一致
    from simverse import formula_registry as FR
    expr = FR.get(FR.F_ENGRAM_POST_WKV)["expr"]
    assert "10·d+5" in expr and "hc_mult" in expr
    row = next(r for r in engram_rows("decode", 1, 3, 5120, _spec(), Q16, 1)
               if r.op == "engram_post_wkv")
    assert row.flops == 3 * 4 * (10 * 5120 + 5)


# ---------------------------------------------------------------------------
# 守恒：开关 Engram 的账本差 = 精确新增；count ↔ memory 对账
# ---------------------------------------------------------------------------

def test_ledger_delta_is_exactly_engram_rows():
    """整模型开/关 engram：逐字段差 == 7 行 engram 的独立求和（精确）。

    V4.1 TP8/Q16/B=1/ctx=1024 独立手算（b_w=b_a=2，part=3，oc=3200）：
      L1+L14 各层：lookup 本 rank 3 heads，flops 768 / wre 792.0 /
      ar 12 / aw 1536（固定 bf16）；
      wkv flops 2·6144·3200=39,321,600 / wr=cap 6144·3200·2=39,321,600；
      post flops 4·(10·5120+5)=204,820 / wr=cap 2·4·5120·2=81,920 /
      ar (4·5120+5·5120)·2+1=92,161 / aw 4·5120·2=40,960；
      仅 L1：hash flops 112 / ar 4 / aw 192
    """
    m_on = _model()
    m_off = replace(m_on, extras={**m_on.extras, "engram":
                                  {**m_on.extras["engram"], "layer_ids": []}})
    on = build_model_work(m_on, "decode", 1, 1024, 1, Q16, tp=8)
    off = build_model_work(m_off, "decode", 1, 1024, 1, Q16, tp=8)
    delta_rows = [r for r in on if r.op.startswith("engram_")]
    assert len(delta_rows) == 7
    t_on, t_off = totals(on), totals(off)
    assert t_on["flops"] - t_off["flops"] == sum(r.flops for r in delta_rows)
    assert (t_on["weight_read"] - t_off["weight_read"]
            == sum(r.weight_read for r in delta_rows))
    assert (t_on["act_read"] - t_off["act_read"]
            == sum(r.act_read for r in delta_rows))
    assert (t_on["act_write"] - t_off["act_write"]
            == sum(r.act_write for r in delta_rows))
    assert (t_on["weight_read_expected"] - t_off["weight_read_expected"]
            == pytest.approx(sum(r.weight_read_expected or 0.0
                                 for r in delta_rows)))
    # 独立手算（TP8；R22：wkv 值 1B + ue8m0 scale 3200×6144/32=614,400/层；post_wkv q/k bf16 不变）
    per_common = (3 * 256 + 2 * 6144 * 3200 + 4 * (10 * 5120 + 5))
    assert t_on["flops"] - t_off["flops"] == 2 * per_common + 112
    assert t_on["weight_read"] - t_off["weight_read"] == \
        2 * (6144 * 3200 * 1 + 3200 * (6144 // 32) + 81920)
    assert t_on["act_read"] - t_off["act_read"] == \
        2 * (3 * 4 + 6144 * 2 + (4 * 5120 + 5 * 5120) * 2 + 1) + 4
    assert t_on["act_write"] - t_off["act_write"] == \
        2 * (3 * 256 * 2 + 3200 * 2 + 40960) + 192
    assert t_on["weight_read_expected"] - t_off["weight_read_expected"] \
        == pytest.approx(2 * 3 * 264.0)
    assert t_on["kv_read"] == t_off["kv_read"]
    assert t_on["kv_write"] == t_off["kv_write"]


def test_count_capacity_matches_memory_engram_objects():
    """count 行 weight_capacity ↔ memory engram 对象逐层相等（rank0，同精度）。"""
    m = _model()
    for tp in (1, 8):
        rows = build_model_work(m, "decode", 1, 1024, 1, Q16, tp)
        allocs = {a.alloc_id: a.bytes for a in target_weight_allocations(
            m, Q16, tp, engram_table_placement="cpu_offload")
            if "engram" in a.alloc_id}
        for li in (1, 14):
            lookup = next(r for r in rows
                          if r.op == "engram_lookup" and r.layer == li)
            wkv = next(r for r in rows
                       if r.op == "engram_wkv" and r.layer == li)
            post = next(r for r in rows
                        if r.op == "engram_post_wkv" and r.layer == li)
            assert lookup.weight_capacity == \
                allocs[f"target-w-engram-table-L{li}"]
            assert wkv.weight_capacity == allocs[f"target-w-engram-wkv-L{li}"]
            assert post.weight_capacity == allocs[f"target-w-engram-qk-L{li}"]


# ---------------------------------------------------------------------------
# 表位置（placement）与场景输入
# ---------------------------------------------------------------------------

def _rank_memory(tp: int, placement: str) -> RankMemory:
    from simverse.devices import DEVICES
    dev = DEVICES["h100_sxm"]
    rm = RankMemory("target-r0", "h100_sxm",
                    DeviceBudget("h100_sxm", int(dev.mem_capacity)))
    for a in target_weight_allocations(_model(), Q16, tp,
                                       engram_table_placement=placement):
        a.consumers = ["target"]
        rm.add(a)
    return rm


def test_table_placement_cpu_offload_excluded_from_gpu_peak():
    """默认 cpu_offload：表在 host（host_resident_bytes），GPU steady 不含；
    gpu 驻留：101.4 GiB/层计入 → 已知下界超 H100 标称 → 如实 OOM。"""
    cpu = _rank_memory(8, "cpu_offload")
    gpu_steady = cpu.steady_bytes()
    assert cpu.host_resident_bytes() == sum(
        sum(_independent_primes(16_000_000, 2, 4, 8)[i][:3])
        for i in (0, 1)) * 264
    assert cpu.host_resident_bytes() > 0
    assert cpu.transient_peak_bytes() == 0
    rep = cpu.peak_vs_budget()
    assert rep["known_peak_bytes"] == gpu_steady        # 表不在 GPU 峰值
    assert "host_resident_bytes" in rep
    # 显式 gpu 驻留：表计入 GPU。R22 量化映射（MXFP4 打包 0.5B+scale +
    # Marlin 对齐）后 TP8 已知下界 75.4 GiB < H100 标称 80 GiB——不再按
    # 下界判 OOM；可分配预算仍 unknown → 判定保持 unknown（R03b 四态链
    # 路不变，只是数值不再超界）。旧 Q16 口径（2B 专家）曾为 oom。
    gpu = _rank_memory(8, "gpu")
    assert gpu.host_resident_bytes() == 0
    rep_gpu = gpu.peak_vs_budget()
    assert rep_gpu["status"] == "unknown"
    assert rep_gpu["known_peak_bytes"] == gpu_steady + cpu.host_resident_bytes()
    assert rep_gpu["known_peak_bytes"] < rep_gpu["nominal_bytes"]
    assert rep_gpu["known_peak_bytes"] == 75_410_707_008   # R22 手算对账


def test_scenario_engram_table_placement_validation():
    """非法值拒绝（镜像 draft_ctx_kv_path 模式）；默认 cpu_offload 可跑。"""
    with pytest.raises(ValueError, match="engram_table_placement"):
        ScenarioV2(model_id="deepseek_v41_flash", method_id="baseline",
                   draft_id=None, quant=V41_QUANT,
                   deployment={"mode": "baseline", "target_tp": 8,
                               "engram_table_placement": "hbm"}).validate()
    sc = ScenarioV2(model_id="deepseek_v41_flash", method_id="baseline",
                    draft_id=None, quant=V41_QUANT,
                    deployment={"mode": "baseline", "target_tp": 8})
    assert sc.validate().deployment.engram_table_placement == "cpu_offload"
    # 默认 V4.1 run 的内存块披露 host 表 + staged 下界
    j = run_v2(sc).to_json()
    r0 = j["cost"]["memory"]["target_r0"]
    assert r0["host_resident_bytes"] == 25_344_443_520   # 2×rank0 分片×264
    # R20：staged 下界 + 共享索引/候选缓冲下界（1×512×4 + 1×2048×4）
    assert r0["transient_peak_bytes"] == 2 * 1 * 3 * 256 * 2 + 1 * 512 * 4         + 1 * 2048 * 4
    inv = [a for a in r0["inventory"] if "engram" in a["alloc_id"]]
    assert {a["placement"] for a in inv if "table" in a["alloc_id"]} == \
        {"cpu_offload"}
    assert j["capability"]["checks"]["engram_host_lookup_latency"]["status"] \
        == "unknown"
    assert j["metrics"]["M01_tpot_analytical"]["support"] == "approximated"
    assert "host" in j["metrics"]["M01_tpot_analytical"]["reason"]


def test_staged_allocations_lower_bound():
    """staged_rows 下界 = T×part×256×2B/层（max_num_batched_tokens 未 pin）。"""
    allocs = engram_staged_allocations(_spec(), 8, tokens=16)
    assert [a.alloc_id for a in allocs] == [
        "target-t-engram-staged-L1", "target-t-engram-staged-L14"]
    assert all(a.bytes == 16 * 3 * 256 * 2 for a in allocs)
    assert all(a.lifecycle == "cross-round" for a in allocs)


# ---------------------------------------------------------------------------
# 通信行（TP>1：每 engram 层 head/wkv 两次 all-gather）
# ---------------------------------------------------------------------------

def test_engram_gathers_per_rank_bytes():
    """per-rank 接收 = (tp−1)/tp×T×width×bf16（源码固定 2B）：
    wkv width=25600；head width=24×256=6144；TP1 无行。"""
    from simverse.devices import LINKS
    from simverse.lowering.sharding import LinkLike
    m = _model()
    link = LinkLike(bw=LINKS["pcie5"].bw, fixed_ms=LINKS["pcie5"].fixed_ms)
    plan = validate_parallel(m, 8, 1)
    cols = model_collectives(m, plan, m.hidden, 1.0, tokens=1, link=link)
    wkv = [c for c in cols if c.kind == "engram_wkv_gather"]
    head = [c for c in cols if c.kind == "engram_head_gather"]
    assert len(wkv) == 2 and len(head) == 2       # 层 1/14 各一次
    assert all(c.bytes_on_wire == int(7 / 8 * 25600 * 2) for c in wkv)
    assert all(c.bytes_on_wire == int(7 / 8 * 6144 * 2) for c in head)
    plan1 = validate_parallel(m, 1, 1)
    cols1 = model_collectives(m, plan1, m.hidden, 1.0, tokens=1, link=link)
    assert not any(c.kind.startswith("engram_") for c in cols1)
