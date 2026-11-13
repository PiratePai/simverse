# SPDX-License-Identifier: MIT
"""R24：V4.1 SwiGLU clamp + compress_ratios 尾部测试。

证据分层（G10）：本文件全部为源码阅读（source-checked）对应的解析口径、
纯 stdlib 数值 oracle 与整数算术对账；未执行任何 vLLM kernel/CPU
forward/GPU（clamp CUDA kernel 与 native 的数值等价以 vLLM 侧在册测试
登记，本仓未跑，不冒充 measured）。

组织（镜像 tests/test_v41_quant.py 结构）：
  §1 pin manifest：swiglu-tail-pin.json 13 文件 blob sha1 对 vllm-tree.json
     + 磁盘 SHA256/size 逐项校验；
  §2 pinned 源码锚行在场（防漂移）；
  §3 纯 stdlib 数值 oracle：clamp 单/双侧不对称、边界值、饱和、NaN 传播、
     SWIGLUOAI 家族常数差异（V4.1 域外）；
  §4 ModelIR：extras.swiglu / draft_compress_ratios / 覆盖与缺口事实；
  §5 手算绝对值：moe_act/moe_shared_act 行（TP1/TP8/EP8）；
  §6 负例：config 域（swiglu_limit 0/负/NaN/inf/缺失/字符串；尾部非零/
     坏值/长度域外）；
  §7 replay 归因冻结 + 账本守恒（字节单一所有者、面外行冻结）。
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simverse.count.moe import moe_rows  # noqa: E402
from simverse.count.work import build_model_work, totals  # noqa: E402
from simverse.model_ir.catalog import build_catalog  # noqa: E402
from simverse.model_ir.families import parse_v41_csa2  # noqa: E402
from simverse.model_ir.schema import SourceIdentity  # noqa: E402

EVID = ROOT / "docs/evidence/swiglu-tail"
REFS = ROOT / "docs/rebuild-20261106/references"
Q8 = {"dense_w": 8, "expert_w": 4, "act": 8, "kv": 8, "state": 32}
Q16 = {"dense_w": 16, "expert_w": 16, "act": 16, "kv": 16, "state": 32}

H, INTER, E, K = 5120, 2304, 384, 6
L, MTP = 40, 3
LIMIT = 10.0


def _model():
    return build_catalog()["models"]["deepseek_v41_flash"]


def _official_full():
    return json.loads(
        (ROOT / "assets/config/deepseek_v41_config.json")
        .read_text(encoding="utf-8"))


def _parse(full):
    return parse_v41_csa2(
        full["text_config"], SourceIdentity("deepseek-ai/x", "1", "s", "t"),
        "t", "t", full_config=full, text_prefix="text_config")


def _variant(**over):
    """官方 config 的 text_config 叶键覆写（深拷贝）。"""
    full = _official_full()
    tc = full["text_config"]
    for key, val in over.items():
        if val is _DROP:
            tc.pop(key, None)
        else:
            tc[key] = val
    return full


class _DROP:
    pass


# ---------------------------------------------------------------------------
# §1 pin manifest
# ---------------------------------------------------------------------------

def test_swiglu_pin_manifest_files_match_tree_and_disk():
    pin = json.loads((REFS / "vllm/swiglu-tail-pin.json").read_text(
        encoding="utf-8"))
    tree = {x["path"]: x["sha"] for x in json.loads(
        (REFS / "vllm-tree.json").read_text(encoding="utf-8"))["tree"]}
    assert pin["commit"] == "f358f0578a83f4e1fe6fc49a492d3ae9362de8e5"
    assert len(pin["files"]) == 13
    for item in pin["files"]:
        data = (REFS / "vllm" / item["path"]).read_bytes()
        blob = hashlib.sha1(
            b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        assert blob == tree[item["path"]] == item["blob_sha1"], item["path"]
        assert hashlib.sha256(data).hexdigest() == item["sha256"]
        assert len(data) == item["size"]
        assert item["tree_match"] is True


def test_swiglu_pin_covers_semantics_and_both_consumers():
    pin = json.loads((REFS / "vllm/swiglu-tail-pin.json").read_text(
        encoding="utf-8"))
    paths = {f["path"] for f in pin["files"]}
    for must in (
        # 语义本体（native + cuda 分派）
        "vllm/model_executor/layers/activation.py",
        # fused MoE clamp 消费链（config → from_configs → 应用点）
        "vllm/model_executor/layers/fused_moe/config.py",
        "vllm/model_executor/layers/fused_moe/activation.py",
        "vllm/model_executor/layers/fused_moe/utils.py",
        # 注册执行域（Marlin）激活位
        "vllm/model_executor/layers/fused_moe/experts/marlin_moe.py",
        # 模型接线（共享专家 MLP + fused 工厂透传；CPU 变体同构）
        "vllm/models/deepseek_v4/nvidia/model.py",
        "vllm/models/deepseek_v4/cpu/model.py",
        # 尾部归属
        "vllm/models/deepseek_v41/attention.py",
        "vllm/transformers_utils/deepseek_v4_utils.py",
        "vllm/models/deepseek_v41/nvidia/dspark.py",
        "vllm/v1/attention/backends/mla/sparse_swa.py",
        # vLLM 侧数值等价测试（在册未跑，登记用）
        "tests/kernels/core/test_activation.py",
    ):
        assert must in paths, must


# ---------------------------------------------------------------------------
# §2 pinned 源码锚行（防漂移；行号漂移 = 需要复核 pin）
# ---------------------------------------------------------------------------

def test_pinned_source_key_lines_present():
    def has(path, needle):
        text = (REFS / "vllm" / path).read_text(encoding="utf-8")
        assert needle in text, (path, needle)

    # 语义本体：gate 单侧 max、up 双侧、alpha/beta 一般式（V4.1 退化）
    has("vllm/model_executor/layers/activation.py",
        "class SiluAndMulWithClamp(CustomOp):")
    has("vllm/model_executor/layers/activation.py",
        "gate = torch.clamp(x[..., :d], max=self.swiglu_limit)")
    has("vllm/model_executor/layers/activation.py",
        "up = torch.clamp(x[..., d:], min=-self.swiglu_limit, "
        "max=self.swiglu_limit)")
    has("vllm/model_executor/layers/activation.py",
        "return gate * torch.sigmoid(self.alpha * gate) * (up + self.beta)")
    # fused MoE：clamp_limit 回退链 + 家族常数（SWIGLUOAI 7.0 域外）
    has("vllm/model_executor/layers/fused_moe/activation.py",
        "clamp_limit = quant_config.gemm1_clamp_limit")
    has("vllm/model_executor/layers/fused_moe/activation.py",
        "clamp_limit = moe_config.swiglu_limit")
    has("vllm/model_executor/layers/fused_moe/activation.py",
        "masked_clamp_limit = 7.0")
    has("vllm/model_executor/layers/fused_moe/config.py",
        "swiglu_limit: float | None = None")
    # Triton 触发条件（与 MLP is-not-None 在 0 处分叉）
    has("vllm/model_executor/layers/fused_moe/utils.py",
        "HAS_LIMIT=swiglu_limit > 0,")
    # 注册执行域（Marlin）在 gemm1/gemm2 之间应用 clamp 激活
    has("vllm/model_executor/layers/fused_moe/experts/marlin_moe.py",
        "activation_input = intermediate_cache1.view(-1, w13_num_shards * N)")
    has("vllm/model_executor/layers/fused_moe/experts/marlin_moe.py",
        "apply_moe_activation(")
    # 模型接线：共享专家 MLP + swiglu_limit 读取/透传
    has("vllm/models/deepseek_v4/nvidia/model.py",
        "self.act_fn = SiluAndMulWithClamp(swiglu_limit)")
    has("vllm/models/deepseek_v4/nvidia/model.py",
        "self.swiglu_limit = config.swiglu_limit")
    has("vllm/models/deepseek_v4/nvidia/model.py",
        "swiglu_limit=self.swiglu_limit,")
    has("vllm/models/deepseek_v4/cpu/model.py",
        "self.act_fn = SiluAndMulWithClamp(swiglu_limit)")
    # 尾部：MTP included + 按层号消费 + else 分支 0 + is_backbone 门控
    has("vllm/transformers_utils/deepseek_v4_utils.py",
        "Per-layer compress ratios (0 = sliding window), MTP layers included.")
    has("vllm/transformers_utils/deepseek_v4_utils.py",
        "return ratios + [0] * config.num_nextn_predict_layers")
    has("vllm/models/deepseek_v41/attention.py",
        "compress_ratios has one entry per layer (MTP layers included):")
    has("vllm/models/deepseek_v41/attention.py",
        "self.compress_ratio = int(compress_ratios[layer_id])")
    has("vllm/models/deepseek_v41/attention.py",
        "# MTP layers past the configured list are pure sliding-window.")
    has("vllm/models/deepseek_v41/attention.py",
        "is_backbone = layer_id < config.num_hidden_layers")
    has("vllm/models/deepseek_v41/nvidia/dspark.py",
        "self.num_dspark_layers = (")
    has("vllm/models/deepseek_v41/nvidia/dspark.py",
        'f"layers.{self.num_hidden_layers + i}")')
    # 后端只按去重层型集合消费（尾部 0 与主干 0 同组）
    has("vllm/v1/attention/backends/mla/sparse_swa.py",
        "compress_ratios = get_compress_ratios(hf_config) or [1]")
    # vLLM 侧数值等价测试在册（本仓未执行）
    has("tests/kernels/core/test_activation.py",
        "cuda kernel must match native reference.")


# ---------------------------------------------------------------------------
# §3 纯 stdlib 数值 oracle（不 import 生产代码当 oracle）
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    # 数值稳定 sigmoid（与 torch.sigmoid 的解析定义一致）
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _clamp_minmax(x: float, lo: float, hi: float) -> float:
    if x != x:
        return x                       # torch.clamp 保留 NaN
    return hi if x > hi else (lo if x < lo else x)


def _clamp_max(x: float, hi: float) -> float:
    if x != x:
        return x
    return hi if x > hi else x


def silu_and_mul_with_clamp(gate_up, limit, alpha=1.0, beta=0.0):
    """独立实现 pinned forward_native 语义（activation.py L239-243）。"""
    d = len(gate_up) // 2
    gate_raw, up_raw = gate_up[:d], gate_up[d:]
    gate = [_clamp_max(g, limit) for g in gate_raw]
    up = [_clamp_minmax(u, -limit, limit) for u in up_raw]
    return [g * _sigmoid(alpha * g) * (u + beta) for g, u in zip(gate, up)]


def test_oracle_gate_clamp_is_one_sided_and_up_two_sided():
    # 关键不对称：gate 只 clamp max——负 gate 原样通过；up 双侧。
    v = silu_and_mul_with_clamp([-100.0, -100.0], LIMIT)
    # gate=-100 未被钳（若被钳到 -10，输出会不同）
    assert v[0] == pytest.approx(-100.0 * _sigmoid(-100.0) * -10.0)
    assert v[0] < -99.0 * -10.0        # 远小于钳制值对应的输出
    v2 = silu_and_mul_with_clamp([100.0, -100.0], LIMIT)
    # gate=100 → 钳到 10；up=-100 → 钳到 -10
    assert v2[0] == pytest.approx(10.0 * _sigmoid(10.0) * -10.0)


def test_oracle_boundary_at_and_beyond_limit():
    at_g = silu_and_mul_with_clamp([LIMIT, LIMIT], LIMIT)
    beyond_g = silu_and_mul_with_clamp([LIMIT + 5.0, LIMIT], LIMIT)
    assert at_g[0] == beyond_g[0]      # gate 恰在限值 = 超限（连续，无跳变）
    up_at = silu_and_mul_with_clamp([0.0, LIMIT], LIMIT)
    up_beyond = silu_and_mul_with_clamp([0.0, LIMIT + 1e6], LIMIT)
    assert up_at[0] == up_beyond[0]    # up 上界钳制
    up_lo = silu_and_mul_with_clamp([0.0, -LIMIT], LIMIT)
    up_lo_beyond = silu_and_mul_with_clamp([0.0, -1e6], LIMIT)
    assert up_lo[0] == up_lo_beyond[0]  # up 下界钳制


def test_oracle_saturation_and_zero():
    # gate 超限后输出对 gate 饱和（导数 0 语义：进一步增大不变）
    a = silu_and_mul_with_clamp([2 * LIMIT, 1.0], LIMIT)[0]
    b = silu_and_mul_with_clamp([3 * LIMIT, 1.0], LIMIT)[0]
    assert a == b == LIMIT * _sigmoid(LIMIT) * 1.0
    # gate=0 → silu(0)=0 → 输出 0（up 任意）
    assert silu_and_mul_with_clamp([0.0, 123.0], LIMIT)[0] == 0.0


def test_oracle_nan_propagates():
    v = silu_and_mul_with_clamp([float("nan"), 1.0], LIMIT)
    assert v[0] != v[0]                # clamp/sigmoid 均保留 NaN
    v2 = silu_and_mul_with_clamp([1.0, float("nan")], LIMIT)
    assert v2[0] != v2[0]


def test_oracle_swigluoai_family_constants_differ_v41_out_of_domain():
    # SWIGLUOAI（alpha=1.702/beta=1.0/limit=7.0；fused_moe/activation.py
    # L268-270 硬编码）与 V4.1（10.0/1.0/0.0）在同一输入上输出不同——
    # 证明家族常数实质影响数值且 V4.1 域外，不得混用。
    x = [8.0, -8.0, 8.0, -8.0]        # d=2：gate=[8,-8]、up=[8,-8]
    v41 = silu_and_mul_with_clamp(x, 10.0)
    oai = silu_and_mul_with_clamp(x, 7.0, alpha=1.702, beta=1.0)
    assert v41[0] != oai[0] and v41[1] != oai[1]
    # gate=-8：单侧 max 钳不作用于负值 → gate=-8 原样；up=-8 → 钳 -7；
    # (up+beta) = -7+1 = -6
    assert oai[1] == pytest.approx(-8.0 * _sigmoid(1.702 * -8.0) * (-6.0))


def test_oracle_ops_per_elem_accounting():
    # 计量口径：clamp 3（gate max 1 + up min 1 + up max 1）+ sigmoid 1 +
    # 乘 2 = 6 算术 op/元素——与 extras.swiglu.ops_per_elem 一致。
    assert _model().extras["swiglu"]["ops_per_elem"] == 6


# ---------------------------------------------------------------------------
# §4 ModelIR：覆盖/缺口/extras 事实
# ---------------------------------------------------------------------------

def test_model_ir_swiglu_and_tail_mapped():
    m = _model()
    sw = m.extras["swiglu"]
    assert sw["limit"] == 10.0 and sw["alpha"] == 1.0 and sw["beta"] == 0.0
    assert sw["gate"] == "clamp_max" and sw["up"] == "clamp_minmax"
    assert sw["applies_to"] == ["moe_routed", "moe_shared"]
    cov = m.coverage["text_config.swiglu_limit"]
    assert cov["status"] == "mapped"
    assert cov["consumer"] == "model_ir+count"
    assert "SiluAndMulWithClamp" in cov["note"]
    # 尾部归属：3 项全 0（纯 SWA 草稿层）
    assert m.extras["kv_spec"]["draft_compress_ratios"] == [0, 0, 0]
    assert m.mtp_layers == MTP
    assert m.extras["kv_spec"]["compress_ratios"][L:] == [0, 0, 0]


def test_model_ir_gaps_updated_legitimately():
    # R24 前缺口（swiglu_limit / compress_ratios[40:43]）按 pin 解除；
    # structure_complete 仍 False（注意力核本体 unverified_paths 阻断）。
    m = _model()
    gap_keys = {g.split(":")[0] for g in m.gaps}
    assert "text_config.swiglu_limit" not in gap_keys
    assert not any("compress_ratios[" in g for g in m.gaps)
    assert not m.structure_complete
    assert m.config_coverage_complete          # 全部叶键已分类
    assert not m.forward_evidence_complete     # 核本体未核 → 结构资格仍缺
    assert any("csa2_attention_core" in u for u in m.evidence.unverified_paths)
    # compress_ratios 覆盖注记提到尾部归属
    note = m.coverage["text_config.compress_ratios"]["note"]
    assert "尾部 [40:43]" in note and "DSpark" in note


def test_candidate_block_size_gap_fixed_registration():
    # R24 补漏：R20 已 mapped 的键此前漏在 struct 集合，长期误挂
    # "未登记 config 键"缺口；现应 mapped（语义不变，非放宽）。
    m = _model()
    cov = m.coverage["text_config.candidate_block_size"]
    assert cov["status"] == "mapped"
    assert "candidate_blocks.py" in cov["note"]


# ---------------------------------------------------------------------------
# §5 手算绝对值：moe_act / moe_shared_act 行
# ---------------------------------------------------------------------------

def _moe():
    return _model().layers[0].moe


def test_act_rows_tp1_hand_computed():
    m = _model()
    rows = build_model_work(m, "decode", 1, 0, 1, Q8, tp=1)
    act = [r for r in rows if r.op in ("moe_act", "moe_shared_act")]
    assert len(act) == 2 * L
    routed = next(r for r in act if r.op == "moe_act")
    shared = next(r for r in act if r.op == "moe_shared_act")
    # TP1：inter_rank=2304、act_assign=k=6 / T=1
    assert routed.flops == 6 * INTER * 6            # 82,944
    assert shared.flops == 1 * INTER * 6            # 13,824
    assert routed.flops == 82_944 and shared.flops == 13_824
    # 字节全零（单一所有者：已归 moe_routed/moe_shared）
    for r in act:
        assert (r.weight_read, r.weight_capacity, r.act_read, r.act_write) \
            == (0, 0, 0, 0)
    assert routed.statistic == "exact" == shared.statistic


def test_act_rows_tp8_hand_computed_and_per_layer_sum():
    m = _model()
    rows = build_model_work(m, "decode", 1, 32768, 1, Q16, tp=8)
    act = [r for r in rows if r.op in ("moe_act", "moe_shared_act")]
    assert len(act) == 2 * L
    # TP8：inter_rank = 2304//8 = 288；routed = 6 行 × 288 × 6 op
    assert {r.flops for r in act if r.op == "moe_act"} == {6 * 288 * 6}
    assert {r.flops for r in act if r.op == "moe_shared_act"} == {1 * 288 * 6}
    total = sum(r.flops for r in act)
    assert total == L * (6 * 288 * 6 + 288 * 6) == 483_840


def test_act_rows_ep8_expectation_and_local_share():
    m = _model()
    # EP8（tp=1, ep=8）：E_l = 384/8 = 48，share = 48/384 = 0.125
    rows = build_model_work(m, "decode", 1, 32768, 1, Q8, tp=1, ep=8, rank=0)
    routed = next(r for r in rows if r.layer == 0 and r.op == "moe_act")
    # act_assign = T·k·share = 6·0.125 = 0.75；inter 整专家 2304；
    # flops = int(0.75 × 2304 × 6 op) = 10,368（与 moe_routed 同 int 截断惯例）
    assert routed.flops == int(6 * 48 / 384 * 2304 * 6) == 10_368
    assert routed.statistic == "expectation"
    # 与同层 moe_routed 的 assignment 口径一致（比值 = 6 op/elem）
    moe_routed = next(r for r in rows if r.layer == 0 and r.op == "moe_routed")
    assert moe_routed.flops == int(2 * 1 * 6 * 3 * H * INTER * 48 / 384)


def test_act_rows_absent_without_recipe():
    # Qwen36 无 extras.swiglu → 不加激活行（R06 冻结口径不变）
    q = build_catalog()["models"]["qwen36_35b_a3b"]
    rows = build_model_work(q, "decode", 1, 128, 1, Q8, tp=1)
    assert not [r for r in rows if r.op in ("moe_act", "moe_shared_act")]
    assert "swiglu" not in (q.extras or {})


def test_moe_rows_act_recipe_bytes_single_owner():
    moe = _moe()
    with_act = moe_rows(moe, 1, H, Q8, tp=8, act_recipe={
        "limit": 10.0, "ops_per_elem": 6})
    without = moe_rows(moe, 1, H, Q8, tp=8)
    # 面外行（routed/shared/gate）逐字段不变；act 行字节为 0
    by_op = lambda rs, op: [r for r in rs if r["layer_op"] == op]
    for op in ("moe_router", "moe_routed", "moe_shared", "shared_gate"):
        assert by_op(without, op) == by_op(with_act, op), op
    for op in ("moe_act", "moe_shared_act"):
        row = by_op(with_act, op)[0]
        assert row["weight_read"] == row["act_read"] == row["act_write"] == 0
        assert row["flops"] > 0
    # V4.1 共享专家无 gate（R21）：无 shared_gate 行
    assert {r["layer_op"] for r in without} == {
        "moe_router", "moe_routed", "moe_shared"}


# ---------------------------------------------------------------------------
# §6 负例：config 域
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [0, -10.0, float("nan"), float("inf"), "10"])
def test_swiglu_limit_out_of_domain_rejected(bad):
    with pytest.raises(ValueError, match="swiglu_limit"):
        _parse(_variant(swiglu_limit=bad))


def test_swiglu_limit_missing_rejected():
    # official 资产恒 10.0；无 clamp 变体不在 source-checked 域
    with pytest.raises(ValueError, match="swiglu_limit"):
        _parse(_variant(swiglu_limit=_DROP))


@pytest.mark.parametrize("tail", [
    [0, 0, 1],      # 非零尾部：草稿层消费压缩 KV 未建模
    [2, 0, 0],
    [0, 3, 0],      # 值域外（attention.py L387-392 只认 0/1/2）
])
def test_tail_nonzero_or_bad_value_rejected(tail):
    with pytest.raises(ValueError):
        _parse(_variant(compress_ratios=[0, 0] + [2] * 18 + [1] * 20 + tail))


@pytest.mark.parametrize("ratios_len_note", [
    ("too_long", [0] * 44),          # 超出 L+mtp：无层号消费的死配置
    ("partial_tail", [0] * 41),      # 部分 tail：既非 L 也非 L+mtp
])
def test_tail_length_domain(ratios_len_note):
    note, ratios = ratios_len_note
    base = [0, 0] + [2] * 18 + [1] * 20
    if note == "too_long":
        full_ratios = base + [0, 0, 0, 0]
    else:
        full_ratios = base + [0]
    with pytest.raises(ValueError, match="compress_ratios 长度"):
        _parse(_variant(compress_ratios=full_ratios))


def test_tail_absent_len_l_maps_default_zero_draft():
    # len == L：attention.py else 分支 → 草稿层默认 0（纯 SWA），可映射
    base = [0, 0] + [2] * 18 + [1] * 20
    m = _parse(_variant(compress_ratios=base))
    assert m.extras["kv_spec"]["draft_compress_ratios"] == [0, 0, 0]


# ---------------------------------------------------------------------------
# §7 replay 归因冻结 + 账本守恒
# ---------------------------------------------------------------------------

def test_replay_attribution_and_freeze():
    att = json.loads((EVID / "attribution.json").read_text(encoding="utf-8"))
    assert set(att["expected_changed_ops"]) == {"moe_act", "moe_shared_act"}
    # 证据包自校验
    assert (hashlib.sha256((EVID / "before.json").read_bytes()).hexdigest()
            == att["before_sha256"])
    assert (hashlib.sha256((EVID / "scenario.json").read_bytes()).hexdigest()
            == att["scenario_sha256"])
    prov = json.loads((EVID / "baseline-provenance.json")
                      .read_text(encoding="utf-8"))
    assert prov["before_sha256"] == att["before_sha256"]
    assert prov["captured_before_edits"] is True
    assert prov["head"] == "b8585a1bd2733cc3e282dfd31bc92f546e48c1ee"
    before = json.loads((EVID / "before.json").read_text(encoding="utf-8"))
    after = json.loads((EVID / "after.json").read_text(encoding="utf-8"))
    old, new = before["work"]["target_decode"], after["work"]["target_decode"]
    AUDITED = set(att["expected_changed_ops"])
    frozen = lambda rows: [r for r in rows if r["op"] not in AUDITED]
    assert frozen(old["rows"]) == frozen(new["rows"])
    for key in old["totals"]:
        assert (new["totals"][key] - old["totals"][key]
                == sum(g["delta"][key] for g in att["by_op"].values())), key
    # 手算对账：decode TP8/B1，40 层 × (routed 6·288·6 + shared 288·6)
    d = att["delta"]
    assert d["flops"] == 483_840
    assert d["flops"] == L * (6 * 288 * 6 + 288 * 6)
    assert d["rows"] == 80
    for byte_key in ("weight_read", "weight_read_expected", "act_read",
                     "act_write", "kv_read", "kv_write", "state_read",
                     "state_write"):
        assert d[byte_key] == 0, byte_key       # 字节单一所有者：无重复计费
    # 内存账本不受激活行影响（clamp 不改布局）
    assert att["after_known_peak"] == att["before_known_peak"]
    assert att["t_ms_delta"] > 0
    assert att["execution"].startswith("analytical replay only")


def test_replay_tool_check_mode_reproduces():
    import subprocess
    out = subprocess.run(
        [sys.executable, str(ROOT / "tools/replay_v41_swiglu.py"), "--check"],
        capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    assert "check OK" in out.stdout


# ---------------------------------------------------------------------------
# §8 门禁事实：公式/证据引用/工作台不回归
# ---------------------------------------------------------------------------

def test_formula_and_evidence_registered():
    import simverse.formula_registry as FR
    f = FR.get(FR.F_SWIGLU_CLAMP_ACT)
    assert f["expr"].startswith("ops = rows·inter_rank·6")
    assert "SiluAndMulWithClamp" in f["source"]
    from simverse.validation.registry import build_registry
    rec = build_registry()["ev-vllm-v41-swiglu-tail"]
    assert rec.executed is False
    m = _model()
    assert "ev-vllm-v41-swiglu-tail" in m.evidence_refs
    from simverse.importers import source_map_json
    sm = {e["entry_id"]: e for e in source_map_json()["entries"]}
    entry = sm["sm-v41-swiglu-tail"]
    assert entry["both_sides_covered"] is True and entry["uncovered_domain"]
    assert "sm-v41-swiglu-tail" in m.evidence.checked_paths
