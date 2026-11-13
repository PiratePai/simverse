# SPDX-License-Identifier: MIT
"""R22：V4.1 量化映射（quantization_config → 存储布局/字节/执行域）测试。

证据分层（G10）：本文件全部为源码阅读（source-checked）对应的解析口径与
整数算术 oracle；未执行任何 vLLM kernel/CPU forward/GPU（执行域结论以
extras.quant 登记的形式披露，不冒充 measured）。

组织（镜像 tests/test_v41_moe.py 结构）：
  §1 pin manifest：quant-pin.json 20 文件 blob sha1 对 vllm-tree.json +
     磁盘 SHA256/size 逐项校验；
  §2 pinned 源码锚行在场（防漂移）；
  §3 纯 stdlib dispatch/布局 oracle（不 import 生产代码当 oracle）；
  §4 小张量语义 oracle：e8m0 scale=2 的幂、ceil(log2(amax/448))+127 截断、
     fp4 打包 2 值/byte；
  §5 手算绝对值：逐类字节（dense scale/shared/expert round-up/embed-lm_head
     -norm BF16）；
  §6 负例：config 域 + 场景 quant 冻结；
  §7 count↔memory 对账 + replay 归因冻结 + 门禁事实。
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
from simverse.count.quant_layout import (  # noqa: E402
    marlin_round_up_hidden, marlin_round_up_intermediate,
    mxfp4_expert_bytes, mxfp8_linear_bytes,
    require_mxfp8_input_divisible)
from simverse.count.work import build_model_work  # noqa: E402
from simverse.lowering.memory import target_weight_allocations  # noqa: E402
from simverse.model_ir.catalog import build_catalog  # noqa: E402
from simverse.model_ir.families import parse_v41_csa2  # noqa: E402
from simverse.model_ir.schema import SourceIdentity  # noqa: E402
from simverse.scenario import ScenarioV2, V41_QUANT  # noqa: E402

EVID = ROOT / "docs/evidence/quant-map"
REFS = ROOT / "docs/rebuild-20261106/references"
Q8 = {"dense_w": 8, "expert_w": 4, "act": 8, "kv": 8, "state": 32}
Q16 = {"dense_w": 16, "expert_w": 16, "act": 16, "kv": 16, "state": 32}

H, HEADS, D, GROUPS, ORANK = 5120, 64, 512, 8, 1024
QRANK, DI, IH, INTER, E, K = 1280, 132, 32, 2304, 384, 6
VOCAB = 129280


def _model():
    return build_catalog()["models"]["deepseek_v41_flash"]


def _parse(full=None):
    """用官方 config（或变体）解析；无 full_config 时走固定资产默认。"""
    if full is None:
        cfg = json.loads(
            (ROOT / "assets/config/deepseek_v41_config.json")
            .read_text(encoding="utf-8"))
        return parse_v41_csa2(
            cfg["text_config"], SourceIdentity("deepseek-ai/x", "1", "s", "t"),
            "t", "t", full_config=cfg, text_prefix="text_config")
    return parse_v41_csa2(
        full["text_config"], SourceIdentity("deepseek-ai/x", "1", "s", "t"),
        "t", "t", full_config=full, text_prefix="text_config")


def _official_full():
    return json.loads(
        (ROOT / "assets/config/deepseek_v41_config.json").read_text(
            encoding="utf-8"))


# ---------------------------------------------------------------------------
# §1 pin manifest
# ---------------------------------------------------------------------------

def test_quant_pin_manifest_files_match_tree_and_disk():
    pin = json.loads((REFS / "vllm/quant-pin.json").read_text(encoding="utf-8"))
    tree = {x["path"]: x["sha"] for x in json.loads(
        (REFS / "vllm-tree.json").read_text(encoding="utf-8"))["tree"]}
    assert pin["commit"] == "f358f0578a83f4e1fe6fc49a492d3ae9362de8e5"
    assert len(pin["files"]) == 20
    for item in pin["files"]:
        data = (REFS / "vllm" / item["path"]).read_bytes()
        blob = hashlib.sha1(
            b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        assert blob == tree[item["path"]] == item["blob_sha1"], item["path"]
        assert hashlib.sha256(data).hexdigest() == item["sha256"]
        assert len(data) == item["size"]
        assert item["tree_match"] is True


def test_quant_pin_covers_dispatch_and_kernel_chain():
    pin = json.loads((REFS / "vllm/quant-pin.json").read_text(encoding="utf-8"))
    paths = {f["path"] for f in pin["files"]}
    for must in (
        "vllm/models/deepseek_v41/quant_config.py",       # dispatch 主体
        "vllm/config/model.py",                            # override 链
        "vllm/model_executor/layers/quantization/__init__.py",
        "vllm/model_executor/layers/quantization/fp8.py",  # from_config 键表
        "vllm/model_executor/layers/quantization/modelopt.py",  # KMxfp8Static
        "vllm/model_executor/layers/quantization/mxfp4.py",     # 专家 create_weights
        "vllm/model_executor/layers/fused_moe/oracle/mxfp4.py",  # 后端优先级/对齐
        "vllm/model_executor/layers/fused_moe/routed_experts.py",  # roundup 调用点
        "vllm/model_executor/kernels/linear/mxfp8/marlin.py",   # SM90 W8A16
        "vllm/model_executor/layers/vocab_parallel_embedding.py",
    ):
        assert must in paths, must


# ---------------------------------------------------------------------------
# §2 pinned 源码锚行（防漂移；行号漂移 = 需要复核 pin）
# ---------------------------------------------------------------------------

def test_pinned_source_key_lines_present():
    def has(path, needle):
        text = (REFS / "vllm" / path).read_text(encoding="utf-8")
        assert needle in text, (path, needle)

    has("vllm/models/deepseek_v41/quant_config.py",
        '_DEEPSEEK_V4_EXPERT_DTYPES = ("fp4", "fp8")')
    has("vllm/models/deepseek_v41/quant_config.py",
        'return self.expert_dtype == "fp4"')          # is_scale_e8m0
    has("vllm/models/deepseek_v41/quant_config.py",
        'and self.weight_block_size == [32, 32]')     # 线性 MXFP8 分支门
    has("vllm/models/deepseek_v41/quant_config.py",
        'return Mxfp4MoEMethod(layer.moe_config, '
        'activation_key=kMxfp8Dynamic)')
    has("vllm/models/deepseek_v41/quant_config.py",
        'hf_quant_cfg.get("quant_method") in ("fp8", "deepseek_v4_fp8")')
    has("vllm/model_executor/layers/quantization/fp8.py",
        'weight_block_size = cls.get_from_keys_or(config, '
        '["weight_block_size"], None)')
    # scale_fmt 不在 from_config 读取的键集合中（键表逐键列在 L155-172）
    fp8 = (REFS / "vllm/vllm/model_executor/layers/quantization/fp8.py") \
        .read_text(encoding="utf-8").splitlines()
    keys_block = "\n".join(fp8[154:172])
    assert "scale_fmt" not in keys_block, "scale_fmt 若被 loader 读取须复核注记"
    has("vllm/model_executor/layers/quantization/modelopt.py",
        "class KMxfp8Static(QuantKeyScheme):")
    has("vllm/model_executor/layers/quantization/modelopt.py",
        "input_size_per_partition // MXFP8_BLOCK_SIZE")
    has("vllm/model_executor/layers/quantization/modelopt.py",
        "layer.weight_block_size = [1, MXFP8_BLOCK_SIZE]")
    has("vllm/model_executor/layers/quantization/utils/quant_utils.py",
        "kMxfp8StaticScale = ScaleDesc(torch.uint8, True, GroupShape(1, 32))")
    has("vllm/model_executor/layers/quantization/mxfp4.py",
        'hidden_size // 2,')                            # w13 打包 2 值/byte
    has("vllm/model_executor/layers/fused_moe/routed_experts.py",
        "self.quant_method.maybe_roundup_sizes(")
    has("vllm/model_executor/kernels/linear/mxfp8/marlin.py",
        "MXFP8 W8A16 GEMM via Marlin (SM80+)")
    has("vllm/model_executor/layers/vocab_parallel_embedding.py",
        "quant_method = UnquantizedEmbeddingMethod()")
    has("vllm/model_executor/layers/quantization/utils/humming/priority.py",
        "_HUMMING_PREFERRED_CAPABILITIES = frozenset({90})")


# ---------------------------------------------------------------------------
# §3 纯 stdlib dispatch/布局 oracle（独立复算，不调用生产函数）
# ---------------------------------------------------------------------------

def oracle_dispatch(qcfg: dict, model_type: str) -> dict:
    """R22 pinned dispatch 的独立重实现（quant_config.py + __init__.py）。"""
    # override_quantization_method：model_type 族 + quant_method → 方法名
    if qcfg.get("quant_method") not in ("fp8", "deepseek_v4_fp8"):
        return {"config": None}
    if model_type not in ("deepseek_v4", "deepseek_v4_text",
                          "deepseek_v41", "deepseek_v41_text"):
        return {"config": None}
    out = {"config": "DeepseekV41FP8Config"}
    # expert_dtype 惰性解析（config 类把它提到 hf_config 顶层，默认 fp4）
    expert_dtype = qcfg.get("expert_dtype", "fp4")
    assert expert_dtype in ("fp4", "fp8")
    out["expert_dtype"] = expert_dtype
    out["is_scale_e8m0"] = expert_dtype == "fp4"
    # get_quant_method（线性 / 路由专家）
    block = qcfg.get("weight_block_size")
    linear_mxfp8 = (block == [32, 32] and out["is_scale_e8m0"])
    out["linear"] = ("ModelOptLinearMethod(mxfp8_static, mxfp8_dynamic)"
                     if linear_mxfp8 else "Fp8LinearMethod(block)")
    out["experts"] = ("Mxfp4MoEMethod(mxfp8_dynamic)" if expert_dtype == "fp4"
                      else "Fp8MoEMethod(block, fp32 scale)")
    # activation_scheme 约束（fp8.py：块量化必须 dynamic）
    if block is not None and qcfg.get("activation_scheme") != "dynamic":
        raise ValueError("block quant requires dynamic")
    return out


def test_oracle_dispatch_official_config():
    qcfg = _official_full()["quantization_config"]
    d = oracle_dispatch(qcfg, "deepseek_v41")
    assert d["config"] == "DeepseekV41FP8Config"
    assert d["expert_dtype"] == "fp4" and d["is_scale_e8m0"] is True
    assert "ModelOptLinearMethod" in d["linear"]
    assert "Mxfp4MoEMethod" in d["experts"]


def test_oracle_dispatch_variants():
    # fp8 专家（Flash-Base）：scale 是 float32，不是 ue8m0
    d = oracle_dispatch({"quant_method": "fp8", "activation_scheme": "dynamic",
                         "weight_block_size": [32, 32],
                         "expert_dtype": "fp8"}, "deepseek_v41")
    assert d["is_scale_e8m0"] is False
    assert "Mxfp4MoEMethod" not in d["experts"]
    # 128×128 块（V3 式）：线性不走 MXFP8 分支
    d2 = oracle_dispatch({"quant_method": "fp8", "activation_scheme": "dynamic",
                          "weight_block_size": [128, 128],
                          "expert_dtype": "fp8"}, "deepseek_v41")
    assert d2["linear"] == "Fp8LinearMethod(block)"
    # 非 deepseek 模型型：override 不认领 → 普通 Fp8Config
    assert oracle_dispatch({"quant_method": "fp8"}, "qwen3_5_moe")[
        "config"] is None
    # 块量化 + static 激活：源码拒绝
    with pytest.raises(ValueError):
        oracle_dispatch({"quant_method": "fp8", "activation_scheme": "static",
                         "weight_block_size": [32, 32],
                         "expert_dtype": "fp4"}, "deepseek_v41")


def test_scale_fmt_key_is_not_consumed_by_loader():
    """from_config 键表（fp8.py L155-172）不含 scale_fmt：声明性键。

    有效 scale dtype 由 expert_dtype 派生（is_scale_e8m0）；本仓按声明
    矛盾拒绝非 ue8m0 值，而不是静默取派生值。"""
    text = (REFS / "vllm/vllm/model_executor/layers/quantization/fp8.py") \
        .read_text(encoding="utf-8")
    start = text.index("def from_config(cls, config")
    end = text.index("def get_quant_method", start)
    body = text[start:end]
    for key in ("quant_method", "activation_scheme", "ignored_layers",
                "weight_block_size", "store_dtype"):
        assert f'"{key}"' in body
    assert "scale_fmt" not in body


# ---------------------------------------------------------------------------
# §4 小张量语义 oracle（纯 Python；不 import torch/vllm）
# ---------------------------------------------------------------------------

def _e8m0_scale_for_block(block: list[float]) -> int:
    """mxfp8_utils._mxfp8_e4m3_quantize_torch 的 scale 语义独立重实现：
    sb = clamp(ceil(log2(amax/448)) + 127, 0, 254)，uint8 存 2 的幂指数。"""
    amax = max(abs(x) for x in block)
    amax = max(amax, math.ldexp(1.0, -126))            # fp32 tiny 下限
    sb = math.ceil(math.log2(amax / 448.0)) + 127.0
    return max(0, min(254, int(sb)))


@pytest.mark.parametrize("block,expect", [
    ([0.0] * 32, 0),                        # 全零：clamp 下限
    ([448.0] + [0.0] * 31, 127),            # amax=448：log2(1)=0
    ([896.0] + [0.0] * 31, 128),            # 2×448：+1
    ([224.0] + [0.0] * 31, 126),            # 0.5×448：−1
    ([1e100] + [0.0] * 31, 254),            # 上限 clamp（远超 fp8 域）
])
def test_e8m0_scale_power_of_two_semantics(block, expect):
    sb = _e8m0_scale_for_block(block)
    assert sb == expect
    # descale = 2^(sb−127)：scale 是 2 的幂（ue8m0 只有指数位）
    descale = math.ldexp(1.0, sb - 127)
    assert descale > 0


def test_mxfp8_scale_layout_counts():
    """(N, K) 权重 → 值 N·K B + scale N·K/32 B（运行时展开后口径）。"""
    v, s = mxfp8_linear_bytes(1280, 5120)
    assert v == 1280 * 5120
    assert s == 1280 * 160            # 5120/32=160
    # checkpoint 侧是 32×32 块：(N/32, K/32) = 40×160 = 6400 B（装载展开 204800）
    assert s == 32 * (1280 // 32) * (5120 // 32)
    # K 不整除 32 拒绝（kernel 断言域）
    with pytest.raises(ValueError):
        mxfp8_linear_bytes(64, 33)
    require_mxfp8_input_divisible(640, "ok")      # h/tp=640 合法


def test_mxfp4_packing_two_values_per_byte():
    """fp4 打包：w13 (2I, H/2) + w2 (H, I/2) uint8；scale 每维 /32。"""
    v, s = mxfp4_expert_bytes(5120, 2304)
    assert v == 2 * 2304 * (5120 // 2) + 5120 * (2304 // 2) \
        == int(1.5 * 5120 * 2304)
    assert s == 2 * 2304 * (5120 // 32) + 5120 * (2304 // 32) \
        == int(3 * 5120 * 2304 / 32)
    # 非整除 64（2 值/byte 与 32 块）拒绝
    with pytest.raises(ValueError):
        mxfp4_expert_bytes(5120, 2305)


def test_marlin_round_up_intermediate_only_when_needed():
    assert marlin_round_up_intermediate(2304) == 2304   # EP：已对齐
    assert marlin_round_up_intermediate(288) == 384     # TP8：288→384
    assert marlin_round_up_hidden(5120) == 5120


# ---------------------------------------------------------------------------
# §5 手算绝对值（与生产公式独立复算对账）
# ---------------------------------------------------------------------------

def test_dense_projection_scale_bytes_tp1_tp8():
    """逐投影 scale 字节（TP1/TP8；值字节 = 维积×1B 两侧不变）。"""
    m = _model()
    scale = lambda n, k: n * (k // 32)
    for tp in (1, 8):
        ng, heads_tp = GROUPS // tp, HEADS // tp
        rows = {r.op: r for r in
                build_model_work(m, "decode", 1, 1024, 1, Q16, tp)}
        expect = {
            "q_a_proj": scale(QRANK, H),
            "kv_proj": scale(D, H),
            "q_b_proj": scale(heads_tp * D, QRANK),
            "o_a_proj": ng * scale(ORANK, HEADS * D // GROUPS),
            "o_b_proj": scale(H, ng * ORANK),
        }
        for op, s in expect.items():
            base = {"q_a_proj": QRANK * H, "kv_proj": D * H,
                    "q_b_proj": QRANK * heads_tp * D,
                    "o_a_proj": ng * ORANK * (HEADS * D // GROUPS),
                    "o_b_proj": ng * ORANK * H}[op]
            assert rows[op].weight_read == base + s, (tp, op)
            assert rows[op].weight_capacity == base + s, (tp, op)


def test_shared_expert_mxfp8_scale_bytes():
    m = _model()
    rows = [r for r in build_model_work(m, "decode", 1, 1024, 1, Q16, 8)
            if r.op == "moe_shared"]
    si = INTER // 8                                   # 288
    values = 3 * H * si
    scales = 2 * si * (H // 32) + H * (si // 32)      # gate_up + down
    assert len(rows) == 40
    assert all(r.weight_read == r.weight_capacity == values + scales
               for r in rows)
    assert scales == 138_240


def test_routed_expert_bytes_packing_and_marlin_roundup():
    """TP8/EP1：I_r=288 → Marlin I_eff=384；per-expert = 51/32·H·I_eff。"""
    m = _model()
    rows = [r for r in build_model_work(m, "decode", 1, 1024, 1,
                                        Q16, 8) if r.op == "moe_routed"]
    i_eff = 384
    per_expert = int(1.5 * H * i_eff) + int(3 * H * i_eff / 32)
    assert per_expert == 3_133_440
    assert all(r.weight_capacity == E * per_expert for r in rows)
    # T=1 期望并集 = k=6（每 token 恰选 6 个不同专家）
    assert all(r.weight_read_expected == pytest.approx(6 * per_expert)
               for r in rows)
    # EP8：I 不分片（2304 已对齐），值+scale = 18,800,640/专家
    rows_ep = [r for r in build_model_work(m, "decode", 1, 1024, 1, Q16, 8,
                                           ep=8, rank=0)
               if r.op == "moe_routed"]
    per_expert_ep = int(1.5 * H * 2304) + int(3 * H * 2304 / 32)
    assert per_expert_ep == 18_800_640
    assert all(r.weight_capacity == 48 * per_expert_ep for r in rows_ep)


def test_embed_lm_head_final_norm_are_bf16_unquantized():
    m = _model()
    rows = {r.op: r for r in
            build_model_work(m, "decode", 1, 1024, 1, Q16, 8)}
    v_r = VOCAB // 8
    assert rows["lm_head"].weight_read == rows["lm_head"].weight_capacity \
        == H * v_r * 2
    assert rows["embed_lookup"].weight_capacity == H * v_r * 2
    assert rows["embed_lookup"].act_read == 1 * H * 2
    assert rows["final_norm"].weight_read == rows["final_norm"].weight_capacity \
        == H * 2
    # 修正事实：旧口径按 dense_w=1B 计 embed/lm_head（少 2×）、final_norm
    # 随 act 位宽（act=16 时 4B/elem 多计 2×）；R22 固定 BF16 2B。


def test_ensem_wkv_scale_bytes():
    m = _model()
    rows = [r for r in build_model_work(m, "decode", 1, 1024, 1, Q16, 8)
            if r.op == "engram_wkv"]
    oc = H * 5 // 8                                   # 3200
    assert len(rows) == 2
    assert all(r.weight_read == r.weight_capacity
               == 6144 * oc * 1 + oc * (6144 // 32) for r in rows)


def test_memory_moe_object_matches_count_capacity():
    """count moe capacity ↔ memory -w-moe- 对象逐层相等（EP off / on）。"""
    m = _model()
    for tp, ep in ((8, 1), (8, 8)):
        rows = build_model_work(m, "decode", 1, 1024, 1, Q16, tp, ep=ep)
        allocs = {a.alloc_id.split("-L")[-1]: a.bytes
                  for a in target_weight_allocations(m, Q16, tp, ep=ep)
                  if "-w-moe-L" in a.alloc_id}
        route_objs = [a.bytes for a in target_weight_allocations(m, Q16, tp,
                                                                 ep=ep)
                      if "-w-moe-route-L" in a.alloc_id]
        assert route_objs and all(b == 384 * 4 * 2 for b in route_objs)
        for li in range(40):
            cap = sum(r.weight_capacity for r in rows
                      if r.layer == li and r.op in ("moe_routed", "moe_shared"))
            cap += 5120 * 384 * 2                    # gate（R21，BF16 行对象）
            assert cap == allocs[str(li)], (tp, ep, li)


# ---------------------------------------------------------------------------
# §6 负例：config 域 + 场景 quant 冻结
# ---------------------------------------------------------------------------

def _full_with_quant(qcfg):
    full = _official_full()
    full["quantization_config"] = qcfg
    return full


@pytest.mark.parametrize("qcfg,match", [
    ({"quant_method": "gptq", "activation_scheme": "dynamic",
      "weight_block_size": [32, 32], "scale_fmt": "ue8m0",
      "expert_dtype": "fp4"}, "quant_method"),
    ({"quant_method": "fp8", "activation_scheme": "static",
      "weight_block_size": [32, 32], "scale_fmt": "ue8m0",
      "expert_dtype": "fp4"}, "activation_scheme"),
    ({"quant_method": "fp8", "activation_scheme": "dynamic",
      "weight_block_size": [128, 128], "scale_fmt": "ue8m0",
      "expert_dtype": "fp4"}, "weight_block_size"),
    ({"quant_method": "fp8", "activation_scheme": "dynamic",
      "weight_block_size": [32, 32], "scale_fmt": "ue8m0",
      "expert_dtype": "fp8"}, "expert_dtype"),
    ({"quant_method": "fp8", "activation_scheme": "dynamic",
      "weight_block_size": [32, 32], "scale_fmt": "float32",
      "expert_dtype": "fp4"}, "scale_fmt"),
])
def test_parse_rejects_out_of_domain_quant_configs(qcfg, match):
    with pytest.raises(ValueError, match=match):
        _parse(_full_with_quant(qcfg))


def test_parse_rejects_missing_quantization_config():
    full = _official_full()
    del full["quantization_config"]
    with pytest.raises(ValueError, match="quantization_config"):
        _parse(full)


def test_unknown_quantization_leaf_remains_an_explicit_gap():
    full = _official_full()
    full["quantization_config"]["future_kernel_mode"] = "unknown"
    model = _parse(full)
    path = "quantization_config.future_kernel_mode"
    assert model.coverage[path]["status"] == "unsupported-semantic"
    assert any(g.startswith(f"{path}:") for g in model.gaps)
    assert not model.structure_complete


def test_partial_draft_quant_uses_draft_defaults():
    scenario = ScenarioV2(draft_quant={"act": 16})
    assert scenario.draft_quant == {"dense_w": 8, "expert_w": 8,
                                    "act": 16, "kv": 8, "state": 32}


def test_scenario_quant_frozen_for_v41():
    """V4.1 场景位宽与冻结配方冲突 → 拒绝（dense_w/expert_w/act 逐键）。"""
    def sc(quant):
        return ScenarioV2(model_id="deepseek_v41_flash", method_id="baseline",
                          draft_id=None, quant=quant,
                          deployment={"mode": "baseline", "target_tp": 8})
    assert sc(V41_QUANT).validate().quant["act"] == 16
    # 默认档（act=8）也被拒绝：SM90 无激活量化 kernel 路径
    with pytest.raises(ValueError, match="quant.act"):
        sc({**V41_QUANT, "act": 8}).validate()
    with pytest.raises(ValueError, match="quant.dense_w"):
        sc({**V41_QUANT, "dense_w": 16}).validate()
    with pytest.raises(ValueError, match="quant.expert_w"):
        sc({**V41_QUANT, "expert_w": 8}).validate()
    # 非 V4.1 模型不受冻结（qwen 无 quantization_config）
    ScenarioV2(model_id="qwen36_35b_a3b", method_id="dflash_v1").validate()


# ---------------------------------------------------------------------------
# §7 门禁事实 + replay 归因冻结
# ---------------------------------------------------------------------------

def test_quant_spec_and_coverage_registered():
    m = _model()
    q = m.extras["quant"]
    assert q["method"] == "deepseek_v4_fp8_dispatch"
    assert q["dense_linear"]["value_itemsize"] == 1
    assert q["dense_linear"]["runtime_scale_block"] == [1, 32]
    assert q["dense_linear"]["ckpt_scale_block"] == [32, 32]
    assert q["routed_experts"]["value_itemsize"] == 0.5
    assert q["act_itemsize"] == 2 and q["act_dtype"] == "bfloat16"
    assert q["act_domain"] == "registered_sm80_sm90_marlin_w8a16_w4a16"
    assert q["expert_backend"] == "marlin_w4a16"
    assert q["expert_round_up"] == {"intermediate": 128, "hidden": 256}
    assert "kv" in q["kv_cache"]  # 缓存不受 quantization_config 影响的注记
    # checkpoint 声明原样入册
    assert q["checkpoint_quantization_config"] == \
        _official_full()["quantization_config"]
    cov = {k: v["status"] for k, v in m.coverage.items()
           if k.startswith("quantization_config")}
    assert set(cov.values()) == {"mapped"}
    assert "quantization_config" not in {g.split(":")[0] for g in m.gaps}


def test_evidence_record_registered_and_not_executed():
    from simverse.validation.registry import build_registry
    rec = build_registry()["ev-vllm-v41-quant"]
    assert rec.kind == "source-checked" and rec.executed is False
    assert len(rec.content_sha256) == 64
    m = _model()
    assert "ev-vllm-v41-quant" in m.evidence_refs
    assert "sm-v41-quant-dispatch" in m.evidence.checked_paths
    assert not m.structure_complete       # swiglu/尾部/核本体缺口仍阻断


def test_formulas_registered():
    import simverse.formula_registry as FR
    for fid in (FR.F_QUANT_MXFP8_LINEAR, FR.F_QUANT_MXFP4_EXPERT,
                FR.F_QUANT_UNQUANTIZED):
        assert fid.startswith("v2.quant_")


def test_replay_attribution_and_freeze():
    att = json.loads((EVID / "attribution.json").read_text(encoding="utf-8"))
    assert set(att["expected_changed_ops"]) == {
        "q_a_proj", "kv_proj", "q_b_proj", "o_a_proj", "o_b_proj",
        "index_q_proj", "moe_routed", "moe_shared", "engram_wkv",
        "final_norm", "lm_head", "embed_lookup"}
    # 证据包自校验
    assert (hashlib.sha256((EVID / "before.json").read_bytes()).hexdigest()
            == att["before_sha256"])
    assert (hashlib.sha256((EVID / "scenario.json").read_bytes()).hexdigest()
            == att["scenario_sha256"])
    prov = json.loads((EVID / "baseline-provenance.json").read_text(
        encoding="utf-8"))
    assert prov["before_sha256"] == att["before_sha256"]
    before = json.loads((EVID / "before.json").read_text(encoding="utf-8"))
    after = json.loads((EVID / "after.json").read_text(encoding="utf-8"))
    old, new = before["work"]["target_decode"], after["work"]["target_decode"]
    AUDITED = set(att["expected_changed_ops"])
    frozen = lambda rows: [r for r in rows if r["op"] not in AUDITED]
    assert frozen(old["rows"]) == frozen(new["rows"])
    for key in old["totals"]:
        assert (new["totals"][key] - old["totals"][key]
                == sum(g["delta"][key] for g in att["by_op"].values())), key
    # 关键 delta 手算对账（TP8/B1/ctx32768，40 层）
    d = att["delta"]
    assert d["weight_read"] == 120_616_960
    assert d["weight_read_expected"] == 221_184_000.0
    assert d["flops"] == 0 and d["act_read"] == 0 and d["rows"] == 0
    # expert：6 distinct/层 × 40 层 × Δ921,600（288→384 打包+scale）
    assert d["weight_read_expected"] == 40 * 6 * (
        3_133_440 - int(3 * 5120 * 288 * 0.5))
    assert att["after_known_peak"] - att["before_known_peak"] \
        == 14_359_147_520
    assert att["execution"].startswith("analytical replay only")
