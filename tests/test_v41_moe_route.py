"""R21：V4.1 MoE scoring/top-k/分组路由独立 oracle（tests/test_v41_moe_route.py）。

证据链：moe-pin.json 11 文件（vllm @f358f057，blob sha1 对 vllm-tree.json
逐项校验）→ source-checked 建模（count/moe.v41_moe_route_rows、model_ir
extras.moe_route、lowering/memory 路由偏置对象、EP 整除断言）。

独立性声明：§1-§2 的语义 oracle 用**纯 stdlib（math）**重写 pinned Triton
kernel 语义（不 import 生产公式/计数实现）；fp32 下溢按 exp(−87)≈0 门槛
模拟（IEEE-754 单精度解析假设，非 pinned 源码断言——kernel 未执行）。
§3 手算绝对值逐字段核对生产行。§7 replay 冻结核对审计面外行逐字段
不变。source-checked ≠ executed：kernel/CPU forward/GPU 均未执行。
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
REF = ROOT / "docs/rebuild-20261106/references/vllm"
EVID = ROOT / "docs/evidence/moe-route"
PIN = REF / "moe-pin.json"
TREE = ROOT / "docs/rebuild-20261106/references/vllm-tree.json"
CONFIG = ROOT / "assets/config/deepseek_v41_config.json"

Q8 = {"dense_w": 8, "expert_w": 4, "act": 8, "kv": 8, "state": 32}

# V4.1-Flash 常数（config text_config + pinned 源码）
E, K, H = 384, 6, 5120
RSF = 1.5
NEG_GUARD = -1e30


# ---------------------------------------------------------------------------
# §1 pin 完整性（blob sha1 对本地文件与 tree 双侧校验）
# ---------------------------------------------------------------------------

def test_pin_manifest_files_match_tree_and_disk():
    pin = json.loads(PIN.read_text(encoding="utf-8"))
    tree = json.loads(TREE.read_text(encoding="utf-8"))
    by_path = {e["path"]: e["sha"] for e in tree["tree"] if e["type"] == "blob"}
    assert pin["commit"] == tree["sha"] == \
        "f358f0578a83f4e1fe6fc49a492d3ae9362de8e5"
    assert len(pin["files"]) == 11
    for f in pin["files"]:
        data = (REF / f["path"]).read_bytes()
        blob = hashlib.sha1(b"blob %d\x00" % len(data) + data).hexdigest()
        assert blob == f["blob_sha1"] == by_path[f["path"]], f["path"]
        assert hashlib.sha256(data).hexdigest() == f["sha256"]
        assert f["size"] == len(data)
        assert f["tree_match"] is True
    # 核心五文件必须在册（路由语义的直接来源）
    paths = {f["path"] for f in pin["files"]}
    for p in ("vllm/models/deepseek_v4/nvidia/model.py",
              "vllm/model_executor/layers/fused_moe/router/dsv4_topk.py",
              "vllm/model_executor/layers/fused_moe/router/fused_topk_bias_"
              "router.py",
              "vllm/model_executor/layers/fused_moe/router/router_factory.py",
              "vllm/model_executor/layers/fused_moe/router/gate_linear.py"):
        assert p in paths


def test_pinned_source_key_lines_present():
    """路由语义的关键源码片段必须存在于 pin 字节里（防漂移的文本锚点）。"""
    dsv4 = (REF / "vllm/model_executor/layers/fused_moe/router/"
            "dsv4_topk.py").read_text(encoding="utf-8")
    assert "tl.sqrt(tl.where(logits > 20.0, logits" in dsv4        # sqrtsoftplus 护栏
    assert "topk == _TOPK" in dsv4 and "_TOPK = 6" in dsv4
    assert "renormalize" in dsv4 and "gating_output.shape[1] in (256, 384)" in dsv4
    assert "routed_scaling_factor / tl.where" in dsv4               # 归一 ×scale
    assert "tl.min(candidate, axis=0)" in dsv4                      # 平手最小 id
    factory = (REF / "vllm/model_executor/layers/fused_moe/router/"
               "router_factory.py").read_text(encoding="utf-8")
    assert ("num_expert_group and topk_group must be provided when"
            in factory)            # 分组路由需要显式组键
    base = (REF / "vllm/models/deepseek_v4/nvidia/model.py"
            ).read_text(encoding="utf-8")
    assert ("assert self.n_physical_experts % self.ep_size == 0"
            in base)                                                # EP 整除断言
    assert "self.experts_start_idx = ep_rank * self.n_local_experts" in base
    assert "out_dtype=torch.float32" in base                        # FP32 logits
    gate = (REF / "vllm/model_executor/layers/fused_moe/router/"
            "gate_linear.py").read_text(encoding="utf-8")
    assert "class GateLinear(ReplicatedLinear)" in gate             # 复制路由 GEMM
    v41 = (REF / "vllm/models/deepseek_v41/nvidia/model.py"
           ).read_text(encoding="utf-8")
    assert "num_hash_layers=0" in v41                              # 主干无 hash MoE


# ---------------------------------------------------------------------------
# §2 语义 oracle：纯 stdlib 参考（独立于生产实现），镜像 pinned kernel
# ---------------------------------------------------------------------------

def _weight(x: float) -> float:
    """sqrtsoftplus 打分，fp32 语义：>20 取 sqrt(x)；exp(−87) 下溢为 0。

    −87.3 门槛是 **oracle 解析假设**（IEEE-754 单精度最小 normal 2^−126
    ≈1.18e-38 → ln≈−87.3，exp 更小则下溢 0），非 pinned 源码断言——
    Triton/CPU kernel 未执行，实际下溢行为未实测。
    """
    if x != x:                                 # NaN 先于下溢分支（exp(NaN)=NaN）
        return float("nan")
    if x > 20.0:
        return math.sqrt(x)
    e = math.exp(x) if x > -87.0 else 0.0     # fp32 exp 下溢门槛（oracle 假设）
    return math.sqrt(math.log1p(e))


def ref_route(logits, bias, k=K, rsf=RSF):
    """独立参考路由：镜像 dsv4_topk.py L60-128 语义。

    选择列 = weight + bias（NaN→−1e30）；k 轮 argmax、平手最小 id；
    权重列 = 无偏 weight；sum→1 归一 ×rsf（sum≤0 除 1）。
    """
    n = len(logits)
    weights = [_weight(x) for x in logits]
    current = []
    for i in range(n):
        v = weights[i] + bias[i]
        current.append(v if v == v else NEG_GUARD)   # NaN 护栏
    chosen, cur = [], list(current)
    for _ in range(k):
        m = max(cur)
        eid = min(i for i in range(n) if cur[i] == m)  # 平手最小 id
        chosen.append(eid)
        cur[eid] = -math.inf                            # 移除已选
    w = [weights[i] for i in chosen]
    s = sum(w)
    denom = s if s > 0.0 else 1.0
    return chosen, [x * rsf / denom for x in w]


def test_oracle_sqrtsoftplus_values_and_guard():
    assert _weight(0.0) == pytest.approx(math.sqrt(math.log(2.0)))
    # 护栏：>20 直接 sqrt(logits)（softplus(x)≈x 截断）
    assert _weight(21.0) == math.sqrt(21.0)
    assert _weight(20.0) == pytest.approx(math.sqrt(math.log1p(math.exp(20.0))))
    # fp32 下溢：极小 logits 打分为精确 0（sum≤0 护栏入口；−87 门槛为
    # oracle 解析假设，非源码断言）
    assert _weight(-1000.0) == 0.0


def test_oracle_weights_sum_equals_routed_scaling():
    logits = [0.3 * i - 5.0 for i in range(E)]
    bias = [0.002 * ((i * 37) % 11 - 5) for i in range(E)]
    ids, w = ref_route(logits, bias)
    assert len(ids) == K and len(set(ids)) == K      # 无放回 k 个不同专家
    assert sum(w) == pytest.approx(RSF, abs=1e-12)   # 归一后 sum==1.5


def test_oracle_bias_steers_selection_not_weights():
    """noaux_tc：bias 改变选择集合，但权重列仍是无偏 score。"""
    logits = [((i * 13) % 17) * 0.1 - 0.8 for i in range(E)]
    b1 = [0.0] * E
    b2 = list(b1)
    b2[100], b2[101], b2[102] = 5.0, 5.0, 5.0        # 强推 100/101/102
    ids1, w1 = ref_route(logits, b1)
    ids2, w2 = ref_route(logits, b2)
    assert {100, 101, 102} <= set(ids2)
    assert not {100, 101, 102} & set(ids1) or ids1 != ids2
    # 权重 = 被选专家的无偏打分（不含 bias）
    for ids, w in ((ids1, w1), (ids2, w2)):
        for eid, wi in zip(ids, w):
            s = sum(_weight(x) for i, x in enumerate(logits) if i in set(ids))
            assert wi == pytest.approx(_weight(logits[eid]) * RSF / s)


def test_oracle_tie_breaks_to_smallest_expert_id():
    logits = [1.0] * E                                # 全平
    bias = [0.5] * E
    ids, _ = ref_route(logits, bias)
    assert ids == [0, 1, 2, 3, 4, 5]                  # 平手最小 id


def test_oracle_nan_logit_never_selected_over_finite():
    """NaN 选择列被 −1e30 护栏压掉——在有限候选的 score+bias > −1e30 域内
    成立（护栏是常数：极端负 bias 的有限候选可低于 −1e30，该域外不主张）。"""
    logits = [0.0] * E
    logits[7] = float("nan")                          # NaN 分数被 −1e30 护栏压掉
    bias = [0.1] * E
    bias[7] = 100.0                                   # 巨 bias 也救不回 NaN 列
    ids, _ = ref_route(logits, bias)
    assert 7 not in ids


def test_oracle_all_underflow_weights_zero_guard_divide_by_one():
    logits = [-1000.0] * E                            # fp32 softplus 下溢全 0
    bias = [((i * 29) % 13) * 0.01 for i in range(E)]  # bias 决定选择
    ids, w = ref_route(logits, bias)
    assert all(x == 0.0 for x in w)                   # sum=0 → 除 1 → 权重全 0
    assert ids == sorted(
        sorted(range(E), key=lambda i: (-bias[i], i))[:K])


def test_oracle_reference_small_case_properties():
    # 固定小例（确定性）：k 个不同 id、权重正、和=rsf
    logits = [((i * 7) % 23) * 0.05 - 0.6 for i in range(E)]
    bias = [((i * 11) % 19) * 0.01 - 0.09 for i in range(E)]
    ids, w = ref_route(logits, bias)
    assert len(set(ids)) == K
    assert all(x > 0 for x in w)
    assert sum(w) == pytest.approx(RSF, abs=1e-12)


def test_v41_config_has_no_group_keys_flat_topk():
    """V4.1 config 无 n_group/topk_group/num_expert_group → 平坦 top-k。

    pinned router_factory 只有 use_grouped_topk（需 num_expert_group/
    topk_group）才走 GroupedTopKRouter；这些键在 V4.1 config 不存在。
    """
    tc = json.loads(CONFIG.read_text(encoding="utf-8"))["text_config"]
    for key in ("n_group", "topk_group", "num_expert_group", "moe_group"):
        assert key not in tc, key
    assert tc["scoring_func"] == "sqrtsoftplus"
    assert tc["topk_method"] == "noaux_tc"
    assert tc["norm_topk_prob"] is True
    assert tc["routed_scaling_factor"] == 1.5
    assert tc["num_experts_per_tok"] == 6
    assert tc["n_routed_experts"] == 384
    assert tc["n_shared_experts"] == 1


# ---------------------------------------------------------------------------
# §3 生产行手算绝对值（count/moe.v41_moe_route_rows）
# ---------------------------------------------------------------------------

def test_route_rows_hand_computed_absolute_values():
    from simverse.count.moe import v41_moe_route_rows
    from simverse.model_ir import build_catalog
    moe = build_catalog()["models"]["deepseek_v41_flash"].layers[0].moe
    route = build_catalog()["models"]["deepseek_v41_flash"].extras["moe_route"]
    T = 3
    rows = {r["layer_op"]: r
            for r in v41_moe_route_rows(moe, route, T, H, Q8)}
    # gate：2·T·h·E；权重 h·E·1B（dense_w=8）；FP32 logits 写 T·E·4
    g = rows["moe_gate"]
    assert g["flops"] == 2 * 3 * 5120 * 384 == 11_796_480
    assert g["weight_read"] == g["weight_capacity"] == 5120 * 384
    assert g["act_read"] == 3 * 5120
    assert g["act_write"] == 3 * 384 * 4
    # score：T·(9E+2)；偏置 2·E·4（bias+bias_vl）；读 T·(E·4+8)
    s = rows["moe_score"]
    assert s["flops"] == 3 * (9 * 384 + 2) == 10_374
    assert s["weight_read"] == s["weight_capacity"] == 2 * 384 * 4
    assert s["act_read"] == 3 * (384 * 4 + 8)
    assert s["act_write"] == 0
    # topk+归一：T·[k(5E+3)+2k+4]；写 T·k·8（fp32 权重+int32 ids）
    t = rows["moe_topk_renorm"]
    assert t["flops"] == 3 * (6 * (5 * 384 + 3) + 2 * 6 + 4) == 3 * 11_554
    assert t["act_write"] == 3 * 6 * 8
    assert t["weight_read"] == t["weight_capacity"] == 0


def test_build_model_work_route_rows_and_shared_no_gate():
    from simverse.count import build_model_work
    from simverse.model_ir import build_catalog
    m = build_catalog()["models"]["deepseek_v41_flash"]
    rows = build_model_work(m, "decode", 1, 1024, 1, Q8, tp=8)
    gate = [r for r in rows if r.op == "moe_gate"]
    score = [r for r in rows if r.op == "moe_score"]
    topk = [r for r in rows if r.op == "moe_topk_renorm"]
    old_router = [r for r in rows if r.op == "moe_router"]
    shared = [r for r in rows if r.op == "moe_shared"]
    shared_gate = [r for r in rows if r.op == "shared_gate"]
    assert len(gate) == len(score) == len(topk) == 40   # 全 40 MoE 层
    assert not old_router                               # 通用行已被替换
    # V4.1 共享专家无 gate（Qwen3.6 才有 sigmoid gate 行）
    assert len(shared) == 40 and not shared_gate
    assert m.layers[0].moe.shared_intermediate == 2304  # 1×moe_inter
    # 公式绑定：三行各引用 v2.moe_route_*，不得挪用通用公式解释
    from simverse import formula_registry as FR
    assert {r.formula for r in gate} == {FR.F_MOE_ROUTE_GATE}
    assert {r.formula for r in score} == {FR.F_MOE_ROUTE_SCORE}
    assert {r.formula for r in topk} == {FR.F_MOE_ROUTE_TOPK}
    for fid in (FR.F_MOE_ROUTE_GATE, FR.F_MOE_ROUTE_SCORE, FR.F_MOE_ROUTE_TOPK):
        ent = FR.get(fid)
        assert "dsv4_topk.py" in ent["source"] or "model.py" in ent["source"]


def test_route_rows_replicated_across_tp_and_ep():
    """路由三行 per-rank 全量复制：tp/ep 不改变逐行值；跨 rank 和=单 rank×tp。"""
    from simverse.count import build_model_work
    from simverse.model_ir import build_catalog
    m = build_catalog()["models"]["deepseek_v41_flash"]
    ops = ("moe_gate", "moe_score", "moe_topk_renorm")
    def sig(tp, ep):
        rows = [r for r in build_model_work(m, "decode", 1, 512, 1, Q8,
                                            tp=tp, ep=ep, rank=0)
                if r.op in ops]
        return [(r.op, r.flops, r.weight_read, r.weight_capacity,
                 r.act_read, r.act_write) for r in rows]
    base = sig(1, 1)
    assert len(base) == 120
    for tp, ep in ((2, 1), (4, 1), (8, 1), (8, 8)):
        assert sig(tp, ep) == base, (tp, ep)   # 复制：逐行逐字段不变


# ---------------------------------------------------------------------------
# §4 EP 归属：整除断言（V4.1 pinned）与本地专家数
# ---------------------------------------------------------------------------

def test_ep_divisibility_enforced_count_and_memory():
    from simverse.count import build_model_work
    from simverse.model_ir import build_catalog
    from simverse.lowering.memory import target_weight_allocations
    m = build_catalog()["models"]["deepseek_v41_flash"]
    # 384 % 7 ≠ 0 → 拒绝（pinned 断言，无余数 rank 分布）
    with pytest.raises(ValueError, match="整除"):
        build_model_work(m, "decode", 1, 512, 1, Q8, tp=8, ep=7)
    with pytest.raises(ValueError, match="整除"):
        target_weight_allocations(m, Q8, 8, ep=7, rank=0)
    # 整除组合放行（默认 tp8/ep8 → 每 rank 48 专家）
    for ep in (1, 2, 3, 4, 6, 8):
        build_model_work(m, "decode", 1, 512, 1, Q8, tp=8, ep=ep)


def test_ep8_local_experts_and_routed_share():
    from simverse.count import build_model_work
    from simverse.model_ir import build_catalog
    m = build_catalog()["models"]["deepseek_v41_flash"]
    rows = build_model_work(m, "decode", 1, 512, 1, Q8, tp=8, ep=8, rank=3)
    routed = [r for r in rows if r.op == "moe_routed"]
    assert len(routed) == 40
    for r in routed:
        # E_l=48：share=48/384=1/8；flops = 2·T·k·3·h·inter/8
        assert r.flops == 2 * 1 * 6 * 3 * 5120 * 2304 * 48 // 384
        assert "E_local=48" in r.shape


# ---------------------------------------------------------------------------
# §5 路由偏置容量对象（memory ↔ count 对账）
# ---------------------------------------------------------------------------

def test_memory_route_bias_allocations_match_count():
    from simverse.lowering.memory import target_weight_allocations
    from simverse.count import build_model_work
    from simverse.model_ir import build_catalog
    m = build_catalog()["models"]["deepseek_v41_flash"]
    allocs = [a for a in target_weight_allocations(m, Q8, 8, ep=1, rank=0)
              if a.alloc_id.endswith("-w-moe-route-L0")]
    # 40 层 ×（bias 1536 + bias_vl 1536）= 每层 3072 B（FP32 复制）
    assert len(allocs) == 1 and allocs[0].bytes == 3072
    all_route = [a for a in target_weight_allocations(m, Q8, 8, ep=1, rank=0)
                 if "-w-moe-route-L" in a.alloc_id]
    assert len(all_route) == 40
    assert sum(a.bytes for a in all_route) == 40 * 3072 == 122_880
    # count 侧 weight_capacity（moe_score 行）与 memory 侧逐层相等
    rows = [r for r in build_model_work(m, "decode", 1, 512, 1, Q8, tp=8)
            if r.op == "moe_score"]
    assert sum(r.weight_capacity for r in rows) == 122_880


def test_route_spec_constants_from_pinned_source():
    from simverse.model_ir import build_catalog
    spec = build_catalog()["models"]["deepseek_v41_flash"].extras["moe_route"]
    assert spec["scoring"] == "sqrtsoftplus"
    assert spec["topk_method"] == "noaux_tc"
    assert spec["renormalize"] is True
    assert spec["routed_scaling"] == 1.5
    assert spec["gate_out_dtype"] == "fp32" and spec["logits_itemsize"] == 4
    assert spec["bias_bytes"] == 384 * 4
    # V4.1-Flash 是视觉 checkpoint（vision 32 层）→ bias_vl 存在
    assert spec["bias_vl_bytes"] == 384 * 4
    assert spec["hash_layers"] == 0            # 主干无 hash MoE
    assert spec["grouped"] is False            # 平坦 top-k，无分组
    assert spec["shared_expert_weight"] == 1.0  # 共享专家归一后追加，无 gate


# ---------------------------------------------------------------------------
# §6 config 域负例（非法路由组合拒绝，不静默回退）
# ---------------------------------------------------------------------------

def _v41_parse(tc_mutator, full_mutator=None):
    from simverse.model_ir.families import parse_v41_csa2
    from simverse.model_ir.schema import SourceIdentity
    full = json.loads(CONFIG.read_text(encoding="utf-8"))
    tc = full["text_config"]
    tc_mutator(tc)
    if full_mutator:
        full_mutator(full)
    src = SourceIdentity("deepseek-ai/DeepSeek-V4.1-Flash",
                         "314113f8a7dadb388215396ba5fbee2c610a04d6",
                         "sha256:fixture", "测试")
    return parse_v41_csa2(tc, src, "t", "t", full_config=full,
                          text_prefix="text_config")


def test_config_domain_rejects_unsupported_routing():
    with pytest.raises(ValueError, match="scoring_func"):
        _v41_parse(lambda tc: tc.update(scoring_func="softmax"))
    with pytest.raises(ValueError, match="topk_method"):
        _v41_parse(lambda tc: tc.update(topk_method="greedy"))
    with pytest.raises(ValueError, match="norm_topk_prob"):
        _v41_parse(lambda tc: tc.update(norm_topk_prob=False))


def test_config_domain_vision_absent_disables_bias_vl():
    spec = _v41_parse(lambda tc: None,
                      lambda f: f["vision_config"].update(num_hidden_layers=0))
    assert spec.extras["moe_route"]["bias_vl_bytes"] == 0
    assert spec.extras["moe_route"]["bias_bytes"] == 384 * 4


# ---------------------------------------------------------------------------
# §7 structure gate 不解除 + replay 归因冻结
# ---------------------------------------------------------------------------

def test_structure_complete_still_false_and_gaps():
    from simverse.model_ir import build_catalog
    m = build_catalog()["models"]["deepseek_v41_flash"]
    gap_keys = {g.split(":")[0] for g in m.gaps}
    assert not m.structure_complete
    assert {"swiglu_limit", "quantization_config"} <= gap_keys
    assert any("compress_ratios[40:43]" in g for g in m.gaps)
    # 证据四栏：source-checked 未冒充 executed
    assert m.evidence.cpu_forward == "not-run"
    assert m.evidence.gpu_measurement == "none"
    assert "sm-v41-moe-route" in m.evidence.checked_paths
    assert "ev-vllm-v41-moe" in m.evidence_refs


def test_replay_attribution_and_freeze():
    att = json.loads((EVID / "attribution.json").read_text(encoding="utf-8"))
    assert set(att["expected_changed_ops"]) == {
        "moe_router", "moe_gate", "moe_score", "moe_topk_renorm"}
    before = json.loads((EVID / "before.json").read_text(encoding="utf-8"))
    after = json.loads((EVID / "after.json").read_text(encoding="utf-8"))
    old, new = before["work"]["target_decode"], after["work"]["target_decode"]
    # 审计面外的行逐字段冻结
    AUDITED = set(att["expected_changed_ops"])
    def frozen(rows):
        return [r for r in rows if r["op"] not in AUDITED]
    assert frozen(old["rows"]) == frozen(new["rows"])
    # 逐字段 delta = Σ by_op delta
    for key in old["totals"]:
        assert (new["totals"][key] - old["totals"][key]
                == sum(g["delta"][key] for g in att["by_op"].values())), key
    # 旧 moe_router 行消失、新三行进入；gate flops 与通用行相同（拆行不重计）
    ops_before = {r["op"] for r in old["rows"]}
    ops_after = {r["op"] for r in new["rows"]}
    assert "moe_router" in ops_before and "moe_router" not in ops_after
    assert {"moe_gate", "moe_score", "moe_topk_renorm"} <= ops_after
