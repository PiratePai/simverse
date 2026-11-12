"""R06：Qwen3.6 GDN+MoE 小尺寸 CPU forward 对照（D 组 R06）。

Part A（hermetic，CI 必跑）：读 tests/fixtures/qwen36_r06_cpu_fixture.json
（由 tools/probe_qwen36_forward_r06.py 独立执行 pin 的官方 HF forward 产生，
不含任何本仓产物）+ parse_qwen3_5_moe(同一 config) → build_model_work(tp=1)，
逐算子对账（GEMM/conv/attention pairs/MoE assignment/输出门控/递推口径）。

Part B（live）：torch/transformers 可用时重执行探针 runner，对稳健不变量
（GEMM 清单/conv 维度/因果对数/assignment 总量/递推状态形状/logits 形状）
与落盘 artifact 相等；随机路由分布（per_expert_tokens）不参与比较，防跨
torch 版本脆断。

口径披露（不放宽断言，只登记事实）：
  * gdn_recurrence 行 4·T·state_elems 是注册 approximated 口径；官方 torch
    参考递推约 7 ops/elem（chunk L399-406 循环内 decay/读/写/外积/减），
    conv/MoE 内 silu 逐元素与 router softmax/topk 微操作未计，GDN b/a 的
    sigmoid/softplus 未计——与仓库"norm/rope/残差计逐元素、激活不计"的
    既有口径一致。
  * 本仓 conv 状态容量按 kernel−1 列（vLLM 口径）；HF DynamicCache 的
    conv_state 静态宽度为 kernel 列（fixture 观测 state_len=4），差 1 列
    是 HF 缓存实现为 cudagraph 静态地址的填充，不改变逐 token 卷积算量。
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "qwen36_r06_cpu_fixture.json"
# 全 32bit 精度档：本仓行与官方 fixture（fp32 随机权重）同口径
Q32 = {"dense_w": 32, "expert_w": 32, "act": 32, "kv": 32, "state": 32}


def _load_artifact() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _fixture_ir():
    """fixture config → ModelIR（身份指向探针 pin 的官方源，非本仓代码）。"""
    from simverse.model_ir.families import parse_qwen3_5_moe
    from simverse.model_ir.schema import SourceIdentity
    art = _load_artifact()
    src = SourceIdentity(
        repo_id="probe:qwen3_5_moe", revision="hf-transformers@b6401d66",
        config_sha256=hashlib.sha256(json.dumps(
            art["fixture_config"], sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest(),
        captured_at=art["environment"]["date"],
        note="R06 CPU fixture 探针（独立于本仓执行；随机权重缩比）")
    return parse_qwen3_5_moe(dict(art["fixture_config"]), src,
                             "qwen36_r06_fixture", "R06 fixture (2L GDN+Full)")


def _ledger(phases=("prefill", "decode")):
    """两相位本仓账本：{phase: rows}。"""
    from simverse.count import build_model_work
    ir = _fixture_ir()
    out = {}
    if "prefill" in phases:
        out["prefill"] = build_model_work(ir, "prefill", B=1, ctx=5, queries=5,
                                          quant=Q32, prefill=True)
    if "decode" in phases:
        out["decode"] = build_model_work(ir, "decode", B=1, ctx=5, queries=1,
                                         quant=Q32)
    return out


def _gemm_flops(phase_art, layer, module):
    return sum(g["flops"] for g in phase_art["gemms"]
               if g["layer"] == layer and g["module"] == module)


def _row(rows, layer, op):
    hits = [r for r in rows if r.layer == layer and r.op == op]
    assert len(hits) == 1, f"期望恰 1 行 (layer={layer}, op={op})，得到 {len(hits)}"
    return hits[0]


# ---------------------------------------------------------------------------
# Part A：hermetic 对账（无 torch）
# ---------------------------------------------------------------------------

def test_a1_ir_geometry_matches_artifact():
    art = _load_artifact()
    ir = _fixture_ir()
    assert ir.layer_kinds() == art["layer_kinds"] == \
        ["linear_attention", "full_attention"]
    lin, attn_layer = ir.layers[0].linear, ir.layers[1].attn
    g, a = art["phases"]["prefill"]["gdn"], art["phases"]["prefill"]["attn"]
    assert (lin.num_v_heads, lin.num_k_heads, lin.head_k_dim,
            lin.head_v_dim) == (g["nv"], g["nk"], g["head_k_dim"],
                                g["head_v_dim"]) == (4, 2, 8, 8)
    assert lin.conv_kernel == art["phases"]["prefill"]["conv"][0]["kernel"] == 4
    assert lin.state_elems == g["state_elems"] == 256
    assert g["recurrent_state_shape"] == [1, 4, 8, 8]
    # nv/nk=2>1：官方 forward 对 q/k repeat_interleave 到 nv 头（artifact 记录）
    assert g["k_heads_after_repeat"] == g["nv"] == 4
    assert (attn_layer.heads, attn_layer.kv_heads, attn_layer.head_dim) == \
        (a["heads"], a["kv_heads"], a["head_dim"]) == (4, 2, 8)
    assert attn_layer.q_out == 64 == a["q_proj_out"]      # 2× 门控输出
    assert attn_layer.rotary_dim == 2 == a["rotary_dim"]  # partial 0.25·8
    assert attn_layer.output_gated is True
    moe = ir.layers[0].moe
    m = art["phases"]["prefill"]["moe"]
    assert (moe.experts, moe.top_k, moe.intermediate,
            moe.shared_intermediate) == (m["E"], m["k"], m["inter"],
                                         m["shared_inter"]) == (4, 2, 16, 16)
    assert moe.shared_gate is True


def test_a2_per_gemm_rows_exact_both_phases():
    art = _load_artifact()
    led = _ledger()
    for phase in ("prefill", "decode"):
        pa, rows = art["phases"][phase], led[phase]
        T = pa["T"]
        # L0 GDN 投影：in_proj_qkv / in_proj_z / out_proj 同名对账
        for op in ("in_proj_qkv", "in_proj_z", "out_proj"):
            assert _row(rows, 0, op).flops == _gemm_flops(pa, 0, op), \
                (phase, op)
        # in_proj_ba：本仓单行（oc=2·nv）= 官方 in_proj_b + in_proj_a 两条
        assert _row(rows, 0, "in_proj_ba").flops == \
            _gemm_flops(pa, 0, "in_proj_b") + _gemm_flops(pa, 0, "in_proj_a")
        # L1 full-attn 投影
        for op in ("q_proj", "k_proj", "v_proj", "o_proj"):
            assert _row(rows, 1, op).flops == _gemm_flops(pa, 1, op), \
                (phase, op)
        # head：lm_head（词表 64）
        assert _row(rows, -1, "lm_head").flops == _gemm_flops(pa, -1, "lm_head")
        # 每 MoE 层：router（replicated 全量）/shared（3 GEMM 合计）/gate
        for li in (0, 1):
            assert _row(rows, li, "moe_router").flops == \
                pa["moe"]["per_layer"][li]["router"]["router_flops"]
            assert _row(rows, li, "moe_shared").flops == \
                _gemm_flops(pa, li, "gate_proj") + _gemm_flops(pa, li, "up_proj") \
                + _gemm_flops(pa, li, "down_proj")
            assert _row(rows, li, "shared_gate").flops == \
                _gemm_flops(pa, li, "shared_expert_gate")
        # artifact 每条 GEMM 的 token 数与相位一致
        assert all(g["tokens"] == T for g in pa["gemms"])


def test_a3_causal_conv1d_rows_match_both_paths():
    art = _load_artifact()
    led = _ledger()
    for phase, path in (("prefill", "pad_conv"), ("decode", "update")):
        pa = art["phases"][phase]
        conv = pa["conv"][0]
        assert conv["path"] == path and conv["layer"] == 0
        assert (conv["conv_dim"], conv["kernel"]) == (64, 4)
        assert _row(led[phase], 0, "causal_conv1d").flops == conv["flops"]


def test_a4_attention_core_pairs_and_flops():
    art = _load_artifact()
    led = _ledger()
    for phase, (S, Q) in (("prefill", (0, 5)), ("decode", (5, 1))):
        pa = art["phases"][phase]
        row = _row(led[phase], 1, "attention_core")
        pairs = 1 * (S * Q + Q * (Q + 1) // 2)     # 本仓闭式
        assert pairs == pa["attn"]["causal_pairs"] == \
            (15 if phase == "prefill" else 6)
        assert row.flops == pairs * 4 * (4 * 8 + 5)
        assert pa["attn"]["attn_weights_shape"][-1] == \
            pa["attn"]["kv_cache_rows_after"]       # K = S+Q


def test_a5_moe_routed_and_no_replacement_assignment():
    art = _load_artifact()
    led = _ledger()
    for phase in ("prefill", "decode"):
        pa, rows = art["phases"][phase], led[phase]
        T, k = pa["T"], pa["moe"]["k"]
        routed = [r for r in rows if r.op == "moe_routed"]
        assert sum(r.flops for r in routed) == pa["moe"]["routed_flops"]
        # 无放回 top-k：每层 assignment 恰 T·k；两层合计
        assert pa["moe"]["assignment_total"] == 2 * T * k
        for ee in pa["moe"]["per_layer"]:
            assert ee["experts"]["assignment_total"] == T * k


def test_a6_attn_output_gate_row_added():
    """R06 修正：官方 L802 sigmoid(gate)·attn_out 逐元素行（此前漏计）。"""
    art = _load_artifact()
    led = _ledger()
    for phase in ("prefill", "decode"):
        pa = art["phases"][phase]
        rows = led[phase]
        gates = [r for r in rows if r.op == "attn_output_gate"]
        assert len(gates) == 1 and gates[0].layer == 1
        assert gates[0].flops == 2 * pa["T"] * 4 * 8
        assert gates[0].formula == "v2.attn_output_gate"
        # GDN 层（L0）不得出现该行
        assert not any(r.op == "attn_output_gate" and r.layer == 0
                       for r in rows)


def test_a7_gdn_recurrence_convention_and_state_capacity():
    """递推口径（approximated 披露）与每请求常数状态容量。"""
    art = _load_artifact()
    led = _ledger()
    for phase in ("prefill", "decode"):
        pa = art["phases"][phase]
        row = _row(led[phase], 0, "gdn_recurrence")
        assert row.flops == 4 * pa["T"] * 256
        assert row.statistic == "approximated"   # 注册口径，官方参考≈7 ops/elem
        assert row.state_capacity == 1792        # (256·4 + 64·3·4) B
    dec = _row(led["decode"], 0, "gdn_recurrence")
    assert dec.state_read == dec.state_write == 1024   # T=1: 256 elem × 4B
    # 分支事实：prefill chunk（零初始态）/ decode recurrent（带初始态）
    assert art["phases"]["prefill"]["gdn"]["chunk_or_recurrent"] == "chunk"
    assert art["phases"]["decode"]["gdn"]["chunk_or_recurrent"] == \
        "recurrent"
    # HF DynamicCache conv_state 静态宽度 = kernel（4 列），本仓按 kernel−1
    # （vLLM 口径）计容量；差 1 列为 HF 缓存填充，披露不改本仓。
    assert art["phases"]["decode"]["conv"][0]["state_len"] == 4


def test_a8_evidence_registered_and_model_level_not_overclaimed():
    from simverse.model_ir import build_catalog
    from simverse.validation.registry import build_registry
    reg = build_registry()
    rec = reg.get("ev-qwen36-cpu-fixture-r06")
    assert rec is not None
    assert rec.kind == "numeric-checked" and rec.executed is True
    # 内容哈希绑定：registry 实算 == fixture 落盘 SHA256
    assert rec.content_sha256 == hashlib.sha256(
        FIXTURE.read_bytes()).hexdigest()
    q = build_catalog()["models"]["qwen36_35b_a3b"]
    assert "ev-qwen36-cpu-fixture-r06" in q.evidence_refs
    # 模型级三栏不冒充：fixture 级 CPU 对照 ≠ 模型级 cpu-executed
    assert q.evidence.cpu_forward == "not-run"
    assert q.evidence.gpu_measurement == "none"


def test_a9_fixture_independent_of_simverse():
    art = _load_artifact()
    assert art["kind"] == "qwen36-r06-cpu-fixture"
    assert set(art["pinned"]) == {"modeling_sha256", "configuration_sha256"}
    pin_dir = ROOT / "assets" / "identity" / "hf_transformers"
    manifest = json.loads((pin_dir / "manifest.json").read_text(encoding="utf-8"))
    for name, key in (("modeling_qwen3_5_moe.py", "modeling_sha256"),
                      ("configuration_qwen3_5_moe.py", "configuration_sha256")):
        actual = hashlib.sha256((pin_dir / name).read_bytes()).hexdigest()
        assert art["pinned"][key] == manifest["files"][name]["sha256"] == actual
    for key in ("python", "torch", "transformers", "platform", "date"):
        assert art["environment"][key]

    def strings(obj):
        if isinstance(obj, str):
            yield obj
        elif isinstance(obj, dict):
            for kk, vv in obj.items():
                yield kk
                yield from strings(vv)
        elif isinstance(obj, list):
            for vv in obj:
                yield from strings(vv)

    blob = " ".join(strings(art)).lower()
    for banned in ("simverse", "build_model_work", "model_ir", "count/work"):
        assert banned not in blob, f"fixture 引用了本仓产物：{banned}"


# ---------------------------------------------------------------------------
# Part B：live 重执行（torch/transformers 缺失时 skip）
# ---------------------------------------------------------------------------

def _load_probe_module():
    spec = importlib.util.spec_from_file_location(
        "probe_qwen36_forward_r06",
        ROOT / "tools" / "probe_qwen36_forward_r06.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_b1_live_probe_reproduces_artifact_invariants():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    mod = _load_probe_module()
    live = mod.run_probe()
    art = _load_artifact()
    assert live["environment"]["torch"] == art["environment"]["torch"]
    assert live["pinned"] == art["pinned"]
    for phase in ("prefill", "decode"):
        lp, ap = live["phases"][phase], art["phases"][phase]
        # GEMM 清单（层/模块/ic/oc/tokens/flops）逐条相等（与路由随机性无关）
        assert lp["gemms"] == ap["gemms"]
        # conv 维度与执行路径
        assert lp["conv"] == ap["conv"]
        # GDN 状态几何与分支
        assert lp["gdn"]["recurrent_state_shape"] == \
            ap["gdn"]["recurrent_state_shape"]
        assert lp["gdn"]["chunk_or_recurrent"] == \
            ap["gdn"]["chunk_or_recurrent"]
        assert lp["gdn"]["state_elems"] == ap["gdn"]["state_elems"]
        # full-attn 因果对数与 KV 行
        assert lp["attn"]["causal_pairs"] == ap["attn"]["causal_pairs"]
        assert lp["attn"]["attn_weights_shape"] == \
            ap["attn"]["attn_weights_shape"]
        # MoE assignment 总量（无放回 top-k，确定量；分布不比防脆断）
        assert lp["moe"]["assignment_total"] == ap["moe"]["assignment_total"]
        assert lp["moe"]["routed_flops"] == ap["moe"]["routed_flops"]
        assert lp["moe"]["router_flops"] == ap["moe"]["router_flops"]
        # 输出形状
        assert lp["shapes"]["logits_shape"] == ap["shapes"]["logits_shape"]
