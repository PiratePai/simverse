"""R06 fixture probe: tiny-size CPU forward of the pinned Qwen3.6 HF source.

Loads the in-repo pinned qwen3_5_moe configuration/modeling files (HF
transformers @b6401d66) via importlib, executes a random-weight two-layer
model (GDN linear-attention layer 0 + full-attention layer 1, 256->4E MoE
scale-down) for one prefill and one cached decode step, and records every
observable operator geometry (per-Linear GEMM shapes, depthwise causal conv
dims, GDN chunk/recurrent state shapes, full-attention weights shape and
causal pair count, MoE router logits / no-replacement top-k assignment).

The artifact is written to tests/fixtures/qwen36_r06_cpu_fixture.json and is
consumed by tests/test_qwen36_forward_r06.py (hermetic reconciliation against
simverse build_model_work). This module must stay independent of simverse so
the fixture is produced by the pinned official forward alone.

Random weights only: this is a fixture-level CPU execution; it does not
validate real checkpoint weights, MTP, the vision tower, mrope position ids,
or fused kernels (fla / causal-conv1d are not installed here; the pinned
PyTorch reference implementations run instead).

Hook/patch notes (verified against transformers 5.22.1):
  * use_kernelized_func is a no-op in this version, so module-level
    monkeypatching of torch_chunk_gated_delta_rule /
    torch_recurrent_gated_delta_rule / causal_conv1d_fn /
    causal_conv1d_update works: Qwen3_5MoeGatedDeltaNet.forward resolves
    them as module globals at call time.
  * The nn.Conv1d module itself is never called during forward (its weight
    is passed to F.conv1d inside causal_conv1d_fn/update), so a forward hook
    on nn.Conv1d would not fire; both conv paths are recorded by wrapping
    the two module-level functions instead.
  * ALL_ATTENTION_FUNCTIONS has no "eager" entry, so
    get_interface("eager", fallback) returns the pinned module's own
    eager_attention_forward; wrapping it captures attn_weights shapes.
"""

import hashlib
import importlib.util
import json
import platform
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PIN_DIR = REPO / "assets" / "identity" / "hf_transformers"
DEFAULT_OUT = REPO / "tests" / "fixtures" / "qwen36_r06_cpu_fixture.json"
SEED = 20261112

FIXTURE_CONFIG = dict(
    vocab_size=64, hidden_size=32, num_hidden_layers=2,
    layer_types=["linear_attention", "full_attention"],
    num_attention_heads=4, num_key_value_heads=2, head_dim=8,
    linear_num_key_heads=2, linear_num_value_heads=4,
    linear_key_head_dim=8, linear_value_head_dim=8, linear_conv_kernel_dim=4,
    moe_intermediate_size=16, shared_expert_intermediate_size=16,
    num_experts_per_tok=2, num_experts=4,
    rope_parameters={"rope_type": "default", "partial_rotary_factor": 0.25},
)
PREFILL_IDS = [[1, 2, 3, 4, 5]]
DECODE_IDS = [[6]]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_pinned_modules():
    """importlib 注册 pin 的官方 configuration/modeling（挂 pinned_* 名下）。"""
    import torch  # noqa: F401  (探针依赖 torch/transformers 运行环境)
    import transformers

    config_source = PIN_DIR / "configuration_qwen3_5_moe.py"
    config_spec = importlib.util.spec_from_file_location(
        "transformers.models.qwen3_5_moe.pinned_configuration_qwen3_5_moe",
        config_source)
    config_module = importlib.util.module_from_spec(config_spec)
    sys.modules[config_spec.name] = config_module
    config_spec.loader.exec_module(config_module)

    source = PIN_DIR / "modeling_qwen3_5_moe.py"
    spec = importlib.util.spec_from_file_location(
        "transformers.models.qwen3_5_moe.pinned_modeling_qwen3_5_moe", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return config_module, module, torch, transformers


def _layer_of(name: str) -> int:
    """'model.layers.1.mlp.gate_proj' → 1；层外（lm_head 等）→ -1。"""
    parts = name.split(".")
    if len(parts) > 2 and parts[0] == "model" and parts[1] == "layers":
        return int(parts[2])
    return -1


def _pairs_from_weights(shape) -> int:
    """因果对数（j≤i）：Q 行第 i 行可见 min(i+1+(K−Q), K) 个键。"""
    q, k = shape[-2], shape[-1]
    s = k - q
    return sum(min(i + 1 + s, k) for i in range(q))


def run_probe():
    """执行官方 forward 并采集 artifact dict（不落盘；供 main/测试复用）。"""
    config_module, modeling, torch, transformers = load_pinned_modules()

    # 记录容器（每相位清空）；active_layer 由 linear_attn 前 hook 设置，
    # 供模块级 conv 包裹函数归属层号（conv 函数本身无层信息）。
    rec = {"gemms": [], "conv": [], "gdn": [], "attn": []}
    state = {"active_gdn_layer": None}
    router_recs, expert_recs, cos_widths = [], [], []

    # ---- 1) GEMM：nn.Linear 前向 hook（ic/oc 取自 weight.shape） --------
    def make_linear_hook(name):
        def hook(_m, inputs, _out):
            inp = inputs[0]
            tokens = (int(inp.shape[0] * inp.shape[1]) if inp.dim() == 3
                      else int(inp.shape[0]))
            oc, ic = _m.weight.shape
            rec["gemms"].append(dict(
                layer=_layer_of(name), module=name.split(".")[-1],
                ic=int(ic), oc=int(oc), tokens=tokens,
                flops=2 * tokens * int(ic) * int(oc)))
        return hook

    # ---- 2) conv：包裹模块级 causal_conv1d_fn（prefill pad 路径）与
    #          causal_conv1d_update（decode 单步路径）。
    #          注：nn.Conv1d 模块自身从不被调用（weight 直传 F.conv1d），
    #          模块 hook 不触发，故在函数级包裹。 ------------------------
    orig_conv_fn = modeling.causal_conv1d_fn
    orig_conv_upd = modeling.causal_conv1d_update

    def conv_fn_rec(hidden_states, weight, bias=None, activation=None, **kw):
        out = orig_conv_fn(hidden_states, weight, bias, activation, **kw)
        _b, c, t = hidden_states.shape
        rec["conv"].append(dict(
            layer=state["active_gdn_layer"], conv_dim=int(c),
            kernel=int(weight.shape[-1]), tokens=int(t),
            flops=2 * int(t) * int(c) * int(weight.shape[-1]),
            path="pad_conv"))
        return out

    def conv_upd_rec(hidden_states, conv_state, weight, bias=None,
                     activation=None, **kw):
        out = orig_conv_upd(hidden_states, conv_state, weight, bias,
                            activation, **kw)
        _b, c, t = hidden_states.shape
        rec["conv"].append(dict(
            layer=state["active_gdn_layer"], conv_dim=int(c),
            kernel=int(weight.shape[-1]), tokens=int(t),
            state_len=int(conv_state.shape[-1]),
            flops=2 * int(t) * int(c) * int(weight.shape[-1]),
            path="update"))
        return out

    modeling.causal_conv1d_fn = conv_fn_rec
    modeling.causal_conv1d_update = conv_upd_rec

    # ---- 3) GDN 递推：包裹 chunk / recurrent PyTorch 参考实现 -----------
    orig_chunk = modeling.torch_chunk_gated_delta_rule
    orig_recurrent = modeling.torch_recurrent_gated_delta_rule

    def _gdn_common(kind, query, key, value, g, beta, kw, call):
        out, st = call(query, key, value, g=g, beta=beta, **kw)
        rec["gdn"].append(dict(
            chunk_or_recurrent=kind,
            chunk_size=(int(kw["chunk_size"]) if kind == "chunk"
                        and kw.get("chunk_size") is not None else
                        (64 if kind == "chunk" else None)),
            tokens=int(query.shape[1]),
            query_shape=list(query.shape), key_shape=list(key.shape),
            value_shape=list(value.shape), g_shape=list(g.shape),
            beta_shape=list(beta.shape),
            initial_state_shape=(list(kw["initial_state"].shape)
                                 if kw.get("initial_state") is not None
                                 else None),
            core_attn_out_shape=list(out.shape),
            final_state_shape=(list(st.shape) if st is not None else None)))
        return out, st

    def chunk_rec(query, key, value, g=None, beta=None, **kw):
        return _gdn_common("chunk", query, key, value, g, beta, kw, orig_chunk)

    def recurrent_rec(query, key, value, g=None, beta=None, **kw):
        return _gdn_common("recurrent", query, key, value, g, beta, kw,
                           orig_recurrent)

    modeling.torch_chunk_gated_delta_rule = chunk_rec
    modeling.torch_recurrent_gated_delta_rule = recurrent_rec

    # ---- 4) full-attn：包裹 eager 核（attn_weights 形状/因果对数）与
    #          self_attn 前 hook（position_embeddings cos 宽 = rotary_dim）。
    orig_eager = modeling.eager_attention_forward

    def eager_rec(module, query, key, value, attention_mask, scaling,
                  dropout=0.0, **kw):
        out, w = orig_eager(module, query, key, value, attention_mask,
                            scaling, dropout, **kw)
        rec["attn"].append(dict(
            heads=int(query.shape[1]), q_len=int(query.shape[2]),
            kv_heads=int(key.shape[1]),
            attn_weights_shape=list(w.shape),
            causal_pairs=_pairs_from_weights(w.shape)))
        return out, w

    modeling.eager_attention_forward = eager_rec

    def attn_pre_hook(_m, _args, kwargs):
        pe = kwargs.get("position_embeddings")
        if pe is not None:
            cos_widths.append(int(pe[0].shape[-1]))

    # ---- 5) MoE：router 前 hook（logits 形状）+ experts.forward 包裹 ---
    def make_gate_hook(layer_idx):
        def hook(m, _inp, out):
            logits = out[0]
            e, h = m.weight.shape
            router_recs.append(dict(
                layer=layer_idx, router_logits_shape=list(logits.shape),
                E=int(e), hidden=int(h),
                router_flops=2 * int(logits.shape[0]) * int(h) * int(e)))
        return hook

    def wrap_experts(layer_idx, experts):
        orig_forward = experts.forward

        def wrapped(hidden_states, top_k_index, top_k_weights):
            idx = top_k_index.detach().cpu().reshape(-1).tolist()
            counts = [0] * experts.num_experts
            for e in idx:
                counts[int(e)] += 1
            expert_recs.append(dict(
                layer=layer_idx,
                assignment_shape=list(top_k_index.shape),
                assignment_total=len(idx),
                distinct_experts=sum(1 for c in counts if c),
                per_expert_tokens=counts))
            return orig_forward(hidden_states, top_k_index, top_k_weights)

        experts.forward = wrapped

    # ---- 建模与 hook 注册 ---------------------------------------------
    torch.manual_seed(SEED)
    config = config_module.Qwen3_5MoeTextConfig(**FIXTURE_CONFIG)
    config._attn_implementation = "eager"   # 需要观测 attn_weights 形状
    model = modeling.Qwen3_5MoeForCausalLM(config).eval()

    handles = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            handles.append(m.register_forward_hook(make_linear_hook(name)))
        elif type(m).__name__ == "Qwen3_5MoeGatedDeltaNet":
            layer_idx = _layer_of(name)
            handles.append(m.register_forward_pre_hook(
                lambda _m, _a, _k, li=layer_idx: state.update(
                    active_gdn_layer=li), with_kwargs=True))
        elif type(m).__name__ == "Qwen3_5MoeAttention":
            handles.append(m.register_forward_pre_hook(
                attn_pre_hook, with_kwargs=True))
        elif type(m).__name__ == "Qwen3_5MoeTopKRouter":
            handles.append(m.register_forward_hook(
                make_gate_hook(_layer_of(name))))
        elif type(m).__name__ == "Qwen3_5MoeExperts":
            wrap_experts(_layer_of(name), m)

    # ---- 执行两相位 ----------------------------------------------------
    def run_phase(input_ids, past):
        for key in rec:
            rec[key] = []
        router_recs.clear()
        expert_recs.clear()
        cos_widths.clear()
        with torch.inference_mode():
            return model(input_ids=torch.tensor(input_ids),
                         past_key_values=past, use_cache=True)

    prefill = run_phase(PREFILL_IDS, None)
    pre = _snapshot_phase("prefill", PREFILL_IDS, rec, router_recs,
                          expert_recs, cos_widths, prefill, model)
    decode = run_phase(DECODE_IDS, prefill.past_key_values)
    dec = _snapshot_phase("decode", DECODE_IDS, rec, router_recs,
                          expert_recs, cos_widths, decode, model)

    for h in handles:
        h.remove()

    artifact = dict(
        schema_version="1",
        kind="qwen36-r06-cpu-fixture",
        environment=dict(
            python=platform.python_version(),
            torch=torch.__version__, transformers=transformers.__version__,
            platform=platform.platform(), date=date.today().isoformat()),
        pinned=dict(
            modeling_sha256=_sha256_file(PIN_DIR / "modeling_qwen3_5_moe.py"),
            configuration_sha256=_sha256_file(
                PIN_DIR / "configuration_qwen3_5_moe.py")),
        fixture_config=FIXTURE_CONFIG,
        seed=SEED,
        layer_kinds=list(FIXTURE_CONFIG["layer_types"]),
        phases=dict(prefill=pre, decode=dec),
    )
    _self_check(artifact)
    return artifact


def _snapshot_phase(name, input_ids, rec, router_recs, expert_recs,
                    cos_widths, output, model):
    """把单相位记录整理成键稳定的 dict。"""
    convs = list(rec["conv"])
    assert len(convs) == 1 and convs[0]["layer"] is not None, \
        f"{name}: 期望恰 1 次 GDN conv（层号已知）"
    g = rec["gdn"][0]
    nv, dv = g["value_shape"][-2], g["value_shape"][-1]
    dk = g["key_shape"][-1]
    # kernel 边界的 key 已被 repeat_interleave 到 nv 头（nv/nk=2>1），
    # 真实 nk 只能由投影几何反推：in_proj_qkv oc = 2·key_dim + value_dim，
    # in_proj_z oc = value_dim → key_dim = (qkv_oc − z_oc)/2，nk = key_dim/dk。
    qkv_oc = next(x["oc"] for x in rec["gemms"] if x["module"] == "in_proj_qkv")
    z_oc = next(x["oc"] for x in rec["gemms"] if x["module"] == "in_proj_z")
    value_dim = int(z_oc)
    key_dim = (int(qkv_oc) - value_dim) // 2
    nk = key_dim // int(dk)
    assert g["key_shape"][-2] == nv and g["query_shape"][-2] == nv, \
        "nv/nk>1 时官方 forward 对 q/k repeat_interleave 到 nv 头"
    gdn = dict(
        nv=int(nv), nk=int(nk), head_k_dim=int(dk), head_v_dim=int(dv),
        k_heads_after_repeat=int(g["key_shape"][-2]),
        key_dim=int(key_dim), value_dim=value_dim,
        state_elems=int(nv * dk * dv),
        recurrent_state_shape=(g["final_state_shape"]
                               or g["initial_state_shape"]),
        core_attn_out_shape=g["core_attn_out_shape"],
        chunk_or_recurrent=g["chunk_or_recurrent"],
        chunk_size=g["chunk_size"], tokens=g["tokens"],
        initial_state_shape=g["initial_state_shape"],
        query_shape=g["query_shape"], g_shape=g["g_shape"],
        beta_shape=g["beta_shape"])
    a = rec["attn"][0]
    q_proj = next(gem for gem in rec["gemms"] if gem["module"] == "q_proj")
    heads, d = a["heads"], FIXTURE_CONFIG["head_dim"]
    attn = dict(
        heads=int(heads), kv_heads=int(a["kv_heads"]), head_dim=int(d),
        q_proj_out=int(q_proj["oc"]),
        q_proj_out_shape=[1, len(input_ids[0]), int(q_proj["oc"])],
        gate_shape=[1, len(input_ids[0]), int(heads) * int(d)],
        rotary_dim=int(cos_widths[0]) if cos_widths else -1,
        attn_weights_shape=a["attn_weights_shape"],
        causal_pairs=int(a["causal_pairs"]),
        kv_cache_rows_after=int(a["attn_weights_shape"][-1]))
    assert len(router_recs) == 2 and len(expert_recs) == 2, \
        f"{name}: 期望 2 层各一次 router/experts"
    h = FIXTURE_CONFIG["hidden_size"]
    inter = FIXTURE_CONFIG["moe_intermediate_size"]
    moe = dict(
        E=router_recs[0]["E"], k=int(FIXTURE_CONFIG["num_experts_per_tok"]),
        inter=int(inter),
        shared_inter=int(FIXTURE_CONFIG["shared_expert_intermediate_size"]),
        router_logits_shape=router_recs[0]["router_logits_shape"],
        router_flops=router_recs[0]["router_flops"],
        per_layer=[dict(router=rr, experts=ee)
                   for rr, ee in zip(router_recs, expert_recs)],
        assignment_total=sum(ee["assignment_total"] for ee in expert_recs),
        per_expert_tokens=[ee["per_expert_tokens"] for ee in expert_recs],
        distinct_experts=[ee["distinct_experts"] for ee in expert_recs],
        routed_flops=sum(
            sum(6 * t_e * h * inter for t_e in ee["per_expert_tokens"])
            for ee in expert_recs))
    # norm 模块宽度清单（构造期确定，两相位一致；按类型+名字归类）
    norm_widths = {}
    for mod_name, m in model.named_modules():
        w = getattr(m, "weight", None)
        if w is None or w.dim() != 1:
            continue
        tname = type(m).__name__
        base = mod_name.split(".")[-1]
        if tname == "Qwen3_5MoeRMSNorm":
            if base == "input_layernorm":
                norm_widths["attn_norm"] = int(w.shape[0])
            elif base == "post_attention_layernorm":
                norm_widths["mlp_norm"] = int(w.shape[0])
            elif mod_name == "model.norm":
                norm_widths["final_norm"] = int(w.shape[0])
            elif base == "q_norm":
                norm_widths["q_norm"] = int(w.shape[0])
            elif base == "k_norm":
                norm_widths["k_norm"] = int(w.shape[0])
        elif tname == "Qwen3_5MoeRMSNormGated":
            norm_widths["rmsnorm_gated"] = int(w.shape[0])
    seg = next(m for n, m in model.named_modules()
               if n.endswith("shared_expert_gate"))
    shapes = dict(
        logits_shape=list(output.logits.shape),
        shared_expert_gate_shape=[int(seg.weight.shape[1]),
                                  int(seg.weight.shape[0])],
        norm_widths=norm_widths)
    return dict(
        B=len(input_ids),
        S=(0 if name == "prefill" else len(PREFILL_IDS[0])),
        Q=len(input_ids[0]), T=len(input_ids[0]) * len(input_ids),
        gemms=rec["gemms"], conv=convs, gdn=gdn, attn=attn, moe=moe,
        shapes=shapes)


def _self_check(art):
    """探针内置自洽断言：任何失败 → 异常（main 非零退出）。"""
    for phase, want_t in (("prefill", 5), ("decode", 1)):
        p = art["phases"][phase]
        assert p["T"] == want_t
        for g in p["gemms"]:
            assert g["flops"] == 2 * g["tokens"] * g["ic"] * g["oc"], (phase, g)
        for c in p["conv"]:
            assert c["flops"] == 2 * c["tokens"] * c["conv_dim"] * c["kernel"]
            assert (c["conv_dim"], c["kernel"]) == (64, 4)
        g = p["gdn"]
        assert (g["nv"], g["nk"], g["head_k_dim"], g["head_v_dim"]) == \
            (4, 2, 8, 8), g
        assert g["state_elems"] == 256
        assert g["recurrent_state_shape"] == [1, 4, 8, 8]
        a = p["attn"]
        assert a["heads"] == 4 and a["kv_heads"] == 2
        assert a["q_proj_out"] == 64 and a["rotary_dim"] == 2
        assert a["attn_weights_shape"] == \
            ([1, 4, 5, 5] if phase == "prefill" else [1, 4, 1, 6]), a
        assert a["causal_pairs"] == (15 if phase == "prefill" else 6)
        m = p["moe"]
        for ee in m["per_layer"]:       # 无放回 top-k：每层恰 T·k 个 assignment
            assert ee["experts"]["assignment_total"] == want_t * m["k"]
        assert m["assignment_total"] == 2 * want_t * m["k"]
        # 2 层 MoE 各 T·k 个 assignment：routed_flops = 6·Σt_e·h·inter
        assert m["routed_flops"] == 2 * want_t * m["k"] * 3 * 32 * \
            m["inter"] * 2, m["routed_flops"]
        assert p["shapes"]["logits_shape"] == [1, want_t, 64]
    # 分支：prefill chunk（零初始态）/ decode recurrent（带初始态）
    assert art["phases"]["prefill"]["gdn"]["chunk_or_recurrent"] == "chunk"
    assert art["phases"]["prefill"]["gdn"]["initial_state_shape"] is None
    assert art["phases"]["prefill"]["conv"][0]["path"] == "pad_conv"
    assert art["phases"]["decode"]["gdn"]["chunk_or_recurrent"] == "recurrent"
    assert art["phases"]["decode"]["gdn"]["initial_state_shape"] == [1, 4, 8, 8]
    dconv = art["phases"]["decode"]["conv"][0]
    # HF DynamicCache 的 conv state 静态宽度 = kernel（4 列，cache_utils
    # lazy_initialization 按 conv_kernel_size 建，供 cudagraph 静态地址），
    # 比数学最小值 kernel−1=3 多 1 列 —— 与本仓 vLLM 口径 (kernel−1) 的
    # 差异在报告披露，不改本仓。
    assert dconv["path"] == "update" and dconv["state_len"] == 4
    # norm 宽度清单（h=32；q/k_norm 与 RMSNormGated 作用在 head 维 8）
    for phase in ("prefill", "decode"):
        nw = art["phases"][phase]["shapes"]["norm_widths"]
        assert nw == {"attn_norm": 32, "mlp_norm": 32, "final_norm": 32,
                      "q_norm": 8, "k_norm": 8, "rmsnorm_gated": 8}, nw


def main() -> int:
    art = run_probe()
    DEFAULT_OUT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUT.write_text(
        json.dumps(art, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("R06 fixture probe OK")
    print(f"artifact: {DEFAULT_OUT}")
    for phase in ("prefill", "decode"):
        p = art["phases"][phase]
        print(f"[{phase}] T={p['T']} gemms={len(p['gemms'])} "
              f"conv_flops={p['conv'][0]['flops']} "
              f"gdn={p['gdn']['chunk_or_recurrent']} "
              f"causal_pairs={p['attn']['causal_pairs']} "
              f"assignment_total={p['moe']['assignment_total']} "
              f"routed_flops={p['moe']['routed_flops']} "
              f"logits={p['shapes']['logits_shape']}")
    print(f"pinned modeling sha256={art['pinned']['modeling_sha256'][:16]}... "
          f"config sha256={art['pinned']['configuration_sha256'][:16]}...")
    print(f"environment torch={art['environment']['torch']} "
          f"transformers={art['environment']['transformers']} "
          f"python={art['environment']['python']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
