"""Independent tensor oracle for the pinned, tiny Qwen3.6 CPU fixture.

Reads weights from the pinned model but never calls its layer, attention,
GDN, router, expert, or norm forward methods in the reference path. RoPE
cos/sin are supplied as an input boundary, so mrope generation is not tested.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from probe_qwen36_forward_r06 import (
    DECODE_IDS, FIXTURE_CONFIG, PIN_DIR, PREFILL_IDS, SEED, load_pinned_modules,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "fixtures" / "qwen36_r06_numeric_oracle.json"


def _linear(x, layer, torch):
    y = torch.matmul(x, layer.weight.transpose(-1, -2))
    return y if layer.bias is None else y + layer.bias


def _rms(x, module, gated, torch):
    xf = x.float()
    y = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) +
                         (module.variance_epsilon if gated else module.eps))
    if gated:
        return (y.to(x.dtype) * module.weight).to(x.dtype)
    return (y * (1 + module.weight.float())).to(x.dtype)


def _silu(x, torch):
    return x * torch.sigmoid(x)


def _moe(x, block, torch):
    b, t, h = x.shape
    flat = x.reshape(-1, h)
    shared = block.shared_expert
    shared_y = _linear(_silu(_linear(flat, shared.gate_proj, torch), torch) *
                       _linear(flat, shared.up_proj, torch), shared.down_proj, torch)
    shared_y *= torch.sigmoid(_linear(flat, block.shared_expert_gate, torch))

    logits = torch.matmul(flat, block.gate.weight.T)
    probs = torch.softmax(logits.float(), dim=-1)
    vals, inds = torch.topk(probs, block.gate.top_k, dim=-1)
    vals = (vals / vals.sum(-1, keepdim=True)).to(flat.dtype)
    routed = torch.zeros_like(flat)
    # Token-major scalar reference deliberately differs from the official
    # expert-major scatter path.
    for token in range(flat.shape[0]):
        for slot in range(block.gate.top_k):
            expert = int(inds[token, slot])
            gate_up = torch.matmul(block.experts.gate_up_proj[expert], flat[token])
            gate, up = gate_up.chunk(2)
            contribution = torch.matmul(
                block.experts.down_proj[expert], _silu(gate, torch) * up)
            routed[token] += contribution * vals[token, slot]
    return (routed + shared_y).reshape(b, t, h)


def _gdn(x, layer, cache, torch):
    b, t, _ = x.shape
    c = layer.conv_dim
    qkv = _linear(x, layer.in_proj_qkv, torch)
    z = _linear(x, layer.in_proj_z, torch).reshape(b, t, layer.num_v_heads,
                                                   layer.head_v_dim)
    beta = torch.sigmoid(_linear(x, layer.in_proj_b, torch))
    a = _linear(x, layer.in_proj_a, torch)
    decay_log = -torch.exp(layer.A_log.float()) * torch.nn.functional.softplus(
        a.float() + layer.dt_bias)

    history = (torch.zeros(b, c, layer.conv_kernel_size, dtype=x.dtype)
               if cache is None else cache["conv"].clone())
    weight = layer.conv1d.weight[:, 0, :]
    conv_tokens = []
    for i in range(t):
        history = torch.cat((history[:, :, 1:], qkv[:, i, :, None]), dim=-1)
        conv_tokens.append(_silu((history * weight[None]).sum(-1), torch))
    conv = torch.stack(conv_tokens, dim=1)
    q, k, v = torch.split(conv, [layer.key_dim, layer.key_dim,
                                layer.value_dim], dim=-1)
    nk, nv, dk, dv = (layer.num_k_heads, layer.num_v_heads,
                       layer.head_k_dim, layer.head_v_dim)
    q = q.reshape(b, t, nk, dk).repeat_interleave(nv // nk, dim=2)
    k = k.reshape(b, t, nk, dk).repeat_interleave(nv // nk, dim=2)
    v = v.reshape(b, t, nv, dv)
    q = q.float() * torch.rsqrt(q.float().square().sum(-1, keepdim=True) + 1e-6)
    k = k.float() * torch.rsqrt(k.float().square().sum(-1, keepdim=True) + 1e-6)
    q = q / math.sqrt(dk)
    state = (torch.zeros(b, nv, dk, dv, dtype=torch.float32)
             if cache is None else cache["state"].clone().float())
    outs = []
    for i in range(t):
        state = state * torch.exp(decay_log[:, i])[:, :, None, None]
        prediction = torch.einsum("bhk,bhkv->bhv", k[:, i], state)
        correction = (v[:, i].float() - prediction) * beta[:, i, :, None]
        state = state + torch.einsum("bhk,bhv->bhkv", k[:, i], correction)
        outs.append(torch.einsum("bhk,bhkv->bhv", q[:, i], state))
    core = torch.stack(outs, dim=1).to(x.dtype)
    normed = _rms(core, layer.norm, True, torch)
    normed = normed * _silu(z, torch)
    return _linear(normed.reshape(b, t, -1), layer.out_proj, torch), {
        "conv": history, "state": state}


def _rotate(x, cos, sin, torch):
    rot = cos.shape[-1]
    a = x[..., :rot]
    half = rot // 2
    turned = torch.cat((-a[..., half:], a[..., :half]), dim=-1)
    return torch.cat((a * cos[:, None] + turned * sin[:, None], x[..., rot:]), dim=-1)


def _attention(x, layer, pos, cache, torch):
    b, t, _ = x.shape
    heads = layer.config.num_attention_heads
    kvheads = layer.config.num_key_value_heads
    d = layer.head_dim
    packed = _linear(x, layer.q_proj, torch).reshape(b, t, heads, 2 * d)
    q, gate = packed.split(d, dim=-1)
    q = _rms(q, layer.q_norm, False, torch).transpose(1, 2)
    k = _rms(_linear(x, layer.k_proj, torch).reshape(b, t, kvheads, d),
             layer.k_norm, False, torch).transpose(1, 2)
    v = _linear(x, layer.v_proj, torch).reshape(b, t, kvheads, d).transpose(1, 2)
    cos, sin = pos
    q, k = _rotate(q, cos, sin, torch), _rotate(k, cos, sin, torch)
    if cache is not None:
        k = torch.cat((cache["key"], k), dim=2)
        v = torch.cat((cache["value"], v), dim=2)
    keys = k.repeat_interleave(heads // kvheads, dim=1)
    values = v.repeat_interleave(heads // kvheads, dim=1)
    scores = torch.matmul(q, keys.transpose(-1, -2)) / math.sqrt(d)
    past = k.shape[2] - t
    qi = torch.arange(t)[:, None] + past
    kj = torch.arange(k.shape[2])[None, :]
    scores = scores.masked_fill(kj > qi, torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    attended = torch.matmul(weights, values).transpose(1, 2).reshape(b, t, -1)
    gated = attended * torch.sigmoid(gate.reshape(b, t, -1))
    return _linear(gated, layer.o_proj, torch), {"key": k, "value": v}, gated


def _reference(model, ids, pos, caches, torch):
    x = model.model.embed_tokens.weight[torch.tensor(ids)]
    new_caches = []
    intermediates = {}
    for li, decoder in enumerate(model.model.layers):
        residual = x
        normed = _rms(x, decoder.input_layernorm, False, torch)
        if li == 0:
            mixed, state = _gdn(normed, decoder.linear_attn,
                                None if caches is None else caches[li], torch)
            intermediates["gdn_output"] = mixed
            intermediates["gdn_state"] = state["state"]
        else:
            mixed, state, gated = _attention(
                normed, decoder.self_attn, pos,
                None if caches is None else caches[li], torch)
            intermediates["attn_gated"] = gated
            intermediates["attn_output"] = mixed
        x = residual + mixed
        moe_out = _moe(_rms(x, decoder.post_attention_layernorm, False, torch),
                       decoder.mlp, torch)
        intermediates[f"moe_{li}"] = moe_out
        x = x + moe_out
        intermediates[f"layer_{li}"] = x
        new_caches.append(state)
    x = _rms(x, model.model.norm, False, torch)
    logits = _linear(x, model.lm_head, torch)
    intermediates["logits"] = logits
    return intermediates, new_caches


def _compare(name, actual, expected, torch, atol=1e-5, rtol=1e-4):
    diff = (actual.float() - expected.float()).abs()
    bound = atol + rtol * actual.float().abs()
    worst = float(diff.max())
    assert bool((diff <= bound).all()), f"{name}: max_abs={worst}, bound={float(bound.max())}"
    return {"shape": list(actual.shape), "max_abs": worst,
            "rms_error": float(diff.square().mean().sqrt()),
            "reference_sample": expected.flatten()[:8].tolist()}


def run_oracle():
    config_module, modeling, torch, transformers = load_pinned_modules()
    torch.manual_seed(SEED)
    config = config_module.Qwen3_5MoeTextConfig(**FIXTURE_CONFIG)
    config._attn_implementation = "eager"
    model = modeling.Qwen3_5MoeForCausalLM(config).eval()
    captures = {"layers": {}, "position": None, "gdn": None,
                "attention": None, "gated": None, "moe": {}}
    handles = []
    for li, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_hook(
            lambda _m, _a, out, layer_idx=li:
            captures["layers"].__setitem__(layer_idx, out.detach().clone())))
    handles.append(model.model.layers[1].self_attn.register_forward_pre_hook(
        lambda _m, _a, kw: captures.__setitem__(
            "position", tuple(v.detach().clone() for v in kw["position_embeddings"])),
        with_kwargs=True))
    handles.append(model.model.layers[0].linear_attn.register_forward_hook(
        lambda _m, _a, out: captures.__setitem__("gdn", out.detach().clone())))
    handles.append(model.model.layers[1].self_attn.register_forward_hook(
        lambda _m, _a, out: captures.__setitem__("attention", out[0].detach().clone())))
    handles.append(model.model.layers[1].self_attn.o_proj.register_forward_pre_hook(
        lambda _m, args: captures.__setitem__("gated", args[0].detach().clone())))
    for li, layer in enumerate(model.model.layers):
        handles.append(layer.mlp.register_forward_hook(
            lambda _m, _a, out, layer_idx=li:
            captures["moe"].__setitem__(layer_idx, out.detach().clone())))
    phases = {}
    ref_caches = None
    hf_cache = None
    with torch.inference_mode():
        for phase, ids in (("prefill", PREFILL_IDS), ("decode", DECODE_IDS)):
            captures["layers"] = {}
            captures["moe"] = {}
            output = model(input_ids=torch.tensor(ids), past_key_values=hf_cache,
                           use_cache=True)
            hf_cache = output.past_key_values
            ref, ref_caches = _reference(model, ids, captures["position"],
                                         ref_caches, torch)
            checks = {
                "gdn_output": _compare("gdn_output", captures["gdn"], ref["gdn_output"], torch),
                "attn_output": _compare("attn_output", captures["attention"], ref["attn_output"], torch),
                "attn_gated": _compare("attn_gated", captures["gated"], ref["attn_gated"], torch),
                "moe_0": _compare("moe_0", captures["moe"][0], ref["moe_0"], torch),
                "moe_1": _compare("moe_1", captures["moe"][1], ref["moe_1"], torch),
                "layer_0": _compare("layer_0", captures["layers"][0], ref["layer_0"], torch),
                "layer_1": _compare("layer_1", captures["layers"][1], ref["layer_1"], torch),
                "logits": _compare("logits", output.logits, ref["logits"], torch),
                "gdn_state": _compare("gdn_state", hf_cache.get_recurrent_state(0, 0),
                                      ref_caches[0]["state"], torch),
                "conv_state": _compare("conv_state", hf_cache.get_conv_state(0, 0),
                                       ref_caches[0]["conv"], torch),
                "kv_keys": _compare("kv_keys", hf_cache.layers[1].keys,
                                    ref_caches[1]["key"], torch),
                "kv_values": _compare("kv_values", hf_cache.layers[1].values,
                                      ref_caches[1]["value"], torch),
            }
            phases[phase] = checks
    for handle in handles:
        handle.remove()
    return {"schema_version": "1", "seed": SEED,
            "pinned": {name: hashlib.sha256((PIN_DIR / name).read_bytes()).hexdigest()
                       for name in ("modeling_qwen3_5_moe.py",
                                    "configuration_qwen3_5_moe.py")},
            "environment": {"torch": torch.__version__,
                            "transformers": transformers.__version__},
            "boundary": "RoPE cos/sin from pinned forward inputs; mrope generation not checked",
            "tolerance": {"atol": 1e-5, "rtol": 1e-4}, "phases": phases}


if __name__ == "__main__":
    result = run_oracle()
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8", newline="\n")
    print(f"numeric oracle passed: {OUT}")
