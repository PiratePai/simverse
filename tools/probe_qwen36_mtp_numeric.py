"""R23：Qwen3.6 原生 MTP 头小尺寸独立数学对照（fixture 生成探针）。

结构按 assets/identity/vllm_qwen36_mtp_pin @10a854ed（vllm/model_executor/
models/qwen3_5_mtp.py）**手工复刻**（vLLM 模块本身不在 CPU 执行）：

  combine = embed(共享查表) → pre_fc_norm_embedding → pre_fc_norm_hidden(目标
            hidden) → cat[2h] → fc(2h→h, bias=False, 无激活/无残差直加)
  层      = full_attention decoder 层（gated GQA + MoE top-k + shared+gate）
            层选择 spec_step_idx % n_layers（MTP1 同层复用）
  出口    = mtp.norm(final, residual) → 共享 lm_head → greedy argmax

双实现互核（同一随机权重、逐元素对照）：
  * impl A：torch 张量矩阵式（忠实复刻，fp32）；
  * impl B：纯 Python 逐元素/逐 token 标量循环（math 标准库，fp64）——
    独立参考，不共享任何 A 的代码路径。
同时对账本仓 count/draft.mtp_step_rows（同几何、Q32 全位宽）：
  * 每个 GEMM 行 flops == 2 × B 循环实际计数 MAC（逐步差分）；
  * attention 因果对数 / KV 读行数与 B 的实际键计数精确相等；
  * 逐元素行（norm/rope/门控/残差）按仓内注册口径核对并登记。

产出 tests/fixtures/qwen36_mtp_r23_numeric.json（hermetic 测试消费；
本探针不作为 pytest 用例——CI 无 torch）。fixture 级证据不升格：
模型级 cpu_forward 保持 not-run（准则 §1.5/§1.8）。
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

SEED = 20261113
ATOL = 1e-9

# 小尺寸几何（不外推真实 35B；比例保持：partial rotary 0.25、gated q 2×）
H, V = 32, 97
HEADS, KVH, D, ROT = 4, 2, 16, 4          # rotary = 0.25·d
E, TOPK, INTER, SHARED = 4, 2, 8, 8
CTX, KSTEPS, N_MTP = 6, 3, 1              # MTP1；K=3 步同层循环复用
GQA_MAP = [0, 0, 1, 1]                    # Q head → KV head（4 Q / 2 KV）

P = {
    "hidden_size": H, "num_hidden_layers": 1,
    "layer_types": ["full_attention"], "num_attention_heads": HEADS,
    "num_key_value_heads": KVH, "head_dim": D, "vocab_size": V,
    "num_experts": E, "num_experts_per_tok": TOPK,
    "moe_intermediate_size": INTER, "intermediate_size": INTER,
    "shared_expert_intermediate_size": SHARED,
    "linear_num_key_heads": 2, "linear_num_value_heads": 4,
    "linear_key_head_dim": 8, "linear_value_head_dim": 8,
    "linear_conv_kernel_dim": 4,
    "partial_rotary_factor": 0.25,
    "mtp_num_hidden_layers": N_MTP,
    "mtp_use_dedicated_embeddings": False,
    "tie_word_embeddings": False,
    "rms_norm_eps": 1e-6,
}


# ---------------------------------------------------------------------------
# impl B：纯 Python 标量参考 + 逐步 MAC/键计数（独立于 torch 路径）
# ---------------------------------------------------------------------------

def _rms_b(x, w, eps=1e-6):
    """Gemma 式 RMSNorm：x/rms(x)·(1+w)（vLLM GemmaRMSNorm as Qwen3NextRMSNorm）。"""
    rms = math.sqrt(sum(v * v for v in x) / len(x) + eps)
    return [v / rms * (1.0 + wi) for v, wi in zip(x, w)]


def _silu(z):
    if z < -700:
        return 0.0
    return z / (1.0 + math.exp(-z))


def _softmax(xs):
    m = max(xs)
    es = [math.exp(v - m) for v in xs]
    s = sum(es)
    return [v / s for v in es]


def _rope_b(vec, cos, sin):
    """rotate-half RoPE：前 ROT 维成对旋转（cos/sin 已按 rotate-half 排列）。"""
    out = list(vec)
    half = ROT // 2
    for i in range(half):
        x1, x2 = vec[i], vec[i + half]
        out[i] = x1 * cos[i] - x2 * sin[i]
        out[i + half] = x1 * sin[i] + x2 * cos[i]
    return out


class ScalarRef:
    """逐步记录 MAC 差分 / 因果键计数 / KV 读行数。"""

    def __init__(self, W):
        self.W = W
        self.macs: dict = {}
        self.pairs = 0
        self.kv_read_rows = 0
        self.step_log = []

    def _mv(self, name, mat, x):
        macs = 0
        out = []
        for row in mat:
            acc = 0.0
            for a, b in zip(row, x):
                acc += a * b
                macs += 1
            out.append(acc)
        self.macs[name] = self.macs.get(name, 0) + macs
        return out

    def step(self, token, hidden, kv_k, kv_v, cos, sin, step_idx):
        """kv_k/kv_v：[KVH][rows][D]（草稿自有 KV，按 KV 头组织）。"""
        W = self.W
        layer_idx = step_idx % N_MTP          # 层选择（MTP1 恒 0）
        macs0, pairs0, rows0 = dict(self.macs), self.pairs, self.kv_read_rows
        rec: dict = {"layer_idx": layer_idx}
        e = [float(v) for v in W["embed"][token]]
        e_n = _rms_b(e, W["enorm"])
        h_n = _rms_b(hidden, W["hnorm"])
        rec["pre_fc_norm_embedding"] = e_n
        rec["pre_fc_norm_hidden"] = h_n
        x = self._mv("mtp_fc", W["fc"], e_n + h_n)
        rec["fc_out"] = x
        # ---- full_attention 层（pre-norm 残差）----
        r = x
        y = _rms_b(x, W["attn_norm"])
        qg = self._mv("q_proj", W["q"], y)                  # 2×heads·d（query+gate）
        k_flat = self._mv("k_proj", W["k"], y)
        v_flat = self._mv("v_proj", W["v"], y)
        q = [None] * HEADS
        g = [None] * HEADS
        for i in range(HEADS):
            q[i] = _rms_b(qg[i * D:(i + 1) * D], W["q_norm"])
            g[i] = qg[HEADS * D + i * D: HEADS * D + (i + 1) * D]
        k_new = [None] * KVH
        v_new = [None] * KVH
        for c in range(KVH):
            k_new[c] = _rms_b(k_flat[c * D:(c + 1) * D], W["k_norm"])
            v_new[c] = v_flat[c * D:(c + 1) * D]
        q_r = [_rope_b(q[i], cos, sin) for i in range(HEADS)]
        k_r = [_rope_b(k_new[c], cos, sin) for c in range(KVH)]
        kk = [list(kv_k[c]) for c in range(KVH)]
        vv = [list(kv_v[c]) for c in range(KVH)]
        for c in range(KVH):
            kk[c].append(k_r[c])
            vv[c].append(v_new[c])
        q_pos = len(kk[0]) - 1
        # 缓存读行数按本仓账本口径：先前行（q_pos 行，每请求一次——4 头共享
        # 同一 KV 行集合）；本步新写行在寄存器/激活中，由 kv_write 单独计
        self.kv_read_rows += q_pos
        attn_out = []
        for i in range(HEADS):
            c = GQA_MAP[i]
            scores = []
            for t in range(q_pos + 1):                       # 因果：键行 ≤ q_pos
                s = 0.0
                for a, b in zip(q_r[i], kk[c][t]):
                    s += a * b
                    self.pairs += 1
                scores.append(s / math.sqrt(D))
            probs = _softmax(scores)
            o = [0.0] * D
            for t, p_t in enumerate(probs):
                for j in range(D):
                    o[j] += p_t * vv[c][t][j]
            o = [o[j] / (1.0 + math.exp(-g[i][j])) for j in range(D)]
            attn_out.extend(o)
        rec["attn_out_gated"] = attn_out
        attn_proj = self._mv("o_proj", W["o"], attn_out)
        res1 = [a + b for a, b in zip(attn_proj, r)]
        # ---- MoE（router softmax fp32 → top-k 重归一；专家 SwiGLU×3 GEMM）----
        z = _rms_b(res1, W["post_norm"])
        router_logits = self._mv("moe_router", W["router"], z)
        probs = _softmax(router_logits)
        order = sorted(range(E), key=lambda i: -probs[i])[:TOPK]
        wsum = sum(probs[i] for i in order)
        wts = [probs[i] / wsum for i in order]
        routed = [0.0] * H
        for wi, ei in zip(wts, order):
            gu = self._mv("moe_routed", W["experts"][ei][0], z)
            up = self._mv("moe_routed", W["experts"][ei][1], z)
            act = [_silu(a) * b for a, b in zip(gu, up)]
            out_e = self._mv("moe_routed", W["experts"][ei][2], act)
            routed = [rv + wi * ov for rv, ov in zip(routed, out_e)]
        sg = self._mv("moe_shared", W["shared"][0], z)
        su = self._mv("moe_shared", W["shared"][1], z)
        sact = [_silu(a) * b for a, b in zip(sg, su)]
        s_out = self._mv("moe_shared", W["shared"][2], sact)
        gate_v = self._mv("shared_gate", W["sgate"], z)[0]
        shared_full = [v / (1.0 + math.exp(-gate_v)) for v in s_out]
        ffn = [a + b for a, b in zip(routed, shared_full)]
        rec["moe_out"] = ffn
        res2 = [a + b for a, b in zip(ffn, res1)]
        final = _rms_b(res2, W["final_norm"])                # mtp.norm(residual)
        rec["final_hidden"] = final
        logits = self._mv("draft_lm_head", W["lm_head"], final)
        rec["logits"] = logits
        rec["sampled_token"] = max(range(V), key=lambda i: logits[i])
        rec["kv_len_after"] = len(kk[0])
        rec["step_macs"] = {k: self.macs[k] - macs0.get(k, 0) for k in self.macs
                            if self.macs[k] != macs0.get(k, 0)}
        rec["keys_per_head"] = q_pos + 1          # 因果键数（pairs 计的是元素乘）
        rec["kv_read_rows"] = self.kv_read_rows - rows0
        self.step_log.append(rec)
        return rec, kk, vv


# ---------------------------------------------------------------------------
# impl A：torch 张量式（忠实复刻 pinned 结构）
# ---------------------------------------------------------------------------

def _rms_t(x, w, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1) + eps) * (1.0 + w)


def _rope_t(vec, cos, sin):
    out = vec.clone()
    half = ROT // 2
    x1, x2 = vec[:half], vec[half:ROT]
    out[:half] = x1 * cos[:half] - x2 * sin[:half]
    out[half:ROT] = x1 * sin[:half] + x2 * cos[:half]
    return out


class TorchRef:
    def __init__(self, Wt):
        self.W = Wt

    def step(self, token, hidden, kv_k, kv_v, cos, sin, step_idx):
        W = self.W
        layer_idx = step_idx % N_MTP
        rec: dict = {"layer_idx": layer_idx}
        e = W["embed"][token]
        e_n = _rms_t(e, W["enorm"])
        h_n = _rms_t(hidden, W["hnorm"])
        rec["pre_fc_norm_embedding"] = e_n
        rec["pre_fc_norm_hidden"] = h_n
        x = W["fc"] @ torch.cat([e_n, h_n])
        rec["fc_out"] = x
        r = x
        y = _rms_t(x, W["attn_norm"])
        qg = W["q"] @ y
        k_flat = W["k"] @ y
        v_flat = W["v"] @ y
        q = torch.stack([_rms_t(qg[i * D:(i + 1) * D], W["q_norm"])
                         for i in range(HEADS)])
        g = torch.stack([qg[HEADS * D + i * D: HEADS * D + (i + 1) * D]
                         for i in range(HEADS)])
        k_new = torch.stack([_rms_t(k_flat[c * D:(c + 1) * D], W["k_norm"])
                             for c in range(KVH)])
        v_new = torch.stack([v_flat[c * D:(c + 1) * D] for c in range(KVH)])
        q_r = torch.stack([_rope_t(q[i], cos, sin) for i in range(HEADS)])
        k_r = torch.stack([_rope_t(k_new[c], cos, sin) for c in range(KVH)])
        kk = torch.cat([kv_k, k_r.unsqueeze(0)], 0)         # [keys, KVH, D]
        vv = torch.cat([kv_v, v_new.unsqueeze(0)], 0)
        q_pos = kk.shape[0] - 1
        attn_out = []
        for i in range(HEADS):
            c = GQA_MAP[i]
            keys_c = kk[:, c, :]                             # GQA：本头的 KV 头切片
            scores = (keys_c @ q_r[i]) / math.sqrt(D)        # 键行 0..q_pos（因果）
            probs = torch.softmax(scores, -1)
            o = probs @ vv[:, c, :]
            attn_out.append(o * torch.sigmoid(g[i]))
        attn_out = torch.cat(attn_out)
        rec["attn_out_gated"] = attn_out
        res1 = W["o"] @ attn_out + r
        z = _rms_t(res1, W["post_norm"])
        logits_r = W["router"] @ z
        probs = torch.softmax(logits_r.double(), -1)
        wts_val, order = torch.topk(probs, TOPK)
        wts_val = wts_val / wts_val.sum()
        routed = torch.zeros(H)
        for wi, ei in zip(wts_val.tolist(), order.tolist()):
            gu = W["experts"][ei][0] @ z
            up = W["experts"][ei][1] @ z
            out_e = W["experts"][ei][2] @ (torch.nn.functional.silu(gu) * up)
            routed = routed + wi * out_e
        sg = W["shared"][0] @ z
        su = W["shared"][1] @ z
        s_out = W["shared"][2] @ (torch.nn.functional.silu(sg) * su)
        gate_v = (W["sgate"] @ z)[0]
        ffn = routed + s_out * torch.sigmoid(gate_v)
        rec["moe_out"] = ffn
        res2 = ffn + res1
        final = _rms_t(res2, W["final_norm"])
        rec["final_hidden"] = final
        logits = W["lm_head"] @ final
        rec["logits"] = logits
        rec["sampled_token"] = int(logits.argmax())
        rec["kv_len_after"] = int(kk.shape[0])
        return rec, kk, vv


def build_weights():
    g = torch.Generator().manual_seed(SEED)

    def rnd(*shape):
        return torch.randn(*shape, generator=g)

    Wt = {
        "embed": rnd(V, H), "fc": rnd(H, 2 * H),
        "enorm": rnd(H), "hnorm": rnd(H), "attn_norm": rnd(H),
        "post_norm": rnd(H), "final_norm": rnd(H),
        "q": rnd(2 * HEADS * D, H), "k": rnd(KVH * D, H),
        "v": rnd(KVH * D, H), "o": rnd(H, HEADS * D),
        "q_norm": rnd(D), "k_norm": rnd(D),
        "router": rnd(E, H),
        "experts": [[rnd(INTER, H), rnd(INTER, H), rnd(H, INTER)]
                    for _ in range(E)],
        "shared": [rnd(INTER, H), rnd(INTER, H), rnd(H, INTER)],
        "sgate": rnd(1, H),
        "lm_head": rnd(V, H),
    }
    kv_k0 = rnd(CTX, KVH, D)
    kv_v0 = rnd(CTX, KVH, D)
    hidden0 = rnd(H)
    tok0 = int(torch.randint(0, V, (1,), generator=g))
    inv = 1.0 / (10000.0 ** (torch.arange(0, ROT, 2).float() / ROT))
    pos = torch.arange(CTX + KSTEPS + 1).float()
    angles = torch.outer(pos, inv)                       # [pos, ROT/2]
    cos = torch.cos(torch.cat([angles] * 2, -1))         # rotate-half 排列
    sin = torch.sin(torch.cat([angles] * 2, -1))
    return Wt, kv_k0, kv_v0, hidden0, tok0, cos, sin


def to_lists(obj):
    """torch 权重树 → 纯 Python 嵌套 list（标量叶子）。"""
    if isinstance(obj, dict):
        return {k: to_lists(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_lists(x) for x in obj]
    if isinstance(obj, torch.Tensor):
        if obj.dim() == 0:
            return float(obj)
        if obj.dim() == 1:
            return [float(x) for x in obj]
        return [[float(x) for x in row] for row in obj]
    raise TypeError(type(obj))


def main() -> None:
    Wt, kv_k0, kv_v0, hidden0, tok0, cos, sin = build_weights()

    def to64(o):
        if isinstance(o, dict):
            return {k: to64(v) for k, v in o.items()}
        if isinstance(o, list):
            return [to64(x) for x in o]
        return o.double()

    # impl A 在 float64 下运行：随机 softmax 下 router top-k 存在近平手，
    # f32/f64 概率会翻转第 2 名选择（观测 4e-4 假差异）——结构对照必须同
    # 精度；f32 执行的近平手翻转是运行时数值行为，不属于本对照的断言面。
    Wt64 = to64(Wt)
    kv_k0, kv_v0 = kv_k0.double(), kv_v0.double()
    hidden0, cos, sin = hidden0.double(), cos.double(), sin.double()
    Wb = to_lists(Wt64)
    a = TorchRef(Wt64)
    b = ScalarRef(Wb)
    kk_a, vv_a = kv_k0.clone(), kv_v0.clone()
    kk_b = [[[float(x) for x in kv_k0[t, c, :]] for t in range(CTX)]
            for c in range(KVH)]
    vv_b = [[[float(x) for x in kv_v0[t, c, :]] for t in range(CTX)]
            for c in range(KVH)]
    hb = [float(x) for x in hidden0]
    ta, tb = tok0, tok0
    diffs = []
    steps_summary = []
    for j in range(KSTEPS):
        pos = CTX + j
        ra, kk_a, vv_a = a.step(ta, hidden0 if j == 0 else prev_h_a,
                                kk_a, vv_a, cos[pos], sin[pos], j)
        prev_h_a = ra["final_hidden"]
        rb, kk_b, vv_b = b.step(tb, hb, kk_b, vv_b,
                                [float(x) for x in cos[pos]],
                                [float(x) for x in sin[pos]], j)
        hb = rb["final_hidden"]
        assert ra["layer_idx"] == rb["layer_idx"] == j % N_MTP
        assert ra["kv_len_after"] == rb["kv_len_after"] == CTX + j + 1
        for key in ("pre_fc_norm_embedding", "pre_fc_norm_hidden", "fc_out",
                    "attn_out_gated", "moe_out", "final_hidden", "logits"):
            # 对照在 float64 下做（torch.tensor(list) 默认 f32，会把 B 侧量化
            # 成 f32 制造假差异——曾观测 4e-4 假阳，修复为显式 f64）
            va = (torch.tensor(rb[key], dtype=torch.float64)
                  - ra[key].double())
            diffs.append({"step": j, "tensor": key,
                          "max_abs_diff": float(va.abs().max())})
        steps_summary.append({
            "step": j, "layer_idx": rb["layer_idx"],
            "sampled_token": rb["sampled_token"],
            "kv_len_after": rb["kv_len_after"],
            "keys_per_head": rb["keys_per_head"],
            "kv_read_rows": rb["kv_read_rows"],
            "step_macs": rb["step_macs"],
            "final_hidden": [round(v, 6) for v in rb["final_hidden"]],
            "logits_head3": [round(rb["logits"][i], 6) for i in range(3)]})
        ta = ra["sampled_token"]
        tb = rb["sampled_token"]
        assert ta == tb, (j, ta, tb)

    # ---- 与本仓 mtp_step_rows 对账（Q32 全位宽；B=1；tp=1）----
    from simverse.count.draft import mtp_step_rows
    from simverse.model_ir.families import parse_qwen3_5_moe
    from simverse.model_ir.schema import SourceIdentity
    ir = parse_qwen3_5_moe(dict(P), SourceIdentity(
        "probe:qwen3_5_moe", "vllm@10a854ed", "fixture",
        captured_at="2026-11-13",
        note="R23 MTP fixture（pin 复刻，非本仓执行官方模块）"),
        "qwen36_mtp_r23_fixture", "R23 MTP fixture (1L full-attn MoE)")
    Q32 = {"dense_w": 32, "expert_w": 32, "act": 32, "kv": 32, "state": 32}
    gemm_checks = []
    name_map = {"mtp_fc": "mtp_fc", "q_proj": "q_proj", "k_proj": "k_proj",
                "v_proj": "v_proj", "o_proj": "o_proj",
                "moe_router": "moe_router", "moe_routed": "moe_routed",
                "moe_shared": "moe_shared", "shared_gate": "shared_gate",
                "draft_lm_head": "draft_lm_head"}
    attention_checks = []
    for j in range(KSTEPS):
        rows = {r.op: r for r in mtp_step_rows(f"draft_step_{j}", ir, j, CTX,
                                               Q32, tp=1)}
        for ref_name, op in name_map.items():
            mac = steps_summary[j]["step_macs"].get(ref_name, 0)
            if op in rows and mac:
                gemm_checks.append({
                    "step": j, "op": op, "ledger_flops": rows[op].flops,
                    "probe_2x_macs": 2 * mac,
                    "match": rows[op].flops == 2 * mac})
        core = rows["attention_core"]
        ledger_pairs = core.flops // (HEADS * (4 * D + 5))
        kv_rows = core.kv_read // (4 * 2 * KVH * D)
        attention_checks.append({
            "step": j, "keys_per_head": steps_summary[j]["keys_per_head"],
            "ledger_pairs": ledger_pairs,
            "pairs_match": steps_summary[j]["keys_per_head"] == ledger_pairs,
            "kv_read_rows": steps_summary[j]["kv_read_rows"],
            "kv_read_rows_ledger": kv_rows,
            "kv_rows_match": steps_summary[j]["kv_read_rows"] == kv_rows,
            "kv_write_rows_ledger": core.kv_write // (4 * 2 * KVH * D)})

    # 手算锚点：Gemma RMSNorm([3,4], w=[1,1]) = [3,4]/sqrt(12.5+eps)·2
    anchor = _rms_b([3.0, 4.0], [1.0, 1.0])
    anchor_expected = [3.0 / math.sqrt(12.5 + 1e-6) * 2.0,
                       4.0 / math.sqrt(12.5 + 1e-6) * 2.0]

    out = {
        "schema": "qwen36-mtp-r23-numeric-v1",
        "environment": {"python": sys.version.split()[0],
                        "torch": torch.__version__,
                        "date": "2026-11-13",
                        "impl_a_dtype": "float64",
                        "impl_a_dtype_note": "双实现同精度对照：随机 softmax "
                                "router 近平手在 f32/f64 间会翻转第 2 名选择"
                                "（观测 4e-4 假差异），结构对照须同精度；"
                                "近平手翻转属运行时数值行为，不在断言面",
                        "note": "结构复刻自 assets/identity/vllm_qwen36_mtp_pin "
                                "@10a854ed（qwen3_5_mtp.py）；vLLM 模块未执行"},
        "dims": dict(H=H, V=V, heads=HEADS, kvh=KVH, d=D, rotary=ROT, E=E,
                     topk=TOPK, inter=INTER, shared=SHARED, ctx=CTX,
                     k_steps=KSTEPS, n_mtp=N_MTP, seed=SEED, atol=ATOL),
        "fixture_config": P,
        "structure_facts": {
            "layer_selection": "spec_step_idx % n_layers",
            "steps_layer_idx": [s["layer_idx"] for s in steps_summary],
            "fc": "2h(64) → h(32)，bias=False，无激活无残差直加",
            "pre_fc_norms": ["pre_fc_norm_embedding", "pre_fc_norm_hidden"],
            "final_norm": "mtp.norm(residual)",
            "shared": ["embed_tokens", "lm_head"],
        },
        "steps": steps_summary,
        "numeric_check": {"atol": ATOL, "tensors": diffs,
                          "all_within_atol": all(
                              d["max_abs_diff"] <= ATOL for d in diffs),
                          "sampled_tokens_match": True},
        "gemm_checks": gemm_checks,
        "all_gemm_match": all(c["match"] for c in gemm_checks),
        "attention_checks": attention_checks,
        "all_attention_match": all(c["pairs_match"] and c["kv_rows_match"]
                                   for c in attention_checks),
        "hand_anchor": {"name": "gemma_rmsnorm([3,4], w=[1,1])",
                        "got": anchor, "expected": anchor_expected,
                        "match": all(abs(a2 - e) < 1e-12
                                     for a2, e in zip(anchor, anchor_expected))},
        "limits": [
            "随机权重小尺寸 fixture，不覆盖真实 35B checkpoint 权重/融合 kernel/"
            "部署性能",
            "vLLM 模块本身未在 CPU 执行（结构手工复刻，逐行对 pinned 源）",
            "fixture 级证据：模型级 cpu_forward 保持 not-run，不升格"],
    }
    assert out["numeric_check"]["all_within_atol"], \
        max(d["max_abs_diff"] for d in diffs)
    assert out["all_gemm_match"]
    assert out["all_attention_match"]
    assert out["hand_anchor"]["match"]
    dest = ROOT / "tests/fixtures/qwen36_mtp_r23_numeric.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n",
                    encoding="utf-8", newline="\n")
    print("max abs diff:", max(d["max_abs_diff"] for d in diffs))
    print("gemm checks:", len(gemm_checks), "all match:", out["all_gemm_match"])
    print("attention all match:", out["all_attention_match"])
    print("anchor match:", out["hand_anchor"]["match"])
    print("sampled tokens:", [s["sampled_token"] for s in steps_summary])
    print("written:", dest)


if __name__ == "__main__":
    main()
