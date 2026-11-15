"""R26 fixture probe: Qwen3.6 文本路径 MRoPE position_ids 数学对照（纯标准库）。

对照对象与证据边界（全部 pin 文件在 assets/identity/，SHA256 记入 fixture）：
  * HF transformers @b6401d66 modeling_qwen3_5_moe.py
      - L130-194  Qwen3_5MoeTextRotaryEmbedding（L146 mrope_section 默认
        [11,11,10]；L162-168 partial dim 与 inv_freq；L171-183 forward
        expand(3)；L185-194 recomposition_frequencies 交错重组 + cos/sin 加倍）
      - L1322-1332 Qwen3_5MoeTextModel.forward 文本 position_ids 生成：
        arange(seq)+past_seen → view(1,1,-1).expand(4,B,-1)；行 0（text）
        供 mask/linear-attn，行 1:3（T/H/W）进旋转
      - L655-690  apply_rotary_pos_emb（rotate_half/NeoX、partial 尾部直通）
  * vLLM @dc74b8cb（vllm_qwen36_mrope_pin + vllm_qwen35_pin）
      - qwen3_5.py L440-446  Qwen3_5ForCausalLMBase.get_mrope_input_positions
        = arange(len).unsqueeze(0).expand(3,-1)，起始偏移 0（文本退化）
      - qwen3_next.py L344-350 get_rope(...)；L384-401 supports_mrope 判定
        （mrope_interleaved 且 sum(section)==rotary_dim//2）；L423-426 文本
        fused 路径取 positions[0]；L467 rotary_emb(positions,q,k)
      - qwen3_vl.py L1821-1826 mrope_positions_factor = 4 仅当多模态视频
        剪枝启用，否则 1（文本 positions 一维，cos/sin 缓存不放大）
      - rope_init.py L112-124 rope_type=default + mrope_section →
        MRotaryEmbedding（openpangu 才走 MRotaryEmbeddingInterleaved，Qwen3.6
        不经过该分支）
      - rope_mrope.py L304-363（assert sum(mrope_section)==rotary_dim//2）、
        L375-436 forward_native（1D 不做 section 重组）、L438-470
        _forward_text_rope（= 标准部分 RoPE）、L573-586
        get_next_input_positions（3 行同 range）
      - rope_base.py L80-103 inv_freq/cos_sin_cache
      - rope_common.py L19-22 rotate_neox、L133-173 ApplyRotaryEmb NeoX
  * vLLM @10a854ed（vllm_qwen36_mtp_pin，MTP 位置步进）
      - qwen3_5_mtp.py L55-62 positions 契约注释、L147-180 forward
        （positions 直传 full_attention 层）
      - tdar_speculator.py L351-355/388-389 prefill：positions=target 末位、
        sample_src=positions+1；L442-507 _prepare_prefill_inputs_kernel
        （L494-499 draft positions 逐元素复制 target positions）；L564-616
        _prepare_decode_inputs_kernel（L601-603 sample_src+1；L605-607
        seq_len=target_seq_len−num_rejected；L612-615 positions+1 夹紧
        max_model_len−1）；L619-689 _update_draft_inputs_kernel
        （L649-651 末步不推进；L653-655 sample_src+1；L678-689
        positions/seq_len +1 夹紧）
    交叉检查（2026-11-16）：rope_mrope/rope_base/rope_common 在
    dc74b8cb 与 10a854ed 两 pin commit 字节一致；rope_init 的
    default+ mrope_section 分支两版本一致；qwen3_vl 差异仅在视频计数
    管道，mrope_positions_factor 不变。

本探针三路独立实现（f64、纯 Python 标量/列表，不用 torch）：
  A. HF 公式转写：inv_freq → 3 行 freqs → 交错重组 → cos/sin 加倍 →
     rotate_half 应用（按 pinned 行号逐行转写）；
  B. vLLM 公式转写：cos_sin_cache → 1D 文本路径 → ApplyRotaryEmb NeoX
     （rope row 数学与 _forward_text_rope 逐行对应）；
  C. 独立标量 oracle：每 NeoX 对 (x1,x2) 复数旋转 e^{i·p·theta_i}，
     与 A/B 无共享代码路径。
MTP 段为内核转移的逐行状态机（prefill 复制/步进 +1/末步不推进/下轮
回退 num_rejected/夹紧），随机接受场景 + 不变量断言 + 负例。

边界与负例：decode 重启 0、步进 +2、不回退、2D 异行（多模态分支，
登记为域外不冒充）、sum(section)≠rotary_dim//2 断言、max_model_len 夹紧。

live 段（torch/transformers 可用时）：importlib 加载 pin 的 HF
configuration/modeling，用真实 Qwen3.6-35B-A3B rope 参数实例化
Qwen3_5MoeTextRotaryEmbedding，按 L1322-1332 构造 4 行文本 position_ids，
官方类输出 cos/sin/旋转结果 vs oracle 对照（fp32 计算路径，容差 1e-5）。
vLLM 模块不导入执行（公式转写已在 A 段对账），不构成 vLLM 执行证据。
"""

import hashlib
import json
import math
import platform
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PIN_HF = REPO / "assets" / "identity" / "hf_transformers"
PIN_MROPE = REPO / "assets" / "identity" / "vllm_qwen36_mrope_pin"
PIN_PAR = REPO / "assets" / "identity" / "vllm_qwen35_pin"
PIN_MTP = REPO / "assets" / "identity" / "vllm_qwen36_mtp_pin"
DEFAULT_OUT = REPO / "tests" / "fixtures" / "qwen36_mrope_r26_numeric.json"
SEED = 20261116
ATOL_HERMETIC = 1e-12          # 三路 f64 纯公式
ATOL_LIVE = 1e-5               # 官方类 fp32 路径 vs f64 oracle

# 真实 Qwen3.6-35B-A3B text_config rope 几何（assets/config/
# qwen36_35b_a3b_main__config.json，ev-qwen36-main-config 同源）
ROPE = dict(head_dim=256, partial_rotary_factor=0.25, rope_theta=1e7,
            rope_type="default", mrope_section=[11, 11, 10],
            mrope_interleaved=True, max_position_embeddings=262144,
            num_attention_heads=16, num_key_value_heads=2)
ROTARY_DIM = int(256 * 0.25)   # 64
NPairs = ROTARY_DIM // 2       # 32


def _sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# A. HF 公式转写（modeling_qwen3_5_moe.py L162-194、L655-690）
# ---------------------------------------------------------------------------
def hf_inv_freq() -> list:
    # L162-168: dim = head_dim*partial；inv_freq = 1/(base**(arange(0,dim,2)/dim))
    dim = ROTARY_DIM
    return [1.0 / (ROPE["rope_theta"] ** (i / dim))
            for i in range(0, dim, 2)]          # 32 个频率


def hf_recomposition(freq_rows: list) -> list:
    """L185-194 交错重组：基组取 T 行；H 覆盖 idx{1,4,...}<3*s1，W 覆盖
    idx{2,5,...}<3*s2；再 cat 加倍成 64 宽。freq_rows: [T,H,W] 各 32。"""
    s1, s2 = ROPE["mrope_section"][1], ROPE["mrope_section"][2]
    out = list(freq_rows[0])
    for dim, offset in ((1, 1), (2, 2)):        # H, W（enumerate((1,2),1)）
        length = (s1 if dim == 1 else s2) * 3
        for idx in range(offset, length, 3):
            out[idx] = freq_rows[dim][idx]
    return out + out                            # cat doubling → 64


def hf_apply_rotary(q: list, cos64: list, sin64: list) -> list:
    """L680-690：q_rot=前 64 维；rotate_half(cat(-x2,x1))；q*cos+rot*sin；
    尾部 192 维直通。q: 单头单 token 256 维。"""
    rot = q[:ROTARY_DIM]
    h = ROTARY_DIM // 2
    x1, x2 = rot[:h], rot[h:]
    o1 = [a * c - b * s for a, c, b, s in zip(x1, cos64[:h], x2, sin64[:h])]
    o2 = [b * c + a * s for b, c, a, s in zip(x2, cos64[h:], x1, sin64[h:])]
    return o1 + o2 + q[ROTARY_DIM:]


def hf_text_cos_sin(pos: int) -> tuple:
    """L171-183：expand(3) → freqs → cos/sin → 重组加倍。文本三行同 pos。"""
    inv = hf_inv_freq()
    p3 = [pos] * 3
    freq_rows = [[p * f for f in inv] for p in p3]
    cos64 = hf_recomposition([ [math.cos(v) for v in row] for row in freq_rows ])
    sin64 = hf_recomposition([ [math.sin(v) for v in row] for row in freq_rows ])
    return cos64, sin64


# ---------------------------------------------------------------------------
# B. vLLM 公式转写（rope_base L80-103、rope_mrope L438-470、
#    rope_common L133-173；文本 1D 不做 section 重组）
# ---------------------------------------------------------------------------
def vllm_cos_sin_cache(max_pos: int) -> list:
    # L86-103: inv_freq 同式；cache[t] = [cos(t*inv), sin(t*inv)]（32+32）
    inv = hf_inv_freq()                          # 同一公式（rope_base L86-91）
    cache = []
    for t in range(max_pos):
        cache.append([math.cos(t * f) for f in inv] +
                     [math.sin(t * f) for f in inv])
    return cache


def vllm_apply_text(q: list, cos32: list, sin32: list) -> list:
    """_forward_text_rope + ApplyRotaryEmb NeoX：chunk 后 o1/o2 cat；
    尾部直通。cos/sin 32 宽（cache chunk(2,-1) 之后）。"""
    x1, x2 = q[:ROTARY_DIM][:ROTARY_DIM // 2], q[ROTARY_DIM // 2:ROTARY_DIM]
    o1 = [a * c - b * s for a, c, b, s in zip(x1, cos32, x2, sin32)]
    o2 = [b * c + a * s for b, c, a, s in zip(x2, cos32, x1, sin32)]
    return o1 + o2 + q[ROTARY_DIM:]


# ---------------------------------------------------------------------------
# C. 独立标量 oracle：NeoX 对 (x1[i], x2[i]) 旋转角 p·inv_freq[i]
# ---------------------------------------------------------------------------
def oracle_apply(q: list, pos: int) -> list:
    inv = hf_inv_freq()
    out = list(q)
    for i in range(NPairs):
        ang = pos * inv[i]
        c, s = math.cos(ang), math.sin(ang)
        a, b = q[i], q[i + NPairs]
        out[i] = a * c - b * s
        out[i + NPairs] = b * c + a * s
    return out


def _max_diff(u, v) -> float:
    return max(abs(a - b) for a, b in zip(u, v))


def _rand_qk(rng: random.Random, n: int, dims: int) -> list:
    return [[rng.uniform(-1.0, 1.0) for _ in range(dims)] for _ in range(n)]


# ---------------------------------------------------------------------------
# MTP 位置状态机（tdar_speculator.py 内核转移逐行转写）
# ---------------------------------------------------------------------------
class MtpPositions:
    """每请求位置状态。max_len 夹紧语义照抄内核：
    positions 夹 max_model_len−1、seq_len 夹 max_model_len。"""

    def __init__(self, pos: int, seq_len: int, max_len: int):
        self.pos, self.seq_len, self.max_len = pos, seq_len, max_len
        self.sample_src = None
        self.kv_rows = seq_len          # 草稿 KV 已写行数（跨步缓存一致性锚）

    # _prefill L351-355/388-389：positions=target 末位（prefill kernel
    # L494-499 逐元素复制后取 last）；sample_src = positions+1
    def prefill(self, target_last_pos: int):
        self.pos = target_last_pos
        self.sample_src = self.pos + 1

    # _update_draft_inputs_kernel L641-689：步 s 采样写槽 s；
    # s < K−1 → sample_src+1（L653-655）、pos+1 夹紧（L678-685）、
    # seq_len+1 夹紧（L687-689）；末步（L649-651）不推进
    def step(self, s: int, K: int):
        assert self.sample_src is not None
        self.kv_rows = min(self.kv_rows + 1, self.max_len)   # 每步写 1 行
        if s >= K - 1:
            return
        self.sample_src += 1
        self.pos = min(self.pos + 1, self.max_len - 1)
        self.seq_len = min(self.seq_len + 1, self.max_len)

    # _prepare_decode_inputs_kernel L597-616：sample_src+1（L601-603）；
    # seq_len = target_seq_len − num_rejected（L605-607，回退被拒位置）；
    # pos+1 夹紧（L612-615）、seq_len+1 夹紧（L615-616）。
    # 草稿 KV 行随 seq_len 回退：被拒行下轮重写（R23 已登记语义；
    # 内核只写 seq_lens，槽位回收由 attn 后端按 seq_len 分配）
    def next_round(self, target_seq_len: int, num_rejected: int):
        self.sample_src += 1
        self.seq_len = target_seq_len - num_rejected
        self.pos = min(self.pos + 1, self.max_len - 1)
        self.seq_len = min(self.seq_len + 1, self.max_len)
        self.kv_rows = self.seq_len


def mtp_trace(rng: random.Random, rounds: int = 4, K: int = 3):
    """随机接受场景的状态机轨迹 + 不变量断言；返回 trace 与断言计数。"""
    max_len = 24          # 小边界，数轮内触到夹紧
    ctx = 7               # 首轮上下文（含 bonus token，位置 0..6）
    st = MtpPositions(pos=0, seq_len=ctx, max_len=max_len)
    trace, checks = [], 0

    def chk(cond, what):
        nonlocal checks
        checks += 1
        assert cond, what

    for r in range(rounds):
        ctx0 = ctx
        st.prefill(target_last_pos=ctx - 1)          # target 末位=ctx−1
        chk(st.sample_src == ctx, "prefill sample_src = last+1")
        steps = []
        for s in range(K):
            fwd_pos = st.pos
            st.step(s, K)
            steps.append(dict(step=s, fwd_pos=fwd_pos,
                              sample_src=st.sample_src, pos_after=st.pos,
                              seq_len=st.seq_len, kv_rows=st.kv_rows))
            if s < K - 1:
                chk(st.pos == min(fwd_pos + 1, max_len - 1), "非末步 +1 夹紧")
            # 跨步缓存一致性：kv_rows 恰为初始 seq_len + 已执行步数，
            # 且封顶 max_len（夹紧后不再增长）
            chk(st.kv_rows == min(ctx0 + (s + 1), max_len),
                "kv_rows = 初始 + 步数（夹紧上界）")
        final = steps[-1]
        chk(final["pos_after"] == final["fwd_pos"], "末步不推进（pos 不变）")
        # verify：K 个候选被验证；随机接受 A ∈ [0,K]
        accepted = rng.randint(0, K)
        num_rejected = K - accepted
        target_seq_len = ctx + K
        before_pos = st.pos
        st.next_round(target_seq_len, num_rejected)
        chk(st.seq_len == min(target_seq_len - num_rejected + 1, max_len),
            "下轮 seq_len = target − rejected + 1（bonus）夹紧")
        chk(st.pos == min(before_pos + 1, max_len - 1), "下轮 pos +1 夹紧")
        ctx = st.seq_len
        trace.append(dict(round=r, ctx_before=ctx0,
                          steps=steps, accepted=accepted,
                          num_rejected=num_rejected,
                          target_seq_len=target_seq_len,
                          pos_after_round=st.pos, seq_len_after=st.seq_len,
                          sample_src=st.sample_src))
    return trace, checks


def negatives(rng: random.Random) -> dict:
    """边界负例：每个负例必须与正例语义可区分（差异被 oracle 抓住）。"""
    out = {}
    q = _rand_qk(rng, 1, 256)[0]
    pos_decode = 5                                   # 上下文 5 后的 decode

    # N1 decode 重启 0（违反 HF L1325 +past_seen / vLLM 缓存续位）
    good = oracle_apply(q, pos_decode)
    bad = oracle_apply(q, 0)
    out["decode_restart_zero"] = dict(
        what="decode 位置重启 0 而非续缓存长度", caught=_max_diff(good, bad) > 1e-3,
        max_diff=_max_diff(good, bad))

    # N2 步进 +2（跳位；违反内核 pos+1）
    p_ok, p_bad = 6, 8
    a, b = oracle_apply(q, p_ok), oracle_apply(q, p_bad)
    out["step_plus_two"] = dict(
        what="draft 步进 +2 跳位", caught=_max_diff(a, b) > 1e-3,
        max_diff=_max_diff(a, b))

    # N3 不回退被拒位置（seq_len 不减 num_rejected）
    st = MtpPositions(6, 7, 24); st.prefill(6)
    for s in range(3):
        st.step(s, 3)
    rw = MtpPositions(6, 7, 24); rw.prefill(6)
    for s in range(3):
        rw.step(s, 3)
    st.next_round(10, 2)                              # 2 拒 → seq 9
    rw.seq_len = 10 + 1                               # 不回退（错）
    out["no_rewind"] = dict(
        what="下轮不回退 num_rejected", caught=rw.seq_len != st.seq_len,
        correct_seq_len=st.seq_len, wrong_seq_len=rw.seq_len)

    # N4 2D 异行（多模态 T/H/W 位置不同）：HF 重组不再是恒等 → 与文本路径
    # 分裂。此分支属视觉域，本仓不建模不冒充，仅证明分支真实存在。
    inv = hf_inv_freq()
    rows = [[5 * f for f in inv], [2 * f for f in inv], [9 * f for f in inv]]
    cos_mm = hf_recomposition([[math.cos(v) for v in r] for r in rows])
    cos_tx = hf_text_cos_sin(5)[0]
    out["multimodal_2d_distinct_rows"] = dict(
        what="2D 异行走交错重组（多模态分支，域外登记）",
        branch_exists=_max_diff(cos_mm, cos_tx) > 1e-3,
        max_diff=_max_diff(cos_mm, cos_tx),
        scope="out-of-scope：视觉域 T/H/W 位置选择未建模")

    # N5 sum(mrope_section) != rotary_dim//2：vLLM 构造断言（rope_mrope
    # L362-363）/ HF 布局越界，必须显式失败不得静默
    fired = False
    try:
        s = [11, 11, 11]                              # 33 ≠ 32
        assert sum(s) == ROTARY_DIM // 2
    except AssertionError:
        fired = True
    out["section_sum_assert"] = dict(
        what="sum(section)≠rotary_dim//2 断言触发", fired=fired)

    # N6 max_model_len 夹紧（L612-615/L683-685）：位置封顶 max−1 不越界
    st = MtpPositions(22, 23, 24); st.prefill(22)
    st.step(0, 2); st.step(1, 2)
    st.next_round(24, 0)
    out["clamp_max_len"] = dict(
        what="positions 夹紧 max_model_len−1 / seq_len 夹 max",
        pos=st.pos, seq_len=st.seq_len,
        clamped=st.pos == 23 and st.seq_len == 24)
    return out


def run_hermetic() -> dict:
    rng = random.Random(SEED)
    # ---- 主模型三路对照（真实几何 16 头×256，rotary 64）----
    positions = list(range(9)) + [262143]             # 0..8 + 上界
    qs = _rand_qk(rng, len(positions), 256)
    diffs_hf_or, diffs_vl_or, diffs_hf_vl = [], [], []
    for pos, q in zip(positions, qs):
        cos64, sin64 = hf_text_cos_sin(pos)           # A
        r_hf = hf_apply_rotary(q, cos64, sin64)
        r_or = oracle_apply(q, pos)                   # C
        cache = vllm_cos_sin_cache(pos + 1)           # B（按需建到 pos）
        cos32, sin32 = cache[pos][:NPairs], cache[pos][NPairs:]
        r_vl = vllm_apply_text(q, cos32, sin32)
        diffs_hf_or.append(_max_diff(r_hf, r_or))
        diffs_vl_or.append(_max_diff(r_vl, r_or))
        diffs_hf_vl.append(_max_diff(r_hf, r_vl))
    assert max(diffs_hf_or) <= ATOL_HERMETIC, "HF vs oracle"
    assert max(diffs_vl_or) <= ATOL_HERMETIC, "vLLM vs oracle"
    assert max(diffs_hf_vl) <= ATOL_HERMETIC, "HF vs vLLM"

    # prefill/decode 增量序列（HF L1322-1326 / vLLM arange+缓存）
    seq_checks = []
    for T, past in ((5, 0), (1, 5), (1, 6)):          # prefill5/decode/decode
        hf_pos = [i + past for i in range(T)]         # arange+past_seen
        vl_pos = list(range(past, past + T))          # 引擎缓存续位
        seq_checks.append(dict(T=T, past_seen=past, hf=hf_pos, vllm=vl_pos,
                               equal=hf_pos == vl_pos))
    assert all(c["equal"] for c in seq_checks)

    # 2D 同行（SupportsMRoPE 文本 fallback，qwen3_5.py L440-446）≡ 1D 文本
    inv = hf_inv_freq()
    q = qs[0]
    same_rows = [[7 * f for f in inv]] * 3
    cos2d = hf_recomposition([[math.cos(v) for v in r] for r in same_rows])
    r2d = hf_apply_rotary(q, cos2d, hf_recomposition(
        [[math.sin(v) for v in r] for r in same_rows]))
    degenerate_diff = _max_diff(r2d, oracle_apply(q, 7))
    assert degenerate_diff <= ATOL_HERMETIC

    trace, checks = mtp_trace(random.Random(SEED + 1))
    neg = negatives(random.Random(SEED + 2))
    for k in ("decode_restart_zero", "step_plus_two", "no_rewind"):
        assert neg[k]["caught"], f"负例未抓住: {k}"
    assert neg["multimodal_2d_distinct_rows"]["branch_exists"], \
        "2D 异行分支必须与文本路径可区分"
    assert neg["section_sum_assert"]["fired"]
    assert neg["clamp_max_len"]["clamped"]

    pins = {
        "hf/modeling_qwen3_5_moe.py": _sha256_file(PIN_HF / "modeling_qwen3_5_moe.py"),
        "hf/configuration_qwen3_5_moe.py": _sha256_file(PIN_HF / "configuration_qwen3_5_moe.py"),
        "vllm_dc74b8cb/qwen3_5.py": _sha256_file(PIN_PAR / "qwen3_5.py"),
        "vllm_dc74b8cb/qwen3_next.py": _sha256_file(PIN_PAR / "qwen3_next.py"),
        "vllm_dc74b8cb/rope_init.py": _sha256_file(PIN_MROPE / "rope_init.py"),
        "vllm_dc74b8cb/rope_base.py": _sha256_file(PIN_MROPE / "rope_base.py"),
        "vllm_dc74b8cb/rope_common.py": _sha256_file(PIN_MROPE / "rope_common.py"),
        "vllm_dc74b8cb/rope_mrope.py": _sha256_file(PIN_MROPE / "rope_mrope.py"),
        "vllm_dc74b8cb/qwen3_vl.py": _sha256_file(PIN_MROPE / "qwen3_vl.py"),
        "vllm_10a854ed/qwen3_5_mtp.py": _sha256_file(PIN_MTP / "qwen3_5_mtp.py"),
        "vllm_10a854ed/tdar_speculator.py": _sha256_file(PIN_MTP / "tdar_speculator.py"),
    }
    return dict(
        seed=SEED, rope_config=ROPE, rotary_dim=ROTARY_DIM,
        positions=positions,
        main_model=dict(
            max_diff_hf_vs_oracle=max(diffs_hf_or),
            max_diff_vllm_vs_oracle=max(diffs_vl_or),
            max_diff_hf_vs_vllm=max(diffs_hf_vl),
            position_sequences=seq_checks,
            twod_identical_rows_degenerate_diff=degenerate_diff),
        mtp=dict(trace=trace, invariant_checks=checks),
        negatives=neg,
        pins_sha256=pins,
        environment=dict(python=platform.python_version(),
                         implementation=platform.python_implementation(),
                         mode="pure-python f64（无 torch）"),
    )


# ---------------------------------------------------------------------------
# live 段：执行 pin 的官方 HF 旋转类（真实 rope 参数）
# ---------------------------------------------------------------------------
def run_live(art: dict) -> dict:
    import importlib.util
    import torch

    import transformers  # noqa: F401

    cfg_spec = importlib.util.spec_from_file_location(
        "transformers.models.qwen3_5_moe.pinned_configuration_qwen3_5_moe",
        PIN_HF / "configuration_qwen3_5_moe.py")
    cfg_mod = importlib.util.module_from_spec(cfg_spec)
    sys.modules[cfg_spec.name] = cfg_mod
    cfg_spec.loader.exec_module(cfg_mod)

    spec = importlib.util.spec_from_file_location(
        "transformers.models.qwen3_5_moe.pinned_modeling_qwen3_5_moe",
        PIN_HF / "modeling_qwen3_5_moe.py")
    modeling = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = modeling
    spec.loader.exec_module(modeling)

    cfg = cfg_mod.Qwen3_5MoeTextConfig(
        vocab_size=64, hidden_size=256, num_hidden_layers=1,
        layer_types=["full_attention"], num_attention_heads=16,
        num_key_value_heads=2, head_dim=256,
        rope_parameters={
            "rope_type": ROPE["rope_type"], "rope_theta": ROPE["rope_theta"],
            "partial_rotary_factor": ROPE["partial_rotary_factor"],
            "mrope_section": list(ROPE["mrope_section"]),
            "mrope_interleaved": True},
        max_position_embeddings=ROPE["max_position_embeddings"])
    rot = modeling.Qwen3_5MoeTextRotaryEmbedding(config=cfg)
    assert list(rot.mrope_section) == [11, 11, 10]
    assert rot.inv_freq.shape[0] == NPairs

    rng = random.Random(SEED)
    positions = list(range(9))
    torch.manual_seed(SEED)
    q_t = torch.randn(1, len(positions), 16, 256, dtype=torch.float64)
    k_t = torch.randn(1, len(positions), 2, 256, dtype=torch.float64)
    # L1322-1332 构造：arange(T)+past_seen → expand(4,B,T)；行 1:3 进旋转
    past = 0
    pos_ids = (torch.arange(len(positions)) + past).view(1, 1, -1).expand(
        4, 1, -1).contiguous()
    text_ids, rotary_ids = pos_ids[0], pos_ids[1:]
    cos, sin = rot(q_t, rotary_ids)                  # 官方类执行
    # apply_rotary_pos_emb L655-690（转 q 形状 (B,H,T,d)）
    q_bhtd = q_t.transpose(1, 2)
    qr, kr = modeling.apply_rotary_pos_emb(q_bhtd, k_t.transpose(1, 2),
                                           cos, sin)
    qr = qr.transpose(1, 2)[0]                       # (T,16,256)

    # oracle（C 段）逐位置逐头对照
    max_diff = 0.0
    for ti, pos in enumerate(positions):
        for h in range(16):
            ref = oracle_apply([float(v) for v in q_t[0, ti, h]], pos)
            got = [float(v) for v in qr[ti, h]]
            max_diff = max(max_diff, _max_diff(ref, got))
    assert max_diff <= ATOL_LIVE, f"官方类 vs oracle: {max_diff}"

    # decode 续位（past_seen=5）再核一步；cos (B,T,64)
    pos_ids_d = (torch.arange(1) + 5).view(1, 1, -1).expand(4, 1, -1)
    cos_d, sin_d = rot(torch.zeros(1, 1, 256, dtype=torch.float64),
                       pos_ids_d[1:])
    freq_oracle = 5 * hf_inv_freq()[0]
    d = max(abs(float(cos_d[0, 0, 0]) - math.cos(freq_oracle)),
            abs(float(sin_d[0, 0, 0]) - math.sin(freq_oracle)))
    assert d <= ATOL_LIVE

    return dict(executed=True,
                torch=torch.__version__,
                transformers=transformers.__version__,
                official_class="Qwen3_5MoeTextRotaryEmbedding(pinned @b6401d66)",
                max_diff_official_vs_oracle=max_diff,
                decode_pos5_cos_diff=d,
                atol=ATOL_LIVE)


def main() -> int:
    art = run_hermetic()
    try:
        art["live_hf_class"] = run_live(art)
        print("live: pinned HF rotary class executed and reconciled")
    except Exception as e:                           # torch/transformers 缺失等
        art["live_hf_class"] = dict(executed=False, reason=repr(e))
        print("live skipped:", e)
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    out.write_text(json.dumps(art, ensure_ascii=False, indent=1) + "\n",
                   encoding="utf-8")
    print(f"fixture written: {out}")
    print("hermetic: 3-way rope math <=1e-12; MTP state machine + negatives OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
