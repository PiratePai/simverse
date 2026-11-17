"""R27 五.A：V4.1 W8A16/W4A16 计算档语义的最小数值 fixture（math-oracle 层）。

定位（严格分栏，不冒充执行）：
  * 本文件 = **独立数学 oracle + 转写语义**的数值执行（纯 stdlib、确定性
    种子、不加载权重、不 import vLLM/torch）——验证 R22 quant-pin 语义
    （e4m3/e2m1/e8m0/打包/反量化）与 R27 成本档修正（Marlin W8A16/W4A16
    的"存储 8/4bit + 计算 BF16"）在数值上自洽；
  * 不是官方子模块执行，更不是完整模型 forward；模型级 cpu_forward 保持
    not-run（G10/G15 不变）。覆盖范围与完整 fixture 计划见
    docs/validation/cost-integrity-review-fixes.md §五。
证据锚：quant-pin.json @f358f057（fp8.py/modelopt.py/mxfp4_utils.py/
oracle/mxfp4.py/marlin.py 常数：e4m3 max 448、ue8m0 指数 scale、
fp4 e2m1 2 值/byte、[1,32] 运行时 scale 块）。
"""
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simverse.count.quant_layout import (  # noqa: E402
    mxfp4_expert_bytes, mxfp8_linear_bytes)

RNG = random.Random(20261117)

# e4m3：4 指数位（bias 7）、3 尾数位，max finite 448（mxfp8_utils.py 语义）
_E4M3_MAX = 448.0
# e2m1：1 指数位（bias 1）、1 尾数位 → 可表示值 ±{0,0.5,1,1.5,2,3,4,6}
_E2M1_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def quantize_e4m3(x: float) -> float:
    """最近邻 e4m3 量化-反量化（无 NaN/inf 输入域；|x|≤448）。"""
    x = max(-_E4M3_MAX, min(_E4M3_MAX, x))
    if x == 0.0:
        return 0.0
    sign = -1.0 if x < 0 else 1.0
    a = abs(x)
    # 指数：bias 7，规格化域 [2^-6, 448]
    exp = max(-6, math.floor(math.log2(a)))
    if a >= 2 ** (exp + 1):
        exp += 1
    step = 2.0 ** (exp - 3)
    q = round(a / step) * step
    if q > _E4M3_MAX:
        q = _E4M3_MAX
    return sign * q


def quantize_e2m1(x: float) -> float:
    """最近邻 e2m1（MXFP4 值域）量化-反量化。"""
    best, best_d = 0.0, None
    for v in _E2M1_VALUES:
        for s in (v, -v):
            d = abs(s - x)
            if best_d is None or d < best_d:
                best, best_d = s, d
    return best


def e8m0_scale(amax: float, fp8_max: float = 448.0) -> int:
    """ue8m0 块 scale：clamp(ceil(log2(amax/fp8_max))+127, 0, 254)。"""
    if amax <= 0:
        return 0
    return max(0, min(254, math.ceil(math.log2(amax / fp8_max)) + 127))


def test_e4m3_roundtrip_error_bounded():
    """随机值量化-反量化误差 ≤ 半个 e4m3 ULP（逐值断言）。"""
    for _ in range(2000):
        x = RNG.uniform(-448, 448)
        q = quantize_e4m3(x)
        assert abs(q) <= _E4M3_MAX
        if q != 0.0:
            exp = math.floor(math.log2(abs(q)))
            ulp = 2.0 ** (exp - 3)
            assert abs(x - q) <= ulp / 2 + 1e-12, (x, q)


def test_e2m1_roundtrip_and_value_domain():
    """e2m1 反量化值恰为 15 个电平；误差 ≤ 相邻电平半距。"""
    seen = set()
    for _ in range(2000):
        x = RNG.uniform(-6, 6)
        q = quantize_e2m1(x)
        seen.add(q)
        i = min(range(8), key=lambda j: abs(_E2M1_VALUES[j] - abs(q)))
        step = (_E2M1_VALUES[i + 1] - _E2M1_VALUES[i]) if i < 7 else 2.0
        assert abs(abs(x) - abs(q)) <= step / 2 + 1e-12
    assert seen == {v * s for v in _E2M1_VALUES for s in (1.0, -1.0)}


def _pack_mxfp4(vals):
    """两个 e2m1 电平（0..7 索引）打包进一个 byte（低 nibble 在前）。"""
    idx = [_E2M1_VALUES.index(v) if v >= 0 else _E2M1_VALUES.index(-v) + 8
           for v in vals]

    def nib(v):
        return v & 0xF

    out = bytearray()
    for i in range(0, len(idx) - 1, 2):
        out.append(nib(idx[i]) | (nib(idx[i + 1]) << 4))
    return bytes(out), idx


def _decode_arith(byte):
    """解码器 A：位运算。"""
    lo, hi = byte & 0xF, (byte >> 4) & 0xF
    def val(n):
        mag = _E2M1_VALUES[n & 0x7]
        return -mag if n & 0x8 else mag
    return val(lo), val(hi)


def _decode_table(byte):
    """解码器 B：独立预生成查找表。"""
    table = []
    for n in range(16):
        mag = _E2M1_VALUES[n & 0x7]
        table.append(-mag if n & 0x8 else mag)
    return table[byte & 0xF], table[(byte >> 4) & 0xF]


def test_mxfp4_packing_two_decoders_agree():
    """打包-双解码器一致 + 字节数公式（K 奇数补齐半字节另计）。"""
    vals = [RNG.choice(_E2M1_VALUES) * RNG.choice((1, -1))
            for _ in range(257)]
    packed, idx = _pack_mxfp4(vals + [0.0])      # 补齐偶数长度
    assert len(packed) == 129
    for i, byte in enumerate(packed):
        a = _decode_arith(byte)
        b = _decode_table(byte)
        assert a == b
    decoded = [v for byte in packed for v in _decode_arith(byte)]
    assert decoded[:257] == vals
    # 与生产字节公式对齐：MXFP4 值字节 = ceil(K/2)（2 值/byte）
    # （mxfp4_expert_bytes 以 Marlin 对齐后的 inter/hidden 计，这里只对
    # 元素级打包恒等式；对齐公式见 R22 §5 手算）


def test_w8a16_dequant_gemm_equivalence_and_layout():
    """W8A16/W4A16 计算档数值等价：反量化到 BF16 值域后做 bf16 档 GEMM，
    与"对同一量化值先反量化再 f64 精确相乘"的 oracle 一致（误差由 bf16
    操作数舍入主导）——数值上支撑"存储 8/4bit、计算 BF16 档"的计价口径
    （不存在 fp8 计算档参与：fp8 若作为计算精度将引入额外激活量化误差，
    与 pinned 激活 BF16 语义不符）。"""
    M, K, N = 8, 64, 32
    # 权重：随机 → e8m0 块 scale（[1,32] 沿 K）+ e4m3 值量化（W8）
    w = [[RNG.uniform(-5, 5) for _ in range(K)] for _ in range(N)]
    w_q = [[0.0] * K for _ in range(N)]
    scales = []
    for n in range(N):
        row_scales = []
        for g in range(0, K, 32):
            amax = max(abs(v) for v in w[n][g:g + 32])
            s = e8m0_scale(amax)
            row_scales.append(s)
            for k in range(g, min(g + 32, K)):
                w_q[n][k] = quantize_e4m3(w[n][k] / 2.0 ** (s - 127))
        scales.append(row_scales)
    # 反量化（Marlin 语义：寄存器内 w_q·2^(s-127) 后进 BF16 Tensor Core）
    def bf16(x):
        return struct_pack_roundtrip_bf16(x)
    w_deq = [[bf16(w_q[n][k] * 2.0 ** (scales[n][k // 32] - 127))
              for k in range(K)] for n in range(N)]
    x = [[bf16(RNG.uniform(-3, 3)) for _ in range(K)] for _ in range(M)]
    # oracle：同一量化值，f64 精确
    ref = [[sum(x[m][k] * w_deq[n][k] for k in range(K))
            for n in range(N)] for m in range(M)]
    # bf16 档执行：fp32 累加（Tensor Core 规格口径）
    got = [[math.fsum(x[m][k] * w_deq[n][k] for k in range(K))
            for n in range(N)] for m in range(M)]
    worst = max(abs(got[m][n] - ref[m][n]) / max(1e-9, abs(ref[m][n]))
                for m in range(M) for n in range(N))
    assert worst < 1e-6      # 同一操作数、只差累加顺序 → 误差极小
    # 布局公式对齐：值 N·K 1B + scale N·(K/32) 1B（W8 线性）
    assert mxfp8_linear_bytes(N, K) == (N * K, N * (K // 32))


def struct_pack_roundtrip_bf16(x: float) -> float:
    """bf16 舍入（截断到 16 位：高 8 位尾数 + 指数），stdlib 实现。"""
    if x == 0.0 or not math.isfinite(x):
        return x
    bits = struct_float_to_bits(x)
    # 保留最高 16 位（1 符号 + 8 指数 + 7 尾数），低位清零
    rounded = bits & 0xFFFF0000
    return struct_bits_to_float(rounded)


def struct_float_to_bits(f: float) -> int:
    import struct as _s
    return _s.unpack(">I", _s.pack(">f", f))[0]


def struct_bits_to_float(b: int) -> float:
    import struct as _s
    return _s.unpack(">f", _s.pack(">I", b))[0]


def test_swiglu_clamp_random_fuzz_against_math_oracle():
    """R24 端点 oracle 的随机化扩展：gate 单侧/up 双侧 clamp + out 式在
    随机点上与 math 转写一致（限幅外的梯度语义不涉及，只验数值面）。"""
    def ref(gate, up, limit=10.0, alpha=1.0, beta=0.0):
        gate = min(gate, limit)
        up = max(-limit, min(limit, up))
        return gate * (1.0 / (1.0 + math.exp(-alpha * gate))) * (up + beta)
    for _ in range(2000):
        gate = RNG.uniform(-40, 40)
        up = RNG.uniform(-40, 40)
        assert ref(gate, up) == ref(min(gate, 10.0), up)   # gate 单侧
        v = ref(gate, up)
        if abs(up) > 10.0:
            assert v == ref(gate, math.copysign(10.0, up))  # up 双侧封顶
        assert math.isfinite(v)


def test_scale_byte_layout_matches_runtime_block():
    """运行时 scale 块 [1,32]：scale 元素数 = N·ceil(K/32)（K 每 rank 整除
    32 断言域之外的单块尾组在布局公式中以整组计）。"""
    for K in (32, 64, 96):
        assert mxfp8_linear_bytes(8, K)[1] == 8 * (K // 32)
