"""M0 红灯测试：已知错误类别必须被 v0.2 契约/实现捕获。

每条测试注明“哪个错误由哪个断言发现”（plan §8.1）。这些测试针对新管线；
v0.1 legacy 模块的对应缺陷在 evidence-audit E05/E07/E16/E20/E21 登记，
由迁移后旧测试退休/改写时逐项对账（见 docs/IMPLEMENTATION_STATUS.md）。
"""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simverse.contracts import (Metric, Quantity, ResultKind, Statistic,  # noqa: E402
                                Support, canonical_json, digest, js_restore, js_safe)
from simverse.validation import build_registry, code_revision

_spec = importlib.util.spec_from_file_location("golden_v02", ROOT / "golden" / "oracles_v02.py")
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)


# ---------------------------------------------------------------------------
# E05/口径：期望值不得 int() 冒充精确；exact 不得携带非整浮点
# ---------------------------------------------------------------------------

def test_expectation_quantity_must_be_float():
    """错误：int(union*bytes) 把期望截断成“精确整数”。
    断言：Quantity(statistic=expectation) 且传 int → TypeError。"""
    with pytest.raises(TypeError):
        Quantity(value=int(G.MOE_E4_K2_M2_RATIO * 1000), unit="B",
                 statistic=Statistic.EXPECTATION, formula_ref="f")


def test_exact_metric_rejects_fractional_float():
    """错误：把期望均值直接写成 exact 字段。
    断言：statistic=exact + 非整浮点 → ValueError。"""
    with pytest.raises(ValueError):
        Metric(value=G.MOE_E4_K2_M2_WITH_REPL, unit="experts",
               statistic=Statistic.EXACT)


def test_moe_formula_distinction_from_with_replacement():
    """错误：M·k 次有放回抽签近似被当作并集公式（v0.1 ledger 现状）。
    断言：黄金枚举 ≠ 有放回值，两口径必须可区分。"""
    assert G.MOE_E4_K2_M2_ENUM == 3.0
    assert G.MOE_E4_K2_M2_WITH_REPL == 2.734375
    assert G.MOE_E4_K2_M2_ENUM != G.MOE_E4_K2_M2_WITH_REPL
    assert G.MOE_M1_GIVES_K == 8.0                      # 单 token 恰好 k 个专家
    assert G.MOE_M1_WITH_REPL_LT_K < 8.0                # 有放回在 M=1 端点即露馅


# ---------------------------------------------------------------------------
# E15/E16/V01：null+reason、非有限数拒绝、完整 hash、大整数往返
# ---------------------------------------------------------------------------

def test_null_requires_reason():
    """错误：缺数据填 0 或空。断言：value=None 无 reason → ValueError。"""
    with pytest.raises(ValueError):
        Metric(value=None, unit="ms")


def test_zero_is_deliberate():
    """已确定不存在才是 0：value=0 合法且带 support=exact 语义。"""
    m = Metric(value=0, unit="tokens", statistic=Statistic.EXACT)
    assert m.value == 0 and m.reason is None


def test_nan_rejected_everywhere():
    """错误：NaN/Infinity 进入 JSON 结果。断言：finite/canonical_json 拒绝。"""
    with pytest.raises(ValueError):
        Metric(value=float("nan"), unit="ms")
    with pytest.raises(ValueError):
        canonical_json({"x": float("inf")})
    with pytest.raises(ValueError):
        js_safe({"x": float("nan")})


def test_digest_is_full_sha256():
    """E16：v0.1 digest 截断 16 hex。断言：v0.2 digest 为完整 64 hex。"""
    d = digest({"a": 1})
    assert d.startswith("sha256:") and len(d) == 7 + 64
    assert digest({"a": 1}) == digest({"a": 1})
    assert digest({"a": 1}) != digest({"a": 2})


def test_bigint_roundtrip_js():
    """V01：>2^53 整数经 js_safe/js_restore 无损往返。"""
    big = 2**60 + 12345
    payload = {"flops": big, "small": 42}
    assert js_safe(payload)["flops"] == {"$bigint": str(big)}
    assert js_restore(js_safe(payload)) == payload


# ---------------------------------------------------------------------------
# E07/§7.7：mean 不决定分布；动态 K 缺联合律 → 未知
# ---------------------------------------------------------------------------

def test_mean_does_not_determine_pfull():
    """E07：同均值两种分布 P_full 0 vs 0.5。
    断言：黄金反例中两世界的 P_full 必须不同 → mean 输入不可反解分布。"""
    assert G.MEAN_WORLD_A["mean"] == G.MEAN_WORLD_B["mean"] == 2.0
    assert G.MEAN_WORLD_A["P_full"] != G.MEAN_WORLD_B["P_full"]


def test_dynamic_k_joint_law_counterexample():
    """Z2-01：同边际不同联合律 → E[N]/TPOT 可辨识地不同。
    断言：边际相同而 E[N] 不同 → 缺联合律时只能返回 null+reason。"""
    assert G.DYN_MARGINAL_A == G.DYN_MARGINAL_B == 0.7
    assert G.DYN_WORLD_A["E_N"] != G.DYN_WORLD_B["E_N"]
    assert abs(G.DYN_WORLD_A["TPOT"] - 3.883495) < 1e-4
    assert abs(G.DYN_WORLD_B["TPOT"] - 4.040404) < 1e-4
    # 契约侧：动态 K 无联合律的指标必须显式 unknown（不是数值）
    m = Metric.unknown("ms", "动态 K 缺联合/条件分布：边际 α_i 与 K 分布不足以确定 E[N]")
    assert m.value is None and m.support is Support.UNKNOWN


# ---------------------------------------------------------------------------
# E09/V03：逻辑算量不被融合吞掉
# ---------------------------------------------------------------------------

def test_residual_adds_are_counted():
    """E09：融合后残差逻辑加法不消失。断言：黄金值 4×d=1024 存在且非零。"""
    assert G.RESIDUAL_ADDS_M4_D256 == 1024


def test_causal_pairs_formula_boundaries():
    """V03：S=100/Q=1→101；Q=3→306；禁止 S 与 Q 重复计数。"""
    assert G.CAUSAL_S100_Q1 == 101
    assert G.CAUSAL_S100_Q3 == 306


# ---------------------------------------------------------------------------
# M0 证据登记
# ---------------------------------------------------------------------------

def test_evidence_registry_content_hashes():
    """证据绑定内容 hash；必需场景身份证据在册。"""
    reg = build_registry()
    for ev_id in ("ev-v41-config", "ev-qwen36-main-config",
                  "ev-qwen36-dflash-config", "ev-dflash-impl"):
        assert ev_id in reg, ev_id
        assert reg[ev_id].content_sha256 not in ("", "MISSING"), ev_id
        assert len(reg[ev_id].content_sha256) == 64


def test_code_revision_available():
    """代码身份可追溯（git HEAD 或显式 unknown）。"""
    rev = code_revision()
    assert rev and rev != ""


def test_roofline_and_link_anchors():
    """V04-cost 独立手算锚点：F=1000,P=100,Q=200,BW=50 → 10 s；60KB/64GBs ≈ 0.94µs。"""
    assert G.ROOFLINE_ANCHOR["t_roofline"] == 10.0
    assert abs(G.LINK_ANCHOR["t_transfer_s"] - 0.9375e-6) < 1e-9
    assert G.LINK_ANCHOR["t_total_s"] > G.LINK_ANCHOR["t_transfer_s"] * 100


def test_metric_oracle_trace_facts():
    """V08 人工轨迹：同批同刻 token、窗口计数、busy 积分不双计重叠。"""
    assert G.TRACE_TOKENS["r2"][0] == G.TRACE_TOKENS["r2"][1]   # 允许同刻
    assert G.WINDOW_TOKEN_COUNT == 6
    assert G.BUSY_INTEGRAL == 25                                 # (0,12)+(20,33) 无重叠
    assert G.R1_MEAN_ITL == 10.0
