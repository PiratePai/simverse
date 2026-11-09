"""手算 oracle 对账：golden/expectations.py（独立推导）vs 实现（engine/ledger）。

验收规则（方案 §8 误差规则）：整数计数精确相等；E[n]/通信字节精确相等；
roofline 浮点按重排相对误差 ≤1e-12。
"""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "golden_expectations", ROOT / "golden" / "expectations.py")
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)

from simverse.contracts import ScenarioSpec            # noqa: E402
from simverse.engine import run                        # noqa: E402


def _tiny_scenario(mode="colocated"):
    return ScenarioSpec(
        name="oracle", model_id="tiny_dense_oracle", draft_id="custom_chain",
        k=G.K, tp=1, target_device="h100_sxm", draft_device="l40",
        link="pcie5", batch=G.B, ctx_tokens=G.CTX,
        deployment={"mode": mode},
        acceptance={"kind": "chain_alpha", "chain_alpha": G.ALPHA}).validate()


def _stage_ops(result, stage):
    return [o for o in result.ops if o.stage == stage]


def test_acceptance_math_exact():
    r = run(_tiny_scenario())
    assert r.status == "ok"
    assert r.metrics["E_committed_len"] == pytest.approx(G.E_N, abs=0)      # 1.75
    assert r.metrics["P_full"] == pytest.approx(G.P_FULL, abs=0)            # 0.25
    assert r.metrics["E_draft_step0_tokens"] == pytest.approx(G.E_M0, abs=0)


def test_baseline_decode_integers():
    r = run(_tiny_scenario())
    ops = _stage_ops(r, "baseline_decode")
    assert sum(o.flops for o in ops) == G.TOTAL_FLOPS_BASE
    assert sum(o.weight_read for o in ops) == G.TOTAL_WEIGHT_BASE
    assert sum(o.kv_read for o in ops) == G.TOTAL_KV_READ_BASE
    assert sum(o.kv_write for o in ops) == G.TOTAL_KV_WRITE_BASE
    # 逐层绑定：层 0/1 各有 11 个层内算子，lm_head/embed 在 layer=-1
    assert len({o.layer for o in ops if o.layer >= 0}) == G.LAYERS
    per_layer = [o for o in ops if o.layer == 0]
    names = [o.op for o in per_layer]
    assert names == ["q_proj", "k_proj", "v_proj", "attention_core", "out_proj",
                     "gate_up_proj", "down_proj", "attn_norm", "mlp_norm",
                     "attn_add", "mlp_add"]


def test_verify_step_integers():
    """验证步 M=K+1=3：GEMM ×3、attention 对数 306、KV 读共享一次。"""
    r = run(_tiny_scenario())
    ops = _stage_ops(r, "verify")
    assert sum(o.flops for o in ops) == G.TOTAL_FLOPS_VERIFY
    assert sum(o.kv_read for o in ops) == G.TOTAL_KV_READ_VERIFY
    assert sum(o.kv_write for o in ops) == G.TOTAL_KV_WRITE_VERIFY
    core = next(o for o in ops if o.op == "attention_core" and o.layer == 0)
    assert core.flops == G.VERIFY_ATTN_FLOPS


def test_draft_ledger_integers():
    """草稿步 0 展开为 M=1 / M=2 两条整数账本；KV 折算按参考步 batch。"""
    r = run(_tiny_scenario())
    m1 = _stage_ops(r, "draft_step_0_m1")
    m2 = _stage_ops(r, "draft_step_0_m2")
    assert sum(o.flops for o in m1) == G.DRAFT_M1_FLOPS + 0  # 无其他 FLOPs 行
    assert sum(o.weight_read for o in m1) == G.DRAFT_M1_WEIGHT
    assert sum(o.kv_read for o in m1) == G.DRAFT_KV_READ_M1
    assert sum(o.flops for o in m2) == G.DRAFT_M2_FLOPS
    assert sum(o.weight_read for o in m2) == G.DRAFT_M1_WEIGHT  # 权重读不随 M 放大（摊薄）
    assert sum(o.kv_read for o in m2) == G.DRAFT_KV_READ_M2


def test_comm_bytes_exact():
    r = run(_tiny_scenario())
    assert r.metrics["bytes_d2t"] == G.BYTES_D2T
    assert r.metrics["bytes_t2d"] == G.BYTES_T2D


def test_roofline_spot_value():
    """q_proj 单行 roofline：与手算逐位一致（重排相对误差 ≤1e-12）。"""
    r = run(_tiny_scenario())
    q = next(o for o in _stage_ops(r, "baseline_decode") if o.op == "q_proj")
    assert q.t_ms == pytest.approx(G.QPROJ_T, rel=1e-12)
    assert q.bound == "memory"


def test_replay_determinism():
    """同一场景文件 → 同一 digest、同一关键数字（可重放性）。nan 归一化后比较。"""
    import math

    def _norm(o):
        if isinstance(o, float) and math.isnan(o):
            return "<nan>"
        if isinstance(o, dict):
            return {k: _norm(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_norm(v) for v in o]
        return o

    a, b = run(_tiny_scenario()), run(_tiny_scenario())
    assert a.scenario_digest == b.scenario_digest
    assert _norm(a.to_json()["metrics"]) == _norm(b.to_json()["metrics"])


def test_unsupported_method_is_explicit():
    sc = ScenarioSpec(name="bad", model_id="tiny_dense_oracle", draft_id="dflash2",
                      k=4, deployment={"mode": "colocated"})
    r = run(sc)
    assert r.status == "unsupported_method"
    assert "dflash2" in r.status_detail


def test_mtp_k_over_limit():
    sc = ScenarioSpec(name="mtp-k9", model_id="deepseek_v41_flash", draft_id="mtp",
                      k=9, deployment={"mode": "colocated"})
    r = run(sc)
    assert r.status == "unsupported_combo"
