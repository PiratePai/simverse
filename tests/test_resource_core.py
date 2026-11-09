"""M3 资源事件核测试：调度合法性、流水/串行分列、周期≠请求延迟。

人工手排轨迹（V07/V08 口径）：draft=1, d2t=0.1, verify=2, t2d=0.1（ms）。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simverse.execute.reduce import busy_integral  # noqa: E402
from simverse.execute.resource_core import (dt_pipeline_schedule,  # noqa: E402
                                            schedule_earliest,
                                            single_request_lookahead)
from simverse.execute.resource_core import TaskSpec  # noqa: E402


def test_schedule_respects_dependencies_and_resources():
    """A(0-1,g0) → B(1-2,g0)串行互斥；C 与 A 并行在 g1。"""
    tasks = [
        TaskSpec("a", "g0", 1.0),
        TaskSpec("b", "g0", 1.0, depends_on=["a"]),
        TaskSpec("c", "g1", 0.5, depends_on=["a"]),
    ]
    res = schedule_earliest(tasks)
    assert res.violations == []
    by = {e.event_id: e for e in res.events}
    assert by["b"].start_ms == 1.0 and by["c"].start_ms == 1.0
    assert res.makespan_ms == 2.0


def test_pipeline_overlaps_draft_and_verify():
    """两微批流水 vs 串行：按"每轮等价吞吐"比较（总轮数不同，不比 makespan）。

    手排：串行每轮 1+0.1+2+0.1=3.2ms；流水 8 轮等价（4 轮×2 微批）实测
    makespan 18.6 → 2.325ms/轮等价 < 3.2（重叠收益来自具体调度，非 max 式）。
    """
    D, L, V = 1.0, 0.1, 2.0
    rounds = 4
    pipe = dt_pipeline_schedule(D, L, V, L, rounds=rounds, microbatches=2)
    assert pipe.violations == []
    assert pipe.resource_busy_ms["draft-gpu"] > 0
    assert pipe.resource_busy_ms["target-gpu"] > 0
    serial = single_request_lookahead(D, L, V, L, rounds=rounds * 2)  # 同总轮数
    assert serial.violations == []
    assert serial.makespan_ms == 8 * (D + L + V + L)          # 串行=轮和
    per_round_pipe = pipe.makespan_ms / (rounds * 2)
    per_round_serial = D + L + V + L
    assert per_round_pipe < per_round_serial                   # 流水收益为真
    # 周期与请求延迟分列：微批周期 ≥ 反馈链下界（不套 max 通则）
    for p in pipe.cohort_periods_ms.values():
        assert p >= per_round_serial - 1e-9


def test_link_half_duplex_no_overlap():
    """半双工链路：d2t/t2d 同资源互斥（不重叠）。"""
    pipe = dt_pipeline_schedule(1.0, 0.5, 1.0, 0.5, rounds=3, microbatches=2)
    assert pipe.violations == []
    link_events = sorted((e for e in pipe.events if e.resource_id == "link"),
                         key=lambda e: e.start_ms)
    for a, b in zip(link_events, link_events[1:]):
        assert b.start_ms >= a.end_ms - 1e-9


def test_cohort_period_vs_request_tpot_distinct():
    """M18：吞吐周期 ≠ 单请求 TPOT（分列不混用）。

    手算：draft=1+d2t0.1 → 阶段和 1.1；verify=2。反馈链每微批每轮
    1.1+2+0.1=3.2（下界）；两微批交错后每微批稳态周期≈max(3.2, …)。
    断言只验证分列与量级关系，不套闭式当通则。
    """
    D, L, V = 1.0, 0.1, 2.0
    pipe = dt_pipeline_schedule(D, L, V, L, rounds=6, microbatches=2)
    periods = pipe.cohort_periods_ms
    assert set(periods) == {"b0", "b1"}
    for p in periods.values():
        assert p >= (D + L + V + L) - 1e-9      # 每微批反馈链下界
    # 单请求口径：同链路串行轮时延 = 3.2（请求 TPOT 分母 E[N] 由上层）
    serial_round = D + L + V + L
    assert periods["b0"] >= serial_round - 1e-9


def test_warmup_drain_visible():
    """有限 cohort：首轮与末轮事件可见（warmup/drain 不裁剪）。"""
    pipe = dt_pipeline_schedule(1.0, 0.1, 2.0, 0.1, rounds=3, microbatches=2)
    ids = {e.event_id for e in pipe.events}
    assert "r0b0.draft" in ids and "r2b1.t2d" in ids
    # busy 积分（独立重算）与核自报一致
    by_res: dict = {}
    for e in pipe.events:
        by_res.setdefault(e.resource_id, []).append((e.start_ms, e.end_ms))
    for r, segs in by_res.items():
        assert busy_integral(segs) == pipe.resource_busy_ms[r]


def test_cycle_dependency_rejected():
    """依赖环 → 显式报错（不静默输出）。"""
    import pytest
    with pytest.raises(ValueError, match="环"):
        schedule_earliest([
            TaskSpec("a", "g0", 1.0, depends_on=["b"]),
            TaskSpec("b", "g0", 1.0, depends_on=["a"]),
        ])


def test_unknown_dependency_rejected():
    import pytest
    with pytest.raises(ValueError, match="不存在"):
        schedule_earliest([TaskSpec("a", "g0", 1.0, depends_on=["ghost"])])
