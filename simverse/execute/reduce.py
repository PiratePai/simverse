"""指标归约器（M3 / V08）：从事件/token 轨迹独立重算系统指标。

与调度器分离的纯函数：输入 token emission 轨迹（time_basis 分列）与资源
busy 段，输出 TPOT（请求口径）/吞吐（窗口口径）/goodput（SLO）/busy 积分/
分位（最近秩）。独立 oracle 见 golden/oracles_v02.py §5（人工轨迹手算）。
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def request_mean_itl(emitted_ms: Sequence[float]) -> Optional[float]:
    """M01 请求口径平均 token 间隔 = (last−first)/(N−1)；N≤1 不适用（None）。"""
    n = len(emitted_ms)
    if n <= 1:
        return None
    return (emitted_ms[-1] - emitted_ms[0]) / (n - 1)


def window_token_count(emissions: Dict[str, Sequence[float]],
                       start_ms: float, end_ms: float) -> int:
    """窗口 [start,end) 内 token 事件数（§7.6：与“窗口内完成请求”不同名）。"""
    return sum(1 for ts in emissions.values() for t in ts if start_ms <= t < end_ms)


def window_completed_requests(emissions: Dict[str, Sequence[float]],
                              completed: Iterable[str],
                              start_ms: float, end_ms: float) -> int:
    """窗口内完成的请求数（完成=最后一个 emission 在窗口内且请求已完结）。"""
    n = 0
    for rid in completed:
        ts = emissions.get(rid) or ()
        if ts and start_ms <= ts[-1] < end_ms:
            n += 1
    return n


def throughput_tokens_per_ms(emissions: Dict[str, Sequence[float]],
                             completed: Iterable[str],
                             start_ms: float, end_ms: float,
                             count_basis: str = "completed_requests") -> Optional[float]:
    """吞吐：completed_requests 口径=窗口内完成请求的 token/墙钟。

    count_basis=token_events 口径=窗口内全部 token 事件/墙钟（另一命名，不混用）。
    """
    wall = end_ms - start_ms
    if wall <= 0:
        return None
    if count_basis == "token_events":
        return window_token_count(emissions, start_ms, end_ms) / wall
    toks = sum(len(emissions.get(rid) or ()) for rid in completed
               if (emissions.get(rid) or () and start_ms <= emissions[rid][-1] < end_ms))
    return toks / wall


def goodput_tokens_per_ms(emissions: Dict[str, Sequence[float]],
                          completed: Iterable[str],
                          start_ms: float, end_ms: float,
                          slo_itl_max_ms: float) -> Optional[float]:
    """goodput：满足逐请求平均间隔 ≤ SLO 的完成请求 token 数/墙钟。

    取消/未完成请求不计入（单独列示）。窗口口径与吞吐同声明。
    """
    wall = end_ms - start_ms
    if wall <= 0:
        return None
    toks = 0
    for rid in completed:
        ts = emissions.get(rid) or ()
        if not (ts and start_ms <= ts[-1] < end_ms):
            continue
        itl = request_mean_itl(ts)
        # N≤1 的请求平均间隔不适用：单 token 请求无间隔约束，视为满足
        if itl is None or itl <= slo_itl_max_ms:
            toks += len(ts)
    return toks / wall


def busy_integral(segments: Sequence[Tuple[float, float]]) -> float:
    """资源 busy 时间积分：区间并集（同资源重叠不双计，V08 反例）。"""
    if not segments:
        return 0.0
    segs = sorted(segments)
    total = 0.0
    cur_s, cur_e = segs[0]
    for s, e in segs[1:]:
        if s > cur_e:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    total += cur_e - cur_s
    return total


def quantile_nearest_rank(samples: Sequence[float], p: float) -> Optional[float]:
    """最近秩分位（estimator 显式；样本不足由调用方标注）。"""
    if not samples or not 0 < p <= 1:
        return None
    xs = sorted(samples)
    import math
    idx = max(0, math.ceil(p * len(xs)) - 1)
    return xs[idx]


def summarize_run(emissions: Dict[str, Sequence[float]],
                  completed: Iterable[str],
                  cancelled: Iterable[str],
                  window_ms: Tuple[float, float],
                  slo_itl_max_ms: float = 15.0) -> Dict[str, object]:
    """汇总（含取消单列与样本数）。"""
    start, end = window_ms
    completed = list(completed)
    cancelled = list(cancelled)
    itls = [request_mean_itl(emissions[r]) for r in completed
            if r in emissions and len(emissions[r]) > 1]
    return {
        "window": {"start_ms": start, "end_ms": end, "basis": "[start,end)"},
        "n_completed": len(completed),
        "n_cancelled": len(cancelled),
        "cancelled_ids": cancelled,
        "token_events_in_window": window_token_count(emissions, start, end),
        "throughput_completed_tok_per_ms":
            throughput_tokens_per_ms(emissions, completed, start, end,
                                     "completed_requests"),
        "goodput_tok_per_ms": goodput_tokens_per_ms(
            emissions, completed, start, end, slo_itl_max_ms),
        "per_request_mean_itl_ms": {r: request_mean_itl(emissions[r])
                                    for r in completed if r in emissions},
        "itl_p50": quantile_nearest_rank([v for v in itls if v is not None], 0.5),
        "itl_p95": quantile_nearest_rank([v for v in itls if v is not None], 0.95),
        "itl_sample_count": len([v for v in itls if v is not None]),
    }
