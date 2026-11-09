"""接受率模型（方案 §4.1 三字段契约 + §4.5 增量② Catch-up）。

E[n]：链式 α → Σαⁱ（含 bonus）；τ 直给。
P_full = α^K：上轮全接受概率 → 下一轮草稿首步须喂 2 token（Catch-up）。
事件采样（hist / 截断几何）属 M3 事件核，本切片仅解析期望。
"""
from __future__ import annotations

from .contracts import AcceptanceSpec


def resolve_alpha(acc: AcceptanceSpec, k: int) -> float:
    """链式逐位接受率 α；committed_mean 输入按截断几何反解（标注假设）。"""
    if acc.kind == "chain_alpha":
        return float(acc.chain_alpha)
    tau = float(acc.committed_mean)
    # 二分反解 (1-α^{k+1})/(1-α) = τ；τ=k+1 → α=1
    lo, hi = 0.0, 0.999999
    for _ in range(80):
        mid = (lo + hi) / 2
        if (1 - mid ** (k + 1)) / (1 - mid) < tau:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def expected_committed_length(acc: AcceptanceSpec, k: int) -> float:
    """E[n] ∈ [1, k+1]：一轮验证实际写回的 token 数（接受草稿数 + 1 bonus）。"""
    if acc.kind == "committed_mean":
        return min(float(acc.committed_mean), k + 1)
    a = float(acc.chain_alpha)
    if a >= 1.0:
        return float(k + 1)
    return (1 - a ** (k + 1)) / (1 - a)


def catchup_prob_full(alpha: float, k: int) -> float:
    """P_full = α^K：上轮 K 个草稿全接受（提交 K+1）的概率。"""
    return alpha ** k


def expected_draft_step0_tokens(alpha: float, k: int) -> float:
    """E[M_draft,0] = 1 + α^K（RoundTransitionSpec；部分接受喂 1，全接受喂 2）。"""
    return 1.0 + catchup_prob_full(alpha, k)


def breakeven_alpha(speedup_fn, lo: float = 0.0, hi: float = 0.9999) -> float:
    """反解 speedup(α*) = 1（二分；假设 speedup 随 α 单调不减）。"""
    if speedup_fn(hi) <= 1.0:
        return float("nan")   # 全接受率区间都不盈利
    if speedup_fn(lo) >= 1.0:
        return lo             # α=0（退化普通 decode）即已不亏 —— 实际不会发生，防御
    for _ in range(60):
        mid = (lo + hi) / 2
        if speedup_fn(mid) < 1.0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
