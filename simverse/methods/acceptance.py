"""接受模型 v0.2：逐位条件接受率为主输入；均值/动态 K 显式假设或未知。

修复的 v0.1 缺陷：
  * E07：committed_mean 不再静默反解几何 α —— 需显式登记
    assumption="truncated-geometric" 才给 approximated 数值，否则 unknown；
  * Z2-01：动态 K 缺联合/条件律 → unknown + reason，不合成边际；
  * 端点：α=1 → K+1；α=0 → 1（v0.1 二分在 α→1 处数值化 0.999999 的补丁移除）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class AcceptanceLaw:
    """接受分布的显式身份。

    kinds:
      position_alphas   逐位条件接受率 β_i（链式/块内逐位同式 F_EN_CHAIN/F_EN_BLOCK）
      geometric         β_i = α 全同（position_alphas 特例，输入便捷）
      committed_mean    仅一阶矩（需 assumption 才可数值化）
      dynamic_k         K 随状态变化：需联合/条件律（law 字段），否则 unknown
      empirical_hist    A/K 直方图（后续 trace 扩展；首版 unsupported）
    """
    kind: str
    position_alphas: Optional[Tuple[float, ...]] = None
    k_limit: Optional[int] = None                    # 链式 K 上限
    mean: Optional[float] = None                     # committed_mean 输入
    assumption: Optional[str] = None                 # 显式假设（truncated-geometric…）
    joint_law: Optional[dict] = None                 # 动态 K 的联合/条件律描述
    source: str = "hypothesis"                       # Q8：首版显式假设扫描
    source_ref: str = ""                             # trace/数据集引用（后续）


def law_from_scenario(acc: dict, k: int) -> AcceptanceLaw:
    kind = acc.get("kind", "geometric")
    if kind == "geometric":
        alpha = float(acc["alpha"])
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha ∈ [0,1]")
        return AcceptanceLaw(kind="position_alphas",
                              position_alphas=(alpha,) * k, k_limit=k,
                              source=acc.get("source", "hypothesis"))
    if kind == "position_alphas":
        alphas = tuple(float(a) for a in acc["alphas"])
        if any(not 0.0 <= a <= 1.0 for a in alphas):
            raise ValueError("alphas ∈ [0,1]")
        return AcceptanceLaw(kind="position_alphas", position_alphas=alphas,
                              k_limit=len(alphas),
                              source=acc.get("source", "hypothesis"))
    if kind == "committed_mean":
        return AcceptanceLaw(kind="committed_mean", mean=float(acc["mean"]),
                              k_limit=k, assumption=acc.get("assumption"),
                              source=acc.get("source", "hypothesis"))
    if kind == "dynamic_k":
        return AcceptanceLaw(kind="dynamic_k", k_limit=k,
                             joint_law=acc.get("joint_law"),
                             source=acc.get("source", "hypothesis"))
    if kind == "empirical_hist":
        return AcceptanceLaw(kind="empirical_hist", k_limit=k,
                              source_ref=acc.get("source_ref", ""))
    raise ValueError(f"未知接受输入 kind={kind}")


def expected_committed(law: AcceptanceLaw) -> Tuple[Optional[float], Optional[str]]:
    """E[N]（含 bonus）。返回 (value, reason)；value=None 时 reason 必填。"""
    if law.kind == "position_alphas":
        e, prefix = 1.0, 1.0
        for a in law.position_alphas:      # E[N] = 1 + Σ Π_{j≤i} β_j
            prefix *= a
            e += prefix
        return e, None
    if law.kind == "committed_mean":
        if law.assumption == "truncated-geometric":
            tau = min(float(law.mean), (law.k_limit or 1) + 1)
            return float(tau), ("approximated: mean 按『截断几何』假设数值化"
                                "（用户显式选择；同均值其他分布不成立）")
        return None, ("mean-only：均值不唯一决定分布（反例 N≡2 vs N∈{1,3} 各半，"
                      "P_full 0 vs 0.5）；登记 assumption=truncated-geometric 后给近似")
    if law.kind == "dynamic_k":
        if law.joint_law:
            return None, ("动态 K 需联合/条件律的数值模型：当前仅登记描述，"
                          "数值生成在 methods 联合律引擎接入后开放")
        return None, ("动态 K 缺联合/条件律：边际 α_i 与 K 分布不足以确定 E[N]"
                      "（同边际不同联合 → E[N] 2.06 vs 1.98 反例）")
    if law.kind == "empirical_hist":
        return None, "empirical_hist 属 trace 扩展，首版 unsupported"
    return None, f"未知 kind {law.kind}"


def p_full(law: AcceptanceLaw) -> Optional[float]:
    """P(A=K)（全接受）。mean-only 无法确定（E07）。"""
    if law.kind == "position_alphas":
        p = 1.0
        for a in law.position_alphas:
            p *= a
        return p
    return None


def position_prefix_probs(law: AcceptanceLaw) -> List[float]:
    """P(A≥i) for i=1..K。"""
    out, p = [], 1.0
    for a in (law.position_alphas or ()):
        p *= a
        out.append(p)
    return out


def alpha_effective(law: AcceptanceLaw) -> Optional[float]:
    """几何特例的 α（显示用）；非几何返回 None，不折算。"""
    if law.kind == "position_alphas" and law.position_alphas:
        a0 = law.position_alphas[0]
        if all(a == a0 for a in law.position_alphas):
            return a0
    return None
