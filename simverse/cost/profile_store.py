"""蒸馏 profile 集的加载与查表（MERGE_CRITERIA §3）。

来源：tools/distill_h200_profiles.py 产出的 assets/profiles/<set>/；
加载时重验 manifest 内产物哈希（G3），不匹配即拒绝。
查表策略（最简可满足数据）：
  * 线性算子：全量训练网格上按 log10(tokens) 括号分段线性插值；域外 None。
  * decode 注意力块：per-op (batch 取 log 括号, kv 线性括号) 双线性插值后求和；
    域外 None。留出划分仅用于误差评估（manifest.holdout），生产查表用全量网格。
本模块只读；不修改任何账本。
"""
from __future__ import annotations

import bisect
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

PROFILES_ROOT = Path(__file__).resolve().parent.parent.parent / "assets" / "profiles"

PROFILE_SETS = {
    "h200-qwen3-30b-a3b-tiny-op-v1": "h200_qwen3_30b_a3b_tiny",
    "h200-qwen3-235b-a22b-op-v1": "h200_qwen3_235b_a22b",
}

# 历史校准资产绑定（2026-11-10 审查）：profile 只对其采集模型有效，跨模型
# 直接拒绝，不靠“形状碰巧不一致”兜底。两款活跃主模型目前没有匹配 profile。
PROFILE_MODEL_BINDINGS = {
    "h200-qwen3-30b-a3b-tiny-op-v1": "qwen3_30b_a3b",
    "h200-qwen3-235b-a22b-op-v1": "qwen3_235b_a22b",
}
PROFILE_SET_STATUS = {
    "h200-qwen3-30b-a3b-tiny-op-v1": "archived",
    "h200-qwen3-235b-a22b-op-v1": "archived",
}
PRODUCT_PROFILE_SETS = frozenset(
    sid for sid, status in PROFILE_SET_STATUS.items() if status == "active")


def profile_model_id(set_id: str) -> str:
    if set_id not in PROFILE_MODEL_BINDINGS:
        raise ValueError(f"profile {set_id} 未登记模型绑定；禁止在无绑定情况下使用")
    return PROFILE_MODEL_BINDINGS[set_id]


def archived_profiles_json() -> list:
    """只读归档清单（UI/API 不可选择；用于明确旧资产边界）。"""
    return [{
        "id": sid, "model_id": PROFILE_MODEL_BINDINGS[sid],
        "status": "archived",
        "reason": "历史模型校准资产；模型已移出活跃 catalog；禁止跨模型使用",
    } for sid in sorted(PROFILE_SETS)]

DECODE_BLOCK_OPS = ("attn_input_reshape", "attn_kv_cache_save", "attn_decode",
                    "attn_output_reshape")


class ProfileIntegrityError(ValueError):
    """产物哈希与 manifest 不符 / 结构缺失 → 拒绝加载（不降级静默）。"""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class ProfileSet:
    set_id: str
    manifest: dict
    linear_ms: Dict[str, Dict[Tuple[int, int], float]]   # op → (tp,tokens) → ms
    linear_kc: Dict[str, Dict[Tuple[int, int], int]]     # op → (tp,tokens) → kc
    attn_ms: Dict[Tuple[int, int, int], Dict[str, float]]  # (tp,batch,kv) → op → ms
    attn_kc: Dict[Tuple[int, int, int], Dict[str, int]]
    lin_axes: Dict[Tuple[str, int], list] = field(default_factory=dict,
                                                  repr=False)

    # ---- 域信息透传 ----
    @property
    def device(self) -> str:
        return self.manifest["domain"]["device"]

    @property
    def precision(self) -> str:
        return self.manifest["domain"]["precision"]

    @property
    def tensor_parallel(self):
        """TP 域：int（单档）或 list（多档集合，如 235B 的 [2,8]）。"""
        return self.manifest["domain"]["tensor_parallel"]

    def tp_allowed(self, tp: int) -> bool:
        dom = self.tensor_parallel
        return tp in dom if isinstance(dom, list) else tp == dom

    @property
    def shape_identity(self) -> dict:
        return self.manifest["domain"]["shape_identity"]

    def eager_gap_us(self, tp: int = 1) -> float:
        for g in self.manifest["kernel_gap"]:
            if g["execution_mode"] == "eager" and g.get("tensor_parallel", 1) == tp:
                return g["kernel_gap_us"]
        return 0.0

    @property
    def graph_gap_us(self) -> float:
        for g in self.manifest["kernel_gap"]:
            if g["execution_mode"] == "cuda_graph":
                return g["kernel_gap_us"]
        return 0.0

    # ---- 查表 ----
    def lookup_linear(self, op: str, tokens: int, tp: int = 1) -> Optional[float]:
        """log10(tokens) 括号插值（同 tp 训练点）；域外/无表 None。"""
        tab = {t: v for (t2, t), v in self.linear_ms.get(op, {}).items()
               if t2 == tp}
        if not tab:
            return None
        xs = self.lin_axes.get((op, tp))
        if xs is None:
            xs = sorted(tab)
            self.lin_axes[(op, tp)] = xs
        if tokens < xs[0] or tokens > xs[-1]:
            return None
        if tokens in tab:
            return tab[tokens]
        i = bisect.bisect_left(xs, tokens)
        x0, x1 = xs[i - 1], xs[i]
        lx = math.log10(tokens)
        lx0, lx1 = math.log10(x0), math.log10(x1)
        w = (lx - lx0) / (lx1 - lx0) if lx1 != lx0 else 0.0
        return tab[x0] * (1 - w) + tab[x1] * w

    def lookup_linear_kc(self, op: str, tokens: int, tp: int = 1) -> int:
        """kernel_count 为阶梯量：取 ≤tokens 的最近网格点（域内）。"""
        tab = {t: v for (t2, t), v in self.linear_kc.get(op, {}).items()
               if t2 == tp}
        if not tab:
            return 0
        xs = sorted(tab)
        if tokens <= xs[0]:
            return tab[xs[0]]
        if tokens >= xs[-1]:
            return tab[xs[-1]]
        i = bisect.bisect_left(xs, tokens)
        return tab[xs[i - 1]] if xs[i] != tokens else tab[tokens]

    def _attn_interp_op(self, op: str, batch: int, kv: int,
                        tp: int = 1) -> Optional[float]:
        pts: Dict[int, Dict[int, float]] = {}
        for (t2, b, k), ops in self.attn_ms.items():
            if t2 != tp:
                continue
            v = ops.get(op)
            if v is not None:
                pts.setdefault(b, {})[k] = v
        bs = sorted(pts)
        if not bs or batch < bs[0] or batch > bs[-1]:
            return None

        def kv_interp(b: int) -> Optional[float]:
            row = pts[b]
            ks = sorted(row)
            if kv < ks[0] or kv > ks[-1]:
                return None
            if kv in row:
                return row[kv]
            i = bisect.bisect_left(ks, kv)
            k0, k1 = ks[i - 1], ks[i]
            w = (kv - k0) / (k1 - k0) if k1 != k0 else 0.0
            return row[k0] * (1 - w) + row[k1] * w

        if batch in pts:
            return kv_interp(batch)
        i = bisect.bisect_left(bs, batch)
        b0, b1 = bs[i - 1], bs[i]
        v0, v1 = kv_interp(b0), kv_interp(b1)
        if v0 is None or v1 is None:
            return None
        lb, lb0, lb1 = math.log10(batch), math.log10(b0), math.log10(b1)
        w = (lb - lb0) / (lb1 - lb0) if lb1 != lb0 else 0.0
        return v0 * (1 - w) + v1 * w

    def lookup_attn_block(self, batch: int, kv: int,
                          tp: int = 1) -> Optional[float]:
        """decode 注意力块 = 4 算子插值之和（域外 None）。"""
        total = 0.0
        for op in DECODE_BLOCK_OPS:
            v = self._attn_interp_op(op, batch, kv, tp)
            if v is None:
                return None
            total += v
        return total

    def lookup_attn_block_kc(self, batch: int, kv: int, tp: int = 1) -> int:
        """最近网格点的块 kernel 数（阶梯量，同 tp）。"""
        best = None
        for (t2, b, k), ops in self.attn_kc.items():
            if t2 != tp:
                continue
            d = (abs(b - batch), abs(k - kv))
            if best is None or d < best[0]:
                best = (d, sum(ops.get(op, 0) for op in DECODE_BLOCK_OPS))
        return best[1] if best else 0


def load_profile_set(set_id: str) -> ProfileSet:
    if set_id not in PROFILE_SETS:
        raise ValueError(f"未知 profile 集 {set_id}；可用: {sorted(PROFILE_SETS)}")
    d = PROFILES_ROOT / PROFILE_SETS[set_id]
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("profile_set_id") != set_id:
        raise ProfileIntegrityError(f"manifest set_id 不匹配: {manifest.get('profile_set_id')}")
    for name, sha in manifest.get("distill_files", {}).items():
        p = d / name
        if not p.exists():
            raise ProfileIntegrityError(f"缺产物 {name}")
        if _sha(p) != sha:
            raise ProfileIntegrityError(f"产物哈希不符 {name}（G3 拒绝加载）")
    lin = json.loads((d / "linear_ops.json").read_text(encoding="utf-8"))
    lkc = json.loads((d / "linear_kernel_counts.json").read_text(encoding="utf-8"))
    att = json.loads((d / "attn_decode.json").read_text(encoding="utf-8"))
    tp_keyed = bool(lin.get("tp_keyed"))

    def _lin_key(k: str) -> Tuple[int, int]:
        if tp_keyed:
            tp, t = k.split(":")
            return (int(tp), int(t))
        return (1, int(k))

    linear_ms = {op: {_lin_key(k): v for k, v in tab.items()}
                 for op, tab in lin["grid"].items()}
    linear_kc = {op: {_lin_key(k): v for k, v in tab.items()}
                 for op, tab in lkc["grid"].items()}
    attn_ms, attn_kc = {}, {}
    for p in att["points"]:
        tp = int(p.get("tp", 1))
        key = (tp, int(p["batch"]), int(p["kv"]))
        attn_ms[key] = {op: v for op, v in p["median_ms"].items()}
        attn_kc[key] = {op: v for op, v in p.get("kernel_count", {}).items()}
    return ProfileSet(set_id=set_id, manifest=manifest, linear_ms=linear_ms,
                      linear_kc=linear_kc, attn_ms=attn_ms, attn_kc=attn_kc)
