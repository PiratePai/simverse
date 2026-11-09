"""数据身份：完整 SHA256、规范序列化、大整数无损（V01）。

与 v0.1 的差异（evidence-audit E16）：
  * digest 输出完整 64 hex，不再截断 16 位；显示端自行缩短；
  * canonical_json 保持 Python int 原样（json 标准无精度损失），浮点用
    repr 安全往返；NaN/Infinity 拒绝入 JSON（V01 非有限数拒绝）；
  * js_safe() 为浏览器通道把 |n|>2^53 的整数包成 {"$bigint": "..."}，
    前端按标记还原，双向约定单一。
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

_JS_MAX = (1 << 53) - 1


def canonical_json(obj: Any) -> str:
    """键排序 + 紧凑分隔；拒绝 NaN/Infinity（契约：非有限数不得入结果）。"""
    text = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)
    return text


def sha256_hex(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def digest(obj: Any) -> str:
    """完整 SHA256（显示可截断，存储/比对用全值）。"""
    return "sha256:" + sha256_hex(obj)


def short(value: str, n: int = 12) -> str:
    return value if not value.startswith("sha256:") else value[: n + 7]


def js_safe(obj: Any) -> Any:
    """递归把超出 JS Number 安全域的整数包成 {"$bigint": "<十进制串>"}。"""
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        return obj if -_JS_MAX <= obj <= _JS_MAX else {"$bigint": str(obj)}
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError("结果中的浮点必须有限（NaN/Infinity 禁止）")
        return obj
    if isinstance(obj, dict):
        return {k: js_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [js_safe(v) for v in obj]
    return obj


def js_restore(obj: Any) -> Any:
    """js_safe 的逆变换（测试/导入侧使用）。"""
    if isinstance(obj, dict):
        if set(obj) == {"$bigint"} and isinstance(obj["$bigint"], str):
            return int(obj["$bigint"])
        return {k: js_restore(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [js_restore(v) for v in obj]
    return obj


def finite(value: float, name: str = "value") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} 必须为数值")
    if not math.isfinite(value):
        raise ValueError(f"{name} 必须有限，收到 {value}")
    return value


class RngScheme:
    """随机流身份（§7.3）：算法/版本/主种子/子流派生/语义键。"""

    ALGORITHM = "pcg64-xsl-rr"       # numpy default_rng 位流（版本固定见 VERSION）
    STREAMS = ("workload", "acceptance", "routing", "cost_noise")

    def __init__(self, master_seed: int, semantic_keys: dict | None = None):
        self.master_seed = int(master_seed)
        self.semantic_keys = semantic_keys or {}
        self.derivation = "sha256(master_seed, stream_name, semantic_key) → seed"

    def stream_seed(self, stream: str, semantic_key: str = "") -> int:
        if stream not in self.STREAMS:
            raise ValueError(f"未知随机流 {stream}；可用: {self.STREAMS}")
        basis = canonical_json([self.master_seed, stream, semantic_key])
        return int.from_bytes(hashlib.sha256(basis.encode("utf-8")).digest()[:8], "big")

    def to_json(self) -> dict:
        return {"algorithm": self.ALGORITHM, "master_seed": self.master_seed,
                "stream_derivation": self.derivation,
                "semantic_key_fields": self.semantic_keys}
