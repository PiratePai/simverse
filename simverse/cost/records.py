"""Scoped archived-observation costs, separate from admitted measured profiles.

Only costs change. An explicitly selected set is a conditional implementation /
environment assumption; source gaps remain visible and block deployment ranking.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from .profile_store import ProfileIntegrityError


def _data_root() -> Path:
    """源码检出→仓库根；wheel→包内 _data（cost_records 不在 wheel 清单，
    缺失时按 ProfileIntegrityError 如实报错，不静默降级）。"""
    from ..assets import repo_root
    root = repo_root()
    if root is not None:
        return root
    return Path(__file__).resolve().parent.parent / "_data"


ROOT = _data_root()
SETS = {"ascend-r29-hccl": "ascend_r29_hccl.json",
        "ascend-r28-gemm": "ascend_r28_gemm.json"}


def catalog():
    return [{"id": key, "label": label, "admission": "conditional-only"}
            for key, label in (("ascend-r29-hccl", "R29 双卡通信记录（有条件对照）"),
                               ("ascend-r28-gemm", "R28 独立 GEMM 记录（有条件对照）"))]


def load(sid, root=ROOT):
    try:
        return _load(sid, root)
    except (OSError, json.JSONDecodeError) as exc:
        raise ProfileIntegrityError(f"成本记录文件缺失/损坏：{sid} ({type(exc).__name__})") from exc


def _load(sid, root):
    if sid not in SETS:
        raise ValueError(f"未知成本记录档 {sid}")
    path = root / "assets/cost_records" / SETS[sid]
    raw = path.read_bytes()
    data = json.loads(raw)
    if data["id"] != sid or data["admission"] != "conditional-only":
        raise ProfileIntegrityError("成本记录身份/准入状态不符")
    for item in data["sources"]:
        content = (root / item["path"]).read_bytes()
        if hashlib.sha256(content).hexdigest() != item["sha256"]:
            raise ProfileIntegrityError(f"成本记录原始来源哈希不符：{item['path']}")
    # Pin the derived file too; never trust edits to costs on the strength of raw hashes.
    manifest = json.loads((root / "assets/cost_records/manifest.json").read_text(encoding="utf-8"))
    if hashlib.sha256(raw).hexdigest() != manifest[SETS[sid]]:
        raise ProfileIntegrityError(f"派生成本记录哈希不符：{sid}")
    points = [p for curve in data.get("curves", {}).values() for p in curve] + data.get("operators", [])
    if not points or any(not math.isfinite(p["selected_ms"]) or p["selected_ms"] <= 0 for p in points):
        raise ProfileIntegrityError(f"成本记录含无效时长：{sid}")
    return data, hashlib.sha256(raw).hexdigest()


def curve_lookup(data, op, input_bytes, ranks, dtype, device):
    """Latency by logical input bytes; never invert estimated wire bytes generally."""
    if device != data["domain"]["device"]:
        return None, [], "device-mismatch"
    if dtype != "bfloat16":
        return None, [], "dtype-mismatch"
    if ranks != 2:
        return None, [], "rank-domain"
    points = data.get("curves", {}).get(op, [])
    if not points:
        return None, [], "primitive-unresolved"
    xs = [p["input_bytes"] for p in points]
    if input_bytes < xs[0] or input_bytes > xs[-1]:
        return None, [], "bytes-out-of-range"
    i = bisect.bisect_left(xs, input_bytes)
    if i < len(xs) and xs[i] == input_bytes:
        return points[i]["selected_ms"], [points[i]["id"]], "exact"
    lo, hi = points[i - 1], points[i]
    f = (input_bytes - lo["input_bytes"]) / (hi["input_bytes"] - lo["input_bytes"])
    return lo["selected_ms"] + f * (hi["selected_ms"] - lo["selected_ms"]), [lo["id"], hi["id"]], "interpolated"


REASONS = {
    "device-mismatch": "设备不匹配，保留解析值",
    "dtype-mismatch": "仅有独立 BF16 样本，当前精度不匹配",
    "rank-domain": "通信仅有 world=2，不外推其他 rank 数",
    "bytes-out-of-range": "载荷在记录曲线外，不外推",
    "primitive-unresolved": "实际通信原语/切分与记录不等价，不能按字节碰巧相等替换",
    "operator-set-missing": "所选 R29 档只有通信；不能混入 R28 不同栈的算子成本",
    "communication-set-missing": "所选 R28 档只有独立算子；不能混入 R29 不同栈的通信成本",
    "model-role-mismatch": "不是采集模型的 target 行，草稿/其他模型无匹配实现记录",
    "operator-not-equivalent": "无等价独立 GEMM 记录，组合/代理/融合边界未核验",
    "shape-missing": "算子有记录但当前 M/K/N 无匹配点；不插值、不外推",
    "no-communication": "单卡或无实际通信，不使用通信样本",
    "point-to-point-uncovered": "独立草稿/目标点到点链路无匹配记录，不能复用集合通信曲线",
}


class RecordPricer:
    def __init__(self, scenario, model):
        self.data, self.content_digest = load(scenario.cost_records)
        self.scenario, self.model = scenario, model
        self.entries = []

    def _entry(self, kind, stage, name, shape, analytic, selected=None, refs=(), code="", **kw):
        hit = selected is not None
        item = dict(kind=kind, stage=stage, name=name, shape=shape,
                    analytic_ms=analytic, selected_ms=selected if hit else analytic,
                    source="archived-observation-assumption" if hit else "analytic",
                    lookup=code if hit else None, record_refs=list(refs),
                    fallback_code=None if hit else code,
                    fallback_reason=None if hit else REASONS[code], **kw)
        self.entries.append(item)
        return item

    def rows(self, rows, prices, device, quant):
        out = []
        by_key = {(p["op"], tuple(p["shape"])): p for p in self.data.get("operators", [])}
        for row, price in zip(rows, prices):
            code, selected, refs = "operator-not-equivalent", None, []
            shape = None
            match = re.match(r"^\[(\d+)[,;](\d+)\]@\[(\d+)[,;](\d+)\]", row.shape or "")
            if match:
                m, k, k2, n = map(int, match.groups())
                if k == k2 and row.flops == 2 * m * k * n:
                    shape = (m, k, n)
            elif row.op == "lm_head" and row.role == "target":
                k = self.model.hidden
                n = self.model.vocab // self.scenario.deployment.target_tp
                denom = 2 * k * n
                if row.flops and row.flops % denom == 0:
                    shape = (row.flops // denom, k, n)
            if device.id != "ascend_910b3":
                code = "device-mismatch"
            elif "operators" not in self.data:
                code = "operator-set-missing"
            elif (self.model.id != self.data["domain"]["model_id"] or row.role != "target"
                  or row.stage.startswith("draft")):
                code = "model-role-mismatch"
            elif any(quant.get(k) != 16 for k in ("dense_w", "act")):
                code = "dtype-mismatch"
            elif row.op in {key[0] for key in by_key}:
                record = by_key.get((row.op, shape))
                if record:
                    selected, refs, code = record["selected_ms"], [record["id"]], "exact"
                else:
                    code = "shape-missing"
            item = self._entry("operator", row.stage, row.op, list(shape) if shape else row.shape,
                               price["t_ms"], selected, refs, code,
                               row_id=f"{row.stage}:{row.layer}:{row.op}", device=device.id)
            out.append({**price, "t_ms": item["selected_ms"],
                        "analytic_ms": price["t_ms"], "cost_source": item["source"],
                        "record_refs": item["record_refs"], "fallback_reason": item["fallback_reason"],
                        "cost_basis": price["cost_basis"] + ("；本行选用独立算子记录作条件对照（含host/sync，非kernel实测校准）"
                                                               if item["record_refs"] else "")})
        return out

    def collective(self, c, ranks, dtype, device, stage):
        analytic, selected, refs, code = c.t_ms, None, [], "primitive-unresolved"
        # world=2: allreduce wire coefficient is exactly 1, and explicit
        # allgather recv=(P-1)*input. Other worlds fail before conversion.
        op = "all_reduce" if c.kind == "allreduce" else (
            "all_gather" if c.kind == "mtp_fc_allgather" else None)
        if not c.bytes_on_wire:
            code = "no-communication"
        elif device.id != "ascend_910b3":
            code = "device-mismatch"
        elif "curves" not in self.data:
            code = "communication-set-missing"
        elif ranks != 2:
            code = "rank-domain"
        elif op:
            selected, refs, code = curve_lookup(self.data, op, c.bytes_on_wire, ranks, dtype, device.id)
        item = self._entry("communication", stage, c.kind, {"ranks": ranks, "input_bytes":
                            c.bytes_on_wire if ranks == 2 and op else None,
                            "analytic_wire_bytes": c.bytes_on_wire},
                           analytic, selected, refs, code, device=device.id, note=c.note)
        c.t_ms = item["selected_ms"]  # replace entire cost; do not add latency again
        return c

    def point_to_point(self, direction, link, payload_bytes, analytic_ms):
        self._entry("link", direction, direction,
                    {"link": link, "bytes": payload_bytes}, analytic_ms,
                    code="point-to-point-uncovered", device="target/draft")

    def report(self):
        entries = self.entries
        by_stage = {}
        for stage in sorted({e["stage"] for e in entries}):
            es = [e for e in entries if e["stage"] == stage]
            hits = [e for e in es if e["record_refs"]]
            selected = sum(e["selected_ms"] for e in es)
            by_stage[stage] = {
                "matched": len(hits), "total": len(es),
                "record_ms": sum(e["selected_ms"] for e in hits),
                "fallback_ms": sum(e["selected_ms"] for e in es if not e["record_refs"]),
                "selected_ms": selected,
                "time_fraction_basis": "成本评估实例的记录值/选用成本；非E2E实测时间覆盖",
                "record_cost_fraction": sum(e["selected_ms"] for e in hits) / selected if selected else 0,
            }
        grouped = defaultdict(list)
        for e in entries:
            if e["fallback_code"] and e["analytic_ms"] > 0:
                grouped[e["fallback_code"]].append(e)
        followups = []
        actions = {
            "operator-set-missing": "在R29统一栈补采当前模型真实算子；R28独立GEMM可先单独对照，不跨栈拼接",
            "communication-set-missing": "先单独使用R29通信档；若需同次组合，在统一栈采集匹配算子与通信",
            "rank-domain": "先验收已有双卡；TP4/8需更多卡或匹配trace，不用双卡带宽外推",
            "shape-missing": "按下列M/K/N补采BF16独立GEMM，再核实际后端融合/布局；已有点不重采",
            "operator-not-equivalent": "采实际后端的GDN/attention/MoE/融合组，逐组登记账本所有权，禁止以代理替代",
            "model-role-mismatch": "为草稿模型单独绑定实现/形状/状态；目标模型样本不跨模型复用",
            "primitive-unresolved": "先固定Ascend实际gather/allgather或EP variable-split实现，再采对应通信，不按同字节硬配",
            "bytes-out-of-range": "只补当前场景缺失的载荷点，保留两rank联合批ID/最慢窗口，禁止外推",
            "device-mismatch": "切换匹配设备，或另采当前设备，禁止跨SKU/跨后端复用",
            "dtype-mismatch": "先完成当前dtype执行配方与数值校验，再独立采样；不按位宽倍率换算",
            "point-to-point-uncovered": "独立采target/draft的token/hidden/KV点到点往返、方向和同步边界，不拿HCCL collective替代",
        }
        for code, es in grouped.items():
            unique = {}
            for e in sorted(es, key=lambda x: -x["analytic_ms"]):
                key = json.dumps([e["name"], e["shape"]], sort_keys=True, ensure_ascii=False)
                case = unique.setdefault(key, {"name": e["name"], "shape": e["shape"],
                                               "stages": [], "instances": 0, "fallback_analytic_ms": 0.0})
                case["instances"] += 1
                case["fallback_analytic_ms"] += e["analytic_ms"]
                if e["stage"] not in case["stages"]:
                    case["stages"].append(e["stage"])
            cases = sorted(unique.values(), key=lambda x: -x["fallback_analytic_ms"])
            followups.append({"reason_code": code, "reason": REASONS[code],
                              "instances": len(es), "fallback_analytic_ms": sum(e["analytic_ms"] for e in es),
                              "action": actions.get(code, "按对应边界补齐独立证据"),
                              "unique_cases": len(unique), "examples": cases[:8], "cases": cases})
        followups.sort(key=lambda x: -x["fallback_analytic_ms"])
        return {"set_id": self.data["id"], "label": self.data["label"],
                "content_digest": self.content_digest, "source_kind": self.data["source_kind"],
                "admission": self.data["admission"], "domain": self.data["domain"],
                "limits": self.data["limits"], "sources": self.data["sources"],
                "available_curves": [{"op": op, "points": len(points),
                                      "input_bytes": [points[0]["input_bytes"], points[-1]["input_bytes"]]}
                                     for op, points in self.data.get("curves", {}).items()],
                "available_operator_shapes": [{"op": p["op"], "shape": p["shape"]}
                                              for p in self.data.get("operators", [])],
                "curve_diagnostics": self.data.get("curve_diagnostics"),
                "excluded_records": self.data.get("excluded", []),
                "matched": sum(bool(e["record_refs"]) for e in entries), "total": len(entries),
                "stages": by_stage, "entries": entries, "measurement_plan": followups,
                "prerequisites": ["P0：补CANN/runtime commit/dirty补丁/实际拓扑/每run唯一ID与计时原样窗口",
                                  "P1：同栈至少三次独立进程，按shape分组留出；通信联合批窗口对齐；不重复收费host/launch"],
                "deployment_ranking_eligible": False}
