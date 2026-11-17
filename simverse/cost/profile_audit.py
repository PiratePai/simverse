"""Ascend profiling 审计包只读加载（A-N5 离线阶段，未准入）。

本模块消费 `tools/import_ascend_profiles.py` 生成在 `assets/profile_audit/<pack>/`
的审计包（identity/census/mapping/diff/shape_costs + manifest 哈希清单）。

铁律（MERGE_CRITERIA §3 / AGENTS.md）
------------------------------------
* **只读、不计价**：不 import `pipeline`/`cost.calibrated`，不产生 `selected_ms`，
  不修改任何 Work/Traffic/Memory 账本；`calibration_admission` 恒为
  `not-admitted`——审计包永不进入 `PROFILE_SETS`/`CalibratedPricer`。
* **完整性**：加载时对 manifest.distill_files 逐文件重算 SHA256，不符即
  `AuditIntegrityError`（不降级、不静默）。
* **域门禁显式化**：设备家族/SKU/实现/TP/模型绑定逐项检查；不匹配时给出
  `fallback_reason` 并保持 `selected_ms=None`，绝不填 0。
* **不拼接**：不同 run（不同 draft 方法/日期/栈）只并列展示，不聚合成一个
  “profile”；跨模型查询直接拒绝。

对外 API
--------
  load_audit_pack(set_id) -> AuditPack
  audit_packs_json() -> list[dict]          # 工作台/API 索引
  audit_report_json(set_id, model_id=None, device_id=None, tp=None) -> dict
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent.parent
AUDIT_ROOT = ROOT / "assets" / "profile_audit"

SCHEMA = "ascend-profile-audit/1"

# 审计包白名单（显式；文件系统不自动发现）。未登记的 set_id 一律拒绝。
AUDIT_PACKS = {
    "ascend-910b-profile-audit-v1": "ascend_910b_profile_audit_v1",
}
# 与 validation.registry 的证据登记一致（executed=False：本仓未执行 NPU 采集）
AUDIT_EVIDENCE = {
    "ascend-910b-profile-audit-v1": "ev-ascend-profile-audit-v1",
}
# 审计包当前一律未准入；此常量存在是为了让"准入"必须改代码+测试，不能靠数据字段漂移
NOT_ADMITTED = "not-admitted"


class AuditIntegrityError(ValueError):
    """产物哈希与 manifest 不符 / 结构缺失 / 被错误标记为已准入 → 拒绝加载。"""


class AuditAdmissionBlocked(ValueError):
    """调用方试图把审计数据当校准/测量使用 → 显式拒绝并给原因。"""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditIntegrityError(f"审计包产物不可读 {path.name}: {exc}") from exc


@dataclass(frozen=True)
class AuditPack:
    set_id: str
    manifest: dict
    identity: dict
    census: dict
    mapping: dict
    diff: dict
    shape_costs: dict
    ledger_rows: dict

    # ---- 透传的身份/边界 ----
    @property
    def admission(self) -> str:
        return self.manifest.get("calibration_admission", "unknown")

    @property
    def admitted(self) -> bool:
        return bool(self.manifest.get("admitted"))

    @property
    def device_label(self) -> str:
        return self.manifest.get("device_family", "unknown")

    @property
    def candidate_models(self) -> List[str]:
        return list(self.manifest.get("candidate_model_ids") or [])

    @property
    def limits(self) -> List[str]:
        return list(self.manifest.get("limits") or [])

    @property
    def admission_reasons(self) -> List[str]:
        return list((self.manifest.get("admission") or {}).get("reasons") or [])

    @property
    def run_ids(self) -> List[str]:
        return [r["run_id"] for r in (self.manifest.get("source") or {}).get(
            "runs", [])]

    @property
    def evidence_ref(self) -> str:
        return AUDIT_EVIDENCE.get(self.set_id, "")

    def runs_for_model(self, model_id: Optional[str]) -> List[dict]:
        """按 run 的候选模型过滤（绝不跨模型聚合）。"""
        ids = {r["run_id"]: r.get("candidate_model_id")
               for r in self.identity.get("runs", [])}
        return [r for r in self.diff.get("runs", [])
                if model_id is None or ids.get(r["run_id"]) == model_id]

    def verified(self) -> bool:
        return self.admission == NOT_ADMITTED and not self.admitted

    def ledger_match_runs(self, model_id: Optional[str] = None,
                          include_excluded: bool = False) -> List[dict]:
        """row-level 候选提取（只对匹配模型；不跨模型借用）。

        默认剔除 owner 声明排除的 run（如自研 minidraft）；报告可用
        include_excluded=True 展示其排除原因。
        """
        ids = {r["run_id"]: r.get("candidate_model_id")
               for r in self.identity.get("runs", [])}
        out = []
        for r in self.ledger_rows.get("runs", []):
            tid = (r.get("target_model") or {}).get("id")
            if model_id is not None and tid != model_id:
                continue
            if model_id is None and ids.get(r["run_id"]) != tid:
                continue
            if r.get("excluded_by_owner") and not include_excluded:
                continue
            out.append(r)
        return out


def _check_manifest(manifest: dict, set_id: str) -> None:
    if manifest.get("schema_version") != SCHEMA:
        raise AuditIntegrityError(
            f"审计包 schema 不符：{manifest.get('schema_version')}")
    if manifest.get("profile_set_id") != set_id:
        raise AuditIntegrityError(
            f"manifest set_id 不匹配：{manifest.get('profile_set_id')}")
    if manifest.get("kind") != "ascend-profile-audit":
        raise AuditIntegrityError(
            f"kind 必须是 ascend-profile-audit，实际 {manifest.get('kind')}")
    if manifest.get("calibration_admission") != NOT_ADMITTED or \
            manifest.get("admitted"):
        # 审计包不得被标记为已准入；否则拒绝加载而不是悄悄忽略
        raise AuditIntegrityError(
            "审计包不得标记为已准入/measured（calibration_admission 必须为 "
            f"{NOT_ADMITTED}、admitted=false）")
    for key in ("distill_files", "source", "domain", "admission", "limits"):
        if key not in manifest:
            raise AuditIntegrityError(f"manifest 缺关键字段 {key}")


def load_audit_pack(set_id: str) -> AuditPack:
    if set_id not in AUDIT_PACKS:
        raise ValueError(
            f"未知审计包 {set_id}；可用: {sorted(AUDIT_PACKS)}"
            "（审计包不得当作 profile_set 使用）")
    d = AUDIT_ROOT / AUDIT_PACKS[set_id]
    manifest = _load_json(d / "manifest.json")
    _check_manifest(manifest, set_id)
    for name, sha in sorted((manifest.get("distill_files") or {}).items()):
        p = d / name
        if not p.is_file():
            raise AuditIntegrityError(f"缺产物 {name}")
        if _sha256(p) != sha:
            raise AuditIntegrityError(f"产物哈希不符 {name}（拒绝加载）")
    return AuditPack(
        set_id=set_id, manifest=manifest,
        identity=_load_json(d / "identity.json"),
        census=_load_json(d / "census.json"),
        mapping=_load_json(d / "mapping.json"),
        diff=_load_json(d / "diff.json"),
        shape_costs=_load_json(d / "shape_costs.json"),
        ledger_rows=_load_json(d / "ledger_rows.json"),
    )


def _domain_reasons(pack: AuditPack, model_id: Optional[str],
                    device_id: Optional[str], tp: Optional[int]) -> List[str]:
    """设备/实现/形状/TP 域门禁（与 CalibratedPricer 同语义；不匹配即回退）。"""
    from ..devices import DEVICES

    reasons: List[str] = []
    ident = {r["run_id"]: r for r in pack.identity.get("runs", [])}
    if model_id is not None:
        cands = pack.candidate_models
        if model_id not in cands:
            reasons.append(
                f"跨模型拒绝：审计包的采集模型为 {cands}，不适用于 {model_id}"
                "（历史 profile 只绑定采集模型，MERGE_CRITERIA §1.9）")
    if device_id is not None:
        dev = DEVICES.get(device_id)
        if dev is not None and getattr(dev, "backend", "cuda") == "ascend":
            reasons.append(
                f"设备家族可能匹配但 SKU 未证明：注册档 {device_id} 为 assumed "
                "旧仓解析映射（spec 未官方核验），profile 只有 NPU 家族名 "
                "Ascend910B、无 npu-smi board/SKU/驱动 → 不能证明同一 SKU/驱动，"
                "不满足设备域匹配")
        else:
            reasons.append(
                f"设备不符：审计包设备为 {pack.device_label}（SKU 未建立），场景设备 "
                f"{device_id} 不是该 profile 的 Ascend 设备家族；A-N4 未完成前无匹配"
                "设备域")
    else:
        reasons.append(
            "无已证明匹配的注册设备：审计包设备仅到 Ascend910B 家族（无 board/SKU）；"
            "本仓 ascend_910b3 档为 assumed 旧仓映射，不能据此宣称同 SKU/驱动，"
            "A-N4 官方规格与设备身份仍是校准前置")
    if tp is not None:
        hints = {r.get("tp_hint_from_groups") for r in ident.values()}
        if hints == {None}:
            reasons.append("TP 域未知：profile 未记录 tp 并行组大小，无法与场景 TP 比对")
        elif tp not in hints:
            reasons.append(f"TP 不符：场景 TP{tp} ∉ profile TP 域 {sorted(h for h in hints if h)}")
    return reasons


def _pack_index_row(pack: AuditPack) -> dict:
    return {
        "set_id": pack.set_id,
        "kind": "ascend-profile-audit",
        "device_label": pack.device_label,
        "candidate_model_ids": pack.candidate_models,
        "run_ids": pack.run_ids,
        "calibration_admission": pack.admission,
        "admitted": pack.admitted,
        "evidence_ref": pack.evidence_ref,
        "ledger_row_match": "candidate-only（not admitted）",
        "excluded_runs": [x.get("run_id") for x in
                          (pack.manifest.get("owner_declared") or {}).get(
                              "excluded_runs", [])],
        "report_endpoint": f"/api/v2/profile-audit/{pack.set_id}",
        "note": "只读审计（原始 trace 审核/映射/差异/行级候选）；不参与计价与排名",
    }


def audit_packs_json() -> List[dict]:
    """索引（工作台/API 用）；不暴露可选择的 cost_profile。"""
    out = []
    for set_id in sorted(AUDIT_PACKS):
        try:
            out.append(_pack_index_row(load_audit_pack(set_id)))
        except (AuditIntegrityError, ValueError) as exc:  # 资产损坏如实传播
            out.append({"set_id": set_id, "kind": "ascend-profile-audit",
                        "calibration_admission": "unknown",
                        "admitted": False, "integrity_error": str(exc)})
    return out


def _run_report(pack: AuditPack, run: dict) -> dict:
    ident = next((r for r in pack.identity.get("runs", [])
                  if r["run_id"] == run["run_id"]), {})
    census = next((r for r in pack.census.get("runs", [])
                   if r["run_id"] == run["run_id"]), {})
    return {
        "run_id": run["run_id"],
        "boundary": run.get("boundary"),
        "observed_kernel_us": run.get("observed_kernel_us"),
        "mapped_ledger_time_us": run.get("mapped_ledger_time_us"),
        "communication_time_us": run.get("communication_time_us"),
        "unmapped_time_us": run.get("unmapped_time_us"),
        "mapped_ledger_time_share": run.get("mapped_ledger_time_share"),
        "communication_time_share": run.get("communication_time_share"),
        "unmapped_time_share": run.get("unmapped_time_share"),
        "coverage_note": run.get("coverage_note"),
        "ledger_row_coverage": run.get("ledger_row_coverage"),
        "groups": run.get("groups", []),
        "identity": {k: ident.get(k) for k in (
            "candidate_model_id", "owner_declared", "tp_hint_from_groups",
            "observed_geometry", "geometry_check", "geometry_verification",
            "source_files_sha256")},
        "observed_dtypes": census.get("observed_dtypes"),
        "union_gap_stats": census.get("union_gap_stats"),
        "stream_gap_stats": census.get("stream_gap_stats"),
        "op_statistic_consistency": census.get("op_statistic_consistency"),
        "step_trace": census.get("step_trace"),
    }


def audit_report_json(set_id: str, model_id: Optional[str] = None,
                      device_id: Optional[str] = None,
                      tp: Optional[int] = None) -> dict:
    """完整审计报告（API/CLI 导出）。任何域不匹配都只落到 reasons，
    `candidate_calibration.selected_ms` 恒为 None、不产生成本结论。"""
    pack = load_audit_pack(set_id)
    reasons = _domain_reasons(pack, model_id, device_id, tp)
    runs = pack.runs_for_model(model_id)
    if model_id is not None and not runs:
        reasons.append(f"审计包内没有 {model_id} 对应的 run（不跨模型借用）")
    limits = list(pack.limits)
    if model_id is not None and runs:
        geometry = [r.get("identity", {}).get("geometry_check") for r in runs]
        bad = [g for g in geometry if g and g.get("checked")
               and not g.get("shape_consistent")]
        if bad:
            reasons.append("形状域不匹配：" + "; ".join(
                f"{g['model_id']}: {g['failures']}" for g in bad))
    return {
        "schema_version": SCHEMA,
        "set_id": pack.set_id,
        "kind": "ascend-profile-audit",
        "source": {
            "kind": (pack.manifest.get("source") or {}).get("kind"),
            "roots": (pack.manifest.get("source") or {}).get("roots"),
            "runs": (pack.manifest.get("source") or {}).get("runs"),
            "identity_stacks": (pack.manifest.get("source") or {}).get("identity_stacks"),
            "files_sha256": {
                name: {"sha256": sha,
                       "size": (AUDIT_ROOT / AUDIT_PACKS[pack.set_id] / name)
                       .stat().st_size if (AUDIT_ROOT / AUDIT_PACKS[pack.set_id] /
                                           name).is_file() else None}
                for name, sha in sorted(pack.manifest["distill_files"].items())},
        },
        "identity_gap_summary": pack.identity.get("identity_gap_summary"),
        "owner_declared": pack.manifest.get("owner_declared"),
        "device": {
            "family": pack.device_label,
            "sku": (pack.manifest.get("domain") or {}).get("device_sku"),
            "implementation_id": (pack.manifest.get("domain") or {}).get(
                "implementation_id"),
            "precision_observed": (pack.manifest.get("domain") or {}).get(
                "precision_observed"),
        },
        "domain": {
            "candidate_model_ids": pack.candidate_models,
            "requested_model_id": model_id,
            "runs_used": [r["run_id"] for r in runs],
            "merge_allowed": False,
            "merge_reason": "不同 run（draft 方法/日期/会话）身份不同；审计只并列，"
                            "不聚合成一个 profile（禁止跨 run 拼接）",
            "declared": pack.manifest.get("domain"),
        },
        "admission": {
            "calibration_admission": pack.admission,
            "admitted": pack.admitted,
            "prerequisites": (pack.manifest.get("admission") or {}).get(
                "prerequisites"),
            "reasons": reasons,
            "admission_reasons_from_data": pack.admission_reasons,
        },
        "runs": runs,
        "mapping": pack.mapping,
        "coverage": {
            "rows": None,
            "time_note": "mapped_ledger_time_share 为设备 kernel 时间口径；"
                         "通信类/未建模类单列，分母含所有设备 kernel 时间（未做"
                         "关键路径去重）",
            "runs": [{"run_id": r["run_id"],
                      "mapped_ledger_time_share": r.get("mapped_ledger_time_share"),
                      "communication_time_share": r.get("communication_time_share"),
                      "unmapped_time_share": r.get("unmapped_time_share")}
                     for r in runs],
        },
        "candidate_calibration": {
            "selected_ms": None,
            "analytic_base_ms": None,
            "fallback_reason": "审计包 calibration_admission=not-admitted："
                              "无匹配注册设备（A-N4）+ 采集身份不完整"
                              "（引擎 commit/权重哈希/warmup/step 边界）→ 不进入 "
                              "CalibratedPricer",
            "note": "本字段存在只是为未来接入保留形状；当前恒为 None，"
                    "禁止用审计数据生成成本或排名",
        },
        "evidence_ref": pack.evidence_ref,
        "ledger_row_match": {
            "file": "ledger_rows.json",
            "status": "candidate-only（audit-only；not admitted）",
            "runs": [{
                "run_id": r["run_id"],
                "target_model": r.get("target_model"),
                "draft_method": r.get("draft_method"),
                "excluded_by_owner": r.get("excluded_by_owner"),
                "pass_structure": r.get("pass_structure"),
                "coverage": r.get("coverage"),
                "rows_with_allocations": sum(
                    1 for x in r.get("rows", []) if x.get("allocated_to")),
                "unallocated_groups": [g.get("group") for g in
                                       r.get("unallocated_groups", [])],
                "reason": r.get("reason"),
            } for r in pack.ledger_match_runs(model_id)],
        },
        "limits": limits,
    }


def require_not_admitted(set_id: str, purpose: str = "校准") -> None:
    """任何试图把审计包当校准/测量用的调用必须经过这里并显式失败。"""
    pack = load_audit_pack(set_id)
    raise AuditAdmissionBlocked(
        f"审计包 {set_id} 未准入（calibration_admission={pack.admission}），"
        f"不能用于{purpose}；前置：A-N4 设备身份 + 匹配采数")
