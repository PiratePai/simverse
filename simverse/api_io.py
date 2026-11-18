"""入口输入解码（R10d）：/api/v2/run 与 /api/v2/sweep 共用的场景校验。

背景：两个端点此前直接 ``ScenarioV2.from_json(body)``，顶层数组/
``deployment`` 数组等结构性错误会抛 AttributeError/TypeError → Flask
HTML 500；合法但不受支持的组合又与真正的内部异常混在同一个 except 里。

契约（与任务验收对齐）：
  * 非法顶层类型 / 非法嵌套结构 / 非法字段值 → ScenarioInputError
    （ValueError 子类），携带可定位字段路径（JSON Pointer 风格 ``$.a.b``）；
  * 合法但不受支持的场景：照常进 run_v2，沿用结果状态与能力说明
    （manifest.status=unsupported_*，不是 4xx）；
  * 其余未预期异常：不在此捕获，保持服务端 500 语义。
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from .scenario import DeploymentSpec, ScenarioV2

# ScenarioV2 / DeploymentSpec / WorkloadHypothesis / BudgetInput 的
# 结构类型表：字段名 → (允许的 Python 类型, 是否允许 None)
_SCENARIO_FIELDS = {
    "schema_version": (str, False), "name": (str, False),
    "model_id": (str, False), "draft_id": (str, True),
    "method_id": (str, False), "k": (int, False),
    "block_size": (int, True), "deployment": (dict, False),
    "workload": (dict, False), "acceptance": (dict, False),
    "quant": (dict, False), "draft_quant": (dict, False),
    "draft_ctx_kv_path": (str, False), "cost_profile": (str, True),
    "cost_records": (str, True), "seed": (int, False), "notes": (str, False),
}
_DEPLOY_FIELDS = {
    "mode": (str, False), "target_device": (str, False),
    "target_tp": (int, False), "target_ep": (int, False),
    "draft_device": (str, False), "draft_tp": (int, False),
    "link": (str, False), "draft_on_target_group": (bool, False),
    "pipeline_microbatches": (int, False), "pipeline_rounds": (int, False),
    "budget": (dict, True), "draft_budget": (dict, True),
    "engram_table_placement": (str, False), "v41_kv_record": (str, False),
}
_WORKLOAD_FIELDS = {
    "batch": (int, False), "ctx_tokens": (int, False),
    "osl_tokens": (int, True), "arrival": (str, False),
    "source_kind": (str, False), "notes": (str, False),
}
_BUDGET_FIELDS = {
    "allocatable_bytes": (int, True), "allocatable_kind": (str, False),
    "allocatable_source": (str, False), "extra_overhead_bytes": (int, False),
    "overhead_source": (str, False), "note": (str, False),
}

# 语义校验（validate 的 ValueError）到字段路径的确定性映射：按序取
# 消息中第一个被点名的字段；无匹配归 "$"（整个场景）。
_FIELD_TOKENS = (
    ("k ∈", "$.k"), ("batch", "$.workload.batch"), ("ctx", "$.workload.ctx_tokens"),
    ("target_ep", "$.deployment.target_ep"), ("target_tp", "$.deployment.target_tp"),
    ("draft_tp", "$.deployment.draft_tp"), ("allocatable_bytes", "$.deployment.budget.allocatable_bytes"),
    ("allocatable_kind", "$.deployment.budget.allocatable_kind"),
    ("allocatable_source", "$.deployment.budget.allocatable_source"),
    ("extra_overhead_bytes", "$.deployment.budget.extra_overhead_bytes"),
    ("draft_budget", "$.deployment.draft_budget"),
    ("部署模式", "$.deployment.mode"), ("baseline 部署", "$.deployment.mode"),
    ("schema_version", "$.schema_version"), ("cost_records", "$.cost_records"),
    ("cost_profile", "$.cost_profile"), ("draft_ctx_kv_path", "$.draft_ctx_kv_path"),
    ("engram_table_placement", "$.deployment.engram_table_placement"),
    ("v41_kv_record", "$.deployment.v41_kv_record"),
    ("dense_w", "$.quant.dense_w"), ("expert_w", "$.quant.expert_w"),
    ("act", "$.quant.act"), ("state", "$.quant.state"),
    ("draft_quant", "$.draft_quant"), ("model_id", "$.model_id"),
    ("method_id", "$.method_id"), ("未知模型", "$.model_id"),
    ("alpha ∈", "$.acceptance.alpha"), ("alphas ∈", "$.acceptance.alphas"),
    ("'alpha'", "$.acceptance.alpha"), ("'alphas'", "$.acceptance.alphas"),
    ("'mean'", "$.acceptance.mean"),
)


class ScenarioInputError(ValueError):
    """场景输入非法（结构/类型/取值）；field 为可定位路径。"""

    def __init__(self, field: str, message: str):
        super().__init__(message)
        self.field = field


def _check_types(obj: Any, table: Dict[str, tuple], path: str,
                 what: str, *, allow_unknown: bool) -> None:
    if not isinstance(obj, dict):
        raise ScenarioInputError(path, f"{what} 必须是 JSON 对象，非 {type(obj).__name__}")
    for key, value in obj.items():
        if not isinstance(key, str):
            raise ScenarioInputError(path, f"{what} 字段名必须是字符串：{key!r}")
        if key not in table:
            if allow_unknown:
                continue
            raise ScenarioInputError(
                f"{path}.{key}", f"未知字段 {key!r}（{what}）；未登记字段不默认忽略")
        types, nullable = table[key]
        types = types if isinstance(types, tuple) else (types,)
        if value is None:
            if not nullable:
                raise ScenarioInputError(f"{path}.{key}", f"{key} 不允许为 null")
            continue
        if isinstance(value, bool) and bool not in types:
            raise ScenarioInputError(
                f"{path}.{key}", f"{key} 必须为 {types[0].__name__}，非布尔")
        if not isinstance(value, types):
            raise ScenarioInputError(
                f"{path}.{key}",
                f"{key} 必须为 {'/'.join(t.__name__ for t in types)}，"
                f"实际 {type(value).__name__}")


def _check_acceptance(acc: Dict[str, Any]) -> None:
    """接受率假设的字段校验（与 methods.acceptance.law_from_scenario 的
    域一致：越界 kind/alpha 属于字段值非法，入口 4xx，不留给管线 failed）。"""
    def num(value) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    kind = acc.get("kind", "geometric")
    if not isinstance(kind, str):
        raise ScenarioInputError("$.acceptance.kind",
                                 f"kind 必须为 str，实际 {type(kind).__name__}")
    if kind not in ("geometric", "position_alphas", "committed_mean",
                    "dynamic_k", "empirical_hist"):
        raise ScenarioInputError("$.acceptance.kind", f"未知接受输入 kind={kind}")
    if kind == "geometric":
        alpha = acc.get("alpha")
        if not num(alpha) or not 0.0 <= float(alpha) <= 1.0:
            raise ScenarioInputError("$.acceptance.alpha",
                                     f"alpha ∈ [0,1]，实际 {alpha!r}")
    elif kind == "position_alphas":
        alphas = acc.get("alphas")
        if not isinstance(alphas, list) or not alphas or \
                not all(num(a) and 0.0 <= float(a) <= 1.0 for a in alphas):
            raise ScenarioInputError("$.acceptance.alphas",
                                     "alphas 必须为 [0,1] 内非空数值数组")
    elif kind == "committed_mean":
        mean = acc.get("mean")
        if not num(mean) or float(mean) <= 0:
            raise ScenarioInputError("$.acceptance.mean",
                                     f"mean 必须为正数，实际 {mean!r}")


def decode_scenario_v2(body: Any) -> ScenarioV2:
    """结构/类型校验（带路径）→ ScenarioV2.from_json 语义校验。

    顶层未知键沿用 from_json 的既有行为（不因添加未知键拒绝调用方；
    嵌套对象键严格登记，防止拼错键静默丢配置）。语义校验的 ValueError
    尽力映射到字段路径，映射不到的归 "$"。
    """
    _check_types(body, _SCENARIO_FIELDS, "$", "场景顶层", allow_unknown=True)
    if not isinstance(body, dict):
        raise ScenarioInputError("$", "顶层必须是 JSON 对象")  # pragma: no cover
    if "deployment" in body and body["deployment"] is not None:
        _check_types(body["deployment"], _DEPLOY_FIELDS, "$.deployment",
                     "deployment", allow_unknown=False)
        for bkey in ("budget", "draft_budget"):
            bval = body["deployment"].get(bkey)
            if isinstance(bval, dict):
                _check_types(bval, _BUDGET_FIELDS,
                             f"$.deployment.{bkey}", bkey, allow_unknown=False)
    if "workload" in body and body["workload"] is not None:
        _check_types(body["workload"], _WORKLOAD_FIELDS, "$.workload",
                     "workload", allow_unknown=False)
    if isinstance(body.get("acceptance"), dict):
        _check_acceptance(body["acceptance"])
    for qkey in ("quant", "draft_quant"):
        qval = body.get(qkey)
        if isinstance(qval, dict):
            for sub, val in qval.items():
                if not isinstance(sub, str) or isinstance(val, bool) \
                        or not isinstance(val, int):
                    raise ScenarioInputError(
                        f"$.{qkey}.{sub}",
                        f"{qkey}.{sub} 必须为整数位宽（bit），实际 {type(val).__name__}")
    try:
        return ScenarioV2.from_json(body)
    except ScenarioInputError:
        raise
    except (ValueError, TypeError, KeyError) as exc:
        message = str(exc)
        field: Optional[str] = None
        for token, path in _FIELD_TOKENS:
            if token in message:
                field = path
                break
        raise ScenarioInputError(field or "$", message) from exc


def error_payload(exc: ScenarioInputError) -> Dict[str, Any]:
    """结构化 4xx 响应体（workbench.js 读 error 字段，兼容既有形状）。"""
    return {"error": str(exc), "field": exc.field}
