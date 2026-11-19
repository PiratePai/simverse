"""工作台审查修复（2026-11-17）专属测试：文档路由/导出端点/CSV 转义/
产品目录边界/后端组合事实（UI 镜像的权威来源）。

与后端成本任务的测试文件分离；不修改计算管线。浏览器端交互验证
（串行矩阵在途=1、Sweep 400/网络失败、控件联动）见
docs/validation/workbench-review-fixes.md §验证记录。
"""
import csv
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simverse.contracts import js_restore, js_safe  # noqa: E402
from simverse.pipeline import run_v2  # noqa: E402
from simverse.scenario import ScenarioV2  # noqa: E402

DOC_LINKS = [
    "/docs/rebuild-20261106/plan.md",
    "/docs/rebuild-20261106/principles.md",
    "/docs/rebuild-20261106/ui-data-contract.md",
    "/docs/rebuild-20261106/prototype/index.html",
]
# 原型页引用的相对资源（链接修复后必须仍可解析）
DOC_ASSETS = [
    "/docs/rebuild-20261106/prototype/v3-workbench.js",
    "/docs/rebuild-20261106/prototype/style.css",
    "/docs/rebuild-20261106/plan.md",  # 原型页内 ../plan.md 相对引用
]


def _client():
    from simverse.webapp import app
    app.config["TESTING"] = True
    return app.test_client()


# ---------- A. 文档导航 ----------

def test_doc_links_served():
    c = _client()
    for url in DOC_LINKS + DOC_ASSETS:
        r = c.get(url)
        assert r.status_code == 200, (url, r.status_code)
        assert len(r.data) > 200, url


def test_docs_route_only_docs_root():
    """只开放 docs 根：穿越/绝对路径/不存在一律 404，不泄露文件系统。"""
    c = _client()
    for url in (
        "/docs/../simverse/webapp.py",
        "/docs/..%2F..%2Fsimverse%2Fwebapp.py",
        "/docs/rebuild-20261106/../../../../simverse/scenario.py",
        "/docs/nonexistent.md",
        "/docs/",
    ):
        r = c.get(url)
        assert r.status_code == 404, (url, r.status_code)


def test_markdown_served_as_escaped_html():
    c = _client()
    r = c.get("/docs/rebuild-20261106/principles.md")
    assert r.status_code == 200
    assert r.content_type.startswith("text/html")
    body = r.get_data(as_text=True)
    assert "white-space:pre-wrap" in body  # 轻量阅读视图（无渲染依赖）
    # 原文标题存在且 HTML 转义（防注入）
    src = (ROOT / "docs/rebuild-20261106/principles.md").read_text(encoding="utf-8")
    assert src.splitlines()[0].lstrip("# ")[:20] in body


def test_workbench_html_links_are_absolute_docs_paths():
    html = (ROOT / "simverse/static/workbench.html").read_text(encoding="utf-8")
    assert "../../docs/" not in html, "顶栏不得再用相对 ../../docs/ 链接"
    for url in DOC_LINKS:
        assert f'href="{url}"' in html, url
    js = (ROOT / "simverse/static/workbench.js").read_text(encoding="utf-8")
    assert "../../docs/" not in js


# ---------- I. 导出端点与 CSV ----------

def test_export_endpoint_explicitly_deprecated():
    """旧 /api/v2/export 恒 400 且注释误导 → 现明示弃用（410+原因），
    不再伪装可用；导出统一从 /api/v2/run 的 bundle 派生。"""
    c = _client()
    r = c.get("/api/v2/export/csv")
    assert r.status_code == 410
    assert "弃用" in r.get_json()["error"]


def _qwen_dflash_bundle():
    sc = ScenarioV2.from_json({
        "schema_version": "0.2", "name": "wb-csv-check",
        "model_id": "qwen36_35b_a3b", "draft_id": "dflash_qwen36_35b_a3b",
        "method_id": "dflash_v1", "k": 15, "block_size": 16,
        "deployment": {"mode": "colocated", "target_device": "h100_sxm",
                       "target_tp": 8, "target_ep": 1},
        "workload": {"batch": 1, "ctx_tokens": 4096},
        "acceptance": {"kind": "geometric", "alpha": 0.72},
    })
    return run_v2(sc).to_json()


def test_ops_csv_v2_rfc4180_roundtrip():
    """CSV 解析器逐字段往返（非 split(',')）：真实 bundle 行 + 含逗号 shape。"""
    from simverse.cli import _ops_csv_v2
    bundle = _qwen_dflash_bundle()
    text = _ops_csv_v2(bundle)
    rows = list(csv.reader(io.StringIO(text)))
    header, body = rows[0], rows[1:]
    src = bundle["work"]["target_decode"]["rows"]
    assert len(body) == len(src) > 0
    # 至少一行 shape 含逗号（证明走的是引号转义路径而非替换）
    assert any("," in (r["shape"] or "") for r in src)
    for rec, orig in zip(body, src):
        d = dict(zip(header, rec))
        assert d["stage"] == orig["stage"] and d["op"] == orig["op"]
        assert d["layer"] == str(orig["layer"])
        assert d["flops"] == str(orig["flops"])
        assert d["shape"] == (orig["shape"] or "")
        # t_ms/bound 与后端计价行 row_id 对齐（无计价行留空，不伪造）
        pts = {p["row_id"]: p for p in bundle["cost"].get(
            "row_prices", bundle["cost"]["roofline_points"])}
        p = pts.get(f"{orig['stage']}:{orig['layer']}:{orig['op']}")
        if p:
            assert d["t_ms"] == str(p["t_ms"])
            assert d["bound"] == p["bound"]
        else:
            assert d["t_ms"] == "" and d["bound"] == ""


def test_csv_escape_poisoned_fields():
    """逗号/双引号/换行/CRLF 混合字段逐字节往返（CLI 与 Python csv 解析器）。"""
    from simverse.exports import _csv
    cols = ["a", "b", "c"]
    records = [
        {"a": "x,y", "b": 'say "hi"', "c": "line1\nline2"},
        {"a": "plain", "b": 'quote""double', "c": "cr\r\nlf"},
        {"a": "", "b": None, "c": 123},
    ]
    text = _csv(cols, records)
    back = list(csv.reader(io.StringIO(text)))
    assert back[0] == cols
    for rec, orig in zip(back[1:], records):
        assert rec[0] == ("" if orig["a"] is None else str(orig["a"]))
        assert rec[1] == ("" if orig["b"] is None else str(orig["b"]))
        assert rec[2] == str(orig["c"])


def test_trace_csv_v2_escapes_dep_lists():
    """depends_on 列表 str() 含逗号/引号——标准转义后可往返。"""
    from simverse.cli import _trace_csv_v2
    bundle = _qwen_dflash_bundle()
    text = _trace_csv_v2(bundle)
    rows = list(csv.reader(io.StringIO(text)))
    events = [e for rnd in bundle["execution"]["rounds"] for e in rnd["events"]]
    assert len(rows) - 1 == len(events) > 0
    for rec, e in zip(rows[1:], events):
        d = dict(zip(rows[0], rec))
        assert d["event_id"] == e["event_id"]
        assert d["depends_on"] == str(e.get("depends_on", ""))


def test_csv_post_serializes_old_bundle_and_exact_large_integers(monkeypatch):
    from simverse import pipeline
    from simverse.exports import _groups_csv_v2, _ops_csv_v2

    def unexpected_run(*args, **kwargs):
        raise AssertionError("CSV export should use the supplied result")

    monkeypatch.setattr(pipeline, "run_v2", unexpected_run)
    exact_flops = 2**53 + 1
    bundle = {
        "schema_version": "0.2",
        "work": {"target_decode": {"rows": [
            {"stage": "target_decode", "layer": 0, "op": "k_proj",
             "flops": exact_flops, "shape": '报价, "原值"\r\n下一行'},
        ]}},
        "cost": {
            "record_cost": {"matched": 0},
            "roofline_points": [{"row_id": "target_decode:0:k_proj",
                                 "t_ms": 0.125, "bound": "memory"}],
            "groups": [{"stage": "unpriced_runtime", "selected_ms": None}],
        },
    }
    c = _client()
    body = js_safe(bundle)
    ops = c.post("/api/v2/export/ops", json=body)
    assert ops.status_code == 200 and ops.mimetype == "text/csv"
    assert ops.get_data(as_text=True) == _ops_csv_v2(bundle)
    row, = csv.DictReader(io.StringIO(ops.get_data(as_text=True)))
    assert row["flops"] == str(exact_flops)
    assert row["t_ms"] == "0.125" and row["bound"] == "memory"
    assert row["shape"] == bundle["work"]["target_decode"]["rows"][0]["shape"]
    groups = c.post("/api/v2/export/groups", json=body)
    assert groups.status_code == 200
    assert groups.get_data(as_text=True) == _groups_csv_v2(bundle)
    assert c.post("/api/v2/export/ops", json=[]).status_code == 400
    assert c.post("/api/v2/export/unknown", json=body).status_code == 404


# ---------- D. 产品目录边界与后端组合事实（UI 联动的权威来源） ----------

def test_catalog_product_scope_no_chain_no_tiny():
    c = _client()
    j = c.get("/api/v2/catalog").get_json()
    assert set(j["catalog"]["models"]) <= {"qwen36_35b_a3b", "deepseek_v41_flash"}
    methods = j["methods"]
    selectable = set(methods["supported"]) | set(methods.get("research", []))
    # chain_*/tiny 不出现在任何产品可选入口（内部研究资产不重新开放）
    assert selectable <= {"baseline", "dflash_v1", "mtp"}
    for blob in (methods["supported"], methods.get("research", []),
                 list(methods.get("unsupported", {}))):
        assert "chain_draft" not in blob and "chain_moe_proxy" not in blob
        assert not any(str(m).startswith("tiny_") for m in blob)


def test_tp4_is_legal_for_both_models():
    """UI 补 TP4 的后端依据：两产品模型 TP4 并行配方合法。"""
    from simverse.lowering.parallel import validate_parallel
    from simverse.model_ir import build_catalog
    cat = build_catalog()
    for mid in ("qwen36_35b_a3b", "deepseek_v41_flash"):
        assert validate_parallel(cat["models"][mid], 4, 1).valid, mid


def test_v41_dflash_and_mtp_rejected_by_backend():
    """UI 冻结 V4.1 方法选择的权威事实：后端对非 baseline 显式拒绝。"""
    for method in ("dflash_v1", "mtp"):
        sc = ScenarioV2.from_json({
            "schema_version": "0.2", "name": "probe", "model_id": "deepseek_v41_flash",
            "draft_id": "dflash_qwen36_35b_a3b" if method == "dflash_v1" else None,
            "method_id": method, "k": 4,
            "quant": {"dense_w": 8, "expert_w": 4, "act": 16, "kv": 8, "state": 32},
            "deployment": {"mode": "colocated", "target_device": "h100_sxm",
                           "target_tp": 8},
            "acceptance": {"kind": "geometric", "alpha": 0.72},
        })
        b = run_v2(sc).to_json()
        assert b["manifest"]["status"] == "failed", method
        assert "error" in b["metrics"]


def test_qwen_mtp_dt_rejected_by_backend():
    sc = ScenarioV2.from_json({
        "schema_version": "0.2", "name": "probe", "model_id": "qwen36_35b_a3b",
        "draft_id": None, "method_id": "mtp", "k": 1,
        "deployment": {"mode": "dt_serial", "target_device": "h100_sxm",
                       "target_tp": 8, "draft_device": "l40"},
        "acceptance": {"kind": "geometric", "alpha": 0.72},
    })
    b = run_v2(sc).to_json()
    assert b["manifest"]["status"] == "failed"
    assert "MTP" in b["metrics"]["error"]["reason"]
    assert "dt" in b["metrics"]["error"]["reason"]


def test_tp4_run_completes():
    """TP4 全链路（UI 新档位）实跑完成。"""
    sc = ScenarioV2.from_json({
        "schema_version": "0.2", "name": "tp4", "model_id": "qwen36_35b_a3b",
        "draft_id": "dflash_qwen36_35b_a3b", "method_id": "dflash_v1",
        "k": 15, "block_size": 16,
        "deployment": {"mode": "colocated", "target_device": "h100_sxm",
                       "target_tp": 4, "target_ep": 1},
        "workload": {"batch": 1, "ctx_tokens": 4096},
        "acceptance": {"kind": "geometric", "alpha": 0.72},
    })
    b = run_v2(sc).to_json()
    assert b["manifest"]["status"] == "completed"
    assert b["metrics"]["M01_tpot_analytical"]["value"] > 0


# ---------- G. M06 双端显存 ----------

def test_m06_both_ends_present_dt():
    """DT 分离：target/draft 两端 M06 各自成键；_detail 为账本字段非 Metric。"""
    c = _client()
    r = c.post("/api/v2/run", json={
        "schema_version": "0.2", "name": "mem-dt", "model_id": "qwen36_35b_a3b",
        "draft_id": "dflash_qwen36_35b_a3b", "method_id": "dflash_v1",
        "k": 15, "block_size": 16,
        "deployment": {"mode": "dt_serial", "target_device": "h100_sxm",
                       "target_tp": 8, "draft_device": "l40", "draft_tp": 1},
        "workload": {"batch": 1, "ctx_tokens": 4096},
        "acceptance": {"kind": "geometric", "alpha": 0.72},
    })
    assert r.status_code == 200
    b = js_restore(r.get_json())
    keys = [k for k in b["metrics"] if k.startswith("M06_memory")
            and not k.endswith("_detail")]
    assert "M06_memory_target_r0" in keys
    assert "M06_memory_draft_r0" in keys
    for k in keys:  # Metric 四态字段齐备（UI 按此区分 unknown/oom/feasible/条件）
        m = b["metrics"][k]
        assert m["feasibility"] in ("unknown", "oom", "feasible")
        assert "support" in m
        assert isinstance(b["metrics"][k + "_detail"], dict)
    # 两端不同设备：容量与判定不得合并（UI 分端展示的依据）
    t = b["metrics"]["M06_memory_target_r0"]
    d = b["metrics"]["M06_memory_draft_r0"]
    assert t["scope"]["role"] == "target" and d["scope"]["role"] == "draft"
    assert t["value"] != d["value"]


def test_m06_conditional_expressed_as_approximated():
    """假设预算 → feasible + support=approximated（条件可行）：UI 判别依据。"""
    sc = ScenarioV2.from_json({
        "schema_version": "0.2", "name": "mem-cond", "model_id": "qwen36_35b_a3b",
        "draft_id": "dflash_qwen36_35b_a3b", "method_id": "dflash_v1",
        "k": 15, "block_size": 16,
        "deployment": {"mode": "dt_serial", "target_device": "h100_sxm",
                       "target_tp": 8, "draft_device": "l40", "draft_tp": 1,
                       "draft_budget": {"allocatable_bytes": 40 << 30,
                                        "allocatable_kind": "hypothesis",
                                        "allocatable_source": "测试假设",
                                        "extra_overhead_bytes": 0,
                                        "overhead_source": ""}},
        "workload": {"batch": 1, "ctx_tokens": 4096},
        "acceptance": {"kind": "geometric", "alpha": 0.72},
    })
    b = run_v2(sc).to_json()
    m = b["metrics"]["M06_memory_draft_r0"]
    # 峰值完整性缺口下仍 unknown；若判定到 feasible 必为 approximated（条件），
    # 且带假设引用——不允许出现 supported 的“实测可部署”
    if m["feasibility"] == "feasible":
        assert m["support"] == "approximated"
        assert m.get("assumption_refs")
    else:
        assert m["feasibility"] == "unknown"


# ---------- B. 矩阵串行/防重叠（源码回归绊线；行为由浏览器验证） ----------

def test_matrix_serialization_source_guards():
    js = (ROOT / "simverse/static/workbench.js").read_text(encoding="utf-8")
    # runCell 返回真 Promise（请求处理完才 resolve，供串行链 await）
    assert "return runScenario(" in js
    # runMatrix 串行 reduce + 防重复点击 + finally 恢复按钮
    assert "state.matrixRunning" in js
    assert ".reduce(function (chain, td)" in js
    assert ".finally(function ()" in js
    # 失败/unsupported 不显示成功数值
    assert "不支持/失败" in js
    # Sweep：HTTP 状态检查 + 非 JSON 处理 + {error} 不渲染成功表
    assert "!r.ok" in js
    assert "响应非 JSON" in js
    assert "不渲染为空成功表" in js


def test_sweep_400_error_shape():
    """Sweep 后端 400 返回 {error:...}——前端必须显示原因而非空表。"""
    c = _client()
    r = c.post("/api/v2/sweep", json={
        "schema_version": "0.2", "model_id": "qwen36_35b_a3b", "method_id": "dflash_v1",
        "k": 15, "deployment": {"mode": "baseline", "target_device": "h100_sxm"},
        "acceptance": {"kind": "geometric", "alpha": 0.72}})
    assert r.status_code == 400
    assert "error" in r.get_json()


# ---------- E/F. 透视与 Roofline 的数据面依据 ----------

def test_roofline_row_ids_join_work_rows():
    """work 行无 t_ms/bound；计价行以 row_id=stage:layer:op 稳定关联
    （UI 透视聚合/Roofline 选点/CSV 关联的契约）。"""
    bundle = _qwen_dflash_bundle()
    rows = bundle["work"]["target_decode"]["rows"]
    ids = [f"{r['stage']}:{r['layer']}:{r['op']}" for r in rows]
    assert len(ids) == len(set(ids)), "row_id 必须可唯一寻址"
    pts = bundle["cost"]["roofline_points"]
    assert pts and all(p["row_id"] in set(ids) for p in pts)
    for p in pts:
        for k in ("ai_flop_per_byte", "attained_tflops", "bound", "t_ms"):
            assert k in p, k


def test_full_row_prices_cover_every_work_row_and_stage_total():
    bundle = _qwen_dflash_bundle()
    rows = bundle["work"]["target_decode"]["rows"]
    priced = bundle["cost"]["row_prices"]
    assert len(priced) == len(rows) > len(bundle["cost"]["roofline_points"])
    assert {p["row_id"] for p in priced} == {
        f"{r['stage']}:{r['layer']}:{r['op']}" for r in rows}
    assert sum(p["t_ms"] for p in priced) <= bundle["cost"]["target_decode"]["t_ms"]


def test_v41_multi_shape_ops_exist():
    """V4.1 存在同 op 多 shape 的行组（透视“多种 shape”展示的对象）。"""
    sc = ScenarioV2.from_json({
        "schema_version": "0.2", "name": "v41-shapes", "model_id": "deepseek_v41_flash",
        "draft_id": None, "method_id": "baseline", "k": 1,
        "quant": {"dense_w": 8, "expert_w": 4, "act": 16, "kv": 8, "state": 32},
        "deployment": {"mode": "baseline", "target_device": "h100_sxm",
                       "target_tp": 8},
        "workload": {"batch": 1, "ctx_tokens": 4096},
        "acceptance": {"kind": "geometric", "alpha": 0.72},
    })
    b = run_v2(sc).to_json()
    shapes = {}
    for r in b["work"]["target_decode"]["rows"]:
        shapes.setdefault(r["op"], set()).add(r["shape"])
    multi = [op for op, s in shapes.items() if len(s) > 1]
    assert multi, "V4.1 应存在多 shape 算子组（如 sparse_attention）"
