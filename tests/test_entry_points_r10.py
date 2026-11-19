"""R10 入口修复回归（批次一）：

* R10c 资产定位：源码检出/wheel 双模式解析、wheel 清单覆盖受哈希约束
  资产且排除归档采集（完整 wheel 安装冒烟在 CI 的 tools/wheel_smoke.py，
  不在 pytest 内重复构建）；
* R10d 共享预设：预设文件与原 CLI 硬编码字典 digest 逐位一致；CLI 默认
  run-v2、API run、目录 catalog 三入口同源；run/sweep 共用解码——畸形
  输入结构化 4xx 带字段路径，合法场景照常运行，内部异常保持 500；
* R10b 根目录文档：白名单（ROADMAP/AGENTS）可从状态页相对链接打开，
  白名单外根路径与不存在的文档仍 404。

修复前最小复现见 docs/validation/entry-scan-compare-fixes-20261119.md。
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simverse.webapp import app  # noqa: E402


# ---------------------------------------------------------------------------
# R10c 资产定位
# ---------------------------------------------------------------------------

def test_r10c_asset_locator_prefers_source_checkout():
    from simverse import assets
    root = assets.repo_root()
    assert root == ROOT
    assert assets.assets_dir() == ROOT / "assets"
    assert assets.scenario_dir() == ROOT / "assets" / "scenarios"
    assert assets.docs_dir() == ROOT / "docs"
    # CLI 默认输出仍落仓库 outputs/（既有行为；wheel 模式才改 cwd）
    assert assets.default_output_dir() == ROOT / "outputs"


def test_r10c_wheel_manifest_files_exist_and_exclude_archives():
    from simverse.assets import WHEEL_DATA_MANIFEST, iter_manifest_files
    files = iter_manifest_files(ROOT)
    assert files, "清单不应为空"
    rels = [str(dst).replace("\\", "/") for _, dst in files]
    # 受哈希约束的配置与证据身份必须入包（catalog/证据哈希实际消费）
    assert "assets/config/qwen36_35b_a3b_main__config.json" in rels
    assert any(r.startswith("assets/identity/") for r in rels)
    assert any(r.startswith("docs/rebuild-20261106/references/") for r in rels)
    # 大体积采集/归档数据不进 wheel（按用途留仓库）
    assert not any("cost_records" in r for r in rels)
    assert not any("linear_ops.json" in r or "attn_decode.json" in r
                   for r in rels)
    # profile 证据 manifest 入包（证据登记哈希引用），数据文件不入
    assert "assets/profiles/h200_qwen3_30b_a3b_tiny/manifest.json" in rels
    # 目录条目确实存在于仓库
    for src, _ in files:
        assert src.is_file(), src


def test_r10c_locator_wheel_mode_falls_back_to_packaged_data(tmp_path,
                                                             monkeypatch):
    """wheel 模式（repo_root=None）回退包内 _data；用临时包目录模拟。"""
    from simverse import assets
    pkg = tmp_path / "simverse"
    (pkg / "_data" / "assets" / "config").mkdir(parents=True)
    (pkg / "_data" / "assets" / "config" / "x.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(assets, "_PACKAGE_DIR", pkg.resolve())
    monkeypatch.setattr(assets, "repo_root", lambda: None)
    assert assets.assets_dir() == pkg / "_data" / "assets"
    assert assets.docs_dir() == pkg / "_data" / "docs"
    assert assets.default_output_dir() == Path.cwd() / "outputs"


# ---------------------------------------------------------------------------
# R10d 共享预设（三入口同源）
# ---------------------------------------------------------------------------

def _historical_cli_default() -> dict:
    """修复前 CLI 硬编码的 V2_DEFAULT_SCENARIO（逐键对照锚点）。"""
    return {
        "schema_version": "0.2", "name": "cli-default",
        "model_id": "qwen36_35b_a3b", "draft_id": "dflash_qwen36_35b_a3b",
        "method_id": "dflash_v1", "k": 15, "block_size": 16,
        "draft_ctx_kv_path": "fused_raw",
        "deployment": {"mode": "colocated", "target_device": "h100_sxm",
                       "target_tp": 8, "draft_device": "l40", "draft_tp": 1},
        "workload": {"batch": 1, "ctx_tokens": 32768},
        "acceptance": {"kind": "geometric", "alpha": 0.72},
    }


def test_r10d_preset_matches_historical_cli_default_digest():
    from simverse.presets import DEFAULT_PRESET_ID, default_preset, list_presets
    from simverse.scenario import ScenarioV2
    preset = default_preset()
    assert preset == _historical_cli_default(), "预设必须与原 CLI 默认逐键一致"
    d_preset = ScenarioV2.from_json(dict(preset)).scenario_digest()
    d_old = ScenarioV2.from_json(_historical_cli_default()).scenario_digest()
    assert d_preset == d_old, "场景 digest 应与历史默认 run 可回放对齐"
    ids = [e["id"] for e in list_presets()]
    assert DEFAULT_PRESET_ID in ids
    entry = next(e for e in list_presets() if e["id"] == DEFAULT_PRESET_ID)
    assert entry["scenario_digest"] == d_preset


def test_r10d_catalog_exposes_preset_and_legacy_index_untouched():
    client = app.test_client()
    cat = client.get("/api/v2/catalog").get_json()
    ids = {s["id"] for s in cat["scenarios"]}
    assert "qwen36_dflash_default" in ids
    # legacy v0.1 目录行为保留：v0.2 预设不混入（ScenarioSpec 读不了 0.2）
    legacy = client.get("/api/scenarios").get_json()["scenarios"]
    assert "qwen36_dflash_default" not in legacy
    assert "v41_flash_dt_verification" in legacy  # 旧场景读取行为不变


def test_r10d_same_preset_cli_and_api_same_digest_and_metric():
    """同一份预设走 CLI 与 API：场景 digest 与关键指标一致。"""
    from simverse.pipeline import run_v2
    from simverse.presets import default_preset
    from simverse.scenario import ScenarioV2
    client = app.test_client()
    r = client.post("/api/v2/run", json=default_preset())
    assert r.status_code == 200
    bundle = r.get_json()
    assert bundle["manifest"]["status"] == "completed"
    assert bundle["scenario_snapshot"]
    api_bundle = run_v2(ScenarioV2.from_json(default_preset())).to_json()
    assert bundle["manifest"]["input_digest"] == \
        api_bundle["manifest"]["input_digest"]
    assert bundle["manifest"]["run_id"] == api_bundle["manifest"]["run_id"]
    for key in ("M01_tpot_analytical", "M02_speedup_vs_baseline"):
        assert bundle["metrics"][key]["value"] == \
            pytest.approx(api_bundle["metrics"][key]["value"])


def test_r10d_cli_default_run_v2_executes_with_run_id(tmp_path):
    out = tmp_path / "run.json"
    proc = subprocess.run(
        [sys.executable, "-m", "simverse.cli", "run-v2", "-o", str(out)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(tmp_path), env={**__import__("os").environ,
                                "PYTHONPATH": str(ROOT)})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "status=completed" in proc.stdout and "run_id=" in proc.stdout
    bundle = json.loads(out.read_text(encoding="utf-8"))
    assert bundle["replay"]["scenario_file"] == "preset:qwen36_dflash_default"
    assert bundle["manifest"]["run_id"].startswith("sha256:")
    assert bundle["metrics"]["M01_tpot_analytical"]["value"] > 0


# ---------------------------------------------------------------------------
# R10d run/sweep 共用解码：结构化 4xx + 字段路径
# ---------------------------------------------------------------------------

MALFORMED_CASES = [
    # (名称, body, 期望字段路径子串)
    ("顶层数组", [1, 2, 3], "$"),
    ("顶层字符串", "scenario", "$"),
    ("deployment 数组", {"model_id": "qwen36_35b_a3b",
                         "method_id": "dflash_v1", "deployment": [1]},
     "$.deployment"),
    ("workload 类型非法", {"model_id": "qwen36_35b_a3b",
                           "method_id": "dflash_v1", "workload": "x"},
     "$.workload"),
    ("batch 字符串", {"model_id": "qwen36_35b_a3b", "method_id": "dflash_v1",
                      "workload": {"batch": "1"}}, "$.workload.batch"),
    ("k 越界", {"model_id": "qwen36_35b_a3b", "method_id": "dflash_v1",
                "k": 99}, "$.k"),
    ("alpha 越界", {"model_id": "qwen36_35b_a3b", "method_id": "dflash_v1",
                    "acceptance": {"kind": "geometric", "alpha": 1.5}},
     "$.acceptance.alpha"),
    ("acceptance kind 未知", {"model_id": "qwen36_35b_a3b",
                              "method_id": "dflash_v1",
                              "acceptance": {"kind": "weird"}},
     "$.acceptance.kind"),
    ("deployment 未知键", {"model_id": "qwen36_35b_a3b",
                           "method_id": "dflash_v1",
                           "deployment": {"mode": "colocated", "tp": 8}},
     "$.deployment.tp"),
    ("budget 类型非法", {"model_id": "qwen36_35b_a3b", "method_id": "dflash_v1",
                         "deployment": {"mode": "colocated", "budget": {
                             "allocatable_bytes": "7e10",
                             "allocatable_kind": "hypothesis"}}},
     "$.deployment.budget.allocatable_bytes"),
    ("负块长", {"block_size": -1}, "$.block_size"),
    ("未知设备", {"deployment": {"target_device": "missing"}}, "$.deployment.target_device"),
    ("未知链路", {"deployment": {"link": "missing"}}, "$.deployment.link"),
    ("微批越界", {"deployment": {"mode": "dt_pipeline_sim", "pipeline_microbatches": 0}},
     "$.deployment.pipeline_microbatches"),
    ("流水轮数越界", {"deployment": {"mode": "dt_pipeline_sim", "pipeline_rounds": 0}},
     "$.deployment.pipeline_rounds"),
    ("输出上限越界", {"workload": {"osl_tokens": -1}}, "$.workload.osl_tokens"),
    ("上下文越界", {"workload": {"ctx_tokens": 0}}, "$.workload.ctx_tokens"),
    ("目标 KV 位宽非法", {"quant": {"kv": 3}}, "$.quant.kv"),
    ("草稿 KV 位宽非法", {"draft_quant": {"kv": 3}}, "$.draft_quant.kv"),
    ("草稿预算越界", {"deployment": {"draft_budget": {"allocatable_bytes": -1,
                             "allocatable_kind": "hypothesis"}}},
     "$.deployment.draft_budget.allocatable_bytes"),
    ("预算超过标称容量", {"deployment": {"budget": {"allocatable_bytes": 10**12,
                             "allocatable_kind": "hypothesis"}}},
     "$.deployment.budget.allocatable_bytes"),

]


@pytest.mark.parametrize("name,body,path_prefix", MALFORMED_CASES,
                         ids=[c[0] for c in MALFORMED_CASES])
def test_r10d_run_rejects_malformed_with_field_path(name, body, path_prefix):
    client = app.test_client()
    r = client.post("/api/v2/run", json=body)
    assert r.status_code == 400, f"{name}: {r.status_code}"
    payload = r.get_json()
    assert payload["error"], name
    assert payload["field"] == path_prefix, (name, payload["field"])


@pytest.mark.parametrize("name,body,path_prefix", MALFORMED_CASES,
                         ids=[c[0] for c in MALFORMED_CASES])
def test_r10d_sweep_rejects_malformed_with_field_path(name, body, path_prefix):
    client = app.test_client()
    r = client.post("/api/v2/sweep", json=body)
    assert r.status_code == 400, f"{name}: {r.status_code}"
    payload = r.get_json()
    assert payload["field"] == path_prefix, (name, payload["field"])


def test_r10d_malformed_json_body_returns_structured_400():
    client = app.test_client()
    for url in ("/api/v2/run", "/api/v2/sweep"):
        r = client.post(url, data="{not json",
                        content_type="application/json")
        assert r.status_code == 400
        payload = r.get_json()
        assert payload["field"] == "$" and "JSON" in payload["error"]


def test_r10d_valid_scenarios_still_run_on_both_endpoints():
    client = app.test_client()
    body = {"model_id": "qwen36_35b_a3b", "method_id": "dflash_v1",
            "k": 4, "workload": {"ctx_tokens": 8192}}
    r = client.post("/api/v2/run", json=body)
    assert r.status_code == 200
    assert r.get_json()["manifest"]["status"] == "completed"
    explicit_defaults = {**body, "quant": {}, "draft_quant": {}}
    same = client.post("/api/v2/run", json=explicit_defaults)
    assert same.status_code == 200
    assert same.get_json()["manifest"]["input_digest"] == r.get_json()["manifest"]["input_digest"]
    scan = client.post("/api/v2/sweep", json=explicit_defaults)
    assert scan.status_code == 200
    assert scan.get_json()["base_digest"] == r.get_json()["manifest"]["input_digest"]


def test_r10d_unsupported_combo_keeps_result_status_not_4xx():
    """合法但不受支持的组合：沿用结果状态与能力说明（200 + failed）。

    MTP+dt 通过场景校验（拦截在管线层），返回 unsupported_combo bundle；
    对照：V4.1+dflash 是 validate 拒绝的非法字段值（quant 配方冲突）→ 4xx。
    """
    client = app.test_client()
    r = client.post("/api/v2/run", json={
        "model_id": "qwen36_35b_a3b", "method_id": "mtp",
        "deployment": {"mode": "dt_serial"}})
    assert r.status_code == 200
    bundle = r.get_json()
    assert bundle["manifest"]["status"].startswith("unsupported") or \
        bundle["manifest"]["status"] == "failed"
    assert "MTP" in json.dumps(bundle, ensure_ascii=False)
    # 域非法（V4.1 冻结量化配方冲突）是字段值非法 → 4xx 带路径
    r2 = client.post("/api/v2/run", json={
        "model_id": "deepseek_v41_flash", "method_id": "dflash_v1", "k": 4})
    assert r2.status_code == 400
    assert r2.get_json()["field"].startswith("$.")


def test_r10d_internal_error_keeps_server_error_semantics(monkeypatch):
    """入参异常与内部 ValueError/KeyError/TypeError 分开处理。"""
    import simverse.pipeline as pipeline_mod
    import simverse.experiments.sweep_v2 as sweep_mod

    def fail(error):
        def broken(*args, **kwargs):
            raise error("内部缺陷")
        return broken

    monkeypatch.setitem(app.config, "PROPAGATE_EXCEPTIONS", False)
    for error in (ValueError, KeyError, TypeError):
        monkeypatch.setattr(pipeline_mod, "run_v2", fail(error))
        monkeypatch.setattr(sweep_mod, "run_v2", fail(error))
        with app.test_client() as client:
            for url in ("/api/v2/run", "/api/v2/sweep"):
                r = client.post(url, json={})
                assert r.status_code == 500, (url, error)


# ---------------------------------------------------------------------------
# R10b 根目录文档白名单
# ---------------------------------------------------------------------------

def test_r10b_root_docs_whitelisted_and_others_404():
    client = app.test_client()
    for fn in ("ROADMAP.md", "AGENTS.md"):
        r = client.get(f"/{fn}")
        assert r.status_code == 200, fn
        assert "text/html" in r.headers["Content-Type"]
        body = r.get_data(as_text=True)
        assert "simverse" in body.lower() or "ROADMAP" in body or fn == "AGENTS.md"
    # 白名单之外的根路径与不存在的文档仍 404
    for url in ("/README.md", "/NOPE.md", "/pyproject.toml",
                "/docs/no-such-doc.md"):
        assert client.get(url).status_code == 404, url
    # 穿越拒绝
    assert client.get("/docs/../pyproject.toml").status_code == 404


def test_r10b_roadmap_links_resolve_and_status_pages_open():
    """状态页 → Roadmap → Roadmap 指向的状态/验证文档逐级可打开。"""
    client = app.test_client()
    # 状态文档本身可打开（/docs/<fn> 路由：fn 不带 docs/ 前缀）
    for fn in ("FOCUS_STATUS.md", "IMPLEMENTATION_STATUS.md",
               "ascend-validation-plan.md"):
        r = client.get(f"/docs/{fn}")
        assert r.status_code == 200, fn
        assert 'href="../ROADMAP.md"' in r.get_data(as_text=True), fn
    roadmap = client.get("/ROADMAP.md")
    assert roadmap.status_code == 200
    text = roadmap.get_data(as_text=True)
    # ROADMAP 指向的入口文档均可打开（合入要求与状态入口段）；
    # 渲染后的相对链接保持相对（从 /ROADMAP.md 解析到 /docs/...）
    for target in ("docs/MERGE_CRITERIA.md", "docs/FOCUS_STATUS.md",
                   "docs/IMPLEMENTATION_STATUS.md",
                   "docs/ascend-validation-plan.md"):
        assert f'href="{target}"' in text, target
        assert client.get(f"/{target}").status_code == 200, target


def test_r10b_workbench_html_has_preset_selector_and_doc_links():
    html = (ROOT / "simverse/static/workbench.html").read_text(encoding="utf-8")
    assert 'id="sel-preset"' in html
    js = (ROOT / "simverse/static/workbench.js").read_text(encoding="utf-8")
    # 预设原样提交（digest 对齐）与用户改动守卫的关键逻辑
    assert "if (state.preset)" in js
    assert "clearPreset()" in js and "userTouched" in js
    assert "selectPreset('qwen36_dflash_default')" in js
    assert "$('sel-preset').value = hit.id" in js
    assert 'href="/docs/FOCUS_STATUS.md"' in html
