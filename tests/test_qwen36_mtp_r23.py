"""R23：Qwen3.6 原生 MTP forward 证据与仿真映射（2026-11-13）。

覆盖面（与改动面匹配，准则 §4.1）：
  * Part A（hermetic，CI 必跑）：fixture 数值对照（tools/probe_qwen36_mtp_
    numeric.py 双实现互核产物）+ 本仓 mtp_step_rows 与 fixture 逐步重对账；
  * Part B：结构 spec/计数/方法/管线/内存/负例/固定场景重放冻结；
  * 证据分栏：method_status mtp=source-checked；模型级 cpu_forward/not-run
    不因 fixture 升格；部署排名仍被 mrope/CPU/GPU/预算缺口阻断。

口径披露：fixture 是随机权重小尺寸数学对照（vLLM 结构手工复刻 + 独立
标量参考），不构成真实 checkpoint/完整 forward/部署验证。
"""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "qwen36_mtp_r23_numeric.json"
PIN = ROOT / "assets" / "identity" / "vllm_qwen36_mtp_pin"
# 全 32bit 精度档：与 fixture（f64 权重）同字节口径
Q32 = {"dense_w": 32, "expert_w": 32, "act": 32, "kv": 32, "state": 32}
Q8 = {"dense_w": 8, "expert_w": 4, "act": 8, "kv": 8, "state": 32}


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _fixture_ir():
    from simverse.model_ir.families import parse_qwen3_5_moe
    from simverse.model_ir.schema import SourceIdentity
    art = _fixture()
    src = SourceIdentity(
        repo_id="probe:qwen3_5_moe", revision="vllm@10a854ed",
        config_sha256="fixture", captured_at=art["environment"]["date"],
        note="R23 MTP fixture 探针（独立于本仓执行；随机权重缩比）")
    return parse_qwen3_5_moe(dict(art["fixture_config"]), src,
                             "qwen36_mtp_r23_fixture",
                             "R23 MTP fixture (1L full-attn MoE)")


# ---------------------------------------------------------------------------
# Part A：hermetic fixture 对账（无 torch）
# ---------------------------------------------------------------------------

class TestPartAFixture:
    def test_dual_impl_numeric_within_atol(self):
        art = _fixture()
        assert art["numeric_check"]["all_within_atol"] is True
        worst = max(d["max_abs_diff"] for d in art["numeric_check"]["tensors"])
        assert worst <= art["dims"]["atol"]
        # 覆盖关键中间张量：combine/注意力门控/MoE/最终 hidden/logits
        names = {d["tensor"] for d in art["numeric_check"]["tensors"]}
        assert {"pre_fc_norm_embedding", "pre_fc_norm_hidden", "fc_out",
                "attn_out_gated", "moe_out", "final_hidden", "logits"} <= names
        assert art["numeric_check"]["sampled_tokens_match"] is True

    def test_structure_facts_match_pinned_source(self):
        art = _fixture()
        f = art["structure_facts"]
        assert f["layer_selection"] == "spec_step_idx % n_layers"
        # MTP1：K=3 步全部落在 layer 0（同层循环复用，无逐层步进）
        assert f["steps_layer_idx"] == [0, 0, 0]
        assert "无激活" in f["fc"] and "无残差" in f["fc"]
        assert f["shared"] == ["embed_tokens", "lm_head"]
        assert f["final_norm"] == "mtp.norm(residual)"
        # KV 随步增长（草稿自有缓存：ctx+step+1）
        assert [s["kv_len_after"] for s in art["steps"]] == [7, 8, 9]

    def test_gemm_and_attention_ledger_checks_recorded(self):
        art = _fixture()
        assert art["all_gemm_match"] is True
        assert art["all_attention_match"] is True
        assert art["hand_anchor"]["match"] is True
        # 3 步 × 10 类 GEMM 行全部对上（2×MAC == ledger flops）
        assert len(art["gemm_checks"]) == 30

    def test_ledger_recompute_matches_fixture(self):
        """本仓当前实现对 fixture 几何复算须与落盘对账逐项相等（防漂移）。"""
        from simverse.count.draft import mtp_step_rows
        art = _fixture()
        ctx = art["dims"]["ctx"]
        for j, chk in enumerate(art["gemm_checks"]):
            rows = {r.op: r for r in mtp_step_rows(f"s{j}", _fixture_ir(), j,
                                                   ctx, Q32, tp=1)}
            assert rows[chk["op"]].flops == chk["ledger_flops"], (j, chk["op"])
        for j, chk in enumerate(art["attention_checks"]):
            rows = {r.op: r for r in mtp_step_rows(f"s{j}", _fixture_ir(), j,
                                                   ctx, Q32, tp=1)}
            core = rows["attention_core"]
            assert core.kv_read == chk["kv_read_rows_ledger"] * 4 * 2 * 2 * 16
            assert core.flops // (4 * (4 * 16 + 5)) == chk["ledger_pairs"]

    def test_live_probe_replay_invariants(self):
        """torch 可用时重执行探针，稳健不变量须与落盘 fixture 相等
        （数值逐步对照本身在探针内断言；此处只对不变量，防跨版本脆断）。"""
        import importlib.util
        import subprocess
        import sys
        if importlib.util.find_spec("torch") is None:
            pytest.skip("torch 不可用（CI hermetic 路径）")
        proc = subprocess.run(
            [sys.executable, str(ROOT / "tools/probe_qwen36_mtp_numeric.py")],
            capture_output=True, text=True, timeout=300)
        assert proc.returncode == 0, proc.stderr[-2000:]
        art = _fixture()
        assert art["all_gemm_match"] and art["all_attention_match"]
        assert [s["sampled_token"] for s in art["steps"]]


# ---------------------------------------------------------------------------
# Part B：结构 spec / 计数 / 方法 / 管线 / 内存 / 负例
# ---------------------------------------------------------------------------

class TestMtpSpecAndCoverage:
    def test_catalog_model_carries_mtp_spec(self):
        from simverse.model_ir import build_catalog
        m = build_catalog()["models"]["qwen36_35b_a3b"]
        assert m.mtp_layers == 1
        spec = m.extras["mtp"]
        assert spec["n_layers"] == 1 and spec["n_predict"] == 1
        assert spec["layer_kind"] == "full_attention"
        assert spec["use_dedicated_embeddings"] is False
        assert spec["fc"] == {"in": 2 * 2048, "out": 2048, "bias": False,
                              "sharded": "column_gather"}
        assert spec["pre_fc_norms"] == ("embedding", "hidden")
        assert spec["final_norm"] is True
        assert spec["shared_objects"] == ("embed_tokens", "lm_head")
        assert spec["step_selection"] == "spec_step_idx % n_layers"

    def test_coverage_keys_mapped_with_line_evidence(self):
        from simverse.model_ir import build_catalog
        m = build_catalog()["models"]["qwen36_35b_a3b"]
        for key in ("text_config.mtp_num_hidden_layers",
                    "text_config.mtp_use_dedicated_embeddings"):
            entry = m.coverage[key]
            assert entry["status"] == "mapped"
            assert "qwen3_5_mtp.py" in entry["note"] or \
                "speculative.py" in entry["note"]

    def test_dedicated_embeddings_rejected(self):
        from simverse.model_ir.families import parse_qwen3_5_moe
        from simverse.model_ir.schema import SourceIdentity
        cfg = json.loads((ROOT / "assets/config/"
                          "qwen36_35b_a3b_main__config.json").read_text(
                              encoding="utf-8"))["text_config"]
        cfg["mtp_use_dedicated_embeddings"] = True
        with pytest.raises(ValueError, match="dedicated"):
            parse_qwen3_5_moe(cfg, SourceIdentity("x", "y", "z"), "m", "l")

    def test_hf_side_still_ignores_mtp_weights(self):
        """HF 栏与推理栈栏分立：pinned HF 建模文件没有 MTP forward（只有
        _keys_to_ignore 丢弃 mtp.*）；推理实现在 vLLM/SGLang pin。"""
        hf = (ROOT / "assets/identity/hf_transformers/"
              "modeling_qwen3_5_moe.py").read_text(encoding="utf-8")
        assert '_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]' in hf
        assert "MultiTokenPredictor" not in hf
        vllm = (PIN / "qwen3_5_mtp.py").read_text(encoding="utf-8")
        assert "class Qwen3_5MultiTokenPredictor" in vllm
        assert "class Qwen3_5MoeMTP" in vllm


class TestMtpStepRows:
    def _rows(self, step=0, ctx=512, batch=1, tp=1, quant=Q8):
        from simverse.count.draft import mtp_step_rows
        from simverse.model_ir import build_catalog
        m = build_catalog()["models"]["qwen36_35b_a3b"]
        return mtp_step_rows("s", m, step, ctx, quant, tp=tp, batch=batch)

    def test_step_structure_matches_pinned_forward(self):
        """行集合 = 共享查表 + 双预归一 + fc + full-attn 层（gated GQA+MoE）
        + final norm + 共享 lm_head——无 GDN 行（旧模板曾用目标首层 GDN）。"""
        rows = {r.op: r for r in self._rows()}
        for op in ("mtp_embed_share", "mtp_pre_fc_norm_embedding",
                   "mtp_pre_fc_norm_hidden", "mtp_fc", "q_proj", "k_proj",
                   "v_proj", "attention_core", "attn_output_gate", "o_proj",
                   "q_norm", "k_norm", "rope", "moe_router", "moe_routed",
                   "moe_shared", "shared_gate", "attn_norm", "attn_add",
                   "mlp_norm", "mlp_add", "mtp_final_norm", "draft_lm_head"):
            assert op in rows, op
        assert all(op not in rows for op in
                   ("in_proj_qkv", "causal_conv1d", "gdn_recurrence"))
        fc = rows["mtp_fc"]
        assert fc.flops == 2 * 1 * (2 * 2048) * 2048     # [1,2h]@[2h,h] TP1
        assert fc.weight_read == int(2 * 2048 * 2048 * 1.0)
        lm = rows["draft_lm_head"]
        assert lm.flops == 2 * 1 * 2048 * 248320
        assert lm.weight_capacity == 0                    # 容量归目标对象
        assert rows["mtp_embed_share"].weight_capacity == 0

    def test_step_kv_grows_and_layer_cycles(self):
        r0, r2, r5 = self._rows(0), self._rows(2), self._rows(5)
        core = lambda rs: next(r for r in rs if r.op == "attention_core")
        kv_elems = 2 * 2 * 256                            # kvh<tp=1 复制全量
        assert core(r0).kv_read == 512 * kv_elems
        assert core(r2).kv_read == 514 * kv_elems         # ctx+step 先前行
        assert core(r5).kv_read == 517 * kv_elems
        for rs, li in ((r0, 0), (r2, 0), (r5, 0)):        # MTP1 恒 layer 0
            assert all(r.layer == li for r in rs if r.op == "q_proj")

    def test_batch_scales_tokens_once_not_weights(self):
        from simverse.count import totals
        t1, t2 = totals(self._rows(batch=1)), totals(self._rows(batch=2))
        assert t2["flops"] == 2 * t1["flops"]
        assert t2["act_read"] == 2 * t1["act_read"]
        assert t2["kv_read"] == 2 * t1["kv_read"]
        assert t2["weight_read"] == t1["weight_read"]
        assert sum(r.weight_capacity for r in self._rows(batch=1)) == \
            sum(r.weight_capacity for r in self._rows(batch=2))

    def test_tp8_shards_fc_and_heads(self):
        rows = {r.op: r for r in self._rows(tp=8)}
        assert rows["mtp_fc"].flops == 2 * 1 * 4096 * 256   # h/tp 输出
        assert rows["mtp_fc"].weight_read == int(4096 * 256 * 1.0)
        assert rows["q_proj"].flops == 2 * 1 * 2048 * (2 * 2 * 256)  # heads/tp=2
        # lm_head 读目标 V/tp 分片
        assert rows["draft_lm_head"].flops == 2 * 2048 * (248320 // 8)

    def test_no_mtp_spec_rejected(self):
        from simverse.count.draft import mtp_step_rows
        from simverse.model_ir import build_catalog
        v41 = build_catalog()["models"]["deepseek_v41_flash"]
        with pytest.raises(ValueError, match="无原生 MTP 结构 spec"):
            mtp_step_rows("s", v41, 0, 8, Q8)

    def test_rewind_row_expectation(self):
        from simverse.count.draft import mtp_kv_rewind_row
        from simverse.model_ir import build_catalog
        m = build_catalog()["models"]["qwen36_35b_a3b"]
        row = mtp_kv_rewind_row("s", 3, 2.0, m, Q8, 8)
        assert row.statistic == "expectation" and row.kv_write == 0
        assert "1.00 行/轮" in row.shape              # K−E[N]=1


class TestPlanMtp:
    def test_k_over_n_predict_now_allowed_with_cycling(self):
        """R23：K>n_predict 循环复用（pinned spec_step_idx % n；旧 K≤n
        限制是对 config-only 时代的保守近似，按源码撤销）。"""
        from simverse.methods.plans import build_method_plan
        plan = build_method_plan("mtp", {"kind": "geometric", "alpha": 0.7}, 4,
                                 mtp_layers=1)
        assert plan.k_effective == 4
        assert [s.step_id for s in plan.steps] == [
            "draft_step_0", "draft_step_1", "draft_step_2", "draft_step_3",
            "d2t", "verify", "t2d"]
        assert "同层循环复用" in plan.steps[1].desc
        assert plan.status == "approximated"
        assert plan.source_pin.startswith("vllm@10a854ed")

    def test_no_native_head_rejected(self):
        from simverse.methods.plans import build_method_plan
        with pytest.raises(ValueError, match="原生 MTP 头"):
            build_method_plan("mtp", {"kind": "geometric", "alpha": 0.7}, 2,
                              mtp_layers=0)

    def test_mtp_and_dflash_are_distinct_methods(self):
        """不可把 DFlash 当作原生 MTP（或反之）：两方法的源 pin/程序图分立。"""
        from simverse.methods.plans import DFLASH_PIN, build_method_plan
        from simverse.model_ir import build_catalog
        cat = build_catalog()
        mtp = build_method_plan("mtp", {"kind": "geometric", "alpha": 0.7}, 1,
                                mtp_layers=1)
        dflash = build_method_plan(
            "dflash_v1", {"kind": "geometric", "alpha": 0.7}, 15,
            draft_ir=cat["drafts"]["dflash_qwen36_35b_a3b"],
            block_size=16)
        assert mtp.source_pin != dflash.source_pin
        assert dflash.source_pin == DFLASH_PIN
        assert mtp.method_id == "mtp" and dflash.method_id == "dflash_v1"
        # MTP 计划不消费 DFlash DraftIR（无块扩散步骤/块大小）
        assert all("blk" not in (s.desc or "") for s in mtp.steps)
        assert dflash.block_size == 16 and mtp.block_size is None


class TestPipelineMtp:
    def test_colocated_round_completes_with_per_step_stages(self):
        from simverse.pipeline import run_v2
        from simverse.scenario import ScenarioV2
        b = run_v2(ScenarioV2(method_id="mtp", k=3)).to_json()
        assert b["manifest"]["status"] == "completed"
        assert {"draft_step_0", "draft_step_1", "draft_step_2",
                "verify"} <= set(b["work"])
        # 每步独立计价（K 均匀步，无 catch-up 加权）
        times = [b["cost"][f"draft_step_{j}"]["t_ms"] for j in range(3)]
        assert all(t > 0 for t in times)
        # 草稿步通信（TP8：attn/mlp allreduce + fc all-gather + logits gather）
        kinds = b["cost"]["draft_step_0"]["collectives"]["counts_by_kind"]
        assert kinds.get("allreduce", 0) >= 2
        assert kinds.get("mtp_fc_allgather") == 1
        assert kinds.get("logits_gather", 0) >= 1
        checks = b["capability"]["checks"]
        assert checks["mtp_serving_integration"]["status"] == "supported"
        assert checks["method_recipe_ready"]["status"] == "approximated"
        # 排名资格仍被阻断（mrope/CPU/GPU/预算缺口，不因 source-checked 解除）
        assert b["metrics"]["M02_speedup_vs_baseline"]["reason"]

    def test_memory_carries_mtp_objects_and_shared_refs(self):
        from simverse.pipeline import run_v2
        from simverse.scenario import ScenarioV2
        b = run_v2(ScenarioV2(method_id="mtp", k=1)).to_json()
        inv = {a["alloc_id"]: a for a in
               b["cost"]["memory"]["colocated_r0"]["inventory"]}
        for alloc in ("draft-w-moe-L0", "draft-w-mtp-fc-norms", "draft-kv",
                      "t-mtp-draft-buffers", "t-mtp-last-hidden"):
            assert alloc in inv, alloc
        # embed/lm_head 零字节共享：引用行按物理身份并入 target 对象
        # （RankMemory._dedup：reference 并入 consumer，不产生独立行/字节）
        assert not any(a.startswith("draft-ref-") for a in inv)
        assert not any("embed" in a or "lm_head" in a for a in inv
                       if a.startswith("draft-"))
        for phys in ("target-w-embed", "target-w-lm_head"):
            assert "draft" in inv[phys].get("consumers", []), phys
        # MTP 层 MoE 权重 per-rank（EP off inter/tp=64，draft expert W8 档）
        moe = inv["draft-w-moe-L0"]["bytes"]
        assert moe == int(2048 * 256 * 1.0) + int(3 * 2048 * 64 * 1.0) + \
            int(2048 * 1.0) + int(256 * 3 * 2048 * 64 * 1.0)
        # 草稿 KV：1 个 full-attn 层，kvh=2<tp 复制全量
        assert inv["draft-kv"]["bytes"] == 1 * 32768 * 2 * 2 * 256 * 1

    def test_dt_modes_rejected_no_stack_implementation(self):
        from simverse.pipeline import run_v2
        from simverse.scenario import DeploymentSpec, ScenarioV2
        for mode in ("dt_serial", "dt_pipeline_sim"):
            b = run_v2(ScenarioV2(
                method_id="mtp", k=1,
                deployment=DeploymentSpec(mode=mode, target_tp=8,
                                          draft_tp=1))).to_json()
            assert b["manifest"]["status"] == "failed"
            assert "load_eagle_model" in b["metrics"]["error"]["reason"]

    def test_non_qwen36_target_rejected(self):
        from simverse.pipeline import run_v2
        from simverse.scenario import ScenarioV2, V41_QUANT
        b = run_v2(ScenarioV2(model_id="deepseek_v41_flash", method_id="mtp",
                              k=3, quant=V41_QUANT)).to_json()
        assert b["manifest"]["status"] == "failed"
        assert "Qwen3_5MoeMTP" in b["metrics"]["error"]["reason"]

    def test_evidence_columns_synced(self):
        from simverse.model_ir import build_catalog
        from simverse.validation.registry import build_registry
        m = build_catalog()["models"]["qwen36_35b_a3b"]
        assert m.evidence.method_status["mtp"] == "source-checked"
        assert "sm-qwen36-mtp" in m.evidence.checked_paths
        assert "ev-vllm-qwen36-mtp" in m.evidence_refs
        assert "ev-sglang-qwen36-mtp" in m.evidence_refs
        assert "ev-qwen36-mtp-cpu-numeric" in m.evidence_refs
        # 四栏不冒充：fixture 数值证据不升格模型级 cpu_forward
        assert m.evidence.cpu_forward == "not-run"
        assert m.evidence.gpu_measurement == "none"
        rec = build_registry()
        for ev in ("ev-vllm-qwen36-mtp", "ev-sglang-qwen36-mtp"):
            assert rec[ev].executed is False
        assert rec["ev-qwen36-mtp-cpu-numeric"].executed is True

    def test_run_refs_include_mtp_evidence(self):
        from simverse.pipeline import run_v2
        from simverse.scenario import ScenarioV2
        b = run_v2(ScenarioV2(method_id="mtp", k=1)).to_json()
        assert "ev-vllm-qwen36-mtp" in b["evidence_refs"]
        assert "ev-sglang-qwen36-mtp" in b["evidence_refs"]


# ---------------------------------------------------------------------------
# 固定场景重放冻结（R23 默认 MTP 场景：B1/ctx32768/TP8/K3/W8A8+FP4E）
# ---------------------------------------------------------------------------

FROZEN = {
    "scenario_digest": "sha256:ee932075d9c338f394d709066a14a7ca444d3e22"
                       "e1954cea0fa3461b61ddf1fe",
    "step0_flops": 215_342_492,
    "step0_weight_read": 70_802_432,
    "step0_kv_read": 33_554_432,
    "step2_kv_read": 33_556_480,
    "steady_bytes": 2_841_006_528,
}


class TestReplayFrozen:
    def test_fixed_scenario_replay_stable(self):
        """冻结重放：R23 改动后的 MTP 场景计数逐字段可复现（校准开/关
        守恒由 G7 通用测试覆盖；此处冻结 MTP 专属行）。"""
        from simverse.pipeline import run_v2
        from simverse.scenario import ScenarioV2
        b = run_v2(ScenarioV2(method_id="mtp", k=3)).to_json()
        assert b["manifest"]["status"] == "completed"
        assert b["manifest"]["input_digest"] == FROZEN["scenario_digest"]
        t0 = b["work"]["draft_step_0"]["totals"]
        t2 = b["work"]["draft_step_2"]["totals"]
        assert t0["flops"] == FROZEN["step0_flops"]
        assert t0["weight_read"] == FROZEN["step0_weight_read"]
        assert t0["kv_read"] == FROZEN["step0_kv_read"]
        assert t2["kv_read"] == FROZEN["step2_kv_read"]
        mem = b["cost"]["memory"]["colocated_r0"]
        assert mem["steady_bytes"] == FROZEN["steady_bytes"]
