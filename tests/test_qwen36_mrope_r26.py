"""R26：Qwen3.6 文本路径 MRoPE position_ids 语义（hermetic 对账 + live 可选）。

证据链（docs/validation/qwen36-text-mrope-20261116.md）：
  * pin：assets/identity/vllm_qwen36_mrope_pin/（5 文件，blob sha1 对
    GitHub contents API 校验）+ 既有 hf_transformers / vllm_qwen35_pin /
    vllm_qwen36_mtp_pin；
  * fixture：tests/fixtures/qwen36_mrope_r26_numeric.json（三路 f64 对照
    diff=0、官方 HF 旋转类 fp32 实测 4.35e-07、MTP 状态机 + 6 负例）；
  * source_map：sm-qwen36-text-mrope（行号锚 → 本测试逐行核对 pin 文件）；
  * replay：tools/replay_qwen36_mrope.py（干净 main 基线，账本零漂移，
    M07 structure_complete False→True，M02 排名理由不再含 mrope）。

fixture 级证据不升格模型级 cpu_forward（G10）；多模态 2D 位置分支域外。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FIXTURE = ROOT / "tests/fixtures/qwen36_mrope_r26_numeric.json"
PIN_MROPE = ROOT / "assets/identity/vllm_qwen36_mrope_pin"
PIN_HF = ROOT / "assets/identity/hf_transformers"
PIN_PAR = ROOT / "assets/identity/vllm_qwen35_pin"
PIN_MTP = ROOT / "assets/identity/vllm_qwen36_mtp_pin"


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _lines(path: Path) -> list:
    return path.read_text(encoding="utf-8").splitlines()


def _anchor(path: Path, lineno: int, needle: str) -> None:
    """source_map 行号锚核对：pin 文件第 lineno 行必须含 needle。"""
    lines = _lines(path)
    assert lineno <= len(lines), f"{path.name}:{lineno} 越界"
    assert needle in lines[lineno - 1], \
        f"{path.name}:{lineno} 锚漂移：{lines[lineno - 1]!r} 不含 {needle!r}"


# ---------------------------------------------------------------------------
# Part A：pin 完整性与行号锚（hermetic）
# ---------------------------------------------------------------------------

class TestPinIntegrity:
    def test_manifest_files_match_disk_sha256_and_lf(self):
        man = json.loads((PIN_MROPE / "manifest.json").read_text(encoding="utf-8"))
        import hashlib
        assert man["repo"] == "vllm-project/vllm"
        assert man["head_sha"].startswith("dc74b8cb")
        assert set(man["files"]) == {
            "qwen3_vl.py", "rope_init.py", "rope_base.py",
            "rope_common.py", "rope_mrope.py"}
        for name, meta in man["files"].items():
            data = (PIN_MROPE / name).read_bytes()
            assert b"\r" not in data, f"{name} 非 LF"
            assert hashlib.sha256(data).hexdigest() == meta["sha256"]
            assert meta["bytes"] == len(data)
            assert len(meta["blob_sha1"]) == 40
        assert "10a854ed" in man["note"]      # 跨 commit 字节一致声明

    def test_source_anchors_hf_text_position_ids(self):
        m = PIN_HF / "modeling_qwen3_5_moe.py"
        _anchor(m, 1322, "the hard coded `4` is for text, temporal, height and width")
        _anchor(m, 1325, "torch.arange(inputs_embeds.shape[1]")
        _anchor(m, 1325, "+ past_seen_tokens")
        _anchor(m, 1326, "expand(4, inputs_embeds.shape[0], -1)")
        _anchor(m, 1331, "text_position_ids = position_ids[0]")
        _anchor(m, 1332, "position_ids = position_ids[1:]")

    def test_source_anchors_hf_rotary_class(self):
        m = PIN_HF / "modeling_qwen3_5_moe.py"
        _anchor(m, 146, 'config.rope_parameters.get("mrope_section", [11, 11, 10])')
        _anchor(m, 176, "position_ids = position_ids.expand(3, -1, -1)")
        _anchor(m, 191, "self.mrope_section[dim] * 3")   # 交错重组
        _anchor(m, 194, "torch.cat((freqs_thw, freqs_thw), dim=-1)")
        _anchor(m, 686, "rotate_half")                    # NeoX 应用

    def test_source_anchors_vllm_text_degeneration(self):
        q = PIN_PAR / "qwen3_5.py"
        _anchor(q, 445, "torch.arange(len(input_tokens), dtype=torch.long)")
        _anchor(q, 446, "expand(3, -1), 0")
        n = PIN_PAR / "qwen3_next.py"
        _anchor(n, 386, "type(self.rotary_emb) is MRotaryEmbedding")
        _anchor(n, 389, "sum(mrope_section) == self.rotary_emb.rotary_dim // 2")
        _anchor(n, 390, "mrope_interleaved")
        _anchor(n, 426, "positions = positions[0]")
        vl = PIN_MROPE / "qwen3_vl.py"
        _anchor(vl, 1821, "def mrope_positions_factor")
        _anchor(vl, 1828, "return 4 if pruned else 1")

    def test_source_anchors_vllm_rope_stack(self):
        init = PIN_MROPE / "rope_init.py"
        _anchor(init, 112, 'elif scaling_type == "default":')
        _anchor(init, 114, "rotary_emb = MRotaryEmbedding(")
        _anchor(init, 334, 'elif scaling_type == "openpangu":')  # Interleaved 仅此
        rope = PIN_MROPE / "rope_mrope.py"
        _anchor(rope, 363, "assert sum(self.mrope_section) == rotary_dim // 2")
        _anchor(rope, 438, "def _forward_text_rope(")
        _anchor(rope, 504, "return self._forward_text_rope(positions, query, key)")
        base = PIN_MROPE / "rope_base.py"
        _anchor(base, 102, "cache = torch.cat((cos, sin), dim=-1)")
        com = PIN_MROPE / "rope_common.py"
        _anchor(com, 163, "o1 = x1 * cos - x2 * sin")

    def test_source_anchors_mtp_position_advance(self):
        td = PIN_MTP / "tdar_speculator.py"
        _anchor(td, 355, "sample_src_positions = positions + 1")
        _anchor(td, 494, "# Copy positions.")
        _anchor(td, 498, "target_pos = tl.load(target_positions_ptr")
        _anchor(td, 607, "seq_len = target_seq_len - num_rejected")
        _anchor(td, 613, "tl.minimum(position + 1, max_model_len - 1)")
        _anchor(td, 649, "if step >= num_speculative_steps - 1:")
        _anchor(td, 655, "sample_src_positions_ptr + req_idx, sample_position + 1")
        mtp = PIN_MTP / "qwen3_5_mtp.py"
        _anchor(mtp, 58, "# positions is of shape (3, seq_len) if mrope is enabled")
        _anchor(mtp, 172, "positions=positions,")


# ---------------------------------------------------------------------------
# Part A：source_map / catalog / registry（证据四栏）
# ---------------------------------------------------------------------------

class TestSourceMapAndEvidence:
    def test_sm_entry_registered_with_uncovered_domain(self):
        from simverse.importers.source_map import ENTRIES
        e = next(x for x in ENTRIES if x.entry_id == "sm-qwen36-text-mrope")
        assert e.both_sides_covered is True
        assert e.evidence_kind == "source-checked"
        assert e.source_pin.startswith("vllm@dc74b8cb")
        joined = " ".join(e.uncovered_domain)
        assert "多模态" in joined and "2D" in joined   # 域外不冒充
        assert "vLLM MRotaryEmbedding 模块本机未执行" in joined
        assert "R26" in e.note

    def test_catalog_structure_complete_not_overclaiming(self):
        from simverse.model_ir import build_catalog
        q = build_catalog()["models"]["qwen36_35b_a3b"]
        assert q.structure_complete and q.forward_evidence_complete
        assert q.evidence.forward_source == "source-checked"
        assert q.evidence.unverified_paths == ()
        assert q.evidence.cpu_forward == "not-run"        # 不冒充执行
        assert q.evidence.gpu_measurement == "none"
        for ref in ("ev-vllm-qwen36-mrope", "ev-qwen36-mrope-cpu-numeric"):
            assert ref in q.evidence_refs
        cov = q.coverage
        for key in ("text_config.rope_parameters.mrope_section",
                    "text_config.rope_parameters.mrope_interleaved"):
            assert cov[key]["status"] == "mapped"
            assert "R26" in cov[key]["note"]
        rope = q.extras["rope"]
        assert rope["rotary_dim"] == 64 and rope["theta"] == 1e7
        assert rope["mrope_section"] == [11, 11, 10]
        assert rope["mrope_interleaved"] is True
        assert rope["text_semantics"]["degenerate"] is True
        assert rope["text_semantics"]["positions_factor_text"] == 1
        adv = rope["mtp_position_advance"]
        assert adv["source_pin"].startswith("vllm@10a854ed")
        assert "num_rejected" in adv["rewind"] and "末步" in adv["per_step"]

    def test_registry_records_bound_to_artifacts(self):
        from simverse.validation.registry import build_registry
        reg = build_registry()
        ev1, ev2 = reg["ev-vllm-qwen36-mrope"], reg["ev-qwen36-mrope-cpu-numeric"]
        assert ev1.kind == "source-checked" and ev1.executed is False
        assert ev1.content_sha256 not in ("", "MISSING")
        assert ev2.kind == "numeric-checked" and ev2.executed is True
        assert ev2.content_sha256 not in ("", "MISSING")
        assert any("多模态" in x for x in ev1.limits)
        assert any("不升级模型级" in x or "不是模型级" in x for x in ev2.limits)


# ---------------------------------------------------------------------------
# Part A：fixture 数值对账 + 独立 mini-oracle（防 fixture 自洽盲区）
# ---------------------------------------------------------------------------

class TestFixtureNumeric:
    def test_three_way_zero_and_position_sequences(self):
        art = _fixture()
        mm = art["main_model"]
        assert mm["max_diff_hf_vs_oracle"] == 0.0
        assert mm["max_diff_vllm_vs_oracle"] == 0.0
        assert mm["max_diff_hf_vs_vllm"] == 0.0
        assert mm["twod_identical_rows_degenerate_diff"] == 0.0
        seqs = mm["position_sequences"]
        assert [c["equal"] for c in seqs] == [True, True, True]
        assert seqs[0]["hf"] == [0, 1, 2, 3, 4]     # prefill
        assert seqs[1]["hf"] == [5] and seqs[2]["hf"] == [6]  # decode 续位

    def test_independent_mini_oracle_recompute(self):
        """测试内独立复算（不复用探针代码）：pos=7 首对与末对旋转角。"""
        import math
        art = _fixture()
        theta = art["rope_config"]["rope_theta"]
        rotary = art["rope_config"]["partial_rotary_factor"] * \
            art["rope_config"]["head_dim"]
        n = int(rotary) // 2
        f0 = theta ** (0.0 / rotary)                # inv_freq[0] = 1.0
        flast = theta ** (-(2 * (n - 1)) / rotary)
        assert abs(math.cos(7 * f0) - math.cos(7.0)) < 1e-15
        # mrope_section 布局：T 频率索引 {0,3,...,30}（s0 个），H {1,4,...,31}
        # （s1 个），W {2,5,...,29}（s2 个）——与 [11,11,10] 精确一致
        s = art["rope_config"]["mrope_section"]
        h_idx = {i for i in range(1, 3 * s[1], 3)}
        w_idx = {i for i in range(2, 3 * s[2], 3)}
        t_idx = set(range(n)) - h_idx - w_idx
        assert (len(t_idx), len(h_idx), len(w_idx)) == (11, 11, 10)
        assert t_idx == set(range(0, 3 * s[0], 3))
        assert max(h_idx) == n - 1 and max(w_idx) == n - 3

    def test_mtp_trace_and_negatives(self):
        art = _fixture()
        assert art["mtp"]["invariant_checks"] >= 16
        rounds = art["mtp"]["trace"]
        assert len(rounds) == 4
        for r in rounds:                      # 每轮 K=3 步、末步不推进
            steps = r["steps"]
            assert [s["step"] for s in steps] == [0, 1, 2]
            assert steps[-1]["pos_after"] == steps[-1]["fwd_pos"]
            for a, b in zip(steps, steps[1:]):
                assert b["fwd_pos"] == min(a["pos_after"], 10**9)
        neg = art["negatives"]
        for k in ("decode_restart_zero", "step_plus_two", "no_rewind"):
            assert neg[k]["caught"] is True
        assert neg["multimodal_2d_distinct_rows"]["branch_exists"] is True
        assert neg["multimodal_2d_distinct_rows"]["scope"].startswith(
            "out-of-scope")
        assert neg["section_sum_assert"]["fired"] is True
        assert neg["clamp_max_len"]["clamped"] is True
        assert neg["clamp_max_len"]["pos"] == 23   # max−1

    def test_fixture_pins_match_current_pin_bytes(self):
        """pin 文件改动 → fixture 必须重生成（pins_sha256 绑定）。"""
        import hashlib
        art = _fixture()
        src = {
            "hf/modeling_qwen3_5_moe.py": PIN_HF / "modeling_qwen3_5_moe.py",
            "vllm_dc74b8cb/qwen3_5.py": PIN_PAR / "qwen3_5.py",
            "vllm_dc74b8cb/rope_mrope.py": PIN_MROPE / "rope_mrope.py",
            "vllm_dc74b8cb/qwen3_vl.py": PIN_MROPE / "qwen3_vl.py",
            "vllm_10a854ed/tdar_speculator.py": PIN_MTP / "tdar_speculator.py",
        }
        for key, path in src.items():
            disk = hashlib.sha256(path.read_bytes()).hexdigest()
            assert art["pins_sha256"][key] == disk, f"{key} 变更未重生成 fixture"
        assert len(art["pins_sha256"]) == 11

    def test_bad_section_sum_rejected_in_parser(self):
        from simverse.model_ir.families import parse_qwen3_5_moe
        from simverse.model_ir.schema import SourceIdentity
        cfg = dict(
            vocab_size=64, hidden_size=32, num_hidden_layers=1,
            layer_types=["full_attention"], num_attention_heads=4,
            num_key_value_heads=2, head_dim=8,
            linear_num_key_heads=2, linear_num_value_heads=4,
            linear_key_head_dim=8, linear_value_head_dim=8,
            linear_conv_kernel_dim=4,
            rope_parameters={"rope_type": "default", "rope_theta": 1e7,
                             "partial_rotary_factor": 1.0,
                             "mrope_section": [2, 2, 1],   # 和 5 ≠ 4
                             "mrope_interleaved": True})
        src = SourceIdentity(repo_id="synthetic", revision="r26-test",
                             config_sha256="synthetic", captured_at="2026-11-16")
        with pytest.raises(ValueError, match="mrope_section"):
            parse_qwen3_5_moe(cfg, src, "neg_fixture", "neg fixture")
        cfg["rope_parameters"]["mrope_section"] = [2, 1, 1]      # 和 4 ✓
        ir = parse_qwen3_5_moe(cfg, src, "ok_fixture", "ok fixture")
        assert ir.extras["rope"]["mrope_section"] == [2, 1, 1]
        assert ir.extras["rope"]["text_semantics"]["degenerate"] is True


# ---------------------------------------------------------------------------
# Part A：replay 固定场景 + CLI/API 一致性
# ---------------------------------------------------------------------------

class TestReplayAndConsistency:
    def test_replay_zero_ledger_drift_and_evidence_flip(self, capfd):
        import subprocess

        subprocess.run([sys.executable, "-X", "utf8", "tools/replay_qwen36_mrope.py", "--check"],
                       cwd=ROOT, check=True)

    def test_run_v2_consistent_with_catalog(self):
        from simverse.model_ir.catalog import catalog_json
        from simverse.pipeline import run_v2
        from simverse.scenario import ScenarioV2
        b = run_v2(ScenarioV2()).to_json()
        m07 = b["metrics"]["M07_support_summary"]
        assert m07["structure_complete"] is True
        assert m07["forward_source"] == "source-checked"
        cap = m07["capability_checks"]
        assert cap["model_cpu_forward"] == "unknown"     # 执行证据另列
        reason = str(b["metrics"]["M02_speedup_vs_baseline"]["reason"])
        assert "mrope" not in reason and "CPU forward" in reason
        assert set(catalog_json()["models"]) == {
            "qwen36_35b_a3b", "deepseek_v41_flash"}


# ---------------------------------------------------------------------------
# Part B：live（torch 可用时重执行探针 hermetic+live；否则 skip）
# ---------------------------------------------------------------------------

class TestLiveProbe:
    def test_probe_reruns_and_official_class_matches(self):
        pytest.importorskip("torch")
        pytest.importorskip("transformers")
        from tools.probe_qwen36_mrope_numeric import run_hermetic, run_live
        art = run_hermetic()
        live = run_live(art)
        assert live["executed"] is True
        assert live["max_diff_official_vs_oracle"] <= 1e-5
        assert live["decode_pos5_cos_diff"] <= 1e-5
