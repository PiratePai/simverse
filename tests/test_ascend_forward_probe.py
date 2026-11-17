"""Offline tests for tools/ascend_forward_probe.py.

These tests never import vLLM/torch for plan construction and never touch an
Ascend device.  The execution-wiring test passes stub factories explicitly and
asserts that the artifact is marked ``executed-stub`` with
``hardware_executed=false``.
"""

from __future__ import annotations

import builtins
import copy
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "ascend_forward_probe.py"
SOURCE_CONFIG = ROOT / "assets" / "config" / "qwen36_35b_a3b_main__config.json"
FIXTURE_SPEC = (
    ROOT
    / "assets"
    / "identity"
    / "ascend_forward"
    / "qwen36_35b_a3b_text_fixture_config.json"
)
SOURCE_MANIFEST = (
    ROOT / "assets" / "identity" / "ascend_forward" / "source_manifest.json"
)

SPEC = importlib.util.spec_from_file_location("ascend_forward_probe_under_test", TOOL)
AFP = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(AFP)

VERSIONS = {
    "vllm": "0.26.0",
    "vllm-ascend": "0.26.0rc1",
    "torch": "2.10.0",
    "torch-npu": "2.10.0.post4",
}


@pytest.fixture()
def workdir():
    base = ROOT / ".tmp" / "r28" / "pytest_ascend_forward"
    shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True, exist_ok=True)
    try:
        yield base
    finally:
        shutil.rmtree(base, ignore_errors=True)


def _write_model(
    base: Path,
    name: str,
    config: dict,
    *,
    weights: bool = True,
    tokenizer: bool = True,
    quant_file: bool = False,
) -> Path:
    model = base / name
    model.mkdir(parents=True, exist_ok=True)
    (model / "config.json").write_text(
        json.dumps(config, ensure_ascii=False), encoding="utf-8"
    )
    if weights:
        (model / "model.safetensors").write_bytes(b"")
    if tokenizer:
        (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    if quant_file:
        (model / "quant_model_description.json").write_text('{"model.layers.0.mlp.experts.0.gate_proj.weight":"W8A8"}', encoding="utf-8")
    return model


def _qwen36_config() -> dict:
    return json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))


def _v41_config() -> dict:
    return {
        "architectures": ["DeepseekV41ForCausalLM"],
        "model_type": "deepseek_v41",
        "text_config": {
            "num_attention_heads": 128,
            "num_key_value_heads": 8,
            "vocab_size": 129280,
        },
    }


def _reject(profile: str, options: dict, code: str) -> None:
    with pytest.raises(AFP.ProbeError) as excinfo:
        AFP.build_plan(profile, options, versions=VERSIONS)
    assert excinfo.value.code == code


def test_dry_run_fixture_does_not_import_torch_or_vllm(monkeypatch, workdir):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "torch_npu", "vllm"}:
            raise AssertionError(f"dry-run imported {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    plan = AFP.build_plan(
        "qwen36-fixture",
        {"runtime_dir": str(workdir / "fixture_model")},
        versions=VERSIONS,
    )

    assert plan["mode"] == "dry-run"
    assert plan["status"] == "planned"
    assert plan["engine"]["args"]["load_format"] == "dummy"
    assert plan["engine"]["args"]["skip_tokenizer_init"] is True
    assert plan["engine"]["args"]["quantization"] is None
    assert plan["engine"]["args"]["tensor_parallel_size"] == 1
    assert plan["identity"]["full_model"] is False
    assert plan["claims"]["random_weights"] is True
    assert plan["claims"]["hardware_executed"] is False
    assert plan["claims"]["performance_measured"] is False
    assert plan["claims"]["acceptance_rate_measured"] is False
    assert plan["fixture"]["source_config_sha256"] == AFP._sha256_file(SOURCE_CONFIG)
    assert plan["fixture"]["overrides"]["num_hidden_layers"] == 2
    assert "execution" not in plan


def test_fixture_config_provenance_matches_pinned_source():
    spec = json.loads(FIXTURE_SPEC.read_text(encoding="utf-8"))
    source = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
    original = source[spec["source_field"]]
    effective = spec["config"]
    overrides = spec["overrides"]

    assert spec["source_config_sha256"] == AFP._sha256_file(SOURCE_CONFIG)
    for key, value in original.items():
        assert key in effective, f"fixture dropped source key {key}"
        if key in overrides:
            assert effective[key] == overrides[key]
        else:
            assert effective[key] == value
    for key in overrides:
        assert key in original
        assert original[key] != overrides[key]
    assert effective["architectures"] == ["Qwen3_5MoeForCausalLM"]
    assert effective["model_type"] == "qwen3_5_moe_text"


def test_fixture_config_is_accepted_by_pinned_transformers_config():
    pin = ROOT / "assets" / "identity" / "hf_transformers"
    try:
        spec = importlib.util.spec_from_file_location(
            "transformers.models.qwen3_5_moe.pinned_configuration_qwen3_5_moe",
            pin / "configuration_qwen3_5_moe.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        fixture = json.loads(FIXTURE_SPEC.read_text(encoding="utf-8"))["config"]
        config = module.Qwen3_5MoeTextConfig(**fixture)
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.skip(f"pinned transformers config unavailable: {exc}")

    assert config.num_hidden_layers == 2
    assert config.layer_types == ["linear_attention", "full_attention"]
    assert config.head_dim == 256
    assert sum(config.rope_parameters["mrope_section"]) == (
        config.head_dim * config.rope_parameters["partial_rotary_factor"] // 2
    )


def test_main_dry_run_writes_json_and_returns_zero(capsys, workdir):
    out = workdir / "plan.json"
    rc = AFP.main(
        [
            "--profile",
            "qwen36-fixture",
            "--runtime-dir",
            str(workdir / "fixture_model"),
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    artifact = json.loads(out.read_text(encoding="utf-8"))
    assert artifact["kind"] == "ascend-forward-probe-plan"
    assert artifact["mode"] == "dry-run"
    assert artifact["execute_blocked"] is False
    assert "artifact:" in capsys.readouterr().out


def test_invalid_tokens_and_prompt_are_rejected(workdir):
    for ids in ([], [-1], [1024], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 20):
        _reject("qwen36-fixture", {"prompt_token_ids": ids}, "tokens")
    _reject("qwen36-fixture", {"prompt_token_ids": "1,not-a-number"}, "tokens")

    model = _write_model(workdir, "local", _qwen36_config())
    _reject(
        "qwen36-local",
        {"model_path": str(model), "prompt": "   "},
        "prompt",
    )
    _reject(
        "qwen36-local",
        {"model_path": str(model), "prompt_token_ids": [16]},
        "tokens",
    )


def test_tp_and_device_dtype_are_rejected(workdir):
    _reject("qwen36-fixture", {"tensor_parallel_size": 3}, "tp")
    _reject("qwen36-fixture", {"tensor_parallel_size": 0}, "tp")
    _reject("qwen36-fixture", {"device": "cuda"}, "device")
    _reject("qwen36-fixture", {"device": "cpu"}, "device")
    _reject("qwen36-fixture", {"dtype": "float16"}, "dtype")

    model = _write_model(workdir, "local", _qwen36_config())
    _reject(
        "qwen36-local",
        {"model_path": str(model), "tensor_parallel_size": 3},
        "tp",
    )


def test_local_architecture_mismatch_is_rejected(workdir):
    model = _write_model(
        workdir,
        "llama",
        {"architectures": ["LlamaForCausalLM"], "model_type": "llama"},
    )
    _reject("qwen36-local", {"model_path": str(model)}, "architecture")


def test_quantization_mismatches_are_rejected(workdir):
    _reject("qwen36-fixture", {"quantization": "ascend"}, "quantization")

    bf16 = _write_model(workdir, "bf16", _qwen36_config())
    _reject(
        "qwen36-local",
        {"model_path": str(bf16), "quantization": "ascend"},
        "quantization",
    )

    w8a8 = _write_model(workdir, "w8a8", _qwen36_config(), quant_file=True)
    _reject(
        "qwen36-local",
        {"model_path": str(w8a8), "quantization": "none"},
        "quantization",
    )

    fp8_config = copy.deepcopy(_qwen36_config())
    fp8_config["quantization_config"] = {"quant_method": "fp8"}
    fp8 = _write_model(workdir, "fp8", fp8_config)
    _reject("qwen36-local", {"model_path": str(fp8)}, "quantization")


def test_w8a8_local_resolves_to_ascend_quantization(workdir):
    model = _write_model(workdir, "w8a8_ok", _qwen36_config(), quant_file=True)
    plan = AFP.build_plan(
        "qwen36-local", {"model_path": str(model)}, versions=VERSIONS
    )
    assert plan["engine"]["args"]["quantization"] == "ascend"
    assert plan["identity"]["quantization"] == "ascend"


def test_versions_are_gated(workdir):
    with pytest.raises(AFP.ProbeError) as excinfo:
        AFP.build_plan(
            "qwen36-fixture",
            {},
            versions={"vllm": "0.17.0", "vllm-ascend": "0.26.0rc1"},
        )
    assert excinfo.value.code == "vllm_version"

    with pytest.raises(AFP.ProbeError) as excinfo:
        AFP.build_plan(
            "qwen36-fixture",
            {},
            versions={"vllm": "0.26.0", "vllm-ascend": "0.17.0"},
        )
    assert excinfo.value.code == "vllm_ascend_version"


def test_v41_plan_is_blocked_and_lists_real_gates(workdir):
    model = _write_model(
        workdir, "v41", _v41_config(), quant_file=True
    )
    plan = AFP.build_plan(
        "deepseek-v41", {"model_path": str(model)}, versions=VERSIONS
    )

    assert plan["status"] == "blocked"
    assert plan["execute_blocked"] is True
    assert plan["identity"]["architecture"] == "DeepseekV41ForCausalLM"
    assert plan["identity"]["model_type"] == "deepseek_v41"
    assert plan["engine"]["args"]["tensor_parallel_size"] == 8
    assert plan["engine"]["args"]["enable_expert_parallel"] is True
    assert plan["engine"]["args"]["quantization"] == "ascend"
    assert plan["v41_serve"]["global_topology"] == "DP4/TP8/EP32"
    assert plan["commands"]["execute"] is None
    assert any(
        gate["id"] == "quantization" and gate["status"] == "pass"
        for gate in plan["gates"]
    )
    with pytest.raises(AFP.ProbeError) as excinfo:
        AFP.execute_plan(plan)
    assert excinfo.value.code == "execute_blocked"


def test_v41_rejects_v4_config_and_029(workdir):
    v4 = _write_model(
        workdir,
        "v4",
        {"architectures": ["DeepseekV4ForCausalLM"], "model_type": "deepseek_v4"},
        quant_file=True,
    )
    with pytest.raises(AFP.ProbeError) as excinfo:
        AFP.build_plan("deepseek-v41", {"model_path": str(v4)}, versions=VERSIONS)
    assert excinfo.value.code == "v4_substitution"

    v41 = _write_model(workdir, "v41_029", _v41_config(), quant_file=True)
    plan = AFP.build_plan(
        "deepseek-v41",
        {"model_path": str(v41)},
        versions={
            "vllm": "0.29.0",
            "vllm-ascend": "0.26.0rc1",
            "torch": "2.10.0",
            "torch-npu": "2.10.0.post4",
        },
    )
    assert plan["execute_blocked"] is True
    assert any(
        gate["id"] == "vllm_version" and gate["status"] == "fail"
        for gate in plan["gates"]
    )

    rc_plan = AFP.build_plan(
        "deepseek-v41",
        {"model_path": str(v41)},
        versions={
            "vllm": "0.29.0rc1",
            "vllm-ascend": "0.26.0rc1",
            "torch": "2.10.0",
            "torch-npu": "2.10.0.post4",
        },
    )
    assert not any(
        gate["id"] == "vllm_version" and gate["status"] == "fail"
        for gate in rc_plan["gates"]
    )


class _FakeCompletion:
    token_ids = [7, 8]
    text = ""
    finish_reason = "length"


class _FakeRequestOutput:
    outputs = [_FakeCompletion()]


class _FakeLLM:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.prompts = None
        self.sampling = None

    def generate(self, prompts, sampling_params, use_tqdm=False):
        self.prompts = prompts
        self.sampling = sampling_params
        return [_FakeRequestOutput()]


def test_stub_execution_wiring_is_explicitly_non_hardware(monkeypatch, workdir):
    plan = AFP.build_plan(
        "qwen36-fixture",
        {"runtime_dir": str(workdir / "fixture_model")},
        versions=VERSIONS,
    )
    captured: dict = {}

    def llm_factory(**kwargs):
        llm = _FakeLLM(**kwargs)
        captured["llm"] = llm
        return llm

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "torch_npu", "vllm"}:
            raise AssertionError(f"stub execution imported {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    result = AFP.execute_plan(
        plan, llm_factory=llm_factory, sampling_factory=lambda **kwargs: kwargs
    )

    llm = captured["llm"]
    assert llm.kwargs["load_format"] == "dummy"
    assert llm.kwargs["skip_tokenizer_init"] is True
    assert llm.kwargs["model"] == str(workdir / "fixture_model")
    assert llm.prompts == [{"prompt_token_ids": [16, 17, 18, 19, 20]}]
    assert result["status"] == "executed-stub"
    assert result["execution"]["stub"] is True
    assert result["claims"]["hardware_executed"] is False
    assert result["claims"]["hardware_verified"] is False
    assert result["outputs"] == [
        {"token_ids": [7, 8], "text": "", "finish_reason": "length"}
    ]
    runtime_config = workdir / "fixture_model" / "config.json"
    assert runtime_config.is_file()
    assert AFP._sha256_canonical(
        json.loads(runtime_config.read_text(encoding="utf-8"))
    ) == result["fixture"]["effective_config_sha256"]


def test_source_manifest_matches_tool_constants():
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    from pathlib import PurePosixPath, PureWindowsPath
    assert manifest["schema_version"] == "2"
    assert manifest["references"]["vllm"]["url"] == AFP.VLLM_REPO
    assert manifest["references"]["vllm_ascend"]["url"] == AFP.VLLM_ASCEND_REPO
    for reference in manifest["references"].values():
        assert reference["url"].startswith("https://github.com/")
        assert "path" not in reference
        assert len(reference["commit"]) == 40
    for item in manifest["hashes"]:
        assert item["repository"] in manifest["references"]
        assert not PurePosixPath(item["path"]).is_absolute()
        assert not PureWindowsPath(item["path"]).drive
        assert ".." not in PurePosixPath(item["path"]).parts
    assert manifest["references"]["vllm"]["commit"] == AFP.VLLM_COMMIT
    assert manifest["references"]["vllm_ascend"]["commit"] == AFP.VLLM_ASCEND_COMMIT
    assert manifest["verified_pins"]["vllm_main"] == AFP.VLLM_VERIFIED_MAIN
    assert manifest["qwen36"]["repo_id"] == AFP.QWEN36_REPO_ID
    assert (
        manifest["qwen36"]["revision_main_sha"] == AFP.QWEN36_REVISION_MAIN
    )
    assert manifest["qwen36"]["source_config_sha256"] == AFP._sha256_file(
        SOURCE_CONFIG
    )
    expected_suffix = "assets/config/qwen36_35b_a3b_main__config.json"
    assert any(
        item["repository"] == "simverse" and item["path"] == expected_suffix
        and item["sha256"] == AFP._sha256_file(SOURCE_CONFIG)
        for item in manifest["hashes"]
    )
    assert (
        manifest["deepseek_v41"]["vllm_ascend_registration"]["condition"]
        == "not vllm_version_is('0.29.0')"
    )


def test_probe_runs_from_relocated_checkout_without_reference_repositories(workdir):
    import subprocess
    relocated = workdir / "contributor checkout"
    # Only the runner and its three in-repo data files; no upstream checkout.
    for source in (TOOL, SOURCE_CONFIG, FIXTURE_SPEC, SOURCE_MANIFEST):
        target = relocated / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    output = workdir / "relocated-plan.json"
    result = subprocess.run(
        [sys.executable, "-I", str(relocated / "tools/ascend_forward_probe.py"),
         "--profile", "qwen36-fixture", "--out", str(output)],
        cwd=workdir, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    plan = json.loads(output.read_text(encoding="utf-8"))
    assert plan["sources"]["vllm"]["repo"] == "https://github.com/vllm-project/vllm"
    assert plan["sources"]["vllm_ascend"]["repo"] == "https://github.com/vllm-project/vllm-ascend"
    assert Path(plan["identity"]["source_config_path"]).is_relative_to(relocated)
    assert Path(plan["engine"]["args"]["model"]).is_relative_to(relocated)
    assert not Path(plan["engine"]["args"]["model"]).exists()
    assert plan["claims"]["hardware_executed"] is False


def test_default_factories_use_keyword_parameters(monkeypatch, workdir):
    from types import SimpleNamespace
    def sampling(*, max_tokens, temperature, top_p, seed, ignore_eos, detokenize):
        return dict(max_tokens=max_tokens, temperature=temperature, seed=seed,
                    ignore_eos=ignore_eos, detokenize=detokenize)
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=_FakeLLM, SamplingParams=sampling))
    monkeypatch.setattr(AFP, "_preflight_hardware", lambda plan: {"hardware_verified": False})
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
    plan = AFP.build_plan("qwen36-fixture", {"runtime_dir": str(workdir / "fixture")}, versions=VERSIONS)
    result = AFP.execute_plan(plan)
    assert result["outputs"][0]["token_ids"] == [7, 8]
    assert result["claims"]["hardware_verified"] is False


def test_registry_resolution_uses_class_api(monkeypatch):
    from types import SimpleNamespace
    def resolve(architecture, config):
        assert architecture == AFP.QWEN36_ARCH_WRAPPER
        return _FakeLLM, architecture
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models.registry",
                        SimpleNamespace(ModelRegistry=SimpleNamespace(resolve_model_cls=resolve)))
    result = AFP._inspect_model_cls({"identity": {"architecture": AFP.QWEN36_ARCH_WRAPPER}},
                                    SimpleNamespace(model_config=object()))
    assert result["class"] == "_FakeLLM"
    assert "error" not in result


def test_hardware_preflight_accepts_two_cards_and_rejects_other_family(monkeypatch):
    from types import SimpleNamespace
    import importlib.metadata
    npu = SimpleNamespace(device_count=lambda: 2, get_device_name=lambda index: "Ascend910B3")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__="2.10.0", npu=npu))
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(__version__="2.10.0.post4"))
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(__version__="0.26.0"))
    monkeypatch.setitem(sys.modules, "vllm.platforms", SimpleNamespace(current_platform=SimpleNamespace(device_type="npu")))
    monkeypatch.setattr(importlib.metadata, "version", lambda package: "0.26.0")
    plan = {"profile": "qwen36-local", "engine": {"args": {"tensor_parallel_size": 2}}}
    assert AFP._preflight_hardware(plan)["device_count"] == 2
    npu.get_device_name = lambda index: "Ascend310P3" if index else "Ascend910B3"
    with pytest.raises(AFP.ProbeError, match="Unverified A2/A3"):
        AFP._preflight_hardware(plan)
    npu.device_count = lambda: 1
    with pytest.raises(AFP.ProbeError) as error:
        AFP._preflight_hardware(plan)
    assert error.value.code == "device_count"


def test_fixture_preserves_kernel_dimensions_and_existing_config(workdir):
    plan = AFP.build_plan("qwen36-fixture", {"runtime_dir": str(workdir / "fixture")}, versions=VERSIONS)
    source = _qwen36_config()["text_config"]
    for key in ("head_dim", "hidden_size", "linear_key_head_dim", "linear_value_head_dim",
                "rope_parameters", "moe_intermediate_size", "num_experts_per_tok"):
        assert plan["fixture"]["effective_config"][key] == source[key]
    target = Path(plan["fixture"]["runtime_config_path"])
    target.parent.mkdir()
    target.write_text('{"existing":"checkpoint"}', encoding="utf-8")
    with pytest.raises(AFP.ProbeError) as error:
        AFP._materialize_fixture(plan)
    assert error.value.code == "fixture_target"
    assert json.loads(target.read_text())["existing"] == "checkpoint"


def test_matching_architecture_does_not_allow_changed_model_structure(workdir):
    config = _qwen36_config()
    config["text_config"]["hidden_size"] = 512
    model = _write_model(workdir, "other", config)
    _reject("qwen36-local", {"model_path": str(model)}, "checkpoint_structure")


def test_empty_modelslim_descriptor_does_not_prove_w8a8(workdir):
    model = _write_model(workdir, "emptyquant", _qwen36_config(), quant_file=True)
    (model / "quant_model_description.json").write_text("{}", encoding="utf-8")
    _reject("qwen36-local", {"model_path": str(model)}, "quantization")


def test_replay_command_preserves_effective_settings(workdir):
    import shlex
    options = {"runtime_dir": str(workdir / "fixture"), "seed": 42,
               "gpu_memory_utilization": 0.25, "max_num_batched_tokens": 256,
               "enable_expert_parallel": True, "trust_remote_code": True}
    plan = AFP.build_plan("qwen36-fixture", options, versions=VERSIONS)
    args = AFP._parser().parse_args(shlex.split(plan["commands"]["plan"])[2:])
    for key, value in options.items():
        assert getattr(args, key) == value
