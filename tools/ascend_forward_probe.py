"""Offline Ascend forward/collection entry for Qwen3.6 and DeepSeek-V4.1.

This tool does not implement a model forward.  It drives the *real* forward
owned by the installed vLLM engine and the vllm-ascend platform plugin:

* ``qwen36-fixture``: a 2-layer, text-only, random-weight fixture derived from
  the pinned Qwen3.6-35B-A3B text config, loaded with vLLM
  ``load_format="dummy"`` and ``skip_tokenizer_init=True``.
* ``qwen36-local``: a real local Qwen3.6-35B-A3B checkpoint loaded by vLLM
  with the Ascend NPU platform selected by the vllm-ascend entry point.
* ``deepseek-v41``: a plan-only/blocked entry.  It records the real vLLM and
  vllm-ascend registrations and the documented W8A8 multi-node gates, but this
  single-process runner refuses to execute V4.1.  A V4/V4-Flash checkpoint is
  rejected as a substitute.

Safety defaults:
  * dry-run unless ``--execute`` is passed;
  * no torch/vllm import during dry-run or plan construction;
  * no model download, no remote GPU call, no credentials;
  * no performance or acceptance-rate claim.  A short random-weight fixture is a
    forward-wiring check, not Qwen3.6-35B-A3B whole-model performance.

The source audit is recorded in
``assets/identity/ascend_forward/source_manifest.json`` and the fixture identity
in ``assets/identity/ascend_forward/qwen36_35b_a3b_text_fixture_config.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
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
DEFAULT_RUNTIME_DIR = ROOT / ".tmp" / "r28" / "ascend_forward"

# Source pins actually read for this change.  These are intentionally constants
# (not resolved from the local file system) so the artifact records what was
# reviewed even when the tool runs on a different machine.
VLLM_REPO = "D:/Github/vllm"
VLLM_COMMIT = "6b858751f6e99c4f33913657dc0c2afd34ff13e6"
VLLM_ASCEND_REPO = "D:/Github/vllm-ascend-main"
VLLM_ASCEND_COMMIT = "baa023ada41a5b889cac6249845a688aab130892"
VLLM_VERIFIED_MAIN = "84030bbe3d74d99bad477a3d2e37a973ccd8865c"
VLLM_RELEASE_TAG = "v0.29.0"

QWEN36_ARCH_WRAPPER = "Qwen3_5MoeForConditionalGeneration"
QWEN36_ARCH_TEXT = "Qwen3_5MoeForCausalLM"
QWEN36_MODEL_TYPE = "qwen3_5_moe"
QWEN36_TEXT_MODEL_TYPE = "qwen3_5_moe_text"
PINNED_QWEN36_CONFIG_SHA256 = (
    "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99"
)
QWEN36_REPO_ID = "Qwen/Qwen3.6-35B-A3B"
QWEN36_REVISION_MAIN = "345d467b117d1fd5eda9b879a5988de9ae5f47a7"
V41_REPO_ID = "deepseek-ai/DeepSeek-V4.1-Flash"
V41_ARCH = "DeepseekV41ForCausalLM"
V41_MODEL_TYPE = "deepseek_v41"

MIN_VLLM = (0, 18, 0)
MIN_VLLM_ASCEND = (0, 18, 0)
V41_MIN_VLLM = (0, 26, 0)
V41_MIN_VLLM_ASCEND = (0, 26, 0)

DEFAULT_SEED = 20261117
FIXTURE_PROMPT_IDS = [16, 17, 18, 19, 20]

PROFILES = {
    "qwen36-fixture": {
        "model_name": "Qwen3.6-35B-A3B",
        "checkpoint_kind": "random-weight-text-fixture",
        "full_model": False,
        "entrypoint": "vllm.LLM",
    },
    "qwen36-local": {
        "model_name": "Qwen3.6-35B-A3B",
        "checkpoint_kind": "local-checkpoint",
        "full_model": True,
        "entrypoint": "vllm.LLM",
    },
    "deepseek-v41": {
        "model_name": "DeepSeek-V4.1-Flash",
        "checkpoint_kind": "local-checkpoint",
        "full_model": True,
        "entrypoint": "vllm serve (multi-node, plan only)",
    },
}

DEFAULTS = {
    "qwen36-fixture": dict(
        tensor_parallel_size=1,
        quantization="none",
        dtype="bfloat16",
        device="npu",
        prompt=None,
        prompt_token_ids=FIXTURE_PROMPT_IDS,
        max_tokens=4,
        max_model_len=128,
        max_num_batched_tokens=128,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
        enable_expert_parallel=False,
        trust_remote_code=False,
        seed=DEFAULT_SEED,
    ),
    "qwen36-local": dict(
        tensor_parallel_size=2,
        quantization="auto",
        dtype="bfloat16",
        device="npu",
        prompt="Hello, my name is",
        prompt_token_ids=None,
        max_tokens=8,
        max_model_len=4096,
        max_num_batched_tokens=4096,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        enable_expert_parallel=True,
        trust_remote_code=False,
        seed=DEFAULT_SEED,
    ),
    "deepseek-v41": dict(
        tensor_parallel_size=8,
        quantization="ascend",
        dtype="bfloat16",
        device="npu",
        prompt="Hello, my name is",
        prompt_token_ids=None,
        max_tokens=4,
        max_model_len=4096,
        max_num_batched_tokens=4096,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        enable_expert_parallel=True,
        trust_remote_code=True,
        seed=DEFAULT_SEED,
    ),
}


class ProbeError(RuntimeError):
    """A refused plan or execution, with a stable machine-readable code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ProbeError("read", f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ProbeError("json", f"invalid JSON in {path}: {exc}") from exc


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ProbeError("hash", f"cannot hash {path}: {exc}") from exc


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_canonical(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _version_tuple(value: object) -> tuple[int, int, int] | None:
    if value is None:
        return None
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", str(value))
    if not match:
        return None
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


def _is_exact_release(value: object, target: str) -> bool:
    """Match vllm_version_is semantics for a plain release (ignore +local)."""

    if value is None:
        return False
    return str(value).split("+", 1)[0] == target


def installed_versions() -> dict[str, str | None]:
    """Read distribution metadata without importing torch or vllm."""

    try:
        from importlib.metadata import version
    except ImportError:  # pragma: no cover - Python 3.12 always has it
        return {
            "vllm": None,
            "vllm-ascend": None,
            "torch": None,
            "torch-npu": None,
        }
    result: dict[str, str | None] = {}
    for dist in ("vllm", "vllm-ascend", "torch", "torch-npu"):
        try:
            result[dist] = version(dist)
        except Exception:
            result[dist] = None
    return result


def _gate(gates: list[dict], gate_id: str, status: str, detail: str) -> None:
    gates.append({"id": gate_id, "status": status, "detail": detail})


def _normalise_options(profile: str, options: dict | None) -> dict:
    opts = dict(DEFAULTS[profile])
    for key, value in (options or {}).items():
        if value is not None:
            opts[key] = value

    if opts["device"] != "npu":
        raise ProbeError(
            "device",
            f"profile {profile} targets Ascend NPU; device={opts['device']!r} is "
            "rejected. This runner never falls back to CUDA/CPU.",
        )
    if opts["dtype"] not in ("bfloat16",):
        raise ProbeError(
            "dtype",
            f"profile {profile} uses bfloat16 on A2/A3; dtype={opts['dtype']!r} "
            "is outside the documented stack.",
        )
    if opts["quantization"] not in ("auto", "none", "ascend"):
        raise ProbeError(
            "quantization",
            "quantization must be one of auto/none/ascend; "
            f"got {opts['quantization']!r}",
        )
    if int(opts["max_tokens"]) < 1:
        raise ProbeError("sampling", "max_tokens must be >= 1")
    if int(opts["max_model_len"]) <= int(opts["max_tokens"]):
        raise ProbeError(
            "sampling", "max_model_len must be greater than max_tokens"
        )
    token_ids = opts.get("prompt_token_ids")
    if isinstance(token_ids, str):
        try:
            opts["prompt_token_ids"] = [
                int(part) for part in token_ids.split(",") if part.strip()
            ]
        except ValueError as exc:
            raise ProbeError("tokens", f"invalid --prompt-token-ids: {exc}") from exc
    return opts


def _hf_text(config: dict) -> dict:
    text = config.get("text_config")
    return text if isinstance(text, dict) else config


def _architectures(config: dict) -> list[str]:
    arches = config.get("architectures")
    if not isinstance(arches, list) or not arches:
        raise ProbeError(
            "architecture", "config.json has no non-empty 'architectures' list"
        )
    return [str(arch) for arch in arches]


def _check_architecture(profile: str, config: dict, gates: list[dict]) -> None:
    arches = _architectures(config)
    model_type = config.get("model_type")
    text_model_type = _hf_text(config).get("model_type")

    if profile == "qwen36-fixture":
        expected_arch = QWEN36_ARCH_TEXT
        expected_type = QWEN36_TEXT_MODEL_TYPE
    elif profile == "qwen36-local":
        expected_arch = QWEN36_ARCH_WRAPPER
        expected_type = QWEN36_MODEL_TYPE
    else:
        expected_arch = V41_ARCH
        expected_type = V41_MODEL_TYPE

    if profile == "deepseek-v41":
        v4_aliases = {"DeepseekV4ForCausalLM", "DeepseekV4ForConditionalGeneration"}
        if any(arch in v4_aliases for arch in arches):
            raise ProbeError(
                "v4_substitution",
                "DeepSeek-V4/V4-Flash config is not DeepSeek-V4.1; the V4 "
                "four-card e2e test and V4 checkpoints cannot substitute for V4.1.",
            )

    if expected_arch not in arches:
        raise ProbeError(
            "architecture",
            f"profile {profile} expects architecture {expected_arch!r}, got {arches!r}. "
            "Refusing name-based or cross-model adaptation.",
        )
    if model_type != expected_type:
        raise ProbeError(
            "model_type",
            f"profile {profile} expects model_type {expected_type!r}, got {model_type!r}.",
        )
    if profile == "qwen36-local" and text_model_type != QWEN36_TEXT_MODEL_TYPE:
        raise ProbeError(
            "text_model_type",
            "Qwen3.6-35B-A3B local checkpoint must carry "
            f"text_config.model_type={QWEN36_TEXT_MODEL_TYPE!r}; got "
            f"{text_model_type!r}. Use qwen36-fixture for the text-only random "
            "fixture instead.",
        )
    _gate(
        gates,
        "architecture",
        "pass",
        f"architectures={arches}, model_type={model_type}, text_model_type={text_model_type}",
    )


def _check_tp(profile: str, tp: int, text_config: dict, gates: list[dict]) -> None:
    if not isinstance(tp, int) or tp < 1:
        raise ProbeError("tp", f"tensor_parallel_size must be a positive integer; got {tp!r}")

    if profile == "qwen36-fixture":
        allowed = (1, 2)
    elif profile == "qwen36-local":
        allowed = (1, 2, 4, 8)
    else:
        allowed = (8,)
    if tp not in allowed:
        raise ProbeError(
            "tp",
            f"profile {profile} documents tensor_parallel_size in {allowed}; got {tp}.",
        )

    heads = text_config.get("num_attention_heads")
    kv_heads = text_config.get("num_key_value_heads")
    if heads is not None and int(heads) % tp != 0:
        raise ProbeError(
            "tp", f"num_attention_heads={heads} is not divisible by TP={tp}"
        )
    if kv_heads is not None:
        kv = int(kv_heads)
        if kv % tp != 0 and tp % kv != 0:
            raise ProbeError(
                "tp",
                f"num_key_value_heads={kv} cannot be sharded/replicated by TP={tp}",
            )
    for key in ("linear_num_key_heads", "linear_num_value_heads"):
        value = text_config.get(key)
        if value is not None and int(value) % tp != 0:
            raise ProbeError(
                "tp", f"{key}={value} is not divisible by TP={tp}"
            )
    _gate(
        gates,
        "tp",
        "pass",
        f"TP={tp} fits attention/GDN head counts "
        f"(heads={heads}, kv_heads={kv_heads})",
    )


def _quant_signature(model_path: Path, config: dict) -> str | None:
    if (model_path / "quant_model_description.json").is_file():
        description = _read_json(model_path / "quant_model_description.json")
        # ModelSlim loader performs the complete per-parameter validation.
        # Presence alone (including an empty JSON object) proves no quantization.
        if not any(value == "W8A8" for value in description.values()):
            raise ProbeError("quantization", "ModelSlim descriptor contains no W8A8 entries; this probe only covers W8A8")
        return "modelslim"
    qc = config.get("quantization_config")
    if isinstance(qc, dict):
        method = str(qc.get("quant_method", "")).strip().lower()
        if method:
            return method
    return None


def _resolve_quantization(
    profile: str,
    requested: str,
    signature: str | None,
    gates: list[dict],
) -> str | None:
    requested = requested.lower()
    if requested not in ("auto", "none", "ascend"):
        raise ProbeError("quantization", f"unsupported quantization request {requested!r}")

    if profile == "qwen36-fixture":
        if requested == "ascend":
            raise ProbeError(
                "quantization",
                "qwen36-fixture uses vLLM dummy random BF16 weights; "
                "--quantization ascend is not valid for this fixture.",
            )
        _gate(
            gates,
            "quantization",
            "pass",
            "dummy random weights, BF16, quantization disabled",
        )
        return None

    if profile == "qwen36-local":
        if signature == "modelslim":
            if requested == "none":
                raise ProbeError(
                    "quantization",
                    "model path has quant_model_description.json (ModelSlim W8A8) "
                    "but --quantization none was requested.",
                )
            _gate(
                gates,
                "quantization",
                "pass",
                "ModelSlim W8A8 entries detected; tensor coverage/dtypes require backend validation",
            )
            return "ascend"
        if signature is not None:
            raise ProbeError(
                "quantization",
                f"unsupported quantization signature {signature!r} for the Ascend "
                "Qwen3.6 path; expected BF16 or ModelSlim W8A8.",
            )
        if requested == "ascend":
            raise ProbeError(
                "quantization",
                "no quant_model_description.json / quantization_config found, so "
                "--quantization ascend would be a false claim.",
            )
        _gate(
            gates,
            "quantization",
            "pass",
            "no quantization metadata; requesting BF16 (weight dtypes not audited)",
        )
        return None

    # DeepSeek-V4.1
    if signature != "modelslim":
        _gate(
            gates,
            "quantization",
            "fail",
            "V4.1 documented path requires W8A8 ModelSlim "
            "(quant_model_description.json); no such signature found",
        )
        return "ascend" if requested == "ascend" else None
    if requested == "none":
        _gate(
            gates,
            "quantization",
            "fail",
            "V4.1 W8A8 checkpoint requires --quantization ascend",
        )
        return None
    _gate(
        gates,
        "quantization",
        "pass",
        "ModelSlim W8A8 signature found; --quantization ascend",
    )
    return "ascend"


def _weight_inventory(model_path: Path, profile: str) -> tuple[list[str], list[str]]:
    index_files = sorted(model_path.glob("*.safetensors.index.json"))
    if index_files:
        index = _read_json(index_files[0])
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ProbeError(
                "weights", f"{index_files[0]} has no usable weight_map"
            )
        shards = sorted({str(value) for value in weight_map.values()})
        missing = [name for name in shards if not (model_path / name).is_file()]
        return shards, missing

    candidates = []
    for path in sorted(model_path.iterdir()):
        if path.is_file() and path.suffix in (".safetensors", ".bin", ".pt"):
            candidates.append(path.name)
    if profile == "deepseek-v41":
        candidates = [name for name in candidates if name.endswith(".safetensors")]
    return candidates, []


def _check_weights(profile: str, model_path: Path, gates: list[dict]) -> None:
    if profile == "qwen36-fixture":
        _gate(
            gates,
            "weights",
            "pass",
            "vLLM load_format=dummy; no checkpoint weights are read",
        )
        return
    files, missing = _weight_inventory(model_path, profile)
    if missing:
        raise ProbeError(
            "weights",
            f"index references missing shard(s): {missing[:5]}",
        )
    if not files:
        raise ProbeError(
            "weights",
            f"{model_path} has no .safetensors/.bin/.pt weight file; "
            "downloading is disabled, so provide a complete local checkpoint.",
        )
    _gate(
        gates,
        "weights",
        "pass",
        f"{len(files)} local weight shard(s) present; not loaded in dry-run",
    )


def _check_tokenizer(profile: str, model_path: Path, gates: list[dict]) -> None:
    if profile == "qwen36-fixture":
        _gate(
            gates,
            "tokenizer",
            "pass",
            "skip_tokenizer_init=True; prompts are token IDs",
        )
        return
    present = [
        name
        for name in ("tokenizer.json", "tokenizer.model", "vocab.txt")
        if (model_path / name).is_file()
    ]
    has_bpe = (model_path / "vocab.json").is_file() and (
        model_path / "merges.txt"
    ).is_file()
    if not present and not has_bpe:
        raise ProbeError(
            "tokenizer",
            f"{model_path} has no usable local tokenizer (tokenizer.json / "
            "tokenizer.model / vocab.txt / vocab.json+merges.txt). Text-prompt "
            "mode requires a complete checkpoint, and downloading is disabled.",
        )
    if has_bpe:
        present.append("vocab.json+merges.txt")
    _gate(
        gates,
        "tokenizer",
        "pass",
        f"local tokenizer marker(s): {present}",
    )


def _check_processor(profile: str, model_path: Path, gates: list[dict]) -> None:
    if profile == "qwen36-fixture":
        return
    markers = (
        "preprocessor_config.json",
        "processor_config.json",
        "video_preprocessor_config.json",
    )
    present = [name for name in markers if (model_path / name).is_file()]
    if present:
        _gate(
            gates,
            "processor",
            "pass",
            f"local multimodal processor marker(s): {present}",
        )
    else:
        _gate(
            gates,
            "processor",
            "warn",
            "no multimodal processor JSON found; text-only forward may work, but "
            "image/video preprocessing is not covered by this probe.",
        )


def _check_prompt(
    profile: str,
    opts: dict,
    text_config: dict,
    gates: list[dict],
) -> dict:
    vocab_size = text_config.get("vocab_size")
    max_tokens = int(opts["max_tokens"])
    max_model_len = int(opts["max_model_len"])

    if profile == "qwen36-fixture":
        ids = opts.get("prompt_token_ids")
        if isinstance(ids, str):
            try:
                ids = [int(part) for part in ids.split(",") if part.strip()]
            except ValueError as exc:
                raise ProbeError("tokens", f"invalid --prompt-token-ids: {exc}") from exc
        if not isinstance(ids, list) or not ids:
            raise ProbeError("tokens", "fixture requires a non-empty token-ID prompt")
        for token in ids:
            if not isinstance(token, int):
                raise ProbeError("tokens", f"token IDs must be integers; got {token!r}")
            if token < 0 or (vocab_size is not None and token >= int(vocab_size)):
                raise ProbeError(
                    "tokens",
                    f"token id {token} is outside [0, vocab_size={vocab_size})",
                )
        if len(ids) + max_tokens > max_model_len:
            raise ProbeError(
                "tokens",
                f"prompt({len(ids)}) + max_tokens({max_tokens}) exceeds "
                f"max_model_len({max_model_len})",
            )
        _gate(
            gates,
            "prompt",
            "pass",
            f"{len(ids)} token id(s) in [0, {vocab_size}); no tokenizer",
        )
        return {"mode": "token_ids", "prompts": [{"prompt_token_ids": list(ids)}]}

    if opts.get("prompt_token_ids") is not None:
        raise ProbeError(
            "tokens",
            "token-ID prompts are fixture-only. The real-checkpoint path uses a "
            "text prompt so the checkpoint's own tokenizer is exercised.",
        )

    prompt = opts.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ProbeError("prompt", "real-checkpoint path requires a non-empty --prompt")
    _gate(
        gates,
        "prompt",
        "pass",
        f"text prompt of {len(prompt)} chars; max_tokens={max_tokens}",
    )
    return {"mode": "text", "prompts": [prompt]}


def _check_versions(
    profile: str,
    versions: dict,
    gates: list[dict],
) -> None:
    vllm = versions.get("vllm")
    ascend = versions.get("vllm-ascend")
    vllm_t = _version_tuple(vllm)
    ascend_t = _version_tuple(ascend)
    torch_v = versions.get("torch")
    torch_npu_v = versions.get("torch-npu")
    torch_t = _version_tuple(torch_v)
    torch_npu_t = _version_tuple(torch_npu_v)

    if torch_t is None or torch_npu_t is None:
        _gate(
            gates,
            "stack",
            "unknown",
            "torch/torch-npu metadata unavailable; the vllm-ascend baa023a "
            "requirements pin torch 2.10.0 / torch-npu 2.10.0.post4 and CANN 9.1.0",
        )
    elif torch_t < (2, 10) or torch_npu_t < (2, 10):
        _gate(
            gates,
            "stack",
            "warn",
            f"installed torch={torch_v} torch-npu={torch_npu_v}; vllm-ascend "
            "baa023a requirements pin torch 2.10.0 / torch-npu 2.10.0.post4. "
            "Execution will refuse this stack.",
        )
    else:
        _gate(
            gates,
            "stack",
            "pass",
            f"torch={torch_v} torch-npu={torch_npu_v} meet the 2.10 major/minor floor",
        )

    if profile == "deepseek-v41":
        if _is_exact_release(vllm, "0.29.0"):
            _gate(
                gates,
                "vllm_version",
                "fail",
                "vllm-ascend registers DeepseekV41ForCausalLM only when "
                "vllm_version_is('0.29.0') is false; installed vLLM is exactly 0.29.0",
            )
        elif vllm_t is not None and vllm_t < V41_MIN_VLLM:
            _gate(
                gates,
                "vllm_version",
                "fail",
                f"vLLM {vllm} is below the documented V4.1 minimum {V41_MIN_VLLM}",
            )
        elif vllm_t is None:
            _gate(
                gates,
                "vllm_version",
                "unknown",
                "vLLM distribution metadata unavailable; execution would verify after import",
            )
        else:
            _gate(
                gates,
                "vllm_version",
                "pass",
                f"vLLM {vllm} is not the excluded 0.29.0 release",
            )

        if ascend_t is not None and ascend_t < V41_MIN_VLLM_ASCEND:
            _gate(
                gates,
                "vllm_ascend_version",
                "fail",
                f"vllm-ascend {ascend} is below the documented V4.1 minimum "
                f"{V41_MIN_VLLM_ASCEND}",
            )
        elif ascend_t is None:
            _gate(
                gates,
                "vllm_ascend_version",
                "unknown",
                "vllm-ascend distribution metadata unavailable",
            )
        else:
            _gate(
                gates,
                "vllm_ascend_version",
                "pass",
                f"vllm-ascend {ascend}",
            )
        return

    if vllm_t is None:
        _gate(
            gates,
            "vllm_version",
            "unknown",
            "vLLM distribution metadata unavailable; dry-run plan only. "
            "Execution verifies vllm.__version__ after import.",
        )
    elif vllm_t < MIN_VLLM:
        raise ProbeError(
            "vllm_version",
            f"Qwen3.6 Ascend support starts at vLLM/vllm-ascend 0.18.0; got {vllm}",
        )
    else:
        _gate(gates, "vllm_version", "pass", f"vLLM {vllm} >= 0.18.0")

    if ascend_t is None:
        _gate(
            gates,
            "vllm_ascend_version",
            "unknown",
            "vllm-ascend distribution metadata unavailable; execution verifies "
            "importlib.metadata after import.",
        )
    elif ascend_t < MIN_VLLM_ASCEND:
        raise ProbeError(
            "vllm_ascend_version",
            f"Qwen3.6 is first documented in vllm-ascend v0.18.0rc1; got {ascend}",
        )
    else:
        _gate(
            gates,
            "vllm_ascend_version",
            "pass",
            f"vllm-ascend {ascend} >= 0.18.0rc1",
        )


def _load_fixture() -> tuple[dict, dict, dict]:
    spec = _read_json(FIXTURE_SPEC)
    source_path = ROOT / str(spec["source_config"])
    actual_hash = _sha256_file(source_path)
    if actual_hash != spec["source_config_sha256"]:
        raise ProbeError(
            "fixture_identity",
            f"fixture source hash mismatch: expected "
            f"{spec['source_config_sha256']}, got {actual_hash}",
        )
    source_config = _read_json(source_path)
    field = spec.get("source_field", "text_config")
    original = source_config.get(field)
    if not isinstance(original, dict):
        raise ProbeError("fixture_identity", f"source field {field!r} is not an object")
    effective = spec.get("config")
    overrides = spec.get("overrides")
    if not isinstance(effective, dict) or not isinstance(overrides, dict):
        raise ProbeError("fixture_identity", "fixture config/overrides must be objects")

    for key, value in original.items():
        if key not in effective:
            raise ProbeError(
                "fixture_identity", f"effective fixture config dropped source key {key!r}"
            )
        if key in overrides:
            if effective[key] != overrides[key]:
                raise ProbeError(
                    "fixture_identity",
                    f"override {key!r}={overrides[key]!r} does not match effective "
                    f"value {effective[key]!r}",
                )
        elif effective[key] != value:
            raise ProbeError(
                "fixture_identity",
                f"source key {key!r} changed without being listed in overrides",
            )
    for key in overrides:
        if key not in original:
            raise ProbeError(
                "fixture_identity", f"override {key!r} is not present in the source text config"
            )
        if original[key] == overrides[key]:
            raise ProbeError(
                "fixture_identity", f"override {key!r} does not change the source value"
            )
    for key in spec.get("added_keys", []):
        if key not in effective:
            raise ProbeError(
                "fixture_identity", f"declared added key {key!r} is missing"
            )
    return spec, original, effective


def _engine_args(profile: str, opts: dict, model: Path, quantization: str | None) -> dict:
    args = {
        "model": str(model),
        "dtype": opts["dtype"],
        "tensor_parallel_size": int(opts["tensor_parallel_size"]),
        "max_model_len": int(opts["max_model_len"]),
        "max_num_seqs": 1,
        "max_num_batched_tokens": int(opts["max_num_batched_tokens"]),
        "gpu_memory_utilization": float(opts["gpu_memory_utilization"]),
        "enforce_eager": bool(opts["enforce_eager"]),
        "enable_prefix_caching": False,
        "trust_remote_code": bool(opts["trust_remote_code"]),
        "seed": int(opts["seed"]),
        "disable_log_stats": True,
        "quantization": quantization,
        "enable_expert_parallel": bool(opts["enable_expert_parallel"]),
    }
    if profile == "qwen36-fixture":
        args.update(
            load_format="dummy",
            skip_tokenizer_init=True,
        )
    else:
        args.update(
            load_format="auto",
            skip_tokenizer_init=False,
        )
    if int(opts["tensor_parallel_size"]) > 1:
        args["distributed_executor_backend"] = "mp"
    return args


def _sampling_args(profile: str, opts: dict) -> dict:
    return {
        "max_tokens": int(opts["max_tokens"]),
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": int(opts["seed"]),
        "detokenize": profile != "qwen36-fixture",
        # The random fixture has an arbitrary eos id; force the requested
        # number of decode steps so prefill+decode wiring is exercised.
        "ignore_eos": profile == "qwen36-fixture",
    }


def _plan_command(profile: str, opts: dict) -> str:
    parts = [
        "python3",
        "tools/ascend_forward_probe.py",
        "--profile",
        profile,
        "--tensor-parallel-size",
        str(opts["tensor_parallel_size"]),
        "--quantization",
        str(opts["quantization"]),
        "--max-model-len",
        str(opts["max_model_len"]),
        "--max-tokens",
        str(opts["max_tokens"]),
    ]
    if opts.get("model_path"):
        parts.extend(["--model-path", str(opts["model_path"])])
    if opts.get("prompt") and profile != "qwen36-fixture":
        parts.extend(["--prompt", str(opts["prompt"])])
    if opts.get("prompt_token_ids") and profile == "qwen36-fixture":
        parts.extend(
            [
                "--prompt-token-ids",
                ",".join(str(token) for token in opts["prompt_token_ids"]),
            ]
        )
    if profile == "qwen36-fixture" and opts.get("runtime_dir"):
        parts.extend(["--runtime-dir", str(opts["runtime_dir"])])
    if not opts["enforce_eager"]:
        parts.append("--no-enforce-eager")
    if not opts["enable_expert_parallel"]:
        parts.append("--no-enable-expert-parallel")
    for key in ("dtype", "device", "seed", "max_num_batched_tokens", "gpu_memory_utilization"):
        parts.extend(["--" + key.replace("_", "-"), str(opts[key])])
    if opts["trust_remote_code"]:
        parts.append("--trust-remote-code")
    if opts["enable_expert_parallel"]:
        parts.append("--enable-expert-parallel")
    return shlex.join(parts)


def _v41_serve_plan(opts: dict) -> dict:
    return {
        "documented_hardware": (
            "2x Atlas 800 A3 (16 logical devices/server) or "
            "4x Atlas 800 A2 (8 devices/server)"
        ),
        "global_topology": "DP4/TP8/EP32",
        "alternative": "single A3: TP8/DP2/EP16, INT8 Engram host offload; see section 5.3; not executed",
        "node_layout": "A3: 2 DP ranks/server; A2: 1 DP rank/server",
        "checkpoint": "Ascend W8A8 ModelSlim checkpoint with INT8 Engram tables",
        "quantization": "ascend",
        "model_runner": "V1",
        "speculative": "DSpark, eager draft",
        "graph": "FULL_DECODE_ONLY",
        "max_model_len": 1048576,
        "common_flags": [
            "--data-parallel-size 4",
            "--tensor-parallel-size 8",
            "--enable-expert-parallel",
            "--quantization ascend",
            "--tokenizer-mode deepseek_v41",
            "--block-size 128",
            '--compilation-config {"cudagraph_mode":"FULL_DECODE_ONLY"}',
            '--speculative-config {"method":"dspark","num_speculative_tokens":5,"enforce_eager":true}',
        ],
        "reference": (
            "vllm-ascend docs/source/tutorials/models/DeepSeek-V4.1-Flash.md "
            "sections 5.1/5.3 at commit baa023a"
        ),
        "not_executed_by_this_tool": (
            "This runner is a single-process offline probe. It refuses to fake "
            "a 2x A3 / 4x A2 multi-node DP4/TP8/EP32 launch."
        ),
    }


def build_plan(
    profile: str,
    options: dict | None = None,
    *,
    versions: dict | None = None,
) -> dict:
    """Build a dry-run plan without importing torch or vllm."""

    if profile not in PROFILES:
        raise ProbeError("profile", f"unknown profile {profile!r}")
    opts = _normalise_options(profile, options)
    versions = dict(versions) if versions is not None else installed_versions()
    gates: list[dict] = []
    limitations: list[str] = []

    model_path: Path | None = None
    source_config_hash: str | None = None
    effective_config: dict
    effective_config_hash: str
    overrides: dict
    fixture_section = None
    text_config: dict

    if profile == "qwen36-fixture":
        spec, _source, effective_config = _load_fixture()
        effective_config_hash = _sha256_canonical(effective_config)
        overrides = dict(spec["overrides"])
        source_config_hash = str(spec["source_config_sha256"])
        text_config = effective_config
        runtime_dir = Path(
            opts.get("runtime_dir") or (DEFAULT_RUNTIME_DIR / "qwen36_fixture_model")
        )
        if not runtime_dir.is_absolute():
            runtime_dir = (ROOT / runtime_dir).resolve()
        model_path = runtime_dir
        fixture_section = {
            "enabled": True,
            "source_config": str(spec["source_config"]),
            "source_config_sha256": source_config_hash,
            "source_field": spec.get("source_field", "text_config"),
            "effective_config": effective_config,
            "effective_config_sha256": effective_config_hash,
            "runtime_config_path": str(runtime_dir / "config.json"),
            "overrides": overrides,
            "added_keys": list(spec.get("added_keys", [])),
            "random_weights": True,
            "text_only": True,
            "vision": False,
            "mtp": False,
            "full_model": False,
            "note": spec.get("note", ""),
        }
        _gate(
            gates,
            "fixture_identity",
            "pass",
            "source hash and every override match the pinned text_config",
        )
    else:
        if not opts.get("model_path"):
            raise ProbeError(
                "model_path",
                f"profile {profile} requires --model-path pointing at an existing "
                "local checkpoint; downloading is disabled.",
            )
        raw_path = Path(str(opts["model_path"]))
        model_path = raw_path if raw_path.is_absolute() else (ROOT / raw_path)
        model_path = model_path.resolve()
        if not model_path.is_dir():
            raise ProbeError(
                "model_path",
                f"{model_path} is not an existing local directory. The probe does "
                "not download models or call a remote GPU.",
            )
        config_path = model_path / "config.json"
        if not config_path.is_file():
            raise ProbeError("model_path", f"{model_path} has no config.json")
        config = _read_json(config_path)
        source_config_hash = _sha256_file(config_path)
        effective_config = config
        effective_config_hash = source_config_hash
        overrides = {}
        text_config = _hf_text(config)

    _check_architecture(profile, effective_config, gates)
    checkpoint_identity = None
    if profile == "qwen36-fixture":
        checkpoint_identity = {
            "repo_id": QWEN36_REPO_ID,
            "pinned_revision_main_sha": QWEN36_REVISION_MAIN,
            "pinned_main_config_sha256": PINNED_QWEN36_CONFIG_SHA256,
            "matches_pinned_main": True,
            "note": "fixture derives from the pinned Qwen3.6-35B-A3B text_config",
        }
    elif profile == "qwen36-local":
        pinned_text = _read_json(SOURCE_CONFIG)["text_config"]
        mismatches = [key for key, value in pinned_text.items()
                      if key not in {"dtype", "torch_dtype"} and text_config.get(key) != value]
        if mismatches:
            raise ProbeError("checkpoint_structure", "Text config differs from Qwen3.6 pin: " + ", ".join(mismatches))
        matches = source_config_hash == PINNED_QWEN36_CONFIG_SHA256
        checkpoint_identity = {
            "repo_id": QWEN36_REPO_ID,
            "pinned_revision_main_sha": QWEN36_REVISION_MAIN,
            "pinned_main_config_sha256": PINNED_QWEN36_CONFIG_SHA256,
            "matches_pinned_main": matches,
            "note": (
                "config.json matches the pinned Qwen3.6-35B-A3B main config"
                if matches
                else "config.json differs from the pinned main config; this can be "
                "a W8A8/converted local checkpoint, but the artifact does not claim "
                "the pinned main revision"
            ),
        }
        _gate(
            gates,
            "checkpoint_identity",
            "pass" if matches else "warn",
            checkpoint_identity["note"],
        )
    else:
        checkpoint_identity = {
            "repo_id": V41_REPO_ID,
            "pinned_revision_main_sha": None,
            "matches_pinned_main": None,
            "note": "no local HF revision pin for V4.1; only the model id and "
            "architecture/model_type are checked",
        }
        _gate(
            gates,
            "checkpoint_identity",
            "warn",
            checkpoint_identity["note"],
        )
    _check_tp(profile, int(opts["tensor_parallel_size"]), text_config, gates)

    model_path = model_path or Path(".")
    _check_weights(profile, model_path, gates)
    _check_tokenizer(profile, model_path, gates)
    _check_processor(profile, model_path, gates)

    signature = (
        None
        if profile == "qwen36-fixture"
        else _quant_signature(model_path, effective_config)
    )
    quantization = _resolve_quantization(
        profile, str(opts["quantization"]), signature, gates
    )
    request = _check_prompt(profile, opts, text_config, gates)
    _check_versions(profile, versions, gates)

    allowed_versions = {
        "vllm": versions.get("vllm"),
        "vllm-ascend": versions.get("vllm-ascend"),
        "torch": versions.get("torch"),
        "torch-npu": versions.get("torch-npu"),
    }
    _gate(
        gates,
        "sources",
        "pass",
        f"vLLM @{VLLM_COMMIT} (verified main pin {VLLM_VERIFIED_MAIN}); "
        f"vllm-ascend @{VLLM_ASCEND_COMMIT}",
    )
    limitations.append(
        "Source audit is read-only against local reference trees; "
        f"vLLM {VLLM_COMMIT} is not vllm-ascend's documented verified main "
        f"{VLLM_VERIFIED_MAIN}."
    )

    engine_args = _engine_args(profile, opts, model_path, quantization)
    sampling = _sampling_args(profile, opts)

    if profile == "qwen36-fixture":
        limitations.extend(
            [
                "Random-weight, text-only, 2-layer fixture: not the Qwen3.6-35B-A3B "
                "checkpoint, not whole-model performance, and not an acceptance-rate "
                "measurement.",
                "Vision tower, image/video tokens, MTP and DFlash are not exercised.",
                "A fixture pass only proves the vLLM/vllm-ascend forward wiring on "
                "the installed Ascend stack.",
            ]
        )
    elif profile == "qwen36-local":
        limitations.extend(
            [
                "Real local checkpoint forward through vLLM; this change did not run "
                "it on hardware, so no measured/performance claim is made.",
                "Official docs include DP1/TP2; A2 8-card/A3 16-card describes the server. "
                "Only TP participating NPUs are required; capacity and this stack still need validation.",
                "MTP/DFlash acceptance and product inference are out of scope.",
            ]
        )
    else:
        limitations.extend(
            [
                "DeepSeek-V4.1 is plan-only/blocked here; no V4.1 forward was run.",
                "V4.1 docs give W8A8 DP4/TP8/EP32, and a single-A3 TP8/DP2/EP16 "
                "Engram host-offload alternative; neither is a two-910B3 deployment.",
                "DeepSeek-V4/V4-Flash tests and checkpoints are not V4.1 evidence.",
            ]
        )

    blocked_reasons = [gate["detail"] for gate in gates if gate["status"] == "fail"]
    if profile == "deepseek-v41":
        blocked_reasons.append(
            "This single-process offline runner does not execute the documented "
            "multi-node V4.1 deployment. Use the vllm serve command from "
            "docs/source/tutorials/models/DeepSeek-V4.1-Flash.md on a matching "
            "W8A8 multi-node cluster."
        )
    execute_blocked = bool(blocked_reasons)
    if execute_blocked and profile != "deepseek-v41":
        raise ProbeError("gate_failed", "; ".join(blocked_reasons))

    source_manifest_hash = (
        _sha256_file(SOURCE_MANIFEST) if SOURCE_MANIFEST.is_file() else None
    )
    plan = {
        "schema_version": "1",
        "kind": "ascend-forward-probe-plan",
        "profile": profile,
        "mode": "dry-run",
        "status": "blocked" if execute_blocked else "planned",
        "generated_at": _utc_now(),
        "claims": {
            "hardware_executed": False,
            "hardware_verified": False,
            "full_model_forward": False,
            "random_weights": profile == "qwen36-fixture",
            "performance_measured": False,
            "acceptance_rate_measured": False,
            "v41_substitute_for_v4": False,
        },
        "identity": {
            "model_name": PROFILES[profile]["model_name"],
            "profile": profile,
            "checkpoint_kind": PROFILES[profile]["checkpoint_kind"],
            "architecture": (
                QWEN36_ARCH_TEXT
                if profile == "qwen36-fixture"
                else QWEN36_ARCH_WRAPPER
                if profile == "qwen36-local"
                else V41_ARCH
            ),
            "architectures": _architectures(effective_config),
            "model_type": (
                QWEN36_TEXT_MODEL_TYPE
                if profile == "qwen36-fixture"
                else QWEN36_MODEL_TYPE
                if profile == "qwen36-local"
                else V41_MODEL_TYPE
            ),
            "dtype": opts["dtype"],
            "quantization": quantization,
            "implementation": "vllm-ascend",
            "entrypoint": PROFILES[profile]["entrypoint"],
            "source_config_path": str(
                SOURCE_CONFIG if profile == "qwen36-fixture" else model_path / "config.json"
            ),
            "source_config_sha256": source_config_hash,
            "effective_config_path": str(model_path / "config.json"),
            "effective_config_sha256": effective_config_hash,
            "overrides": overrides,
            "config": None if profile == "qwen36-fixture" else effective_config,
            "checkpoint_identity": checkpoint_identity,
            "weights_sha256": None,
            "weights_identity_verified": False,
            "full_model": bool(PROFILES[profile]["full_model"]),
        },
        "sources": {
            "vllm": {
                "repo": VLLM_REPO,
                "commit": VLLM_COMMIT,
                "installed_version": allowed_versions["vllm"],
                "commit_role": "reviewed-reference-not-runtime",
                "runtime_commit": None,
            },
            "vllm_ascend": {
                "repo": VLLM_ASCEND_REPO,
                "commit": VLLM_ASCEND_COMMIT,
                "installed_version": allowed_versions["vllm-ascend"],
                "commit_role": "reviewed-reference-not-runtime",
                "runtime_commit": None,
            },
            "vllm_main_verified_commit": VLLM_VERIFIED_MAIN,
            "vllm_release_tag": VLLM_RELEASE_TAG,
            "source_manifest": str(SOURCE_MANIFEST.relative_to(ROOT)),
            "source_manifest_sha256": source_manifest_hash,
            "stack_requirements": {
                "torch": "2.10.0",
                "torch-npu": "2.10.0.post4",
                "cann": "9.1.0",
                "evidence": "vllm-ascend baa023a requirements.txt / release notes",
            },
            "installed": {
                "torch": allowed_versions["torch"],
                "torch-npu": allowed_versions["torch-npu"],
            },
        },
        "engine": {
            "args": engine_args,
            "sampling": sampling,
            "request": request,
        },
        "gates": gates,
        "execute_blocked": execute_blocked,
        "blocked_reason": "; ".join(blocked_reasons) if blocked_reasons else None,
        "commands": {
            "plan": _plan_command(
                profile,
                {
                    **opts,
                    "model_path": None
                    if profile == "qwen36-fixture"
                    else model_path,
                    "runtime_dir": str(model_path)
                    if profile == "qwen36-fixture"
                    else opts.get("runtime_dir"),
                    "quantization": quantization or "none",
                },
            ),
            "execute": None
            if profile == "deepseek-v41"
            else _plan_command(
                profile,
                {
                    **opts,
                    "model_path": None
                    if profile == "qwen36-fixture"
                    else model_path,
                    "runtime_dir": str(model_path)
                    if profile == "qwen36-fixture"
                    else opts.get("runtime_dir"),
                    "quantization": quantization or "none",
                },
            )
            + " --execute",
        },
        "limitations": limitations,
        "notes": [
            "Dry-run does not import torch or vllm and does not load weights.",
            "The only hardware path is vLLM LLM + the installed vllm-ascend "
            "platform plugin; there is no hand-written CUDA/Ascend forward here.",
            "No automatic model download and no remote GPU call.",
        ],
    }
    if profile == "qwen36-fixture":
        plan["fixture"] = fixture_section
    if profile == "deepseek-v41":
        plan["v41_registration"] = {
            "vllm": {
                "architecture": V41_ARCH,
                "target": "vllm.models.deepseek_v41:DeepseekV41ForCausalLM",
                "evidence": "vllm/model_executor/models/registry.py:378-381",
            },
            "vllm_ascend": {
                "architecture": V41_ARCH,
                "target": (
                    "vllm_ascend.models.deepseek_v41.vl_model:"
                    "AscendDeepseekV41ForCausalLM"
                ),
                "condition": "not vllm_version_is('0.29.0')",
                "evidence": "vllm_ascend/models/__init__.py:40-44",
            },
            "dspark": {
                "architecture": "DeepseekV41DSparkModel",
                "target": (
                    "vllm_ascend.models.deepseek_v41.dspark:"
                    "DSparkDeepseekV41ForCausalLM"
                ),
                "condition": "not vllm_version_is('0.29.0')",
                "evidence": "vllm_ascend/models/__init__.py:58-62",
            },
            "v4_not_substitute": (
                "DeepseekV4ForCausalLM / DeepseekV4ForConditionalGeneration and "
                "tests/e2e/pull_request/four_card/test_deepseek_v4.py are V4, "
                "not V4.1."
            ),
        }
        plan["v41_serve"] = _v41_serve_plan(opts)
        plan["commands"]["serve_reference"] = (
            "see docs/source/tutorials/models/DeepSeek-V4.1-Flash.md "
            "(vllm serve, DP4/TP8/EP32, W8A8)"
        )
    return plan


def _materialize_fixture(plan: dict) -> Path:
    fixture = plan.get("fixture")
    if not fixture:
        raise ProbeError("fixture", "plan has no fixture section")
    target = Path(fixture["runtime_config_path"])
    if target.exists() and _sha256_canonical(_read_json(target)) != fixture["effective_config_sha256"]:
        raise ProbeError("fixture_target", f"Refusing to overwrite a different config: {target}")
    if any(target.parent.glob("*.safetensors")) or any(target.parent.glob("*.bin")):
        raise ProbeError("fixture_target", "Fixture directory contains checkpoint weights")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(fixture["effective_config"], ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    actual = _sha256_canonical(_read_json(target))
    if actual != fixture["effective_config_sha256"]:
        raise ProbeError(
            "fixture",
            f"materialized fixture config hash mismatch: {actual} != "
            f"{fixture['effective_config_sha256']}",
        )
    return target


def _preflight_hardware(plan: dict) -> dict:
    try:
        import torch
        import torch_npu  # noqa: F401
    except Exception as exc:  # pragma: no cover - hardware path
        raise ProbeError(
            "hardware_import",
            f"cannot import torch/torch_npu on this host: {type(exc).__name__}: {exc}",
        ) from exc

    torch_version = getattr(torch, "__version__", None)
    torch_npu_version = getattr(torch_npu, "__version__", None)
    torch_t = _version_tuple(torch_version)
    torch_npu_t = _version_tuple(torch_npu_version)
    if torch_t is None or torch_npu_t is None:
        raise ProbeError(
            "stack_version",
            "cannot determine torch/torch-npu versions from the imported stack",
        )
    if torch_t < (2, 10) or torch_npu_t < (2, 10):
        raise ProbeError(
            "stack_version",
            f"vllm-ascend {VLLM_ASCEND_COMMIT} requirements pin torch 2.10.0 / "
            f"torch-npu 2.10.0.post4; this host has torch={torch_version} "
            f"torch-npu={torch_npu_version}.",
        )

    try:
        count = int(torch.npu.device_count())
        name = str(torch.npu.get_device_name(0)) if count else "unknown"
    except Exception as exc:  # pragma: no cover - hardware path
        raise ProbeError(
            "hardware",
            f"torch.npu device query failed: {type(exc).__name__}: {exc}",
        ) from exc

    tp = int(plan["engine"]["args"]["tensor_parallel_size"])
    if count < tp:
        raise ProbeError(
            "device_count",
            f"TP={tp} needs at least {tp} visible NPUs, but torch.npu reports {count}",
        )
    # Collection domain from vllm-ascend device/hardware.py at baa023a.
    # Check every participating rank; server card count is not TP size.
    names = [str(torch.npu.get_device_name(rank)) for rank in range(tp)]
    for device_name in names:
        normal = device_name.strip().lower()
        if not (re.fullmatch(r"ascend910b(?:1|2|2c|3|4|4-1)", normal)
                or normal in {"910b", "910c", "ascend910_9391", "ascend910_9381",
                              "ascend910_9372", "ascend910_9392", "ascend910_9382",
                              "ascend910_9362", "ascend910_9363"}):
            raise ProbeError("device_family", f"Unverified A2/A3 collection device: {device_name}")

    import vllm

    from vllm.platforms import current_platform

    if current_platform.device_type != "npu":
        raise ProbeError(
            "platform",
            f"vLLM selected current_platform.device_type={current_platform.device_type!r}; "
            "expected 'npu' from the installed vllm-ascend plugin.",
        )
    vllm_version = str(getattr(vllm, "__version__", "")) or None
    vllm_t = _version_tuple(vllm_version)
    if vllm_t is None or vllm_t < MIN_VLLM:
        raise ProbeError(
            "vllm_version", f"installed vLLM {vllm_version} is below 0.18.0"
        )
    try:
        from importlib.metadata import version as dist_version

        ascend_version = dist_version("vllm-ascend")
    except Exception:
        ascend_version = None
    ascend_t = _version_tuple(ascend_version)
    if ascend_t is None or ascend_t < MIN_VLLM_ASCEND:
        raise ProbeError(
            "vllm_ascend_version",
            f"installed vllm-ascend {ascend_version} is below 0.18.0",
        )
    if plan["profile"] == "deepseek-v41" and _is_exact_release(
        vllm_version, "0.29.0"
    ):
        raise ProbeError(
            "v41_registration",
            "vllm 0.29.0 disables the vllm-ascend DeepseekV41ForCausalLM override",
        )
    return {
        "device_type": "npu",
        "device_count": count,
        "device_name": name,
        "participating_device_names": names,
        "stack_pair_validated": False,
        "numerical_correctness_verified": False,
        "torch": getattr(torch, "__version__", None),
        "torch_npu": getattr(torch_npu, "__version__", None),
        "vllm": vllm_version,
        "vllm_ascend": ascend_version,
        "platform_class": type(current_platform).__name__,
        "ascend_rt_visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "hardware_verified": True,
        "stub": False,
    }


def _default_factories():
    from vllm import LLM, SamplingParams

    return LLM, SamplingParams


def _extract_outputs(outputs) -> list[dict]:
    result = []
    for request_output in outputs:
        for completion in getattr(request_output, "outputs", []) or []:
            result.append(
                {
                    "token_ids": [int(token) for token in getattr(completion, "token_ids", [])],
                    "text": getattr(completion, "text", None),
                    "finish_reason": getattr(completion, "finish_reason", None),
                }
            )
    return result


def _inspect_model_cls(plan: dict, llm) -> dict:
    try:
        from vllm.model_executor.models.registry import ModelRegistry

        cls, arch = ModelRegistry.resolve_model_cls(
            plan["identity"]["architecture"], llm.model_config
        )
        return {
            "architecture": arch,
            "module": cls.__module__,
            "class": cls.__qualname__,
        }
    except Exception as exc:  # pragma: no cover - hardware path
        return {"error": f"{type(exc).__name__}: {exc}"}


def execute_plan(
    plan: dict,
    *,
    llm_factory=None,
    sampling_factory=None,
    preflight=None,
) -> dict:
    """Execute a previously built plan.

    Tests pass both factories to prove the wiring without importing vLLM or
    touching hardware.  In that case the artifact is explicitly marked as a stub.
    """

    if plan.get("execute_blocked"):
        raise ProbeError("execute_blocked", plan.get("blocked_reason") or "plan is blocked")
    if plan["profile"] == "deepseek-v41":
        raise ProbeError(
            "v41_execute",
            "DeepSeek-V4.1 is plan-only in this runner; use the documented "
            "multi-node vllm serve command on matching W8A8 hardware.",
        )
    stub = llm_factory is not None or sampling_factory is not None
    if stub and (llm_factory is None or sampling_factory is None):
        raise ProbeError(
            "stub", "provide both llm_factory and sampling_factory, or neither"
        )

    result = json.loads(json.dumps(plan))
    if result["profile"] == "qwen36-fixture":
        _materialize_fixture(result)

    if stub:
        hardware = (
            preflight(result)
            if preflight is not None
            else {
                "device_type": "stub",
                "device_count": None,
                "device_name": None,
                "hardware_verified": False,
                "stub": True,
            }
        )
    else:
        hardware = preflight(result) if preflight is not None else _preflight_hardware(result)
        llm_factory, sampling_factory = _default_factories()

    if int(result["engine"]["args"]["tensor_parallel_size"]) > 1:
        # vllm-ascend e2e tests use spawn for multi-process Ascend workers.
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    # Local checkpoints only; propagate to spawned workers before engine import.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    llm = llm_factory(**result["engine"]["args"])
    sampling = sampling_factory(**result["engine"]["sampling"])
    started = time.perf_counter()
    outputs = llm.generate(
        result["engine"]["request"]["prompts"], sampling, use_tqdm=False
    )
    elapsed_s = time.perf_counter() - started

    result["kind"] = "ascend-forward-probe-result"
    result["mode"] = "execute"
    result["status"] = "executed-stub" if stub else "executed"
    result["execution"] = {
        "hardware": hardware,
        "elapsed_s": elapsed_s,
        "elapsed_boundary": "host wall clock around LLM.generate; not device kernel time",
        "stub": stub,
        "model_class": _inspect_model_cls(result, llm) if hasattr(llm, "model_config") else None,
    }
    result["outputs"] = _extract_outputs(outputs)
    result["claims"]["hardware_executed"] = not stub
    result["claims"]["hardware_verified"] = bool(hardware.get("hardware_verified"))
    result["claims"]["full_model_forward"] = (
        bool(result["identity"]["full_model"]) and not stub
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Ascend forward/collection probe: dry-run by default, --execute only "
            "on a matching vLLM + vllm-ascend NPU stack."
        )
    )
    parser.add_argument("--profile", required=True, choices=sorted(PROFILES))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument(
        "--quantization", default=None, choices=["auto", "none", "ascend"]
    )
    parser.add_argument("--dtype", default=None, choices=["bfloat16", "float16"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt-token-ids", default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument(
        "--enforce-eager", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--enable-expert-parallel",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--trust-remote-code", action="store_true", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--runtime-dir", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    options = {
        "model_path": args.model_path,
        "tensor_parallel_size": args.tensor_parallel_size,
        "quantization": args.quantization,
        "dtype": args.dtype,
        "device": args.device,
        "prompt": args.prompt,
        "prompt_token_ids": args.prompt_token_ids,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
        "enable_expert_parallel": args.enable_expert_parallel,
        "trust_remote_code": args.trust_remote_code,
        "seed": args.seed,
        "runtime_dir": args.runtime_dir,
    }
    try:
        plan = build_plan(args.profile, options)
        if args.execute:
            plan = execute_plan(plan)
    except ProbeError as exc:
        print(f"REFUSED [{exc.code}] {exc}", file=sys.stderr)
        return 2

    text = json.dumps(plan, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = Path.cwd() / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"artifact: {out}")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
