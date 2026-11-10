"""Optional tiny CPU shape probe for the pinned Qwen3.6 HF source files.

Requires torch and transformers in the caller's environment. This runs a
random-weight two-layer text model, including GDN prefill/decode and a full
attention layer. It checks shapes and shared-gate construction; it does not
validate real checkpoint weights, MTP, fused kernels, or deployment timing.
"""

import hashlib
import importlib.util
import sys
from pathlib import Path

import torch
import transformers

config_source = Path(__file__).resolve().parents[1] / "assets/identity/hf_transformers/configuration_qwen3_5_moe.py"
config_spec = importlib.util.spec_from_file_location(
    "transformers.models.qwen3_5_moe.pinned_configuration_qwen3_5_moe", config_source)
config_module = importlib.util.module_from_spec(config_spec)
sys.modules[config_spec.name] = config_module
config_spec.loader.exec_module(config_module)

source = Path(__file__).resolve().parents[1] / "assets/identity/hf_transformers/modeling_qwen3_5_moe.py"
spec = importlib.util.spec_from_file_location(
    "transformers.models.qwen3_5_moe.pinned_modeling_qwen3_5_moe", source)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

text_config = dict(
    vocab_size=64, hidden_size=32, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, head_dim=8,
    linear_conv_kernel_dim=4, linear_key_head_dim=8,
    linear_value_head_dim=8, linear_num_key_heads=2,
    linear_num_value_heads=2, moe_intermediate_size=16,
    shared_expert_intermediate_size=16, num_experts_per_tok=2,
    num_experts=4, layer_types=["linear_attention", "full_attention"],
    rope_parameters={"rope_type": "default", "partial_rotary_factor": 0.25},
)
config = config_module.Qwen3_5MoeTextConfig(**text_config)
model = module.Qwen3_5MoeForCausalLM(config).eval()
with torch.inference_mode():
    prefill = model(input_ids=torch.tensor([[1, 2, 3, 4]]), use_cache=True)
    decode = model(input_ids=torch.tensor([[5]]),
                   past_key_values=prefill.past_key_values, use_cache=True)
print("pinned_source", source)
print("pinned_model_sha256", hashlib.sha256(source.read_bytes()).hexdigest())
print("pinned_config_sha256", hashlib.sha256(config_source.read_bytes()).hexdigest())
print("environment", f"torch={torch.__version__}",
      f"transformers={transformers.__version__}")
print("prefill_logits", tuple(prefill.logits.shape))
print("decode_logits", tuple(decode.logits.shape))
gates = sum("shared_expert_gate" in name
            for name, _ in model.named_modules())
print("shared_gate_modules", gates)
assert tuple(prefill.logits.shape) == (1, 4, 64)
assert tuple(decode.logits.shape) == (1, 1, 64)
assert gates == 2
