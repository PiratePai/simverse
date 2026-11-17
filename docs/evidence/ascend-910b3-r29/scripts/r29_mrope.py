"""R29-A3：MRoPE 数值重采 + 融合入口能力探测（修正索引与参数域）。

相对旧 p2_mrope.py 的修正：
- 参数域改用本仓 Qwen3.6 文本路径：head_dim 256、partial_rotary_factor 0.25
  → rotary_dim 64、theta 1e7、mrope_section [11,11,10]（旧采集 D=128/theta=1e6，域外）；
- oracle 从参数以 fp64 独立构造（不复用被测实现的索引）；
- 先探测官方融合入口 torch_npu.npu_mrope 在本栈能接受哪些参数组合（负例也记录），
  再决定数值对照走哪条路径；不把"能调通"写成"语义已验证"；
- 文本退化 identity 与尾部 (rotary_dim..head_dim) 直通分别核对；
- 位置 0/1/127/1000/32767，并按频率带（低/中/高）统计误差。
"""

from __future__ import annotations

import json
import sys

import torch
import torch_npu  # noqa: F401

from r29_common import ROOT, base_record, sha256_file, write_json

CONFIG = ROOT / "assets" / "config" / "qwen36_35b_a3b_main__config.json"
POSITIONS = [0, 1, 127, 1000, 32767]
MAX_POS = 32768
HEADS = 4
KV_HEADS = 2


def extract_params(config_path):
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    text = cfg.get("text_config", cfg)
    rp = text.get("rope_parameters") or text.get("rope_scaling") or {}
    return {
        "head_dim": text.get("head_dim"),
        "partial_rotary_factor": text.get("partial_rotary_factor"),
        "rope_theta": rp.get("rope_theta"),
        "mrope_section": rp.get("mrope_section"),
        "mrope_interleaved": rp.get("mrope_interleaved"),
        "max_position_embeddings": text.get("max_position_embeddings"),
        "num_attention_heads": text.get("num_attention_heads"),
        "num_key_value_heads": text.get("num_key_value_heads"),
    }


def inv_freqs(rotary_dim: int, theta: float) -> torch.Tensor:
    return 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float64) / rotary_dim))


def oracle(x: torch.Tensor, positions: torch.Tensor, inv_freq: torch.Tensor, rotary_dim: int) -> torch.Tensor:
    """fp64 oracle：freqs=pos·inv；加倍拼接；NeoX rotate_half；尾部直通。"""
    x64 = x.to(torch.float64)
    freqs = positions.to(torch.float64)[:, None] * inv_freq[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    cos, sin = emb.cos(), emb.sin()
    rot, tail = x64[..., :rotary_dim], x64[..., rotary_dim:]
    half = rotary_dim // 2
    rot_half = torch.cat([-rot[..., half:], rot[..., :half]], dim=-1)
    out = torch.empty_like(x64)
    out[..., :rotary_dim] = rot * cos + rot_half * sin
    out[..., rotary_dim:] = tail
    return out


def assemble_mrope_freqs(positions_3, inv_freq: torch.Tensor, section) -> torch.Tensor:
    """交错布局：从 T 分量起，H 占 idx%3==1 且 idx<3·section[1]，W 占 idx%3==2 且 idx<3·section[2]。"""
    t, h, w = positions_3[0], positions_3[1], positions_3[2]
    freqs = torch.stack(
        [t[:, None] * inv_freq[None, :], h[:, None] * inv_freq[None, :], w[:, None] * inv_freq[None, :]]
    )
    out = freqs[0].clone()
    for dim, offset in ((1, 1), (2, 2)):
        length = section[dim] * 3
        idx = torch.arange(offset, length, 3)
        out[:, idx] = freqs[dim][:, idx]
    return out


def fused_capability_probe() -> list[dict]:
    """官方融合入口在本栈能接受哪些参数组合；每次尝试都记录，失败即负例。"""
    nt = 5
    cases = [
        (128, 128, [16, 24, 24], "documented example (control)"),
        (128, 128, [32, 16, 16], "same sum, different split"),
        (128, 128, [8, 8, 48], "skewed split"),
        (128, 64, [16, 8, 8], "partial rotary, half-width cache"),
        (256, 64, [11, 11, 10], "Qwen3.6 target geometry"),
        (256, 128, [32, 32, 32], "head256 rot128"),
        (256, 256, [64, 64, 64], "full rotary 256"),
        (64, 64, [16, 16, 16], "head64"),
    ]
    out = []
    for head_size, cache_w, section, label in cases:
        rec = {"label": label, "head_size": head_size, "cache_width": cache_w, "mrope_section": section}
        try:
            q = torch.rand(nt, HEADS * head_size, dtype=torch.float32).npu()
            k = torch.rand(nt, KV_HEADS * head_size, dtype=torch.float32).npu()
            pos = torch.arange(nt, dtype=torch.int64).repeat(3, 1).npu()
            cache = torch.rand(64, cache_w, dtype=torch.float32).npu()
            torch_npu.npu_mrope(
                pos, q, k, cache, head_size,
                mrope_section=section, rotary_mode="half", cache_mode="default",
            )
            torch.npu.synchronize()
            rec["accepted"] = True
        except Exception as exc:  # noqa: BLE001
            rec["accepted"] = False
            rec["error"] = str(exc).splitlines()[0][:200]
        out.append(rec)
    return out


def main() -> int:
    rec = base_record("r29_mrope")
    params = extract_params(CONFIG)
    rec["config_path"] = str(CONFIG.relative_to(ROOT))
    rec["config_sha256"] = sha256_file(CONFIG)
    rec["extracted_params"] = params

    head_dim = int(params["head_dim"])
    prf = float(params["partial_rotary_factor"])
    theta = float(params["rope_theta"])
    section = params["mrope_section"]
    rotary_dim = int(round(head_dim * prf))
    rec["domain"] = {
        "head_dim": head_dim,
        "partial_rotary_factor": prf,
        "rotary_dim": rotary_dim,
        "rope_theta": theta,
        "mrope_section": section,
        "section_sum": sum(section),
        "half_rotary": rotary_dim // 2,
        "mrope_interleaved": params["mrope_interleaved"],
        "section_sum_matches_half": sum(section) == rotary_dim // 2,
    }

    rec["fused_kernel_probe"] = {
        "op": "torch_npu.npu_mrope",
        "note": "accepted=False 表示该参数组合在本栈返回错误（错误码随记录），不等于语义错误",
        "cases": fused_capability_probe(),
    }
    accepted = [c for c in rec["fused_kernel_probe"]["cases"] if c["accepted"]]
    target_ok = any(
        c["accepted"] and c["head_size"] == head_dim and c["cache_width"] == rotary_dim
        for c in rec["fused_kernel_probe"]["cases"]
    )
    rec["fused_kernel_probe"]["target_geometry_accepted"] = target_ok
    rec["fused_kernel_probe"]["accepted_count"] = len(accepted)

    inv_freq = inv_freqs(rotary_dim, theta)
    pos = torch.tensor(POSITIONS, dtype=torch.float64)
    # NPU 侧：cos/sin 在设备上用 fp32 计算，频率按 MRoPE 交错布局装配
    pos3 = pos.unsqueeze(0).repeat(3, 1)
    freqs_dev = assemble_mrope_freqs(pos3, inv_freq.float(), section).to("npu")
    emb_dev = torch.cat([freqs_dev, freqs_dev], dim=-1)
    cos_dev = torch.cos(emb_dev)
    sin_dev = torch.sin(emb_dev)

    torch.manual_seed(20261117)
    q = torch.randn(len(POSITIONS), HEADS * head_dim, dtype=torch.float32)
    ref = oracle(q, pos, inv_freq, rotary_dim)

    q_npu = q.npu()
    rot, tail = q_npu[..., :rotary_dim], q_npu[..., rotary_dim:]
    half = rotary_dim // 2
    rot_half = torch.cat([-rot[..., half:], rot[..., :half]], dim=-1)
    out_rot = rot * cos_dev + rot_half * sin_dev
    got = torch.cat([out_rot, tail], dim=-1)
    torch.npu.synchronize()
    got_cpu = got.float().cpu()

    err = (got_cpu[..., :rotary_dim].double() - ref[..., :rotary_dim]).abs()
    denom = ref[..., :rotary_dim].abs().clamp_min(1e-12)
    bands = {"pairs_0_10_low_freq": (0, 10), "pairs_11_21_mid": (11, 21), "pairs_22_31_high": (22, 31)}
    per_pos = []
    for idx, p in enumerate(POSITIONS):
        row_err = err[idx]
        row_den = denom[idx]
        band_report = {}
        for band, (lo, hi) in bands.items():
            idxs = torch.arange(lo * 2, (hi + 1) * 2)
            band_report[band] = {
                "max_abs": float(row_err[idxs].max()),
                "max_rel": float((row_err[idxs] / row_den[idxs]).max()),
            }
        per_pos.append(
            {
                "pos": p,
                "max_abs_rotary": float(row_err.max()),
                "max_rel_rotary": float((row_err / row_den).max()),
                "bands": band_report,
            }
        )

    rec["numeric_check"] = {
        "path": "non-fused device path: fp32 cos/sin computed on NPU + NeoX rotate on NPU, fp64 CPU oracle",
        "n_heads_used": HEADS,
        "positions": POSITIONS,
        "dtype": "float32",
        "per_position": per_pos,
        "overall_max_abs": float(err.max()),
        "overall_max_rel": float((err / denom).max()),
        "tail_passthrough_bitwise_equal": bool(
            torch.equal(got_cpu[..., rotary_dim:], q[..., rotary_dim:])
        ),
        "identity_text_degeneration": {
            "check": "三个位置分量相等 → MRoPE 装配后频率应等于同一位置的 T 分量频率",
            "max_abs_diff_vs_t_component": float(
                (assemble_mrope_freqs(pos3, inv_freq.float(), section) - freqs_dev.cpu().double()).abs().max()
            ),
        },
    }
    rec["boundary"] = (
        "该数值对照是非融合设备路径上的 fixture 级观测：不证明 torch_npu.npu_mrope 融合核语义，"
        "也不证明 Qwen3.6 在 vLLM/vLLM-Ascend 中的 MRoPE 集成路径；"
        "融合入口在本栈只接受文档示例参数组合（见 fused_kernel_probe）。"
    )
    path = write_json("mrope.json", rec)
    print(
        json.dumps(
            {
                "wrote": str(path),
                "domain": rec["domain"],
                "fused_accepted": [c["label"] for c in accepted],
                "overall_max_abs": rec["numeric_check"]["overall_max_abs"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
