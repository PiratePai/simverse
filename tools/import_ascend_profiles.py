"""Ascend NPU profiling 导入与离线审计（A-N5「profiling 校对」第一阶段）。

用途
----
把外部 Ascend msprof/torch_npu profiler 产物（`ASCEND_PROFILER_OUTPUT/` 下的
kernel/op/summary CSV + sqlite 身份表）蒸馏为**仓内自包含、字节可重放**的审计包：

  * identity.json    每个 run 记录到的栈身份/拓扑，以及**缺失**身份字段（不补造）
  * census.json      逐 OP Type 的 count / total / min / median / max / core（设备 kernel 边界）
  * shape_costs.json 关键算子类的 shape → 采样统计（未来局部校准的候选查表键）
  * mapping.json     采集侧 kernel/OP Type → 本仓账本行组的显式 ownership 表
  * diff.json        结构差异表：账本覆盖行/未建模时间占比/通信类/设备内核间隙
  * README.md        使用边界（只读审计，不是 measured 准入）

铁律（与 MERGE_CRITERIA §3 / AGENTS.md 一致）
--------------------------------------------
1. 本工具只**读**外部目录，只**写** `--out`；不 import torch/torch_npu/vllm，不在 NPU 上执行。
2. 不做任何计价，不产生 `selected_ms`；`calibration_admission` 恒为 `not-admitted`。
3. 缺失字段一律 `null` + 原因，绝不填 0；不把不同 run/不同栈拼成一个 profile。
4. 产物 JSON 固定 `sort_keys` + 紧凑分隔符 + LF；manifest 内逐文件 SHA256，
   `--check` 可离线重放校验（源目录只需只读可读）。
5. 设备/实现/形状域未匹配登记设备前，本包**不得**接入 `PROFILE_SETS`/`CalibratedPricer`
   （A-N4 是前置，见 docs/ascend-validation-plan.md §1/§3）。

用法
----
  python tools/import_ascend_profiles.py --src D:/github/qwen3_6_mtp_domino \\
      --src D:/github/qwen3-8b-0811 \\
      --out assets/profile_audit/ascend_910b_profile_audit_v1
  python tools/import_ascend_profiles.py --check --out <同一目录>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SCHEMA = "ascend-profile-audit/1"
SET_ID = "ascend-910b-profile-audit-v1"
OUT_DIR_NAME = "ascend_910b_profile_audit_v1"
CREATED_AT = "2026-11-17"

# ---------------------------------------------------------------------------
# Owner 声明的身份（2026-11-17 用户消息；记录为 owner-declared，不等于
# repo/revision/checkpoint 验证——身份缺口仍按准则登记）
# ---------------------------------------------------------------------------
OWNER_DECLARATION = {
    "declared_by": "repository owner（会话消息，2026-11-17）",
    "roots": {
        "qwen3_6_mtp_domino": {
            "target_model_id": "qwen36_35b_a3b",
            "target_model_declared": "Qwen3.6-35B-A3B"},
        "qwen3-8b-0811": {
            "target_model_id": "qwen3_8b_dense",
            "target_model_declared": "Qwen3-8B"},
    },
    "draft_methods": ["mtp", "dflash", "dspark"],
    "draft_method_note": "mtp/dflash/dspark 为三种现有草稿模型",
    "excluded": {"minidraft": "自研草稿模型；owner 声明本轮暂不考虑"},
    "limits": ["声明只覆盖“模型/方法叫什么”；权重 revision/checkpoint 哈希、"
               "vLLM/vllm-ascend commit、量化配方仍未验证"],
}
_DRAFT_KEYWORDS = ("minidraft", "dflash", "dspark", "mtp")


def _run_method(run_label: str) -> str:
    low = run_label.lower()
    for kw in _DRAFT_KEYWORDS:
        if kw in low:
            return kw
    return "baseline"

ROUND_US = 3          # Duration/统计保留 3 位小数（1e-3 µs = 1 ns）
MAX_SHAPES_PER_CLASS = 12   # 每算子类保留的 shape 点数（按总时间取前 N，其余合并）

# AICore 时钟来自 profile（仅记录，不作设备规格）
# 观测到的设备名：Ascend910B（家族级；SKU 未由本数据建立）
EXPECTED_GEOMETRY = {
    "qwen36_35b_a3b": {
        "catalog_model": True,
        "hidden_size": 2048,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "linear_conv_kernel_dim": 4,
        "num_hidden_layers": 40,
        "skip_layers": 4,
        "vocab_size": 248320,
    },
    "qwen3_8b_dense": {
        "catalog_model": False,
        "hidden_size": 4096,
        "num_attention_heads_total": 32,
        "num_key_value_heads_total": 8,
        "head_dim": 128,
        "intermediate_size": 12288,
        "num_hidden_layers": 36,
        "vocab_size": 151936,
        "note": "Qwen3-8B 量级的 dense 模型（owner 声明为 Qwen3-8B；权重 revision/"
                "checkpoint 未提供，身份未核实；不在活跃 catalog，禁止用于主模型校准）",
    },
}

# ---------------------------------------------------------------------------
# kernel/OP Type → 语义组 → 本仓账本行 ownership
# coverage:
#   mapped                             组内 kernel 的语义可归到给定账本行（组级聚合）
#   fused_group                        多 kernel/单 kernel 同时拥有多行（融合边界）
#   communication                      成本由 pipeline 通信/链路口径建模，不是账本行
#   unmapped_no_ledger_row             本仓无对应账本行（结构性缺口）
#   unmapped_implementation_mismatch   语义行存在，但实现分解不同（不可 1:1 计价）
# ---------------------------------------------------------------------------
_GROUP_RULES: List[Tuple[str, Tuple[str, ...], str, Tuple[str, ...], str, str]] = [
    # (group_id, op_types, coverage, ledger_rows, boundary, note)
    ("gemm_dense_quant",
     ("MatMulV2", "MatMulV3", "QuantBatchMatmulV3", "WeightQuantBatchMatmulV2",
      "BatchMatMulV2", "BatchMatMulV3"),
     "mapped",
     ("q_proj", "k_proj", "v_proj", "o_proj", "lm_head", "in_proj_qkv",
      "in_proj_z", "in_proj_ba", "out_proj", "moe_shared", "shared_gate"),
     "device-kernel",
     "单核 GEMM（含 W8A8/W4A8 动态量化变体）。layer/op 归因需 correlation id，"
     "本审计只到组级；具体行归属未知。"),
    ("gemm_grouped_moe",
     ("GroupedMatmul", "GroupedMatmulSwigluQuantV2"),
     "mapped",
     ("moe_routed", "moe_act"),
     "device-kernel",
     "EP 分组专家 GEMM；SwigluQuantV2 为 fusion（gemm1+激活+量化）→ 同时拥有 "
     "moe_routed 与 moe_act 两行的设备时间。"),
    ("attention_core",
     ("FusedInferAttentionScore",),
     "mapped",
     ("attention_core",),
     "device-kernel",
     "融合注意力核（GQA/MQA）；输出门控 attn_output_gate 是否融合进该核未由"
     "profile 证明（见 rows[attn_output_gate]）。"),
    ("attention_kv_write",
     ("ScatterPaKvCache",),
     "fused_group",
     ("attention_core",),
     "device-kernel",
     "KV cache 写入单独成核；语义属于 attention_core 的 kv_write 字节，与"
     "FusedInferAttentionScore 构成同一账本行的多核时间（组内合并计一次）。"),
    ("attn_qkv_norm_rope_fused",
     ("split_qkv_rmsnorm_rope_kernel", "split_qkv_rmsnorm_mrope_kernel"),
     "fused_group",
     ("q_norm", "k_norm", "rope"),
     "device-kernel",
     "已投影 QKV 的 split + QK-Norm + RoPE；不含投影 GEMM。参考源码中 qkv_proj "
     "先执行；采集栈 commit 未知，当前仅是语义候选，不能重复认领 q/k/v 投影行。"),
    ("norm_add_fused",
     ("AddRmsNormBias",),
     "mapped",
     ("attn_add", "attn_norm", "mlp_add", "mlp_norm"),
     "device-kernel",
     "融合 残差Add+RMSNorm；每层两处（attn/mlp），具体归属由调用序决定，本审计"
     "只到组级。"),
    ("norm_generic",
     ("RmsNorm", "GemmaRmsNorm", "_layer_norm_fwd_1pass_kernel_npu"),
     "mapped",
     ("attn_norm", "mlp_norm", "q_norm", "k_norm", "final_norm"),
     "device-kernel",
     "通用/融合归一化核（含 MTP 层 final norm）；行归属为组级近似。"),
    ("gdn_core",
     ("RecurrentGatedDeltaRule", "fused_gdn_gating_kernel", "l2norm_fwd_kernel2",
      "l2norm_fwd_kernel2_loop"),
     "mapped",
     ("gdn_recurrence", "v_norm_gated", "in_proj_ba"),
     "device-kernel",
     "GDN 递推/门控/l2norm 的 Ascend 融合核；本仓按 token 递推公式计 gdn_recurrence，"
     "实现分解与核数不同（组级时间可比，行级不 1:1）。"),
    ("gdn_conv",
     ("CausalConv1d",),
     "mapped",
     ("causal_conv1d",),
     "device-kernel",
     "depthwise 因果卷积核（线性注意力 conv state 更新）。"),
    ("gdn_chunked_prefill",
     ("ChunkFwdO", "ChunkGatedDeltaRuleFwdH", "recompute_w_u_fwd_kernel",
      "merge_16x16_to_64x64_inverse_kernel", "solve_tril_16x16_kernel",
      "chunk_scaled_dot_kkt_fwd_kernel", "chunk_local_cumsum_scalar_kernel"),
     "unmapped_implementation_mismatch",
     ("gdn_recurrence", "causal_conv1d"),
     "device-kernel",
     "chunked-prefill GDN 的 7 核分解（chunk O/状态/TRIL 求逆/KKT/cumsum）；语义数学"
     "与本仓 token 递推行对应，但实现分解不同 → 不计入账本覆盖。"),
    ("moe_router_topk",
     ("MoeGatingTopK",),
     "mapped",
     ("moe_router",),
     "device-kernel",
     "router 打分 + top-k + 归一化（含返回 logits），对应本仓 moe_router 行。"),
    ("hccm_collective",
     (),
     "communication",
     (),
     "device-kernel",
     "HCCL/hcom_* 集合通信核（alltoallv/allGather/allReduce 等）。本仓通信成本"
     "在 pipeline 执行层按链路/collective 口径计，不与设备 kernel 时间相加；"
     "kernel duration 含同步等待，属 device-kernel 边界。"),
    ("moe_token_shuffle",
     ("MoeTokenPermute", "MoeTokenUnpermute", "RepeatInterleave"),
     "unmapped_no_ledger_row",
     (),
     "device-kernel",
     "MoE token 重排/去重/复制：本仓无账本行（本仓把 MoE 计成 GEMM+通信，未计"
     "permute 的设备时间）。"),
    ("moe_ep_comm",
     ("MoeDistributeDispatchV2", "MoeDistributeCombineV2"),
     "communication",
     (),
     "device-kernel",
     "EP dispatch/combine 的设备侧通信核；本仓以链路字节/延迟口径建模通信，"
     "不以账本行表示（计时边界不同，不可与链路成本直接相加）。"),
    ("tp_collective",
     ("allreduceAicpuKernel", "allgatherAicpuKernel", "AivKernel"),
     "communication",
     (),
     "device-kernel",
     "TP AllReduce / logits all-gather 的 AICPU kernel；本仓通信成本在 pipeline.执行层"
     "按链路口径计，不与设备 kernel 相加。"),
    ("mlp_activation",
     ("SwiGlu",),
     "mapped",
     ("moe_act", "moe_shared_act"),
     "device-kernel",
     "SwiGLU 激活（dense MLP 路径；MoE 侧已被 GroupedMatmulSwigluQuantV2 融合）。"),
    ("sampler_accept",
     ("ApplyTopKTopPWithSorted", "_temperature_kernel", "_gumbel_sample_kernel",
      "sample_recovered_tokens_kernel", "rejection_random_sample_kernel",
      "prepare_inputs_padded_kernel", "DSARandomUniform", "Sort", "MaskedFill",
      "ArgMaxV2", "HistogramV2", "TopK"),
     "unmapped_no_ledger_row",
     (),
     "device-kernel",
     "采样/接受/拒绝重采样/随机数：本仓把接受律作为显式假设输入，不建模设备侧采样核。"),
    ("draft_runtime",
     ("_prepare_dflash_inputs_kernel_ascend", "_resample_kernel",
      "_compute_local_logits_stats_kernel", "_grouped_k_norm_kernel",
      "_triton_rope", "_compute_slot_mapping_kernel",
      "_compute_slot_mappings_kernel", "_clear_ssm_states_kernel",
      "_apply_write_kernel", "_bias_kernel",
      "_combine_sampled_and_draft_tokens_kernel", "_expand_idx_mapping_kernel",
      "_fused_gru_cell_gather_kernel", "_gather_block_tables_kernel",
      "_get_num_sampled_and_rejected_kernel", "_insert_resampled_kernel",
      "_post_update_kernel", "_prepare_pos_seq_lens_kernel",
      "_prepare_prefill_inputs_kernel", "_prepare_rope_positions_kernel",
      "_probabilistic_rejection_kernel", "_scatter_num_accepted_kernel",
      "Swish"),
     "unmapped_implementation_mismatch",
     (),
     "device-kernel",
     "草稿/服务运行时的 Ascend 专用 kernel（DFlash/MTP 输入准备、采样统计、slot "
     "mapping、block table、状态更新、非 catalog 草稿 MLP 的 Swish 激活等）；"
     "本仓草稿按账本/事件建模，不按 kernel 计，也不归入目标模型行。"),
    ("quant_cast",
     ("DynamicQuant", "RmsNormDynamicQuant", "AddRmsNormDynamicQuant"),
     "unmapped_no_ledger_row",
     (),
     "device-kernel",
     "动态量化/反量化 cast（含融合 norm+quant）：本仓按字节/位宽计，未计设备侧"
     " cast kernel 时间。"),
    ("data_move",
     ("Transpose", "Slice", "TensorMove", "ConcatD", "Cast", "ViewCopy",
      "BroadcastTo", "Fill", "MemSet", "ZerosLike", "Index", "IndexCheck",
      "IndexPutV2", "IndexFill", "ScatterUpdate", "ScatterElementsV2",
      "GatherV2", "GatherV3", "GatherElements", "GatherElementsV2", "PadV3",
      "expand_kernel", "Range", "LinearIndex", "NotEqual", "GreaterEqual",
      "IndexPut", "Tile", "OnesLike"),
     "unmapped_no_ledger_row",
     (),
     "device-kernel",
     "布局/索引/填充/拷贝类 kernel：本仓把字节计入各算子行，未单列设备 kernel 时间。"),
    ("elementwise_glue",
     ("Add", "Mul", "Sub", "Neg", "RealDiv", "Sigmoid", "SoftmaxV2", "Log",
      "Greater", "Equal", "Less", "LogicalAnd", "LogicalNot", "Minimum",
      "FloorMod", "FloorDiv", "ClipByValueV2", "SelectV2", "ReduceSum",
      "Cumsum", "Softmax"),
     "unmapped_no_ledger_row",
     (),
     "device-kernel",
     "逐元素/规约黏合 kernel：本仓只计 FLOPs/字节，无对应设备任务级时间。"),
]

_GROUP_BY_OP: Dict[str, Tuple[str, Tuple[str, ...], str, str]] = {}
for _gid, _ops, _cov, _rows, _bound, _note in _GROUP_RULES:
    for _op in _ops:
        if _op in _GROUP_BY_OP:
            raise SystemExit(f"映射表重复登记 OP Type {_op}")
        _GROUP_BY_OP[_op] = (_gid, _rows, _cov, _note)

GROUP_META = {g[0]: {"coverage": g[2], "ledger_rows": list(g[3]),
                     "boundary": g[4], "note": g[5]}
              for g in _GROUP_RULES}

# 本仓账本行的静态覆盖状态（与具体模型无关的行级判定；组级归属见 mapping.json）
ROW_STATUS = {
    "q_proj": "mapped_group", "k_proj": "mapped_group", "v_proj": "mapped_group",
    "o_proj": "mapped_group", "lm_head": "mapped_group",
    "in_proj_qkv": "mapped_group", "in_proj_z": "mapped_group",
    "in_proj_ba": "mapped_group", "out_proj": "mapped_group",
    "moe_shared": "mapped_group", "shared_gate": "mapped_group",
    "moe_routed": "mapped_group", "moe_act": "mapped_group",
    "moe_shared_act": "mapped_group", "moe_router": "mapped_group",
    "attention_core": "mapped_group",
    "attn_output_gate": "fused_unknown",
    "q_norm": "mapped_group", "k_norm": "mapped_group", "rope": "mapped_group",
    "attn_add": "mapped_group", "attn_norm": "mapped_group",
    "mlp_add": "mapped_group", "mlp_norm": "mapped_group",
    "causal_conv1d": "mapped_group", "gdn_recurrence": "mapped_group",
    "v_norm_gated": "mapped_group",
    "final_norm": "mapped_group",
    "embed_lookup": "not_observed_in_device_kernels",
}

ROW_REASON = {
    "fused_unknown": "该行在 Ascend 实现中可能融合进 FusedInferAttentionScore；"
                     "profile 未提供 correlation，无法证明是否单独成核。",
    "not_observed_in_device_kernels": "该行未在设备 kernel 列表中出现（嵌入查表可能"
                                      "由框架/拷贝路径完成，profile Level1 未单列）。",
}


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _canon(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _num(text: Optional[str]) -> Optional[float]:
    if text is None:
        return None
    s = str(text).strip().strip('"')
    if not s or s.upper() in ("N/A", "NA", "NAN", "NONE"):
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def _int(text: Optional[str]) -> Optional[int]:
    v = _num(text)
    return None if v is None else int(v)


def _round(v: Optional[float], nd: int = ROUND_US) -> Optional[float]:
    if v is None or not math.isfinite(v):
        return None
    return round(v, nd)


def _quantile(sorted_vals: List[float], q: float) -> Optional[float]:
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (len(sorted_vals) - 1) * q
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def _stats(vals: Iterable[float]) -> dict:
    vs = sorted(v for v in vals if v is not None and math.isfinite(v))
    if not vs:
        return {"count": 0, "total_us": None, "min_us": None, "median_us": None,
                "p90_us": None, "max_us": None, "mean_us": None}
    total = sum(vs)
    return {
        "count": len(vs),
        "total_us": _round(total),
        "min_us": _round(vs[0]),
        "median_us": _round(_quantile(vs, 0.5)),
        "p90_us": _round(_quantile(vs, 0.9)),
        "max_us": _round(vs[-1]),
        "mean_us": _round(total / len(vs)),
    }


def _split(text: Optional[str]) -> List[str]:
    if not text:
        return []
    return [p for p in (x.strip() for x in str(text).split(";")) if p]


def _shape_dims(text: Optional[str]) -> List[Optional[Tuple[int, ...]]]:
    out: List[Optional[Tuple[int, ...]]] = []
    for part in _split(text):
        part = part.strip().strip('"')
        if not part:
            continue
        try:
            out.append(tuple(int(x) for x in part.split(",") if x.strip()))
        except ValueError:
            out.append(None)
    return out


def _shape_key(text: Optional[str]) -> Optional[str]:
    dims = _shape_dims(text)
    if not dims:
        return None
    return ";".join(",".join(str(d) for d in d) if d is not None else "?"
                    for d in dims)


def _op_type(row: dict) -> str:
    return (row.get("Type") or row.get("OP Type") or row.get("Name") or "").strip()


def _classify(op: str):
    hit = _GROUP_BY_OP.get(op)
    if hit:
        return hit
    if op.startswith("hcom_"):
        return ("hccm_collective", (), "communication",
                "HCCL/hcom_* 集合通信核（本仓通信在链路/collective 口径另计）")
    return ("other_unmapped", (), "unmapped_no_ledger_row",
            "未登记 OP Type（本审计显式列出，不静默丢弃）")


# ---------------------------------------------------------------------------
# profile 目录读取
# ---------------------------------------------------------------------------

def discover_runs(src_roots: List[Path]) -> List[dict]:
    runs: List[dict] = []
    for src in src_roots:
        if not src.is_dir():
            raise SystemExit(f"--src 不是目录：{src}")
        outs = sorted(p for p in src.rglob("ASCEND_PROFILER_OUTPUT") if p.is_dir())
        for out in outs:
            run_dir = out.parent
            label = run_dir.name
            run_id = f"{src.resolve().name}/{label}"
            runs.append({"run_id": run_id, "run_label": label,
                         "src_root_label": src.resolve().name,
                         "run_dir": run_dir, "out_dir": out})
    if not runs:
        raise SystemExit("未在 --src 下发现任何 ASCEND_PROFILER_OUTPUT 目录")
    ids = [r["run_id"] for r in runs]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"run_id 重复（不同根下同名 run）：{ids}")
    return runs


def _sqlite_tables(db: Path) -> Dict[str, List[tuple]]:
    """只读打开 profiler sqlite；不可用时返回空（身份缺失如实记录）。"""
    if not db.is_file():
        return {}
    uri = db.resolve().as_uri() + "?mode=ro&immutable=1"
    con = sqlite3.connect(uri, uri=True)
    try:
        cur = con.cursor()
        tables = [r[0] for r in cur.execute(
            "select name from sqlite_master where type='table'").fetchall()]
        out: Dict[str, List[tuple]] = {}
        for name in ("HOST_INFO", "NPU_INFO", "RANK_DEVICE_MAP",
                     "SESSION_TIME_INFO", "AICORE_FREQ", "META_DATA"):
            if name not in tables:
                out[name] = []
                continue
            try:
                out[name] = [tuple(r) for r in
                             cur.execute(f'select * from "{name}" limit 256').fetchall()]
            except sqlite3.DatabaseError:
                out[name] = []
        return out
    finally:
        con.close()


def _parse_identity(run: dict) -> dict:
    src = run["run_dir"]
    out = run["out_dir"]
    recorded: Dict[str, object] = {}
    absent: List[dict] = []
    files: Dict[str, str] = {}

    info_path = src / "profiler_info_0.json"
    if info_path.is_file():
        files["profiler_info_0.json"] = _sha256(info_path)
        info = json.loads(info_path.read_text(encoding="utf-8"))
        recorded["torch_npu_version"] = info.get("torch_npu_version")
        recorded["cann_version"] = info.get("cann_version")
        recorded["rank_id"] = info.get("rank_id")
        exp = (info.get("config") or {}).get("experimental_config") or {}
        recorded["profiler_level"] = exp.get("_profiler_level")
        recorded["aic_metrics"] = exp.get("_aic_metrics")
        recorded["data_simplification"] = exp.get("_data_simplification")
        recorded["record_shapes"] = (info.get("config") or {}).get(
            "common_config", {}).get("record_shapes")
        recorded["activities"] = (info.get("config") or {}).get(
            "common_config", {}).get("activities")
    else:
        absent.append({"field": "torch_npu_version/cann_version/rank_id",
                       "reason": "缺 profiler_info_0.json"})

    meta_path = src / "profiler_metadata.json"
    if meta_path.is_file():
        files["profiler_metadata.json"] = _sha256(meta_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        recorded["env_variables"] = meta.get("ENV_VARIABLES")
        groups = meta.get("parallel_group_info") or {}
        topo = {}
        for key, g in sorted(groups.items()):
            topo[g.get("group_name") or key] = {
                "group_rank": g.get("group_rank"),
                "global_ranks": g.get("global_ranks"),
                "world_size": len(g.get("global_ranks") or []),
            }
        recorded["parallel_groups"] = topo
    else:
        absent.append({"field": "parallel_group_info", "reason":
                       "缺 profiler_metadata.json"})

    db = out / "ascend_pytorch_profiler_0.db"
    if db.is_file():
        files["ascend_pytorch_profiler_0.db"] = _sha256(db)
        tables = _sqlite_tables(db)
        npu = tables.get("NPU_INFO") or []
        recorded["npu_name"] = npu[0][1] if npu and len(npu[0]) == 2 else None
        host = tables.get("HOST_INFO") or []
        recorded["host_name"] = host[0][1] if host and len(host[0]) == 2 else None
        rdm = tables.get("RANK_DEVICE_MAP") or []
        recorded["rank_device_map"] = [{"rank": r[0], "device_id": r[1]}
                                       for r in rdm if len(r) == 2]
        sess = tables.get("SESSION_TIME_INFO") or []
        if sess and len(sess[0]) == 2:
            recorded["session_start_ns"] = sess[0][0]
            recorded["session_end_ns"] = sess[0][1]
            recorded["session_ms"] = _round((sess[0][1] - sess[0][0]) / 1e6, 3)
        freqs = tables.get("AICORE_FREQ") or []
        recorded["aicore_freq_mhz"] = sorted({r[2] for r in freqs if len(r) >= 3})
        schema = {r[0]: r[1] for r in (tables.get("META_DATA") or [])
                  if len(r) == 2 and str(r[0]).startswith("SCHEMA_VERSION")}
        recorded["profiler_db_schema"] = schema.get("SCHEMA_VERSION")
    else:
        absent.append({"field": "npu_name/rank_device_map/session",
                       "reason": "缺 ascend_pytorch_profiler_0.db"})

    # 关键缺失字段（采集身份完整性；plan §3.1 清单）
    absent.extend([
        {"field": "device_sku_board",
         "reason": "profile 只记录 NPU 家族名（Ascend910B），未含 npu-smi board/"
                   "SKU/物理卡号；910B3/910C/A2/A3 能力不可互相继承"},
        {"field": "npu_driver_version",
         "reason": "profile 记录 CANN/torch_npu，未记录 driver/CANN 精确 patch 与"
                   "固件版本"},
        {"field": "vllm_version/vllm_ascend_commit",
         "reason": "推理引擎版本未随 profile 提供；kernel ownership 无法按源码 pin"},
        {"field": "model_repo/revision/config_sha256/checkpoint_sha256",
         "reason": "权重身份未提供；owner 已声明目标模型名（Qwen3.6-35B-A3B / "
                   "Qwen3-8B），但 repo/revision/config/checkpoint 哈希仍未验证；"
                   "形状一致 + 名称声明不等于权重身份（准则 §1.1/§3.1）"},
        {"field": "quantization_recipe_declaration",
         "reason": "量化配方未声明；只能从 kernel 输入 dtype 观察（W8A8/W4A8 混合）"},
        {"field": "tp/ep/pp/dp_full_topology",
         "reason": "只有并行组名与 global_ranks；TP 组缺失/EP 组存在，"
                   "无法证明本 rank 的完整切分"},
        {"field": "batch/seq/phase/seed",
         "reason": "未记录请求 batch/长度/阶段/seed；token 数只能由 shape 反推"},
        {"field": "eager_or_graph/graph_capture_size",
         "reason": "执行模式与图捕获尺寸未记录"},
        {"field": "warmup_active_window/repeat_runs/sync_policy",
         "reason": "warmup/active 区间、重复次数、同步策略未标记；不能宣称统计稳定"},
        {"field": "step_boundaries",
         "reason": "trace step 节点为空（profiler日志 WARNING 'step node list is "
                   "empty'），step_trace_time 无 Step 分段 → 无法切分 prefill/decode 窗口"},
        {"field": "request_trace/e2e",
         "reason": "无到达轨迹/输出 token 数；不能做 E2E 或接受率结论"},
        {"field": "collection_command",
         "reason": "采集命令/开关未归档（除 profiler_info 的 experimental_config）"},
    ])
    return {"run_id": run["run_id"], "run_label": run["run_label"],
            "source_root_label": run["src_root_label"],
            "recorded": recorded, "absent": absent, "source_files_sha256": files,
            "owner_declared": _owner_declaration(run)}


def _owner_declaration(run: dict) -> dict:
    """Owner 声明的目标模型/草稿方法（不是权重/版本验证）。"""
    decl = OWNER_DECLARATION["roots"].get(run["src_root_label"], {})
    method = _run_method(run["run_label"])
    reason = OWNER_DECLARATION["excluded"].get(method)
    return {
        "target_model_id": decl.get("target_model_id"),
        "target_model_declared": decl.get("target_model_declared"),
        "draft_method": method,
        "excluded": reason is not None,
        "exclusion_reason": reason,
        "declared_by": OWNER_DECLARATION["declared_by"],
        "status": "owner-declared（模型/方法名）；权重 revision 与引擎 commit 未验证",
    }


def _parse_step_trace(out: Path) -> dict:
    path = out / "step_trace_time.csv"
    if not path.is_file():
        return {"present": False, "rows": []}
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            row = {}
            for k, v in r.items():
                if k in ("Device_id", "Step"):
                    row[k] = v if v not in (None, "") else None
                else:
                    row[k] = _num(v)
            rows.append(row)
    return {"present": True, "rows": rows}


def _parse_analysis_db(out: Path) -> dict:
    """analysis.db 的 StepTraceTime（与 step_trace_time.csv 对照；字段口径可能不同）。"""
    db = out / "analysis.db"
    if not db.is_file():
        return {"present": False, "step_trace_time": []}
    uri = db.resolve().as_uri() + "?mode=ro&immutable=1"
    con = sqlite3.connect(uri, uri=True)
    try:
        cur = con.cursor()
        cols = [r[1] for r in cur.execute("PRAGMA table_info(StepTraceTime)")]
        if not cols:
            return {"present": False, "step_trace_time": []}
        rows = []
        for r in cur.execute("select * from StepTraceTime").fetchall():
            row = dict(zip(cols, r))
            for k, v in list(row.items()):
                if isinstance(v, (int, float)) and k not in ("deviceId",):
                    row[k] = _num(str(v))
            rows.append(row)
        return {"present": True, "step_trace_time": rows,
                "note": "analysis.db 与 step_trace_time.csv 是同一采集的两份输出；"
                        "字段语义可能不同（computing/communication 归类），"
                        "两者不可互换或相减（MERGE_CRITERIA §2.4 计时边界）"}
    finally:
        con.close()


def _step_trace_consistency(csv_rows: List[dict], db_rows: List[dict]) -> dict:
    def total(row, keys):
        vals = [row.get(k) for k in keys]
        if any(v is None for v in vals):
            return None
        return sum(vals)

    csv_stage = [r.get("Stage") for r in csv_rows]
    db_stage = [r.get("stage") for r in db_rows]
    return {
        "csv_stage_us": [_round(v) for v in csv_stage if isinstance(v, (int, float))],
        "db_stage_us": [_round(v) for v in db_stage if isinstance(v, (int, float))],
        "stage_values_match": (
            bool(csv_stage) and len(csv_stage) == len(db_stage) and
            all(isinstance(a, (int, float)) and isinstance(b, (int, float)) and
                abs(a - b) / max(a, 1.0) < 1e-4
                for a, b in zip(csv_stage, db_stage))),
        "csv_computing_us": [_round(r.get("Computing")) for r in csv_rows],
        "db_computing_us": [_round(r.get("computing")) for r in db_rows],
        "csv_communication_us": [_round(r.get("Communication")) for r in csv_rows],
        "db_communication_us": [_round(r.get("communication")) for r in db_rows],
        "computing_field_equal": (
            bool(csv_rows) and len(csv_rows) == len(db_rows) and all(
                csv_rows[i].get("Computing") is not None and
                db_rows[i].get("computing") is not None and
                abs(csv_rows[i]["Computing"] - db_rows[i]["computing"]) < 1e-3
                for i in range(min(len(csv_rows), len(db_rows))))),
        "computing_delta_db_minus_csv_us": [
            _round(db_rows[i].get("computing") - csv_rows[i].get("Computing"))
            for i in range(min(len(csv_rows), len(db_rows)))
            if csv_rows[i].get("Computing") is not None and
            db_rows[i].get("computing") is not None],
        "overlapped_communication_delta_us": [
            (_round(db_rows[i]["overlapped"] - csv_rows[i]["Overlapped"])
             if db_rows[i].get("overlapped") is not None and
             csv_rows[i].get("Overlapped") is not None else None)
            for i in range(min(len(csv_rows), len(db_rows)))],
        "note": "CSV 与 analysis.db 的 computing/communication 归类不一致时记录"
                "两侧原值，不做换算；step 节点为空 → 无逐步窗口。",
    }


def _parse_op_statistic(out: Path) -> List[dict]:
    path = out / "op_statistic.csv"
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "op_type": (r.get("OP Type") or "").strip(),
                "core_type": (r.get("Core Type") or "").strip() or None,
                "count": _int(r.get("Count")),
                "total_us": _num(r.get("Total Time(us)")),
                "min_us": _num(r.get("Min Time(us)")),
                "avg_us": _num(r.get("Avg Time(us)")),
                "max_us": _num(r.get("Max Time(us)")),
                "ratio_pct": _num(r.get("Ratio(%)")),
            })
    return [r for r in rows if r["op_type"]]


# 首维为 token 数的算子（用于 token 桶/阶段启发式；非白名单一律 unknown）
_TOKEN_DIM_OPS = frozenset({
    "MatMulV2", "MatMulV3", "QuantBatchMatmulV3", "WeightQuantBatchMatmulV2",
    "BatchMatMulV3", "GroupedMatmul", "GroupedMatmulSwigluQuantV2",
    "FusedInferAttentionScore", "AddRmsNormBias", "RmsNorm", "GemmaRmsNorm",
    "_layer_norm_fwd_1pass_kernel_npu", "MoeGatingTopK", "MoeTokenPermute",
    "MoeTokenUnpermute", "RecurrentGatedDeltaRule", "CausalConv1d",
    "split_qkv_rmsnorm_rope_kernel", "split_qkv_rmsnorm_mrope_kernel",
    "DynamicQuant", "RmsNormDynamicQuant", "SwiGlu", "l2norm_fwd_kernel2",
    "l2norm_fwd_kernel2_loop", "fused_gdn_gating_kernel", "ChunkFwdO",
    "ChunkGatedDeltaRuleFwdH", "recompute_w_u_fwd_kernel",
    "merge_16x16_to_64x64_inverse_kernel", "solve_tril_16x16_kernel",
    "chunk_scaled_dot_kkt_fwd_kernel", "chunk_local_cumsum_scalar_kernel",
    "MoeDistributeDispatchV2", "MoeDistributeCombineV2", "ScatterPaKvCache",
    "_prepare_dflash_inputs_kernel_ascend", "_resample_kernel",
    "_compute_local_logits_stats_kernel", "_grouped_k_norm_kernel",
    "_triton_rope", "Neg",
})


def _token_bucket(op: str, input_shapes: Optional[str]) -> Optional[str]:
    """按首维 token 数做**逐调用**桶（不是连续阶段窗口，启发式）。"""
    if op not in _TOKEN_DIM_OPS:
        return None
    dims = _shape_dims(input_shapes)
    if not dims or not dims[0]:
        return None
    tok = dims[0][0]
    if tok <= 0:
        return None
    if tok <= 64:
        return "decode(<=64 tok)"
    if tok <= 1024:
        return "draft/verify(65..1024 tok)"
    return "prefill(>1024 tok)"


def _parse_kernels(out: Path) -> dict:
    """流式解析 kernel_details.csv：逐 OP Type 汇总 + shape 采样 + 设备内核间隙。

    通信口径分两层：
      * hcom_* 为 HCCL 通信核，`op_statistic.csv` 不含它们（用于对账）；
      * 全部 communication 类（含 MoeDistribute*/allreduce/allgather 等设备侧
        通信核）用于"账本覆盖"分母的排除。
    """
    path = out / "kernel_details.csv"
    if not path.is_file():
        raise ValueError(f"缺必需源文件 {path}")
    per_op: Dict[str, dict] = {}
    shape_vals: Dict[str, Dict[str, List[float]]] = {}
    dtypes: Dict[str, set] = {}
    streams: Dict[str, List[Tuple[float, float]]] = {}
    buckets: Dict[str, dict] = {}
    gemm_samples = {}
    total_us = 0.0
    comm_us = 0.0
    hcom_us = 0.0
    comm_rows = 0
    hcom_rows = 0
    rows = 0
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            op = _op_type(row)
            if not op:
                continue
            dur = _num(row.get("Duration(us)"))
            if dur is None or dur < 0:
                raise ValueError(f"kernel 时长无效：{path.name}，行 {rows + 2}")
            rows += 1
            total_us += dur
            is_comm = _classify(op)[2] == "communication"
            is_hcom = op.startswith("hcom_")
            if is_comm:
                comm_us += dur
                comm_rows += 1
            if is_hcom:
                hcom_us += dur
                hcom_rows += 1
            agg = per_op.setdefault(op, {"count": 0, "total_us": 0.0,
                                         "min_us": None, "max_us": None,
                                         "communication": is_comm})
            agg["count"] += 1
            agg["total_us"] += dur
            agg["min_us"] = dur if agg["min_us"] is None else min(agg["min_us"], dur)
            agg["max_us"] = dur if agg["max_us"] is None else max(agg["max_us"], dur)
            key = _shape_key(row.get("Input Shapes"))
            if key:
                shape_vals.setdefault(op, {}).setdefault(key, []).append(dur)
            dt = _split(row.get("Input Data Types"))
            if dt:
                dtypes.setdefault(op, set()).update(x for x in dt if x)
            if key and op in _GEMM_TYPES:
                gemm_samples.setdefault((op, key, tuple(dt[:2])), []).append(dur)
            start = _num(row.get("Start Time(us)"))
            stream = (row.get("Stream ID") or "").strip()
            if start is not None and stream:
                streams.setdefault(stream, []).append((start, dur))
            bucket = _token_bucket(op, row.get("Input Shapes"))
            if bucket is not None:
                b = buckets.setdefault(bucket, {"kernel_count": 0,
                                                "total_us": 0.0})
                b["kernel_count"] += 1
                b["total_us"] += dur
    return {"present": True, "op_stats": per_op, "shape_stats": shape_vals,
            "dtypes": {k: sorted(v) for k, v in dtypes.items()},
            "streams": streams, "total_us": total_us, "rows": rows,
            "comm_us": comm_us, "hcom_us": hcom_us,
            "comm_rows": comm_rows, "hcom_rows": hcom_rows,
            "compute_us": total_us - comm_us, "gemm_samples": gemm_samples,
            "token_buckets": {k: {"kernel_count": v["kernel_count"],
                                  "total_us": _round(v["total_us"])}
                              for k, v in sorted(buckets.items())}}


def _gap_stats(intervals: List[Tuple[float, float]]) -> dict:
    """设备内核间隙（device-kernel 边界）：同一 stream 内相邻 kernel 的 idle。"""
    ivs = sorted(intervals)
    gaps: List[float] = []
    for (s0, d0), (s1, _d1) in zip(ivs, ivs[1:]):
        g = s1 - (s0 + d0)
        if g > 1e-9:
            gaps.append(g)
    return _stats(gaps)


def _union_gap_stats(streams: Dict[str, List[Tuple[float, float]]]) -> dict:
    """全设备忙区间并集后的空闲间隙（跨 stream 重叠去重；未跨 rank 校时）。"""
    ivs = sorted(iv for vs in streams.values() for iv in vs)
    merged: List[List[float]] = []
    for s, d in ivs:
        e = s + d
        if merged and s <= merged[-1][1] + 1e-9:
            if e > merged[-1][1]:
                merged[-1][1] = e
        else:
            merged.append([s, e])
    gaps = [merged[i + 1][0] - merged[i][1] for i in range(len(merged) - 1)]
    busy = sum(e - s for s, e in merged)
    window = (merged[-1][1] - merged[0][0]) if merged else None
    stats = _stats(g for g in gaps if g > 1e-9)
    stats["busy_union_us"] = _round(busy)
    stats["window_us"] = _round(window)
    stats["busy_fraction_of_window"] = (
        _round(busy / window, 6) if window else None)
    return stats


# ---------------------------------------------------------------------------
# 语义归组 / 差异表
# ---------------------------------------------------------------------------

def _group_census(kernels: dict) -> List[dict]:
    per_op = kernels["op_stats"]
    groups: Dict[str, dict] = {}
    for op, agg in per_op.items():
        gid, rows, cov, note = _classify(op)
        g = groups.setdefault(gid, {
            "group": gid, "coverage": cov, "ledger_rows": list(rows),
            "boundary": GROUP_META.get(gid, {}).get("boundary", "device-kernel"),
            "note": GROUP_META.get(gid, {}).get("note", note),
            "op_types": {}, "kernel_count": 0, "observed_us": 0.0,
        })
        g["op_types"][op] = {"count": agg["count"],
                             "total_us": _round(agg["total_us"])}
        g["kernel_count"] += agg["count"]
        g["observed_us"] += agg["total_us"]
    out = sorted(groups.values(), key=lambda g: (-g["observed_us"], g["group"]))
    for g in out:
        g["observed_us"] = _round(g["observed_us"])
        g["op_types"] = dict(sorted(g["op_types"].items()))
    return out


def _shape_costs(kernels: dict) -> Dict[str, list]:
    out: Dict[str, list] = {}
    for op, table in sorted(kernels["shape_stats"].items()):
        points = []
        for key, vals in table.items():
            st = _stats(vals)
            st["shape"] = key
            points.append(st)
        points.sort(key=lambda p: (-(p["total_us"] or 0), p["shape"]))
        head, tail = points[:MAX_SHAPES_PER_CLASS], points[MAX_SHAPES_PER_CLASS:]
        out[op] = head
        if tail:
            vals = [v for p in tail for v in [p["total_us"]] if v is not None]
            out[op].append({
                "shape": f"<other {len(tail)} shape keys pooled>",
                "count": sum(p["count"] for p in tail),
                "total_us": _round(sum(vals)),
                "min_us": min((p["min_us"] for p in tail
                               if p["min_us"] is not None), default=None),
                "median_us": None, "p90_us": None,
                "max_us": max((p["max_us"] for p in tail
                               if p["max_us"] is not None), default=None),
                "mean_us": None,
            })
    return out


def _derived_rates(kernels: dict) -> dict:
    """只对 2 输入 2D、1 输出 2D 的 MatMul 计算观测算力（设备实测速率，非规格）。"""
    per_op = kernels["op_stats"]
    return {"method": "仅 MatMulV2/MatMulV3 且 shape 可唯一解释为 "
                      "([M,K] @ [N,K] → [M,N]) 时派生；观测实现速率不是设备规格",
            "note": "本审计不把观测速率当作 peak_by_dtype（A-N4 禁止实测峰值填 spec）",
            "classes": {op: {"kernels_with_derivable_shape":
                             per_op[op]["count"],
                             "tflops_observed": None,
                             "reason": "shape 逐核变体多，聚合需 correlation；"
                                       "shape_costs.json 保留逐 shape 点"}
                        for op in ("MatMulV2", "MatMulV3", "GroupedMatmul",
                                   "GroupedMatmulSwigluQuantV2")
                        if op in per_op}}


def _ledger_row_coverage(model_id: Optional[str]) -> dict:
    """账本行静态覆盖：mapped_group / fused_unknown / not_observed / unmapped。"""
    if model_id is None:
        return {"model_id": None, "rows": {}, "coverage": None}
    from simverse.count.work import build_model_work
    from simverse.model_ir import build_catalog
    cat = build_catalog(include_internal=True)
    model = (cat["models"].get(model_id))
    if model is None:
        return {"model_id": model_id, "rows": {}, "coverage": None,
                "reason": "模型不在 catalog（不得用于活跃模型校准）"}
    q = {"dense_w": 8, "expert_w": 8, "act": 8, "kv": 8, "state": 32}
    rows = build_model_work(model, "decode", 1, 1024, 1, q, 1, device_sms=None)
    ops = sorted({r.op for r in rows})
    table = {}
    for op in ops:
        status = ROW_STATUS.get(op, "unmapped")
        table[op] = {"status": status, "reason": ROW_REASON.get(status)}
    mapped = sum(1 for v in table.values() if v["status"] == "mapped_group")
    return {"model_id": model_id, "rows": table,
            "coverage": [mapped, len(ops)],
            "note": "组级归属：一个设备 kernel/融合核可拥有多行；行级 attribution "
                    "需 correlation id（本审计不含）。"}


def _run_diff(run_id: str, kernels: dict, groups: List[dict],
              model_id: Optional[str]) -> dict:
    total = kernels["total_us"]
    compute = kernels["compute_us"]
    comm = kernels["comm_us"]
    ledger_us = sum(g["observed_us"] or 0 for g in groups
                    if g["coverage"] in ("mapped", "fused_group"))
    comm_us = sum(g["observed_us"] or 0 for g in groups
                  if g["coverage"] == "communication")
    unmapped_us = sum(g["observed_us"] or 0 for g in groups
                      if g["coverage"].startswith("unmapped"))
    rows = []
    for g in groups:
        share = (g["observed_us"] / total) if total else None
        rows.append({
            "group": g["group"],
            "kernel_classes": sorted(g["op_types"]),
            "coverage": g["coverage"],
            "owned_ledger_rows": g["ledger_rows"],
            "kernel_count": g["kernel_count"],
            "observed_us": g["observed_us"],
            "time_share": _round(share, 6),
            "time_share_of_compute": (
                _round(g["observed_us"] / compute, 6)
                if compute and g["coverage"] != "communication" else None),
            "analytic_base_ms": None,
            "selected_ms": None,
            "fallback_reason": "无匹配注册设备（profile 设备仅到 Ascend910B 家族；"
                              "SKU/实现/精度域未建立）→ A-N4 前置未满足，不做 T0 计价",
            "boundary": g["boundary"],
            "note": g["note"],
        })
    row_cov = _ledger_row_coverage(model_id)
    return {
        "run_id": run_id,
        "boundary": "device-kernel（profiler task duration；未含 host launch/E2E）",
        "observed_kernel_us": _round(total),
        "compute_kernel_us": _round(compute),
        "communication_kernel_us": _round(comm),
        "hcom_communication_kernel_us": _round(kernels["hcom_us"]),
        "mapped_ledger_time_us": _round(ledger_us),
        "communication_time_us": _round(comm_us),
        "unmapped_time_us": _round(unmapped_us),
        "mapped_ledger_time_share": _round(ledger_us / total, 6) if total else None,
        "communication_time_share": _round(comm_us / total, 6) if total else None,
        "unmapped_time_share": _round(unmapped_us / total, 6) if total else None,
        "mapped_ledger_time_share_of_compute": (
            _round(ledger_us / compute, 6) if compute else None),
        "unmapped_time_share_of_compute": (
            _round(unmapped_us / compute, 6) if compute else None),
        "coverage_note": "mapped_ledger_time 只含设备 kernel 时间；两个分母都披露："
                         "总设备 kernel 时间（含 hcom_* 通信核、多 stream 可能重叠）"
                         "与非通信('compute')时间。通信类单列、不计入账本覆盖；"
                         "未建模类为结构性缺口。",
        "ledger_row_coverage": row_cov,
        "groups": rows,
    }


def _identity_gap_summary(identities: List[dict]) -> List[dict]:
    seen: Dict[str, str] = {}
    for ident in identities:
        for item in ident["absent"]:
            seen.setdefault(item["field"], item["reason"])
    return [{"field": k, "reason": seen[k]} for k in sorted(seen)]


def _qwen36_expected_rows() -> dict:
    """Qwen3.6-35B-A3B 的 GEMM/注意力行签名（来自 pinned config + 本仓账本口径）。

    只用于 A-N5 离线 row-level 匹配演示：签名=(K,N)（同一 K 下 N 决定 FLOPs），
    mult=每 pass 调用数。TP/EP 取观测档（TP1/EP8），GEMM 输出宽按单卡。
    """
    cfg = json.loads((ROOT / "assets" / "config" /
                      "qwen36_35b_a3b_main__config.json").read_text(
                          encoding="utf-8"))["text_config"]
    h = cfg["hidden_size"]
    heads, kv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    hd = cfg["head_dim"]
    q_out = 2 * heads * hd                      # output-gated q_proj
    n_full = sum(1 for t in cfg["layer_types"] if t == "full_attention")
    n_gdn = len(cfg["layer_types"]) - n_full
    n_layers = len(cfg["layer_types"])
    key_dim = cfg["linear_key_head_dim"] * cfg["linear_num_key_heads"]
    value_dim = cfg["linear_value_head_dim"] * cfg["linear_num_value_heads"]
    qkv_out, z_out = 2 * key_dim + value_dim, value_dim
    ba_out = 2 * cfg["linear_num_value_heads"]
    moe_i = cfg["moe_intermediate_size"]
    shared_i = cfg["shared_expert_intermediate_size"]
    experts, vocab = cfg["num_experts"], cfg["vocab_size"]
    rows = []

    def add(op, role, K, N, mult, group, note=""):
        rows.append({"op": op, "role": role, "K": K, "N": N,
                     "mult_per_pass": mult, "kernel_group": group, "note": note})

    add("q_proj", "target", h, q_out, n_full, "attn_qkvg")
    add("k_proj", "target", h, kv * hd, n_full, "attn_qkvg")
    add("v_proj", "target", h, kv * hd, n_full, "attn_qkvg")
    add("o_proj", "target", heads * hd, h, n_full, "o_same")
    add("in_proj_qkv", "target", h, qkv_out, n_gdn, "gdn_qkvz")
    add("in_proj_z", "target", h, z_out, n_gdn, "gdn_qkvz")
    add("in_proj_ba", "target", h, ba_out, n_gdn, "gdn_ba")
    add("out_proj", "target", value_dim, h, n_gdn, "o_same")
    add("moe_router", "target", h, experts, n_layers, "router")
    add("shared_gate", "target", h, 1, n_layers, "shared_gate")
    add("moe_shared", "target", h, 2 * shared_i, n_layers, "shared_gu")
    add("moe_shared", "target", shared_i, h, n_layers, "shared_down")
    add("lm_head", "target", h, vocab, 1, "lm_head")
    # MTP 草稿：每步 1 个 full-attention MoE 层 + fc（K 步/pass，K 由计数反解）
    add("mtp_fc", "draft", 2 * h, h, "<K>", "o_same")
    add("q_proj", "draft", h, q_out, "<K>", "draft_attn_qkvg")
    add("k_proj", "draft", h, kv * hd, "<K>", "draft_attn_qkvg")
    add("v_proj", "draft", h, kv * hd, "<K>", "draft_attn_qkvg")
    add("o_proj", "draft", heads * hd, h, "<K>", "o_same")
    add("moe_router", "draft", h, experts, "<K>", "router")
    add("shared_gate", "draft", h, 1, "<K>", "shared_gate")
    add("moe_shared", "draft", h, 2 * shared_i, "<K>", "shared_gu")
    add("moe_shared", "draft", shared_i, h, "<K>", "shared_down")
    add("draft_lm_head", "draft", h, vocab, "<K>", "lm_head")
    return {"rows": rows, "hidden": h, "q_out": q_out, "kv_dim": kv * hd,
            "qkv_out": qkv_out, "z_out": z_out, "ba_out": ba_out,
            "value_dim": value_dim, "heads_dim": heads * hd,
            "experts": experts,
            "n_full": n_full, "n_gdn": n_gdn, "n_layers": n_layers,
            "vocab": vocab, "moe_i": moe_i, "shared_i": shared_i}


def _gemm_observed(kernels: dict) -> Dict[Tuple[int, int, str], dict]:
    """从 shape 统计提取观测 GEMM 签名 (K,N,dtype)：[T,K] @ [N,K] → [T,N]。"""
    gemm_ops = ("MatMulV2", "MatMulV3", "QuantBatchMatmulV3",
                "WeightQuantBatchMatmulV2", "BatchMatMulV2", "BatchMatMulV3")
    out: Dict[Tuple[int, int, str], dict] = {}
    for op in gemm_ops:
        table = kernels["shape_stats"].get(op)
        if not table:
            continue
        samples = kernels.get("gemm_samples")
        if samples is None:
            items = []  # No joint shape/dtype observations: do not infer from class unions.
        else:
            items = []
            for (sample_op, key, pair), vals in samples.items():
                if sample_op != op:
                    continue
                dtype = ";".join(pair).upper()
                bucket = ("int4" if "INT4" in dtype else "int8" if "INT8" in dtype
                          else "bf16" if len(pair) == 2 and all("BF16" in p.upper() or
                               "BFLOAT16" in p.upper() for p in pair) else "unknown")
                items.append((key, vals, bucket))
        for key, vals, bucket in items:
            parts = key.split(";")
            if len(parts) < 2:
                continue
            try:
                a = tuple(int(x) for x in parts[0].split(","))
                b = tuple(int(x) for x in parts[1].split(","))
            except ValueError:
                continue
            if len(a) != 2 or len(b) != 2 or b[1] != a[1]:
                continue
            k, n = a[1], b[0]
            if k <= 0 or n <= 0 or k > 1_000_000 or n > 1_000_000:
                continue
            st = _stats(vals)
            item = out.setdefault((k, n, bucket), {"count": 0, "total_us": 0.0,
                                                   "durations": []})
            item["count"] += st["count"]
            item["total_us"] += st["total_us"] or 0.0
    return out


_GEMM_TYPES = ("MatMulV2", "MatMulV3", "QuantBatchMatmulV3",
               "WeightQuantBatchMatmulV2", "BatchMatMulV2", "BatchMatMulV3")


def _shape_geometry_candidates(kernels: dict) -> dict:
    """从**具体算子输入/输出 shape** 反推 hidden/heads/vocab/intermediate 等维度。

    每个候选都带证据（哪些 OP Type、出现次数、总时长），不靠名字猜。
    """
    shapes = kernels["shape_stats"]
    out = {"hidden_candidates": {}, "gemm_k_candidates": {},
           "gemm_n_candidates": {}, "vocab_candidates": {},
           "intermediate_candidates": {}, "local_experts_observed": []}

    def bump(bucket, val, op, n):
        item = out[bucket].setdefault(str(val), {"count": 0, "op_types": {}})
        item["count"] += n
        item["op_types"][op] = item["op_types"].get(op, 0) + n

    for op in ("AddRmsNormBias", "RmsNorm", "GemmaRmsNorm",
               "_layer_norm_fwd_1pass_kernel_npu", "AddRmsNormDynamicQuant"):
        for key, vals in (shapes.get(op) or {}).items():
            parts = key.split(";")
            try:
                first = tuple(int(x) for x in parts[0].split(","))
            except (ValueError, IndexError):
                continue
            if len(first) == 2 and first[1] > 0:
                bump("hidden_candidates", first[1], op, len(vals))
            elif first:
                bump("hidden_candidates", first[-1], op, len(vals))
    hidden_mode = None
    if out["hidden_candidates"]:
        hidden_mode = int(max(out["hidden_candidates"].items(),
                              key=lambda kv: kv[1]["count"])[0])
    for op in _GEMM_TYPES:
        for key, vals in (shapes.get(op) or {}).items():
            parts = key.split(";")
            if len(parts) < 2:
                continue
            try:
                a = tuple(int(x) for x in parts[0].split(","))
                b = tuple(int(x) for x in parts[1].split(","))
            except ValueError:
                continue
            if len(a) != 2 or len(b) != 2 or b[1] != a[1]:
                continue
            K, N = a[1], b[0]
            bump("gemm_k_candidates", K, op, len(vals))
            bump("gemm_n_candidates", N, op, len(vals))
            if N >= 50_000:
                bump("vocab_candidates", N, op, len(vals))
            if hidden_mode and K == hidden_mode and 64 <= N <= 65_536:
                bump("intermediate_candidates", N, op, len(vals))
    for key, vals in (shapes.get("GroupedMatmul") or {}).items():
        parts = key.split(";")
        if len(parts) < 2:
            continue
        try:
            w = tuple(int(x) for x in parts[1].split(","))
        except ValueError:
            continue
        if len(w) == 3 and w[0] > 0:
            out["local_experts_observed"].append(
                {"grouped_weight_shape": list(w), "calls": len(vals)})
    return {"hidden_dim_mode": hidden_mode,
            "hidden_candidates": dict(sorted(out["hidden_candidates"].items())),
            "gemm_k_candidates": dict(sorted(out["gemm_k_candidates"].items())),
            "gemm_n_candidates": dict(sorted(
                out["gemm_n_candidates"].items(),
                key=lambda kv: -kv[1]["count"])[:12]),
            "vocab_candidates": dict(sorted(out["vocab_candidates"].items())),
            "intermediate_candidates": dict(sorted(
                out["intermediate_candidates"].items(),
                key=lambda kv: -kv[1]["count"])[:8]),
            "local_experts_observed": out["local_experts_observed"][:4],
            "basis": "由具体算子输入/输出 shape 反推（norm 输入末维、GEMM K/N、"
                     "lm_head 输出宽、grouped 专家权重的 E 维）；带调用计数证据"}


def _geometry_verify(geom: dict, model_id: Optional[str],
                     tp_hint: Optional[int]) -> dict:
    """用 shape 证据核对 hidden/heads/vocab/intermediate 是否吻合声明模型。"""
    expected = EXPECTED_GEOMETRY.get(model_id or "")
    cands = geom.get("shape_candidates") or {}
    out = {"model_id": model_id, "checked": bool(expected),
           "tp_hint_from_groups": tp_hint, "fields": {}, "failures": [],
           "basis": "shape 维度核对（hidden/heads/kv/head_dim/experts/vocab/"
                    "intermediate）；仍是形状一致，不等于权重身份"}
    if not expected:
        out["reason"] = "无期望几何参考"
        return out
    hidden_c = {int(k) for k in cands.get("hidden_candidates", {})}
    gemm_k = {int(k) for k in cands.get("gemm_k_candidates", {})}
    exp_hidden = expected.get("hidden_size")
    hidden_mode = cands.get("hidden_dim_mode")
    ok_hidden = (exp_hidden == hidden_mode and (exp_hidden in gemm_k))
    out["fields"]["hidden_size"] = {
        "expected": exp_hidden, "observed_norm_candidates": sorted(hidden_c),
        "observed_gemm_k_candidates": sorted(gemm_k),
        "hidden_dim_mode": hidden_mode, "match": ok_hidden,
        "evidence": {k: cands["hidden_candidates"][k]
                     for k in cands.get("hidden_candidates", {})
                     if int(k) == exp_hidden}}
    if not ok_hidden:
        out["failures"].append(
            f"hidden {exp_hidden} 未同时命中 norm 输入末维众数（{hidden_mode}）"
            f"与 GEMM K 候选 {sorted(gemm_k)}")
    vocab_c = {int(k) for k in (cands.get("vocab_candidates") or {})}
    exp_vocab = expected.get("vocab_size")
    if exp_vocab and vocab_c:
        tp = tp_hint or 1
        ok_vocab = exp_vocab in vocab_c or (exp_vocab // tp) in vocab_c
        out["fields"]["vocab_size"] = {
            "expected": exp_vocab, "observed_lm_head_N": sorted(vocab_c),
            "tp_hint": tp, "match": ok_vocab}
        if not ok_vocab:
            out["failures"].append(f"vocab {exp_vocab} 未被 lm_head 输出宽命中")
    elif exp_vocab:
        out["fields"]["vocab_size"] = {"expected": exp_vocab, "observed": None,
                                       "match": None,
                                       "reason": "未观测到 >=100000 的 GEMM 输出宽"}
    exp_inter = expected.get("intermediate_size")
    if exp_inter:
        inter_c = {int(k) for k in (cands.get("intermediate_candidates") or {})}
        tp = tp_hint or 1
        ok_inter = any(exp_inter in (n, n * tp, 2 * n)
                       for n in inter_c) if inter_c else None
        out["fields"]["intermediate_size"] = {
            "expected": exp_inter, "observed_gemm_N": sorted(inter_c),
            "match": ok_inter}
        if ok_inter is False:
            out["failures"].append(
                f"intermediate {exp_inter} 未被 hidden-K GEMM 的 N 命中")
    if expected.get("num_experts") is not None:
        exps = geom.get("moe_experts_observed") or []
        out["fields"]["num_experts"] = {
            "expected": expected.get("num_experts"), "observed": exps,
            "match": expected["num_experts"] in exps if exps else None}
        if exps and expected["num_experts"] not in exps:
            out["failures"].append(f"num_experts {exps} 不含 "
                                   f"{expected['num_experts']}")
    core = _geometry_check(geom, model_id, tp_hint)
    out["failures"].extend(core.get("failures", []))
    out["unknown"] = list(core.get("unknown", []))
    if tp_hint is None:
        out["unknown"].append("tp_for_global_heads/vocab")
    for name, field in out["fields"].items():
        if field.get("match") is None:
            out["unknown"].append(name)
    out["observed_compatible"] = not out["failures"]
    out["verification_passed"] = out["checked"] and not out["failures"] and not out["unknown"]
    return out


def _unallocated_groups(groups: List[dict], rows: List[dict]) -> List[dict]:
    allocations = {}
    for row in rows:
        group = {"MatMul/QuantBatchMatmul": "gemm_dense_quant",
                 "FusedInferAttentionScore": "attention_core"}.get(row["kernel_class"])
        if group:
            allocations[group] = allocations.get(group, 0.0) + sum(
                x["allocated_us"] for x in row.get("allocated_to", []))
    result = []
    for group in groups:
        if group["coverage"] not in ("mapped", "fused_group"):
            continue
        remaining = _round(max(0.0, group["observed_us"] - allocations.get(group["group"], 0.0)))
        if remaining and remaining > .001:
            result.append({"group": group["group"], "observed_us": group["observed_us"],
                           "unallocated_us": remaining, "coverage": group["coverage"],
                           "reason": "无逐调用 ownership，或候选签名/计数未匹配；保留全部未分摊时间"})
    return result


def _ledger_row_match(run_id: str, run_label: str, owner: dict,
                      kernels: dict, groups: List[dict]) -> dict:
    """A-N5 离线 row-level 提取：shape 签名 + 计数方程 → 假设分摊候选表。

    只对活跃模型 Qwen3.6-35B-A3B 的目标 run 做；minidraft 按 owner 声明排除。
    输出仍是 audit-only：不产生 selected_ms，不改变任何账本；融合/同形核的
    拆分依据显式登记（宽度份额或计数方程），不是源码 ownership pin。
    """
    target = (owner or {}).get("target_model_id")
    result = {
        "schema_version": SCHEMA, "run_id": run_id, "run_label": run_label,
        "target_model": {
            "id": target,
            "declared": (owner or {}).get("target_model_declared"),
            "identity_status": "owner-declared 模型名 + 形状一致；"
                               "repo/revision/checkpoint 与引擎 commit 未验证",
        },
        "draft_method": (owner or {}).get("draft_method"),
        "excluded_by_owner": bool((owner or {}).get("excluded")),
        "calibration_admission": "not-admitted",
        "rows": [], "unallocated_groups": [], "coverage": None,
        "pass_structure": None, "limits": [],
    }
    if result["excluded_by_owner"]:
        result["reason"] = f"owner 声明排除：{owner.get('exclusion_reason')}"
        return result
    if target != "qwen36_35b_a3b":
        result["reason"] = ("目标模型不在活跃 catalog（Qwen3-8B 为历史/离线模型）："
                            "按跨模型规则不做 row-level 匹配，避免借用为主模型校准")
        return result

    exp = _qwen36_expected_rows()
    obs = _gemm_observed(kernels)
    n_full, n_gdn, n_layers = exp["n_full"], exp["n_gdn"], exp["n_layers"]
    # ---- 计数反解 pass 数与 MTP 步数（不依赖 step 边界；因此是 derived） ----
    passes = mtp_k = None
    pass_basis = []
    ba = obs.get((exp["hidden"], exp["ba_out"], "bf16"))
    if ba and ba["count"] % n_gdn == 0:
        passes = ba["count"] // n_gdn
        pass_basis.append(f"in_proj_ba 调用 {ba['count']} ÷ GDN 层 {n_gdn} = {passes}")
    attn_type = kernels["op_stats"].get("FusedInferAttentionScore")
    attn_count = attn_type["count"] if attn_type else None
    if passes and attn_count and attn_count % passes == 0:
        mtp_k = attn_count // passes - n_full
        pass_basis.append(f"attention 调用 {attn_count} ÷ passes {passes} − 全注意力层 "
                          f"{n_full} = MTP K={mtp_k}")
    result["pass_structure"] = {
        "passes": passes, "mtp_k": mtp_k,
        "basis": pass_basis,
        "status": ("derived-from-counts（非 step 边界证明；chunked-prefill/decode "
                   "混合时 pass 语义是近似）" if passes and mtp_k is not None
                   else "unknown（计数不可整除，不做行级分摊）"),
    }
    if not passes or mtp_k is None or mtp_k < 0 or (result["draft_method"] == "mtp" and mtp_k == 0):
        result["pass_structure"]["status"] = "unknown（不能证明正整数 passes/MTP K；不做行级分摊）"
        result["reason"] = "调用数无法确定合法 pass 结构；保留全部未分摊组"
        result["unallocated_groups"] = _unallocated_groups(groups, [])
        return result
    K = mtp_k
    rows = [dict(r) for r in exp["rows"]]
    for r in rows:
        if r["mult_per_pass"] == "<K>":
            r["mult_per_pass"] = K

    # 规则：(签名, [(op, role, 每pass次数, 宽度或None)], 每pass核次数,
    #        依据, 分摊方式 width|count)
    def _w(v):
        return v if v else 0

    qkvg_w = [exp["q_out"], exp["kv_dim"], exp["kv_dim"]]
    rules = [
        ((exp["hidden"], exp["q_out"] + 2 * exp["kv_dim"], "int8"),
         [("q_proj", "target", n_full, qkvg_w[0]),
          ("k_proj", "target", n_full, qkvg_w[1]),
          ("v_proj", "target", n_full, qkvg_w[2])],
         n_full, "fused-width: qkv+output-gate（target W8A8）", "width"),
        ((exp["hidden"], exp["q_out"] + 2 * exp["kv_dim"], "bf16"),
         [("q_proj", "draft", K, qkvg_w[0]),
          ("k_proj", "draft", K, qkvg_w[1]),
          ("v_proj", "draft", K, qkvg_w[2])],
         K, "fused-width: draft qkv+gate（MTP 层）", "width"),
        ((exp["hidden"], exp["qkv_out"] + exp["z_out"], "bf16"),
         [("in_proj_qkv", "target", n_gdn, exp["qkv_out"]),
          ("in_proj_z", "target", n_gdn, exp["z_out"])],
         n_gdn, "fused-width: GDN in_proj_qkvz(8192+4096)", "width"),
        ((exp["hidden"], exp["ba_out"], "bf16"),
         [("in_proj_ba", "target", n_gdn, None)],
         n_gdn, "unique: GDN gating ba", "count"),
        ((exp["heads_dim"], exp["hidden"], "bf16"),
         [("o_proj", "target", n_full, None), ("out_proj", "target", n_gdn, None),
          ("o_proj", "draft", K, None), ("mtp_fc", "draft", K, None)],
         n_full + n_gdn + 2 * K,
         "count-equation: 同形 o_proj/out_proj/mtp_fc", "count"),
        ((exp["hidden"], exp["experts"], "bf16"),
         [("moe_router", "target", n_layers, None),
          ("moe_router", "draft", K, None)],
         n_layers + K, "count-equation: router（target 全部层 + MTP 步）", "count"),
        ((exp["hidden"], 1, "bf16"),
         [("shared_gate", "target", n_layers, None),
          ("shared_gate", "draft", K, None)],
         n_layers + K, "count-equation: shared expert gate", "count"),
        ((exp["hidden"], 2 * exp["shared_i"], "bf16"),
         [("moe_shared", "target", n_layers, None),
          ("moe_shared", "draft", K, None)],
         n_layers + K, "count-equation: shared gate+up", "count"),
        ((exp["shared_i"], exp["hidden"], "bf16"),
         [("moe_shared", "target", n_layers, None),
          ("moe_shared", "draft", K, None)],
         n_layers + K, "count-equation: shared down", "count"),
        ((exp["hidden"], exp["kv_dim"], "bf16"),
         [("k_proj", "target", n_full, None), ("v_proj", "target", n_full, None)],
         2 * n_full, "ambiguous-equal: k/v 同形同计数", "count"),
        ((exp["hidden"], exp["vocab"], "bf16"),
         [("lm_head", "target", 1, None), ("draft_lm_head", "draft", K, None)],
         1 + K, "count-equation: target/draft lm_head", "count"),
    ]
    for (k, n, dt), cand, count_factor, basis, alloc_mode in rules:
        if count_factor <= 0:
            continue
        item = obs.get((k, n, dt))
        if not item:
            continue
        predicted = None if passes is None else passes * count_factor
        alloc = []
        residual = None
        if predicted is not None:
            residual = item["count"] - predicted
            if residual == 0:
                total_w = sum(_w(c[3]) if alloc_mode == "width" else c[2]
                              for c in cand)
                for op, role, mult, width in cand:
                    share = ((width or 0) if alloc_mode == "width" else mult) / total_w
                    alloc.append({
                        "op": op, "role": role,
                        "predicted_count": passes * mult,
                        "allocated_us": _round(item["total_us"] * share),
                        "split_basis": ("width" if alloc_mode == "width" else
                                        "count")})
                status = "matched"
            else:
                status = "ambiguous_count_residual"
        else:
            status = "unknown_pass_structure"
        result["rows"].append({
            "signature": {"K": k, "N": n, "dtype": dt},
            "kernel_class": "MatMul/QuantBatchMatmul",
            "observed_count": item["count"],
            "observed_us": _round(item["total_us"]),
            "per_call_us": _round(item["total_us"] / item["count"]
                                  if item["count"] else None),
            "expected_count": predicted,
            "count_residual": residual,
            "status": status,
            "allocation_basis": basis,
            "allocation_evidence": "width/count-derived-assumption; not source-pinned",
            "signature_scope": "K/N/dtype 聚合；M/阶段/角色关联仍未知，非逐调用实测归属",
            "allocated_to": alloc,
        })
    # ---- 注意力核：同一 FusedInferAttentionScore 内按计数分 target/draft ----
    if attn_type and passes and K is not None:
        predicted = passes * (n_full + K)
        residual = attn_count - predicted
        alloc = []
        if abs(residual) <= max(1, round(0.02 * attn_count)):
            alloc = [
                {"op": "attention_core", "role": "target",
                 "predicted_count": passes * n_full,
                 "allocated_us": _round(attn_type["total_us"] * n_full /
                                        (n_full + K))},
                {"op": "attention_core", "role": "draft",
                 "predicted_count": passes * K,
                 "allocated_us": _round(attn_type["total_us"] * K / (n_full + K))},
            ]
            status = "matched"
        else:
            status = "ambiguous_count_residual"
        result["rows"].append({
            "signature": None, "kernel_class": "FusedInferAttentionScore",
            "observed_count": attn_count, "observed_us": _round(attn_type["total_us"]),
            "per_call_us": _round(attn_type["total_us"] / attn_count),
            "expected_count": predicted, "count_residual": residual,
            "status": status, "allocation_basis": "count-equation: 全注意力层+MTP 步",
            "allocated_to": alloc,
        })
    # ---- 未做行级分摊的组（融合/实现分解，显式列出） ----
    result["unallocated_groups"] = _unallocated_groups(groups, result["rows"])

    target_rows = sorted({r["op"] for r in rows if r["role"] == "target"})
    measured_ops = sorted({a["op"] for r in result["rows"] for a in r["allocated_to"]
                           if a["role"] == "target"})
    total_us = sum(item["total_us"] for item in obs.values()) + (
        attn_type["total_us"] if attn_type else 0.0)
    allocated_us = sum(a["allocated_us"] for r in result["rows"]
                       for a in r["allocated_to"] if a["role"] == "target")
    result["coverage"] = {
        "rows_total": len(target_rows),
        "rows_scope": "参与 2D GEMM/注意力签名的目标行（非模型全部 27 行；"
                      "其余非 GEMM 行见 unallocated_groups）",
        "rows_with_measured": len(measured_ops),
        "rows_with_measured_ids": measured_ops,
        "rows_without_measured": sorted(set(target_rows) - set(measured_ops)),
        "matched_time_us": _round(allocated_us),
        "candidate_time_us": _round(total_us),
        "note": "分母只含参与签名的 2D GEMM + 注意力核时间；行级 measured 是"
                "按计数方程/融合宽度分摊的候选值，不是逐调用实测归属",
    }
    result["limits"] = [
        "owner 已声明模型名（Qwen3.6-35B-A3B + MTP），但权重 revision/checkpoint、"
        "vLLM/vllm-ascend commit、设备 SKU 未验证 → not-admitted",
        "passes/MTP K 由调用计数反解，不是 step 边界证明；chunked-prefill 与 decode "
        "混合时 pass 语义近似",
        "融合/同形核按宽度或计数方程分摊，未做源码 ownership pin；行级 measured "
        "只能作为候选校准表，不得直接进入 CalibratedPricer",
        "target q/k/v 投影在融合 split_qkv_rmsnorm_mrope 核内、lm_head 调用计数有"
        "残差 → 未分摊（显式列出，不填 0）",
        "minidraft（自研草稿）按 owner 声明排除，不参与本匹配",
    ]
    return result


def _protocol(candidate_models, run_count: int) -> dict:
    return {
        "schema_version": SCHEMA,
        "created_at": CREATED_AT,
        "extractor": "tools/import_ascend_profiles.py",
        "calibration_admission": "not-admitted",
        "admitted": False,
        "admission_reason": "A-N5 离线审计：身份不完整（设备 SKU/board/引擎版本/"
                            "权重/warmup/step 边界缺失）+ 本仓 ascend_910b3 档为 "
                            "assumed 旧仓映射、不能证明同 SKU/驱动 → 不进入"
                            "CalibratedPricer；生产准入依赖 A-N4 官方规格与匹配采数",
        "result_kind": "audit-only（无计价；selected_ms 恒为 null）",
        "device_family": "Ascend910B（NPU_INFO.name；SKU 未建立）",
        "run_count": run_count,
        "candidate_model_ids": list(candidate_models),
        "boundary": "device-kernel",
        "limits": [
            "本包不是 measured 校准集：无 selected_ms、无成本结论、不改变任何账本",
            "未在本仓执行 NPU 采集；只重放外部已归档 profile 产物",
            "设备 SKU/驱动/引擎 commit/权重身份/量化声明缺失 → 不可跨 SKU/跨模型/跨栈复用",
            "多个 run 身份不同，禁止相减或拼接",
            "逐 shape 点为设备内核采样统计；有重复观测但无独立重复 run，"
            "不宣称统计稳定或留出误差",
        ],
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

README = """# Ascend profiling 审计包（只读 · 未准入）

本目录由 `tools/import_ascend_profiles.py` 从外部 Ascend NPU profiler 产物蒸馏生成，
**不是实测校准集、不参与计价**。用途是 A-N5「profiling 校对」第一阶段的
原始 trace 审核、映射与差异表（docs/ascend-validation-plan.md §1/§3）。

- `manifest.json`：本包身份、来源 run 与逐文件 SHA256、准入裁决（恒 `not-admitted`）。
- `identity.json`：每 run 记录到的栈身份/并行组/会话窗口，以及**缺失**身份字段与原因；
  另含 `shape_candidates`（由具体算子输入/输出 shape 反推的 hidden/GEMM K/N/lm_head
  输出宽/专家维度，带调用计数）与 `geometry_verification`（与 owner 声明模型的
  维度核对结果）。
- `census.json`：逐 OP Type 的设备 kernel 计数与耗时统计（min/median/p90/max）。
- `shape_costs.json`：关键算子类按输入 shape 的采样统计（未来局部校准候选键）。
- `mapping.json`：采集侧 kernel/OP Type → 本仓账本行组 ownership 与覆盖状态。
- `diff.json`：结构差异表（账本覆盖时间占比、通信类、未建模类、设备内核间隙）。
- `ledger_rows.json`：A-N5 离线 **row-level 候选提取**——GEMM shape 签名 + 调用计数
  方程 → 账本行 measured 候选表（目标模型 Qwen3.6-35B-A3B 的 MTP run；minidraft
  按 owner 声明排除；Qwen3-8B 为历史模型不做跨模型匹配）。融合/同形核按宽度或
  计数分摊，**不是源码 ownership pin**。

## 身份声明

owner 已声明两个根目录的目标模型：`qwen3_6_mtp_domino` → Qwen3.6-35B-A3B，
`qwen3-8b-0811` → Qwen3-8B；mtp/dflash/dspark 为三种现有草稿模型，
minidraft 为自研草稿、本轮不考虑（对应 run 标记 `excluded_by_owner`）。
该声明只覆盖名称：权重 revision/config/checkpoint、推理引擎 commit、量化配方
仍未验证，故不影响下方准入结论。

## 为什么不能直接用于校准

1. 设备身份只到 `Ascend910B` 家族，无 `npu-smi` board/SKU/驱动；本仓已有
   `ascend_910b3` 解析档但为 **assumed 旧仓映射**（spec 未官方核验），不能证明
   同 SKU/驱动；910B3/910C/A2/A3 能力不可互相继承（A-N4 仍需官方规格核验）。
2. 推理引擎版本（vLLM/vllm-ascend）与权重 repo/revision/checkpoint 哈希未随
   profile 提供 → 算子 ownership 不能按源码 pin，只能按 kernel 语义归组。
3. 无 warmup/active 区间、无重复 run、step 节点为空 → 不能宣称统计稳定或切分
   prefill/decode 窗口。
4. `CalibratedPricer` 的设备/精度/TP/形状域门禁无法匹配（assumed 档不等于
   已证明同域）→ 校准一律解析回退。

## 复现

    python tools/import_ascend_profiles.py --check --out <本目录>

`--check` 只校验本轮产物哈希与 manifest 一致：不重写产物，也不读取原始采集目录。
"""


def build_pack(src_roots: List[Path]) -> Tuple[Dict[str, bytes], dict]:
    runs = discover_runs(src_roots)
    identities, census_runs, shape_runs, diff_runs = [], [], [], []
    ledger_runs: List[dict] = []
    observed_ops: Dict[str, set] = {}

    for run in runs:
        out = run["out_dir"]
        ident = _parse_identity(run)
        kernels = _parse_kernels(out)
        step_trace = _parse_step_trace(out)
        analysis_db = _parse_analysis_db(out)
        op_stat = _parse_op_statistic(out)
        for name in ("kernel_details.csv", "op_statistic.csv", "step_trace_time.csv", "analysis.db"):
            source_file = out / name
            if source_file.is_file():
                ident["source_files_sha256"]["ASCEND_PROFILER_OUTPUT/" + name] = _sha256(source_file)

        # op_statistic（msprof 聚合，**不含 hcom_* 通信核**）与 kernel_details
        # 非通信行逐行求和的一致性核对
        stat_total = sum(r["total_us"] for r in op_stat if r["total_us"] is not None)
        stat_count = sum(r["count"] for r in op_stat if r["count"] is not None)
        consistency = {
            "kernel_detail_rows": kernels["rows"],
            "communication_kernel_rows_all": kernels["comm_rows"],
            "hcom_kernel_rows": kernels["hcom_rows"],
            "op_statistic_count_sum": stat_count or None,
            "kernel_total_us": _round(kernels["total_us"]),
            "kernel_hcom_communication_us": _round(kernels["hcom_us"]),
            "kernel_communication_us_all": _round(kernels["comm_us"]),
            "kernel_non_hcom_us": _round(kernels["total_us"] - kernels["hcom_us"]),
            "op_statistic_total_us": _round(stat_total) if op_stat else None,
            "count_match_non_hcom": (
                stat_count == kernels["rows"] - kernels["hcom_rows"])
            if op_stat else None,
            "total_us_rel_delta_non_hcom": (
                _round(abs(stat_total -
                           (kernels["total_us"] - kernels["hcom_us"])) /
                       (kernels["total_us"] - kernels["hcom_us"]), 9)
                if op_stat and kernels["total_us"] > kernels["hcom_us"] else None),
            "note": "op_statistic.csv 不含 hcom_* 通信核，因此只与 kernel_details "
                    "的非 hcom 行对账；MoeDistribute*/allreduce 等设备侧通信核在 "
                    "op_statistic 中，已在 communication 类中单列。",
        }
        if op_stat:
            for r in op_stat:
                observed_ops.setdefault(r["op_type"], set()).add(run["run_id"])

        groups = _group_census(kernels)
        run_ops = set(kernels["op_stats"]) | {r["op_type"] for r in op_stat}
        for op in run_ops:
            observed_ops.setdefault(op, set()).add(run["run_id"])
        model_guess = ident["owner_declared"].get("target_model_id") or \
            _candidate_model_for_run(run_ops)
        # 由本 run 的 attention/GDN/MoE 形状再核对一次（几何证据见 identity.json）
        shapes = kernels["shape_stats"]
        geom = {}
        if "AddRmsNormBias" in shapes:
            dims = []
            for key in shapes["AddRmsNormBias"]:
                parts = key.split(";")
                if parts and "," in parts[0]:
                    try:
                        dims.append(tuple(int(x) for x in parts[0].split(",")))
                    except ValueError:
                        pass
            if dims:
                geom["hidden_size_observed"] = sorted({d[-1] for d in dims})
        if "FusedInferAttentionScore" in shapes:
            # 同一 OP Type 可能混有目标层与非 catalog 草稿头（如 minidraft），
            # 因此按 (heads, head_dim) 分组，取耗时最大组为"主注意力几何"，
            # 其余显式登记为污染组（不静默丢弃、也不让它们使目标几何"失败"）。
            attn_groups: Dict[Tuple[int, int], dict] = {}
            kv_by_hd: Dict[int, set] = {}
            for key, durs in shapes["FusedInferAttentionScore"].items():
                parts = key.split(";")
                try:
                    q = tuple(int(x) for x in parts[0].split(","))
                except (ValueError, IndexError):
                    continue
                if len(q) != 3:
                    continue
                g = (q[1], q[2])
                item = attn_groups.setdefault(g, {"count": 0, "total_us": 0.0})
                item["count"] += len(durs)
                item["total_us"] += sum(durs)
                for part in parts[1:]:
                    try:
                        kv = tuple(int(x) for x in part.split(","))
                    except ValueError:
                        continue
                    if len(kv) == 3 and kv[2] == q[2] and 0 < kv[1] <= 128:
                        kv_by_hd.setdefault(q[2], set()).add(kv[1])
            if attn_groups:
                prim = max(attn_groups.items(),
                           key=lambda kv: (kv[1]["total_us"], kv[1]["count"]))
                geom["attention_primary"] = {
                    "heads": prim[0][0], "head_dim": prim[0][1],
                    "shape_keys": prim[1]["count"],
                    "total_us": _round(prim[1]["total_us"])}
                geom["attention_geometry_groups"] = [
                    {"heads": k[0], "head_dim": k[1],
                     "shape_keys": v["count"], "total_us": _round(v["total_us"])}
                    for k, v in sorted(attn_groups.items(),
                                       key=lambda kv: -kv[1]["total_us"])]
                geom["attention_calls_observed"] = sum(
                    v["count"] for v in attn_groups.values())
                geom["attention_kv_heads_by_head_dim"] = {
                    str(hd): sorted(v) for hd, v in sorted(kv_by_hd.items())}
        if "MoeGatingTopK" in shapes:
            experts = set()
            for key in shapes["MoeGatingTopK"]:
                parts = key.split(";")
                if parts and "," in parts[0]:
                    try:
                        experts.add(tuple(int(x) for x in
                                          parts[0].split(","))[-1])
                    except ValueError:
                        pass
            if experts:
                geom["moe_experts_observed"] = sorted(experts)
        ident["observed_geometry"] = geom
        geom["shape_candidates"] = _shape_geometry_candidates(kernels)
        ident["candidate_model_id"] = model_guess
        tp_hint = _tp_hint(ident["recorded"].get("parallel_groups") or {})
        ident["tp_hint_from_groups"] = tp_hint
        ident["geometry_check"] = _geometry_check(geom, model_guess, tp_hint)
        ident["geometry_verification"] = _geometry_verify(
            geom, model_guess, tp_hint)
        identities.append(ident)

        census_runs.append({
            "run_id": run["run_id"],
            "run_label": run["run_label"],
            "source_root_label": run["src_root_label"],
            "profiler_level": ident["recorded"].get("profiler_level"),
            "kernel_rows": kernels["rows"],
            "kernel_total_us": _round(kernels["total_us"]),
            "compute_kernel_us": _round(kernels["compute_us"]),
            "communication_kernel_us": _round(kernels["comm_us"]),
            "hcom_communication_kernel_us": _round(kernels["hcom_us"]),
            "communication_kernel_rows": kernels["comm_rows"],
            "session_ms": ident["recorded"].get("session_ms"),
            "kernel_total_over_session": (
                _round(kernels["total_us"] / 1000 / ident["recorded"]["session_ms"], 6)
                if ident["recorded"].get("session_ms") else None),
            "step_trace": step_trace,
            "analysis_db_step_trace": analysis_db,
            "step_trace_consistency": _step_trace_consistency(
                step_trace.get("rows") or [], analysis_db.get("step_trace_time") or []),
            "token_bucket_split_heuristic": kernels["token_buckets"],
            "token_bucket_note": "按每个 kernel 首维 token 数分桶（decode<=64 / "
                                 "draft/verify 65..1024 / prefill>1024）；这是逐调用"
                                 "启发式，不是连续阶段窗口（step 节点为空）。",
            "op_statistic": sorted(op_stat, key=lambda r: (-(r["total_us"] or 0),
                                                           r["op_type"])),
            "op_statistic_consistency": consistency,
            "groups": groups,
            "stream_gap_stats": {sid: _gap_stats(ivs)
                                 for sid, ivs in sorted(kernels["streams"].items())},
            "union_gap_stats": _union_gap_stats(kernels["streams"]),
            "observed_dtypes": {op: dts for op, dts in sorted(kernels["dtypes"].items())},
            "derived_rates": _derived_rates(kernels),
        })
        shape_runs.append({"run_id": run["run_id"],
                           "classes": _shape_costs(kernels)})
        diff_runs.append(_run_diff(run["run_id"], kernels, groups, model_guess))
        ledger_runs.append(_ledger_row_match(
            run["run_id"], run["run_label"], ident["owner_declared"],
            kernels, groups))

    # 跨 run：不允许拼接；只登记互不相同的身份（栈 = 版本/家族，不含 run 时间）
    stacks = sorted({(ident["recorded"].get("torch_npu_version"),
                      ident["recorded"].get("cann_version"),
                      ident["recorded"].get("npu_name"))
                     for ident in identities})
    candidate_models = sorted({ident["candidate_model_id"] for ident in identities
                               if ident["candidate_model_id"]})
    mapping = {
        "schema_version": SCHEMA,
        "ownership_basis": "OP Type → 语义组 → 账本行组的显式表（按 kernel 语义与"
                           "shape 位置）；采集侧实现源码（vLLM/vllm-ascend commit）"
                           "未随 profile 提供 → 不宣称 source-pinned ownership",
        "boundary_rule": "一个设备 kernel 可以拥有多行（融合）或多核对应一行；"
                         "组内合并只计一次设备时间",
        "rows": [
            {"group": gid, "coverage": meta["coverage"],
             "ledger_rows": meta["ledger_rows"], "boundary": meta["boundary"],
             "note": meta["note"],
             "op_types": sorted(op for op in observed_ops
                                if _classify(op)[0] == gid)}
            for gid, meta in sorted(GROUP_META.items())
        ],
        "observed_op_types": {
            op: {"runs": sorted(runs_), "group": _classify(op)[0],
                 "coverage": _classify(op)[2],
                 "ledger_rows": list(_classify(op)[1])}
            for op, runs_ in sorted(observed_ops.items())},
        "ledger_row_status": ROW_STATUS,
        "ledger_row_status_reason": ROW_REASON,
    }

    files: Dict[str, bytes] = {}
    files["identity.json"] = (_canon({
        "schema_version": SCHEMA,
        "identity_gap_summary": _identity_gap_summary(identities),
        "runs": sorted(identities, key=lambda r: r["run_id"]),
        "note": "profile 只到 Ascend910B 家族级身份；缺失字段如实登记，不补造。",
    }) + "\n").encode()
    files["census.json"] = (_canon({
        "schema_version": SCHEMA,
        "runs": sorted(census_runs, key=lambda r: r["run_id"]),
        "note": "统计口径=profiler task duration（device-kernel 边界）；"
                "多 stream 可重叠，未做关键路径去重。",
    }) + "\n").encode()
    files["shape_costs.json"] = (_canon({
        "schema_version": SCHEMA,
        "statistic": "min/median/p90/max/total（µs）；按 shape 键分组",
        "runs": sorted(shape_runs, key=lambda r: r["run_id"]),
    }) + "\n").encode()
    files["mapping.json"] = (_canon(mapping) + "\n").encode()
    files["diff.json"] = (_canon({
        "schema_version": SCHEMA,
        "calibration_admission": "not-admitted",
        "runs": sorted(diff_runs, key=lambda r: r["run_id"]),
        "note": "analytic_base_ms 恒为 null：无匹配注册设备（A-N4 前置），"
                "不做 T0 计价；不产生 selected_ms。",
    }) + "\n").encode()
    files["ledger_rows.json"] = (_canon({
        "schema_version": SCHEMA,
        "calibration_admission": "not-admitted",
        "note": "A-N5 离线 row-level 候选提取：GEMM shape 签名 + 调用计数方程 → "
                "账本行 measured 候选表；融合/同形核按宽度或计数分摊，未做源码 "
                "ownership pin。audit-only，无 selected_ms。",
        "runs": sorted(ledger_runs, key=lambda r: r["run_id"]),
    }) + "\n").encode()

    manifest = _protocol(candidate_models, len(runs))
    manifest.update({
        "profile_set_id": SET_ID,
        "kind": "ascend-profile-audit",
        "source": {
            "kind": "external Ascend profiler output（只读导入）",
            "roots": sorted({r["src_root_label"] for r in runs}),
            "runs": [{"run_id": r["run_id"],
                      "run_label": r["run_label"],
                      "source_root_label": r["src_root_label"]}
                     for r in sorted(runs, key=lambda r: r["run_id"])],
            "identity_stacks": [list(s) for s in stacks],
            "files_sha256": {r["run_id"]: ident["source_files_sha256"]
                             for r, ident in sorted(zip(runs, identities),
                                                    key=lambda x: x[0]["run_id"])},
            "not_hashed": ["ASCEND_PROFILER_OUTPUT/trace_view.json",
                           "ASCEND_PROFILER_OUTPUT/operator_details.csv",
                           "ASCEND_PROFILER_OUTPUT/api_statistic.csv",
                           "ASCEND_PROFILER_OUTPUT/communication.json",
                           "FRAMEWORK/*", "logs/*"],
            "not_hashed_reason": "本审计只消费 kernel_details/op_statistic/"
                                 "step_trace_time 与身份 sqlite；未消费的大文件不入哈希"
                                 "（避免以体积冒充证据；需要时可单列）",
        },
        "domain": {
            "device_label": "Ascend910B（NPU_INFO.name）",
            "device_sku": None,
            "implementation_id": None,
            "precision_observed": "BF16（dense 路径）+ INT8/INT4（MoE/量化 kernel "
                                  "输入 dtype）",
            "tensor_parallel": None,
            "parallel_groups_observed": "见 identity.json",
            "shape_domain": "见 identity.json.observed_geometry / shape_costs.json",
            "applicable_stage": "未切分（step 节点为空；prefill/decode 不可分）",
            "registered_device_candidates": {
                "ascend_910b3": "assumed 旧仓解析映射（model-verse H3 → "
                                "ascend_910B_64G；spec 未官方核验）；profile 无 "
                                "npu-smi board/SKU/驱动，不能证明同一 SKU/驱动"},
            "owner_declared_targets": {
                root: decl for root, decl in sorted(
                    OWNER_DECLARATION["roots"].items())},
            "draft_methods_declared": OWNER_DECLARATION["draft_methods"],
        },
        "owner_declared": {
            "declared_by": OWNER_DECLARATION["declared_by"],
            "status": "owner-declared（模型/方法名）；权重 revision/引擎 commit 未验证",
            "excluded": OWNER_DECLARATION["excluded"],
            "excluded_runs": sorted(
                [{"run_id": i["run_id"],
                  "reason": (i.get("owner_declared") or {}).get("exclusion_reason")}
                 for i in identities
                 if (i.get("owner_declared") or {}).get("excluded")],
                key=lambda x: x["run_id"]),
            "limits": OWNER_DECLARATION["limits"],
        },
        "ledger_row_match": {
            "file": "ledger_rows.json",
            "status": "candidate-only（audit-only；not admitted）",
            "scope": "仅活跃模型 Qwen3.6-35B-A3B 的 MTP run；minidraft 按 owner "
                     "声明排除；Qwen3-8B 为历史/离线模型，不做跨模型行级匹配",
        },
        "admission": {
            "calibration_admission": "not-admitted",
            "admitted": False,
            "prerequisites": ["A-N4：独立 Ascend 设备身份/官方 dense 规格/实现档",
                              "带 vLLM/vllm-ascend commit、权重 revision/checkpoint "
                              "哈希、warmup/active 区间、重复 run、step 边界的匹配采数"],
            "reasons": [item["field"] for item in _identity_gap_summary(identities)],
        },
        "files": sorted(files) + ["README.md"],
        "distill_files": {},
    })
    # README 与产物哈希必须在字节写定后计算
    payloads = dict(files)
    payloads["README.md"] = README.encode()
    for name, data in payloads.items():
        manifest["distill_files"][name] = hashlib.sha256(data).hexdigest()
    files = payloads
    files["manifest.json"] = (_canon(manifest) + "\n").encode()
    return files, manifest


def _candidate_model_for_run(run_ops: set) -> Optional[str]:
    """按本 run 自身出现的 OP Type 判定候选模型（形状一致 ≠ 身份，见 identity）。"""
    if "MoeGatingTopK" in run_ops:
        return "qwen36_35b_a3b"
    if "SwiGlu" in run_ops or "split_qkv_rmsnorm_rope_kernel" in run_ops:
        return "qwen3_8b_dense"
    return None


def _tp_hint(parallel_groups: dict) -> Optional[int]:
    for name, g in (parallel_groups or {}).items():
        if str(name).lower() == "tp":
            return g.get("world_size")
    return None


def _geometry_check(geom: dict, model_id: Optional[str],
                    tp_hint: Optional[int]) -> dict:
    """观测几何 vs catalog 形状身份（只到形状一致，不声称权重身份）。"""
    if model_id is None or model_id not in EXPECTED_GEOMETRY:
        return {"model_id": model_id, "checked": False,
                "reason": "无候选模型/无期望几何表"}
    exp = EXPECTED_GEOMETRY[model_id]
    out = {"model_id": model_id, "checked": True,
           "catalog_model": exp.get("catalog_model"),
           "tp_hint_from_groups": tp_hint,
           "failures": [], "unknown": [], "contamination": [],
           "note": "形状一致不等于权重/量化身份；缺失维度列为 unknown，不猜。"}
    hidden = geom.get("hidden_size_observed")
    if hidden:
        ok = exp.get("hidden_size") in hidden
        out["hidden_size_observed"] = hidden
        out["hidden_size_match"] = ok
        if not ok:
            out["failures"].append(
                f"hidden_size {hidden} ≠ {exp.get('hidden_size')}")
    else:
        out["unknown"].append("hidden_size")
    primary = geom.get("attention_primary") or {}
    if primary:
        q_heads = primary["heads"]
        head_dim = primary["head_dim"]
        tp = tp_hint or 1
        out["attention_primary"] = primary
        out["q_heads_per_rank_observed"] = q_heads
        out["q_heads_total_inferred"] = q_heads * tp
        out["head_dim_observed"] = head_dim
        exp_heads = exp.get("num_attention_heads") or \
            exp.get("num_attention_heads_total")
        if exp_heads is not None and q_heads * tp != exp_heads:
            out["failures"].append(
                f"q_heads_total {q_heads * tp} ≠ {exp_heads}")
        if exp.get("head_dim") is not None and head_dim != exp["head_dim"]:
            out["failures"].append(f"head_dim {head_dim} ≠ {exp['head_dim']}")
        kv_heads = (geom.get("attention_kv_heads_by_head_dim") or {}).get(
            str(head_dim)) or []
        out["kv_heads_per_rank_observed"] = kv_heads
        exp_kv = exp.get("num_key_value_heads") or \
            exp.get("num_key_value_heads_total")
        if kv_heads and exp_kv is not None and len(kv_heads) == 1:
            out["kv_heads_total_inferred"] = kv_heads[0] * tp
            if kv_heads[0] * tp != exp_kv:
                out["failures"].append(
                    f"kv_heads_total {kv_heads[0] * tp} ≠ {exp_kv}")
        elif len(kv_heads) > 1:
            out["unknown"].append(f"kv_heads_ambiguous {kv_heads}")
        else:
            out["unknown"].append("kv_heads")
        groups = geom.get("attention_geometry_groups") or []
        others = [g for g in groups
                  if (g["heads"], g["head_dim"]) != (q_heads, head_dim)]
        for g in others:
            out["contamination"].append(
                f"同 OP Type 另有注意力几何 heads={g['heads']} "
                f"head_dim={g['head_dim']} calls={g['shape_keys']} "
                f"total_us={g['total_us']}（可能为非 catalog 草稿头/异构层，"
                "不并入目标形状身份）")
    else:
        out["unknown"].extend(["attention_geometry", "head_dim"])
    experts = geom.get("moe_experts_observed")
    if model_id == "qwen36_35b_a3b":
        if experts:
            out["moe_experts_observed"] = experts
            if exp["num_experts"] not in experts:
                out["failures"].append(
                    f"num_experts {experts} ≠ {exp['num_experts']}")
        else:
            out["unknown"].append("num_experts")
    out["shape_consistent"] = out["checked"] and not out["failures"]
    return out



def _out_dir(repo_root: Path, cli_out: Optional[str]) -> Path:
    if cli_out:
        p = Path(cli_out)
        return p if p.is_absolute() else (repo_root / p)
    return repo_root / "assets" / "profile_audit" / OUT_DIR_NAME


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Ascend profiling 导入/离线审计（只读，不执行 NPU）")
    ap.add_argument("--src", action="append", default=[],
                    help="外部 profile 根目录（可重复；根下递归找 ASCEND_PROFILER_OUTPUT）")
    ap.add_argument("--out", default=None,
                    help="输出目录（默认 assets/profile_audit/<set>）")
    ap.add_argument("--check", action="store_true",
                    help="只校验产物与 manifest 哈希，不重写")
    ap.add_argument("--json", action="store_true", help="打印 manifest 摘要 JSON")
    args = ap.parse_args(argv)

    out = _out_dir(ROOT, args.out)
    if args.check:
        from simverse.cost.profile_audit import validate_audit_directory
        try:
            man = validate_audit_directory(out, SET_ID)
        except ValueError as exc:
            print(f"FAIL {exc}", file=sys.stderr)
            return 1
        print(f"OK 审计包哈希一致：{man['profile_set_id']}（"
              f"{len(man['distill_files'])} 个产物，{man['run_count']} 个 run）")
        return 0

    if not args.src:
        print("FAIL 需要至少一个 --src（--check 模式除外）", file=sys.stderr)
        return 1
    payloads, manifest = build_pack([Path(s) for s in args.src])
    out.mkdir(parents=True, exist_ok=True)
    for name in sorted(payloads):
        (out / name).write_bytes(payloads[name])
    # 写出后自检（防写盘编码/换行破坏哈希）
    for name, sha in manifest["distill_files"].items():
        if _sha256(out / name) != sha:
            print(f"FAIL 写盘后哈希漂移 {name}", file=sys.stderr)
            return 1
    print(f"OK 审计包已生成 → {out}")
    print(f"   runs={manifest['run_count']}  op_types="
          f"{len(json.loads((out / 'mapping.json').read_text(encoding='utf-8'))['observed_op_types'])}"
          f"  准入={manifest['calibration_admission']}")
    if args.json:
        print(json.dumps(manifest, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
