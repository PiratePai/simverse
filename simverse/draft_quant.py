"""DFlash 草稿 ctx-KV 的**实现路径**（显式配方，不由位宽推断）。

pinned vLLM `qwen3_dflash.py` @dc74b8cb 的三条分支：

  * L108-113 `_reads_raw_weights`：非 CPU 且全部草稿层 `qkv_proj` 的
    `quant_method` 为 `UnquantizedLinearMethod` → 后加载的 `.weight` 是
    普通张量，可被 `F.linear` 直接消费；
  * L529-531：全部 `qkv_proj` 为 `ModelOptMxFp8Config`（且非未量化）→
    构造独立 `_context_kv_proj`（ReplicatedLinear，仅装载 K/V 行）；
  * 其余（量化方法打包存储 / CPU repack）→ L688-697 逐层执行**完整**
    `proj(normed_context_states)`，再 `qkv[:, q_size:]` 切掉 Q。

容量/权重副本的差异（L541-568, L679-697）：

  * `_fused_kv_weight` 的新 `torch.cat` 副本**只**在 `_reads_raw_weights`
    为真时建立；
  * `_k_norm_weights` 的 `torch.stack` 在**所有**路径建立（L565-568），
    与量化分支无关；
  * MXFP8 的 `_context_kv_proj` 是**另一物理对象**（量化打包存储），
    既不是 raw 融合副本，也不是逐层回退；
  * 逐层回退不产生融合副本，但每轮读的是完整 QKV 权重（含 Q 行）。

`draft_quant` 的 `dense_w/expert_w/act/kv` 是**计价位宽**，不是 vLLM
`quant_method`；单个位宽阈值不能推出上述任一分支。路径必须由场景显式
给出（`ScenarioV2.draft_ctx_kv_path`），缺失/未知时计数与容量层显式拒绝，
不猜默认执行分支。
"""
from __future__ import annotations

FUSED_RAW = "fused_raw"
MXFP8_CONTEXT_PROJ = "mxfp8_context_proj"
PER_LAYER_FALLBACK = "per_layer_fallback"
UNKNOWN = "unknown"

KNOWN_PATHS = (FUSED_RAW, MXFP8_CONTEXT_PROJ, PER_LAYER_FALLBACK)
ALL_PATHS = KNOWN_PATHS + (UNKNOWN,)

PATH_LABELS = {
    FUSED_RAW: "raw 融合 GEMM（_fused_kv_weight 新 cat 副本）",
    MXFP8_CONTEXT_PROJ: "MXFP8 独立 context KV 投影（_context_kv_proj）",
    PER_LAYER_FALLBACK: "逐层完整 qkv_proj 后切 Q（量化/repacked 回退）",
    UNKNOWN: "未知（未提供实现路径，不猜）",
}

# RMSNorm 参数 dtype 与量化 Linear 的计价位宽是两件事。pinned vLLM
# layernorm.py L42-58 在未显式传 dtype 时用 torch.get_default_dtype()；
# base_loader.py L68-85 以 model_config.dtype 包住模型初始化；草稿经
# dflash_utils.py L51-54 get_model(draft_model_config) 加载。因此 k_norm
# stack 继承草稿 config dtype，而不是 draft_quant.dense_w。
_DTYPE_BYTES = {
    "bfloat16": 2, "bf16": 2,
    "float16": 2, "fp16": 2, "half": 2,
    "float32": 4, "fp32": 4, "float": 4,
    "float64": 8, "fp64": 8, "double": 8,
}


def dtype_bytes(dtype_hint: str) -> int:
    """草稿模型 dtype → 每元素字节；未知 dtype 显式拒绝。"""
    key = (dtype_hint or "").strip().lower()
    if key.startswith("torch."):
        key = key[len("torch."):]
    if key not in _DTYPE_BYTES:
        raise ValueError(
            f"草稿 dtype 未知/不支持：{dtype_hint!r}；RMSNorm 参数/其堆叠副本"
            "不能回退到 draft_quant.dense_w（量化 GEMM 位宽不是参数 dtype）。"
            "请从草稿 config 的 dtype/torch_dtype 登记 DraftIR.dtype_hint。")
    return _DTYPE_BYTES[key]


def normalize_ctx_kv_path(value) -> str:
    """规范化场景输入；非法字符串直接拒绝（不静默当 unknown）。"""
    if value is None:
        return UNKNOWN
    if not isinstance(value, str):
        raise ValueError(
            f"draft_ctx_kv_path 必须是字符串，收到 {type(value).__name__}")
    path = value.strip().lower()
    if path not in ALL_PATHS:
        raise ValueError(
            f"未知 draft_ctx_kv_path={value!r}；允许 {', '.join(ALL_PATHS)}"
            "（显式配方，不能由 dense_w 推断）")
    return path


def require_ctx_kv_path(value) -> str:
    """计数/容量入口：未知路径 = unsupported，不返回任何猜测值。"""
    path = normalize_ctx_kv_path(value)
    if path == UNKNOWN:
        raise ValueError(
            "draft ctx-KV 执行路径未知：dense_w 只是计价位宽，不能推出 "
            "UnquantizedLinearMethod/MXFP8/逐层回退。请显式提供 "
            "draft_ctx_kv_path ∈ {fused_raw, mxfp8_context_proj, "
            "per_layer_fallback}。")
    return path


def path_note(path: str) -> str:
    return PATH_LABELS[normalize_ctx_kv_path(path)]


def path_unknown_components(path: str) -> tuple[str, ...]:
    """该路径下无源码尺寸证据、需传播到峰值完整性的组件。"""
    if normalize_ctx_kv_path(path) == MXFP8_CONTEXT_PROJ:
        return (
            "MXFP8 _context_kv_proj 的 block-scale 存储布局"
            "（modelopt.py 未入 pin；当前只计 packed 8bit 权重下界）",
        )
    return ()
