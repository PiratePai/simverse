"""simverse.model_ir —— 共享模型结构/forward 证据子包（Q3：独立子包，单独版本）。

版本 0.3.0。职责：ModelIR/DraftIR schema、config 家族解析、字段覆盖、
结构完备性标志。不依赖仿真器、成本或 UI；两旧仓通过固定版本消费本包。

证据绑定：catalog 中每个模型携带 SourceIdentity（repo/revision/config sha256）；
forward 分支证据见 simverse/validation/registry.py 与 assets/identity。
"""
from .schema import (FULL, LINEAR, MLA, MLA_CSA2, SLIDING, AttentionIR, Csa2IR,
                     DraftIR, LayerIR, LinearAttnIR, MlaIR, ModelIR, MoeIR,
                     SourceIdentity)
from .families import (parse_deepseek_v3, parse_dflash_draft, parse_gqa,
                       parse_qwen3_5_moe, parse_qwen3_moe, parse_qwen3_next,
                       parse_v41_csa2)
from .catalog import PENDING_ASSETS, build_catalog, catalog_json

__version__ = "0.3.0"

__all__ = [
    "__version__", "FULL", "LINEAR", "MLA", "MLA_CSA2", "SLIDING",
    "AttentionIR", "Csa2IR", "DraftIR", "LayerIR", "LinearAttnIR", "MlaIR",
    "ModelIR", "MoeIR", "SourceIdentity",
    "parse_deepseek_v3", "parse_dflash_draft", "parse_gqa", "parse_qwen3_5_moe",
    "parse_qwen3_moe", "parse_qwen3_next", "parse_v41_csa2",
    "PENDING_ASSETS", "build_catalog", "catalog_json",
]
