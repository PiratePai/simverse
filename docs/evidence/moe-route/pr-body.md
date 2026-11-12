## R21：V4.1 MoE scoring/top-k/分组路由 source-checked 建模

独立 PR，不合入 main 前交 Codex 验收。详细报告：[docs/validation/v41-moe-route-20261112.md](docs/validation/v41-moe-route-20261112.md)

### 源码 pin

- **moe-pin.json 11 文件**（`docs/rebuild-20261106/references/vllm/moe-pin.json`）
  - vllm-project/vllm @`f358f0578a83f4e1fe6fc49a492d3ae9362de8e5`（与 R04/R05/R20 既有 pin 同一上游快照）
  - GitHub raw 网络拉取，**blob SHA1 对仓内 vllm-tree.json 逐项校验** + SHA256 清单；tests/test_v41_moe_route.py §1 双侧复验
  - 范围：deepseek_v4 基类 nvidia/model.py + eplb_util、fused_moe/router 8 文件（dsv4_topk/fused_topk_bias/router_factory/base/fused_moe/grouped_topk/cpu/gate_linear）+ moe_output
  - 未读：FusedMoE runner 内核、bf16_mega_gate（deep_gemm 未入 pin）、csrc 专核

### 语义核对结论（逐行号，见报告 §2）

| 项 | 结论 |
|---|---|
| scoring | `sqrt(softplus(logits))`，>20 取 sqrt(logits) 护栏（dsv4_topk.py L85） |
| noaux_tc | e_score_correction_bias[E] fp32 **只导选择不进权重**（选择列 L86-87、无偏权重 L96-98） |
| 分组 | **无分组**：config 无 n_group/topk_group 键，router_factory 仅 use_grouped_topk 走 GroupedTopKRouter——V4.1 平坦偏置 top-6 |
| top-k | 6 轮 argmax、无放回、平手最小 id（L94-95）、NaN→−1e30（L87） |
| 归一/缩放 | sum→1 ×routed_scaling 1.5（sum≤0 除 1 护栏，L104-107）；只作用 routed |
| 共享专家 | 1×inter2304 **无 gate**（≠Qwen3.6），归一后权重 1.0 追加（L447-464，默认 L383-385） |
| 路由 GEMM | GateLinear(h→E, out_dtype=fp32) Replicated（L270-277）→ logits FP32 |
| TP/EP 归属 | 路由三行 per-rank 全量复制；EP **整除域**（mega 断言 L357-362 + fused floor 覆盖性 L419-421 的保守域规则，≠qwen35 余数规则） |
| 其他 | 主干 num_hash_layers=0（无 hash MoE）；视觉 checkpoint bias_vl 哨兵逐行判定 |

### 变更摘要

- `simverse/count/moe.py`：v41_moe_route_rows 三行（moe_gate/moe_score/moe_topk_renorm）+ require_divisible_ep
- `simverse/count/work.py`：ffn_or_moe_rows route 参数（v41 层替换通用 moe_router 行）；build_model_work EP 整除校验
- `simverse/model_ir/families.py`：extras.moe_route spec + 四键逐行号注记 + config 域校验（softmax/greedy/norm=false 拒绝）
- `simverse/lowering/memory.py`：`*-w-moe-route-L{i}` 路由偏置对象（40×3,072B）+ EP 镜像校验
- `simverse/formula_registry.py`：v2.moe_route_gate/score/topk 三条
- `simverse/validation/registry.py`：ev-vllm-v41-moe（source-checked，executed=False）
- `simverse/importers/source_map.py`：sm-v41-moe-route（双侧覆盖）；旧 sm-v41-csa2-moe-unverified 缩窄
- `simverse/model_ir/catalog.py`：checked_paths/unverified_paths/note/evidence_refs 更新
- `tools/replay_v41_moe.py` + `docs/evidence/moe-route/`（before/after/attribution/provenance）
- 文档：报告 / capability-gaps（G-C13 解除）/ IMPLEMENTATION_STATUS / ROADMAP / FOCUS_STATUS

### 测试原始结果

```
py -3.12 -m pytest tests -q          → 332 passed in 11.82s（exit 0；311→332 = 本 PR 新增 21 项）
py -3.12 -m pytest tests/test_v41_moe_route.py -q → 21 passed
node --check simverse/static/workbench.js → exit 0
py -3.12 -m simverse.cli run-v2      → 正常；structure gate 仍按预期拦截（两模型均未排名）
py -3.12 tools/replay_v41_moe.py     → 归因通过（审计面外行逐字段冻结）
```

断言随事实更新（非放宽，逐项归因）：G10/G16/test_model_ir_v02 的 `csa2_moe_routing ∈ unverified_paths` 改指 `moe_deploy_variants`（主干解除后剩余 MegaMoE/DSpark/padding skip 变体）；test_v41_hc replay 冻结面追加 MoE 四行名（由本 PR replay 归因单独冻结）。

### 数值差异（默认 TP8/B1/ctx32768 V4.1 replay，逐 op 归因）

| 量 | 前→后 | Δ | 解释 |
|---|---|---|---|
| decode FLOPs | 7,175,856,742→7,176,457,222 | +600,480 | 40×(score 3,458 + topk 11,554)；gate GEMM 同式 0Δ |
| weight_read | +122,880 B | 40×3,072 | 路由偏置（bias+bias_vl FP32） |
| act_read | +61,760 B | 40×1,544 | FP32 logits 1,536 + input_ids 8 |
| act_write | +48,000 B | 40×(1,584−384) | FP32 logits 出口 + topk 写 − 旧 bf16 logits |
| rows | 927→1,007 | +80 | 每层 1 行拆 3 行 |
| M01 | 116.160230→116.160299 ms | +0.0000695 | 三行 T0 计价 |
| steady/known_peak | +122,880 B | | 偏置常驻对象 |

M02/M04/M05/M06 降级与 unknown 传播不变；**structure_complete 仍 False**（量化映射/swiglu clamp/稀疏 MLA 注意力核本体/MTP 尾部/compress_ratios 尾部缺口原样保留）。

### 剩余限制

1. source-checked ≠ executed：kernel/CPU forward/GPU 均未执行；op 为解析算术口径。
2. 部署变体未建模：native MegaMoE（bf16_mega_gate，需 EP+fp4+SP）、FP32/bf16x3 router GEMM 分层、dsv4_topk Triton 档约束（E∈{256,384}/k=6）外的 E/k 值未独立核。
3. DSpark 草稿层 MoE（E=128/k=3）方法层字段；padding skip/CUDA-graph 整宽路由缓冲、defer_finalize（需 scaling==1.0，本模型 1.5 不适用）未入账。
4. quantization_config（fp8/FP4 专家）不在本 PR；input_ids 按 int64 上界计。


---

### 首轮 Codex 验收（2026-11-12）：不通过 → 已修复（commit b5688fa）

| 意见 | 处置 |
|---|---|
| 高｜报告 §2 行号指错 | 全部对实销行号更正（sqrtsoftplus dsv4 L85、bias L86-87、无偏 gather L96-98、归一 L104-107、共享追加 L447-464/默认 L383-385）；报告/公式注册/source_map/证据 claim/代码注释同步 |
| 高｜replay 后工作区漂移 | 工具新增 `--check` 无漂移复算模式；plain run 的 manifest 身份字段刷新（code_revision/run_id）注明为登记性漂移；`--check` 验证账本确定性与归因仍成立 |
| 中｜oracle −87 阈值无源码依据 | 标注为 IEEE-754 单精度解析假设（2^−126 → ln≈−87.3，非 pinned 断言）；NaN→−1e30 主张收窄到有限候选 score+bias > −1e30 域 |
| 中｜（复核补充）EP 依据不实 | fused 路径实为 floor 除法（L419-421）无断言；EP 校验改写为 mega 断言（L357-362）+ fused 覆盖性的保守域规则，全部文档同步 |
| 附带 | baseline-provenance 补 before_sha256（3addaa5f…）；PR 描述副本入仓 docs/evidence/moe-route/pr-body.md |

修复后：332 passed；node check OK；`tools/replay_v41_moe.py --check` 通过。详见报告 §8。
