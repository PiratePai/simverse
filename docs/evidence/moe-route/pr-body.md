## R21：V4.1 MoE scoring/top-k/分组路由 source-checked 建模

独立 PR，不合入 main 前交 Codex 验收。详细报告：[docs/validation/v41-moe-route-20261112.md](docs/validation/v41-moe-route-20261112.md)

### 源码 pin

- **moe-pin.json 14 文件**（`docs/rebuild-20261106/references/vllm/moe-pin.json`）
  - vllm-project/vllm @`f358f0578a83f4e1fe6fc49a492d3ae9362de8e5`（与 R04/R05/R20 既有 pin 同一上游快照）
  - GitHub API 拉取，**blob SHA1 对仓内 vllm-tree.json 逐项校验** + SHA256 清单；tests/test_v41_moe_route.py §1 双侧复验
  - 范围：deepseek_v4 基类 nvidia/model.py + eplb_util、fused_moe/router 8 文件（dsv4_topk/fused_topk_bias/router_factory/base/fused_moe/grouped_topk/cpu/gate_linear）+ moe_output + **linear.py/base_loader.py/fused_moe/layer.py（gate 权重 dtype/工厂默认值）**
  - 未读：FusedMoE runner/dispatch/combine 内核、bf16_mega_gate（deep_gemm 未入 pin）、ExpertMapManager 全文、csrc 专核

### 语义核对结论（逐行号，见报告 §2）

| 项 | 结论 |
|---|---|
| scoring | `sqrt(softplus(logits))`，>20 取 sqrt(logits) 护栏（dsv4_topk.py L85） |
| noaux_tc | e_score_correction_bias[E] fp32 **只导选择不进权重**（选择列 L86-87、无偏权重 L96-98） |
| 分组 | **无分组**：config 无 n_group/topk_group 键，router_factory 仅 use_grouped_topk 走 GroupedTopKRouter——V4.1 平坦偏置 top-6。pinned model 调用未传该参数；默认 False 已随 fused_moe/layer.py 入 pin |
| top-k | 6 轮 argmax、无放回、平手最小 id（L94-95）、NaN→−1e30（L87） |
| 归一/缩放 | sum→1 ×routed_scaling 1.5（sum≤0/NaN 时分母取 1，L104-107；NaN 权重不被清除）；只作用 routed |
| 共享专家 | 1×inter2304 **无 gate**（≠Qwen3.6），routed 归一后权重 1.0 **独立相加**（moe_output.py L151-175；FusedMoEFactory 调用未传 n_shared_experts/fuse_shared_experts，L447-464 的融合槽未启用） |
| 路由 GEMM | GateLinear(h→E, out_dtype=fp32) Replicated（L270-277）；**权重沿 model loader BF16**（未传 quant_config/params_dtype；linear.py L287-295/L179-199、base_loader.py L86、gate_linear.py L306-341）；**输入由 pinned mHC layer_input_cur BF16 证明**（tilelang.py L900/L1096，R05 pin；model.py L731-816 直接 self.ffn(x)），不是由任意 model dtype 泛化；有效域仅 official V4.1 bf16，非 BF16 拒绝；fp32 只 logits 出口；每层 weight 3,932,160 B |
| TP/EP 归属 | 路由三行 per-rank 全量复制；EP **保守整除域**（仅 mega 断言 L357-362；fused L419-421 的 start_idx 只赋值、未被所有权消费，fused 通用余数映射未入 pin/未建模，不主张 fused 非法；≠qwen35 余数规则） |
| 其他 | 主干 num_hash_layers=0（无 hash MoE）；视觉 checkpoint bias_vl 哨兵逐行判定 |

### 变更摘要

- `simverse/count/moe.py`：v41_moe_route_rows 三行（moe_gate/moe_score/moe_topk_renorm）+ require_divisible_ep
- `simverse/count/work.py`：ffn_or_moe_rows route 参数（v41 层替换通用 moe_router 行）；build_model_work 保守 EP 整除域校验（仅 mega 断言）
- `simverse/model_ir/families.py`：extras.moe_route spec + 四键逐行号注记 + config 域校验（softmax/greedy/norm=false 拒绝）
- `simverse/lowering/memory.py`：`router_b=h·E·2B`（gate BF16）+ `*-w-moe-route-L{i}` 路由偏置对象（40×3,072B）+ EP 镜像校验
- `simverse/formula_registry.py`：v2.moe_route_gate/score/topk 三条
- `simverse/validation/registry.py`：ev-vllm-v41-moe（source-checked，executed=False）
- `simverse/importers/source_map.py`：sm-v41-moe-route（双侧覆盖）；旧 sm-v41-csa2-moe-unverified 缩窄
- `simverse/model_ir/catalog.py`：checked_paths/unverified_paths/note/evidence_refs 更新
- `tools/replay_v41_moe.py` + `docs/evidence/moe-route/`（before/after/attribution/provenance）
- 文档：报告 / capability-gaps（G-C13 解除）/ IMPLEMENTATION_STATUS / ROADMAP / FOCUS_STATUS

### 测试原始结果

```
py -3.12 -m pytest tests -q          → 收集 334；Codex 宿主 gate 修复后 333 passed，dtype 域负例后待复测；本机沙箱 328 passed + 6 tmp_path ACL 错误
py -3.12 -m pytest tests/test_v41_moe_route.py -q → 23 passed
node --check simverse/static/workbench.js → exit 0
py -3.12 -m simverse.cli run-v2      → 正常；structure gate 仍按预期拦截（两模型均未排名）
py -3.12 tools/replay_v41_moe.py     → 归因通过（审计面外行逐字段冻结）
```

断言随事实更新（非放宽，逐项归因）：G10/G16/test_model_ir_v02 的 `csa2_moe_routing ∈ unverified_paths` 改指 `moe_deploy_variants`（主干解除后剩余 MegaMoE/DSpark/padding skip 变体）；test_v41_hc replay 冻结面追加 MoE 四行名（由本 PR replay 归因单独冻结）。

### 数值差异（默认 TP8/B1/ctx32768 V4.1 replay，逐 op 归因）

| 量 | 前→后 | Δ | 解释 |
|---|---|---|---|
| decode FLOPs | 7,175,856,742→7,176,457,222 | +600,480 | 40×(score 3,458 + topk 11,554)；gate GEMM 同式 0Δ |
| weight_read | 1,570,196,928→1,648,963,008 | +78,766,080 | 40×(gate 3,932,160−1,966,080)=78,643,200 + 40×3,072 bias |
| act_read | 10,246,334→10,512,894 | +266,560 | 40×5,120 gate 输入 BF16 + 61,760 score（FP32 logits+ids） |
| act_write | 6,282,096→6,330,096 | +48,000 | 40×(1,584−384)：FP32 logits 出口 + topk 写 − 旧通用 act=1B 的 384/层 |
| rows | 927→1,007 | +80 | 每层 1 行拆 3 行 |
| M01 | 116.160230→116.183836 ms | +0.023606 | 三行 T0 计价；gate 字节主导 |
| steady/known_peak | 35,690,092,992/35,690,106,304→35,768,859,072/35,768,872,384 | +78,766,080 | gate 权重 40×1,966,080 + 偏置 40×3,072 |

M02/M04/M05/M06 降级与 unknown 传播不变；**structure_complete 仍 False**（量化映射/swiglu clamp/稀疏 MLA 注意力核本体/MTP 尾部/compress_ratios 尾部缺口原样保留）。

### 剩余限制

1. source-checked ≠ executed：kernel/CPU forward/GPU 均未执行；op 为解析算术口径。
2. 部署变体未建模：native MegaMoE（bf16_mega_gate，需 EP+fp4+SP）、GateLinear bf16 专核/cuBLAS 档未执行（fp32/bf16x3 只适用 fp32 router 权重）、dsv4_topk Triton 档约束（E∈{256,384}/k=6）外的 E/k 值未独立核。
3. DSpark 草稿层 MoE（E=128/k=3）方法层字段；padding skip/CUDA-graph 整宽路由缓冲、defer_finalize（需 scaling==1.0，本模型 1.5 不适用）未入账。
4. quantization_config（fp8/FP4 专家）不在本 PR；**gate 权重沿 model loader BF16、输入由 mHC layer_input BF16 证明（2B）；有效域固定 official V4.1 bf16，非 BF16 拒绝**；专家/其余 dense 权重布局仍缺；input_ids 按 int64 上界计。


---

### 首轮 Codex 验收（2026-11-12）：不通过 → 已修复（commit b5688fa）

| 意见 | 处置 |
|---|---|
| 高｜报告 §2 行号指错 | 全部对实销行号更正（sqrtsoftplus dsv4 L85、bias L86-87、无偏 gather L96-98、归一 L104-107、共享追加 L447-464/默认 L383-385）；报告/公式注册/source_map/证据 claim/代码注释同步 |
| 高｜replay 后工作区漂移 | 工具新增 `--check` 无漂移复算模式；plain run 的 manifest 身份字段刷新（code_revision/run_id）注明为登记性漂移；`--check` 验证账本确定性与归因仍成立 |
| 中｜oracle −87 阈值无源码依据 | 标注为 IEEE-754 单精度解析假设（2^−126 → ln≈−87.3，非 pinned 断言）；NaN→−1e30 主张收窄到有限候选 score+bias > −1e30 域 |
| 中｜（复核补充）EP 依据不实 | fused 路径实为 floor 除法（L419-421）无断言；EP 校验改写为 mega 断言（L357-362）+ fused 覆盖性的保守域规则，全部文档同步 |
| 附带 | baseline-provenance 补 before_sha256（3addaa5f…）；PR 描述副本入仓 docs/evidence/moe-route/pr-body.md |

修复后：332 passed（gate 修复前；gate 修复后 333、dtype 负例后 334 收集，见下）；node check OK；`tools/replay_v41_moe.py --check` 通过。详见报告 §8。

---

### 第三轮复核（2026-11-13）：修正证据链残留，不改变数值

| 发现 | 处置 |
|---|---|
| 高｜fused EP 依据仍不实 | L419-421 的 `experts_start_idx` 只赋值、未被 FusedMoE 所有权消费；fused 实际映射在未入 pin 的 ExpertMapManager（余数分配）。删除"floor 覆盖性/尾部无 owner/与 generic 失配"表述；EP 规则改为**仅 mega 断言**的保守建模域，不主张 fused 非法。行为不变（仍拒绝非整除组合） |
| 中｜共享专家路径 | V4.1 FusedMoEFactory 调用未传 n_shared_experts/fuse_shared_experts → `num_fused_shared_experts=0`；共享专家独立相加权重 1.0（moe_output.py L151-175）。L447-464 是通用融合槽能力，不是本模型实际路径 |
| 中｜分组默认证据链 | pinned model 调用未传 use_grouped_topk；默认 False 在 fused_moe/layer.py，已随 gate 修复入 pin；平坦 top-k 结论有字节级依据 |
| 低｜oracle −87 措辞 | −87.3 是最小 normal 的对数，不是 underflow 门槛；改为 normal-range 近似，subnormal/FTZ 未实测 |
| 低｜EP 测试域 | 只列实际合法 TP/EP {1,2,4,8}（3/6 被 o_groups=8 拒绝）；移除 dead `experts_start_idx` 锚点；加 before/scenario SHA256 自校验 |

验证（本机沙箱）：`tests/test_v41_moe_route.py` 23 passed；`tools/replay_v41_moe.py --check` 通过；全量 `pytest tests` 为 328 passed + 6 个 `tmp_path` setup PermissionError（沙箱 ACL，测试本体未失败；Codex 宿主 gate 修复后 333 passed，dtype 域负例后 334 收集待复测）。未改 main、未合入 PR。

---

### GateLinear 权重字节修复（2026-11-13，Codex 数值验收）

**问题**：`moe_gate` 的 weight_read/capacity 用场景 `dense_w/8`（Q8=1B），memory `router_b` 同；实际 **gate 权重沿 model loader BF16、gate 输入由 mHC layer_input BF16 证明**，均 2B。每层 per-rank 1,966,080→3,932,160 B，40 层少计 78,643,200 B（含 bias 的 weight_read Δ78,766,080 B）。

**源码链（已入 pin 14 文件；mHC 用已有 R05 pin）**：
- 权重：model.py L270-277（未传 quant_config/params_dtype）→ linear.py L287-295（`params_dtype=torch.get_default_dtype()`、`quant_config None→UnquantizedLinearMethod`）L179-199（weight dtype=params_dtype）→ base_loader.py L86（`set_default_torch_dtype(model_config.dtype)`）→ official config dtype=bfloat16；
- 输入：mHC tilelang.py L900/L1096 的 `layer_input_cur` 硬编码 `dtype=torch.bfloat16`（R05 pin），model.py L731-816 的 fused_post_pre 输出直接 `self.ffn(x)`；
- `out_dtype=fp32` 只约束 logits 出口；不能把输入泛化成任意 model dtype。

**dtype 域收紧**：`extra.moe_route` 的 `dtype_domain=official_v41_bfloat16`、`gate_act_source=mhc_layer_input_cur_bf16`；full_config dtype=float16/float32/float64/缺失一律拒绝；无 full_config 的内部调用只对固定 V4.1 资产采用 BF16 默认（注释登记）。

**修复**：count gate 权重/输入改 2B；memory router_b 改 2B；新增 `test_gate_bytes_follow_loader_and_mhc_dtype_not_scenario_quant`（Q8/Q4 gate 字节不变 + memory 手算）与 `test_config_domain_rejects_non_bf16_dtype`（非 BF16 负例 + 无 full_config 固定资产默认）；pin 14 文件；before/after/attribution 重放。数值：weight_read +78,766,080B、act_read +266,560B、act_write +48,000B、rows +80、M01 +0.023606ms、steady/known_peak +78,766,080B。structure_complete 仍 False，量化专家/其余 dense 映射、swiglu、MLA 核、MTP/compress 尾部缺口不变。
