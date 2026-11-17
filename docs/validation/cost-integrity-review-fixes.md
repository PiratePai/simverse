# R27：cost-integrity 轮修复报告（计算口径/统计身份/指标元数据/运行身份）

日期：2026-11-17。基线 HEAD `11965105d03252dc2a45d22b68d6d668c2c61504`
（main，R25+R26 合入后）。执行环境：本机 ZCode 会话，Windows/py3.12；
全套验证在干净 HEAD+本轮改动的独立 worktree（`../nv-r27-verify`）完成，
隔离并行工作台任务在主工作树的在途编辑（simverse/webapp.py、static/、
cli.py、test_consistency.py 等，均属对方职责文件，本轮未触碰）。
证据层级：本报告全部修复为**工程修复 + 解析复算**；未执行任何 vLLM/
SGLang 模块、kernel、CPU forward 或 GPU 测量。证据包
`docs/evidence/cost-integrity-r27/`（before 于 pre-R27 工作区捕获，
provenance 入册；after/attribution 由 `tools/replay_cost_integrity_r27.py`
生成，`--check` 为活性无漂移入口）。

## 1. 修复项

### A. 执行配方与计算峰值路由（cost/t0.py、model_ir/families.py、pipeline.py）

**问题**：`dtype_route()` 只按 `dense_w` 存储位宽选 GEMM 计算档。V4.1
既定 Marlin W8A16/W4A16 路径（R22 已 pin）保留 BF16 激活、在 BF16
Tensor Core 上执行，却按 FP8 峰值（1979 TFLOPS dense H100）计价——
计算吞吐乐观 ~2×。R22 §6.3 已披露该限制并注明"需 H100 marlin 吞吐
证据，防无据改价"。

**修复**（区分存储精度/计算精度/执行单元类别）：
- `model_ir/families.py` extras.quant 新增 `gemm_compute_dtype="fp16"`
  （bf16 与 fp16 同档）与 `gemm_compute_basis`（来源：quant-pin
  @f358f057 marlin.py L14-26 混合精度 GEMM、oracle/mxfp4.py L461-486
  SM90 优先级；未量化类 BF16 本身同档）。这是**模型执行配方**层面的
  注册，不是 cost 层的新启发。
- `cost/t0.dtype_route(quant, quant_recipe=None)`：有配方时 GEMM 档由
  配方选择；无配方（Qwen3.6、DFlash 草稿、MTP、chain 草稿——均无
  extras.quant）沿用 dense_w 存储位宽启发，**不引入"所有 GEMM 只看
  act 位宽"的新全局规则**，启发档在 cost_basis 中披露为乐观假设。
- 元素/注意力核路由不变（看 act）：配方只钉 GEMM kernel 执行域；非
  GEMM 行"按路由档 Tensor Core 峰值近似计价"如实披露（`_BASIS_
  ELEMENTWISE`：非 Tensor Core 执行单元/launch/融合未建模，带宽主导行
  影响小）——不把逐元素算子当作 Tensor Core GEMM，也不虚构 CUDA core/
  SFU 峰值（不扩大未验证执行域）。
- **理论峰值路由修正 ≠ 实测吞吐校准**：修正后取 bf16 档理论 dense 峰值
  ×mfu_cap 0.75；Marlin 实际吞吐低于 bf16 理论峰值，无 H100 实测证据
  不做进一步折减（校准缺口与 profile 路径分立；Humming W4A8/SM100
  W8A8/emulation 域外变体不进路由，G-C14 不变）。
- launch/融合/调度/CUDA Graph 不足**登记为建模缺口**（assumption
  `A2b_kernel_gaps`）：T0 无逐算子 launch（kernel 映射未建立，逐逻辑
  算子加 launch 是无据双计），不实施"删除重复 launch"类改动。
- 成本修正不改原始计数：三场景 work/cost.memory 逐字段零漂移
  （evidence/cost-integrity-r27/attribution.json counts_invariant；测试
  `test_cost_route_fix_left_ledgers_untouched` 用 before 证据钉死）。

### B. Roofline 后端元数据（cost/t0.py、pipeline.py）

**问题**：`_roofline_points()` 中
`dtype = gemm_dtype if row.op not in _ELEMENTWISE else gemm_dtype` 两个
分支相同且 `dtype` 未使用；前端无从获知物理参数。

**修复**：`price_row` 在**同一次计价**内附带行级元数据（不另建第二套
路由）：
- `compute_dtype`：本行计算档 tier 键（"fp16" 档含 BF16）；
- `peak_tflops`：该档设备规格峰值（dense 口径；TFLOP/s 十进制）；
- `effective_peak_tflops`：本次计价采用的峰值 = 规格 × mfu_cap；
- `bandwidth_tb_s`：本次计价带宽（十进制 TB/s）；
- `cost_basis`：口径短述（配方档/启发档/元素近似；host Engram 期望读
  未计价时追加说明）。
`_roofline_points` 重复分支删除，字段直接从计价输出复制；既有字段
（row_id/layer/op/stage/ai/attained/bound/t_ms）全保留，**新增为可选
字段，旧前端继续工作**（workbench.js 只读既有键，已核）；row_id 稳定
（stage:layer:op）。无法提供依据的行本设计不存在（每行必有路由依据：
配方或位宽启发），若未来出现则按契约用 null+reason（本轮未触发）。
T0 估算吞吐不表述为实测（attained_tflops 仍为解析量，result_kind=
analytical-estimate 不变）。

### C. 期望通信字节（pipeline.py）

**问题**：DFlash/MTP 的 t2d 期望字节 `int()` 截断。锚点：Qwen3.6+
DFlash、K=15、α=0.72（E[N]=3.552800756624948，8 层×h2048×1B）：
58209.08759654315 被截成 58209（复现于 before-qwen-dflash.json）。

**修复**：`t2d` 保 `float()`（E[N] 推导的期望值，expectation 身份）；
`d2t` 为协议确定字节（6B/token）保持整数。下游核对：M19 指标
statistic=expectation（契约允许非整浮点）；`cost.link.bytes` 序列化
（canonical_json/js_safe 浮点原样）；链路成本 t_t2d = fixed+bytes/bw
（浮点运算）；round/序列化无二次截断（测试
`test_expected_bytes_survive_json_roundtrip`）。**不改接受率模型与原
消息公式**；消息包长取整、对齐、采样实际消息与期望值分列处理（不在
本处混算——per-round 期望口径维持）。冻结 replay 场景数值影响：MTP
k=1 α0.75 与 DFlash K4 α0.75 的 t2d 恰为整数值（3584.0/50000.0），
浮点化后数值不变（`==` 成立），mtp-map/mrope-map 证据零漂移。

### D. 内部 chain 计价（pipeline.py）

- 草稿计数（chain_step_rows 用 dquant）与计价同源：计价改用
  `dtype_route(dquant)`（旧代码误用目标 quant 的 gemm/ew 档）。
- 设备：colocated 用目标设备；dt_serial/dt_pipeline_sim 用草稿设备
  （旧代码恒用目标设备计价草稿步）。
- 不把目标 Engram Host 开关（engram_lookup_host）套到不含 Engram 的
  tiny_gqa 草稿计价。
- 不支持的并行组合显式拒绝：chain + DT + draft_tp>1 → `unsupported_
  combo`（tiny 无分片/独立组配方，不静默按 TP1 计价）。
- 内部研究隔离不变：chain_draft/chain_moe_proxy 仍仅
  `allow_internal_methods=True` 可运行（产品入口失败于
  internal_research_method；tiny 不在活跃目录），不补造真实草稿身份
  或部署资格（method_recipe_ready=approximated 不变）。

### E. 指标警告覆盖（pipeline.py）

**问题**：dt_pipeline_sim 模式对 M01 整体重建 Metric，覆盖 Engram
host 下界声明（复现：before-v41-chain-pipeline.json 的 M01 reason 只
剩流水注记）——该组合只能内部触发（capability 中 Host 警告仍在）。

**修复**：增补说明时保留既有 value/support/reason/evidence_refs/
formula_ref/input_refs/assumption_refs，追加流水注记并**去重**（不堆叠）。
内部可触发场景同时表达"Engram host 表查找与预取时延未计"与"串行 TPOT、
流水周期不可混用"；capability 侧 `engram_host_lookup_latency`/
`stage_cost_complete` unknown 降级保留（测试钉死）。

### F. 运行身份（pipeline.py `_device_snapshot`）

职责分立：
- **device_snapshot_digest**：实际计价消费的硬件/链路**数值与来源**
  （target/draft：peak_by_dtype/hbm_bw/mem_capacity/sm_count/mfu_cap/
  launch_overhead_ms/card_count/sources；link：bw/fixed_ms/source；
  携带 target_tp/target_ep/draft_tp 便于独立解读）——不只设备 ID：
  同 ID 改带宽/峰值/链路 → 快照与 run_id 变化（测试
  `test_device_value_change_changes_identity` 用 dataclasses.replace
  验证）。
- **input_digest**（scenario 全量）：部署语义参数（target_ep/draft_tp/
  v41_kv_record/engram_table_placement/budget/接受率/精度档）——完整
  身份不遗漏（关键参数变化 → run_id 变化，测试覆盖）。
- 非语义字段（时间戳等）不入快照；键序经 canonical_json 排序归一，
  字段排序不影响身份（测试钉死）。

**run_id 基变更（版本变化，明确兼容方式）**：本轮起 run_id 输入含
数值快照，**全部历史 run_id 不再可复现**。兼容方式：
1. 历史证据包（docs/evidence/*/before|after.json）**不回改**——其价值
   是账本/指标切片与归因，不依赖 run_id 可复现性；replay `--check` 均
   不比较 run_id（已核 replay_v41_*、replay_qwen36_* 实现）。
2. run_id 语义本来就是"身份哈希"（run.py docstring），非持久外键；新
   证据（cost-integrity-r27）在新基上生成。
3. 若未来需要跨版本对账，用 scenario_digest + content_digest（不含
   设备数值快照的输入侧）做关联，run_id 只做同版本幂等键。

## 2. 成本变化（修复前后分解，evidence/cost-integrity-r27）

场景与归因（before=HEAD 1196510 生产代码；after=本轮修复）：

| 场景 | 指标 | before | after | Δ | 归因 |
|---|---|---|---|---|---|
| V4.1 baseline TP8/B1/ctx32768 | M01/M01_baseline/M04 | 116.33572727816723 | 116.33641192134922 | **+0.00068464 ms** | GEMM 类逐行 t 和 0.7694574845888567→0.7701421277708517（fp8→bf16 档，t_comp ×2 只经 max+0.15·min 的 min 项进入；elementwise 类 0.03859879357835783 不变，t_mem 全部不变；M01 Δ == GEMM 类 t Δ，逐类对账闭环） |
| 同上 | work/cost.memory | — | — | **0（逐字段）** | 计数不可变 |
| Qwen3.6+DFlash colocated TP8 K15/α0.72 blk16 | M01 | 36.859180745326654 | 36.859180745326654 | 0 | t2d 58209→58209.08759654315 对 t_t2d 影响 ≈4e-10 ms，低于该量级双精度分辨率；无配方模型路由不变 |
| 同上 | cost.link.bytes.t2d | 58209（int 截断） | **58209.08759654315（float 期望）** | +0.08759654315 B | 修复 C |
| V4.1 chain+dt_pipeline_sim（内部） | M01 reason | 仅流水注记（Host 警告丢失） | Host 警告+流水注记并存 | — | 修复 E；数值不变 |
| 全部场景 | run_id | 旧基 | 新基（数值快照） | — | 修复 F（见 §1.F 兼容方式） |

**"选错计算峰值"≠"整模型 TPOT 错 2 倍"**：V4.1 decode 行带宽主导
（B=1），t_comp 翻倍只通过 0.15·min 项进入，模型级影响 +5.9e-6 相对
量。反向说明：若未来出现计算主导场景（大批量/prefill），档位修正影响
会显著放大——这正是把峰值档位做对、且把 Marlin 实测吞吐校准另列的
原因。

**csa2-core 证据重基**：`docs/evidence/csa2-core/after.json` 按新成本
档重写（work 逐字段与 R25 版一致——已断言；M01/M01_baseline/M04 三项
重定价，old→new 见 rebase-r27-provenance.json 的 changed_metrics）；
attribution.json 保留 R25 历史记录；`tools/replay_v41_csa2_core.py`
main() 归因改**同价复算**（两侧行都用当前 dtype_route 计价，t_ms_delta
只反映 R25 结构变化；M01 差含全局重定价不再与逐行 delta 绑定）。
链式入口 `replay_v41_quant/moe/swiglu --check`（比较 work）与
`replay_qwen36_mtp/mrope --check` 全部 exit 0。

## 3. 测试与 replay

- 新增 `tests/test_cost_integrity_fixes.py`（20 项）：路由配方/启发档
  分立、行级计数不变、元数据单位与一致性（t_comp/t_mem 与
  peak/effective/bw 自洽）、V4.1 配方 basis 与 Qwen 启发 basis、可选
  字段兼容、t2d 锚点（58209.08759654315/d2t=60 int/M19 expectation/
  MTP 3481.6）、JSON 往返、chain 精度同源（目标档变化草稿步不变+
  dquant 变化草稿步变化）、chain DT 设备（l40 vs h100 可区分）、
  draft_tp>1 拒绝、内部隔离、Host+pipeline 警告并存不堆叠+capability
  保留、run_id 稳定/设备数值变化/部署参数变化/键序不变、before 证据
  钉死 work/memory 零漂移。
- 新增 `tests/test_v41_quant_numeric_r27.py`（6 项，纯 stdlib）：
  e4m3 量化-反量化半 ULP 界、e2m1 值域/误差、MXFP4 打包双解码器一致
  （位运算 vs 查找表）、W8A16 反量化 GEMM 等价（bf16 操作数+fp32 累加
  vs f64 oracle，worst rel < 1e-6）、SwiGLU clamp 随机模糊（R24 端点
  oracle 的随机化扩展）、scale 布局公式对齐（mxfp8_linear_bytes）。
- 全量：干净 worktree `py -3.12 -m pytest tests -q` → **495 passed,
  exit 0**（含 G6 计数门禁 495）。主工作树同期 2 项 CSV 导出失败 +
  G6 差异均来自并行工作台任务的未提交在途修改（cli.py/
  test_consistency.py/static/csv.js/test_workbench_review.py），本轮
  未触碰对方文件，其落地时自行更新计数。
- replay：`tools/replay_cost_integrity_r27.py --check` exit 0（计数
  不变+锚点+归因一致）；`replay_v41_csa2_core/quant/moe/swiglu/
  qwen36_mtp/qwen36_mrope --check` 全部 exit 0；`py -3.12 -m simverse.
  cli run-v2` status=completed。

## 4. 主要变更文件

| 文件 | 变更 |
|---|---|
| simverse/cost/t0.py | dtype_route 配方入参+档位修正；price_row 行级计价元数据；cost_basis 文案函数 |
| simverse/model_ir/families.py | extras.quant 注册 gemm_compute_dtype/gemm_compute_basis |
| simverse/pipeline.py | 路由接线（target/dflash/mtp/chain 四路）；roofline 元数据；t2d float；chain 计价修复；pipeline_sim Metric 合并；_device_snapshot；A2b 假设 |
| tools/replay_v41_csa2_core.py | 归因同价复算 + 重基注记 |
| tools/replay_cost_integrity_r27.py（新） | R27 活性无漂移入口 |
| docs/evidence/cost-integrity-r27/（新） | 三场景 before/after/attribution + provenance |
| docs/evidence/csa2-core/ | after.json 重基 + rebase-r27-provenance.json |
| tests/test_cost_integrity_fixes.py（新）、tests/test_v41_quant_numeric_r27.py（新） | 26 项回归/数值 fixture |
| docs/IMPLEMENTATION_STATUS.md | R27 状态段 + 收集数 495 |

未触碰（并行任务职责）：simverse/static/、simverse/webapp.py、前端/
CLI 导出实现、ROADMAP.md、docs/gaps/capability-gaps.md、v0.1 路由与
历史模块组织。

## 5. 执行与数据缺口（独立于本轮修复验收；不以缺硬件为由停止修复）

### 5.A V4.1 CPU fixture（设计 + 本轮已运行的最小验证）

**已实现并运行**（tests/test_v41_quant_numeric_r27.py，math-oracle 层，
纯 stdlib、确定性种子、不加载权重）：量化语义数值链 e4m3/e2m1/e8m0/
MXFP4 打包/反量化 GEMM 等价 + SwiGLU clamp 随机模糊。该层直接支撑 A
 修复的"存储 8/4bit + 计算 BF16"口径（若把计算档当 FP8，将隐含激活
  量化，与 pinned 激活 BF16 语义矛盾——数值 fixture 排除该混淆）。

**完整 fixture 覆盖矩阵与分层**（严格区分：数学 oracle / 转写实现 /
官方子模块执行 / 完整 forward；不自动升级模型级 cpu_forward）：

| 模块 | 数学 oracle | 转写实现（torch 小张量） | 官方子模块执行 | 完整 forward |
|---|---|---|---|---|
| 量化（MXFP8/4/scale/打包） | ✅ 本轮（6 项）+R22 §4 | 计划：torch 复刻 dequant-GEMM vs stdlib oracle | 需 vLLM 环境+GPU（Marlin SM80+） | — |
| SwiGLU clamp | ✅ 本轮随机模糊 + R24 端点 | 计划：torch SiluAndMulWithClamp 转写对照 | clamp kernel 数值等价未执行（R24 剩余） | — |
| MoE 路由（sqrtsoftplus/noaux_tc/top6/归一×1.5） | R21 stdlib oracle（含 NaN/平手/归一反例） | 计划：torch 转写小 E/k 对照 | — | — |
| mHC（prenorm/Sinkhorn/comb） | R05 算术 oracle（精确式 vs 惯例近似分列） | 计划：Sinkhorn 迭代数值 vs 闭式投影对照（m×m 小阵） | TileLang kernel 未执行 | — |
| 注意力（CSA2 索引/top-k/candidate/稀疏核） | R20/R25 调度器重演+手算绝对值 | 计划：小尺寸 sparse index 选择 vs 暴力 O(N²) 参照 | FlashMLA SM90 kernel 需 GPU | — |
| Engram（素数桶/哈希/查表/wkv） | R04 Miller-Rabin 桶和精确相等+手算 | 计划：桶哈希→查表→注入数值链（随机 key，264B/行布局） | nvidia ops kernel 未执行 | — |

**具体后续任务**（按依赖序）：①torch 转写层（本机有 torch CPU）：量化
dequant-GEMM、Sinkhorn、Engram 哈希链三个转写 vs stdlib oracle（工作量
小，无新资产需求）；②CSA2 索引暴力参照（需把 candidate/topk 语义转写，
中等工作量）；③官方子模块执行（Marlin/FlashMLA/Engram ops）：需要
Linux+GPU+vLLM @f358f057 源码环境与编译链——**额外环境前置**，本机
不可完成；④完整 V4.1 小尺寸 forward：需要官方 checkpoint 分片或
config 一致的随机权重组装（h=极小化）+ 前向脚本，**额外资产/环境前
置**。①②落地前模型级 cpu_forward 保持 not-run。

### 5.B Engram Host 成本（核对方案；本轮不新增假设性成本模型）

现状（保留）：`engram_lookup_host` 时 host 表期望读字节不计价（不套
GPU HBM 带宽），capability `engram_host_lookup_latency`/`stage_cost_
complete` unknown，M01 标注"仅 GPU 侧解析耗时下界"。

核对方案（按序，防重复计价）：
1. **UVA 访问链**：pinned 源码（common/engram.py + nvidia ops）确认
   查表是按行 UVA 远端读（每行 264B×命中行数）还是整桶预取驻留；确
   认 cudaMemcpyAsync/预取触发点与粒度。
2. **预取与重叠**：若部署栈有异步预取（表分片/页级），识别预取带宽
   （PCIe/NVLink-C2C 视 placement）与计算重叠度——重叠部分不进关键
   路径，未重叠部分按链路模型计（`link_ms` 同式），**不得同时**按
   host 读时延与预取带宽双计。
3. **Host 内存侧**：pinned memory 容量/带宽（host 侧）与 GPU 互连带
   宽分列；表驻留 host_resident_bytes 已单列（R04），不进 GPU 峰值。
4. 若补假设性成本模型：独立标明假设与适用域（如"未重叠 UVA 远端读
   下界"），support=approximated，不冒充完整实测 TPOT；缺证据时维持
   现状（GPU 侧已建模成本的明确限制）。
前置条件：需要 pinned nvidia/ops 源码逐行核对 + （可选）真实 UVA 延
迟测量（Linux+GPU）。

### 5.C 可部署性数据采集方案（最小采集 schema；无真实环境不填数据）

**采集项**（每设备/rank 一行，全部字段必填才准入 measured）：
1. **可分配预算**：`torch.cuda.mem_get_info()` 空载差 + 框架预留后
   实测（allocatable_bytes，gross 语义——预算已扣运行时开销，不与
   extra_overhead 重复扣）；方法/环境字符串（驱动/CUDA/框架版本）。
2. **workspace 尺寸**：Marlin workspace、FlashMLA o_accum/lse_accum、
   MoE 中间缓冲——按算子记录（CUDA memory stats 峰值差，warmup 后
   稳态多次取中位）。
3. **CUDA Graph**：捕获后固定池增量（graph_pool 友好分配器差值）+
   是否启用（enforce_eager 反例）。
4. **NCCL/通信池**：NCCL_BUFFSIZE/协议内存 + 自定义通信缓冲，按
   TP/EP 配置分列。
5. **测量边界**：device kernel（CUDA_EVENT）/host launch/graph piece
   分栏（TIMING_BOUNDARIES 契约）；warmup 次数、重复数、sync 策略、
   样本统计（min/max/mean/median/std/count）。
6. **身份字段**：设备 SKU/驱动/CUDA/框架/实现版本、精度（含 scale
   布局）、TP/EP/shape 域、采集代码位置（profile 所有权，准则 §1.6）。

**口径纪律**：预算已扣开销 vs 额外开销（extra_overhead_bytes）分列，
禁双扣（R03a 契约沿用）；无真实环境**不填任何数据**（保持 unknown）；
补预算不自动解除 CPU/GPU/服务集成/部署排名门禁（G15 独立判定）。
前置条件：目标 GPU 环境（H100/H200 SXM）+ 固定软件栈版本清单。

## 6. 结论

六项确定性修复全部落地并回归（495 passed，独立 worktree）；成本变化
仅限计算档/期望字节/内部 chain 与身份基，原始计数零漂移；执行与实测
缺口（V4.1 CPU fixture 分层计划、Engram Host 核对、可部署性采集
schema）以可执行的前置条件形式登记，未以缺硬件为由停止修复，也未虚
构任何执行/实测证据。两款主模型部署排名门禁（forward 执行/GPU 实测/
预算）不变。

## 集成审查补记（2026-11-17）

两任务合并复核发现：原 `roofline_points` 的 top-40 截断使算子透视无法完整汇总时延。新增 `cost.row_prices`，从同一次 `base_prices` 为每个 target_decode 工作量行输出 row_id、t_ms、bound、AI、吞吐与成本元数据；原 `roofline_points` 仍为 top-40，历史证据与 replay 语义不变。计数、计价公式及 TPOT 均未改变。
