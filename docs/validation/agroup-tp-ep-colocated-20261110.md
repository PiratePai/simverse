# A 组验证报告：Qwen3.6-35B-A3B + DFlash TP/EP 分片与同卡账本（R01–R03b）

> **历史轮次（2026-11-10 标注）**：本报告记录当轮实现与数值；之后的两模型
> 审查修复了 shared gate、V4.1 o_a/memory 单位与草稿 B>1 口径。当前状态以
> [两模型审查记录](two-model-forward-merge-gate-20261110.md) 与合入门禁为准，
> 本报告数值不自动继承。

日期：2026-11-10。本轮交付 ROADMAP A 组五项的工程实现与验证。本文是合入
说明的证据入口；实现细节见 `simverse/lowering/parallel.py`（配方判定）与
对应测试。

## 1. 源码身份与映射链

| 环节 | 身份 | 位置 |
|---|---|---|
| 并行配方 pin | vLLM @`dc74b8cbbbe3dbb9399fe369757d63defd614cc2`（main，2026-11-10 GitHub raw 固定获取） | `assets/identity/vllm_qwen35_pin/`（16 文件逐 SHA256 入 `manifest.json`；证据 `ev-vllm-parallel-recipe`） |
| HF 模型语义 | transformers @`b6401d66`（modeling_qwen3_5_moe.py，原有 pin） | `assets/identity/hf_transformers/` |
| DFlash 协议 | z-lab/dflash @`07ebd93`（原有 pin） | `assets/identity/dflash_repo/model.py` |
| 配方实现 | `simverse/lowering/parallel.py`（ParallelPlan；count/moe、count/work、count/draft、lowering/memory、lowering/sharding、pipeline 六处消费同一 plan） | recipe_id `vllm-fusedmoe-tp-ep-v1` |

配方要点（每条都有 pin 文件:行号，见 `simverse/importers/source_map.py`
新增 3 条 `sm-vllm-*`）：

- **MoE routed**：EP 关（ep=1）→ 每 rank 全部 E 专家、inter/tp 切分
  （fused_moe/config.py:1402）；EP 开（ep=tp）→ 每 rank E//ep(+余数) 整
  专家（FusedMoEParallelConfig.make L1257-1288 + expert_map_manager L62-77）。
  ep∉{1,tp} 无实现配方，**显式拒绝**。
- **router**：GateLinear=ReplicatedLinear（每 rank 复制，不随 tp 缩放）。
- **shared expert**：全局 TP inter/tp；shared_expert_gate Replicated(h,1)。
- **attention**：heads/tp；kvh≥tp 切 kvh/tp（要求整除），kvh<tp 复制全量
  （要求 tp%kvh==0）；o_proj 行切。
- **GDN**：in_proj_qkvz((2k+2v)/tp)、in_proj_ba(2nv/tp)、conv1d(conv_dim/tp)、
  out_proj 行切；状态 temporal=nv/tp×hk×hv、conv=conv_dim/tp×(kernel−1)。
- **embed/lm_head**：词表行切 V/tp。
- **DFlash 草稿**：与 target **同 TP 组**运行，embed_tokens/lm_head 替换为
  target 同一模块对象（零额外驻留）；fc ReplicatedLinear 复制。TP1 草稿挂
  TP>1 组无服务栈配方 → 拒绝（旧 colocated 假设废除）。

## 2. 支持范围与明确拒绝

**支持**（Qwen3.6-35B-A3B+DFlash，结构约束全部满足）：

- TP ∈ {1,2,4,8,16}（heads=16、kvh=2、inter=512、shared=512、V=248320
  对上述 tp 整除/复制约束全部满足；草稿 heads=32/kvh=8 同）；
- EP ∈ {1} 或 {tp}；EP 余数分置（E%ep≠0 时低 rank +1，per-rank 账本显式
  差异）；
- draft 放置：colocated/dt_pipeline_sim（同 TP 组共享配方）与 dt_serial
  （独立设备 TP1；embed/lm_head 无共享配方 → 按"草稿全量副本"保守口径并
  如实标注，非零成本共享假设）。

**明确拒绝**（非静默替换并行模式，capability `parallel_valid=invalid` /
场景校验直接报错）：

- ep∉{1,tp}（如 tp8/ep2）：单引擎副本无实现配方；
- tp=1 且 ep>1；
- heads%tp≠0（tp5/tp3）；kvh 不满足切分或复制整除；
- EP 关时 inter%tp≠0 或 shared_inter%tp≠0；
- tp>E（vLLM qwen3_next L184 同款拒绝）；
- V%tp≠0（词表行切）；草稿词表≠target（需 draft_id_to_target_id 映射路径，
  未建模）。

## 3. 默认场景修复前后（同输入重放归因）

场景：CLI run-v2 内置默认（Qwen3.6+DFlash colocated TP8/EP1 8×H100
B1 ctx32768 α0.72 blk16，链路声明 pcie5）。前后代码各跑一次，逐项归因：

| 项 | 修复前 | 修复后 | 归因 |
|---|---|---|---|
| M01_tpot_baseline | 3.4704 ms | 112.2126 ms | ①集合通信按层计费修复：旧代码整模型只计 2 次 allreduce（detail 却声称 2×40 层），现按 40 层×2 计（80 次；pcie5 假设固定延迟 1.4ms/次 → 112ms 主导）。②per-rank 计数本身变小（行时间 0.67→0.21ms）。①是漏计修复：节点内 NVLink 规格属 G-D8 数据缺口，未虚构；用户可换 link 输入 |
| M01_tpot_analytical | 1.6874 ms | 31.6983 ms | 同上（轮内 verify 的集合通信修复主导） |
| M03_E_N | 3.5528 | 3.5528 | 接受律未动 |
| target_decode flops | 5,300,762,860 | 1,493,127,494 | 整模型口径 → per-rank（GDN 投影/状态、attention、MoE、lm_head 按 plan 切分）；复制对象（router）不缩放 |
| weight_read（/rank） | 1,273,710,080 B | 279,369,216 B | 同上；+ final_norm 补行 |
| weight_read_expected | 503,316,480 B | 62,914,560 B | E[U]×(3h·inter/tp) 分片字节（EP 关） |
| state_read/write | 62,914,560 B | 7,864,320 B | GDN 状态 per-rank（nv/tp、conv/tp），修复前整模型口径 |
| draft_block | 0.1630 ms | 0.0301 ms | 草稿与 target 同 TP 组（heads/8、MLP/8、KV 1 头/rank、lm_head V/8）；+q/k norm、rope、hidden_norm、final_norm 补行（76→96 行） |
| colocated 显存 | 5,680,423,680 B（单"rank0"，口径混乱） | steady 2,827,423,168 B + 结构化临时峰值 2,135,746 B = known_peak 2,829,558,914 B（per-rank r0…r7 + 最坏 rank 报告） | ①废除草稿幻影 embed/lm_head 全量副本（≈1.017GB/轮错误驻留）与失配去重键双重计费；②共享对象改为零字节引用+consumer 登记；③embed/lm_head 词表行切 V/tp；④GDN 权重/状态 per-rank；⑤草稿 KV 8 头切分（修复前 TP1 全量）；⑥attention 投影权重字节补乘输入宽 h（旧式漏乘 h 倍）；⑦+生命周期分列（常驻/每请求/跨轮/阶段临时/未知运行时组件） |

修复后默认场景 memory 状态仍为 **unknown**，原因分列：可分配预算未知
（allocatable=None，G-D7）+ 峰值不完整（workspace/CUDA graph/NCCL 池无
尺寸证据，作为缺口组件传播，不填 0）。已知下界 2.83GB < 80GB 标称，不
构成 OOM 结论。

## 4. 测试与验收

- 新增 `tests/test_parallel_sharding.py`（19 项）：非法/合法组合、EP 余数
  分置、MoE 两配方独立手算 oracle、TP 切分对象 Σranks==整模型守恒、EP
  routed Σranks 守恒、复制对象不求和、KV 切分/复制规则、GDN 状态 per-rank、
  按层通信配方（80 allreduce ↔ 40+40+40）、plan 一致性、场景身份（EP/budget
  入 digest）、端到端 EP 拒绝、UI 静态接线、webapp 场景 JSON 等值。
- 新增 `tests/test_colocated_memory.py`（11 项）：共享对象单一物理身份
  （consumers=[target,draft]、引用零字节）、无幻影草稿头权重、整 rank 稳态
  字节独立手算全量对账、EP 余数 rank 差异、草稿批=target 批（修复硬编码
  1）、生命周期类别、峰值取阶段 max 而非求和、未知组件阻断 feasible、同
  身份重复声明只计一次、pipeline 报告逐 rank+最坏、EP 整除时均匀。
- 新增 `tests/test_budget_feasibility.py`（14 项）：预算输入契约负例
  （kind/来源/范围）、JSON 往返与场景身份、超标称预算拒绝、四态判定
  （unknown/oom-from-nominal/feasible/conditional）、峰值不完整传播、
  开销单次扣除（gross 语义）、假设预算不进实际部署排名（compare 资格）、
  端到端 OOM（B×ctx 巨大使已知 KV 下界>80GB）。
- 既有测试更新（逐项归因，非改期望保绿）：tiny 手算 oracle 补 final_norm
  项；TP 非法消息断言随检查项更名；235B 组合断言随 per-rank MoE 口径修正
  （fallback 38.30→4.85ms，旧断言编码的是整模型高估）；235B manifest defer
  理由从 V04 前置改为 G-D2 数据准入。
- 全量 `py -3.12 -m pytest tests -q`：**176 passed**（G6 测试数与本文件
  同步后）。G7 重放不变性（校准开关不改原始计数）在 176 内通过。

## 5. UI 验证（浏览器实测，2026-11-10）

工作台新增 EP 选择（关/=TP）与预算输入（hypothesis GB）。实测
（`py -3.12 -m simverse.cli serve` → 浏览器交互）：

1. EP 下拉与预算输入框渲染正常（combobox/spinbutton 出现在场景栏）；
2. 选择 EP=开 + 预算 70GB 后运行：run_id `sha256:0d85e95a…` completed；
3. EP 生效证据：T_round=64.603ms（EP8 all2all 配方），与 EP 关的
   112.618ms（80×allreduce）可区分——配方差异在 UI 数字上可见；
4. 预算生效证据：M02 不排名原因显示"峰值不完整：workspace/graph/NCCL…"
   而非"可分配预算未知"——hypothesis 预算已进链路且缺口传播正确；
5. 截图存证 `docs/evidence/workbench_ep_budget_run.png`。

静态接线测试（workbench.js/html EP+预算接线）与 webapp 场景 JSON 等值
测试在 §4 测试清单内。

## 6. 已知限制与未做

1. **未做任何硬件测量**：全部结论为 source-checked + 独立数学 oracle；
   H200 校准闭环数字不受本轮影响（G7 重放相等）。
2. **workspace/CUDA graph/NCCL 池尺寸未 pin**：作为 unknown_components
   传播（峰值不完整 → feasible 判定 unknown），未填 0。
3. **节点内 GPU NVLink 规格缺**（G-D8）：TP 集合通信按场景 link 字段计
   （默认 pcie5 假设延迟主导）；未虚构 NVLink 常数。
4. **dt_serial 草稿独立组的 embed/lm_head**：服务栈共享配方未见（vLLM 为
   同组共享），按草稿全量副本保守计并标注，非"零成本共享"假设。
5. **235B moe.csv TP8/EP1 数据准入**：账本口径已对齐（本报告 §1），蒸馏
   +留出验收属 G-D2 独立验收，本轮未执行。
6. **EPLB 冗余专家、量化 Marlin 分支**：未建模（source_map uncovered
   登记）。
