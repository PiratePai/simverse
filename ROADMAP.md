# Simverse Roadmap

更新：2026-11-10。产品范围固定为 **Qwen3.6-35B-A3B + DFlash** 与 **DeepSeek-V4.1-Flash**。历史模型仅用于内部回归与校准，不继续扩展产品目录。以下为待办，不代表已交付或交付日期承诺。

## 当前已交付

- [x] 重点模型产品目录、DFlash 配对门禁、具体资格失败原因。
- [x] ModelIR、计数/成本/执行分层，CLI/Web/导出共用结果。
- [x] Qwen3.6+DFlash 解析实验、DT 串行及初步微批流水。
- [x] V4.1 部分结构及 mHC 边界计数；Engram 单副本路径已 source-checked 建模
  （R04，2026-11-11；DP 分片交换/cpu_offload 运行时行为未覆盖，见 B 组表）；
  CSA2 索引打分/top-k/candidate block 已 source-checked 建模
  （R20，2026-11-12；注意力核本体/部署变体仍缺，见 B 组表）。
- [x] 历史 H200 部分算子校准、来源与覆盖域门禁；不代表重点模型实测。
- [x] A 组（2026-11-10）：TP/EP per-rank 统一配方、同卡对象身份账本、
  生命周期峰值分列、预算契约与可部署性四态判定链路（见 A 组表；真实
  预算与运行时池证据仍缺，默认场景保持 unknown）。同日二轮审计补漏
  （lm_head logits 通信/草稿二次读/fused 副本，见
  [二轮报告](docs/validation/agroup-lmhead-ctxkv-audit-20261110.md)）。

## 组织方式：独立交付，必要处联合验收

P0/P1/P2 表示优先级，不是要求整阶段完成后才能启动下一阶段。下面各组可独立推进；表内只列真实的硬依赖。可以先完成接口、独立 oracle、数据审计或局部模型，依赖未完成时不得宣称联合能力已通过。

保留 R01–R14 编号便于追踪；拆分项使用后缀，新增设备事项使用 R15–R19。一个子项通过即可单独合入，不能因同组其他项缺数据而阻塞。设备与模型分别维护能力矩阵，增加设备不需要增加产品模型。

## A 组：Qwen 分片与同卡可部署性（P0，最终判定耦合）

状态更新 2026-11-10：五项工程实现全部落地（R01–R03b），详见
[A 组验证报告](docs/validation/agroup-tp-ep-colocated-20261110.md)。
**联合验收只解除到"判定链路完整"**：默认场景同卡可部署性仍为 unknown
（可分配预算未提供 G-D7 + workspace/graph/NCCL 池无尺寸证据），不得
表述为已确认可部署或已确认 OOM。

| ID | 状态 | 交付与证据 |
|---|---|---|
| R01 | ✅ 完成 | per-rank TP/EP 统一配方（vLLM @dc74b8cb pin，`simverse/lowering/parallel.py` ParallelPlan；count/memory/lowering 三处共用）；跨 rank 守恒 + 独立手算 oracle + 非法组合拒绝（tests/test_parallel_sharding.py）；默认场景前后差异逐项归因（验证报告 §3）。**二轮审计（同日）**：补 lm_head logits_gather 集合通信（logits_processor.py 入 pin；decode/verify/草稿各 1 次）与草稿 lm_head/ctx KV 二次读流量（[二轮报告](docs/validation/agroup-lmhead-ctxkv-audit-20261110.md)） |
| R02a | ✅ 完成 | 同卡对象身份：共享 embed/lm_head 单一物理对象 + consumer 引用（零字节），废除草稿幻影副本与失配去重键；草稿批=target 批；per-rank 账本含 EP 余数差异与最坏 rank（tests/test_colocated_memory.py）。**二轮审计**：vLLM `_fused_kv_weight` 全层 K/V 行 cat 驻留副本入账。**三轮修正**：改为显式 `draft_ctx_kv_path`——仅 raw 建融合副本；MXFP8 独立对象；4bit 回退无副本但 k_norm 堆叠仍按草稿 dtype（bfloat16→1,536 B）计 |
| R02b | ✅ 完成（结构化部分） | 生命周期分列：常驻/每请求/跨轮/阶段临时（draft/verify 串行取 max）；workspace/CUDA graph/NCCL 池作为 unknown_components 缺口传播（不填 0，峰值不完整 → 判定 unknown）。**二轮审计**：补 ctx 归一副本/融合 ctx-KV 输出（phase:draft）与 verify aux hidden（phase:verify）三件 pinned 形状临时。**限制**：临时缓冲只含 pinned 形状可推导项；运行时池尺寸无证据，完整运行时峰值未验证 |
| R03a | ✅ 完成 | 预算输入契约（nominal/allocatable(kind: measured/hypothesis)/额外开销分列、gross 语义防重复扣除、缺省 unknown 不套标称）；场景身份包含 EP/budget；CLI/API/UI/导出一致（测试 + 浏览器实测） |
| R03b | ✅ 完成（判定链路） | 四态判定 unknown/oom/feasible/feasible-conditional + 已知下界超标称即 OOM；假设预算仅条件结论且不进实际部署排名（compare 资格排除）；负例全覆盖（tests/test_budget_feasibility.py）。**实际可部署结论仍阻塞于真实预算证据（G-D7）与峰值完整性** |

| ID | 硬依赖／联合验收点 |
|---|---|
| R01 | 固定实现源码；无其他 roadmap 前置项 |
| R02a | 对象测试可独立；分片后的容量验收接 R01 |
| R02b | 对象接口来自 R02a；完整 per-rank 峰值接 R01 |
| R03a | 可独立，不必等容量模型或实测数据 |
| R03b | R01 + R02a/b + R03a；实际预算证据。假设预算仅产生条件结论 |

本组必须联合验收才能解除“同卡可部署性未知”；不能仅凭 R03a 输入一个预算就标记为真实可部署。当前判定链路已贯通：给定 measured 预算 + 完整峰值即可给出 feasible/OOM；两缺口（G-D7 预算、运行时池尺寸证据）解除前默认场景保持 unknown。

## B 组：V4.1 结构补齐（P0，分模块独立）

| ID | 独立交付物与验收标准 | 硬依赖／联合验收点 |
|---|---|---|
| R04 | ✅ 完成（单副本路径，2026-11-11）：common/engram.py pin（vllm @f358f057，blob sha1 对 vllm-tree.json 校验）→ 素数桶表布局/批级哈希/查表/wkv/门控四行入 count、表/wkv/qk/staged 对象入 memory（表默认 cpu_offload 研究假设，显式 engram_table_placement 输入）、TP 通信行；独立 oracle tests/test_v41_engram.py 24 项（层桶和 == engram_num_embeddings 精确相等；PR #3 合入后复核修正 rank 局部查表与固定 BF16 字节）。证据 ev-vllm-v41-engram；[报告](docs/validation/v41-engram-20261111.md)。**剩余（未完）**：DP 分片交换（all-to-all/peer/symmetric memory）与 EngramBatch MXFP8 路径、cpu_offload 运行时行为、amd 变体；host 查表时延 unknown，M01 仅 GPU 侧下界 | Engram 源码（已 pin）；不依赖 Qwen 或设备扩展 |
| R05 | **已完成 2026-11-12**：14 kernel 文件 blob/SHA256 pin；prenorm/comb post/Sinkhorn 1400 op 修正、首层特例与出口两行、broadcast 派生容量；精确算术与 norm 惯例近似分列，独立 oracle + 前后 replay。[报告](docs/validation/v41-mhc-kernels-20261112.md) | source-checked；CPU/GPU 未执行，split/部署变体与物理流量仍缺 |
| R20 | **已完成 2026-11-12**：CSA2 索引打分/top-k/candidate block——csa2-pin.json 20 文件 pin（blob sha1 对 vllm-tree.json）；index_select 拆分 score/topk、candidate select/mask 两级过滤（源 L20 发布 top-2048×8 块、消费层 24/28/32/36 mask）、短上下文旁路（fill 0 flops）、compressor 池化/环形 state、索引 K 132B/行、打包记录 584/528 B/token（显式 v41_kv_record）、bf16 固定权重、共享缓冲下界、sparse_index_build 运行首层一次；独立 oracle 20 项（tests/test_v41_csa2.py）+ 前后 replay 逐 op 归因。证据 ev-vllm-v41-csa2；[报告](docs/validation/v41-csa2-indexer-20261112.md)。**剩余**：注意力核本体（4d+5 移植口径）、SM100 sparse-logits/decode_topk CuTe/行分片等部署变体、CUDA-graph 静态整宽 logits 尾写、MoE scoring/量化映射/MTP 尾部不变 | source-checked；kernel/CPU forward/GPU 未执行；structure_complete 仍 False，部署排名资格不因本模块解除 |

各模块分别解除自己的缺口；整个模型的结构资格需全部相关缺口闭环，不能以一个模块通过代表整体通过。

## C 组：更多设备支持（P1，各型号与层级独立交付）

当前注册表已有 H100 SXM、H200 SXM、L40；这仅表示已有规格/解析入口，并非所有模型、精度和部署方式均已验证。

候选接入批次如下；这是待核对的目标清单，不是已支持声明。每个具体 SKU、显存版本、PCIe/SXM/整机形态独立登记，依据官方资料与可获得证据决定准入，不复制近似型号常数。

| 批次 | 设备候选 | 范围 |
|---|---|---|
| C1 | A100、A800、H800、L40S | 优先审计旧仓已有资产，逐项核对规格与来源后复用 |
| C2 | B200、B300、GB200/GB300 系统 | 单芯片、单卡、节点/机柜口径分离；系统互联不折算成单卡常数 |
| C3 | 昇腾 910B/910C、950DT、950PR | 规格档先行，逐型号核对官方身份；后端算子、精度与通信配方按昇腾实际实现独立验证 |

同一批次无需等待全部设备；C2/C3 的资料审计也无需等待 C1。缺少可核验规格的型号保持候选状态。

| ID | 独立交付物与验收标准 | 硬依赖／联合验收点 |
|---|---|---|
| R15 | 可扩展设备契约与能力矩阵：SKU、来源、dense/sparse、精度、标称容量、带宽单位；不支持精度明确拒绝 | 可独立；现有设备回归不得无解释变化 |
| R16 | 按设备提交规格快照与解析档：一设备一来源清单、一组手算/边界测试 | R15 接口；官方规格。无实测不阻塞解析档，但保持 unmeasured |
| R17 | 拓扑与通信：PCIe、NVLink/NVSwitch、RDMA及对应厂商互联分别登记；方向、跨节点、collective 配方和假设明确 | 拓扑契约可独立；设备绑定接 R16，不能把 C2C 泛用为 GPU 间 NVLink |
| R18 | 按设备/后端/精度接入算子及通信 profile：来源、留出误差、覆盖域和回退 | R16 的设备身份；通信 profile 还需 R17；沿用校准契约，不等待所有重点模型完成 |
| R19 | 异构 target/draft 部署组合：独立设备、TP、链路与容量选择；兼容性门禁 | 对应 R16 + R17；可部署性接 A 组；实测性能声明需 R18 与匹配验证 |

设备能力至少分为“规格登记 / 解析可用 / 局部校准 / 匹配场景 E2E 验证”，逐层显示证据与限制。不得把“设备出现在下拉框”当作全部支持；无实测可分配预算仍保持 unknown。R19 不自动认可跨后端 DFlash 服务集成。

## D 组：验证与数据（P1，可先行准备，结论按场景验收）

| ID | 独立交付物与验收标准 | 硬依赖／联合验收点 |
|---|---|---|
| R06 | 按模型/模块拆分的小尺寸 forward 对照；独立 oracle；区分源码阅读与 CPU 执行。**部分完成（2026-11-12，Qwen3.6）**：GDN+MoE+full-attn fixture 的几何/算量对账（shape-checked）及独立张量数值 oracle（numeric-checked）已做；prefill/缓存 decode 各 12 张量逐元素核对，最大差 2.98e−7；修正 attn_output_gate 漏计；tests/test_qwen36_forward_r06.py 12 项；[报告](docs/validation/qwen36-gdn-moe-cpu-fixture-r06.md)。**未做**：mrope 位置生成、V4.1 侧（MLA/CSA2/Engram 执行对照）、其他模块、真实 checkpoint、模型级完整 forward | 对应模块源码；现有完整模块可立即验证，不等待 R04/R05 全部完成 |
| R07a | DFlash 服务栈配对、协议与计时边界审计 | 固定服务栈源码；独立于新设备接入 |
| R07b | 接受分布、draft/verify trace 的采集 schema、导入器与真实样本验收 | schema/导入器可先做；真实数据准入需 R07a 与匹配环境 |
| R08 | 重点模型 GDN/MoE/attention 按算子/设备/精度/TP 扩展覆盖，独立留出与域外回退 | 对应算子语义及设备身份；MoE 分片域依赖 R01，无需等待其他算子 |
| R09 | 按固定场景做 E2E 对照，披露误差与未校准部分 | 匹配实测记录与已实现的场景语义；无需等待全部算子校准，残差可反向指导 R08 |
| R10a | 修复测试子进程编码警告 | 独立维护项，可立即完成 |
| R10b | 工作台重点场景交互、错误提示、窄屏、导出一致性验证 | 现有功能可立即验证；新功能各自追加，不等待其他组完成 |

R07b/R08/R09 的真实数据验收需要额外数据，不以历史 H200 数据或合成数据替代。采集字段与阻塞详见 [数据缺口](docs/gaps/data-gaps.md)。本 roadmap 不授权启动真实模型/GPU 采集。

## E 组：请求级系统实验（P2，按运行时依赖分组）

| ID | 独立交付物与验收标准 | 硬依赖／联合验收点 |
|---|---|---|
| R11 | lookahead、取消、回滚、浪费与资源账；人工轨迹可独立验证 | 已有事件核、方法状态协议；真实资源/性能结论再接对应容量与成本模型 |
| R12a | 请求到达、动态组批、调度事件与确定性重放 | 已有事件核；可用人工负载独立验收 |
| R12b | KV 分页/回收/抢占、生命周期与状态守恒 | R12a 请求生命周期与容量对象接口；不依赖 V4.1 完整语义 |
| R13a | Chunked Prefill：chunk 预算、调度、请求完成口径 | R12a + prefill 成本/状态接口 |
| R13b | PD 分离：KV 交接、传输依赖、两端资源账 | R12a/b + R17 拓扑/传输接口；不依赖 R13a/c |
| R13c | Prefix Cache：命中、引用计数、驱逐与容量守恒 | R12b；不依赖 PD 或 Chunked Prefill |
| R14a | TTFT/ITL/吞吐/goodput 与 SLO 的人工轨迹 oracle | 指标契约，可独立先行，不等待服务特性全部实现 |
| R14b | 请求级 E2E 联合验收 | R14a + 被测场景实际使用的 R11–R13 子项 + 匹配实测轨迹 |

不存在“全部 P0 完成后才允许写系统模块”的总门槛；但未完成模型语义、容量或成本验证时，系统实验只能声明其已覆盖的范围。不得将流水周期当作单请求延迟。

## 合入要求与状态入口

每项完成必须附源码版本/哈希、影响域、测试命令和结果、数值变化、剩余限制。校准不改变原始计数；缺失证据继续传播 unknown/unsupported，不为参与排名补造数值。

- [强制合入准则](docs/MERGE_CRITERIA.md)
- [功能状态与限制](docs/FOCUS_STATUS.md)
- [实施状态](docs/IMPLEMENTATION_STATUS.md)
- [工程缺口](docs/gaps/capability-gaps.md)

优先级以本 roadmap 为准：旧缺口文档中“继续扩展 235B”等建议不再是产品优先事项。更新完成状态时同步这些入口，避免把“已有测试”误写成“已完成部署验证”。
