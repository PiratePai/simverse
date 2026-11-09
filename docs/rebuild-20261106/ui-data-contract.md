# UI 页面、后端数据与验证契约

状态：2026-11-06 规划增补，尚未实现生产接口。本文是 [重构计划](plan.md) 第 8 节的实施附件。界面基准为 [复用工作台](prototype/index.html)，元素来源见 [复用对照](prototype/reuse-map.md)。待用户选择只登记在 [questions.md](questions.md)，本文的默认值不表示用户已经确认。

## 1. 保留界面，逐项替换展示数据

正式 UI 保留原 HTML 的场景栏、Target/Draft 配方、七项摘要、部署方法矩阵、算子图、七维检查器、Roofline、三轨时间轴。保留这些位置不代表首发支持所有方法、部署或平台。能力目录决定选项、禁用原因和支持范围，不能从硬编码选项推断能力。

原 HTML 的原文审查、重构计划、计算原理也保留为资料导航；正式内容连接本次修订文档和对应证据。`reference-v3.html` 是只读历史材料，不与当前结论混用。`teaching.html` 是独立教学入口。两种示例数据都不得成为生产接口失败时的回退数据。

“真实数据”指后端实际生成且可追溯的产物，不等同于真机实测。结构、逻辑计数、解析估算、事件仿真和 measurement 分开标记；一条结果可以有结构证据，但没有性能测量。沿用 [证据复核](evidence-audit.md) 已记录的 Vidur、vLLM、DeepSpec 固定版本参考：借鉴成本插件/事件核分离、状态正确性和方法语义，不引入它们的整套界面或依赖。本次只补设计契约，没有重新运行这些项目。

## 2. 页面与工作台区域

P02–P05 是主工作台的详情页/抽屉和后续导航，不要求 M1 同时做成独立页面。P06–P08 首发可用版本化 Markdown 页面。所有页面通过 run_id、稳定对象 ID 和筛选条件关联，切页不重新计算业务结果。

| ID / 页面 | 保留或新增的信息 | 后端必须提供的产物与责任模块 | 验证与开放阶段 |
|---|---|---|---|
| P01 实验工作台 | 原版全部区域；配置、运行状态、比较、图、检查器、时间线 | Catalog、Scenario、CapabilityReport、ResultBundle、Comparison；api 汇总各模块产物 | V01–V10、V12；M1 结构/计数，M2 串行成本，M3 流水，M4 服务指标 |
| P02 模型与执行配方详情 | Target/Draft 模型身份、逐层类型、权重共享、缓存关系、逻辑算子到 kernel 的映射、配置覆盖 | importers 的 ModelIR/FieldCoverage/ForwardEvidence；methods 的 MethodPlan；lowering 的 Recipe/LoweredGraph | V02–V05；M1 起，未支持分支可见，不输出受其污染的完整性能 |
| P03 部署、显存与通信详情 | 设备组、TP/EP/PP、rank 放置、链路；每 rank 内存栈、峰值时刻、共享对象、消息 | DeploymentSnapshot、AllocationLedger、MemoryTimeline、MessageLedger、ResourceGraph；lowering/count/execute 负责 | V04–V07；M2 串行，M3 资源竞争；尚无生命周期时只显示静态部分与 unknown |
| P04 方案比较与参数扫描 | 工作台矩阵展开；固定基线、扫描维度、敏感度、收益反转点、资源代价 | experiments 的 Comparison/SweepManifest、每格独立 run_id、配对条件、有效性与排除原因 | V08–V10、V12；M2，分位数与 SLO goodput 到 M4 |
| P05 证据与校准 | 字段来源、公式、独立锚点、forward 状态、原始 profile、留出误差、外推提示 | EvidenceRegistry、ValidationReport、CalibrationBundle；validation/cost 负责 | V01–V11；M0 起可看证据，M4 有测量才开放校准结论 |
| P06 原文审查与差异 | E01–E21、两旧仓差异、DSH/ZCode 问题与裁决、未解决项 | 版本化 ReviewArtifact；MigrationDiff 引用旧/新 run 与独立 oracle，报告由产物生成 | V01、V09、V12；M0 文档，M1/M2 增加结构和账本对账 |
| P07 重构计划与实施状态 | 里程碑、退出条件、阻塞、交付链接、待决策入口 | 版本化计划与 AcceptanceManifest；CI/run 产物链接；无运行证据不自动显示已完成 | V01、V12；M0 文档起步，不为计划单独建设项目管理系统 |
| P08 计算原理与公式 | 本次 principles、适用条件、单位、公式版本、微型算例、对应指标 | FormulaRegistry + 版本化文档 + 独立 OracleFixture；公式引用回到具体字段 | V01、V03、V12；M1；公式重放不替代独立数学验证 |

### P01 每个区域的真实输入

| 原版控件/区域 | 必须接入的数据 | 不允许保留的隐式假设 |
|---|---|---|
| Target 与 Draft | model_id/revision/config_hash/IR_hash/tokenizer、角色、权重来源、结构覆盖；自定义草稿的完整结构 | 总参数或模型名称代替层图；所有草稿共享一个固定 d/L |
| SpecMethod | method_id/version、程序图、K 上限/动态规则、块长、迭代、采样、hidden 依赖和缓存政策 | 把 DSpark、MTP、DFlash 仅转换为 S/M/K；把链式代理当真实 DSpark |
| Deployment | rank/device 角色映射、TP/EP/PP、节点/链路、资源共享、协议、支持性与约束 | TP8+TP1 标签本身证明跨服务运行成立；只除参数不计 collective |
| ISL / OSL | 独立数值或分布、batch/concurrency、初始上下文、到达过程、EOS/输出上限、seed | 原下拉框只保存 ISL，却把 OSL 当同一常量；上下文永远固定 |
| 接受率/τ | 分布类型、条件 α_i 或 A/K 直方图/trace、样本与适用负载、截断规则、来源 | α、平均接受数和 τ 可无条件互换；平均值能恢复分布 |
| Features | graph capture、attention 实现、KV 布局、prefix-cache、调度及 backend 的独立配置与能力检查 | 一组选项统一乘一个加速因子；Prefix Cache 一律按 50% 缩减全部成本 |
| 配方标签 | 与当前 run 绑定的角色/设备/recipe 摘要与缺口 | 切换模型后仍沿用旧标签；显示输入草稿却使用旧计算结果 |
| 七项摘要 | 下节 M01–M07，含具体口径与证据 | 未测显示实测；无数据填零；性能界加入赢家排序 |
| 部署 × 方法矩阵 | 每格独立 RunSummary、baseline_run_id、comparison eligibility、总设备数、null reason | 前端套系数生成所有格；未运行格用相邻格插值；颜色代表已证实更优 |
| 算子图与检查器 | layer/logical_op/kernel/allocation/event 映射、依赖边、实际分支与 M08–M14 | 用列表顺序推依赖；融合后仍按逻辑节点重复计时/占用 |
| Roofline | M15，硬件/精度/shape 适用域、字节层次和样本来源 | 网络通信点使用 HBM 屋顶；所有 dtype 共用 Tensor Core 峰值 |
| 三轨时间线 | M16 的事件、请求、轮次、资源、依赖、状态与消息；串行预算单独类型 | 用阶段占比生成并行条；不同微批的周期当单请求延迟 |
| 状态、过滤与导出 | 输入 digest、run 状态/版本、selection ID、过滤范围、导出 manifest | 修改输入自动改旧指标；筛选隐藏部分后悄悄重算全局 KPI |

## 3. 公共协议和数据归属

Q10 已于 2026-11-09 确认首版按单个活跃实验任务、每次扫描几十个方案设计，保留历史、取消/恢复及大图/事件分页；等待时间与具体限额待真实产物测量。此任务数约束不限制仿真场景中的请求并发或微批。V10 验证任务门控、重复提交、扫描点恢复幂等和取消后清理/后续任务隔离；多任务并行后置。

Q6 已于 2026-11-09 确认本地单用户工作台，通过 JSON/CSV/trace/HTML 文件导出分享。首版后端与存储在本地，远程访问、账号/云服务及多人协作后置。导出由同一 ResultBundle 派生，自包含范围与外部引用明确，静态 HTML 保留结果身份、假设和能力限制，不自动上传。V12 检查 fresh run 的四类导出一致性与离线说明。

不要求首版数据库或微服务。下列是逻辑 DTO/文件契约，可先由本地 Python API 与 CLI 共同生成；HTTP 路由只是建议。

| 产物 | 关键字段 | 生产者 / 建议读取方式 |
|---|---|---|
| Catalog | models/methods/devices/recipes 的固定版本、参数 schema、能力约束 | importers/methods/lowering；`GET /catalog` |
| Scenario / CapabilityReport | 模型、方法、负载、接受分布、placement、dtype、特性、seed；逐条件 status/reason/affected_metrics | contracts + 各模块校验；`POST /scenarios/validate`，不触发真实模型测量 |
| RunManifest | schema_version/run_id/full input digest/code revision/资产 hash/父 run/创建时间；queued/running/completed/failed/cancelled、进度、错误 | experiments；`POST /runs` 创建仿真，`GET /runs/{id}`；真实硬件 profile 是独立动作 |
| ResultBundle | scenario_snapshot、ModelIR/MethodPlan/LoweredGraph 引用、work/traffic/memory/cost/execution、metrics、capability、证据与公式字典 | 引用模块产物，api 组装；按 run_id 固定不可变快照 |
| Comparison | baseline_run_id、candidate_run_ids、metric_id、资源预算、比较资格、差异/比率、排除原因 | experiments；`POST /comparisons`，不修改原 run |
| Evidence / Validation | evidence_id/type/source commit/path/hash、验证命令/环境/结果/时间/覆盖对象、未执行原因 | validation；`GET /evidence/{id}`，源码阅读与实际执行分开 |
| ExportBundle | manifest、JSON/CSV/trace/HTML、单位/精度/过滤/基线、相关证据引用、各文件 hash | 同一个 ResultBundle 派生；分页视图导出必须声明 filtered 或 full |

每个指标至少为 `Metric{value, unit, scope, statistic, result_kind, support, feasibility, evidence_refs, formula_ref, input_refs, assumption_refs, reason}`。这些字段允许引用去重字典。scope 含 phase、role、rank/group、request/round、时间窗和聚合方式；statistic 区分 exact/expectation/sample/quantile/bound。unsupported/unknown 用 `value:null + reason`，已确定不存在的量才是 0。`not-applicable` 与“缺数据”分开，例如 baseline 的 draft 字段不适用。大整数计数需约定无损序列化，避免浏览器 Number 对超过 2^53−1 的整数丢精度；传输字节/秒与显示 GiB/ms 的转换保留单位标记。

输入、资产和执行版本共同构成计算身份；时间戳不代替版本。验证状态绑定同一 hash，改动后旧证据不能自动继承。派生指标保存依赖对象，缺失语义沿受影响路径传递，不必把已独立验证的无关字段一并清空。失败或取消的 run 可以显示已完成结构，但不能冒充完整结果。

前端只做格式化、单位/坐标转换、筛选、ID 联动；正式比较比率和资格由后端提供，确保 CLI/CSV/Web 同口径。缓存键必须包含完整运行身份；切换场景后旧结果标为 stale，晚到的旧响应不能覆盖新选择。大图/trace 通过游标分页与后端聚合读取，分页不会改变全局汇总。

## 4. 指标目录：数据、口径、验证

M01–M07 对应现有七项摘要，M08–M14 完整定义七维检查器（原示例部分字段折叠/缺省），M15–M23 对应展开页、图和报告。测试编号见第 5 节。

| ID / 指标 | 后端真实输入与口径 | 验证、阶段与缺数据行为 |
|---|---|---|
| M01 TPOT / token 间隔 | 分列：解析平稳 `E[T_round]/E[N]`；请求平均间隔 `(last_token-first_token)/(output_tokens-1)`；逐 token emission 时间。绑定是否含排队/prefill、输出数与时间窗；N≤1 的请求平均间隔不适用 | V05/V07/V08；M2 解析、M3 请求轨迹；无事件不能伪造逐 token 延迟 |
| M02 加速比与吞吐 | 同口径 latency_baseline/latency_candidate；吞吐单列为窗口内提交输出 token/窗口墙钟。记录 offered load、完成/未完成请求处理、设备增量、baseline ID。SLO goodput 单列：在明确观察窗口内满足给定 TTFT/间隔约束的完成请求数或其输出 token 数除墙钟，必须说明请求/秒或 token/秒及失败/取消处理；只有满足无空闲等条件的串行单请求模型才可标注吞吐=1/TPOT | V08/V09；M2 有资格的解析对比，M4 服务吞吐；分母 0/null、OOM、资源不可比等禁排名 |
| M03 每轮产出及 S/M/K | MethodPlan/实际轮次的 forward 次数、每次 M、提议 K、接受 A、提交 N、截断原因；条件接受分布推期望，事件返回整数；长期均值说明是否包含尾轮 | V03/V05/V08；M2；mean-only 不生成完整接受分布 |
| M04 单轮耗时与分解 | 相同 round_id 的 propose/catch-up/消息/verify/commit-rewind 时间、等待和同步；串行无重叠才可相加。并行用因果完成时刻差/关键路径，资源忙时另列 | V06/V07/V08；M2 串行、M3 并行；无合法执行图只显示分项预算 |
| M05 MoE 权重读放大 | 每层 token-expert assignments、U、shared/routed 字节、分片/复制/重读；相对同层 M=1 同 recipe 的字节比。跨层摘要用总字节比并标明基准，不平均各层倍数 | V03/V04；M1 逻辑/期望，M2 物理 recipe；只有均匀独立假设时使用对应并集公式，不把 U/k 一律称物理读放大 |
| M06 每卡显存与可行性 | 每 rank 同时存活的 weight/KV/recurrent-state/activation/workspace/通信/预留，容量上限、峰值时刻、余量、shared identity；总峰值不能简单加各对象独立峰值 | V04/V06；M2；缺草稿或运行时内存则 unknown，单个已知下界超过上限可报告 OOM 并说明依据 |
| M07 来源与验证状态 | run manifest、各证据类型、support/feasibility/result_kind、失败/跳过验证、缺口及影响；按字段/模型/方法统计有明确分母的覆盖 | V01/V02/V12；M0；不压成统一可信度百分比，结构验证不升级性能证据 |
| M08 算子/核时延 | cost 输出、compute/memory/launch/sync 各项，kernel 数与聚合粒度，解析/拟合/实测身份；排队时间与服务时间分开 | V04/V07/V11；M2 T0、M4 实测；融合核不能再叠加其全部逻辑算子时延 |
| M09 FLOPs 与操作量 | 张量形状、分支、有效 token/routing，逻辑/执行 FLOPs、exp/rsqrt 等操作分类、padding 与计数约定 | V02/V03/V04；M1；整数精确对账，未知分支不填零 |
| M10 权重/缓存容量 | allocation_id、对象所属/共享、dtype/scale/alignment、shard/replica、活跃范围；检查器显示选中对象占用，不声称是整卡峰值 | V04/V06；M1 静态、M2 生命周期；缺布局只显示逻辑元素量 |
| M11 五类访存与其他流量 | weight_read、activation_read/write、KV_read/write 五类分量；state/index/metadata/workspace 另列不遗漏；logical/expected/physical 层次、memory tier、每步/每请求范围明确 | V03/V04/V11；M1 逻辑，M2 物理估算；网络 bytes 单列，不能混入 HBM 分母 |
| M12 张量形状与精度 | input/output shape、动态维语义、布局、storage/compute/accumulator dtype、量化 scale、kernel recipe 和 rank；原七维的精度单独可见 | V02/V04；M1/M2；无实际 shape 证据仅标 source-checked |
| M13 瓶颈与资源归属 | backend 返回瓶颈判据、compute/memory/launch/network/queue 分项与敏感度、主导资源；等量附近允许 mixed/unknown | V07/V11；M2；Roofline 推断只叫“模型预测瓶颈”，不能冒充 profiler 结论 |
| M14 公式、输入与来源 | 每个量独立 formula_id/version、符号式、参数/单位、input_refs、assumption_refs、source 和 oracle 链接 | V01/V03/V12；M1；不能用一个 FLOPs 公式解释容量/时间所有字段 |
| M15 Roofline | 同一 kernel/聚合范围的 F、指定 memory-tier Q、对应 compute peak/BW、dtype、设备与 profile；AI=F/Q，纵坐标 F/t 说明估算/实测；Q=0/F=0 不硬画对数坐标 | V03/V04/V11/V12；M2 解析、M4 profile；无物理流量只显示逻辑 AI，不称 HBM Roofline |
| M16 时间轴与依赖 | event_id、request/batch/cohort/round、start/end、resource/rank、dependency_ids、logical_op/kernel/message、state_before/after、取消/浪费、trace_kind | V05–V08/V12；M2 串行事件、M3 并发；预算条与事件轨迹分型，baseline 不生成 Draft/Verify 假事件 |
| M17 TTFT / E2E / 分位数 | arrival/admit/prefill/first/last emission、输出 token 数、完成状态、样本数、聚合窗口与 estimator；TTFT=first-arrival，E2E=last-arrival；simulated 与 measured 独立 | V07/V08/V11；M4；未建 arrival/queue 时不称含排队 TTFT；少样本分位数标不足，取消请求单列 |
| M18 周期、利用率与浪费 | cohort 完成时刻、warmup/drain、资源 capacity 与 busy share 积分；取消/错分支算量与传输；单请求输出间隔另列 | V05/V07/V08；M3；不使用 max(D,V) 填请求 TPOT，不把忙时总和直接当墙钟 |
| M19 通信量与耗时 | 消息字段、实际 shape/分片、payload/header、sender/receiver、协议/collective/topology、发送/接收/等待时间；消息与网络 hop 字节分别计数 | V04–V07/V11；M2；协议不全时收益未知，不用常数替代 |
| M20 资源与约束收益 | 设备种类/数量/占用时长、总预算；吞吐/卡或 GPU-second/token 说明异构限制；费用/能耗需单价时间戳、功率轨迹/明确模型 | V08/V09/V11；M2 设备，费用能耗另需来源；无数据不生成美元/电费/节能结论 |
| M21 敏感度与收益反转 | 参数范围及来源、每点 run_id、可行性、比较口径、随机 seed/配对；交点搜索范围、误差、可能多解/无解 | V09/V10；M2；插值和外推标记，未采样区域不声称单调；假设区间不叫置信区间 |
| M22 质量与接受分布 | tokenizer/采样参数、目标分布保持依据、参考输出或 token 概率、任务数据集/revision、A/K/位置直方图；性能验证与质量验证两轴 | V02/V05/V11；M2 小状态正确性，有真实评测才开放任务质量；α 假设不能证明输出质量 |
| M23 覆盖、误差与对账 | source/shape/count/protocol/profile 覆盖分母、旧新值/oracle/单位/差异类别；校准 train/holdout 划分、样本量、绝对/相对误差、适用域、外推状态 | V01–V04/V09/V11/V12；逐阶段；旧新相同不替代 oracle，零参考值不计算相对误差 |

## 5. 验证步骤与交付证据

这是待实施的验收清单，不表示已通过。每次记录输入 hash、实现版本、环境、实际命令、结果、输出产物和覆盖范围；只读源码记录为 source-checked。默认数学计数精确比较；时间/浮点与测量误差的容差按约定量纲预先登记，不在看结果后调整。

| ID / 步骤 | 独立检查、边界与反例 | 通过后产物 / 里程碑 |
|---|---|---|
| V01 身份与协议 | schema/单位/状态/完整 hash；未知与零区分；非有限数拒绝；大整数往返不丢失；旧 schema 显式迁移；资产改动使证据失效 | schema-report、manifest；M0/M1 |
| V02 结构与 forward | config 字段覆盖→固定源码分支→小尺寸 CPU/meta 或可用实现 shape trace；对比层序、tensor、cache 读写。Qwen3-Next 检查混合层序，V3 检查 MLA 与 Dense/MoE 分层；四历史模型固定 revision，V4.1 补 Engram/hc/CSA2 缺口 | field-coverage、source-map、shape-trace、未执行说明；M1，真实大模型 GPU 未跑则不宣称跑过 |
| V03 独立数学 oracle | GEMM/因果可见对数/Residual、GQA/MLA 元素量、MoE E4/k2/M2 枚举、接受 α=0/1/K=0；从独立推导生成 golden，不 import 生产公式 | oracle-cases 和按字段差异；M1 |
| V04 lowering 与守恒 | TP/EP/PP 合法组合、分片/复制总量、融合物化消除、packing/scale/alignment、逻辑到 kernel 多对多；routing/collective 字节独立核对 | lowering-diff、每 rank ledger；M1/M2 |
| V05 方法、协议与状态 | A=0/A=K、拒绝/全接受/EOS/OSL 尾轮、动态 K、bonus、cache prefix、hidden/probability 消息、commit/rewind；小 vocab 精确枚举验收分布，随机检验只是补充 | transition-trace、protocol-report、分布正确性证据；M2 |
| V06 显存生命周期 | 共享/复制、同卡两模型同时存活、verify 槽位、workspace/通信 buffer、分配/释放/预留；内存上限前后边界、各对象峰值错位反例 | allocation-trace、每 rank peak 与 oom/unknown 原因；M2 |
| V07 事件因果与资源 | 事件开始≥依赖完成；资源互斥或显式共享容量；通信先发后用；取消/释放；串行事件与同假设解析相符；双微批与单请求超前独立用例 | trace-validator、critical-path、资源利用报告；M2/M3 |
| V08 系统指标归约 | 从小型人工 timestamp/token trace 独立重算 TTFT/TPOT/吞吐/周期；1-token、EOS、窗口边界、warmup/drain、取消和未完成请求；验证 E[T]/E[N] 不等于 E[T/N] | metric-oracle-report；M2/M3/M4 |
| V09 比较与旧仓对账 | baseline/candidate 口径/精度/资源/质量资格；baseline=自己比率1；OOM/unknown/外推排除；旧/新/oracle 三方逐字段而非只查行数 | comparison-eligibility、migration-diff；M2，旧入口迁移 M5 |
| V10 扫描与运行隔离 | 不可变快照；相同seed重放；并发隔离、取消/失败恢复；每点独立身份；反转点无解/多解/不连续边界 | sweep-manifest、isolation-report；M2/M3 |
| V11 校准与外部真实性 | 核对 raw profile/设备/driver/kernel/shape/dtype、warmup/重复次数；训练与留出不泄漏；预设误差门槛、外推检测；如需任务质量则另有数据集与目标参考 | calibration/quality-report；M4；无机器保持 unmeasured，不阻塞已独立完成的计数证据 |
| V12 UI 与导出一致性 | fresh run→Web/CLI/JSON/CSV 同字段/单位/hash；稳定 ID 四视图联动；stale/失败/取消/空图/OOM/unknown；晚响应不覆盖；过滤/分页不改全局总量；截图检查桌面/手机和键盘可达 | contract snapshot + 关键 E2E/导出对账；M1 起，随能力增量验证 |

验证按依赖放行：结构与计数至少 V01–V04；串行收益还需 V05–V09；流水收益还需 M3 的 V07/V08；测量误差结论需要 V11。V12 验证“显示正确的产物”，不能给底层数学或物理结果增加证据。

## 6. 接入顺序与退出条件

Q9 已于 2026-11-09 确认：保留完整布局，按下表逐区开放；首批结构/计数可独立交付，不等待全部串行/流水区域完成。未支持、未配置、证据不足和运行失败分别显示原因；未接入区域不使用展示数字冒充生产数据。

| 阶段 | 后端到界面的最小纵向交付 | 退出条件 |
|---|---|---|
| M0 | 原版元素清单→页面/指标 ID；能力与证据字典；原资料/计划/原理导航 | 每个区域有数据生产者、验证 ID、缺失行为；示例有固定入口且不能混入生产 run |
| M1 | 微型模型与四历史结构→ModelIR/ledger→图/检查器/公式；catalog/capability/run schema；静态容量 | 改层型/维度可观察真实图与计数变化；Qwen3-Next/V3 层序与字段证据可下钻；V01–V04、V12 对应部分通过 |
| M2 | 普通 decode/同卡/DT 串行→状态/容量/成本→摘要、矩阵、串行预算/事件、基线与导出 | 每个有数值格都有独立 run；协议、资源和容量闭合后才显示有资格的收益；移除生产路径原版 mock 计算 |
| M3 | 两微批/超前方案→合法资源事件→三轨时间线、周期/请求延迟/浪费 | cohort/request/round 可追溯，warmup/drain/取消可见；未过门槛仍保留禁用格和原因 |
| M4 | 服务请求轨迹/profile→TTFT/尾延迟/校准详情 | 样本与窗口完整，holdout 及适用域可见；测量与仿真绝不混成一条无标签序列 |
| M5 | 两旧仓薄适配器与存档重放→差异页/原入口切换 | 逐场景对账有 oracle、版本与回退；旧值不覆盖新数据身份 |

正式接入时须补齐原版检查器未完整显式呈现的精度、瓶颈与公式字段；页面标题称“七维”不能代替实际字段覆盖。固定的 5×5 原型矩阵改为能力目录驱动，未实现组合保留禁用/未知状态，维度增加不要求复制页面。

首发规模、接受分布来源、UI 批次与观察边界等决策统一见 Q1–Q12。技术必须项（ID、单位、缺口、未知不填零、仿真不冒充实测）不是可选开关。

## 7. 第二轮审查补强：可计算、可比较、可证伪

本节补充并约束前述简表。审查原文、采纳范围与反例见 [第二轮裁决](review-ui-round2/resolution.md)；这些是待实施门槛，不是生产验证通过声明。

### 7.1 指标必须有可到达的页面区域

| 指标 ID | 主页面 / region_id | 其他入口 |
|---|---|---|
| M01–M04 | P01 / summary-and-round | P04 比较，P05 原始样本 |
| M05 | P01 / moe-summary | P02 路由/专家详情 |
| M06 | P03 / rank-memory | P01 摘要 |
| M07 | P05 / evidence-gaps | P01 证据条 |
| M08–M14 | P01 / operator-inspector | P02 结构/配方，P08 公式 |
| M15 | P01 / roofline | P05 profile 详情 |
| M16 | P01 / execution-timeline | P03 消息/资源 |
| M17–M18 | P04 / request-and-system | P01 对应指标展开 |
| M19–M20 | P03 / communication-and-budget | P04 资源约束比较 |
| M21 | P04 / sensitivity | P01 矩阵展开 |
| M22 | P05 / quality-and-acceptance | P01 接受分布控件/摘要下钻 |
| M23 | P05 / coverage-and-error | P06 新旧/oracle 差异 |

P05 的质量区必须显示数据集 revision、采样/目标参考、质量轴与接受分布；覆盖区显示分母、blocking gaps 与受影响结论。UI 导航契约检验每个指标有 region，区域的数据与证据均可到达；不是只检查表格中出现指标名。

### 7.2 ComparisonPolicy 与过滤范围

新增 `ComparisonPolicy{policy_id,version,hash,allowed_differences,quality_requirement,resource_budget_rule,workload_pairing,ranking_metric_ref,eligibility_predicates}`。`ranking_metric_ref` 包含 metric_id、TPOT kind、sample_unit、estimator、scope、result_kind、单位及优化方向。不同模型/dtype/recipe 可以是刻意实验变量，但必须显式登记 allowed_differences，绑定质量要求与比较目的；不把“相同 dtype”当所有实验通则。

资源规则至少支持 `equal_total_inventory` 与 `equal_target_plus_disclosed_delta`，异构资源按型号/数量向量比较，不能只算卡数。Q4 已选择两种政策都研究并分别比较：后一政策以目标/草稿分离收益为研究目标，草稿资源变化须显式披露。每格输出 `predicate_results(status,reason,evidence_refs)` 与 resource_delta；失败、未知、仅有理想界不得自动变成绿色赢家。缺 profile 但适用域完整的解析估算可以进入**明确标为估算**的同类比较，不能与 measurement 混排。

矩阵着色绑定同一个 ranking_metric_ref；颜色图例和单位可见，禁用格独立纹理，区间重叠/无显著差异不强行宣称统计胜者。配对对比报告差值与基线定义；baseline 自比为 1 的前提是指标为有效正值。V09a 检验同资源与增量资源两种 policy、允许/不允许的变量改动、资格与排序的一致性；V09b 单列旧仓迁移对账，不把 M5 旧入口切换设成 M2 新实验的前置条件。

全局摘要明确 `aggregation_scope=run/global`；选择聚合是单独后端查询，返回 `selection_scope/filter_hash/population_count`。首版可以只显示“全局结果，当前筛选 3/10”，无需强制同时实现所有子集聚合。导出声明 full/filtered，并包含过滤谓词和 scope；分页不改变任一聚合的定义。V12 用部分筛选、空子集和分页验证这一边界。

### 7.3 随机流与统计身份

运行身份增加 `rng_scheme{algorithm,version,master_seed,stream_derivation,semantic_key_fields,counter_policy}`；为 workload、acceptance、routing、cost noise 等用途区分流，支持按请求/轮次/用途/抽样索引复现外生输入。方法参数改变时采用什么 common-random-number 配对规则须记录；相同 seed 本身不证明配对有效。

V10 在相同语义键和抽样分布下验证请求重排/插入无关请求不会意外错配随机抽样；**不要求事件时间不变**，资源竞争会改变等待。若调度改变动态 K 或接受状态，只要求相同已定义键的原始随机输入一致，并解释发生分支变化的原因。另对相同输入/版本/确定性调度检查完整重放；跨平台浮点仅按已定义容差，不能宣称所有硬件位级一致。

M17/M22 补 `sample_unit=request|token|round`、weighting、population/filter、quantile estimator、缺样本/取消/截尾政策；`resampling_scheme=trace_preserved|iid_marginal|conditional_model` 及相关分组。trace 保序重放保存 request_id/round_id 与上下文，直方图重采样只保存边际。V08 构造“同边际、不同请求内相关性”与长短请求混合反例，分别核对统计结果。无真实 trace 可以输出假设下的 simulated 分位数，但不能叫实测、trace replay 或真实负载预测。

### 7.4 显存容量与 MoE 路由执行

设备快照补 `nominal_bytes, allocatable_budget_bytes, snapshot_time, source, runtime_context, external_occupancy, budget_accounting_policy`。预算必须与 allocation ledger 使用同一计量边界：已经从预算扣除的 context/外部占用不能再次入账；allocator reserved、allocated、active 是嵌套指标，不能无条件相加。每 rank 显示预算来源与局限；单次 free-memory 快照也不保证未来无其他进程变化。只有标称容量时，超过可证明上限可以报 OOM，低于它不能证明 runtime feasible。容量未知不得用任意保守常数悄悄升级为 feasible。

V06 加入标称 80、可分配 78.5、所需 79 的 synthetic fixture，以及同一预留“双扣”负例、active/reserved 重叠、外部占用/碎片与连续块需求。数值只验证契约，不代表任何实际设备。

MoE schema 补路由概率/trace、expert placement、capacity_limit 或 factor 的分母/舍入、overflow/drop/reroute、padding、dispatch/combine policy。分别记录 offered、accepted/executed-valid、dropped、rerouted、padded-slots；在明确 stage 上定义守恒，避免 rerouted 双计。**逻辑 FLOPs 用有效执行 assignment，物理执行 FLOPs 还可能包含 padding**；权重读按实际专家、切块/复用和 rank 计算，不能仅乘有效 token 数。丢弃可能改变输出语义，M22 显示变化并重新审查质量资格。

V03/V04 用小专家热点、容量边界、padding 和跨 rank 不均衡独立枚举。首发未实现的 overflow policy 明确 unsupported；dropless 也必须由具体模型/recipe 证明，不靠缺省 drop=0 推断。无需为每种未支持策略新增用户审批，优先级沿用 Q1/Q5。

### 7.5 从“声明的依赖合法”到“依赖完整且没有预知”

事件补 `decision_time, visible_input_refs, input_ready_at`；batch/cohort 也纳入稳定 ID。但自报这些字段不能单独证明无预知。执行策略只通过受限可见状态接口读数据；验证器从请求到达、消息发送接收、tensor/cache producer-consumer、状态转换规则独立重建必要边，而不是复用调度器的 deps 生成函数。

V07 负例至少包含：删除 arrival→batch_start 边、消费尚未到达的消息、未来队列长度参与当前决策、缺前轮提交/回退、跨 rank ready 误用。加入 future-perturbation 检查：在明确非预知策略和固定外生流下，仅改变当前时刻不可知的后缀输入，不应改变此前决定。策略若合法知道 max_new_tokens 等已声明信息，不将它误报为泄漏。离线理想 oracle 调度必须另标，不能与在线策略默认混排。

V02 拆分为 source coverage、shape/branch trace 与必要的数值语义参考。shape 相同不证明 mask、位置、缓存更新或 recurrence 正确。对所支持的变换用小尺寸确定性输入、独立朴素参考核对输出/状态：例如因果 mask、MLA 展开与吸收、逐 token 与分块递推、Dense/MoE 路由与共享专家；分别定义 dtype 和容差、记录 reference 版本。引用同一生产 helper 的两条路径不能算独立参考。数值参考证明的是该变换/输入域，不证明完整大模型质量，更不证明 GPU 性能。

V03/V05/V07 的测试必须包含能杀死已知错误的负例/变异：mask 错位、丢掉通信依赖、重复统计共享对象、少一个 bonus/回退、漏 padding。记录“哪个错误由哪个断言发现”。独立性是按 claim 的来源、实现与覆盖范围评审，不由目录不同或测试数自动保证。

### 7.6 周期、时间戳与请求观察边界

一般流水周期从合法事件与资源需求推导，不套通用二阶段 max 式。资源负载下界、有限 cohort 循环下界与实际 schedule 分列；反馈消息、transition、launch、共享/双工链路和取消开销都有资源归属。只有明确两阶段独立资源、阶段串行内容、反馈归属、足够在途任务且无额外竞争时，才能给出特例 max 式。V07/V08 用独立手排时间表检查 warmup、drain 和反馈瓶颈，不能仅检查事件核与同源闭式相等。

token 时间区分 `computed/accepted/committed/server_emitted/client_observed`，并记录批量释放/flush/序列化政策。M01/M17 的观测边界必须写入 ranking_metric_ref；缺网络/客户端模型时只显示服务端口径。一次提交多个 token 可以有相同 emission 时刻，不能人为均匀摊开来降低 ITL。跨节点 trace 保存 clock domain、同步/偏移估计与不确定度；无法对齐的时钟不直接相减生成端到端时间。

吞吐明确窗口 [start,end)、按窗口内 token 事件还是窗口内完成请求计数。二者不同命名，不混用；goodput 还需完整 SLO 观察范围，不把尚未完成请求默认为合格。V08 用跨窗口请求、同批多 token、服务端与客户端缓冲差异验证；观察目标的产品选择统一补入 questions.md。

### 7.7 动态 K、分支覆盖与采样证据

动态 K 时，M03 需要联合分布 `P(K,A,state/context)`、逐轮保留这些字段的 trace，或明确登记的条件生成模型。边际 α_i 和 K 分布不能唯一决定 E[N]，更不能确定与状态相关的 E[T]。缺联合信息时，受影响的解析产出/TPOT/收益返回 null+reason；只有显式选择独立性等近似假设后，才产生 approximated 数值，并向 M01/M02/M21 传播。固定 K 也仍需说明条件接受率与上下文/跨轮状态假设。换 K 策略后原 trace 是否可迁移必须另验，不能当作相同实测分布。

V03/V05 增加联合律反例：K=1/2 各半、每轮时间 6/10；一个世界 α1|K=1 为 0.5、α1|K=2 为 0.9，另一个 α1 与 K 独立且为 0.7，第二位置条件接受率都为 0.8。E[N] 分别 2.06/1.98，E[T]/E[N] 分别约 3.8835/4.0404；相同边际不能辨识真值。该例只证明缺字段的问题，不代表任何真实方法的误差大小。

V02 的 source-map 增加 `branch_predicate,source_location,input_domain,thresholds,shape_case_refs,uncovered_domain`。根据**实际源码谓词**选择边界两侧与等值用例，包括长度、batch、dtype、layout/backend；不猜测某个 r 就是切换阈值。小尺寸无法覆盖的真实大 shape 可尝试 meta/static 证据，但仍标明哪些路径未执行。V01/V02 拒绝把域外证据直接挂成该场景已验证；同 shape 的数值语义检查与分支覆盖分别过关。

采样类 Metric 增加 `sample_metadata{population_id,sample_unit,n_observations,n_requests,n_replicates,weighting,estimator,correlation_groups,raw_artifact_ref}`；未知或不适用字段必须说明。区间另含 `interval_kind,method,level,bounds,assumptions` 或 `unavailable_reason`，不是强制为 n=1 编造标准差/CI。记录不足时可以报告观察到的差值，但显示“证据不足以判断稳定优劣”。解析期望没有随机采样时不伪造 n。

若报告显著性，Comparison 保存与实验设计相符的配对差值/比率推断方法、独立重复单位及多比较政策。两个区间是否重叠、差值是否小于单个半宽，都不是通用显著性检验。V08 包含最近秩与插值分位数、长短请求权重、goodput 的完成/取消与 SLO 判定、busy-share 积分的人工 oracle；V09 检验不支持推断时的降级，不能用固定 n 阈值宣称结果可靠。

### 7.8 T0 参数审计与跨指标对账

M2 前即做 V04-cost：设备资产附 `spec-sheet/vendor-document/measurement` 等来源类型、版本/访问日期、单位、适用 dtype/稠密或稀疏/设备组范围，独立核对转录值和换算；F/Q/P/BW/t 做量纲检查。构造 F=1000 FLOPs、P=100 FLOP/s、Q=200 B、BW=50 B/s 的独立手算锚点，roofline 下界为 10 s；launch/有效效率按明确模型另算。注入单位错误和数值转录错误，并说明由哪条检查发现。

同量纲常数错十倍不会自动触发量纲检查；带宽低估导致时间增大也可能仍满足时间下界。因此源值核对、独立锚点、适用域检查与量纲检查缺一不可。物理时间下界只在相同 kernel/计量范围、匹配硬件执行路径和实际物理字节层次下使用；它不能证明估算精度，V11 的真机留出验收仍独立存在。

V06/V07/V08 增加 BundleConsistencyReport，以独立 reducer 从原始对象重算：

- 同一 round 的墙钟为所定义开始到完成的时刻差；串行无重叠且阶段完整分区时才等于阶段时长和。并行展示 elapsed、各资源 service/wait、关键路径及未归因差额，不把全部事件时长相加作墙钟。
- 消息账本按唯一 message_id、逻辑 payload 或 hop bytes 的既定 scope 求和；消息时延分解与事件绑定，带宽/排队/重叠不重复加到轮次总时。字节与时间分别对账，不跨量纲强求相等。
- 利用率从资源 capacity/share 的时间积分重算；同资源重叠按声明容量，不重复计 busy。内存从唯一 live allocation 与预算政策重算每 rank 峰值，不能用各对象峰值之和替代。
- 逻辑工作量与融合 kernel 的映射检查守恒或明确记录 padding/重计算等差额。所有汇总标出排除对象，不能靠未说明的 residual 让账“归零”。

负例分别篡改单个事件时长、消息字节、allocation 共享 ID、cost 版本和过滤范围，至少一个对应不变量必须失败；合法重叠轨迹不能被串行加法规则误拒。UI 的 V12 继续只负责显示/导出一致性，不承担底层正确性的证明。

### 7.9 运行历史、phase 与缓存解释

P01 增加历史/已存场景抽屉，P04 承载扫描索引，消费 `RunIndex{run_id,status,scenario_ref,input_digest,result_digest,created_at,comparison_refs}` 与 SavedScenario 的版本。重开历史只读取冻结结果；修改参数生成派生场景/新 run，不覆写旧记录。删除策略与保留规模按 Q10，未实现自动清理前不悄悄删除产物。V12 验证重开历史 content hash、失败/取消/部分结果、缺失资产的原因；导出注明是否自包含及缺失引用。

无需增加首发顶层页面，P01 摘要下钻/P03 缓存详情/P05 证据区补 M04/M11/M17 的 phase 与 cache 子指标：prefill、首轮、steady decode、尾轮、drain 的时间/算量/流量；cold/warm 初态、有效前缀、命中/查询 token 与 block 分母、实际节省的工作/字节、分配/驱逐/重计算、共享与命名空间隔离。Prefix Cache 不再只有一个 h。仅有 decode 模型时 prefill 指标为 unsupported，M2 可提供孤立 prefill 预算，含到达/排队的服务 TTFT 仍在 M4。

V05/V06/V08 增加 K=0 baseline、空/全/部分缓存命中、页/块边界、eviction、cold→warm、OSL 尾轮与 phase 聚合独立用例。tokenizer 兼容校验针对 token ID/词表/特殊 token/采样映射及方法要求；hash 不同是调查信号，不是所有方法绝对不兼容的证明，未验证映射时拒绝对应组合。

任何比率均可展开分子/分母绝对值、单位、baseline、样本与区间来源。P04 的帕累托视图列为可选后续，不成为首发门槛；Q4 已确定双资源政策，Q11 已确定多目标并列，不为视觉形式新增问题。


## 8. 2026-11-08 Q1 场景决策

生产 Catalog 必须登记两条目标：DeepSeek-V4.1-Flash 主模型支持，以及 Qwen3.6-35BA3B + DFlash 明确主草稿配对。这里沿用用户给出的名称，准确 model_id/revision 与 DFlash checkpoint/结构/方法版本在资产接入时核实。不得用 Qwen3-Next、历史 Qwen3.8 或原型 DFlash2 预设代替该配对。

P01 配方、P02 结构/证据和 P05 缺口分别展示各角色的身份、覆盖与兼容状态。Qwen 配对进入 V02–V06 的双角色验证，再按已核对的部署配方进入 M2 比较。V4.1 主模型独立验收，其草稿尚未指定时配对相关指标返回未配置原因；两条场景的证据和支持状态不能相互代用。Q7 已明确 HF 参考优先，Q2 已确定首版解析估算并保留 profile 校准接口，Q8 已选择首版假设扫描；不能将双场景选择推导为已完成引擎适配、真机测量或接受分布。


## 9. 2026-11-08 Q7：HF 参考与可替换算子

首版的 P02 配方与 P05 证据以 HF 推理代码为主要参考，模型记录 repository/revision、代码 hash、Transformers/PyTorch 版本、配置、实际 module/dispatch、shape/dtype/device 及缓存政策。DFlash 方法程序与草稿绑定各自实际来源。界面应区分参考实现、替换实现和未接入的引擎，不能显示未运行引擎的已支持标记。

后端为稳定 semantic_op_id 注册 reference implementation 与替换 recipe，记录 implementation_id/version、支持域、输入/输出/状态契约、逻辑到 kernel 映射、count/traffic/memory 产物和 cost_model_ref。更换实现后创建新运行身份，P01/P04 能配对查看同一语义下的差异，旧结果保持历史快照。语义变更则建 ModelIR 变体并重新审查比较资格。

V02/V04 验证 HF 路径的实际分支与 shape、数值/缓存状态、融合/物理账本；V04-cost 独立核对替换后的计价；V12 验证切换实现后 recipe/hash/证据同步、stale 正确、旧证据不冒用。未替换的实现保持同一参考路径。具体引擎适配留到后续，服务调度和跨设备执行的证据不能由 HF forward 代替。


## 10. 2026-11-09 Q2：解析估算与 profile 校准接口

首版性能为 analytical-estimate；语义/计数/状态验证、T0 独立审计和证据闸门照常实施，真实延迟误差不作为首版退出前提。M2 定义以下 cost 层契约，M4 有真实数据后再接入完整校准：

- `ProfileRecord`：原始测量产物/hash、implementation/recipe/version、shape、dtype/layout、设备/软件版本、计时边界（device kernel/host/graph/end-to-end）、warmup/重复/同步政策、样本/统计单位与来源。不同计时边界不能直接替代同一字段，避免 launch/通信重复计价。
- `CalibrationModel`：版本/hash、训练与留出集引用、coverage、模型/拟合方法、误差报告、外推与回退政策；profile 接口不限定统一乘法因子。
- `CostEstimate`：保留 analytic_base 与 selected 值、cost_source、profile/calibration 引用、覆盖域与 fallback_reason。实测 kernel 样本不使整模型或整个请求升级为 measurement；拟合输出依然标 fitted estimate。

P01/P04 的性能摘要显示当前成本来源与未测状态；P02/P05 能下钻 profile 覆盖域、原始记录、校准前后差异与留出误差。切换算子/设备/shape 后重新检查覆盖，生成新 run，旧估算和旧报告保留。只校准成本，不用校准值反改工作量、流量或容量计数以追求对账。

M2 的 V01/V04-cost/V12 使用明确标为合成的记录验证字段、单位、匹配/域外、计时边界、版本失效、回退与显示/导出一致性；这只证明接口接线。真实 profile、拟合与 holdout 的 V11 仍在 M4，无数据时显示 unmeasured，不生成误差百分比或已校准标记。


## 11. 2026-11-09 Q8：首版假设扫描

首版 P01 的负载/接受控件和 P04 的扫描消费 `source_kind=hypothesis` 的显式场景快照，包含变量定义、单位、分布/条件或相关假设、范围、固定量、seed 与版本；每个扫描点绑定独立 run_id。具体范围尚未确定，演示默认值不能冒充用户确认或实测数据。P05 展示假设来源与影响指标；M01–M03/M17/M21/M22 的结果分别标注解析期望或假设下模拟统计，不由输入接受率推断真实输出质量。

前端不得暗中把平均值换成几何分布，或将动态 K 和边际接受率当作独立量合成。V03/V05/V08/V10/V12 检查场景→生成过程→指标的假设一致性、端点、联合律缺失、扫描隔离与导出重放。真实 trace 的采集/导入后续实现，首版不强制建设导入 UI；已有契约中的 trace 字段保留为扩展设计。


## 12. 2026-11-09 Q12：服务端时间口径

首版指标选择服务端 committed/server_emitted，Metric scope、ranking_metric_ref、token 轨迹和导出均保存 `time_basis`，两种边界分列。M01/M17 的请求间隔与 TTFT、M02 的 SLO/goodput 使用同一已声明边界；arrival 与队列语义照旧，纯解析成本不因此成为完整 TTFT。Q11 已确定多目标并列，临时排序必须绑定具体指标。

同一批提交或发出的多个 token 允许同时间戳，保留 batch/round 与释放策略；不为界面曲线平滑而均摊。没有 flush/发送模型时 server_emitted 返回缺口，不默认为 committed。V08/V12 用提交与发出不等、批量释放、缺发送事件、跨窗口 token 与导出 scope 的负例核对。client_observed 作为后续扩展，不进入首版默认赢家比较。


## 13. 2026-11-09 Q4：双资源约束、分离收益与异构性价比

P04 提供两组 ComparisonPolicy：固定总设备型号/数量向量，以及固定目标设备组并列明草稿资源。两组分别比较/排序，第二组以“分离收益”命名。Comparison 增加 research_question、factor_changes、matched_controls 与 attribution_limits；资源差额是解释条件，不将分离部署研究改称扩容收益。

基线按研究问题区分：无投机 baseline 衡量整体投机收益；混部投机 baseline 衡量指定条件下的分离差异；同一分离策略更换草稿设备衡量异构替换差异。对照记录相同主/草稿模型、方法、工作负载与采样/质量条件；无法固定的资源/TP/EP/recipe 等变量明确列出，单组比值不自动证明纯分离因果效应。

P03/M20 保存型号/数量向量、角色与 rank 放置、链路、资源占用时长，并区分分配资源与实际忙时。后续 CostBasis 引用版本化价格/来源/日期、租赁或摊销等计费口径与时间范围、闲置及附加成本政策；P04 才能据选定目标报告每 token 成本、单位费用吞吐等性价比指标。不同型号不直接用卡数或 GPU-second 当统一价格，profile 延迟校准也不等于价格来源。成本数据/目标尚未指定时返回 null+reason。

V09a 用同总资源/同目标资源两类 synthetic 配对检查资格和分组，并注入设备/方法/负载变化验证因素提示；V08/V12 对账资源时间与导出。后续成本模型使用带来源的独立手算 fixture 检查计费单位、分配/忙时和重复收费，不把合成价格当真实经济收益。


## 14. 2026-11-09 Q11：多目标并列比较

Q11 已选择延迟/吞吐/资源并列、后续成本，不默认单一赢家或综合分。§7.2 ComparisonPolicy 的 ranking_metric_ref 调整为可空的临时排序引用，增加 objectives 数组；各目标保留 metric_id/statistic/scope/time_basis/unit/direction、适用域、证据和缺口。单个目标不可用不要求丢弃所有独立已知目标，但不据缺失维度宣称整体占优。

P01/P04 按 Q4 两种资源政策分组显示绝对值、相对各自基线的变化、分离与异构变量。矩阵可切换着色指标，图例可见，不使用未说明的综合绿色赢家。用户选择排序只改变视图顺序，结果/目标定义和资格不变；导出保留目标列表与当前视图选择。无选择时保留稳定方案顺序，多列并排即可，帕累托图后续再接入。

V09a/V12 构造一个低延迟高资源与另一个高吞吐低成本的冲突样例，要求保留各维取舍，不自动加权定胜；加入成本/吞吐未知、不同时间边界、切换排序与颜色、policy 切换用例。将来标注 Pareto 支配时必须使用同一目标/可比较域、完整值与明确误差处理，不能以缺失维度当零。


## 15. 2026-11-09 Q5：方法与服务特性扩展顺序

优先级依次为 DSpark、MTP、自定义草稿模型、Chunked Prefill、PD 分离、Prefix Cache；既有 V4.1 主模型与 Qwen3.6-35BA3B/DFlash 必需场景保留。方法在 M2 状态/协议闭环上按序开放，服务特性在 M3 事件底座之后于 M4 按序接入；投入优先级不代替依赖与证据门槛。

| 能力 / 页面归属 | 必须登记的输入与后端产物 | 对应独立验证 |
|---|---|---|
| DSpark / P01 方法、P02 配方、P05 证据 | 实际 backbone/Markov/confidence/hidden 程序图、checkpoint/revision、动态 K 与接受条件、状态/消息与成本 | V02/V03/V05：源码/分支与小尺寸参考、采样/置信度截断、联合分布、拒绝/全接受与缓存追赶；链式代理不能标为 DSpark |
| MTP / P01/P02 | 具体 head/层与权重共享、多 token 调用/提议/验收路径、状态、所需 hidden 与消息 | V02/V04/V05：分支/数值状态、共享去重、提议与验收轨迹；不能统一假设“一层/固定 K” |
| 自定义草稿 / P01 编辑、P02 结构 | 完整 ModelIR、tokenizer/采样映射、方法兼容域、算子/设备配方、双方缓存与容量 | V01–V06：schema/覆盖、独立计数、映射与状态；不支持组合明确拒绝，不用总参数替代结构 |
| Chunked Prefill / P01 Features、P04 请求、P03 缓存 | chunk/token 预算、调度/插队规则、prefix/cache 初态、分块边界、队列与阶段事件 | V05–V08：分块/未分块语义状态对照、处理 token 无漏重、资源竞争与 TTFT/ITL；不能把总时简单除 chunk 数 |
| PD 分离 / P03 部署通信、P04 对比 | Prefill/Decode 角色放置、KV/状态布局、转移/转换/分片/ack、链路/队列与双端生命周期 | V04–V08/V09a：传输字节与布局、转移前后状态、producer/consumer 因果、两端峰值、无漏发/重复接收；区别于目标/草稿 DT 分离 |
| Prefix Cache / P03 缓存、P04 请求、P05 证据 | 命中 token/block 与分母、cache key/版本/命名空间、布局与有效前缀、分配/驱逐/重计算及实际节省工作 | V02/V05–V08：cold/warm、部分命中、块边界、失效/隔离/eviction、输出/状态与无复用参考；命中率不直接乘全部耗时 |

暂不区分 CUDA/Ascend 优先级。Catalog/Recipe/DeviceCapability 保存具体 backend/device/算子/shape/dtype 支持域，P02/P05 展示实际证据和缺口。不存在两平台均已支持或必须同时实测的推论；使用统一接口不代表相同硬件峰值、布局或 kernel。V04/V12 对 backend/设备切换检查配方与成本身份重建、域外拒绝/显式近似、证据失效与正确展示。
