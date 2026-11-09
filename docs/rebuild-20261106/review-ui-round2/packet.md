# 第二轮独立复核：计划与 UI 数据契约
用户明确要求 DSH 和 ZCode 分别复核。你只需审查本文内嵌的冻结文档，无需调用工具读取同一材料。只读审查，不编辑文件，不运行模型或测试，不委派其他 Agent，不调用 Codex。当前方案均为计划，UI 是展示原型，不应因未实现就报缺陷。
重点：1 页面/指标是否遗漏关键研究信息；2 数据字段能否真正生成指标或验证口径，找不可辨识/不可重放反例；3 验证能证明什么，哪里自证循环/检查不够，给最小独立测试；4 边界/采样/量纲/聚合/资源/质量；5 哪些是工程应自行修正，哪些必须用户决策。
请给至多 10 项高价值发现，每项用稳定 ID、严重度、文件+章节或指标 ID、现有文字已经覆盖了什么、仍缺什么、具体反例、最小修订、可执行验收例、是否需要用户决策。不要重复已明确写出的限制，不要只泛泛建议加测试。区分必须修正与可选后续；别把性能实测当所有数学正确性的必要条件。最后列仍有疑问但证据不足的项。直接用中文 Markdown 给最终结果，不输出长篇过程。所有事实仅来自内嵌冻结材料；你的提议不是已经完成的验证。


---
# FILE: plan.md

# Simverse 重构计划：先建立可证伪的实验底座

状态：2026-11-06 送审后修订版；具体审查闭环见 [review-resolution.md](review-resolution.md)。

## 1. 结论与目标

继续新仓重构，但不把 new-viewer v0.1 的「13 项测试通过、差异归零」视为已建立正确底座。保留它的场景 JSON、统一 RunResult、CLI/Web/导出共用入口以及差异追踪方法；先重新核定模型/算法语义，再扩展能力。两个旧仓保持独立，迁移以场景和证据包为单位；不整体搬入旧计算逻辑与大型 vendored 依赖。UI 已确定直接复用用户 HTML 的工作台元素、样式和联动骨架；其中演示计算不进入生产链路。

用户已选择优先支持**新架构与系统方案实验**，其次解释模型和硬件瓶颈。本计划的第一项成功标准：研究者改一个明确假设，可以看到结构、工作量、流量、执行依赖和指标如何变化，也能知道哪个结论尚无依据。首发目标场景保留 DeepSeek-V4.1-Flash 与自定义草稿；在证据不足时允许查看结构和假设，阻断「已验证部署收益」结论。精确微型样例与四个历史模型作为独立回归锚点。

此轮交付计划、原理、审查记录及 UI 示例。没有修改三个项目的生产仿真逻辑，也没有进行真实模型/GPU性能测量。`new-viewer` 当前目录没有 `.git`，下文「新仓」指拟成立的独立仓库；本次没有擅自设置远端或提交用户已有文件。

## 2. 对三份材料的裁决

| 材料 | 保留的价值 | 必须修改的结论 |
|---|---|---|
| verification_report.md | 固定场景、输入快照、逐行对账、可运行的13项测试、诚实列出的T0边界 | DSpark链式替代、专家覆盖数错值、漏通信/容量、baseline时间线错误不能归入「已解释即正确」；同源对账只证明迁移一致 |
| 指定提交的 model-verse 计算原理 | 区分逐算子/量级解析、揭示硬编码常数、已有事件与显存时间线资产 | 512专家/top10/100tokens约18%是算术错误；三点自拟合是代理压缩而非物理校准；倍数对比并不天然更可信 |
| simverse_v3_0.html | 联动工作台、角色级部署、提议/验证解耦、状态转换意识 | catch-up=2、KV只读一次、hidden必传、max(D,V)代表TPOT不能作通则；七种部署与所有服务特性同时首发过大 |

证据与逐项纠正见 [evidence-audit.md](evidence-audit.md)。不能用共识、术语或测试数代替独立证据。

## 3. 重新划清模型、方法和实现的边界

建议保持一个新仓内部单体包，模块职责明确，不先拆成多个服务或多个发布包：

```text
simverse/
  contracts/       版本化类型、单位、状态、来源、canonical serialization
  importers/       config/共享 ModelIR/forward evidence → 模型语义
  methods/         自回归、MTP、DSpark等提议与验证状态机
  lowering/        kernel recipe、量化、融合、分片、collective、缓存布局
  count/           确定性工作量、逻辑张量访问、容量对象
  traffic/         数据依赖的期望流量、复用/路由/缓存假设
  cost/            解析成本、profile模型、通信原语；禁止修改原始计数
  execute/         串行解析与资源约束事件执行
  serving/         到达、组批、预算、KV分配/回收/抢占（后续）
  experiments/     参数扫描、配对比较、敏感度、随机种子、产物索引
  api/ cli/ web/   同一结果协议的三个消费者
assets/            可追溯模型/设备/recipe/场景快照
validation/        手算锚点、forward证据、协议轨迹、真机对照
```

数据流：`Scenario + ModelIR → MethodPlan → LoweredGraph → Work/Traffic/Memory → Cost → Execution → ResultBundle`。事件执行时按当前 batch/ctx/state 调用同一 lowering/count/cost；不是一次启动就预先固定所有算子成本。

几个必须独立的身份：

- **模型**：层序、张量、权重共享、缓存producer/consumer。模型名称不能决定一切分支。
- **提议方法**：前向块、Markov head、置信度截断、采样/验收、状态追赶。`S/M/K` 是摘要字段，无法替代程序图；K也可能由置信度动态决定。
- **执行配方**：框架/commit/backend/kernel/dtype/shape/TP/EP/PP/graph模式。相同 ModelIR 可有不同物理实现。
- **部署**：角色放到哪个设备组、资源是否共用、链路拓扑；独立目标/草稿TP是研究能力，不能冒充某个vLLM内置路径已经支持。
- **负载与采样**：请求到达、ISL/OSL、batch、上下文、tokenizer兼容性、greedy/stochastic以及接受分布。

已有 `mv_model_structure` 继续负责模型结构与forward证据，语义schema为唯一入口；新仓中的临时 `_v41_plan` 经独立测试后移入共享包，不保留永久双实现。共享包应从 model-verse 的业务目录独立发布，最终独立仓库还是新仓子包可在打包里程碑决策。两旧仓固定发布版本/hash，通过薄适配器消费；禁止复制包代码。

DSH提出的字段覆盖问题纳入M0硬门槛：每个配置键登记`mapped / metadata-only / ignored-with-reason / unsupported-semantic`，记录具体消费者与证据；未知语义字段使受影响结论降级。V4.1实际前40层比例是2个r0、18个r2、20个r1，另3项与草稿层的对应须显式表示，不能只用数组长度判断模型错误。Engram与hc路径在本次补读的vLLM实现中确实参与forward，当前账本缺失它们，不能继续发布完整模型容量/性能结论。

## 4. 结果协议与能力闸门

每个对象使用稳定 ID：`model/layer/logical_op/kernel/allocation/event/request/round/message`；支持逻辑算子到融合kernel多对多映射。UI联动依赖ID，不能依赖名称、行号或列表顺序。

每个关键结果字段附带：`value, unit, scope, statistic, evidence_refs, assumptions, formula_id, input_refs`。`statistic`区分精确整数、期望、采样值、界；不把期望流量截成整数后称为精确计数。

采纳两位审查者关于实现负担的意见：来源、假设与公式存为去重字典，行只引用ID，阶段/配方可继承默认，异常覆盖。FLOPs、weight/activation/KV各分量和时间分别绑定公式，不能用一个`gemm_flops`解释整行所有量。公式函数的重放检查属于内部一致性测试，独立oracle仍必须另有推导。

状态分开表示：

| 维度 | 示例 | 约束 |
|---|---|---|
| 支持性 | supported / approximated / unsupported | 不支持的模块不填0；返回原因和仍可计算的字段 |
| 证据 | config / source-checked / shape-checked / counted / measured / fitted | 可同时存在、逐字段记录；没有单一「可信度90%」 |
| 可行性 | feasible / oom / invalid / unknown | OOM/未知容量不得进入默认赢家排序 |
| 结果语义 | analytical-estimate / optimistic-bound / event-simulation / measurement | 理想下界不混入实测部署排名 |

RunResult里不存在的字段用`null + reason`，禁止 NaN/Infinity 的非标准 JSON。完整SHA256持久化，短hash只用于显示。运行身份至少包含输入、ModelIR、method、lowering/cost版本、硬件/校准快照、代码revision、随机种子；相同scenario_digest不足以保证版本升级后重放相同。

特性组合通过显式capability检查：例如未知量化kernel、未建模通信、缺草稿缓存/hidden协议、tokenizer不兼容、无证据的架构字段。允许探索模式继续输出独立可知部分，但污染路径的派生收益必须同时标记，并保留原始缺口。

## 5. 分阶段实施与验收

工期为单人专注开发的估算，依赖真实源码可得和机器可用，不是交付承诺。先做M0–M2形成可用闭环；后续按实验需求排序。

| 阶段 | 范围与交付 | 退出条件 | 估算 |
|---|---|---|---|
| M0 证据冻结与红灯 | 归档三个原材料及hash；引入现有测试；登记错误/近似/未知；建独立仓与最小CI；提交关键反例 | 每个报告结论能追到输入/公式/实现；已知错误测试会失败；确认数据来源和许可 | 2–3天 |
| M1 语义与计数 | 共享ModelIR版本；微型Dense/GQA、MLA、MoE、混合层序样例；V4.1+DSpark证据图；逻辑/期望/物理计数分层；第一版联动UI | 手算与独立枚举一致；shape trace与源码绑定；未知架构阻断；最小配置改动确实更新图与工作量 | 5–8天 |
| M2 串行实验闭环 | 自定义草稿完整结构；普通decode/同卡/DT串行；双方KV状态、协议、collective、容量生命周期；T0成本插件与敏感度 | 单轮/跨轮状态守恒；拒绝/全接受/EOS/OSL尾轮用例；串行事件与解析在相同假设下相等；排名可行性合格 | 5–8天 |
| M3 资源事件核 | 两微批流水与单请求超前分开；资源队列/双工/取消/浪费；跨轮trace与指标 | 任意事件依赖合法；同资源无非法重叠；warmup/drain计入；能重现有限微批吞吐和单请求延迟不同 | 5–8天 |
| M4 服务与校准 | 到达/预算组批/Chunked Prefill/Prefix Cache/KV分页；profile导入、holdout、误差域；PD分离逐项启用 | 结构与状态对照通过；有测量才发布性能误差；固定seed可重放，尾延迟报告样本与区间 | 1–2周起 |
| M5 迁移与扩展 | 四模型两旧仓回归；投机方法逐个注册；Ascend/异构/视频另立scope | 每个迁移场景达到下述准入标准，再切换入口；旧路径有回退版本 | 按场景排期 |

**并行推进而非延期到最后：**M1就接回Qwen3-30B-A3B、用户称Qwen3.8的本地模型（以实际model_type/revision辨认）、Qwen3-Next和DeepSeek-V3作为结构回归，M2完成prefill/decode及TP1/8账本；M5只是旧界面接入和覆盖扩展。不同模型真实支持的TP集合必须由适配器校验，不盲测非法组合。

M2内设三个先后关口：**协议/部署能力先审定 → 资源与容量图闭合 → 成本计价和收益比较**。launch在M2随kernel recipe处理，不拖到事件核之后。DSH建议把各关口独立扩成M2–M4，采纳依赖顺序，暂不增加顶层里程碑。首版保留原 HTML 完整工作台的信息布局，按 M1/M2/M3 的数据能力逐区启用：M1 接结构/计数/证据，M2 接串行摘要、比较与 Roofline，M3 接合法并发事件。复杂大图布局可渐进增强，但不删掉既定信息入口；没有真实数据的区域显示缺口。本轮示例不作为正式产品已实现的证明。

**首发明确暂缓：**所有视频/世界模型管线、训练仿真、跨框架完全等价、无profile的p99精确预测、所有投机方法一键兼容、整个前沿引擎vendoring。这些保留插件契约，不做空实现掩盖缺口。

## 6. 验证体系

1. **数学锚点**：GEMM、因果对数、GQA/MLA容量、MoE小集合枚举、通信字节、接受分布。golden的期望值不import生产公式；验证独立性靠推导和来源，不靠另建目录。
2. **结构/forward**：固定config+源码commit+分支条件；小尺寸CPU/meta shape执行；记录真实算子依赖、共享缓存、layer order。静态AST阅读与执行核验分栏，hook无法覆盖fused kernel时明示。
3. **lowering**：TP/EP分片守恒、复制开销、KV布局、packing/scales/alignment、融合消除哪些物化读写。整数计数在同一约定下精确相等；浮点容差按量纲/运算推导。
4. **协议/状态**：accepted/rejected/bonus token位置、草稿/目标cache length、消息体、commit/rewind/free的生命周期；A=0、A=K、EOS、K=0的普通decode独立分支、空batch、长上下文、显存边界。
5. **执行**：依赖和资源不变量、同输入同seed同结果；概率枚举/长序列统计验证解析期望。解析/事件相等仅限同一语义和资源假设，不要求复杂调度强行对齐闭式。
6. **硬件校准**：真实profile原始数据、设备/driver/kernel/版本、warmup/重复次数、分位数、留出集。误差门槛在看到结果前按使用目的预设；没有机器时状态保持unmeasured，不虚写「误差<5%」。
7. **应用一致性**：Web/CLI/export同run_id与内容hash；基线切换、过滤、单位、四视图联动。UI契约快照和一次关键交互端到端测试足够，不为样式重复造测试。

补充工程闸门：扫描接收不可变的设备/链路快照，不能临时修改全局注册表再恢复；并发run隔离、失败后清理与取消需要测试。CSV/export对比fresh RunResult的逐字段内容，不能只导出旧存档并检查行数。新仓采用标准UTF-8日志，修复本次测试暴露的子进程解码警告后才将CLI输出检查列为稳定验收。

## 7. 两旧仓迁移策略

迁移清单的单位是「模型revision × phase × batch/ctx × dtype/recipe × placement × method」。每项记录旧值、新值、独立期望、差异原因、证据状态；差异只有**修复正确性、近似变化、口径不同比较无效、未解决**四类，禁止用「已解释」作为自动通行。

优先迁移共享ModelIR→逐层账本→串行计算→事件导出→旧UI入口。旧计算器先保持可运行，只修阻断性缺陷；新旧同输入做shadow输出，新版达到结构/计数/状态/性能各自约定的门槛后再转入口。每次适配器升级固定wheel/hash，报告保留旧版本可重放。

新仓不直接依赖旧仓内部文件或任意相对路径。抽取代码先核对LICENSE，保留来源和必要声明；上游源码只存最小验证快照/指纹，完整参考checkout移出正式发布包。

## 8. UI方案正式纳入实施范围

**已确定：直接复用原 HTML 工作台的信息结构、样式与交互元素。** 场景栏、Target/Draft 配方、七项摘要、部署方法矩阵、算子图、七维检查器、Roofline 与三轨时间轴均保留。原文审查、计划、原理继续作为资料导航；证据、配方、部署/内存与扫描详情逐步接入。完整页面/指标映射见 [UI 数据与验证契约](ui-data-contract.md)，复用范围见 [元素对照](prototype/reuse-map.md)。无需再次决策是否采用该 UI 方向。

该契约定义 **8 类页面/详情视图、23 组指标和 12 组验证步骤**，每项都有后端生产者、输入/单位/范围、缺失行为与开放里程碑。P02–P05 可先以抽屉或子视图实现，P06–P08 可继续使用版本化文档；不要求首版同时做八个完整应用页面。

| 阶段 | UI/后端联合交付 | 必须提供的证据 |
|---|---|---|
| M0 | 元素清单与页面/指标 ID；Catalog/Capability/ResultBundle schema；资料导航 | V01 身份及缺口登记；示例入口与生产数据隔离 |
| M1 | 后端模型/配方/层序/账本→算子图、七维字段、公式与证据；静态容量 | V02–V04 结构/数学/lowering，V12 同一 run 的显示/导出 |
| M2 | 状态/通信/内存/成本→七项摘要、矩阵、串行时间线、Roofline 与扫描 | V05–V10 的串行范围；每格独立 run、比较资格、每 rank 可行性 |
| M3 | 资源事件→微批/超前轨迹、周期/延迟、取消浪费 | V07/V08 并发、因果、请求指标；不能将周期代替 TPOT |
| M4 | profile 与服务请求→校准页、TTFT/尾延迟/SLO 指标 | V11 原始样本、留出验证、适用域与误差口径 |
| M5 | 两旧仓入口/存档→三方差异与回退 | V09 新旧/oracle 对账与 V12 全路径一致性 |

选择方案、层/算子或时间事件共享稳定 ID，逻辑算子与融合 kernel 允许多对多映射。修改参数使旧结果 stale；输入草稿与不可变 run 快照分离；晚到响应不能覆盖新场景。筛选只改变视图范围，不悄悄重算全局摘要。比较资格及正式派生指标由后端计算，Web/CLI/export 消费同一个结果。

结果先显示适用域和证据，再显示小数。模型结构不完整时保留已知图和缺口；OOM/容量未知不参与默认赢家排序；steady-state 周期、解析 TPOT、请求 token 间隔、服务吞吐分别展示。原型中未绑定执行核的模拟数字不能作为生产接口失败时的回退；原版 JavaScript 演示公式须在生产接入时移出生产路径。

当前 [主工作台原型](prototype/index.html) 直接复用原版 HTML，数字为交互展示数据；[独立教学样例](prototype/teaching.html) 使用离线可复算预算。两者都不是生产执行核。保持原版完整存档只为追溯，不代表接受其全部公式与结论。本次只更新计划/契约，没有实现上述后端接口。

## 9. 待用户决策

见 [questions.md](questions.md)。其中已有明确答案的事项不重问；尚未决定的内容用推荐默认方案推进可逆工作。是否拥有真机数据决定性能校准的验收上限，不影响先完成语义、计数与状态验证。


---
# FILE: ui-data-contract.md

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

| 阶段 | 后端到界面的最小纵向交付 | 退出条件 |
|---|---|---|
| M0 | 原版元素清单→页面/指标 ID；能力与证据字典；原资料/计划/原理导航 | 每个区域有数据生产者、验证 ID、缺失行为；示例有固定入口且不能混入生产 run |
| M1 | 微型模型与四历史结构→ModelIR/ledger→图/检查器/公式；catalog/capability/run schema；静态容量 | 改层型/维度可观察真实图与计数变化；Qwen3-Next/V3 层序与字段证据可下钻；V01–V04、V12 对应部分通过 |
| M2 | 普通 decode/同卡/DT 串行→状态/容量/成本→摘要、矩阵、串行预算/事件、基线与导出 | 每个有数值格都有独立 run；协议、资源和容量闭合后才显示有资格的收益；移除生产路径原版 mock 计算 |
| M3 | 两微批/超前方案→合法资源事件→三轨时间线、周期/请求延迟/浪费 | cohort/request/round 可追溯，warmup/drain/取消可见；未过门槛仍保留禁用格和原因 |
| M4 | 服务请求轨迹/profile→TTFT/尾延迟/校准详情 | 样本与窗口完整，holdout 及适用域可见；测量与仿真绝不混成一条无标签序列 |
| M5 | 两旧仓薄适配器与存档重放→差异页/原入口切换 | 逐场景对账有 oracle、版本与回退；旧值不覆盖新数据身份 |

正式接入时须补齐原版检查器未完整显式呈现的精度、瓶颈与公式字段；页面标题称“七维”不能代替实际字段覆盖。固定的 5×5 原型矩阵改为能力目录驱动，未实现组合保留禁用/未知状态，维度增加不要求复制页面。

首发规模、接受分布来源、UI 批次等决策统一见 Q1–Q11。技术必须项（ID、单位、缺口、未知不填零、仿真不冒充实测）不是可选开关。


---
# FILE: questions.md

# 留给你的决策

可直接在「我的回答」后填写。无需重答已确定事项：两个旧仓继续独立；共享模型结构与forward核对；新仓优先研究架构/系统方案，其次解释瓶颈；UI 直接复用 simverse_v3_0.html 的工作台元素和信息布局。

本文件是唯一的待决策入口；页面与指标定义见 [UI 数据与验证契约](ui-data-contract.md)，不在其他文档散落提问。推荐值尚未视为你的确认；可先按默认做可逆设计，远端/公开发布、真实机器占用与硬件测量不因默认值自动获得授权。

优先回答 Q1–Q3、Q7–Q8，可以决定首个真实数据闭环。其余可随后补充；未知内容填“暂不确定”即可。Q1–Q7 原编号保留。

## 启动对应实施前需要确定

### Q1 首个真实实验要回答什么

推荐：**DeepSeek-V4.1-Flash + 一个明确的草稿方法，比较同卡与DT串行；DSpark语义核实后再加入**。输入固定ISL/OSL/并发、接受分布和设备预算，输出结构、每轮成本、容量及收益反转点。

备选：先做Qwen3-Next混合层序/硬件架构变化，或DeepSeek-V3 MLA+MoE/TP-EP变化。首个实验决定适配器与profile优先级，不改变共同底座。

我的回答：

### Q2 验收是否包含真机性能校准

推荐：**第一版语义/计数/状态验证必须完成，绝对性能保持T0估算；有现成profile优先导入**。如果必须以真实延迟误差验收，请写设备型号/数量、可用时间、运行框架/版本和已有日志位置。没有profile不承诺某个误差百分比。

我的回答：

### Q3 新仓和共享包的归属

推荐：**新仓暂名simverse，使用现有new-viewer目录承接；共享ModelIR最终独立版本发布，两个旧仓固定版本**。需确认仓库名、远端地址、公有/私有、shared包最终独立仓还是新仓子包。目前new-viewer尚未初始化Git。

我的回答：

## 可以稍后回答

### Q4 实验比较的资源约束

默认：同时显示单请求延迟、吞吐和设备数量；优先做**总设备预算相同**的比较，额外草稿卡另列增量资源。若你主要研究采购成本/功耗，补充约束与来源；若只追单请求延迟，也请注明。

我的回答：

### Q5 方法与平台的第二优先级

默认：自定义链式→真实DSpark→MTP/DFlash逐个接入；CUDA先，Ascend和跨引擎由实验需求驱动；PD分离排在可验证的DT串行与事件核后。请选择最想提前的一项，并给具体场景。

我的回答：

### Q6 使用与分享方式

已确定复用原版工作台的信息密度与元素，无需再选视觉方向。默认：本地单用户浏览器工作台、JSON/CSV/trace/HTML 导出；无需账号/云服务。还需选择：是否必须多人协作、远程访问或离线分享结果快照？若有，请说明接收对象，以及源码路径、模型名称、设备信息是否允许包含在导出中。技术栈由实现按本地部署约束选择，不另增一项技术选型问答。

我的回答：

### Q7 首发需要对齐哪种执行环境

推荐：**内核允许研究性部署，具体结论绑定一份已核对的执行配方；先验证可行的串行路径**。请注明必须使用vLLM单进程、允许目标/草稿独立服务，还是允许自定义runtime。当前所查vLLM独立草稿的进程内路径要求目标和草稿TP相同，不能用它证明TP8+TP1的跨服务方案已实现。

我的回答：


## UI 接入后端新增决策

### Q8 接受分布与负载从哪里来

推荐：**首版用显式假设的条件接受率/分布进行敏感度扫描，保留 trace 导入入口；没有测量时不宣称代表某任务真实接受率**。不再将一个平均值静默换算成完整分布。

请选择首个实验的数据方式，并补充能提供的内容：

- 假设扫描：固定或分布化 ISL/OSL、并发/到达、K/动态规则、α_i 或接受数分布及范围。
- 已有 trace：文件路径、产生模型/草稿与 revision、方法、采样参数、任务/数据集、每轮 K/A/提交数、token 时间戳及设备环境；缺哪些字段也可说明。
- 需要新采集：先确定设备与 runtime（Q2/Q7），再单列采集范围和运行安排。

影响：M03 每轮产出、M01/M02 延迟与收益、M17 尾延迟、M22 质量与接受分布；均值不足以决定回退、全接受或尾延迟。

我的回答：

### Q9 首发 UI 的交付批次

推荐：**一次保留完整工作台布局，真实数据按 M1 结构/计数→M2 串行→M3 流水逐区开放**。保留不可用区域及原因，避免必须等全部后端能力才能评审；配方、内存、证据先做详情抽屉，计划/原理/裁决继续文档页。

备选：首个可发布版本必须全部联动区域都有真实数据，则发布时间需要等到 M3 对应能力完成。两种都沿用已确定 UI，不重复选择风格。若有演示日期或必须首发的区域，请列出。

我的回答：

### Q10 常用实验规模与可接受等待时间

推荐：**先按本地单用户、单次运行、几十个方案的扫描设计；大图/事件按页加载，扫描可取消和恢复**。这是初始工程假设，不是已验证容量承诺。

请给大致规模即可：并行运行数、一次扫描方案数、典型并发请求数/上下文长度、希望保留多少历史 run、单次运行与打开详情可接受的等待时间。未知可沿用默认，M2 用真实产物再测。它决定是否需要额外的后台队列、索引与 trace 压缩，不先引入服务集群。

我的回答：

### Q11 结果排序优先回答哪个业务问题

推荐：**同时展示单请求延迟与系统吞吐，但固定一种主排序指标；在 Q4 的资源约束下，首个串行实验按同口径 TPOT 排序**。OOM、容量未知、协议不完整、仅有乐观界等结果可查看，不进入默认赢家排名。

可选主目标：单请求 TPOT / 服务吞吐 / 满足指定 TTFT 与 token 间隔 SLO 的 goodput / 成本或能耗。若选 SLO，请给阈值与负载；若选成本/能耗，请给价格/功率来源。也可选择只做并列比较，不显示“最优方案”。硬件性能误差要求沿用 Q2，不在此重复提问。

我的回答：

## 决策与交付的对应

| 问题 | 决定的范围 | 最晚需要答案的关口 |
|---|---|---|
| Q1 首个实验 | 模型/草稿适配器、P01/P02 | M1 真实目标适配前；不阻塞微型 oracle |
| Q2 真机验收 | M08/M15/M17/M23 的性能证据上限 | 制订 M2 性能验收与安排 M4 采集前 |
| Q3 仓库归属 | 初始化/发布、共享包版本 | 建远端与共享包发布前；不阻塞本地文档和契约 |
| Q4 资源约束 | 比较资格与 M20 | M2 比较功能前 |
| Q5 方法平台顺序 | 后续插件与 capability 范围 | 首个串行闭环后扩展前 |
| Q6 使用分享 | 部署与导出边界 | UI 部署/分享前 |
| Q7 执行环境 | lowering/协议/设备能力 | M2 协议审定前 |
| Q8 数据来源 | 接受/负载与统计口径 | M2 首个可解释的收益实验前 |
| Q9 UI 批次 | 发布范围与页面交付方式 | 排定首发验收前 |
| Q10 规模 | 运行管理/分页/性能预算 | M2 扫描与 trace 工程验收前 |
| Q11 主指标 | 矩阵颜色、排序与胜出定义 | M2 比较验收前 |

不需要你决定的正确性要求：真实数据身份、未知不填零、公式/单位可追溯、结构验证不冒充性能测量、周期不冒充 TPOT、按 rank 验证容量。这些已作为固定验收条件。


---
# FILE: principles.md

# 原理底座：从语义到性能，逐层保留证据

对应 [重构计划](plan.md)。各原理对应的页面、指标、后端输入及验证步骤见 [UI 数据与验证契约](ui-data-contract.md)。这里定义未来实现应满足的契约；并不声称 new-viewer v0.1 已完成。公式只在明示假设下成立，不能按模型名字套用。

## 1. 六类量不能混算

| 量 | 例子 | 依赖 |
|---|---|---|
| 模型语义 | 第3层是全注意力；层间共享一个KV对象 | config+forward分支 |
| 逻辑工作量 | GEMM乘加、逐元素add、张量访问 | 语义、输入形状、计数约定 |
| 物理流量 | 某kernel实际HBM读、跨rank消息 | lowering、缓存、路由、packing |
| 驻留容量 | 权重、KV块、activation、workspace | 分片/复制、共享、存活区间 |
| 成本 | kernel时延、消息时延 | 设备、recipe、profile与适用域 |
| 系统指标 | TTFT、逐token间隔、吞吐、p99 | 执行依赖、资源竞争、请求过程 |

精确的逻辑算量不保证精确的HBM流量；确定的HBM字节也不保证精确时间。来自分布的期望字节可以是浮点数，绝不通过`int()`升级成精确事实。

## 2. WorkLedger：形状和运算约定先于总参数

对矩阵 `X[M,K] @ W[K,N]`，本项目约定一次乘加计2 FLOPs：`F=2MKN`；权重逻辑字节`KN*b_w`，输入/输出`MK*b_a`、`MN*b_o`。真实实现可能有padding、scale、转置、反量化、融合，另记到LoweredLedger。逻辑FLOPs、执行FLOPs分栏；稀疏峰值不能与稠密工作量混用。

Residual add有`M*d`次加法。融合后不单独产生kernel、可能减少activation物化流量，但并不会使数学加法消失。RMSNorm、softmax的除法/exp/rsqrt分别记录操作类别或明确计数约定，不用Tensor Core峰值直接解释所有逐元素运算。SwiGLU拆gate/up/down及激活，不把总参数乘2当作所有算子。

因果attention有历史前缀S和新增Q个查询时，单序列可见query-key对数为：

```text
P = S*Q + Q*(Q+1)/2
F_QK+AV = 2 * B * Hq * P * (d_qk + d_v)
```

头维相等时为`4*B*Hq*d*P`。softmax、掩码、RoPE、投影单列。S指查询前已存在的历史长度，禁止同一token在S与Q重复。滑窗、压缩、稀疏、线性attention分别生成可见关系或递推状态，不能统一用`L²`。

## 3. KV与层序：结构等价不等于实现等价

标准GQA每token每层逻辑KV元素为`2*Hkv*d`。TP恰好整除KV头时每rank可分到`Hkv/TP`；当TP大于KV头时具体复制/分片由backend决定。不能对所有KV一律除TP，也不能对所有GQA一律复制。

DeepSeek-V3的一种已知压缩MLA布局每token每层保留`kv_lora_rank + qk_rope_head_dim`元素；缓存dtype、scale、对齐另加。其prefill、展开式decode、吸收式decode的投影/乘法图不同；数学等价不意味FLOPs或访存相等。V4.1的CSA2、滑窗、压缩器、索引和跨层producer/consumer单独建图，不能继承V3公式后只改维数。

Qwen3-Next按明确layer_types/配置生成48层计划，再与forward绑定；本地已接入参考的全注意力层为0起始`3,7,...,47`，其余为Gated DeltaNet。此规则是固定模型revision的测试期望，不是对所有Next模型的默认特判。DeepSeek-V3的61层中前3层Dense、后58层MoE，同样由版本化字段和forward共同确认。

共享KV对象的**容量**按对象身份去重；各consumer的**读流量**仍分别计价。FlashAttention或MLA分块可以复用前缀，但`η=1`只是指定tile/cache条件下的乐观实现假设；CTA数量、head分组、split-K、L2重用、调度会改变HBM读放大。稀疏并集也不是完整物理读量：还包括索引检索、块取整、重复装载和预取。

V4.1还需单列Engram查表与注入、hc多流状态和混合操作。官方config前40层的r分布为`0:2 / 2:18 / 1:20`，不是38层都r2；源层与消费者的缓存大小不能套一个全局「890 B/token」。value读取被top-k封顶，不代表indexer候选扫描也封顶：当前ledger的`index_select`在非short路径按`end//r`读取索引K，仍随上下文增长。必须区分index/value/cache-state及其单位（token或block）。

## 4. MoE并集：一个期望模型，不能叫普适物理定律

E个专家，每token从中等概率选k个**不同**专家，M个token之间独立，则某专家未被任何token选择的概率是`(1-k/E)^M`，所以：

```text
E[U] = E * (1 - (1-k/E)^M)
k <= E[U] <= min(E, M*k), M>=1
```

把`M*k`次选择视为有放回独立抽签得到`E*(1-(1-1/E)^(M*k))`，是另一个近似；M=1时甚至小于k。E384/k6/M5时前者29.077034、后者28.894244，都不可能是38.9。对E4/k2/M2，小集合枚举给3，有放回公式给2.734375。这是很便宜的独立golden。

如果专家e在一个token被选中的概率是p_e且token间独立，则`Σ_e[1-(1-p_e)^M]`。真实路由的跨token相关、负载倾斜、分组约束和容量drop会改变分布；有trace时直接计算专家集合和每专家token数。

`E[权重读] ≈ W_shared + E[U]*W_expert`只表示「一次执行内各被用专家权重理想读一次」。kernel切块、重读、cache驻留、EP放置和量化元数据另算。U(M)是递增、边际递减的饱和函数，原报告「随B超线性增长」错误。实际时间可能因实现阈值出现非线性，不能由U式推出。

MoE FLOPs取决于实际token-expert assignments；容量取决于驻留的全部专家；延迟还取决于**最慢rank**而非平均路由。EP dispatch/combine的字节以路由表、远程比例、复制与协议为准，不能无条件用`2*M*d*b`。

## 5. Cost：下界、近似、拟合与实测各有身份

给定某物理kernel的工作量F、物理访问Q、对应执行单元的峰值P与带宽BW：

```text
t_compute = F / P_effective
t_memory  = Q / BW_effective
t_roofline = max(t_compute, t_memory)
```

roofline是在已计资源上的乐观约束。`max + rho*min`可作为显式经验近似，rho=0.15、效率=0.75均须登记假设与适用域，不称为硬件定律；对整模型MFU和局部kernel效率须分开。实际kernel还可有launch、同步、反量化、spill等成本。不能逐逻辑算子加一个launch，然后又按融合kernel重复加。

低比特存储不直接决定计算峰值。W4A16可能反量化到FP16/BF16执行；H100不支持某种FP4 Tensor Core路径，不意味着该kernel自动使用FP8。Recipe须指定`storage_dtype/compute_dtype/accumulator_dtype/scale_layout/kernel`并通过硬件能力检查。

`Σ_i max(C_i,M_i)`与`max(Σ_i C_i,Σ_i M_i)`不同，后者通常更乐观。选择聚合粒度需匹配融合和资源重叠，不可把逐算子结果与整步结果称同口径。

Profile成本插件输入完整shape、dtype、设备、backend与版本；输出值、覆盖域、拟合误差和外推状态。用物理模型生成3点再拟合二次式只是代理压缩，其fit_error不是真机误差。Vidur在所查版本对多个类别拟合profiling median并缓存预测；事件核不自动意味着kernel时延是随机分布采样。

公式溯源按**量**建立绑定，例如`flops_formula/weight_read_formula/kv_index_read_formula/time_formula`。字典中的可调用函数、参数引用与版本hash用于重放，UI显示符号式；一个Attention条目不能用GEMM的公式ID解释softmax、KV读与时间。单一生产维护点允许多种已命名的模型插件，不妨碍独立golden故意采用不同实现。

## 6. 容量与通信：必须进入可行性和关键路径

```text
memory(rank,t) = Σ_unique_live_allocations size(allocation,rank,t)
peak(rank) = max_t memory(rank,t)
```

对象包含权重、KV、状态缓存、activation、workspace、通信buffer、allocator预留与块取整。同卡target+draft要叠加同时驻留对象，并按identity去重真正共享的权重。事件释放只是逻辑长度回退时，不能假定物理block立刻释放；碎片与预留分别列。

点到点的最简近似：`t=latency_setup + payload/BW_effective`，但发送依赖、序列化、DMA、收端消费和链路队列均可上关键路径。collective由算法/拓扑/参与rank/消息大小选模型，TP通信不能因权重已分片而消失。不要再同时加一个含通信的「单步常数」。

跨设备投机协议必须明确各字段：candidate IDs、位置/长度、验收结果、bonus、方法需要的hidden、greedy或stochastic验证所需的概率信息。标准随机拒绝采样失败处要按`max(p-q,0)`重采样；只有候选token及其q值一般不足以恢复全vocab残差分布。可选择接收端重算、传输概率、额外拒绝轮次等协议，但必须计其成本并证明分布正确。6B/token+256B/request只能是某个已定义协议的估算。

hidden是**数据依赖**，不直接等于固定网络payload：可以在目标侧先pool/project再发送，也可以发送各层hidden在草稿侧处理；分片、压缩和算子放置都改变消息与计算。以60KB/64GB/s为例仅约0.94μs（十进制近似），不能说与100μs固定延迟同量级。大batch或更低链路带宽可能改变结论，要逐场景重算。

## 7. 投机接受：均值不是状态

定义K为提议候选数，A为接受的草稿数；在标准链式验收且未遇EOS/输出上限的轮次，N=A+1包含一次目标采样出的纠正/bonus token。每位置的条件接受率α_i给：

```text
P(A>=i) = product(alpha_j, j=1..i)
E[N] = 1 + sum(P(A>=i), i=1..K)
```

α_i都相等时才是`Σ_{i=0}^K α^i`；α=1单独返回K+1，α=0返回1。用户输入`mean`只保证一阶矩，不唯一决定P(A=K)：K2时N恒2与N1/3各半均值都是2，Pfull却是0和0.5。将mean反解几何α可以提供，但必须额外选择「截断几何假设」，不能默默改变输入含义。

标准autoregressive草稿、MTP、DSpark与DFlash不能只靠S/M/K三个字段统一。DSpark参考实现将block backbone与顺序Markov sampling分离，具有target hidden依赖和confidence截断；DeepSeek-V4.1路径还复用目标架构与checkpoint中的mtp前缀权重。层数、重复调用次数、串行采样步数是三个概念。当前新仓把DSpark等效成K次完整MoE草稿，只可保留为改名后的假想方法。

维护`RoundState=(committed_prefix, target_cache_prefix, draft_cache_prefix, speculative_slots, pending_messages, rng_state)`。transition生成下一次各forward真正需要的输入。某些简单链式实现全接受后需要处理2个token、拒绝后1个；padding/重算/EAGLE/块草稿可能不同。通用层不硬编码catch-up=2或`1+α^K`。KV回退成本取决于block管理与同步路径，不能固定1–3μs作为事实。

EOS、max_new_tokens尾轮、动态K都截断N和状态；均值round不能展示为真的「提交3.05 token」事件。可显示解析期望，或采样整数事件并报告统计分布。

## 8. 从轮次到请求，避免错用max和平均值

对合法串行协议：`T_round=T_draft+T_D2T+T_verify+T_T2D+T_transition`。平稳、可更新的长期token平均时间为`E[T_round]/E[N]`，通常不等于`E[T_round/N]`。有限请求还要计首轮、prefill、末轮截断及drain。

两个独立微批可以交错，使稳态服务周期接近`max(T_draft+T_D2T,T_verify)`，但单请求依赖仍是草稿→验证→反馈→下一轮；该周期是吞吐解释，不直接等于该请求TPOT。单请求超前提议必须显式引入假设分支、hit/miss、取消、回退和浪费资源；不能靠一个β加权式假装已构造可执行schedule。

事件schema至少包含`event_id/request_id/round_id/stage_id/resource_id/dependency_ids/state_before/state_after`；非预知策略只能读取当前事件时刻已可见的信息。同一资源允许并发时须定义份额/竞争函数，无此模型按互斥处理。

指标命名：TTFT含排队和首token；每请求token间隔与稳态TPOT区别；throughput按全系统完成token/墙钟；goodput须带SLO；p50/p95/p99基于真实采样请求或事件轨迹及其样本数。随机模型假设可生成模拟分位数，但不是测量分位数。

## 9. 敏感度与比较资格

任何方案排序先检查模型/方法/负载/精度/设备预算与目标是否可比；同目标8GPU再外加1GPU的DT方案可研究，但必须同时报告资源增量。质量等价/目标分布一致与性能计数是不同验证轴。

对未测的效率、复用、协议延迟、接受分布，先扫描合理范围并展示结论反转点。所有区间必须注明是假设范围、统计置信区间还是严格界。只有上下界可证明时才称上下界；不把任意滑条范围叫可信区间。没有可行性和有效成本模型的行保留结构证据，不参与收益赢家判定。


---
# FILE: prototype/reuse-map.md

# 原 HTML 复用对照

来源：`C:/Users/41693/Downloads/simverse_v3_0.html`。完整原样副本为 `reference-v3.html`。主工作台使用原 CSS、工作台 DOM 和 JS 交互骨架，不是仅参考配色重画。

| 原版元素 | 新版位置与处理 |
|---|---|
| 顶栏与导航 | 保留品牌和样式，连接本次计划、原理、裁决、教学样例、原版资料 |
| 模型/方法/部署/ISL/α/特性选择 | 全部保留；切换后联动展示与证据提示 |
| Target / Draft Recipe | 保留标签及布局，目标模型标签随选择更新 |
| 七项 KPI | 全部保留；显存未知不伪装成可部署；流水 TPOT 与加速比待验证 |
| 部署 × 方法矩阵 | 保留 25 个可点击单元格；流水行留位置，数值标为待验证 |
| 角色与算子 DAG | 保留分组、节点和选中样式；模型相关标签修正，明确仅为示意分组 |
| 算子七维详情 | 保留时延、瓶颈、工作量、权重、激活、精度、公式信息结构；增加展示样本提示 |
| Roofline SVG | 保留原图形和算子联动；设备曲线及算子数字均为展示输入 |
| Draft / DT / Target 时间轴 | 保留非流水演示与阶段筛选；流水改为缺口解释与筛选按钮；baseline 仅保留 Target |
| 原版四页资料 | 完整存档可打开；审查结论以新评审包为准 |

修正：原版 DSpark 预设更名为链式 MoE 代理，避免将链式生成误称原生块前向；Catch-up 的固定 M=2 不再作为通用规则；Qwen3-Next 与 V4.1 的算子名称分别显示；场景导出明确标注 `metrics_are_verified:false`。

正式开发时沿用这些界面组件与信息结构，用后端 ResultBundle 替换展示数字。每一项指标需要值、单位、口径、证据等级、适用条件与来源；未知状态必须可见。不能把原版演示公式直接认定为生产模型。

本方案已纳入 [重构计划第 8 节](../plan.md)。各页面/指标的后端产物、口径和验证步骤以 [UI 数据与验证契约](../ui-data-contract.md) 为准；待决策统一在 [questions.md](../questions.md)。
