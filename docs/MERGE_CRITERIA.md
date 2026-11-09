# Simverse 合入准则（长期规范）

版本：1.0（2026-11-09 建立）。本文件是所有合入 new-viewer/simverse 的**强制性准则**，
与 [重构计划](rebuild-20261106/plan.md)、[原理底座](rebuild-20261106/principles.md)、
[UI 数据契约](rebuild-20261106/ui-data-contract.md) 配套：那些文件定义目标架构，本文件定义
每一条改动的**准入与披露义务**。冲突时以用户最新决策为准并在此登记。

自动检查：`tests/test_merge_gate.py` 把可机械化的规则固化为测试（见 §5）；不能机械化的
规则以代码评审核对。**修改本文件需在合入说明中单列原因。**

---

## 1. 模型与仿真语义必须与实际 forward 源码核对

1. **身份冻结**：进入 `assets/config` 的每个模型必须固定 repo/revision/config SHA256
   （`assets/identity`），并在 `model_ir` 携带 `SourceIdentity`。禁止"按名称猜测"或用
   相似模型替代必需场景。
2. **映射链**：源码分支/张量形状/层序/状态与缓存依赖 → ModelIR（0.3.0 schema）→
   count/traffic/memory 账本 → cost/execute。每一环可从 run 的 manifest/evidence_refs
   追溯到上一环。
3. **HF 语义 ≠ 推理后端实现**：融合、量化、分片、通信、缓存配方必须对应**具体实现**
   （某后端、某版本、某 kernel 路径），并登记在该 recipe 的 `implementation_id`；
   没有实现证据的配方只能作为显式研究假设建模，不得标记为已部署验证。
4. **配置字段覆盖**：每个 config 键登记 `mapped / metadata-only / ignored-with-reason /
   unsupported-semantic`（ModelIR.coverage）。`unsupported-semantic` 使
   `structure_complete=False`，受影响指标降级并传播。
5. **验证分层标记**：源码阅读（source-checked）、CPU 执行验证（shape-checked/数值）、
   GPU 实测（measured）分别标记，**不能相互冒充**。小尺寸 CPU forward/独立数学验证
   能做必须做；不能做的在缺口清单登记，不得因此把结论标成已验证。
6. **profile 的算子所有权**：校准映射必须写明被测算子拥有哪些账本行（含融合边界，
   如 fused QKV+QKNorm、fused add+norm），并给出采集代码位置作为证据。

## 2. 数值口径统一

1. **单位与计数**：每条数值带单位；FMA 计数约定固定（1 乘加 = 2 FLOPs）；dense/sparse
   峰值不得混用——`peak_by_dtype` 一律 **dense**，sparse 值只允许出现在 source 说明里。
2. **范围**：单卡 vs 整组、单向 vs 双向带宽、标称容量 vs 可分配预算必须显式。链路 bw
   为单向；聚合/双向值必须换算并注明。`DeviceBudget.allocatable_bytes=None` 时 feasible
   判定必须保持 unknown（§7.4），不得用标称容量冒充。
3. **量的类别**：确定性工作量（精确整数）、期望流量（浮点，禁止取整冒充精确）、存储
   容量、实际耗时（estimate/measurement）四类不得混算；同源对账只证明迁移一致，
   **不构成独立正确性或真实性证据**。
4. **计时边界**：device kernel、CUDA event、host launch、graph piece、端到端必须分栏
   （`TIMING_BOUNDARIES`）；不同边界的数字不可互相替换或直接相减。
5. **设备规格来源**：devices 注册表每个数值带 `spec:`（官方规格，注明 dense/sparse 与
   单向/双向）或 `assumed:` 前缀；禁止无来源数值。修正口径必须保留前后对比与依据链接。

## 3. profile 与校准的来源、适用域和边界

1. **来源分级**：measured / synthetic / assumed / analytic 四级，落在 ProfileRecord
   `source_kind` 与 CostEstimate `cost_source`；synthetic 永不升格为 measured。
2. **测量元数据**：设备、驱动、软件栈、实现/recipe/版本、shape、dtype、布局、TP/EP、
   测量方法（计时边界、warmup、重复次数、sync 策略）、样本统计（min/max/mean/median/
   std/count）与原始数据引用必须齐备。**元数据缺失不得编造**；缺关键字段时该记录
   不得进入 measured 准入（拒绝或降级，写明原因）。
3. **校准只改成本**：校准/查表/拟合只作用于 CostEstimate.selected，**永不修改**
   Work/Traffic/Memory 原始账本。换成本前后的固定场景重放必须证明原始计数逐字段不变。
4. **适用域**：CalibrationModel 显式登记覆盖域（设备/精度/TP/后端/shape 范围）；域内
   用最简单可满足数据的查表/插值，域外**显式回退解析**并保留 analytic_base、selected、
   fallback_reason 三列。禁止外推冒充域内。
5. **禁止全局倍率**：不得用单一全局校正系数掩盖结构错误；不得对 CPU/通信/kernel 开销
   重复计费（每笔开销只归属一个所有者）。
6. **验证独立性**：拟合/插值必须在留出划分（holdout，按记录键去重，同 shape 不得同时
   入训练与留出）上报告误差分布（中位/p90/max）与覆盖量；数据不足时如实声明限制，
   不得用训练集拟合误差替代。

## 4. 验收与结果披露

1. **按变更范围验收**：独立 oracle、守恒检查、负例、来源身份、覆盖域、留出误差、
   UI/CLI/导出一致性——覆盖范围与改动面匹配，不为小改动要求全量、也不为大改动放水。
2. **缺口传播**：无数据、语义缺失、域外、容量未知必须传播到受影响指标的
   support/feasibility/reason 与比较资格，不得静默消失。
3. **禁止为通过而通过**：不得通过调常数、放宽误差门槛、把 synthetic 升格为 measured、
   或改写期望值来让测试变绿。已知错误变异未被测试抓住 → 对应门槛不通过。
4. **合入记录**必须说明：数据/代码来源与身份、适用域、验证方式（源码核对/测试运行/
   已有实测数据验证/本次新做硬件测量，四者严格区分）、结果变化原因、剩余限制。
5. **状态文档诚实**：docs/IMPLEMENTATION_STATUS.md 的测试数必须与实际收集数一致
   （test_merge_gate 自动核对）；过时描述随改动同步更新。

## 5. 自动检查（tests/test_merge_gate.py）

| 检查 | 规则来源 | 内容 |
|---|---|---|
| G1 设备来源完备 | §2.5 | DEVICES/LINKS 每个数值字段有 source；峰值 source 含 dense/sparse 口径；链路 source 含方向 |
| G2 dense 口径 | §2.1 | peak_by_dtype 数值必须与其 source 的 dense 口径一致（抽查双档 2× 关系） |
| G3 profile 清单完整 | §3.2 | assets/profiles/** 清单的源文件 SHA256 与蒸馏产物声明的哈希一致 |
| G4 measured 准入 | §3.1/3.2 | source_kind=measured 的 ProfileRecord 缺环境元数据即拒绝 |
| G5 无倍率校准 | §3.5 | 校准实现不得暴露"单一全局倍率"入口（接口形态检查） |
| G6 状态文档测试数 | §4.5 | IMPLEMENTATION_STATUS.md 中 "N passed" 与 pytest 收集数一致 |
| G7 计数不可变 | §3.3 | 固定重放场景：开/关校准，work/traffic/memory 账本逐字段相等 |

新增可机械化规则时在此登记条目；无法机械化的规则（如 §1.5 分层标记）在评审清单中核对。
