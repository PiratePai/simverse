# Simverse 合入准则（长期规范）

版本：1.1（2026-11-09 建立；2026-11-10 增补两模型证据分层/配置覆盖/CI 门禁）。
本文件是所有合入 new-viewer/simverse 的**强制性准则**，
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
7. **完整 config 覆盖**：必须展平完整 config（含 `text_config`/`vision_config`/
   `quantization_config`/`rope_parameters`/`dflash_config` 等嵌套叶键）逐键登记。
   未知键**不得默认 ignored**；不能映射的键登记 `unsupported-semantic` 并进 gaps。
   视觉塔等文本 scope 外字段可登记 `metadata-only`，但必须写明 scope 与未执行事实。
8. **证据四栏分立**：config 字段映射、forward 源码阅读、CPU forward 执行、
   GPU/部署实测必须分别记录，禁止相互冒充。`structure_complete` 只表示
   config 覆盖 + forward 源码证据同时完备；CPU/GPU 结论另有字段/能力检查。
   config 通过、但 MTP/Engram/CSA2/MoE 等关键路径未核时，不得宣称完整或进入
   实际部署排名。
9. **活跃目录边界**：运行时 catalog/CLI/API/工作台只保留 `deepseek_v41_flash`、
   `qwen36_35b_a3b` 两款主模型及与 Qwen3.6 绑定的 DFlash DraftIR。tiny 合成模型
   只能作内部测试 fixture；历史模型只能作离线归档回归，不得出现在产品/部署选择。
   历史 profile 必须绑定采集模型；跨模型使用直接拒绝。

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
| G6 状态文档测试数 | §4.5 | IMPLEMENTATION_STATUS.md 的 `pytest 收集数：N` 与实际收集数一致；全量通过由验收 owner 记录 |
| G7 计数不可变 | §3.3 | 固定重放场景：开/关校准，work/traffic/memory 账本逐字段相等；归档模型走显式注入 |
| G8 活跃目录边界 | §1.9 | catalog 恰两款主模型 + DFlash；历史/tiny 不可运行；API 不暴露归档 profile |
| G9 完整 config 覆盖 | §1.7 | 嵌套叶键全登记；未知键 unsupported+gap；无默认 ignored |
| G10 证据分栏 | §1.8 | config/forward/CPU/GPU 四栏独立；未执行/未实测不得标 executed/measured |
| G11 数值绝对 oracle | §2.1 | V4.1 o_a 分组在 TP1/2/4/8 的 FLOPs/容量绝对期望 |
| G12 shared gate 完整性 | §1.2 | Qwen3.6 每 MoE 层有 gate 行与 h 容量 |
| G13 草稿批口径 | §2.1/2.3 | B>1 时 token 维恰乘一次；权重/通信不重复乘 |
| G14 证据引用 | §1.2/§4.4 | run refs ⊆ registry 且按实际 model/draft/profile 生成 |
| G15 部署排名阻断 | §4.2/§1.8 | 缺 forward/CPU/GPU/预算证据时 M02 与比较器不排名 |
| G16 source_map 校正 | §1.2 | 行号覆盖实际 gate 分支；未核路径显式登记 |
| G17 CI 工作流 | §6 | PR/push/merge_group 不按路径跳过；Windows/Node/Python 3.12 明确 |

新增可机械化规则时在此登记条目；无法机械化的规则（如 §1.8 的“证据是否足够”
判断）在评审清单中核对。

## 6. GitHub 合入强制（仓库规则，不是工作流文件本身）

本仓提供 `.github/workflows/merge-gate.yml`：`pull_request`、`push`、`merge_group`
均触发，不做路径过滤；Windows runner 安装 Node 20，Python 3.12 下执行
`node --check simverse/static/workbench.js` 与 `python -m pytest tests -q`。
**工作流文件不能强制审查或 required check**：仓库管理员必须在
ruleset/branch protection 中把该 job 的 check 设为 required，并启用
“Require a pull request before merging / required approvals”。
参考 GitHub 官方文档（2026-11-10 查阅，未在远端配置）：
[protected branches](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches)、
[branch protection rule](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/managing-a-branch-protection-rule)、
[rulesets](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/about-rulesets)、
[merge_group event](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#merge_group)。
历史（2026-11-10 门禁轮记录时）：未推送、未配置远端保护。私有仓库时期
外层 Codex 只读查询 main branch protection 与 rulesets API，均得到 HTTP 403：
`Upgrade to GitHub Pro or make this repository public to enable this feature.`
当前（2026-11-10 文档收口时）：仓库已公开，候选分支已推送并建 PR #1；
main branch protection 查询返回 404 `Branch not protected`，rulesets 为空。
保护规则仍未配置，不能把本工作流视为 GitHub 强制 required check/PR review；
合入审查按本文件与本地/CI 结果执行，远端强制能力待仓库管理员配置后再核。
