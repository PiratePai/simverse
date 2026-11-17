# 成本路径与默认粒度简化

2026-11-18；数值基线为 `75c048990b51dc73b7bfb8e6a88989225b61fae4`。

本轮把解析、条件记录和历史 profile 的选用成本接到同一个 `CostSession`，移除记录模式下
第二次完整求值。模型计数、流量、lowering、容量与解析定价各执行一次；只有选用阶段成本
发生变化时，才复用同一执行函数重新计算串行轮次和流水事件。没有新增融合假设或硬件测量。

## 成本组与细粒度账本

默认展示 `cost.groups`，按阶段、角色和成本类别归组；通信继续按原语分开。每个组列出
拥有的计价项、解析/选用耗时、来源、记录引用与回退原因。解析组是原有逐项成本的和，
没有把多个 kernel 合成一个 roofline，也没有删除小算子的 FLOPs/字节计数。

历史 H200 profile 的 `attn_pre_proj` 等真实融合组直接保留组总耗时。例如归档 30B-tiny
的投影组拥有 240 行，耗时 5.6256 ms；不将其平均摊成 240 条逐行实测。未覆盖行继续
解析回退，原采样区间间隙独立列出。历史 provider 的六位小数舍入保留，组和与阶段值的
允许差为 1e-6 ms。未知运行时组耗时为 null，不并入已计价覆盖分母。

原始计数表和逐行成本检查器改为默认折叠，仍可展开、定位和导出。`--groups-csv` 与
工作台的“导出成本组 CSV”提供组级导出；原 `--csv` 保留细粒度诊断用途。融合 profile
的逐行表仍是解析值，不冒称测得的融合组内分配。

## 一个选用口径

`cost.selection` 统一给出来源、阶段成本、基线、执行轮次与流水事件；KPI、矩阵、默认
扫描、时间轴和 trace 导出使用相同选用结果。规范指标为 `M01_tpot_selected`、
`M01_baseline_selected`、`M02_speedup_selected`、`M04_round_selected`；原 record/profile
名称保留为兼容视图，纯解析指标保留供对照。`M18_selected_cohort_period_ms` 单列选用
成本下的流水周期，原 M18 继续表示解析流水；周期不是 TPOT。

记录档零命中时 `requested_source` 保留请求，实际 `source=analytic`、`applied=false`，
不生成条件记录 KPI。目标/草稿身份、TP/EP、dtype、形状、通信原语、域外拒绝和 unknown
传播沿用原门禁；R28/R29 仍互斥，不能跨栈拼接。成本组覆盖不等于独立留出或 E2E 验证。

## 回归与协作证据

- [15 场景基线](../evidence/cost-groups/baseline.json)在生产代码改动前捕获，
  `py -3.12 tools/replay_cost_groups.py` 检查原数值指标、Work 与容量摘要不变。
  覆盖默认 Qwen、V4.1、DFlash、MTP、DT 串行/流水、未知接受律、R28/R29 与历史 profile。
- `tests/test_cost_groups.py` 共 11 项：单次计数/定价、组所有权不交叉、无任意融合、
  融合组不向行分摊、零命中、未知传播、DT 边界、API/CLI/trace 与冻结回放。
- R27 冻结 JSON 不改写。其检查排除明确列出的新增命名空间和随提交变化的运行身份，
  同时独立校验新 manifest 的 run ID，原数值、归因和计数仍严格比较。
- 用户授权后 DSH 通过 native adapter 做只读独立核对，见[审查快照](../evidence/cost-groups/dsh-review.md)
  与[回执](../evidence/cost-groups/dsh-receipt.json)。其报告针对改动中快照；只读沙箱限制了
  临时目录测试，不作为最终全量验收。主控已补齐其提出的 R27 兼容、全回退语义、融合组
  和选用流水周期披露，并负责最终回归。

## 后续计划与审查取舍

本轮全量测试 580 项通过（24.95s），JavaScript 静态检查、Ascend 七场景回放通过。
浏览器验证默认解析成本组、折叠细粒度账本、R29 TP2 命中与 TP8 全解析回退；下载的 7 组
CSV 与同场景 API/CLI 导出逐字段相同，无控制台错误。界面 detector 只报告既有 11px
正文告警；本轮未扩展到全工作台字体重构。

先补已有 trace 的原文件哈希、调用/阶段/角色关联和组边界，能恢复元数据时不重采。
然后按当前组的回退原因补真实 MoE/GDN/attention 与必要通信，不为每条逻辑行另建计时核。
默认 H100 TP8 中假设链路启动时间占比很大，需同栈测量再调整；本轮不修改该常数。

PR #26 的融合组不必再拆成逐行耗时，但原始 kernel/step 数据哈希缺失、K/N 与 dtype
聚合缺少 M/阶段关联、GroupedMoE 与 KV-write 未分配等问题仍阻止合入。#27 的官方 dense
规格与实际设备指纹分别推进；#28 优先来源；#29 明确投影融合组；#30 建立独占开销边界；
#31 只保留离线归档，不扩展活跃模型目录。详见昇腾计划。

方案复用已阅读的成熟实现，不引入框架依赖：[Vidur](https://github.com/microsoft/vidur)
`abae7f63aa857300f5cdc6f5e0d27860cd24721b` 的组件成本组合，及
[llm-analysis](https://github.com/cli99/llm-analysis)
`d841e40aec8c84b9e76ba9cfbce67418adb172bd` 的模块粒度组织。后者维护较早，仅参考组织方式；
本轮未运行这些上游项目。真实缓存、状态、通信、容量和串行/流水边界继续按本仓契约保留。
