# Simverse Roadmap

更新：2026-11-10。产品范围固定为 **Qwen3.6-35B-A3B + DFlash** 与 **DeepSeek-V4.1-Flash**。历史模型仅用于内部回归与校准，不继续扩展产品目录。以下为待办，不代表已交付或交付日期承诺。

## 当前已交付

- [x] 重点模型产品目录、DFlash 配对门禁、具体资格失败原因。
- [x] ModelIR、计数/成本/执行分层，CLI/Web/导出共用结果。
- [x] Qwen3.6+DFlash 解析实验、DT 串行及初步微批流水。
- [x] V4.1 部分结构及 mHC 边界计数；Engram 仍不完整。
- [x] 历史 H200 部分算子校准、来源与覆盖域门禁；不代表重点模型实测。

## P0：重点模型语义与容量正确性

| ID | 待办 | 验收标准 | 依赖 |
|---|---|---|---|
| R01 | Qwen MoE per-rank TP/EP 工作量、容量、通信统一 | 固定实现配方；逐层及跨 rank 守恒；独立 oracle；旧数值变化逐项归因 | 源码配方核对 |
| R02 | Qwen+DFlash 同卡显存完整性 | 共享 embedding/lm_head 去重、草稿 batch、KV 分片测试；verify/workspace/graph 生命周期与峰值分列 | R01、forward/后端证据 |
| R03 | 有来源的可分配预算输入 | nominal/allocatable/额外开销独立；unknown/oom/feasible 负例；UI/CLI 同口径，假设预算不能冒充实测 | R02、实际预算数据 |
| R04 | V4.1 Engram 语义补齐 | 固定源码身份；查表/投影/权重/容量/跨层依赖映射；独立小尺寸验证；仅解除已验证缺口 | Engram 实现源码 |
| R05 | V4.1 mHC/Sinkhorn 复核 | kernel 分支、迭代与计数对照；精确值与近似分列 | 固定 kernel 源码 |

## P1：重点模型验证与实测闭环

- [ ] R06：重点模型小尺寸 forward 执行对照；源码阅读、CPU 执行、硬件实测分层标记。
- [ ] R07：采集/接入 Qwen3.6+DFlash 接受分布、draft/verify 时延与运行环境；核对实际服务栈配对。
- [ ] R08：重点模型 GDN/MoE/attention 的 shape、精度、TP 覆盖及独立留出验证，域外回退。
- [ ] R09：匹配模型/环境/负载/指标边界的 E2E 对照；没有 E2E 证据不得将组合估计标为整步实测。
- [ ] R10：工作台重点场景浏览器交互、错误提示、窄屏与导出验证；修复测试子进程编码警告。

R07–R09 需要额外数据，不以历史 H200 数据或合成数据替代。采集字段与阻塞详见 [数据缺口](docs/gaps/data-gaps.md)。本 roadmap 不授权启动真实模型/GPU 采集。

## P2：请求级系统实验

- [ ] R11：lookahead 分支、取消、回滚、浪费与资源占用账。
- [ ] R12：请求到达、动态组批、抢占、KV 分页/回收及生命周期峰值。
- [ ] R13：Chunked Prefill、PD 分离、Prefix Cache，逐项增加源码映射和可复现场景。
- [ ] R14：请求级 TTFT/ITL/吞吐/goodput 与 SLO 端到端验收。

R11–R14 在 P0 底座稳定后推进；不得将流水周期当作单请求延迟。

## 合入要求与状态入口

每项完成必须附源码版本/哈希、影响域、测试命令和结果、数值变化、剩余限制。校准不改变原始计数；缺失证据继续传播 unknown/unsupported，不为参与排名补造数值。

- [强制合入准则](docs/MERGE_CRITERIA.md)
- [功能状态与限制](docs/FOCUS_STATUS.md)
- [实施状态](docs/IMPLEMENTATION_STATUS.md)
- [工程缺口](docs/gaps/capability-gaps.md)

优先级以本 roadmap 为准：旧缺口文档中“继续扩展 235B”等建议不再是产品优先事项。更新完成状态时同步这些入口，避免把“已有测试”误写成“已完成部署验证”。
