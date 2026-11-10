# 重点模型与功能状态（2026-11-10）

本轮产品目录收敛为已固定身份的 Qwen3.6-35B-A3B + DFlash、DeepSeek-V4.1-Flash。用户已确认 Qwen3.5 为笔误，重点模型为 Qwen3.6-35B-A3B，不新增或替换模型身份。历史模型与 H200 数据保留作内部校准和回归，不继续扩展产品模型数量。

2026-11-10 A 组轮（R01–R03b）：TP/EP per-rank 统一配方 + 同卡对象身份
账本 + 生命周期峰值 + 预算契约与四态判定全部落地（详见
[validation/agroup-tp-ep-colocated-20261110.md](validation/agroup-tp-ep-colocated-20261110.md)）。

## 已实现，但不等于真实部署已验证

| 能力 | 范围与限制 |
|---|---|
| 身份、ModelIR、配置覆盖与逐算子账本 | Qwen3.6 文本结构；V4.1 部分结构，Engram 仍缺失 |
| Qwen3.6+DFlash | pinned 草稿、接受律、draft/verify 解析成本、同卡与 DT 串行；非实测收益 |
| **per-rank TP/EP 配方（A组 R01）** | vLLM @dc74b8cb pin（FusedMoE inter/tp 或 E/ep 整专家、router 复制、KV 头切/复制、GDN 切分、词表行切、DFlash 同 TP 组共享 embed/lm_head）；ep∉{1,tp} 显式拒绝 |
| **同卡对象账本（R02a/b）** | 物理 allocation 身份 + consumer 引用（草稿共享 target embed/lm_head 零额外驻留）；生命周期分列（常驻/每请求/跨轮/阶段临时）+ 峰值取阶段 max；workspace/graph/NCCL 池为缺口组件传播（不填 0） |
| **预算契约与判定（R03a/b）** | nominal/allocatable(measured/hypothesis)/额外开销分列；四态判定 unknown/oom/feasible/conditional；已知下界超标称即 OOM；假设预算不进实际部署排名 |
| mHC | 已有源码边界映射和计数测试；Sinkhorn 操作数仍近似 |
| DT 流水事件核 | 微批依赖、资源互斥、周期和利用率；非完整在线服务 |
| 接受律与比较 | 显式假设、未知传播、资格门禁；不从平均接受长度推造完整概率律 |
| CLI/Web/导出/扫描 | 共用生产结果、精度选择、EP 选择、预算输入、独立运行身份 |
| H200 校准基础设施 | 历史 Qwen3 数据的部分算子组合估计，不能覆盖重点模型 |

## 待完成工程

1. ~~Qwen3.6 优先：统一 MoE per-rank TP/EP 配方~~（2026-11-10 A组 R01 完成）。
2. ~~同卡显存复核（草稿 batch/共享去重/KV 分片/生命周期峰值分列）~~（A组 R02a/b 完成；运行时池尺寸证据仍缺，见下）。
3. 可分配预算：契约已就绪（R03a），**真实 measured 数据仍缺**（G-D7）；解除前默认场景保持 unknown。
4. V4.1：获取并固定 Engram 实现、补查表/投影/参数/容量/跨层依赖；完整前不解除结构门禁。
5. 请求级 lookahead、取消与浪费；在线到达/组批/抢占、KV 分页、Chunked Prefill/PD/Prefix Cache。
6. 235B moe.csv TP8/EP1 准入（G-D2）：分片账本口径已对齐（R01），蒸馏+留出验收待做。

## 待验证或需额外数据

- 重点模型小尺寸 forward 与计数的独立执行对照。
- Qwen3.6+DFlash 实际接受分布、draft/verify 延迟及服务栈配对验证。
- 重点模型各精度/TP 的 GDN、MoE、attention 测量；历史 H200 profile 不能按名称相似迁移。
- V4.1 Engram 与 mHC kernel 数值和操作边界。
- 实际可分配显存（G-D7）、非 KV 运行时开销（workspace/CUDA graph/NCCL 池尺寸，当前按缺口传播）、匹配 E2E 请求轨迹。
- 浏览器交互与像素复核：2026-11-10 A组轮已实测 EP 选择/预算输入/运行链路
  （截图 docs/evidence/workbench_ep_budget_run.png）；其余页面沿此前口径。

## 本轮修复（2026-11-10 A 组）

- **TP/EP 配方统一（R01/G-C11 解除）**：count 的 MoE 行、memory 分配、
  EP 通信三方改用同一 ParallelPlan（vLLM pin）；EP 显式场景字段（影响
  运行身份）；非法组合显式拒绝并列原因。
- **同卡账本修正（R02a）**：废除草稿幻影 embed/lm_head 全量副本（≈1.017GB
  错误驻留）与去重键失配双重计费；embed/lm_head 词表行切；GDN 权重/状态、
  草稿 KV per-rank；attention 投影权重补乘输入宽 h（旧式漏乘 h 倍）；
  草稿批=目标批（修复硬编码 1）。
- **计数补齐**：final norm、草稿 q/k norm、rope、hidden_norm 缺行（对
  pinned dflash/HF 源码逐行核对后补）；TP 集合通信从"整模型 2 次"修正为
  "每层 2 次"（旧代码漏计 40×）。
- **判定链路（R03b）**：M06 指标携带 known_peak/steady/峰值完整性；
  假设预算 → feasible-conditional 且不参与排名。

默认 ScenarioV2 的同卡 per-rank 稳态账本为 2,827,423,168 B（steady）+
2,135,746 B（结构化临时峰值）= 2,829,558,914 B（known_peak，TP8/EP1、
B1 ctx32k、W8A8+FP4 专家）；这不是用户所有场景的峰值，且不完整
（workspace/graph/NCCL 池未计、非 0），不能据此证明可部署，也未超
80GB 标称不构成 OOM。可分配预算（G-D7）与运行时池证据仍是阻塞项。
