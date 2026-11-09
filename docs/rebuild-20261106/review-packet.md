# 重构方案初稿与独立质询任务

日期：2026-11-06。此文件是送审初稿，不是最终结论。

## 用户目标

新建统一推理仿真仓库，主要用于研究新模型架构/系统方案，其次解释模型与硬件瓶颈。两个旧仓库独立维护，已有模型结构共享包。用户要求综合三份资料、分别请 DSH 与 ZCode 提问查漏补缺，再完善计划、计算原理和 UI 示例。

## 本次可读材料

- `D:/Github/new-viewer/docs/verification_report.md`
- `D:/Github/new-viewer/simverse/`、`tests/`、`golden/`
- `D:/Github/new-viewer/docs/rebuild-20261106/references/model-verse-principles-pinned.md`：用户指定 Gitee 提交 360817fbda025d9c1d148199acab446e722a7b11 的原文，经 git fetch/show 获取；不是当前 HEAD。
- `D:/Github/new-viewer/docs/rebuild-20261106/references/simverse-v3-text.txt`：用户 HTML 的正文抽取；原文件 `C:/Users/41693/Downloads/simverse_v3_0.html`。
- `references/DeepSpec/`：官方 005e03b81cec38b7da6399833d609ee89a2587f2 的 DSpark source；`references/vidur/`：官方 abae7f63aa857300f5cdc6f5e0d27860cd24721b；`references/vllm/`：官方 f358f0578a83f4e1fe6fc49a492d3ae9362de8e5。这些外部代码只阅读，未运行。
- `D:/Github/model-verse/docs/simulator-rebuild-plan.md`、`simulator-rebuild-questions.md`；Q2 用户已选 C 为首、A 为次。其他待决策不冒充确认。

## 主控当前判断（请挑战）

1. 保留 Count→Cost→Execute 分层与统一数据出口，但在 Count 前加模型语义、执行 lowering；把逻辑量、期望流量、物理布局、容量、时间区分。精确整数不能靠把期望值 int() 来得到。
2. MVP 应为自定义 target/draft 的可解释实验工作台；tiny 独立锚点 + 首发 V4.1 Flash 的证据闸门。不能因模型新而跳过 forward/capability 审核；结构未知须明确阻断相关结论。
3. 同卡/DT 串行先跑闭环，再做有请求/轮次ID的资源约束事件核。微批稳态吞吐周期不等于单请求 TPOT；不能把 max(D,V) 当同请求依赖合法的时间线。
4. shared ModelIR 是模型语义唯一来源；新仓拥有 lowering/count/cost/execute；旧仓经适配器消费共享版本，逐场景迁移。拒绝复制第三份公式。
5. 每个结果同时携带 support、evidence、feasibility、scope、单位、公式与输入依赖。证据维度不混成一个可信度百分数。
6. 计划：M0 冻结证据/红灯反例；M1 语义+计数+最小UI；M2 成本/双向消息/容量/串行投机；M3 资源事件核；M4 serving与实测校准。每阶段明确验收、非目标、回退与交付。
7. UI 沿用材料的联动工作台概念：场景/部署比较、模型层图、选中算子明细、带 request/round 标识的时间线同屏；收益旁边显示缺失假设和不可用原因。示例数据不冒充硬件实测。

## 已做检查与疑点（不保证每个判断都对）

- new-viewer 原有 pytest：13 passed，1 个 UTF-8 子进程输出解码 warning。
- report 明确 TP allreduce 未计、草稿 attention/embed 未展开、校准类透传；13 tests 只保证其定义下的一致性。
- 官方 DeepSpec draft_ops.py 有 forward_dspark_draft_block、Markov sampling、confidence prefix；当前 dspark S=K/M1 MoE 预设可能根本不是官方算法，应撤销真实性声称。
- MoE top-k 每 token 无放回、跨 token 独立均匀时 E[U]=E*(1-(1-k/E)^M)。当前 E*(1-(1-1/E)^(M*k)) 是有放回近似。E384,k6,M5 时并集必不超过30；report 写38.9可疑。源码注释里的组合式亦要核实。
- HTML 把 catchup_tokens=2、KV前缀只读一次、双向 hidden 必传当通用定律；需要按具体 proposer、kernel、protocol 验证。
- alpha 或均值不足以描述路径相关回退、流水线尾延迟。K2/均值2：N恒2 和 N=1/3各半，同均值但 Pfull 分别0/0.5。
- residual FLOPs 被记0；H100无FP4算力不自动等于所有weight-only kernel用FP8；FP16 activation硬件单元不能由bit数统一判。
- 内存需要同卡target+draft+workspace+activation+KV+block对齐/lifetime；当前仅粗权重+目标KV。
- pinned原理文档的512专家/top10/100tokens约18%算术错（公式实际约86%）；三点拟合自身公式不是物理校准；倍数不必然比绝对值可信。
- Vidur代码拟合median与缓存预测，不应笼统描述成每kernel随机采样时延分布。

## 请独立交付

只读审查，不修改任何文件，不调用其他Agent，不运行真机/模型。用中文输出：
1. 8–12个必须追问主控的具体问题，按P0/P1/P2排序。每个包含被挑战的假设、最小反例/源码锚点、建议解决方式/验收。
2. 指出主控以上判断中可能错误或过度的至少两处（没有就明确没有证据，别凑数）。
3. 建议修改的计划顺序、原理契约和UI误导防护；只将确需用户偏好/资源决策的问题留给用户。
4. 说明实际阅读/运行了什么，哪些只是推断。无需写一份大而全的新方案，目标是质询查漏。
