结论：这版契约已经覆盖了“未知不填零、仿真不冒充实测、按 rank 验容量、解析/事件分型”等大部分硬要求；但仍有一组不会报错、却会让结果**不可比较、不可重放或不可归因**的口径缺口。建议把 D2-01/02/03/05/06/08/10 作为 M1–M2 的必改项，D2-04/07/09 至少要在对应页面或指标上线前补进 schema。

范围：仅交叉核对内嵌的 `plan.md`、`ui-data-contract.md`、`questions.md`、`principles.md`、`prototype/reuse-map.md`；未编辑文件、未运行模型/测试、未委派 Agent。以下不把“尚未实现”当缺陷，只审契约与验证口径。

| ID | 严重度 | 问题 | 处置 | 用户决策 |
|---|---|---|---|---|
| D2-01 | 高 | 比较资格与资源增量规则未定义 | 必须，M2 比较验收前 | Q4/Q11 |
| D2-02 | 高 | 单一种子不足以支持并发/顺序变化下的重放 | 必须，M1 采样/事件前 | 否 |
| D2-03 | 高 | 矩阵颜色与主排序 TPOT 变体未绑定 | 必须，M2 矩阵启用前 | Q11 |
| D2-04 | 中 | 过滤后的聚合范围未定义 | 必须，M1 UI 接入前 | 否 |
| D2-05 | 高 | “容量上限”未区分标称/可用显存 | 必须，M2 容量验收前 | Q2/Q7 |
| D2-06 | 高 | MoE capacity factor / token drop 未进 schema | 必须，M1 MoE 计数前 | 部分（首发范围） |
| D2-07 | 中高 | 分位数单位与相关性重放未定义 | 必须，M4 尾延迟前 | Q8 |
| D2-08 | 中 | V07 只验已声明依赖，未验“不可预知” | 必须，M2 事件校验前 | 否 |
| D2-09 | 中 | M22/M23 无明确页面归属 | 必须，M1 UI 映射前 | 仅 Q9 |
| D2-10 | 中高 | 两微批稳态周期公式漏 T2D/transition | 必须，M3 流水周期前 | 否 |

---

### D2-01 比较资格与资源增量规则未定义

`高` · `ui-data-contract.md §3 Comparison、§4 M02/M20、§5 V09` · `questions.md Q4/Q11` · **必须修正**

- **已有覆盖**：Comparison 有 `baseline_run_id`、`candidate_run_ids`、`metric_id`、资源预算、比较资格、排除原因；Q4 默认“总设备预算相同，额外草稿卡另列增量”；V09 要求口径/资源/质量资格。
- **仍缺**：`eligible` 的判定谓词集合；8+1 卡相对 8 卡是“不合格”还是“合格但标注增量”；谁计算、如何保证 Web/CLI/CSV 完全一致。
- **反例**：baseline 8 卡，candidate 8+1 卡。一个消费者按“总设备预算相同”判不合格，另一个按“目标同预算、草稿另列”判合格并给出 1.2×；同一 run 在矩阵和 CSV 中一个灰一个绿。另一个反例：dtype 或 OSL 不同的两 run 被标为 eligible。
- **最小修订**：新增 `ComparisonPolicy{same_model_revision, same_dtype, same_workload, device_budget_rule: equal_total|equal_target_plus_delta, missing_capability: exclude, quality_equivalence_required, policy_hash}`；输出 `eligibility{status, failed_predicates[], resource_delta}`；`policy_hash` 写入 Comparison 与 ExportBundle。
- **可执行验收**：固定 fixture：8 卡 baseline vs 8+1 卡 candidate vs 16 卡 candidate；在 `equal_total` 下三者分别为 eligible、ineligible（或按用户选择为带增量合格）、ineligible；Web/CLI/CSV 的 status/reason 完全一致；把 policy 改为 `equal_target_plus_delta` 后结果按规则翻转且可追溯。
- **用户决策**：是。Q4 决定资源规则；Q11 决定主指标。默认建议“目标同预算 + 强制显示增量”，但需用户确认。

### D2-02 单一种子不足以支持并发/顺序变化下的重放

`高` · `plan.md §4 运行身份、ui-data-contract.md §3 RunManifest、§5 V10、principles.md §7 RoundState` · **必须修正**

- **已有覆盖**：RunManifest 记录 seed；V10 要求相同 seed 重放、并发隔离；RoundState 含 `rng_state`。
- **仍缺**：随机流的划分键（request/round/purpose）、RNG 算法/版本、事件顺序变化时如何保持每请求序列不变。
- **反例**：全局单一流。场景 A 有 8 个请求；场景 B 是同样 8 个请求加一个更早到达的无关请求。调度顺序改变，第 3 个请求的接受抽样消耗了不同数量的随机数，同一 seed 下 token 序列不同。V10 的“同 seed 同输入”不会发现“同请求在不同并发下不可重放”。
- **最小修订**：RunManifest 增加 `rng_scheme{algorithm, version, key_fields:[run_seed, request_id, round_id, purpose]}`；实现为 counter-based 子流；V10 增加“请求顺序置换/插入无关请求”用例。
- **可执行验收**：同一 scenario + seed，跑 8 请求和“8 请求 + 1 个早到无关请求”；按 `request_id` 对齐后每个请求的 token 序列、A/K、事件时间完全一致；若实现使用全局流，此测试必须失败。
- **用户决策**：否（工程必须项）。

### D2-03 矩阵颜色与主排序 TPOT 变体未绑定

`高` · `ui-data-contract.md §2 P01 矩阵、§4 M01/M02、§6 M2` · `questions.md Q11` · **必须修正**

- **已有覆盖**：M01 明确分列三种 TPOT；Q11 要求固定一种主排序指标；矩阵禁止“颜色代表已证实更优”；每格独立 run。
- **仍缺**：默认排序/着色到底用哪一种 TPOT（解析 `E[T]/E[N]`、请求平均间隔、逐 token emission），以及颜色编码（ratio 分箱、排名、仅资格）。
- **反例**：Run A 的 `E[T]/E[N]` 更好，但因 prefill 长、OSL 短，请求平均间隔更差；P01 按一种排序，P04/CSV 按另一种排序，矩阵行列颜色翻转。另一个反例：1.02× 和 1.50× 都被涂成同一“绿”，或颜色只表示 eligible 不表示大小，用户无法判断。
- **最小修订**：Comparison 增加 `ranking_metric_ref{metric_id, statistic, scope, tpot_kind}` 与 `matrix_encoding{mode, bins, neutral, ineligible_pattern}`；`tpot_kind` 枚举三种；默认值写入 policy_hash。
- **可执行验收**：构造两个 run，使三种 TPOT 排序两两相反；Web/CLI/CSV 必须按 `ranking_metric_ref` 给出同一排序，另外两种作为次级列显示；同一 ResultBundle 在 Web 和 CSV 的 bin 编号一致；ineligible 格不得着色为胜出。
- **用户决策**：是。Q11 选主指标；颜色分箱可由工程默认。

### D2-04 过滤后的聚合范围未定义

`中` · `plan.md §8、ui-data-contract.md §2 P01 状态/过滤、§3 前端职责` · **必须修正**

- **已有覆盖**：筛选只改变视图范围，不悄悄重算全局摘要；分页不改变全局汇总；ExportBundle 声明 filtered/full。
- **仍缺**：哪些指标是 `global_fixed`，哪些允许 `selection_scoped`；过滤后显示的 KPI 是全局还是所选子集；两者如何同时呈现。
- **反例**：10 个请求，过滤到 3 个。若 TPOT 保持全局值，数字与可见的 3 行不符；若前端重算，违反“不悄悄重算”。两种实现都能自称符合文字。
- **最小修订**：Metric 增加 `aggregation_scope: global|selection` 和 `selection_ref`；视图声明 `displayed_scope`；后端同时返回全局与选择聚合；导出写 `filtered=true` 及过滤条件 hash。
- **可执行验收**：过滤 3/10 请求；页面同时显示“全局 n=10”和“所选 n=3”两个 TPOT，标签明确；全局值不变；CSV 两列都有，manifest 记录 filter hash；分页前后全局值不变。
- **用户决策**：否。

### D2-05 “容量上限”未区分标称/可用显存

`高` · `ui-data-contract.md §4 M06、§3 Device/Capability、principles.md §6` · **必须修正**

- **已有覆盖**：每 rank 同时存活对象、峰值时刻、余量、shared identity；allocator 预留与块取整；单个已知下界超上限可报 OOM。
- **仍缺**：容量分母的来源（标称 HBM、驱动可用、扣 context/其他进程/预留），以及 `usable_memory_bytes` 的证据。
- **反例**：标称 80 GiB、实际可用 78.5 GiB；方案峰值 79 GiB，按标称判 feasible，按可用判 OOM；矩阵把它放进赢家排序。同一设备不同 driver/context 配置，可用值不同，但报告只写 80 GiB。
- **最小修订**：DeviceProfile 增加 `nominal_memory_bytes`、`usable_memory_bytes`、`reserved_bytes`、`source/evidence_ref`；M06 用 usable 判 OOM，同时显示两者；`capacity_snapshot_id` 写入 RunManifest。
- **可执行验收**：fixture 标称 80、可用 78.5、分配 79 → `feasible=false, reason=oom`，不进入默认排序；把可用改为 80 后翻转为 feasible；报告必须显示 78.5 的来源。
- **用户决策**：部分。Q2/Q7 提供真实设备/可用值；无数据时用保守默认并标 unknown。

### D2-06 MoE capacity factor / token drop 未进 schema

`高` · `principles.md §4、ui-data-contract.md §4 M05/M09/M22、plan.md §3/§5 M1` · **必须修正或明确标 unsupported**

- **已有覆盖**：原则承认“容量 drop 会改变分布”；M05 要求 token-expert assignments、U、shared/routed 字节；M09 要求有效 token/routing；M22 要求质量轴。
- **仍缺**：`capacity_factor`、drop/padding 策略、被 drop 的 assignment 计数，以及 M05/M09 用“已执行”还是“已提议”assignment。
- **反例**：E=64、k=8、M=1024、capacity_factor=1.0；热点专家溢出，5% assignment 被 drop。Ledger 仍按 8192 个 assignment 计 FLOPs 和专家权重读，M09/M05 高估；输出分布已变，M22 没有 drop 率，无法解释质量差异。
- **最小修订**：MethodPlan/Recipe 增加 `moe_capacity_policy{capacity_factor, drop_policy, padding}`；WorkLedger 增加 `offered_assignments`、`executed_assignments`、`dropped_assignments`、`padding_slots`；M05/M09 明确用 executed；M22 记录 drop 率。
- **可执行验收**：E=4/k=2/M=4 手工构造热点，capacity_factor=1.0；手算 dropped>0；报告必须显示 offered/executed/dropped 三个数，FLOPs/权重字节按 executed 计算；若实现选择不支持 drop，则 MoE 结论必须 `unsupported` 而非填 0。
- **用户决策**：部分。首发是否支持 drop 语义需定；建议至少不能静默忽略。

### D2-07 分位数单位与相关性重放未定义

`中高` · `ui-data-contract.md §4 M17/M22、§5 V08` · `questions.md Q8` · **必须修正**

- **已有覆盖**：M17 要求样本数、聚合窗口、estimator、simulated/measured 分开；Q8 明确均值不足以决定尾延迟；M21 区分假设区间与置信区间。
- **仍缺**：分位数按 request、token 还是 round 加权；从 A/K 直方图重采样时是否保留请求内/请求间相关；iid 重采样只能保边际。
- **反例**：K=2，每轮 A∈{0,2} 各半。Trace X 交替 0,2,0,2；Trace Y 一半请求全 0、一半全 2。合并直方图完全相同，但 Y 的请求级 TPOT p99 明显更差。若按合并直方图 iid 重采样，X/Y 都生成“每请求混合”，p99 既不等于 X 也不等于 Y。另一个反例：99 个 1-token 请求 + 1 个 1000-token 请求，request-weighted 与 token-weighted p99 不同。
- **最小修订**：M17 增加 `quantile_unit: request|token|round`、`resampling_scheme: iid_marginal|trace_preserved|markov`；只有 `trace_preserved` 可称“trace replay”；`iid_marginal` 必须标注“不保尾相关”。
- **可执行验收**：构造 X/Y 两条同直方图 trace；用两种 scheme 重放，报告必须显示 scheme；`iid_marginal` 的 p99 不得被标为“来自 trace”；`trace_preserved` 下 X/Y 的 p99 分别接近手算值；token/request 两种加权各出一列。
- **用户决策**：是。Q8 决定是否有 trace；无 trace 则只能做假设扫描并禁止尾延迟结论。

### D2-08 V07 只验已声明依赖，未验“不可预知”

`中` · `principles.md §8、ui-data-contract.md §5 V07、§4 M16` · **必须修正**

- **已有覆盖**：事件开始≥依赖完成；资源互斥/共享；通信先发后用；原则明确“非预知策略只能读取当前事件时刻已可见的信息”。
- **仍缺**：信息可见性/决策时刻的字段与校验；依赖边漏画时如何发现。
- **反例**：请求 t=10 到达，调度器把它放进 t=5 开始的 batch；只要没有从 arrival 到 batch_start 的依赖边，V07 的“开始≥依赖完成”通过。另一个反例：scheduler 读取了 t=20 才产生的队列长度来决定 t=10 的 batch。
- **最小修订**：Event 增加 `decision_time`、`visible_inputs[]`、`input_ready_at`；validator 断言 `start >= max(dep.end, visible_at)`；调度事件必须声明 `decision_time <= start`。
- **可执行验收**：构造“到达晚于 batch start”的负例，validator 必须报 `causality_violation`；构造“读取未来队列长度”的负例，必须报 `information_leak`；合法 trace 通过。
- **用户决策**：否。

### D2-09 M22/M23 无明确页面归属

`中` · `ui-data-contract.md §2 页面表、§4 M22/M23` · `questions.md Q9` · **必须修正**

- **已有覆盖**：P05 覆盖证据/校准、V11、M4；M22 质量与接受分布、M23 覆盖/误差/对账都有指标定义；P06 有迁移差异。
- **仍缺**：M22 的数据集/revision/参考输出/质量指标由哪个页面承载；M23 的覆盖分母/blocking gap 由哪个页面承载。P05 的表只写“原始 profile、留出误差、外推”，更像性能校准，不含任务质量。
- **反例**：用户从 P01 点“质量”或“覆盖率”，没有页面 ID 可跳；M22 只存在于指标表，M23 只散落在 P05/P06。实现时容易出现“指标有定义、页面无入口”。
- **最小修订**：在 §2 增加 P05a“质量与接受分布”和 P05b“覆盖、误差与对账”，或在 P05 下增加两个有 owner 的子区；建立 `metric_id → page_id/region_id` 映射表。
- **可执行验收**：对 M01–M23 做一次自动检查，每个指标至少有一个 `page_id/region_id`；M22 能显示 dataset revision/reference 输出/质量轴，M23 能显示 blocking gap 列表；缺失即测试失败。
- **用户决策**：仅 Q9 的交付批次；内容归属不需要用户决定。

### D2-10 两微批稳态周期公式漏 T2D/transition

`中高` · `principles.md §8、ui-data-contract.md §4 M18、§5 V07` · **必须修正**

- **已有覆盖**：`T_round=T_draft+T_D2T+T_verify+T_T2D+T_transition`；两微批可交错，稳态周期“接近 `max(T_draft+T_D2T,T_verify)`”；明确该周期不能直接当请求 TPOT；plan 也警告 max(D,V) 不能作通则。
- **仍缺**：该 max 公式漏掉 `T_T2D` 和 `T_transition`，且未定义两微批的 stage 边界；没有手算 timeline 验证。
- **反例**：`T_draft=10`、`T_D2T=1`、`T_verify=10`、`T_T2D=1`、`T_transition=2`。公式给 `max(11,10)=11`；若 verify 阶段包含 T2D+transition，实际周期是 `max(11,13)=13`，低估约 15%。当 T2D/transition 相对 verify 不小，M18 的周期和吞吐解释会偏乐观。
- **最小修订**：明确 stage 划分，写成 `T_cycle ≈ max(T_draft+T_D2T, T_verify+T_T2D+T_transition)`，或声明各项归属并给出误差界；M18 的周期字段绑定该公式 ID；V07 增加两微批手算 timeline。
- **可执行验收**：用上述数值构造两微批事件 trace；事件模拟周期与公式在声明容差内一致；把 `T_T2D/T_transition` 设为 0 时两者退化为 11；若不一致，公式必须标为 invalid 而非“近似”。
- **用户决策**：否。

---

### 必须修正与可选后续

**建议在对应里程碑前必须修正**：D2-01、D2-02、D2-03、D2-05、D2-06、D2-08、D2-10；D2-04、D2-07、D2-09 至少在 schema/页面启用前补齐。

**可选后续（不计入上表 10 项）**：

- V05 补 `K=0` 普通 decode 分支。`plan.md §6` 已要求，V05 只列了 A=0/A=K，未列 K=0。
- 拆分 V09：当前 V09 同时含“比较资格”和“旧仓对账”，而“串行收益还需 V05–V09”会把 M5 的旧仓对账提前变成 M2 门禁；建议拆为 V09a 比较资格、V09b 旧仓对账。
- `plan.md §4` 稳定 ID 列表未含 `cohort/batch`，但 M16 已使用；补进 ID 约定。
- tokenizer 兼容在 CapabilityReport 中有描述，但 V01–V12 没有对应负例；建议加“vocab hash 不一致必须 unsupported”的验收。
- Prefix Cache 的命中 token/复用字节/eviction 没有独立指标；M4 前可补 `cache_stats`，否则无法解释同 TPOT 下不同命中率的差异。
- M03 的 `M` 字段未定义（是每次 forward 的 token 数、还是块大小）；建议在 MethodPlan 中给类型和单位，避免 S/M/K 摘要再次替代程序图。

### 仍有疑问但证据不足

- **V4.1 的 Engram/hc/CSA2 是否已完整进入账本**：冻结材料只给结论性断言，未给字段清单或可复核的源码映射；本轮无法独立判断，不能把“已补读”当作已验证。
- **跨服务 TP8+TP1 的真实支持边界（Q7）**：材料只说“所查 vLLM 进程内路径要求目标和草稿 TP 相同”，未给跨服务协议/版本的完整证据；不能据此推断所有独立服务方案。
- **真实接受分布/trace（Q8）**：无数据，无法判断 M03/M17/M22 的首发数值范围和尾延迟结论；D2-07 只能用构造 fixture 验收。
- **真实可用显存/驱动/context（Q2）**：无测量，D2-05 只能用 80/78.5/79 这类 fixture 验证契约，不代表任何真实设备。
- **扫描规模与分页阈值（Q10）**：无实际产物，无法判断“几十个方案/大 trace 游标分页”是否足够；这属于工程容量假设，不应写成已验证承诺。
- **“七项摘要”是否单列观测接受率**：M03 可由 A/K 派生，但原型中的接受率 KPI 是否保留为独立摘要未在契约中明确；属产品选择，不是正确性问题。

### 交付文件/目录

本次为只读审查，未修改或创建任何文件；无交付产物。

```json
{"version":1,"nonce":"82b5bb84-867d-43c9-8ff6-5efbb5b7fd6c","summary":"第二轮独立复核：10 项高价值发现，7 项建议在对应里程碑前必须修正，3 项需补 schema/页面归属；无文件产物。","files":[]}
```