结论：以 `main@75c0489` 为基线，成本路径的数值锚点、Work/Traffic/Memory 账本和记录/校准投影均自洽复现；主控在审查期间落地的 `CostSession` 已去掉 `run_v2` 的二次全量求值，并通过 15 场景数值/账本锚点与 Ascend 记录回放，但仍有 3 个兼容/披露口子需要收口（R27 冻结回放、全回退的 `selection` 语义、profile 融合组与 M18 的默认展示边界）。

## 审查快照

- 起始：`HEAD=75c0489`，工作区干净；以下基线数值均在该状态下用进程内离线调用取得。
- 审查中主控开始修改：`simverse/pipeline.py`、`simverse/execute/serial.py`、新增 `simverse/cost/session.py`、`tools/replay_cost_groups.py`、`docs/evidence/cost-groups/`。以下 in-flight 结论标注为“读取时”，不是对主控最终实现的验收。
- 只读审查：未修改/提交文件，未启动其他 Agent、模型 CLI、NPU 或凭据。
- 未跑全量 pytest；定向 pytest 受只读沙箱临时目录限制（见“验证边界”）。

## 1. 关键基线与测试锚点

| 场景 | 锚点 |
|---|---|
| 必测 DFlash `_dflash_sc`（B1/ctx8192/K15/α0.72/TP8） | `E[N]=3.552800756624948`；`M01=36.83725440601121`；work 行 1355 = target 623 + draft 109 + verify 623；`t2d=58209.08759654315 B`（float） |
| 默认 `ScenarioV2()`（Qwen+DFlash） | work 1355 行；`target_decode=112.91607144789675`；`M01=42.810383894203646`；`M02=2.6375860521817263`；`row_prices=623` 覆盖全部 target 行 |
| R29 TP2 baseline | 命中 `80/704`；解析 `25.748260145129617` → 记录 `22.110020145129617` |
| R29 TP2 DFlash | 命中 `172/1530`；`18.230667487829347` → `17.027995593604487` |
| R29 TP2 / EP2 baseline | 命中 `40/744`；`25.746704145129616` → `23.927584145129615` |
| R29 TP8 baseline | 命中 `0/704`；全回退，无 `M01_record_tpot`，capability=`unknown` |
| R28 TP2 baseline | 命中 `40/704`；`25.748260145129617` → `26.475700734947658` |
| R28 TP8 baseline / DFlash | baseline `201/704`：`115.58350301050466` → `118.44761934659522`；DFlash `201/1530`：`36.55616000311457` 不变 |
| MTP TP2 | 15 个 `draft_step_*`；命中 `205/1814`；最后一步含 `mtp_kv_rewind_waste`；15 个 `mtp_fc_allgather` 均命中（插值） |
| H200 30B-tiny TP1 profile | 解析 `5.754412` / selected `19.881676` / measured `14.272144` / fallback `5.025285` / gap `0.584247`；覆盖率 `577/675` |
| H200 235B TP8 profile | selected `35.035907`；`M01_tpot_profile_calibrated=299.30603900000006`（= selected + collectives）；覆盖率 `1129/1319` |
| DT 串行 vs 流水（H100 默认） | `M04_round=114.34288685400568`；`M18_per_round=113.34133895549287`；周期 `226.48225447489244`，不可当 TPOT |
| 二次求值成本 | 记录档 `run_v2` 调用 `_evaluate_v2` **2 次**；同进程约 `31ms`（无记录）vs `64ms`（有记录），仅作数量级说明 |

主控已捕获的同一基线见 [baseline.json](D:/Github/new-viewer/docs/evidence/cost-groups/baseline.json)（`source_commit=75c0489...`；我未修改该文件）。

## 2. 最重要的回归约束与反例

| 约束 | 反例 / 测试必须钉住 |
|---|---|
| 计数、流量、容量逐字段不变 | 开/关 profile、开/关 records 时 `work` 整字典、`cost.memory`、`model_ref.digest`、`M05/M06/M11` 必须 `==`，不能只比行数。基线已满足。 |
| 记录实例粒度不能改 | `record_cost.entries` 必须仍是“每行/每 collective/每 link 一条”：704=623 行+81 通信；1530=1355+175。若按成本组合并，`tools/replay_ascend_record_costs.py --check` 与 docs 的 704/1530 锚点会漂。 |
| 融合组总成本不拆成伪实测行 | H200 30B 的 `attn_pre_proj` 拥有 240 行：provider 实测 `5.6256ms`，而 240 行解析和仅 `0.214631ms`。不能在 `row_prices` 里把 5.6256 按比例摊给 q/k/v/q_norm/k_norm，也不能给 profile 生成 `selected_row_prices`；行级价格保持解析诊断。 |
| 解析聚合 ≠ 整体 roofline | 成本组总耗时应是“拥有行解析值之和”；不要用 `max(ΣFLOPs, Σbytes)+0.15·min` 重算整组。H200 30B 因全 memory-bound 恰好相等，不能据此认为等价；V4.1 行和 `0.800466` vs 整体 `0.799608` 已出现差异，compute-bound 会放大。 |
| 全回退不伪装成校准 | R29 TP8：`matched=0`、无 record KPI、`archived_cost_records=unknown`、CSV 全 `cost_source=analytic`。若新增 `cost.selection`，必须带 `applied=false` 且 `source=analytic`，不能写 `source=records`。 |
| 原语/角色/形状门禁 | 仅 `allreduce` 与显式 `mtp_fc_allgather` 可命中；`logits_gather`/EP dispatch/combine 必须 `primitive-unresolved`；draft 行必须 `model-role-mismatch`；GEMM 形状精确匹配、不插值；设备 `ascend_910b3`、BF16、rank=2。 |
| MTP 分支不能合并/平均 | 必须保留 15 个 draft step 的总和（最后一步多 `mtp_kv_rewind_waste`），共享 `lm_head` 按目标位宽；`E[N]`、K、verify token 数不变。 |
| 串行与流水分栏 | `M04_record_round_ms` 是串行阶段和；DT 流水 `makespan`/周期另列。基线记录 DT selected 串行 `64.649770`，而 selected pipeline 周期约 `124.870204`，不可互换。 |
| 未知阶段不填 0 | DFlash `commit_rewind`/`hidden_extract` 可显式 0 占位，但 `stage_cost_complete` 必须 unknown，`unpriced_components.time_ms=null`；Engram host、未知接受率同理。不能因为所有 plan step 都有数值就标 supported。 |
| 阶段 ID 要统一 | 现有 `cost.target_decode` / `M04.decode` / plan step `decode` 三套命名。新增 `selection.stage_costs_ms` 若 baseline 用 `decode`、groups 用 `target_decode`，默认展示/对账会漏。需定义 canonical 映射并测试。 |
| collective 不可原地污染 | `RecordPricer.collective` 会原地写 `c.t_ms`。统一路径必须先 `copy` 再替换，否则解析执行/M01 会被 selected 值污染；测试应断言原 collective 对象不变。 |
| 记录档构造用同一 model | 旧 `run_v2` 里 `RecordPricer` 用 `build_catalog()` 的新 model，而求值用传入 `catalog`。统一后应传同一个 `model`，避免归档/fixture 目录分叉。 |

## 3. 消除二次求值与 calibration/record 分叉的可行切法

建议的 `CostSession` 每阶段至少保存：

```text
rows                 # 原始 OpRow，不可变
analytic_prices      # price_rows 一次算出，永不替换
selected_prices      # record 逐行结果或 = analytic_prices
analytic_row_total
selected_row_total
collective_analytic  # 每笔的原值
collective_selected  # copy 后的 record 值
groups               # 拥有行/原语/来源/回退原因
unknown              # 阶段/组件级 unknown
```

关键语义：

1. **一次计数、一次解析计价、一次选择**：record 在行/collective/link 粒度应用；profile 在 provider group 粒度应用；之后只做投影，不再 `_evaluate_v2`。
2. **两套旧字段的语义必须分别保留**：
   - `calibration.stages[stage].selected_ms`：**不含** collectives；`M01cal = selected_ms + collectives_ms`（现有测试 `test_stage_composition_arithmetic` 钉住）。
   - `record_cost.stages[stage].selected_ms`：**含** 行 + collectives + link；现有 `test_pipeline_preserves...` 钉住。
   统一 session 存 `row_selected / collective_selected / total_selected`，两个投影各取所需，不能把 profile 的 `selected_ms` 改成含通信。
3. **执行由 stage 总成本驱动**：`evaluate_execution(plan, effective_stage_totals, e_n, dep)` 同时产出 serial round 和（DT 时）pipeline schedule。保留 `bundle.execution` 为解析值以兼容旧导出，新增 `cost.selection.execution` 作为选用值；`M04_round_selected`/旧 `M04_record_round_ms` 从同一 round 派生。
4. **默认成本组**：对 profile，直接把 `CalibratedPricer` 的 provider groups 提升为顶层组，并补一个 `fallback` 残差组和 `gap` 组；每组带 `owned_rows`、`analytic_replaced_ms`、`selected_ms`、`source`。组间必须“互斥且穷尽”所有行，即 `owned_rows ∪ fallback_rows = all_rows`。当前 in-flight 把整个 stage 收成一个 `profile_owned_groups`（675 行），`ownership` 里仍有 7 组，但默认 `groups` 会丢掉融合组粒度。
5. **行级价格只做诊断**：`cost.row_prices`、`cost.roofline_points` 保持解析；record 保留 `selected_row_prices`（逐行独立 GEMM 可以合法替换）；profile 不生成 selected 行价，也不生成 selected roofline 点。
6. **旧导出兼容**：`cost.row_prices`/`roofline_points`、`record_cost.selected_row_prices/selected_execution/stages`、`calibration.stages/coverage`、`M01_record_*`、`M01_tpot_profile_calibrated` 都要保留；新增 `cost.selection`/`groups`、`M01_tpot_selected`、`M04_round_selected` 作为 additive 字段。`_ops_csv_v2`、`_trace_csv_v2`、`workbench.js` 的读取顺序加新字段回退，不删除旧键。
7. **M18 单独决策**：`selection.pipeline_sim` 已按 selected stage 重算，但当前 `M18_*` 仍是解析值。建议新增 `M18_selected_*` 并注明边界，或明确 `M18` 只作解析对照；不要出现“selected pipeline 已算、M18 仍解析”而无披露。
8. **全回退/域外**：`selection.source` 应为 `analytic`，并加 `applied=false`/`all_fallback=true`；`record_cost.matched=0` 时仍保留报告，但不产生 record KPI。

## 4. 对当前 in-flight 实现的直接发现（读取时，文件仍在变）

- ✅ `py -3.12 tools/replay_cost_groups.py`：`15 scenarios: unchanged numerical/ledger anchors`。
- ✅ `py -3.12 tools/replay_ascend_record_costs.py --check`：exit 0，704/1530 等记录锚点保持。
- ✅ 定向 5 个成本模块 pytest：`72 passed`；4 errors + 1 failure 全部是只读沙箱无临时目录，不是代码失败。
- ❌ `py -3.12 tools/replay_cost_integrity_r27.py --check`：exit 1，`DRIFT` 仅来自 `cost` 新增 `groups`/`selection` 两个键；逐项比较无旧数值漂移。建议把该工具的可比重算过滤扩展到新键（像现有 `row_prices` 一样），或按 provenance 更新证据；不要静默覆盖 R27 after 文件。
- ⚠️ R29 TP8（`matched=0`）时 `cost.selection.source="records"`、`source_id="ascend-r29-hccl"`；虽无 record KPI，但默认消费者可能误读。需 `applied`/`status`。
- ⚠️ profile 的 `cost.groups` 目前是单个 `profile_owned_groups`（675 members），provider 的 7 个融合组在 `ownership` 里；若 UI 默认读 `groups`，会看不到 `attn_pre_proj`/`attn_decode_block` 粒度。
- ⚠️ `selection.stage_costs_ms` 在 baseline 用 `decode`，而 `groups` 用 `target_decode`；建议统一 canonical 名或同时给出别名。
- ⚠️ `M18_*` 仍用解析 `pipeline_sim`，而 `cost.selection.pipeline_sim` 是 selected；需要明确二者关系。

## 5. 建议新增/保留的测试

| 测试 | 断言 |
|---|---|
| `test_single_cost_pass` | 每 stage 行只调用一次 `price_rows` 与一次 `RecordPricer.rows`；不再出现第二次 `_evaluate_v2`；`work`/`cost.memory` 全量 `==` |
| `test_stage_additivity` | profile：`selected = measured + fallback + gap`；record：`selected = Σentries`；组间 owned/fallback 互斥且覆盖全部行 |
| `test_fused_group_not_split` | `attn_pre_proj` 240 行、`5.6256ms`；owned 行价格仍解析；profile 无 `selected_row_prices`/selected roofline |
| `test_record_gates_and_counts` | 80/704、172/1530、201/704、0/704、40/704、40/744；仅 allreduce/allgather；logits/EP/draft/shape 全回退 |
| `test_all_fallback_disclosure` | `matched=0`、无 record KPI、`selection.applied=false`、`source=analytic`、capability unknown |
| `test_mtp_steps_not_collapsed` | 15 steps、205/1814、最后一步 `mtp_kv_rewind_waste`、15 个 fc gather；stage 总和不变 |
| `test_serial_vs_pipeline_boundary` | M04 selected 串行 vs M18 selected/解析分列；事件和 = stage 和；周期不得当 TPOT |
| `test_unknown_stage_not_zero` | DFlash commit/hidden、Engram host、未知接受率：unknown 传播，`time_ms=null`，不标 supported |
| `test_old_export_shape` | `row_prices` 覆盖所有 target 行、`roofline_points` top-40、record/calibration 旧字段、CSV 表头/JS 回退 |
| `test_replay_contracts` | `replay_cost_groups.py`、`replay_ascend_record_costs.py --check`；R27 工具过滤新键后 `--check` |

## 6. 验证边界与未完成

- 基线数值来自 75c0489 的进程内离线调用；不是 NPU/上游运行验证。未下载依赖、未调用真实模型。
- 全量 pytest 未跑（按要求留给主控）。定向 pytest 的 4 errors/1 failure 均为 `FileNotFoundError: No usable temporary directory`（只读沙箱），不是代码失败；因此不能据此宣称完整测试通过。
- 主控工作区在审查中持续变化，in-flight 结论只对读取时的 `session.py`/`pipeline.py` 成立；最终以主控提交后的复跑为准。
- 未完成项：`replay_cost_integrity_r27.py --check` 仍失败（新键导致），以及上述 `selection.applied`、profile 组粒度、M18 selected 披露三个设计口子，需要主控决定是否在本轮收口。

## 交付文件/目录

无。本次为只读审查，未创建或修改任何文件。