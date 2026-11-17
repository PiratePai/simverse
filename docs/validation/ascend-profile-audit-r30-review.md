# PR #26 审查：修复代码，暂缓合入

2026-11-17。被审提交：`89803d0a09c13bcd237049c2c2ccefe7ec381f4d`。本轮在 Windows 阅读源码、核对仓内派生产物并运行离线测试；未访问原始约 400 MB profiling 目录，未运行 NPU，也没有重做原件导入。原审计包字节保留，复核结果见 [review.json](../evidence/ascend-profile-audit-r30-review/review.json)。

**2026-11-18 后续**：已在本机原件上重导入并以修复后导入器重建审计包，证据与旧→新差异见 [重建审查](ascend-profile-audit-r30-rebuild-20261118.md)。本文件保留为历史审查记录（旧包字节与结论不改写）。

**裁决：暂缓合入。** `not-admitted` 能阻止生产计价，但不能替代审计产物自身的来源与归属正确性。代码侧已修复的缺陷不代表现有数据已由修复后的导入器重建。

| 优先级 | 发现与证据 | 本轮处理 / 重新提交要求 |
|---|---|---|
| P1，阻断 | 六个 run 的 `manifest.source.files_sha256` 实际只登记 profiler_info、metadata 和 profiler DB；未登记已消费的 `kernel_details.csv`、`op_statistic.csv`、`step_trace_time.csv`、`analysis.db`。原报告声称这些 CSV 已逐文件入哈希，与产物不符 | 导入器补齐所有已消费文件哈希；原件当前不可读，不能补造已有包的哈希。须从原件重导入并验证聚合值、来源哈希和字节重放 |
| P1，阻断 | 行级数据将 OP Type 的 dtype 并集套到每个 shape，丢失逐调用 dtype 关联；K/N 聚合也丢失 M、阶段和 target/draft 关联。调用数相同不证明耗时相同，宽度分摊也不是逐行实测 | 新导入按调用 shape 与两个乘法操作数 dtype 联合分组；现有包不能从摘要恢复联合分布。保留候选性质，重新输出含 M/角色/阶段可追溯性与未证明项的表，不把按计数/宽度分摊称为 measured 归属 |
| P1，阻断 | `gemm_grouped_moe` 3,402,028.207 µs 与 `attention_kv_write` 14,337.618 µs 没有行级分摊，却被硬编码从 `unallocated_groups` 排除 | 已修复导入器按实际候选分配总量计算剩余项，包含未匹配 GEMM、残差与所有未分摊组。未猜测归属或改写现有归档；仍须重建并以原件证明每笔设备时间只归属一次 |
| P1 | 缺少 `distill_files` 条目时加载器及导入器 `--check` 可以跳过该文件验证，空清单也能通过；内嵌 `selected_ms` 还可与更新后的 manifest 一起进入 | 已修复：固定七个必需产物、SHA256 格式与 schema 校验，拒绝路径越界键、非有限值、内嵌准入/选用成本；两入口共享验证逻辑，加入负例 |
| P1 | API 从 diff 的不存在的 `identity.geometry_check` 读取形状判断，真实 identity 失败未传播；验证器又把缺少关键维度的“无冲突”当作全部通过，Qwen3.6 未知 TP 也得到通过 | 已修复报告身份读取、按模型过滤 TP 域，并将未知维度/TP 保持 unknown；修正后基于既有摘要重判，两个 Qwen3.6 run 不能得到完整几何通过。旧 identity.json 尚待原件重建 |
| P2 | `split_qkv_rmsnorm_mrope` 被写成包含 q/k/v 投影 GEMM；参考实现先执行 `qkv_proj`，再 split/norm/rope | 已修正导入器的候选 ownership，仅列 q_norm/k_norm/rope，防止重复认领投影成本。旧 mapping.json 尚待重建 |
| P2 | 无法反解 passes/K 时仍进行 None 算术；残差容差让非零调用差也可标 matched；step 比较用 zip 截断并将缺失 overlap 补零 | 已修复未知 pass/K 的安全返回、GEMM 计数严格相等与 step 数量/空值处理；加入回归 |

参考源码为官方 [vllm-ascend @baa023ada41a5b889cac6249845a688aab130892 的 forward](https://github.com/vllm-project/vllm-ascend/blob/baa023ada41a5b889cac6249845a688aab130892/vllm_ascend/patch/worker/patch_qwen3_5.py#L43)，固定字节与 SHA256 见 review.json 和其相邻 reference 文件。这里只做源码阅读，不把该参考版本当作未知采集栈的版本证明，也未执行上游算子。

## 重新提交的最小交付

1. 在拥有原件的机器上，用修复后的导入器重新导入同一批文件，保留本次原导入快照对应的 Git 提交；补齐所有消费文件的 SHA256。先修旧数据的可追溯性，无须仅为补哈希重跑 NPU。
2. 分离原始 kernel 观测、候选语义匹配和假设分摊。对 M/阶段/角色/布局/权重/实现未知的键保持未知；完整保留未分配时间，不从调用数推导实测耗时份额。
3. 更新 identity、mapping、diff、ledger_rows 与 manifest；对新旧差异逐项说明。提供独立总量对账和语义/计数负例；`--check` 仅作文件完整性检查，不能写作原件重放验证。
4. 同步报告、状态文档、测试和 PR 描述；重新跑全量门禁。归档身份/正式校准/部署资格的未决项继续保留。

## Issue 审视（待讨论，未修改 issue 状态）

| Issue | 判断与依赖 | 建议讨论的取舍 |
|---|---|---|
| [#28 身份元数据](https://github.com/PiratePai/simverse/issues/28) | P0，先补本 PR 漏掉的源哈希，再找原启动脚本、config、版本、请求与 step 信息。已有产物的“文件完整性”和正式校准的“身份完整性”要分开验收 | 优先只读补证；缺失且不可恢复的统计/阶段边界再定向重采，不默认整批重跑 |
| [#29 融合核 ownership](https://github.com/PiratePai/simverse/issues/29) | P1，依赖采集实现版本；新增 split/norm/rope 不含投影、未分摊项守恒与逐调用角色要求 | 先保留融合组总时间；没有 correlation 时，宽度/计数分摊只作显式假设，不以拆得越细代表越准确 |
| [#30 实现成本缺口](https://github.com/PiratePai/simverse/issues/30) | P1，14.9%/12.0%/7.8% 是该采集的非通信 task-duration 比例，含混合阶段；不是 TPOT 低估比例。数据搬运与现有字节模型可能重叠 | 先完成该 run 的组/阶段/任务所有权；新增成本项须证明不重复计费，再决定实现插件；保持 Work/Traffic/Memory 不变，不用全局倍率 |
| [#27 官方规格与设备身份](https://github.com/PiratePai/simverse/issues/27) | 拆成实际采集设备 fingerprint 与解析峰值规格两条线。前者是同域使用 trace 的条件；后者决定是否能把旧仓参数从 assumed 改成 spec | 保留 model-verse 的 B3 解析档可用。完整官方峰值表不应自动成为所有局部耗时审计的前置条件；实际 SKU/栈仍需证明 |
| [#31 Qwen3-8B / DSpark](https://github.com/PiratePai/simverse/issues/31) | P2，产品范围决策。现有明确范围是两款主模型及绑定 Qwen3.6 的 DFlash，当前无需扩大目录 | 建议先保留离线研究资产；离线历史模型可独立登记/研究，不需要为消费一份 trace 就加入活跃目录 |

本轮验证结果以实施状态的审查条目为准。代码回归通过只说明列出的拒错路径已修复；数据重建与原件核对仍未完成，PR 应保持 open。
