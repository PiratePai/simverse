# R30 重建审查：从原件重导入后的证据与剩余缺口

2026-11-18。被审历史提交：`8c07518`（PR #26，含审查修复）；本重建在本机原件
`D:\github\qwen3_6_mtp_domino`、`D:\github\qwen3-8b-0811` 只读重导入，未运行 NPU。
历史审查见 [ascend-profile-audit-r30-review.md](ascend-profile-audit-r30-review.md)
（旧包缺消费文件哈希，保留为历史记录）；重建依据
[review.json](../evidence/ascend-profile-audit-r30-rebuild/review.json)、
[旧 review.json](../evidence/ascend-profile-audit-r30-review/review.json)。

## 1. 重建范围与来源哈希

用修复后的 `tools/import_ascend_profiles.py` 重新导入 6 个 run，审计包 manifest 由
`165cfed7…` 变为 `958cb6c4…`。**每个 run 登记 7 个已消费源文件 SHA256**（共 42 条）：

```
profiler_info_0.json
profiler_metadata.json
ascend_pytorch_profiler_0.db
ASCEND_PROFILER_OUTPUT/kernel_details.csv
ASCEND_PROFILER_OUTPUT/op_statistic.csv
ASCEND_PROFILER_OUTPUT/step_trace_time.csv
ASCEND_PROFILER_OUTPUT/analysis.db
```

未消费文件（trace_view.json、operator_details.csv、communication.json、FRAMEWORK、
logs）仍不登记、不冒充来源；`manifest.source.not_hashed` 保留原因。
`--check` 只校验产物与 manifest 字节，**不写作原件重放验证**。

## 2. 旧 → 新差异

| 项 | 旧包（89803d0） | 重建包 |
|---|---|---|
| 消费源文件哈希 | 每 run 3 个（info/metadata/profiler DB） | **每 run 7 个**（含 kernel/op/step/analysis） |
| 逐调用 dtype | OP Type dtype 并集，shape 丢失联合分布 | 按 `(OP Type, shape, 两操作数 dtype)` 采样后联合分组；无联合样本不猜 |
| passes/MTP K | 未知时仍可能做 None 运算 | 未知/非整数/非正 → 显式 unknown，**不做行级分摊** |
| 计数匹配容差 | 2% 残差也可标 matched | **GEMM 计数必须严格相等**；残差显式 `ambiguous_count_residual` |
| 未分配组 | 硬编码排除 GroupedMoE/KV write | 按"组实测量−已分摊量"**列全剩余**：GroupedMoE 3,402,028.207 µs、KV write 14,337.618 µs、dense 剩余 452,630.116 µs 等 |
| split kernel ownership | 认领 q/k/v 投影 + norm/rope 共 6 行 | 只认领 `q_norm/k_norm/rope`；参考 forward 中 qkv_proj 先于 split（vllm-ascend @baa023a 只读参考，不冒充采集栈） |
| 几何验证 | 缺 TP 也判"通过" | Qwen3.6 两个 run：`observed_compatible=true` 但 **`verification_passed=false`**（unknown `tp_for_global_heads/vocab`）；Qwen3-8B（tp=2 已知）通过 |
| 行级数值 | 命名接近 measured 归属 | 明确 `allocation_evidence = width/count-derived-assumption`、`signature_scope=K/N/dtype 聚合，M/阶段/角色未知`；**不标 measured** |

## 3. 独立对账与守恒

- **总量**：6 run 的 `op_statistic.csv` 与 `kernel_details.csv` 非 hcom 行计数完全相等、
  总时长相对差 ≤6e-9（`census.op_statistic_consistency`）。
- **组级剩余量**：对每个 mapped/fused 组，`unallocated_us = max(0, observed_us −
  Σ alloc)`，且必须与 `ledger_rows.unallocated_groups` 逐项相等；重建审查断言全部通过
  （GroupedMoE / KV write 不再被排除）。
- **ownership**：`mapping.rows[attn_qkv_norm_rope_fused].ledger_rows ==
  ["q_norm","k_norm","rope"]`；不得重复认领投影 GEMM。
- **未证明项**：阶段/M/角色/布局/权重/实现 commit 仍未知；row-level 只能作为假设诊断，
  不进入 `CalibratedPricer`。

## 4. 复现

```bash
py -3.12 tools/import_ascend_profiles.py --check \
  --out assets/profile_audit/ascend_910b_profile_audit_v1
py -3.12 tools/review_ascend_profile_audit_r30_rebuild.py --check
py -3.12 tools/review_ascend_profile_audit_r30.py --check   # 历史记录重放
py -3.12 -m simverse.cli ascend-audit --check
```

## 5. 结论与剩余缺口

- 审计包来源哈希与组级剩余量已可追溯：**audit-only 可交付**（decision
  `approve-audit-only`）。
- 仍**不可**接入生产校准/部署排名：设备 SKU/board、vLLM/vllm-ascend commit、权重
  revision、warmup/step 边界缺失（#28）；融合组 ownership 未源码 pin（#29）；
  实现特定成本缺口未建模（#30）；Qwen3-8B/DSpark 保持离线（#31）；官方 dense 规格
  与设备指纹分离推进（#27）。
- 约 400 MB 原始采集不在仓库；本重建只证明"派生包 ↔ 已登记源文件哈希"一致，
  不等于原件在其它机器可重放。
