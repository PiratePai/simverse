# Ascend profiling 审计包（只读 · 未准入）

本目录由 `tools/import_ascend_profiles.py` 从外部 Ascend NPU profiler 产物蒸馏生成，
**不是实测校准集、不参与计价**。用途是 A-N5「profiling 校对」第一阶段的
原始 trace 审核、映射与差异表（docs/ascend-validation-plan.md §1/§3）。

- `manifest.json`：本包身份、来源 run 与逐文件 SHA256、准入裁决（恒 `not-admitted`）。
- `identity.json`：每 run 记录到的栈身份/并行组/会话窗口，以及**缺失**身份字段与原因；
  另含 `shape_candidates`（由具体算子输入/输出 shape 反推的 hidden/GEMM K/N/lm_head
  输出宽/专家维度，带调用计数）与 `geometry_verification`（与 owner 声明模型的
  维度核对结果）。
- `census.json`：逐 OP Type 的设备 kernel 计数与耗时统计（min/median/p90/max）。
- `shape_costs.json`：关键算子类按输入 shape 的采样统计（未来局部校准候选键）。
- `mapping.json`：采集侧 kernel/OP Type → 本仓账本行组 ownership 与覆盖状态。
- `diff.json`：结构差异表（账本覆盖时间占比、通信类、未建模类、设备内核间隙）。
- `ledger_rows.json`：A-N5 离线 **row-level 候选提取**——GEMM shape 签名 + 调用计数
  方程 → 账本行 measured 候选表（目标模型 Qwen3.6-35B-A3B 的 MTP run；minidraft
  按 owner 声明排除；Qwen3-8B 为历史模型不做跨模型匹配）。融合/同形核按宽度或
  计数分摊，**不是源码 ownership pin**。

## 身份声明

owner 已声明两个根目录的目标模型：`qwen3_6_mtp_domino` → Qwen3.6-35B-A3B，
`qwen3-8b-0811` → Qwen3-8B；mtp/dflash/dspark 为三种现有草稿模型，
minidraft 为自研草稿、本轮不考虑（对应 run 标记 `excluded_by_owner`）。
该声明只覆盖名称：权重 revision/config/checkpoint、推理引擎 commit、量化配方
仍未验证，故不影响下方准入结论。

## 为什么不能直接用于校准

1. 设备身份只到 `Ascend910B` 家族，无 `npu-smi` board/SKU/驱动；本仓已有
   `ascend_910b3` 解析档但为 **assumed 旧仓映射**（spec 未官方核验），不能证明
   同 SKU/驱动；910B3/910C/A2/A3 能力不可互相继承（A-N4 仍需官方规格核验）。
2. 推理引擎版本（vLLM/vllm-ascend）与权重 repo/revision/checkpoint 哈希未随
   profile 提供 → 算子 ownership 不能按源码 pin，只能按 kernel 语义归组。
3. 无 warmup/active 区间、无重复 run、step 节点为空 → 不能宣称统计稳定或切分
   prefill/decode 窗口。
4. `CalibratedPricer` 的设备/精度/TP/形状域门禁无法匹配（assumed 档不等于
   已证明同域）→ 校准一律解析回退。

## 复现

    python tools/import_ascend_profiles.py --check --out <本目录>

`--check` 只校验本轮产物哈希与 manifest 一致：不重写产物，也不读取原始采集目录。
