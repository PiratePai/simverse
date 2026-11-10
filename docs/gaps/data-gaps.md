# 数据采集缺口清单（需要额外采集的数据）

与 [能力缺口清单](capability-gaps.md) 分开维护：本页只列**数据**缺口（需要采集/提供
真实数据才能解除）；工程实现缺口见另一页。每项含：影响面/缺什么/旧仓为何不足/需要
什么/建议采集与验收/当前降级/优先级。日期 2026-11-09；2026-11-10 审计更新
（全量旧仓数据审计见 [docs/calibration/oldrepo-data-audit-20261110.md](../calibration/oldrepo-data-audit-20261110.md)）。

## G-D1 Qwen3.6-35BA3B+DFlash 接受长度分布与真实 draft/verify 耗时 ⭐P0

- **影响**：必需场景②（qwen36_35b_a3b + dflash_v1）的 M01 TPOT、M02 加速比、
  M03 E[N]、M04 轮分解——当前全部建立在显式假设（α 几何 0.72/0.75 演示值）上。
- **缺什么**：真实部署的每位置接受率分布（α_i 序列或联合律）、draft 块/verify 块
  在目标硬件的实测延迟、block 16/8 两档对照。
- **旧仓不足**：model-verse 的三点自拟合是代理压缩非物理校准（evidence-audit 已裁）；
  pi-viewer 无 DFlash 方法数据；厂商 README 的 3.61× 是营销口径未复核（已 pin 不入生产）。
- **需要**：DFlash 实际部署环境（SGLang DFLASH 或 vLLM PR#40898）+ 目标卡；trace 记录。
- **建议采集**：per-round {K, A, N, draft_ms, verify_ms, accept 指示序列}；≥3 seed；
  ctx 8k/32k、B 1/8/32；greedy 与 stochastic 各一组。验收：E[N] 与实测均值一致
  （独立 t 检查），接受序列可重放进 acceptance 模块（empirical_hist 路径）。
- **当前降级**：acceptance=geometric 假设（evidence=assumed），结果标记
  analytical-estimate；mean-only 输入→E[N] unknown 不排名。
- **优先级**：P0（必需场景结论上限被它钉死）。依赖：G-C3（服务端集成核对）。

## G-D2 真模型 MoE 实测接入（历史 235B 路径已归档；两主模型需新采集）⭐P0

> **2026-11-10 归档更新**：Qwen3-30B-A3B/235B-A22B 已移出活跃 catalog；下列
> 30B/235B 数据只作历史校准归档，不能用于 Qwen3.6/V4.1。两款主模型的 MoE
> 实测仍未采集。

- **影响**：两条校准闭环的覆盖上限——MoE/router/lm_head 全部解析回退；
  30B B8 时间口径覆盖 71.8%、235B B8 42.5%。
- **2026-11-10 审计更新**：
  - A800 真 30B moe.csv（128E/top8/BF16/TP1-8 全网格）**拒绝 measured 准入**：
    CSV 无环境元数据列（gpu/driver/torch/cuda），违反准则 §3.2（G4 同型）；
    补环境记录后可复审。
  - H200 235B moe.csv TP8/EP1（真 128E vllm_fused，451 token 网格）**defer 为
    首选增量路径**：数据合格且环境齐备；G-C11（per-rank 分片账本统一）已于
    2026-11-10 A组 R01 解除，moe.csv 采集配方（TP8/EP1 inter/tp 切分）与
    新账本同口径——剩余为 G-D2 自身的蒸馏+留出验收。
  - TP1/EP4 行为 DP4 部署口径，与单卡场景不符（不采）。
- **需要**：G-C11 工程落地（先行）→ 235B moe 蒸馏准入；或对真 30B-A3B 在 H200
  补测 moe（tokens 1..32768、EP1、TP1/8、BF16+FP8，字段沿用 pi-viewer 契约，
  环境元数据必备）。lm_head（h→151936）两模型均缺实测。
- **验收**：蒸馏入册+留出 p90<5%；uniform 路由假设在域文档显式登记（测量假设）。
- **当前降级**：moe_routed/moe_router/lm_head → analytic（roofline），bundle 中
  cost.calibration.groups 无 moe 项，fallback_ms 披露。
- **优先级**：P0。依赖：G-C11（工程）或新采集。

## G-D3 ~~Qwen3-235B-A22B 官方身份~~ → 已解除（2026-11-10；同轮归档）

- 官方身份已入册：HF main @`8efa6172…`，config SHA256 `0ecd5d6f…`（网络获取，
  与 pi-viewer 二手副本逐字段核对一致）。
- **该模型已移出活跃 catalog，只作历史归档**；TP2/8 闭环报告保留日期事实，
  不得用于两款主模型（profile 绑定 235B，跨模型直接拒绝）。

## G-D4 prefill/混合批注意力 + verify（多 query）形状数据

- **影响**：投机 verify 步（attention_core Q=block>1）当前域外回退；TTFT/prefill
  未开放。
- **缺什么**：vLLM chunked-prefill/混合批的 (total_tokens, batch, chunk) 网格已有
  35k 行（pi-viewer attention.csv is_prefill/true_mixed），但未蒸馏、未建模
  verify 的"多 query 单序列前缀"形状。
- **旧仓不足**：数据在（H200 三模型均有）；缺的是新仓的形状域建模与蒸馏接线
  （工程部分见 G-C2）。
- **建议**：先蒸馏 prefill-even，再决定 verify 用 prefill 形状近似（显式假设）
  或补测 verify 形状。验收：留出误差报告 + 域文档。
- **当前降级**：verify 注意力解析回退（reason="Q>1 非 decode 形状"）。
- **优先级**：P1。依赖：G-C2。

## G-D5 E2E 请求级实测记录（H200 或目标环境）

- **影响**：本闭环只能宣称"算子级校准"；端到端 TPOT/TTFT/吞吐误差不可宣称。
- **缺什么**：固定负载（模型、B、ctx、输出长度）的整服务实测：逐请求 TTFT/ITL/
  完成时间、引擎版本、图模式开关。
- **旧仓不足**：pi-viewer examples 是冒烟断言（finite/positive），非留出实测记录；
  model-verse 的对账是同源自算。
- **建议**：至少 2×2（B1/B8 × ctx2k/32k）× 3 重复；记录引擎 commit、kernel/graph
  配置、种子。验收：与算子级组合预测独立对照，误差分布入报告（不预设门槛）。
- **当前降级**：不宣称 E2E 验证（本报告 §7）。
- **优先级**：P1（决定性能结论的最终上限）。依赖：G-D1/G-D2（否则对照无意义）。

## G-D6 KERNEL_ONLY + CUDA graph 口径校准列

- **影响**：当前 CUDA_EVENT 逐 op 口径含每 op 固定开销地板（B1 实测 13.4ms），
  不能当 graph 重放运行时预测（graph gap 0.418µs vs eager 1.349µs，3.2×差）。
  **2026-11-10 复核**：host launch 与 kernel 地板在 CUDA_EVENT 口径下不可分——
  G-C9 小 shape 解析模型的归因判定同样依赖本条（两缺口共同前置）。
- **缺什么**：kernel_only CSV 的 per-op 时间列（2026-11-10 审计确认两套 H200
  kernel_only CSV 均只有 kernel_count 有值、median 列空）——即 record_function
  轨迹的 per-op 时间汇总，以及 graph 模式整步时间。
- **建议**：对 30B-tiny 或真模型补 record_function 采集（pi-viewer 采集栈已支持），
  或提供 graph 模式 step 计时。验收：两口径并列披露，用户可选。
- **当前降级**：单一 CUDA_EVENT 口径 + gap 备选档（manifest.kernel_gap 已存，
  未启用）；报告 §5.2 明示口径边界。
- **优先级**：P1（对 B1 结论影响 3× 量级）。

## G-D7 可分配显存预算（H200/H100/L40）

- **影响**：M06 显存与可行性——全部 feasible=unknown（标称容量 ≠ 可分配）。
- **缺什么**：torch.cuda.mem_get_info 空闲值 / MIG 关闭实测 / 驱动预留，每设备一份。
- **旧仓不足**：无此数据（旧仓直接用标称）。
- **建议**：采集脚本输出 {device, driver, free_bytes_after_init}；验收：
  DeviceBudget.allocatable_bytes 非 None 且 feasible 判定切换有测试。
- **当前降级**：allocatable=None → feasible=unknown，不参与排名（准则 §2.2）。
- **优先级**：P2。

## G-D8 H200 节内网络（NVLink）与 DT 链路实测

- **影响**：dt_serial/dt_pipeline_sim 的链路成本（当前 LINKS 为 spec/assumed 常数）。
- **缺什么**：H200 NVLink pairwise/dgx 带宽-延迟曲线（pi-viewer network/ 只有
  h100/a100/a800/a40）；DT 跨机 RDMA 实测。
- **建议**：沿用 pi-viewer network 采集格式（可跨仓复用管线）；验收：LinkSpec
  source 从 assumed 升级 spec/measured。
- **当前降级**：固定延迟+单向带宽的解析式，来源已按 spec/assumed 分列（本轮修正）。
- **优先级**：P2（对 DT 场景结论影响中等）。

## G-D9 价格/成本数据

- **影响**：异构性价比（P04）与多目标比较的成本列——全部 null+reason。
- **缺什么**：版本化单价（$/GPU-hr、电力、机架）、日期与计费范围。
- **建议**：用户提供来源后入册 assets/price/；不接受虚构单价。验收：cost 指标
  从 null 变 sourced 值并带日期。
- **当前降级**：成本 unknown（不造数）。
- **优先级**：P3（等用户决策）。
