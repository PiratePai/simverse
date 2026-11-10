# H200 第一条实测校准闭环报告（算子级）

> **历史归档（2026-11-10 标注）**：本报告绑定 Qwen3-30B-A3B-tiny，模型已移出活跃
> catalog；数值、日期与来源事实保持原样，不自动适用于 Qwen3.6/V4.1，禁止把本
> profile 跨模型使用。合入边界见
> [两模型审查记录](../validation/two-model-forward-merge-gate-20261110.md)。

日期：2026-11-09。执行：ZCode 会话（本地 CPU 分析与测试；**未做任何新的硬件测量**，
全部实测数字来自 pi-viewer 已有 H200 采集，经蒸馏导入）。

## 1. 选择的场景与数据身份

| 项 | 值 |
|---|---|
| 新仓模型 | `qwen3_30b_a3b`（Qwen/Qwen3-30B-A3B @ad44e777，官方 config 快照 SHA256 入册 assets/config） |
| 校准域模型 | pi-viewer `Qwen3-30B-A3B-tiny`（**16 专家缩小复制品**：层数 8↔48、专家 16↔128；逐算子形状 hidden 2048/32q/4kv/head_dim 128/inter 768/vocab 151936 与真模型一致） |
| 数据来源 | pi-viewer 工作区 @c5909a7699a1d2bb422d931bdca6cba399451f00（用户有未提交修改；本任务只读）`data/profiling/compute/h200/Qwen3-30B-A3B-tiny/` + `cpu_overhead/h200_dgx/` |
| 采集环境 | NVIDIA H200，driver 570.124.06，torch 2.8.0+cu128，CUDA 12.8（kernel_gap.csv 元数据） |
| 测量口径 | CUDA_EVENT（算子两侧 CUDA event 的 wall-clock；见 frontier/profiling/README 测量契约），BF16，TP1，FLASHINFER，KV block 16，median×20（linear）/×5（attention）重复 |
| 蒸馏产物 | `assets/profiles/h200_qwen3_30b_a3b_tiny/`（manifest + 3 数据文件，源文件 SHA256 逐个入册；`tools/distill_h200_profiles.py --check` 可重放校验） |

选择理由：三个 H200 数据集中（30B-tiny / 235B / llama2-7b），只有 30B-tiny 的逐算子
形状与新仓已 pin 的 `qwen3_30b_a3b` 完全一致；235B 数据更全（TP1/2/8、128E）但新仓
未登记该模型身份（需先补官方 config 快照，见缺口清单 G-D3）；llama2-7b 为 FP16 dense
结构不同。**本闭环不构成 Qwen3.6-35BA3B+DFlash 必需场景的任何真实性证据。**

## 2. 准入裁决（manifest.admission）

- **admit**：linear_op（6 算子 × 387 token 网格 1..16384）；attention decode-even
  （5184 点 = batch 1..64 × kv 0..4095 全网格）；kernel_gap（环境+间隙）。
- **reject**：moe.csv——复制品 16 专家 ≠ 真模型 128E，grouped GEMM/router 形状域不符；
  MoE 相关成本一律解析回退（这是本场景的主要未覆盖部分）。
- **defer**：attention prefill/true_mixed（v1 域仅 decode）；cpu_overheads（v1 组合未用）；
  235B/llama2 数据（另立闭环）。
- decode 注意力块取 4 算子（input_reshape+kv_cache_save+attn_decode+output_reshape）；
  decode 行的 attn_prefill 列为空转核（~7.5µs）、attn_kv_cache_extract 仅 prefill/mixed
  行——均不属 decode 运行时路径，已排除并在映射注记。

## 3. 算子所有权映射（源码证据）

| 实测算子 | 拥有的新仓账本行 | 证据 |
|---|---|---|
| attn_pre_proj | q_proj+k_proj+v_proj+q_norm+k_norm | pi-viewer `linear_op_impl.py` CausalSelfAttention：fused qkv_proj 定时器内含 QKNorm；qwen3_moe ∈ QK_NORM allowlist |
| input_layernorm | attn_add+attn_norm | qwen3_moe ∈ FUSED_ADD_NORM allowlist（融合残差+RMSNorm） |
| post_attention_layernorm | mlp_add+mlp_norm | 同上 |
| attn_rope / attn_post_proj / emb | rope / o_proj / embed_lookup | 直接对应 |
| attn_decode 块（4 算子） | attention_core（含 kv_write） | frontier/attention/families.py DENSE_ATTENTION_FAMILY 算子划分 |

## 4. 适用域与策略（最简可满足数据）

- 域：h200_sxm / BF16 / TP1 / FLASHINFER / block16；linear tokens 1..16384；
  decode (B 1..64, kv 0..4095)。域门禁不满足 → 整 stage 解析回退并逐条记原因
  （TP/精度/设备/形状任一不符即回退；verify 的 Q>1 注意力块域外回退）。
  2026-11-10 复审补强：形状门禁增加**算子构成**三项核对（QKNorm/全 head_dim
  旋转/无滑窗）——形状逐维一致但构成不同的模型不得套用测量块。
- 查表：linear 按 log10(tokens) 括号分段线性插值；attention per-op 双线性
  （batch 取 log 括号、kv 线性括号）后求和；域外显式 None→回退。
- 留出评估（不参与生产查表）：键 md5 哈希 15% 留出（键=算子×shape，源 CSV
  无重复行——2026-11-10 核验 387/5184 键均唯一），训练点插值预测留出点：

| 算子 | n | 中位误差 | p90 | max |
|---|---|---|---|---|
| attn_pre_proj | 54 | 0.57% | 1.5% | 4.8% |
| attn_post_proj | 59 | 1.12% | 7.2% | 16.7% |
| input_layernorm | 50 | 0.77% | 3.1% | 4.4% |
| attn_decode | 768 | 1.18% | 2.9% | 10.7% |
| attn_kv_cache_save | 739 | 1.15% | 5.0% | 238%（网格边缘点） |
| attn_output_reshape | 765 | 1.10% | 7.0% | 47.9% |

  线性算子中位 ≤1.2%、注意力块核心 attn_decode p90 2.9%；reshape/save 类小值算子
  相对误差尾部大（绝对值 µs 级），生产影响集中在小值项，已在指标假设中披露。
- 组合口径（**2026-11-10 修正，修复重复计费**）：selected = 域内实测 median 之和
  + 域外解析（差额法） + eager 间隙补贴（kernel_gap.csv 1.3493 µs × **计时区间
  边界数**）。原版按全部 kernel 数计费，把已含在 CUDA_EVENT 事件括号内的 op 内
  间隙重复计入（B8 全模型 673 kernel vs 433 区间，高估 ~0.32ms/步 ≈1.6%）；
  CUDA_EVENT 括号内是 device wall-clock（launch-bound 时含 host 空闲）。cuda_graph
  档 0.418 µs 备选未采用。**测量口径边界**（2026-11-10 补记，源码核对）：
  ①括号内含 host-launch 空闲 → 逐 op median 含固定开销地板，不能区分 host launch
  与 kernel 地板（需 KERNEL_ONLY 列，G-D6）；②CUDA_EVENT 路径无 L2 刷新
  （仅 RECORD_FUNCTION 路径有）→ 连续重复下 warm-L2 偏乐观；③重复数
  linear 3/20、attention 3/5；④kernel_gap eager 值为 host-ahead 纯设备间隙，
  用作边界补贴是低估。

## 5. 固定场景重放：前后数值对比（B/ctx 扫描，qwen3_30b_a3b baseline decode H200 TP1 BF16）

数据文件：[h200_calib_compare.json](../evidence/h200_calib_compare.json)
（生成器 `tools/h200_calib_compare.py` 可重放）。TPOT（analytic = T0 roofline
0.75 MFU 上限；selected = 实测组合，**2026-11-10 gap 修正后**）：

| B | ctx | analytic ms | selected ms | measured | fallback(MoE/lm_head 解析) | gap×区间 | 比值 |
|---|---|---|---|---|---|---|---|
| 1 | 2048 | 1.313 | 14.773 | 13.297 | 0.892 | 0.584×433 | 11.2× |
| 8 | 2048 | 5.751 | 19.879 | 14.272 | 5.023 | 0.584×433 | 3.46× |
| 32 | 2048 | 12.527 | 26.492 | 15.158 | 10.749 | 0.584×433 | 2.11× |
| 8 | 4095 | 6.089 | 20.277 | 14.670 | 5.023 | 0.584×433 | 3.33× |

（完整 9 点见 JSON：比值范围 1.95–11.59×；coverage_rows 577/674 行，全部 48 层
注意力侧实测；时间口径覆盖 B8=71.8%——分母含解析回退与间隙估计，非真实性度量。）
修正前（v1 按全部 kernel 数计 gap）：B8 selected 20.203ms——差额 0.324ms 即被
剔除的重复计费，与 (673−433) 个区间内间隙 × 1.3493µs 手算一致。

**差异解释（相对当前 CUDA_EVENT 组合估计的结构归因，不是对部署延迟的实测结论）：**
1. T0 解析档相对 CUDA_EVENT 组合估计低 1.95–11.59×：解析假设 roofline max+ρ·min
   且 MFU≤0.75，无 launch/固定开销项；实测侧逐 op median 含每 op 固定开销地板
   （如 attn_pre_proj 在 T=8 实测 117µs，权重读下界仅 ~5µs，4.8TB/s HBM 也救不了
   2048×6144 的小 GEMM）。**host launch 与 kernel 地板的贡献在此口径下不可分**
   （需 KERNEL_ONLY 列判别，G-D6）——差异的一部分可能是测量口径（launch-bound）
   而非设备行为；无 E2E 对照（G-D5），不宣称实际部署延迟被低估相同倍数。
2. measured 部分对 B 几乎不敏感（13.3→15.8ms，B1→32）：CUDA_EVENT 逐算子计时的
   每 op 固定开销地板主导。这是**测量口径的一部分**（eager、逐 op event、
   warm-L2 无刷新），不是 kernel 纯执行时间；graph 重放运行时会显著低于此口径
   （kernel_gap 数据本身证明 eager↔graph 差 3.2×）。已按准则 §2.4 分栏披露，
   禁止把本口径数字当 graph 运行时预测。
3. MoE 部分（本模型主导成本）不在域内：fallback 随 B 增长（union 权重读），仍是
   解析值——覆盖度按行 577/674、按时间 B8 71.8%（measured/selected，依赖估计）。

## 6. 验证与测试（本会话实际运行）

- `tools/distill_h200_profiles.py --check`：产物哈希与 manifest 一致（可重放）。
- `tests/test_calibration_h200.py`（12 项）：蒸馏值=源 CSV 原值抽查；篡改产物拒绝
  加载（ProfileIntegrityError）；网格外显式 None；stage 组合独立复算（selected =
  measured+fallback+gap，measured 由蒸馏文件+48 层重推）；TP8/精度/模型形状域门禁
  负例；MoE 未覆盖且回退>0；指标 result_kind=measured-composite、evidence 含
  measured+assumed、留出报告随 bundle 披露；chain_draft verify 混合覆盖
  （线性 measured、注意力 Q>1 回退）。
- `tests/test_merge_gate.py` G7：固定场景开/关 cost_profile，work 账本逐字段相等、
  解析 TPOT 不变、run_id 改变（成本身份变更→新身份，重放语义正确）。
- 全量 `py -3.12 -m pytest tests -q`：见 [IMPLEMENTATION_STATUS.md](../IMPLEMENTATION_STATUS.md)。

## 7. 本闭环能/不能宣称的结论

**能：**
- H200（该环境栈）上 Qwen3-30B-A3B 逐算子形状的 decode 侧 linear+attention 实测
  校准闭环已打通：导入可追溯、域门禁/回退工作、原始计数不受校准影响、留出误差已报告。
- T0 解析档相对当前 CUDA_EVENT 口径组合估计在小 batch decode 上低 1.95–11.59×
  （9 点网格）；这是对**解析档与该测量口径组合差异**的实测约束，此前只能猜测。
  gap 补贴重复计费已于 2026-11-10 修复（区间边界口径）。

**不能：**
- 不宣称端到端实测验证（无 E2E 对照数据，见缺口 G-D5）。
- 不宣称实际部署延迟被低估 1.95–11.59×——该倍数是"解析 vs CUDA_EVENT 组合估计"
  的相对差，launch-bound 测量口径贡献了其中不可分的一部分（需 G-D6 KERNEL_ONLY
  列与 G-D5 E2E 记录后才能归因到部署行为）。
- 不宣称 Qwen3.6-35BA3B+DFlash 场景性能可信（不同模型/方法，未校准）。
- 不宣称 TP>1、非 BF16、prefill/混合批、graph 重放口径已校准（域外显式回退）。
- MoE/router/lm_head 仍是解析值（复制品专家数不符，G-D2；235B 真 MoE 数据待
  V04 分片口径后准入，见 oldrepo-data-audit-20261110.md）。
- CUDA_EVENT 逐 op 口径含每 op 固定开销，不能直接当 CUDA graph 运行时延迟
  （备选档已在 manifest，未启用）。

## 8. 解除限制的最少后续工作

1. ~~235B 数据接入（需先登记官方身份）~~ → **2026-11-10 已完成身份与 TP2/8
   linear/attention 闭环**（[h200-loop2-235b.md](h200-loop2-235b.md)）；
   其 MoE（真 128E）待 V04 per-rank 分片账本后准入（首选增量路径）。
2. prefill/true_mixed 蒸馏与 verify 块形状建模（多 query 前缀注意力）→投机 verify
   步可进入实测域（G-D4）。
3. KERNEL_ONLY+cuda_graph 口径作为第二校准列（运行时口径），与 CUDA_EVENT 并列披露（G-D6）。
4. H200 网络数据（当前仅 h100/a100/a800）与 E2E 请求轨迹（G-D5、G-C4）。
