# R30 Ascend profiling 离线审计（A-N5 第一阶段）

更新：2026-11-17。数据：外部 Ascend NPU profiler 产物
`D:\github\qwen3_6_mtp_domino`、`D:\github\qwen3-8b-0811`（未入仓的约 400 MB 采集）；
蒸馏/审计包入册 [assets/profile_audit/ascend_910b_profile_audit_v1](../../assets/profile_audit/ascend_910b_profile_audit_v1/manifest.json)。
入口：[`tools/import_ascend_profiles.py`](../../tools/import_ascend_profiles.py)、
[`simverse/cost/profile_audit.py`](../../simverse/cost/profile_audit.py)、
`py -3.12 -m simverse.cli ascend-audit --check`、证据 `ev-ascend-profile-audit-v1`。

**结论先行：本包是只读审计，`calibration_admission = not-admitted`。**
它没有、也不得产生任何成本/排名结论（`selected_ms` 恒 `null`），不改
`DEVICES`/`PROFILE_SETS`/模型执行状态/部署资格，不解除 R17/R18/G-D7/R02b。
A-N5 的"原始 trace 审核、映射与差异表"已交付；生产校准准入依赖
[A-N4 设备/实现档](../ascend-validation-plan.md)与匹配采数（计划 §1/§3）。

## 1. 数据身份（记录到什么、缺什么）

| 项 | 值（本审计记录） |
|---|---|
| 设备 | `NPU_INFO.name = Ascend910B`（家族级；**SKU/物理卡未记录**） |
| 软件栈 | torch_npu 2.10.0.post2 / CANN 9.1.0（每 run 一致，`identity_stacks` 单栈） |
| profiler | Level1、`_data_simplification=true`、AICore 指标、NPU+CPU activities |
| 逻辑 rank/设备 | rank 0 → device_id 8（`RANK_DEVICE_MAP`） |
| 并行组 | Qwen3.6 runs：`ep` 与 `mc2` 组 8 ranks；Qwen3-8B runs：`tp` 组 2 ranks；PP/DP 未记录 |
| 会话窗口 | 36.5–69.1 s（`SESSION_TIME_INFO`）；step 节点为空 → **无逐步/阶段窗口** |
| AICore 时钟 | 1800/800 MHz 两档（仅记录，不作设备规格） |
| 源哈希 | 每 run 的 `profiler_info_0.json`、`profiler_metadata.json`、`kernel_details.csv`、`op_statistic.csv`、`step_trace_time.csv`、`ascend_pytorch_profiler_0.db` 逐文件 SHA256 在 `manifest.source.files_sha256`；`trace_view.json`/`operator_details.csv`/`communication.json` 未消费、未入哈希 |

**缺失身份（`identity.json.identity_gap_summary`，不补造）**：设备 SKU/board/驱动与固件；
vLLM / vllm-ascend commit；模型 repo/revision/config/checkpoint 哈希；量化配方声明
（只能从 kernel dtype 观察 W8A8 + MoE INT4 混合）；TP/EP/PP/DP 完整拓扑；
batch/长度/阶段/seed；eager/graph 与图捕获尺寸；warmup/active 区间、重复次数、同步策略；
step 边界（profiler 日志 `WARN step node list is empty`）；请求轨迹/E2E；采集命令。

## 2. 形状身份（形状一致 ≠ 权重身份）

| run | 候选模型 | 观测几何 | 判定 |
|---|---|---|---|
| qwen3_6_mtp_domino/mtp_4k_bs64 | `qwen36_35b_a3b` | hidden 2048；Q 16×256、KV 2；256E；GDN conv 4 | 形状一致（`shape_consistent=true`），TP hint 未知 |
| qwen3_6_mtp_domino/minidraft_4k_bs64 | `qwen36_35b_a3b` | 同上 + 同一 `FusedInferAttentionScore` 内混入 (32 heads, d128) 草稿头（640 vs 448 调用） | 目标几何一致；**草稿头记录为 contamination**（非 catalog 头，不并入目标形状） |
| qwen3-8b-0811/*（4 run） | `qwen3_8b_dense`（**不在活跃 catalog**） | hidden 4096；TP2 → 总 Q 32×128、KV 8；36 层 SwiGLU（36×73 计数吻合） | 形状一致；模型身份/revision 未证，禁止用于 Qwen3.6/V4.1 校准 |

- 0 个 run 被标记为"已证明整模身份"：形状一致只说明几何域可比，不说明权重、量化转换器或
  实现版本一致（准则 §1.1/§3.1）。
- `minidraft_4k` 同时出现目标层与非 catalog 草稿层的 kernel；未做 layer/pass 归属。

### 2.1 owner 声明的模型/方法（2026-11-17）

owner 声明：`qwen3_6_mtp_domino` → **Qwen3.6-35B-A3B**，`qwen3-8b-0811` → **Qwen3-8B**；
`mtp`/`dflash`/`dspark` 为三种现有草稿模型；`minidraft` 为自研草稿、本轮不考虑
（对应 run 标记 `excluded_by_owner`，不参与任何匹配/校准）。声明只覆盖"叫什么"：
权重 revision/checkpoint、vLLM/vllm-ascend commit、量化配方仍未验证，审计包
`identity.json.owner_declared` 与 `manifest.owner_declared` 如实分栏。

### 2.2 用具体算子输入/输出 shape 验证维度

不靠目录名猜测，直接由 kernel 的输入/输出张量维度反推并核对（`identity.json.shape_candidates`
+ `geometry_verification`，带调用计数证据）：

| 维度 | Qwen3.6-35B-A3B（mtp run） | Qwen3-8B（baseline run） |
|---|---|---|
| hidden | **2048**：`AddRmsNormBias [T,2048]` 8256 次；GEMM K=2048 | **4096**：`AddRmsNormBias [T,4096]`；GEMM K=4096 |
| attention | Q `[T,16,256]` / KV `[T,2,256]` → 16/2 heads, d=256 | Q `[T,16,128]` / KV `[T,4,128]`，TP2 → 总 32/8, d=128 |
| vocab | **248320** = `lm_head` 输出宽 N | **151936** = N 75968 × TP2 |
| MoE | `MoeGatingTopK [T,256]` + top-8 输出；grouped 权重 E=32/rank → EP8 | 无 MoE |
| intermediate | MoE 512/shared 512（grouped/shared GEMM N） | **12288**（hidden-K GEMM N） |
| 判定 | `geometry_verification.verification_passed = true` | 同左（形状一致） |

结论：hidden/heads/vocab/专家维度都可从算子 shape 交叉验证，且与 owner 声明一致；
但这仍是**形状一致**（不是权重身份）；minidraft run 的 shape 候选额外出现 1536/128
等自研头维度，正是按声明排除它的原因之一。

## 3. 方法与对账（可重放）

1. 流式解析 `kernel_details.csv`（逐 task：Type/Name/Duration/Start/Stream/Shapes/dtype），
   聚合到 OP Type；`op_statistic.csv` 作独立交叉校验。
2. **对账结果（全部 6 run）**：`op_statistic` 与 `kernel_details` 的**非 hcom 行**
   计数完全相等、总时长相对差 ≤6e-9（`census.runs[].op_statistic_consistency`）。
   hcom_* HCCL 通信核不在 `op_statistic.csv` 中，单列为 communication。
3. 通信口径分两层：`hcom_*`（HCCL 通信核）与设备侧通信类
   （`MoeDistributeDispatch/Combine`、`allreduceAicpuKernel`、`allgatherAicpuKernel`）。
   账本覆盖只对**非通信**设备时间计算（通信成本在本仓执行/链路口径另计，边界不同，不可相加）。
4. 设备内核间隙：按 stream 与全设备忙区间并集分别统计（device-kernel 边界；
   含 profiler 扰动与多 stream 重叠，**不是 host launch / E2E**）。
5. token 桶（decode ≤64 / draft-verify 65..1024 / prefill >1024）是**逐调用启发式**，
   不是连续阶段窗口（step 边界缺失）。
6. ownership：`mapping.json` 显式登记 OP Type → 语义组 → 账本行组（融合核可拥有多行）。
   **采集侧 vLLM/vllm-ascend commit 未随 profile 提供 → 不宣称 source-pinned ownership，
   只到组级聚合，行级 attribution 需 correlation id（本审计未提取）。**

## 4. 差异表：账本对设备时间的覆盖`mapped/compute` = 有账本行组归属的设备时间 / 非通信设备时间；`unmapped/compute` 为结构性缺口。

| run | kernel 行 | 设备 kernel 时间 | 非通信 | 通信 | mapped/compute | unmapped/compute |
|---|---|---|---|---|---|---|
| qwen3_6 mtp_4k_bs64 | 219,981 | 58.07 s | 13.35 s | 44.72 s（hcom 43.75） | **0.577** | **0.423** |
| qwen3_6 minidraft_4k_bs64 | 123,068 | 12.69 s | 5.15 s | 7.55 s（hcom 6.76） | **0.459** | **0.541** |
| qwen3-8b baseline bs32/8k | 44,244 | 18.42 s | 11.52 s | 6.90 s（hcom 1.54） | 0.899 | 0.101 |
| qwen3-8b dflash bs64/4k | 30,963 | 19.13 s | 11.61 s | 7.51 s（hcom 1.73） | 0.857 | 0.143 |
| qwen3-8b dspark bs64/4k | 39,268 | 17.94 s | 10.93 s | 7.01 s（hcom 1.75） | 0.863 | 0.137 |
| qwen3-8b minidraft bs64/4k | 48,175 | 20.17 s | 12.58 s | 7.60 s（hcom 1.75） | 0.813 | 0.187 |

Qwen3.6 decode 账本 27 行中 **25 行**有组级归属（`ledger_row_coverage=[25,27]`）：
`attn_output_gate` 是否融合进注意力核未知；`embed_lookup` 未在设备 kernel 中出现。

### 4.1 结构性缺口（本仓无对应账本行/实现分解不同）

- **MoE token 重排/分发/合并（无账本行）**：mtp run 的 `MoeTokenPermute+Unpermute`
  1.995 s（非通信时间的 **14.9%**）、`MoeDistributeDispatchV2/CombineV2` 另 0.967 s
  为通信类；minidraft_4k 亦为 0.312 s + 0.790 s。本仓 MoE 只计 grouped GEMM + 通信字节，
  **未计设备侧 permute/dispatch 核时间**。
- **数据搬运/量化 cast/索引（无账本行）**：mtp run `Transpose/Slice/ScatterUpdate/...`
  1.598 s（**12.0%**）、`DynamicQuant` 0.322 s（2.2%）。本仓按字节计入算子行，
  未单列这些设备任务。
- **chunked-prefill GDN 分解（实现不同）**：`ChunkFwdO/ChunkGatedDeltaRuleFwdH/
  recompute_w_u_fwd_kernel/solve_tril_16x16/...` 1.037 s（**7.8%**）。语义对应该仓
  `gdn_recurrence`/`causal_conv1d` 行，但 Ascend 是 chunked 7 核分解，不可 1:1 计价，
  未计入账本覆盖。
- **采样/接受/草稿运行时（无账本行）**：采样 0.186 s、草稿后端 0.022 s（mtp）；
  Qwen3-8B runs 的 `_resample_kernel/_prepare_dflash_inputs_kernel_ascend/` slot
  mapping/采样辅助等占非通信时间 **10.7–13.0%**（dflash/dspark/minidraft；baseline
  无草稿仅 0.02%）。本仓把接受律作为显式假设、草稿按账本建模，未计设备采样核。
- `mapping.json` 覆盖本次观测到的**全部 126 个 OP Type**（0 个未登记）：无法归入
  目标账本的（草稿运行时/采样/搬运/量化 cast）全部显式列入未建模组，不静默丢弃。
- 结论口径：以上是**该 Ascend 栈（含其融合/量化实现）上的结构性缺口**，
  不能反推 NVIDIA 实现缺少同量时间；因为设备/实现域不匹配，本审计不把这些缺口
  折成倍率或成本项（禁止全局倍率与跨设备继承）。

### 4.2 通信占比（为什么这批 run 不能直接当 TPOT 校准）

`mtp_4k_bs64` 设备时间 75.3% 为 hcom 通信（alltoallv 34.77 s + allGather 8.99 s，
通信/计算 ≈ 3.3×）；`minidraft_4k` 为 53.2%。Qwen3-8B TP2 runs 的 TP AllReduce 占
29–30%。这些比例同时依赖 EP/TP 拓扑、请求批与图模式，**没有请求轨迹/step 边界时
不能把设备 kernel 时间当在线 TPOT**。

### 4.3 计时边界与内部不一致

- 设备内核间隙：全设备忙区间并集的中位 **1.0–1.25 µs**、p90 4.25–124.75 µs；
  busy 占会话窗 61–84%（`census.runs[].union_gap_stats`）。这是 device-kernel 边界测量，
  不能与 CUDA_EVENT/H200 的 1.3493 µs eager 间隙互换或相减。
- `kernel 总时长 / 会话墙钟` = 0.26–0.92：采集含编译/预热/等待等非 steady 区，
  再次说明缺 warmup/active 标记时不能宣称统计稳定。
- `step_trace_time.csv` 与 `analysis.db.StepTraceTime`：stage 总数一致（相对差<1e-4），
  但 `computing` 归类不同——如 mtp run CSV computing 14.13 s vs DB 36.00 s
  （差 ≈ 21.88 s，即通信溢出/overlap 归属不同），8B runs 亦有 39–110 ms 差异。
  两份输出**原件并列记录、不换算、不相减**（准则 §2.4）。

### 4.4 row-level 候选提取（`ledger_rows.json`，仍为 audit-only）

对已声明的活跃模型 Qwen3.6-35B-A3B 的 **MTP run**，用 GEMM shape 签名 + 调用计数方程
把设备时间摊回账本行（minidraft 按声明排除；Qwen3-8B 不在活跃 catalog，不做跨模型
行级匹配）：

- 计数反解：`in_proj_ba` 调用 2880 ÷ GDN 层 30 = **96 passes**；`attention` 调用 1248 ÷ 96
  − 全注意力层 10 = **MTP K=3**（登记为 derived-from-counts，非 step 边界证明）。
- 融合核按**输出宽度**分摊：GDN `in_proj_qkvz`(12288) → `in_proj_qkv` 648.5/972.7 ms、
  `in_proj_z` 324.2/972.7 ms；MTP `qkv+gate`(9216) 按 8192:512:512；
- 同形核按**计数方程**分摊：`(4096→2048)` 4416 次 = o_proj 10 + out_proj 30 + MTP o 3 +
  mtp_fc 3（×96），精确匹配；router/shared/shared_gate 同样由 (40 层+MTP 步) 方程对上。
- 结果：12 个 GEMM/注意力签名行中 **9 行**得到候选 measured 值（`attention_core` 按
  10:3 计数分 target/draft）；候选时间覆盖 2,064.8/2,588.0 ms = **79.8%**（仅 2D GEMM +
  注意力核分母）；`lm_head` 观测 357 次 vs 预测 384 次（残差 27）显式
  `ambiguous_count_residual` 不分摊；非 GEMM 融合组（norm_add、GDN core/conv、
  qkv+norm+rope 融合、router topk 等）列入 `unallocated_groups`，**不填 0**。
- 用途：这是未来接 `CalibratedPricer` 的候选查表数据；**准入仍被 A-N4 设备身份与
  权重/引擎身份阻断**，且融合分摊未按源码 ownership pin，不能直接作为 measured 校准。

## 5. 准入裁决（`admission`）

`calibration_admission = not-admitted`，`admitted = false`。原因：

1. **设备域**：只有 `Ascend910B` 家族名，无 `npu-smi` board/SKU/驱动；本仓虽已注册
   `ascend_910b3` 解析档，但 spec 为 **assumed 旧仓映射**（model-verse H3 →
   `ascend_910B_64G`，未经官方核验），不能据此证明同一 SKU/驱动；910B3/910C/A2/A3
   能力不可互相继承（A-N4 仍需官方规格与设备身份）。
2. **实现域**：vLLM/vllm-ascend commit 未记录，kernel ownership 无法按源码 pin。
3. **模型域**：形状一致但权重 repo/revision/config/checkpoint 哈希缺失；
   Qwen3-8B 类 run 不在活跃 catalog（跨模型直接拒绝）。
4. **测量元数据**：无 warmup/active、无重复 run、step 节点为空、无请求轨迹/同步策略；
   不满足 MERGE_CRITERIA §3.2 的 measured 准入。
5. **拓扑**：Qwen3.6 runs 观测到 ep/mc2=8，但无 TP 组记录与 PP/DP；无法与场景 TP/EP 比对。
6. **阶段**：prefill/decode 只能按逐调用 token 桶近似，无连续窗口，不能做 TTFT/ITL 结论。

解除条件（最少后续工作）：A-N4 登记 Ascend 设备身份与官方 dense 规格 + 实现档；
按 [计划 §3.1](../ascend-validation-plan.md) 重采（固定 vLLM/vllm-ascend commit、权重哈希、
warmup/active 窗口、重复 run、step 标记、请求轨迹）；之后才可按 §3.4 做局部校准与留出评估。

## 6. 可复现与门禁

```bash
py -3.12 tools/import_ascend_profiles.py --check \
  --out assets/profile_audit/ascend_910b_profile_audit_v1
py -3.12 -m simverse.cli ascend-audit --check
py -3.12 -m pytest tests/test_ascend_profile_audit_r30.py \
  tests/test_ascend_profile_audit_ui.py -q
```

- 导入器确定性：JSON 固定 `sort_keys`+紧凑分隔符+LF，写后自检哈希；`--check` 只校验不重写。
- 工作台：`/api/v2/catalog.profile_audit` 索引 + `GET /api/v2/profile-audit/<set_id>`
  （只读详情，`js_safe`）；前端抽屉展示准入/覆盖并可导出 JSON；`profile_sets` 仍为空。
- 测试：hermetic 合成 fixture 覆盖导入/映射/覆盖/回退/哈希拒绝/跨模型/设备/TP/隔离；
  真实 6-run 包只读小 JSON 资产。新增 21 项（本文件 15 + 工作台 6）。

## 7. 本审计能/不能宣称

**能**：6 个 run 的原始 kernel/op 产物已离线审核、映射到账本行组，公开逐 run 覆盖与
未建模占比、设备内核间隙、身份缺口；这些是可复现、可被后续校准消费的结构化证据。
两份数据源的 op 计数/总时长对账一致（非 hcom 行），`hcom_*` 与账本边界已分栏。

**不能**：
- 不是 measured 校准：无 `selected_ms`、无成本/TPOT/排名结论，不改变任何账本与设备表。
- 不证明整模 forward/数值正确性、接受率、吞吐或可部署性。
- 不能跨 SKU/跨模型/跨 run 复用，也不能与 R28/R29（不同栈/卡位）相减或拼接。
- 不能把未建模占比（如 MoE permute 14.9%）当作 NVIDIA 实现或该仓解析模型的成本倍率。
