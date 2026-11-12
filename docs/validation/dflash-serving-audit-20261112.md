# R07a：DFlash 服务栈配对、协议与计时边界审计（G-C3）验证报告

日期：2026-11-12。分支 `r07a-dflash-serving-audit`（独立 worktree，基线 main@7928e18）。
本报告是合入入口文档之一；结论口径遵守 [合入准则](../MERGE_CRITERIA.md)。

## 0. 范围与不做的事

- **审计对象**：两套在维护的服务栈对 Qwen3.6-35B-A3B + DFlash 的集成实现——
  SGLang `DFLASH` 与 vLLM DFlash（HF README 所称 PR#40898 的 main 落地路径）。
- **逐项核对**：draft/verify 调度、hidden 传递、KV 回退、拒绝采样、通信字节、
  host/device/端到端计时边界；与本仓 MethodPlan（methods/plans.py）、
  成本行（count/draft.py、lowering/memory.py、lowering/sharding.py）逐项对账。
- **不做**：真实模型/GPU 采集（roadmap 未授权）；不把源码阅读标成 CPU 执行或
  GPU 实测；不采集真实模型或 GPU 数据；不动其他工作的未跟踪文件
  （V4.1 CSA2 参考文件在主检出保持未跟踪，本分支不包含）。
- **不依赖** V4.1 CSA2 的实现或结论。

## 1. 固定仓库、提交与文件哈希（pin）

### 1.1 vLLM 侧（与既有 VLLM_PARALLEL_PIN 同一提交）

`vllm-project/vllm @dc74b8cbbbe3dbb9399fe369757d63defd614cc2`（2026-11-10 main）。
新增 13 文件 pin（LF 归一 SHA256 逐文件见
`assets/identity/vllm_dflash_serving_pin/manifest.json`）：

| 文件（仓内名） | 源路径 | 作用 |
|---|---|---|
| dflash_speculator.py | v1/worker/gpu/spec_decode/dflash/speculator.py | DFlashSpeculator（proposer；含 _prepare_dflash_inputs 内核） |
| base_speculator.py | v1/worker/gpu/spec_decode/speculator.py | DraftModelSpeculator（sample_draft/DraftLogitsCache/local argmax） |
| cudagraph.py / config.py | …/dflash/ | FULL-only 图管理；num_lookahead_tokens=N+1 |
| eagle3_utils.py | …/spec_decode/eagle/ | target_layer_ids(+1) → aux hidden 层解析 |
| rejection_sampler.py / …_utils.py | …/spec_decode/ | 共享拒绝采样（verify 侧） |
| speculative_config.py | vllm/config/ | draft_sample_method 默认 greedy 等 |
| test_dflash{,_causality,_lookahead,_prepare_inputs}.py、test_draft_model.py | tests/ | 单测/e2e（被拒后缀 PAD 负例、因果分支、调度） |

`dflash_speculator.py` 与既有 `vllm_qwen35_pin/dflash_speculator.py` 同源
（SHA256 `f3f93c76…` 一致），两处 pin 单一事实互验。

### 1.2 SGLang 侧（新 pin）

`sgl-project/sglang @74cb973d78dfa4af85a90d2841891fec27e78db2`
（2026-11-12 main）。12 文件 pin（`assets/identity/sglang_dflash_pin/manifest.json`）：
`dflash_worker_v2.py`（v1 worker 文件在 pin 提交已不存在）、`dflash_info.py`/
`dflash_info_v2.py`、`dflash_utils.py`、`dflash_dp.py`、`dflash_disaggregation.py`、
`spec_info.py`、`models_dflash.py`（srt/models/dflash.py）、`kernels_dflash.py`
（kernels/ops/speculative/dflash.py）、`dflash_candidates.py`、两个 e2e 测试。
两处 `dflash.py` 重名按 `models_`/`kernels_` 前缀落盘（manifest 的 paths 字段
保留源路径）。

### 1.3 阅读的测试（两侧）

- vLLM：`test_dflash_prepare_inputs.py`（**被拒后缀 PAD/空块不可写负例**、
  kernel 单编译变体）、`test_dflash_causality.py`（非因果分支表）、
  `test_dflash_lookahead.py`（调度器 lookahead=N+1）、
  `tests/v1/e2e/spec_decode/acceptance_rates/dflash/test_dflash.py`
  （Qwen3-8B-DFlash-b16：GSM8K accuracy≥0.8、acceptance≥3.5；
  acceptance lengths 对照论文 Table 1：mt-bench 4.24/humaneval 6.50/gsm8k 6.54×0.975）。
- SGLang：`test/registered/spec/dflash/test_dflash.py`（GSM8K≥0.75、
  accept≥2.8；overlap 与非 overlap 变体；grammar/logprob/matched-stop kits）、
  `test_basic_sanity_dflash.py`。

## 2. 逐项核对矩阵（两栈实际路径）

| # | 审计项 | vLLM @dc74b8cb | SGLang @74cb973d | 本仓原口径 | 结论 |
|---|---|---|---|---|---|
| 1 | draft/verify 调度 | 每 engine step：目标 verify 前向 → in-process `propose()`；每请求 1 bonus+N mask（`num_query_per_req=1+num_speculative_steps`）；FULL CG only（PIECEWISE 不支持）；uniform token count 避免 CPU-GPU 同步 | `DFlashWorkerV2` 同驱 overlap/非 overlap（spec-v2；FutureMap 中继）；轮=prepare 块→草稿前向（TARGET_VERIFY 模式非因果、独立 draft runner）→借 lm_head 产候选→目标 verify（因果 capture FULL）→accept→ctx KV→next_draft_input | plan_dflash 五步（draft_block/d2t/verify_block/commit_rewind/hidden_extract） | **结构一致**；dt_serial/dt_pipeline 无栈对应（见 #6） |
| 2 | hidden 传递 | aux hidden（`dflash_config.target_layer_ids`，+1 capture-after，eagle3_utils L77-114）→ `combine_hidden_states`=fc(concat)（qwen3_dflash L948-967）；T→D 同卡零拷贝 | verify `capture_hidden_mode.FULL` 全跨度 hidden → `project_target_hidden`=fc(ReplicatedLinear)+hidden_norm（models/dflash L1193-1208）；prefill 后即刻物化草稿 KV（radix 安全） | T→D 消息按 E[N]·8 层·h 字节（仅 dt 链路计费） | **一致**（8 层拼接+fc 复制权重；两栈 fc 均 ReplicatedLinear 全宽） |
| 3 | ctx-KV 投影跨度 | **整 verify 跨度 B×blk 行投影**（`_precompute_context_kv(0,num_target_tokens)`；FULL 图 `_num_graph_context_tokens=num_reqs×(N+1)`）；被拒后缀 PAD 不落缓存 | **整跨度**（全跨度 hidden→fc→逐层 kv_proj_only/融合前缀直写；commit_lens 前缀写） | 按 produced=B·E[N] 行计（HF [:produced] 口径） | **F1 漏计（已修）**：两栈均按跨度执行 → 本仓改跨度基；kv_write 仍 committed |
| 4 | KV 回退 | 草稿：无物理 crop，被拒后缀 `PAD_SLOT_ID` 不写（测试负例在册）；目标：v1 通用 spec-decode 路径 | 草稿：只写 committed 行；`new_seq_lens=prefix+commit`；**mamba/GDN 目标态回退**（last_steps=commit−1 + extra_buffer 策略） | cache_policy 文字 + kv_rewind_waste 行（浪费期望） | 草稿/目标 KV 语义一致；**GDN 态回退拷贝成本未计价（unknown，登记）** |
| 5 | 拒绝采样 | 共享 RejectionSampler：greedy=target argmax 前缀匹配；random=log p>log u+log q；bonus=目标分布；`draft_sample_method` 默认 greedy，probabilistic 需 DraftLogitsCache | greedy Triton accept（前缀匹配+bonus=argmax）；**采样需 CUDA JIT 可用性**，不可用降级 greedy（分布不保持，显式 warning） | transitions：贪心 argmax/随机残差重采样+bonus | 语义一致；**SGLang 采样可用性门槛登记**（divergence） |
| 6 | 通信字节 | 草稿/验证 logits：**全量 gather**（LogitsProcessor；(max,id) 局部归约 `use_local_argmax_reduction` 存在但默认关且与 probabilistic 互斥）；colocated 零链路字节 | 草稿 greedy：**每 rank 局部 (max,id)→all-gather→首最大**（非全量）；greedy verify 可整跳 logits 聚合（deferred_argmax）；候选 top-k 走 [k vals\|k ids] 单次 all-gather | logits_gather 按 vLLM 全量口径（draft B×(blk−1)、verify B×blk 行） | **两栈分歧（登记不改行）**：本仓 pinned vLLM 口径；SGLang 字节显著更低（O(tp·k) vs O(V)），差异公式入 source_map |
| 7 | 部署形态 | 仅 colocated（proposer in-process，同 TP 组） | SINGLE_GROUP 同 TP 组；disagg=target 侧 PD（bonus/seq_lens FutureMap 中继），**非 DT** | dt_serial/dt_pipeline_sim 建模模式 | **dt 模式 unsupported**（无栈实现，建模假设如实标注） |
| 8 | 计时边界 | 采样/prepare 均 in-graph GPU kernel；uniform token 数避免 host 同步；e2e=engine 指标（acceptance_len） | 同（in-graph sampler；fwd_occupancy 计入）；README 基准口径：**吞吐=输出 token/墙钟（含 prefill+调度）**、5 次运行、accept=completion/spec_verify_ct | commit_rewind/hidden_extract 0.0+unknown | 结构对齐：两栈把 accept/prepare 放 GPU 图内；kernel 耗时仍无独立计价 → **保持 unknown（注记升级）** |

## 3. 对账结论与修正（可重放）

### F1（漏计，已修）：ctx 投影按 verify 跨度而非 committed 前缀

**证据**：上表 #3（vLLM `_precompute_context_kv` 全跨度 + SGLang 全跨度前缀写；
两栈只对 committed 行落缓存）。
**修正**：`count/draft.py` 的 `dflash_fc`/`dflash_hidden_norm`/`ctx_kv_proj`
计算与 act 读写改按 `span=B×blk`；`kv_write` 保持 produced（B·E[N]）。
`lowering/memory.py` 的 phase:draft 临时（fc 输入/ctx 归一/ctx 投影输出/
回退 qkv 暂存）同步跨度基。HF 参考实现的 `[:produced]` 前缀口径是**较少
工作**的路径，作为差异注记保留在 source_map。
**独立 oracle**：tests/test_dflash_serving_r07a.py
`test_fc_hidden_norm_ctx_proj_span_basis_oracle`（手算
fc=2·16·16384·2048=1,073,741,824 FLOPs；ctx/层=16,777,216；kv_write=3×256）
与 `test_span_correction_is_strictly_upward_not_kv_double`
（负例：投影与 E[N] 无关、kv_write 严格线性、草稿层 token 行不受影响——防重复计费）。
既有测试同步：test_draft_ctx_kv_paths.py（3,145,728→16,777,216 等）、
test_colocated_memory.py（t-draft-buffers 跨度基）。

### F2（双口径计费，已修）：草稿行按生效块而非 checkpoint 原生块

重放对账发现：场景 `k` 截断或 `block_size` 覆盖时（默认场景 k=4 → 生效块 5），
`dflash_round_rows` 仍按 `draft.block_size=16` 计草稿行，而 verify/msg/collectives
按 `plan.block_size=5` —— 草稿阶段按 16 行计费、verify 按 5 行验证的双口径。
两栈均按**生效块**执行草稿前向（vLLM num_speculative_tokens；SGLang
`--speculative-dflash-block-size` + `set_block_size`）。
**修正**：`dflash_round_rows` 增显式 `block_size` 参数，pipeline 传
`plan.block_size`。**负例**：
`test_draft_rows_follow_effective_block_not_checkpoint`（k=4→q_proj 按 B×5；
k=7→B×8）。

### F3（分歧，登记不改行）：logits/候选的字节口径两栈不同

vLLM（本仓 pinned 口径）：草稿候选与 verify 各一次全量 logits gather
（TP>1，每 rank 线上界 (tp−1)/tp×T×V×act）。SGLang：greedy 草稿每 rank 局部
(max,id)→all-gather（O(tp·(dtype+8)·行)）；greedy verify 可跳过聚合
（deferred_argmax）；候选 top-k 交换 [k vals|k ids] 单次 all-gather。
本仓成本行保持 vLLM pin 口径（implementation_id 即 vLLM），SGLang 差异公式
登记于 `sm-sglang-dflash-serving.uncovered_domain`——**不用另一引擎的实现
替换本仓行，也不用零值**。

### F4（证据不足 → 保留 unknown）

- accept/bonus/prepare-inputs 的 GPU kernel 耗时：两栈 in-graph，无独立计价
  → `stage_cost_complete` 保持 unknown（注记升级为"in-graph GPU kernel，
  非 host 传输"）。
- SGLang 采样验证 kernel（`compute_dflash_sampling_correct_drafts_and_bonus`）
  的运行时可用性（构建/设备相关）→ unknown；不可用时降级 greedy 且分布不保持
  （部署注意事项，登记于 plan approximations）。
- SGLang GDN/mamba 目标态回退（extra_buffer）拷贝成本 → unknown。
- vLLM DraftLogitsCache（probabilistic：`[max_num_reqs, N, V]`，sharded 按
  owned rows）未入 R02b 账本：场景无 `draft_sample_method` 输入键，按
  "config 完整键逐项登记"规则不擅自加键 → 登记缺口（source_map uncovered +
  G-C3 剩余），由后续（如 R07b schema）接。
- FULL CUDA graph 的 batch 取整 padding 行：按未取整跨度计，图模式下界。

## 4. dflash_serving_integration / G-C3 判定

- **colocated → supported（source-checked）**：双栈源码路径完整
  （调度/hidden/KV 回退/拒绝采样/共享 embed+lm_head 均在 pin 内核与 e2e 测试
  覆盖）；注记显式"source-checked ≠ 已部署实测：served 标记仍需 R07b 真实
  trace"。证据引用 `ev-vllm-dflash-serving`/`ev-sglang-dflash-serving` 随
  dflash 场景加入。
- **dt_serial / dt_pipeline_sim → unsupported**：两栈均无 draft/target 分离
  部署（vLLM in-process；SGLang SINGLE_GROUP + target 侧 PD）；模式是本仓
  建模假设，capability 如实标注，不暗示已有部署形态。
- `PENDING_ASSETS["qwen36_dflash_serving_integration"]` 状态翻转为
  `source-checked-2026-11-12-r07a`（reason 指向证据与 R07b 残留）。
- **G-C3 解除程度**：逐行核对+协议字节对账+两处可重放修正已交付 →
  核对部分解除；剩余（DraftLogitsCache 账本、SGLang 采样可用性、GDN 态回退
  成本、真实 served 证据）转 G-C3 残留/R07b。

## 5. 重放与差异归因

命令：`py -3.12 -m simverse.cli run-v2`（必需场景）与
`py -3.12 -m pytest tests -q`。

- **必需场景（CLI run-v2，k=15/blk=16，B=1）**：F2 不触发（k_limit=15=
  原生 blk−1）；F1 生效——投影行 produced 3.55→span 16（fc FLOPs ×4.51）。
  实测 M01 36.8591→36.8592 ms（+0.0003%）：B=1 下草稿阶段为权重读受限，
  投影增量对时间影响可忽略，但对账本行是 4.5× 的漏计修正。
- **默认 ScenarioV2（k=4/blk=5，α=0.75 → E[N]=3.05）**：F2+F1 同时触发。
  账本行：草稿 token 行（q/k/v/o/MLP/norm/rope/lm_head/noise）16→5 行/请求
  （×5/16），投影行 produced 3.05→span 5（fc FLOPs 1,073,741,824×5/16 块
  折算后按生效块）；kv_write 不变。实测 draft 阶段 17.5660→17.5648 ms、
  M01 42.81079→42.81038 ms（同上，B=1 权重读受限；修正是账本正确性而非
  本场景时延）。行级比例可独立复算（span/produced=blk/E[N]），
  见 `tests/test_dflash_serving_r07a.py` 第 5/6 节。
- **吞吐口径提醒**：两修正在 B≫1（计算受限）场景下才显著改变阶段时延；
  默认单请求场景的时间不变不构成"无影响"结论。
- **校准/G7**：无 profile 变更；原始计数行仅按上述两修正变化，方向与幅度
  均有独立 oracle 锁定。
- 全量门禁：见 §7。

## 6. 影响域

- 代码：`simverse/count/draft.py`（F1/F2）、`simverse/lowering/memory.py`（F1）、
  `simverse/pipeline.py`（capability 分模式/证据引用/F2 传参/注记）、
  `simverse/methods/plans.py`（approximations 更新）、
  `simverse/importers/source_map.py`（SGLANG_DFLASH_PIN + 两条服务栈条目 +
  round-protocol 更新）、`simverse/model_ir/catalog.py`（pending-asset 状态）、
  `simverse/validation/registry.py`（两条证据）。
- 资产：`assets/identity/vllm_dflash_serving_pin/`（13 文件+manifest）、
  `assets/identity/sglang_dflash_pin/`（12 文件+manifest）。
- 测试：新增 `tests/test_dflash_serving_r07a.py`（13 项）；同步
  `tests/test_draft_ctx_kv_paths.py`、`tests/test_colocated_memory.py`
  oracle 至跨度基/生效块。
- 文档：本报告、capability-gaps G-C3、IMPLEMENTATION_STATUS、ROADMAP R07a。

## 7. 测试与命令

```
py -3.12 -m pytest tests -q                 # 全量（收集 277；本报告时点全绿，见状态文件）
py -3.12 -m simverse.cli run-v2             # 必需场景重放
py -3.12 -m pytest tests/test_dflash_serving_r07a.py -q   # 本轮新增 13 项
```

源码阅读证据等级：**source-checked**（非 CPU 执行、非 GPU 实测；
registry 中两条证据 executed=False）。本轮未做任何真实模型或 GPU 采集。

## 8. 剩余限制（不冒充）

1. served/已部署实测：仍无（R07b 真实 trace 与匹配环境）。
2. SGLang 采样验证可用性与 GDN 态回退成本：unknown（构建/设备相关）。
3. DraftLogitsCache 与场景 `draft_sample_method` 键：登记待接。
4. logits/候选通信字节：本仓行=vLLM 口径；SGLang 口径为登记差异（未入行）。
5. e2e 测试门槛数字（GSM8K/accept）是**两栈自己的 CI 断言**，未在本仓重放，
   不构成对本仓数值的验证。
