# A 组二轮审计报告：lm_head logits 通信与 DFlash ctx KV 二次读补漏（R01/R02a/R02b）

日期：2026-11-10（同日晚于 [A 组首轮报告](agroup-tp-ep-colocated-20261110.md) 与
[两模型 forward 审查](two-model-forward-merge-gate-20261110.md)）。本轮不新增
能力面，只对既有 A 组实现做**逐文件证据复核**，修复 4 处与 pinned 源码不符的
漏计并补独立手算 oracle。默认场景同卡可部署性判定不变（仍 unknown）。

## 1. 发现与证据（全部 source-checked，无执行/实测）

新补 pin 文件：`vllm/model_executor/layers/logits_processor.py`
@`dc74b8cbbbe3dbb9399fe369757d63defd614cc2`，SHA256
`e1fac869aef80f6c398b92d61444186d1921740dab7df41b95a260b08b4c25d2`
（GitHub raw 同 commit 获取，入 `assets/identity/vllm_qwen35_pin/manifest.json`）。
source_map 新增 `sm-vllm-lmhead-logits-gather`、`sm-vllm-dflash-ctx-kv-second-read`；
证据登记 `ev-vllm-logits-ctxkv-billing`。

| # | 缺陷（修复前） | 证据位置 | 修复 |
|---|---|---|---|
| ① | lm_head 词表行切后 logits 集合通信**完全未计**；`count/draft.py` 注释声称"logits 聚合通信在 lowering 计"但 `model_collectives` 无此实现 | logits_processor.py L90-91（`use_all_gather` 平台开关）、L122-133（`_gather_logits`：all-gather 或 gather-to-rank0）、L185-198（`_get_logits`：`lm_head.tp_size > 1 → _gather_logits`）；qwen3_5.py L345/428（target ParallelLMHead）；qwen3_dflash.py L920-935（草稿同一路径） | `lowering/sharding.logits_gather`：每 rank 线上字节上界 `(tp−1)/tp × T×V×act`（两分支同界），延迟按单向环 `(tp−1)×fixed_ms`（ring allreduce 减半，登记近似）；`model_collectives` 层外计 1 次/阶段，`lm_head_tokens` 参数区分草稿候选行 B×(blk−1) |
| ② | DFlash 草稿 `draft_lm_head` 行 `weight_read=0`（"权重属目标"混淆了**容量去重**与**每次 GEMM 的读流量**） | dflash/model.py L260（`compute_logits(draft_hidden, _output_head(target))`）、L599-601（独立 `output_head(hidden)` GEMM）；vLLM dflash_speculator.py L364-371（`sample_indices` 切到 B×num_speculative_steps 行后才过 head → 行数=B×(blk−1) 维持不变） | `weight_read = h×(V/tp)×b_w`；`weight_capacity` 保持 0（容量归 target 物理对象） |
| ③ | `ctx_kv_proj` 行 `weight_read=0`（ctx 行第二次过 k/v_proj 未计） | dflash/model.py L384-387（`k_ctx`/`k_noise` 两次独立 `k_proj/v_proj` 调用后 cat）；qwen3_dflash.py L617-698（`_project_context_kv_token_major` 三分支，BF16 走融合 GEMM 读融合副本） | 每层 `weight_read = 2×h×kvh_rank×d×b_w`（融合/逐层回退字节数等价） |
| ④ | vLLM DFlash 集成把全部草稿层 K/V 行 `torch.cat` 成 `_fused_kv_weight` **新驻留副本**（非视图），账本无此对象 | qwen3_dflash.py L529-530（MXFP8 才建 ReplicatedLinear）、L541-568（`_build_context_kv_buffers`：`torch.cat` → `refill_derived`；k_norm 另堆叠 L×d）、L108-113（`_reads_raw_weights`＝未量化+非 CPU） | `draft_extra_allocations` 增 `draft-w-dflash-fused-ctx-kv`（未量化路径 dense_w≥8bit；`(Σ_layers 2·kvh_rank·d)×h×b_w + L×d×b_w`；4bit 量化走逐层回退，无副本、读流量不变） |

另修：M06 显存指标 `formula_ref` 原误写 `v2.moe_routed` → 新增 `v2.capacity`
公式条目并引用；结构化临时补 pinned 形状三件（见 §3 表）。

## 2. 支持范围变化

无扩大：TP/EP 合法组合、拒绝规则、共享对象身份、预算契约均未动。
新增建模仅覆盖已有 pin 内的两条明确配方（logits gather 的两平台分支按同一
字节上界；fused 副本只在未量化路径存在）。MXFP8 ctx 投影分支（独立
ReplicatedLinear 装载 checkpoint 行）未建模——本仓场景无该精度档，
source_map uncovered_domain 登记。

## 3. 默认场景修复前后（同输入重放，HEAD worktree 对照）

场景：CLI run-v2 内置默认（Qwen3.6+DFlash colocated TP8/EP1 8×H100 B1
ctx32768 α0.72 blk5，链路 pcie5）。前=HEAD `59eea51` 干净 worktree，后=本轮。

| 项 | 修复前 | 修复后 | Δ | 归因（独立可复算） |
|---|---|---|---|---|
| M01_tpot_baseline | 112.2127 ms | 112.9161 ms | +0.7034 | decode 每步补 1 次 logits_gather：`7×0.1ms` 环延迟 + `(7/8)×1×248320×1B ÷ 64GB/s = 0.0034ms` |
| draft_block t_ms | 16.8335 | 17.5660 | +0.7325 | logits_gather（T=4 行：0.7ms + 0.0136ms）+ 草稿二次读 66.7MB HBM（≈0.019ms roofline） |
| verify_block t_ms | 112.3233 | 113.0403 | +0.7170 | logits_gather（T=5 行：0.7ms + 0.017ms） |
| M04_round_ms | 129.1568 | 130.6063 | +1.4495 | 上一行两项之和（colocated 无 d2t/t2d） |
| M01_tpot_analytical | 42.3357 | 42.8108 | +0.4751 | `1.4495 / E[N]=3.0508` ✓ |
| draft weight_read 合计 | 77,655,040 B | 144,370,688 B | +66,715,648 | ② 63,569,920（h×V/tp）+ ③ 3,145,728（6 层×2×h×1×128）——手算逐字节吻合 |
| colocated_r0 steady | 2,827,505,088 B | 2,830,651,584 B | +3,146,496 | ④ fused ctx-KV 副本 3,145,728（1536×2048×1B）+ k_norm 堆叠 768 |
| colocated transient peak | 731,008 B | 752,704 B | +21,696 | phase:verify 增 aux hidden +81,920（5×8×2048）后 702,720 超过 phase:draft 691,958 成新 max；draft 侧 +10,934（ctx 归一副本 6,248 + 融合输出 4,686）不主导 |
| known_peak | 2,828,236,096 B | 2,831,404,288 B | +3,168,192 | steady Δ + transient Δ |
| memory status | unknown | unknown | — | 判定链路不变：G-D7 可分配预算未提供 + workspace/CUDA graph/NCCL 池无尺寸证据（缺口照旧传播，不填 0） |

## 4. 测试与验收

新增 4 项（收集数 197→201，全量 `py -3.12 -m pytest tests -q` 201 passed）：

- `test_parallel_sharding.py::test_logits_gather_hand_oracle`：字节
  `(tp−1)/tp×T×V×act`、时间 `(tp−1)×fixed + bytes/bw`、TP1 为零、
  `lm_head_tokens` 缺省=tokens/显式草稿行数两口径；
- `test_parallel_sharding.py::test_draft_second_read_traffic`：
  `draft_lm_head.weight_read == h×(V/tp)×b_w` 且 capacity=0、6 层
  `ctx_kv_proj.weight_read == 2×h×kvh_rank×d×b_w`；
- `test_colocated_memory.py::test_fused_ctx_kv_buffer_unquantized_only`：
  TP{1,2,8} 手算字节数（kvh≥tp 切分/kvh<tp 复制同式）；dense_w=4 → 无副本；
- `test_colocated_memory.py::test_transient_ctx_and_aux_hidden`：
  phase:draft/phase:verify/cross-round 三类新增缓冲逐项手算对账。

更新 2 项（逐项归因，非改期望保绿）：`test_collective_recipe_by_layer`
（counts 增 `logits_gather:1`，allreduce 80/40 不变）；
`test_independent_hand_oracle_steady_bytes` 与 `test_no_phantom_draft_head_weights`
（手算式补 fused ctx-KV 项 `(6·2·kvh_rank·d)×h + 6·d`，与实现差值
3,146,496B 独立验证吻合）。其余 195 项无数值断言受影响（G7 校准重放不变性
在 201 内通过——校准仍不改原始计数）。

## 5. 已知限制与未做

1. 未做任何硬件测量；全部结论为 source-checked + 独立数学 oracle。
2. logits gather 的 all-gather 与 gather-to-rank0 平台分支按同一每 rank
   字节上界建模（二者上界相同）；NCCL 内部缓冲仍属缺口组件。
3. 采样器（top-p/top-k、fp32 cast）在模型 forward 边界外，未计（与 target
   采样同一边界，非本轮范围）。
4. MXFP8 ctx 投影分支、EPLB/Marlin 分支维持 uncovered 登记。
5. 默认场景两缺口（G-D7 measured 预算、运行时池尺寸证据）不变，同卡
   可部署性仍 unknown——本轮不改变 R03b 结论，只修正进入判定链路的账本数值。
