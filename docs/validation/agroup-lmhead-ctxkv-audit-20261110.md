# A 组二轮审计报告：lm_head logits 通信与 DFlash ctx KV 二次读补漏（R01/R02a/R02b；含三轮 ctx-KV 路径修正）

日期：2026-11-10（同日晚于 [A 组首轮报告](agroup-tp-ep-colocated-20261110.md) 与
[两模型 forward 审查](two-model-forward-merge-gate-20261110.md)）。本轮不新增
能力面，只对既有 A 组实现做**逐文件证据复核**，修复 4 处与 pinned 源码不符的
漏计并补独立手算 oracle。审计对照场景同卡可部署性判定不变（仍 unknown）。

> **2026-11-11 第三轮修正（外层验收前）**：③④ 的路径判据有误——
> `dense_w≥8` 不能推出 `UnquantizedLinearMethod`；逐层回退执行**完整**
> qkv_proj（含 Q），不是"读流量不变"。现已改为显式
> `draft_ctx_kv_path ∈ {fused_raw, mxfp8_context_proj, per_layer_fallback}`；
> 默认 `fused_raw` 的依据是活跃草稿 config 无 `quantization_config` + GPU
> `_reads_raw_weights`（pinned L108-113），不是位宽。修正后的手算见 §6。
>
> **复核修正（2026-11-11）**：k_norm stack 的字节仍误用了 `dense_w`；
> RMSNorm 参数由草稿 config dtype（bfloat16）决定，应为 1,536 B，与
> Q4/Q8/Q16 无关。证据与源码行号见 §6.6。

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
| ③ | `ctx_kv_proj` 行 `weight_read=0`（ctx 行第二次过 k/v_proj 未计） | dflash/model.py L384-387（`k_ctx`/`k_noise` 两次独立 `k_proj/v_proj` 调用后 cat）；qwen3_dflash.py L617-698（`_project_context_kv_token_major` 三分支，BF16 走融合 GEMM 读融合副本） | 按显式路径：raw/MXFP8 每层 `2×h×kvh_rank×d×b_w`（MXFP8 用 packed 1B/elem 下界）；**逐层回退**每层 `(q_rank·d + 2·kvh_rank·d)×h×b_w`（完整 qkv 含 Q） |
| ④ | vLLM DFlash 集成把全部草稿层 K/V 行 `torch.cat` 成 `_fused_kv_weight` **新驻留副本**（非视图），账本无此对象 | qwen3_dflash.py L529-530（MXFP8 才建 ReplicatedLinear）、L541-568（`_build_context_kv_buffers`：`torch.cat` → `refill_derived`；k_norm 另堆叠 L×d）、L108-113（`_reads_raw_weights`＝未量化+非 CPU） | 仅 `fused_raw` 路径增 `draft-w-dflash-fused-ctx-kv`（`(Σ_layers 2·kvh_rank·d)×h×b_w`）；`k_norm` 堆叠 L×d **所有路径**单列；MXFP8 独立 `_context_kv_proj` 对象；回退无副本 |

另修：M06 显存指标 `formula_ref` 原误写 `v2.moe_routed` → 新增 `v2.capacity`
公式条目并引用；结构化临时补 pinned 形状三件（见 §3 表）。

## 2. 支持范围变化

无扩大：TP/EP 合法组合、拒绝规则、共享对象身份、预算契约均未动。
新增建模仅覆盖已有 pin 内的三条明确配方（logits gather 的两平台分支按同一
字节上界；ctx-KV 的 raw/MXFP8/逐层回退按显式 `draft_ctx_kv_path` 区分）。
路径不能由位宽推断：`draft_quant.dense_w` 只是计价位宽，默认 8bit 也不等于
`UnquantizedLinearMethod`。MXFP8 ctx 投影分支现按独立 `_context_kv_proj`
对象建模 packed 8bit 权重下界；block-scale 存储布局未入 pin，作为 unknown
缺口传播，不冒充已计。

## 3. 审计对照场景修复前后（同输入重放，HEAD worktree 对照）

场景：Qwen3.6+DFlash colocated TP8/EP1 8×H100 B1、ctx32768、
α0.75、blk5、链路 pcie5。它与 CLI 内置默认（α0.72、blk16）不同。
前=HEAD `59eea51` 干净 worktree，后=本轮；2026-11-11 已用相同审计输入
重放核对 M01_tpot_analytical=42.8108 ms、M04_round_ms=130.6063 ms 和
known_peak=2,831,405,056 B。

| 项 | 修复前 | 修复后 | Δ | 归因（独立可复算） |
|---|---|---|---|---|
| M01_tpot_baseline | 112.2127 ms | 112.9161 ms | +0.7034 | decode 每步补 1 次 logits_gather：`7×0.1ms` 环延迟 + `(7/8)×1×248320×1B ÷ 64GB/s = 0.0034ms` |
| draft_block t_ms | 16.8335 | 17.5660 | +0.7325 | logits_gather（T=4 行：0.7ms + 0.0136ms）+ 草稿二次读 66.7MB HBM（≈0.019ms roofline） |
| verify_block t_ms | 112.3233 | 113.0403 | +0.7170 | logits_gather（T=5 行：0.7ms + 0.017ms） |
| M04_round_ms | 129.1568 | 130.6063 | +1.4495 | 上一行两项之和（colocated 无 d2t/t2d） |
| M01_tpot_analytical | 42.3357 | 42.8108 | +0.4751 | `1.4495 / E[N]=3.0508` ✓ |
| draft weight_read 合计 | 77,655,040 B | 144,370,688 B | +66,715,648 | ② 63,569,920（h×V/tp）+ ③ 3,145,728（6 层×2×h×1×128）——手算逐字节吻合 |
| colocated_r0 steady | 2,827,505,088 B | 2,830,652,352 B | +3,147,264 | ④ fused ctx-KV 副本 3,145,728（1536×2048×1B）+ k_norm 堆叠 1,536（6×128×2B，草稿 dtype=bfloat16） |
| colocated transient peak | 731,008 B | 752,704 B | +21,696 | phase:verify 增 aux hidden +81,920（5×8×2048）后 702,720 超过 phase:draft 691,958 成新 max；draft 侧 +10,934（ctx 归一副本 6,248 + 融合输出 4,686）不主导 |
| known_peak | 2,828,236,096 B | 2,831,405,056 B | +3,168,960 | steady Δ + transient Δ |
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
4. MXFP8 ctx 投影分支已按独立对象建 packed 下界，但 block-scale 布局
   （modelopt.py 未入 pin）与 EPLB/Marlin 分支仍 uncovered。
5. 默认场景两缺口（G-D7 measured 预算、运行时池尺寸证据）不变，同卡
   可部署性仍 unknown——本轮不改变 R03b 结论，只修正进入判定链路的账本数值。

## 6. 第三轮修正：ctx-KV 实现路径（2026-11-11）

### 6.1 两处确认缺陷

| # | 修复前 | pinned 证据 | 修正 |
|---|---|---|---|
| ① | `draft_extra_allocations` 以 `dense_w>=8` 推断 `_fused_kv_weight` 存在；4bit 路径连 `_k_norm_weights` 堆叠也漏计 | qwen3_dflash.py L108-113（`_reads_raw_weights` = 非 CPU 且全部 qkv 为 `UnquantizedLinearMethod`）；L529-531（全层 MXFP8 → 独立 `_context_kv_proj`）；L555-568（raw 才 `torch.cat`，k_norm **恒** `torch.stack`）；L679-697（三分支执行） | 新增显式 `draft_ctx_kv_path`；raw 才建融合副本，MXFP8 建独立对象，回退无副本；k_norm 堆叠单列且三路径都有 |
| ② | `ctx_kv_proj` 对 4bit/其他量化回退只计 K/V 投影 | qwen3_dflash.py L688-697：回退逐层执行完整 `proj(normed_context_states)`，再 `qkv[:, q_size:]` 切 Q | 回退按完整 qkv 输出计 FLOPs/权重读/临时缓冲；raw/MXFP8 仍只 K/V 行 |

位宽只是**计价位宽**，不是 `quant_method`；默认 `dense_w=8` 也不能推出
`UnquantizedLinearMethod`。默认场景之所以是 `fused_raw`，依据是活跃草稿
config 无 `quantization_config`（`assets/config/qwen36_35b_a3b_dflash__config.json`）
且部署在 GPU；量化档必须显式覆盖路径，`4bit + fused_raw` 在场景校验拒绝。
**这是场景/研究假设（config source-checked、未执行）**：它不证明实际
vLLM 部署以“8bit unquantized”运行，也不把 `dense_w=8` 计价档冒充为
部署精度证据；若实际服务栈覆盖 dtype/quant_method，需另取运行配置证据
并显式改 `draft_ctx_kv_path`。

### 6.2 路径与证据边界

| `draft_ctx_kv_path` | 源码分支 | ctx 执行 | 额外常驻对象 |
|---|---|---|---|
| `fused_raw` | `_reads_raw_weights` 真（L108-113） | 全层 `_fused_kv_weight` 一次融合 GEMM，仅 K/V 行（L679-683） | 融合 K/V cat 副本 + k_norm stack |
| `mxfp8_context_proj` | 全层 qkv 为 `ModelOptMxFp8Config`（L529-531） | 独立 `_context_kv_proj`，仅 K/V 行（L684-687） | 独立 MXFP8 投影（packed 8bit 下界；block-scale 未入 pin）+ k_norm stack |
| `per_layer_fallback` | 其余量化打包/CPU repack | 逐层完整 qkv_proj 后切 Q（L688-697） | 无投影副本；仅 k_norm stack |
| `unknown` | — | 不猜；计数/容量层显式拒绝 | — |

### 6.3 独立手算（TP8、B1、E[N]=3、h=2048、Q=4 头/rank、KV=1 头/rank、d=128）

| 路径 | 每层 ctx FLOPs | 每层权重读 | 6 层权重读 | 常驻对象（本 rank） |
|---|---:|---:|---:|---|
| `fused_raw`（Q8 计价） | 2·3·2048·256 = **3,145,728** | 256·2048·1 = 524,288 B | 3,145,728 B | 融合副本 3,145,728 B + k_norm 1,536 B |
| `mxfp8_context_proj` | 同上 **3,145,728** | packed 1B/elem 下界 524,288 B（scale 未计） | 3,145,728 B（下界） | 独立投影 3,145,728 B + k_norm 1,536 B |
| `per_layer_fallback`（Q4） | 2·3·2048·(512+256) = **9,437,184** | 768·2048·0.5 = 786,432 B | 4,718,592 B | 无投影副本；k_norm 1,536 B |

回退的临时缓冲另含逐层完整 qkv（Q 随后切掉）暂存：
`B·E[N]·L·(Q512+2×KV128)·1B = 13,824 B`，再加 stack 后 K/V 4,608 B。
对比修复前：回退每层只计 3,145,728 FLOPs，少计 Q 的 6,291,456/层；
每层权重读在 Q4 下少计 262,144 B（6 层少 1,572,864 B）。
默认 raw 路径的 ③④ 原有数值（融合副本 3,145,728 B、k_norm 1,536 B、
draft weight_read +66,715,648 B）不变，但对象已拆分为两个 id 并改由
显式路径选择。k_norm 的 dtype 证据见 §6.6：不能用 `dense_w` 计。

### 6.4 验证

新增 `tests/test_draft_ctx_kv_paths.py`（11 项）：raw/MXFP8/回退的
FLOPs、权重读、常驻对象、k_norm 三路径、unknown 拒绝、位宽不选分支、
回退 Q 暂存、同输入 raw↔fallback 单对象差、MXFP8 缺口传播、场景校验。
另更新 `test_parallel_sharding.py::test_draft_second_read_traffic`、
`test_colocated_memory.py` 的手算与路径入参、`test_merge_gate.py` 的
G13 调用。实际命令见最终交付报告；未运行全量 pytest（外层 Codex 负责）。

### 6.5 剩余限制

1. 未做任何硬件测量；结论为 source-checked + 独立数学 oracle。
2. MXFP8 的 block-scale 存储字节未入 pin：权重读/驻留为 packed 8bit 下界，
   `draft_ctx_kv_weight_bytes` 标 unknown 并传播到峰值完整性。
3. 量化 qkv 的 scale/zero 元数据与 CPU repack 细节未建模；回退只按完整
   qkv 的权重/输出几何计。
4. 默认场景可部署性仍 unknown（预算与运行时池缺口未变）。

### 6.6 k_norm dtype 复核修正（2026-11-11，外层验收）

复核指出 §6.3 的 k_norm 仍按 `dense_w` 计（Q8=768、Q4=384），错误。
证据链（同一 pin commit `dc74b8cbbbe3dbb9399fe369757d63defd614cc2`）：

- `vllm/model_executor/layers/layernorm.py` L42-58：`RMSNorm` 未显式传
  `dtype` 时 `weight_dtype = dtype or torch.get_default_dtype()`；
- `vllm/model_executor/model_loader/base_loader.py` L68-85：
  `load_model` 在 `set_default_torch_dtype(model_config.dtype)` 内
  `create_model`；
- 本仓 pin `assets/identity/vllm_qwen35_pin/dflash_utils.py` L51-54：
  `get_model(vllm_config=draft_vllm_config, model_config=draft_model_config)`；
- `qwen3_dflash.py` L288-289 `RMSNorm(...)` 未传 dtype；L565-568
  `torch.stack` 继承原 weight dtype；
- `assets/config/qwen36_35b_a3b_dflash__config.json` 声明
  `"dtype": "bfloat16"`。

因此 `_k_norm_weights` = 6×128×**2** = **1,536 B**，与
`draft_quant.dense_w` 是 4/8/16 无关；量化 GEMM 的计价位宽不能复用为
RMSNorm 参数宽度。实现上 `DraftIR` 新增 `dtype_hint`（parser 从 config
`dtype`/`torch_dtype` 登记），`lowering/memory.py` 用 source-backed
`dtype_bytes` 计算；dtype 缺失/不支持时显式拒绝，不回退到 `dense_w`。
审计对照场景 steady 因 k_norm +768 B 变为 **2,830,652,352 B**，known_peak
**2,831,405,056 B**；回退与 MXFP8 路径的 k_norm 同样为 1,536 B。
两份 vLLM 源码本轮未入本地 pin（仅按固定 commit + 行号登记证据；
shell 网络下载被拒），故 registry 证据条目注明行号与来源 commit。
