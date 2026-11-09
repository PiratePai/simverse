**结论**：三段流水、先语义/lowering、证据分级、单请求闭环的方向成立；但送审稿把若干“代理/假设”写成了“已验证事实”。其中 **DSpark 语义、B=1 的 max(D,V) 流水线、V4.1 配置覆盖** 是 P0，足以改变 0.909/1.368 这类头条数字，建议先做硬证据闸门，再谈成本与 UI。

## 一、必须追问的 12 个问题

### P0-1｜`dspark` 到底是官方 DSpark，还是链式代理？

- **被挑战的假设**：`dspark` = 链式 `S=K, M=1`、MoE 128E top-3、FP8、草稿不需要 target hidden、batch 可到 512。
- **源码反例/锚点**：
  - 官方 `forward_dspark_draft_block` 一次处理 **block**，`is_causal=False`，随后 `past_key_values_draft.crop(start)`：`docs/rebuild-20261106/references/DeepSpec/deepspec/eval/dspark/draft_ops.py:22-45`。
  - `build_dspark_proposal` 显式 `assert draft_input_ids.size(0) == 1`，即 proposal 只支持 B=1：`draft_ops.py:105`。
  - 草稿注意力需要 `target_hidden_states`，来自 target 层 37/38/39：`draft_ops.py:35`、`evaluator.py:93-97`、`evaluator.py:139-147`。
  - 提供的官方 decoder 使用 `Qwen3MLP`（dense），不是 128E MoE：`modeling.py:159`；采样走 `markov_head.sample_block_tokens` 与 confidence prefix：`modeling.py:328`、`draft_ops.py:82-93`。
  - 官方 evaluator 以 `torch.bfloat16` 加载草稿：`evaluator.py:72-78`。
- **建议解决/验收**：把当前实现改名为 `dspark_like_chain_proxy`，增加 `semantics: native|proxy`、`block_size`、`hidden_deps`、`batch_limit`、`cache_policy` 字段；原生 DSpark 另建 block 路径。验收必须包含一张“官方 block vs 当前 chain”的逐项对照表，并有一条红灯测试：UI/报告中出现“DSpark 原生”但 `semantics=proxy` 时失败。

### P0-2｜B=1 时 `dt_pipeline_ideal = max(D,V)` 是合法时间线吗？

- **被挑战的假设**：单请求也能把草稿轮与验证轮无损重叠，得到 0.426 ms、1.368×。
- **最小反例**：第 t+1 轮草稿必须消费第 t 轮验证后的接受前缀/bonus；B=1 没有第二个 microbatch 可交错。`max(D,V)` 把同一请求的依赖工作当成并行，得到的是松弛下界，不是可调度时间线。代码在 `engine.py:196-198` 直接对 B=1 计算并放进 `comparisons`；`index.html:266-268` 仍写“跨微批无损交错”。
- **建议解决/验收**：`dt_pipe_microbatch` 只允许显式拆成 ≥2 个 cohort，并用每个 cohort 的 D/V 计算周期；B=1 返回 `unsupported`，或改用 `dt_pipe_speculative`（带分支命中率与回滚）。验收：时间线必须画依赖箭头和 request/round ID；B=1 的 pipeline 结果不能是 `ok`。

### P0-3｜V4.1 的 config 覆盖完整吗？`compress_ratios[:40]` 是不是静默截断？

- **被挑战的假设**：官方 config 快照已经被完整映射为“40 层 MLA+CSA2 + 全层 MoE”，36 GB/卡 + 0.06 GB KV 足以判断无 OOM。
- **反例/锚点**：
  - `compress_ratios` 共 **43** 项，`num_hidden_layers=40`；`_v41_plan` 只取 `[:40]`，没有声明多出的 3 项是什么（很可能是 3 个 MTP 层，但未验证）：`assets/config/deepseek_v41_config.json:27,60-103`、`simverse/models.py:60-91`。
  - 前 40 项实际是 **2 层 r=0、18 层 r=2、20 层 r=1**；报告写“2 层纯 SWA + 38 层压缩 r=2”不准确：`docs/verification_report.md:40`。
  - `engram_layer_ids`、`engram_num_embeddings=[384006168,384016682]`、`engram_head_dim=256`、`hc_mult=4`、`candidate_block_size=8`、`dspark_*` 等未进入 IR；若 engram 两表整表驻留，按 1 B/param 约 +24.6 GB/卡，按 BF16 约 +49 GB/卡（需按实际压缩/哈希方式核实）。
- **建议解决/验收**：M0 增加“config 字段覆盖清单”：每个键标记 `mapped / ignored_with_reason / blocked`；长度不一致必须 fail closed；V4.1 在 engram/hc 未解释前只能是 `partial/blocked`，不能给 `ok`。验收：一条测试遍历 config 键，未登记的键直接失败。

### P0-4｜MoE 并集公式、38.9 和 `int()` 期望值

- **被挑战的假设**：`E[U]=E*(1-(1-1/E)^(T*k))` 是精确式；“逐算子整数账本”是精确整数。
- **最小反例**：E=384、k=6、T=5 时，硬上界是 `min(384, 5×6)=30`。按“每 token 无放回、跨 token 独立”精确式 `384*(1-(378/384)^5)≈29.09`；当前有放回近似 `384*(1-(383/384)^30)≈28.90`。报告写 **38.9** 既超过硬上界，也与 `outputs/ops_v41.csv` 的 verify 行 `E[U]=28.89` 矛盾：`docs/verification_report.md:103`、`outputs/ops_v41.csv:1711`。
- **另一处**：`ledger.py:303,394` 直接内联 union；`cost.expected_unique` 没有被调用；`moe_routed.weight_read` 与 `draft_kv_scan.kv_read` 都是 `int(期望值)`，但 `OpRecord` 只有一个整数字段，无法区分“精确整数”和“期望值截断”。
- **建议解决/验收**：改为 token 级精确式或明确标注近似；增加不变量 `U ≤ min(E,T*k)`；把期望值存为 `ExpectedValue{value, estimator, bound}`，不要 `int()` 进 `exact` 字段；报告与测试同步重算。验收：E/k/T 边界属性测试 + 38.9 这类红灯用例。

### P0-5｜DT 协议：target TP8 + draft TP1、T→D 256 B、链路带宽不敏感，能同时成立吗？

- **被挑战的假设**：旗舰场景 target TP8、draft TP1 是 vLLM 可实现的；T→D 只需 256 B 控制面；因此 DT 链路只受固定延迟影响。
- **源码反例/锚点**：
  - vLLM `DraftModelProposer._raise_if_draft_tp_mismatch` 明确要求 draft TP 与 target TP 相同：`references/vllm/vllm/v1/spec_decode/draft_model.py:64-79`。测试里 `draft_tensor_parallel_size=1` 只验证了 config 构造，没有实例化 proposer：`references/vllm/tests/v1/e2e/spec_decode/draft_model/test_draft_model.py:194-225`。
  - 官方 DSpark 需要 target hidden states，T→D 至少是 `n × 3 × 5120 × 2 B ≈ 30 KB/接受 token`；B=64、n≈3.05 时约 6 MB/轮，PCIe5 上约 0.09 ms，已接近固定延迟量级。
  - 当前 `DraftSpec.dspark.t2d_bytes_per_token=0`：`simverse/models.py:161-174`；`engine.py:181-185` 只发 `nbar*0 + 256`。
- **建议解决/验收**：给每个部署加 `feasibility: vllm_inprocess | separate_server | hypothetical`；target/draft TP 不等时不得声称“对齐 vLLM”；按 proposer 的 hidden 依赖计算 T→D 字节。验收：协议矩阵（TP 相等/不等 × token-only/hidden × B=1/64）逐项标注 supported/blocked，并重算 dt_serial。

### P0-6｜`catchup_tokens=2` 是通用定律，还是某一种 KV 缓存策略的结果？

- **被挑战的假设**：所有链式草稿下一轮首步都是 `M0=2`，且 `E[M0]=1+α^K`。
- **反例/锚点**：标准 K 步自回归草稿的 cache 只写到 `d_{K-1}`；全接受时下一轮需喂 `d_K + bonus`，所以 `M0=2` 是合理的。但这不是通用定律：DSpark block 一次算整块并 `crop`；MTP/自定义草稿可能保留或丢弃最后 token 的 KV；部分接受时通常只需喂 bonus，`M0=1`。当前 `engine.py:90-103` 对所有 `draft` 一律按 `P_full=α^K` 加权，没有 `DraftKVPolicy`。
- **建议解决/验收**：增加 `cache_policy: retain_all | retain_accepted | discard_block` 与 `M0_policy`，由方法注册表给出；测试三种策略的 `M0` 分布；UI 显示“catch-up 假设”。验收：同一 α/K 下，不同 policy 的 `E[M0]` 可以不同，且各自有反例。

### P1-7｜稀疏 KV 的“前缀只读一次”、η_sparse 和 890 B/token

- **被挑战的假设**：verify 的 5 个 query 只读一次前缀；39.9 MB 受 topk=512 封顶；890 B/token 与 55 MB KV 保留一致。
- **反例/锚点**：
  - `ops_v41.csv` 中 verify 的 `index_select` 每层 2.1/4.19 MB，而 `sparse_attention` 每层仅 0.327 MB；39.9 MB 主要是 **indexer 全候选读**，不是 top-k 值读。把 ctx 增长解释成“只抬容量、不抬 decode 时延”会误导：indexer 读随 `ctx/r` 线性增长。
  - 5 个 query 的 top-k 并集可以是 1–5×；HTML 的 `η_sparse=1+0.15K` 没有 kernel/source 依据：`simverse-v3-text.txt:139-142`。代码当前按 1× 计，是乐观下界。
  - 890 B/token × 32768 ≈ 29 MB/源层；4 个源层就约 116 MB，而 `_kv_retained_bytes` 给 55 MB；两个数没有对齐：`engine.py:58-79`、`metrics.kv_retained_per_card`。
- **建议解决/验收**：把 `kv_read_index` 与 `kv_read_value` 分列；KV 并集给 `[1,Q]` 区间或 kernel 实测系数，并做敏感性；890 B 标注“每源层/每逻辑层”口径。验收：KV 表同时有下界、上界和采用值，禁止单点值不带范围。

### P1-8｜“公式单一维护点”目前只是文档，不是可执行单一源

- **被挑战的假设**：每个 `OpRecord.formula_id` 都能追到真实公式；`formulas.py` 是唯一维护点。
- **反例/锚点**：
  - `attention_core`/`sparse_attention` 的 `formula_id` 是 `gemm_flops`，但实际式是 `pairs*heads*(4d+5)`：`simverse/ledger.py:125-131,274-280`。
  - `F_KV_VERIFY` 注册了但 V4.1 路径没用；`moe_routed` 的 `formula_id` 只描述权重读，FLOPs 是另一式。
  - `ledger.py` 重复 union 公式，`cost.expected_unique` 是死代码；`int()` 期望值仍存在。
- **建议解决/验收**：`formula_ids: {flops, weight_read, act_read, act_write, kv_read, kv_write, t}` 逐量绑定；公式注册表存可调用函数而非字符串；CI 对每行重算并比对。验收：任何一行 formula_id 无法解析或重算不一致即失败。

### P1-9｜shared ModelIR 的“唯一来源”在当前切片里成立吗？

- **被挑战的假设**：shared ModelIR 是模型语义唯一来源，新仓只拥有 lowering/count/cost/execute。
- **反例/锚点**：`simverse/models.py` 自己构造 `ModelSpec`，没有 import `mv_model_structure`；`ledger.py` 按 `family` 分支调用 `build_gqa_model`/`build_v41_model`，没有 `Lowering(ModelIR, Recipe) -> OpGraph`；`MV_MODEL_STRUCTURE_SUPPORT` 自己写的是 `unsupported`：`models.py:134-140`。新架构仍要改 `ledger.py`，违反“新模型只写 L0a 适配器”。
- **建议解决/验收**：明确 `mv-model-structure` 的发布所有权与版本节奏；新仓只能依赖/向上游贡献，不能在自己的 `models.py` 再定义一套语义；先落地 `OpGraph` 与 lowering，再谈“唯一来源”。验收：跨仓依赖测试 + 同一 ModelIR 在两个 Recipe 下总参数量/主干 FLOPs 不变量。

### P1-10｜内存/OOM：只算权重+目标 KV，且 draft OOM 不进 `status`

- **被挑战的假设**：36 GB 权重 + 55 MB KV < 80 GB，所以“无 OOM”。
- **反例/锚点**：`oom_target` 只算 `weight_cap + kv_ret`；`oom_draft` 只算草稿权重；没有 activation、workspace、block 对齐、lifetime、draft KV：`engine.py:154-162`。`status` 只反映 target OOM；draft OOM 只写进 `comparisons[].status`，而 `index.html:200-211` 不渲染 `c.status`，dt 行仍会显示加速比。
- **建议解决/验收**：`MemoryLedger` 分项（target/draft 权重、KV、prefill/verify activation、workspace、碎片/对齐、峰值/保留）；每个 role 单独 `oom`；draft OOM 时 dt 行必须 `blocked` 且 UI 显示原因。验收：对 prefill/verify/B/ctx 组合做 OOM 测试。

### P1-11｜Vidur 是“中位数回归 + 缓存预测”，不是每 kernel 随机采样

- **被挑战的假设**：Vidur 对实测 kernel 时延分布插值采样，事件仿真随机抽样。
- **源码反例/锚点**：`SklearnExecutionTimePredictor` 训练目标列是 `time_stats.*.median`，`model.predict` 确定性输出；没有 kernel 分布采样：`references/vidur/.../sklearn_execution_time_predictor.py:476,491,502,513,569,576,435`。`GridSearchCV` 在全部数据上训练，注释明确“不做 train/test split”：同文件 `:369-370`，所以其误差是 in-sample。报告的“分布插值采样”描述不成立：`docs/verification_report.md:58-63`。
- **建议解决/验收**：方法注改为“per-op median regression，确定性预测，in-sample fit”；校准必须留出 holdout，并报告逐算子残差；不要把“三点定解”当校准（3 点解 3 个系数是插值，fit_error=0 不构成证据）。

### P2-12｜`candidate` block 语义、L40 峰值来源、扫描全局状态

- **`candidate_topk_blocks=2048` / `candidate_block_size=8` 与 `index_topk=512` 的 token/block 语义未定义**：若 512 是 block，压缩 KV/attention 读会放大 8×；若 512 是 token，`candidate_publish` 的 2048 个 block ID 又未被 attention 使用。必须从官方 forward 确认，不能靠猜。
- **L40 峰值来源注释自相矛盾**：`devices.py:37` 写“90.5 TFLOPS dense FP16×2(稀疏)=181”，但 90.5 是 FP32，不是 FP16；L40 的 FP16/FP8 公开值需复核（常见口径为 362/733，若如此当前值低一半）。当前草稿是带宽受限，头条对峰值不敏感，但器件表不能这样写。
- **`sweep_1d` 直接改全局 `DEV.LINKS/DEV.DEVICES`**：`sweep.py:31-54`，Web 并发扫描会串数据；`test_csv_export_matches_runresult` 导出的是 `outputs/run_v41.json` 存档而不是 fresh result，可能被 stale archive 骗过。

## 二、主控判断中可能错误或过度的至少两处

1. **判断 #4（shared ModelIR 是唯一来源，新仓拥有 lowering/count/cost/execute，旧仓经适配器消费共享版本）可能把所有权方向写反了，且当前实现已经是第二份适配器。**  
   证据：`models.py` 自己构造 `ModelSpec`，没有消费 `mv_model_structure`；`ledger.py` 按 `family` 分支，没有 lowering。若新仓再“拥有”共享版本，就是第三份来源。建议收窄为：共享包仍是唯一发布源；新仓**依赖**它并向上游提交 V4.1 适配器；新仓只拥有 lowering/count/cost/execute；旧仓继续 pin 同一版本，而不是消费新仓的私有版本。

2. **判断 #2（首发 V4.1 Flash + tiny 锚点 + 证据闸门）在顺序上过度：V4.1 还没通过 config 字段覆盖和官方 forward 审核，就被当成第一证据锚点。**  
   证据：`compress_ratios` 43 vs 40、engram/hc 未建模、DSpark 是链式代理而非官方 block。正确顺序应是“tiny + 一个已支持模型（如 Qwen3/GQA）先做独立锚点；V4.1 先通过字段覆盖/forward 审核，再升级为第二锚点；DSpark 在此之前只能叫 proxy”。

3. **判断 #7（一屏四联动工作台作为 MVP）可能过度。**  
   当前 `index.html` 实际是五个 tab（cmp/ops/tl/src/sweep），不是同屏联动；把 layer graph + op card + Roofline + 多轨时间线全部塞进 M1，会把资源从证据闸门转移到 UI。对用户 Q2 的 C（验证新架构/系统方案），M1 的最小可用界面应是 **两方案对比 + 选中算子明细 + 带 request/round ID 的时间线**，layer graph/Roofline 放 M2。

4. **判断 #5（每个结果同时携带 support/evidence/feasibility/scope/单位/公式/输入依赖）方向正确，但“每个结果”过度。**  
   对每一行 op 都加七类自由文本元数据，会产生大量不可验证的“伪严谨”。建议收窄为：每个**决策性聚合结果**、以及所有**非精确**行必须带；字段用受控枚举 + CI 校验，而不是 prose。当前 `provenance` 是自由文本，正是反例。

5. **判断 #1 本身正确，但报告用它背书“整数精确”属于过度。**  
   `moe_routed.weight_read`、`draft_kv_scan.kv_read` 都是 `int(期望值)`；`OpRecord` 只有一个整数字段，无法区分 exact/expected。报告说“逐算子整数账本精确相等”，但 V4.1 的 MoE 并集和草稿 KV 不是精确整数；13 个测试只覆盖 tiny GQA + 通道一致性，不覆盖这些行。

## 三、建议的计划顺序、原理契约与 UI 误导防护

### 3.1 计划顺序（只列改动，不重写全案）

| 阶段 | 交付 | 进入下一阶段的条件 | 明确非目标 |
|---|---|---|---|
| **M0 证据/红灯** | 用户 Q1 案例；config 字段覆盖清单；官方 forward 锚点；38.9/18%/catch-up/pipeline 反例测试 | 每个 config 键有 mapped/ignored/blocked；每个红灯有可运行测试 | 不选默认模型，不产出 TPOT |
| **M1 独立锚点** | tiny + 一个已支持模型；typed `ModelIR`→`OpGraph`→`Count`；两方案对比 + 算子明细 | 手算逐项相等；Lowering 无架构分支；跨 Recipe 不变量 | 不做 V4.1/DSpark 结论 |
| **M2 V4.1 结构闸门** | V4.1 字段覆盖、CSA2/index 语义、engram/hc 的 blocked 说明；仅 Count | 结构未知项全部显式 blocked；不输出“ok” | 不做成本、不做 UI 结论 |
| **M3 单设备成本/容量** | 逐算子 T0 + MemoryLedger；敏感性；无 DT | 每个数字带 per-quantity formula id；OOM 分 role | 不做链路/事件 |
| **M4 DT 可行性 + 串行投机** | protocol matrix（TP/hidden/token-only）；`dt_serial` 闭环；事件核（request/round ID） | vLLM 可实现性或明确 hypothetical；B=1 无 pipeline | 不做 serving |
| **M5 Serving/校准** | 请求级事件、holdout 校准、误差报告 | 校准与验收样本分离；非三点自拟合 | 不把 T1 当 T0 的证明 |

关键调整：把当前 M2 的“成本/双向消息/容量/串行投机”拆开；**DT 协议可行性必须排在成本之前**，否则会在不可实现的场景上做精细计价。

### 3.2 原理契约（建议直接写进 M0）

```text
ResultEnvelope {
  support:       exact_int | expected | approximate | unsupported
  evidence:      hand | official_source | old_repo | measured | assumption
  feasibility:   feasible | conditional | blocked
  scope:         per_card | group | per_request
  units:         {tpot: ms/token, bytes: B, capacity: B, ...}
  formula_ids:   {flops, weight_read, act_read, act_write,
                  kv_read, kv_write, t}
  input_deps:    [scenario_digest, model_digest, device_digest, recipe_digest]
  calibration:   none | in_sample | holdout
}
ExpectedValue { value, estimator, bound, distribution? }
DraftSpec {
  S, M, block_size, hidden_deps, kv_policy, batch_limit,
  semantics: native | proxy, acceptance_model, verify_rule
}
```

三条硬规则：

- **期望值不得 `int()` 进 exact 字段**；`U ≤ min(E,T*k)` 作为不变量。
- **一个 `formula_id` 不够**：FLOPs、五类字节、时间各自绑定公式；`attention_core` 不能挂 `gemm_flops`。
- **容量与流量分账**：权重容量/读取、KV 保留/读写、activation/workspace/lifetime 分开；draft 与 target 同卡时合并峰值。

### 3.3 UI 误导防护

- 每个收益数字旁挂 **missing-assumptions badge**：`allreduce 未计`、`launch 未计`、`DSpark proxy`、`hidden 未计`、`TP 不匹配/hypothetical`、`OOM 风险`、`T0 未校准`。badge 由结果字段生成，不能由前端自由写。
- `dspark` 行显示 `native/proxy`；proxy 时不要写“官方 DSpark”，改成“DSpark-like chain proxy”。
- B=1 隐藏/禁用 `dt_pipeline_ideal`；B≥2 也必须显示 cohort 拆分和依赖箭头。当前 `renderTl` 没有 request/round ID，也没有依赖箭头。
- OOM 分 role 显示；draft OOM 时 dt 行必须置灰并显示原因，不能仍显示 0.9×/1.3×。
- 表格加 `scope` 列（per-card vs group）、`KV index/value` 两列；当前“共享专家 1.41 GB”和“每卡 0.177 GB”混在一起。
- 公式 tooltip 按量显示；`attention_core`/`sparse_attention` 不能再显示 `gemm_flops`。
- 增加“两方案 diff”视图：同 digest 校验 + 按算子分解 Δt，而不是只有同一场景的四部署表。

### 3.4 需要用户决定的问题（只留这些）

- **Q1**：哪次使用让你觉得现有仿真靠不住？这决定 M0 的第一条验收案例。
- **Q4**：只验结构和相对变化（A），还是要复现绝对性能（B/C）？可接受的误差和可用 profile 数据是什么？
- **Q6**：引擎复用边界：必须 vLLM 进程内、允许独立 draft server，还是可以用自定义 runtime？这直接决定 TP8+TP1 的 DT 场景是否合法。
- **Q3**：如果首发不是 V4.1 Flash，用户真正关心的新架构/场景是什么？
- **是否接受 proxy**：第一版演示是否允许用“DSpark-like 代理”并明确标注，还是必须等原生 block 实现？这是对外声称的偏好，不是技术细节。

## 四、实际阅读/运行了什么，哪些只是推断

**实际阅读**：

- 任务与背景：`docs/rebuild-20261106/review-packet.md`；`D:/Github/model-verse/docs/simulator-rebuild-plan.md`、`simulator-rebuild-questions.md`。
- 报告与产物：`docs/verification_report.md`；`outputs/run_v41.json` 的 metrics/comparisons；`outputs/ops_v41.csv` 的 baseline/verify 关键行；`assets/config/deepseek_v41_config.json`；`assets/scenarios/v41_flash_dt_verification.json`。
- 代码：`simverse/contracts.py`、`formulas.py`、`cost.py`、`ledger.py`、`engine.py`、`acceptance.py`、`models.py`、`devices.py`、`sweep.py`、`static/index.html` 相关段；`tests/test_oracle.py`、`tests/test_consistency.py`；`golden/expectations.py`。
- 外部源码：DeepSpec `draft_ops.py`、`evaluator.py`、`modeling/dspark/qwen3/modeling.py`；vLLM `vllm/v1/spec_decode/draft_model.py`、`tests/.../test_draft_model.py`；Vidur `sklearn_execution_time_predictor.py`、`events/batch_stage_end_event.py`。
- 资料：`docs/rebuild-20261106/references/model-verse-principles-pinned.md`；`simverse-v3-text.txt` 前 315 行及关键 grep 命中。

**未运行**：

- 未跑 `pytest`（只读任务；pytest 会写 `.pytest_cache`/临时文件；13 passed 是送审稿自述，未复跑）。
- 未跑模型、GPU、Web 服务、`tools/oldrepo_diff.py`；未重算 config/digest 的 SHA256。
- 本环境的 `pwsh` stdout 被吞掉（只返回 Electron cache 的 stderr），所以没有做 shell 数值复算；所有算式均为 read/grep 后手算。

**推断/未证实**：

- `compress_ratios` 多出的 3 项“很可能是 MTP 层”是推断，未在官方 forward 中确认；但无论是什么，`[:40]` 是静默截断这一点是确定的。
- engram 整表驻留导致 +24.6/+49 GB/卡是“若整表驻留”的敏感性估算，未按 `engram_compressed_vocab_size=99092` 的压缩/哈希实现核实。
- L40 的 FP16/FP8 公开值未在本会话核对；我只确认 `devices.py` 的来源注释把 90.5（FP32）写成 FP16 依据，至少注释错误。
- 官方 V4.1 forward 源码不在 `references/` 中，因此 CSA2/index 的 token/block 语义、engram/hc 的实际作用只能标为“待官方 forward 审核”，不能下结论。
- 报告 38.9 很可能是笔误，但即使是笔误，也说明“数字—公式—产物”没有自动对账；18% 则是 pinned 文档的独立算术错误（公式给 85.8%）。

## 交付文件/目录

无新增或修改文件；未创建外部资源。本报告即本次唯一交付，所有结论基于上述只读材料与手算。