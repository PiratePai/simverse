# R23：Qwen3.6-35B-A3B 原生 MTP forward 证据与仿真映射（2026-11-13）

## 0. 结论

Qwen3.6-35B-A3B（`qwen3_5_moe`，`mtp_num_hidden_layers=1`）的原生 MTP 微头
已按**双栈官方实现逐行核对**（vLLM 主 + SGLang 对照，9 文件 pin）完成
source-checked 建模，替换了此前"目标首层（GDN）模板 × mtp_layers"的研究
近似。计数/内存/方法资格按 pinned 结构重写；不支持的路径（DT 分离、
专属 embedding、无 MTP 头模型）显式拒绝。**fixture 级独立数学对照已执行**
（双实现互核 + 账本对账），但 vLLM 模块本身未在 CPU/GPU 执行、真实
checkpoint 未加载、无任何实测——部署排名资格不因本轮解除（mrope 未核、
模型级 cpu_forward=not-run、无 GPU 实测、预算 unknown）。DFlash 与 MTP
是两种不同方法，源 pin/程序图/结构分立，不可互替。

## 1. 来源与身份（固定仓库/提交/哈希）

| 对象 | 固定来源 | 本仓证据 |
|---|---|---|
| vLLM MTP 实现 | `vllm-project/vllm @10a854ed6b3bbd7411c65ccbf87594e71c4d322c`（main，2026-11-13 raw 固定字节） | `assets/identity/vllm_qwen36_mtp_pin/`（8 文件：qwen3_5_mtp.py、qwen3_next_mtp.py、config/speculative.py、eagle/utils.py、mtp/target_dependent_ar speculator.py、2 个 config 测试 + e2e 测试；manifest 逐文件 SHA256+blob sha1，blob 对 GitHub contents API 校验） |
| SGLang MTP 对照 | `sgl-project/sglang @1b2870d33de7b5f31844fd74e97eea6d7896c53a` | `assets/identity/sglang_qwen36_mtp_pin/`（qwen3_5_mtp.py 1 文件） |
| Qwen3.6 checkpoint identity | `Qwen/Qwen3.6-35B-A3B @345d467b`（config 快照不变） | `ev-qwen36-main-config`（既有） |
| HF 侧（对照栏） | HF transformers @b6401d66（既有 pin） | `ev-hf-qwen35-modeling`：主模型 `_keys_to_ignore_on_load_unexpected=[^mtp.*]`——**HF 无 MTP forward** |

证据条目：`ev-vllm-qwen36-mtp`（source-checked，executed=False）、
`ev-sglang-qwen36-mtp`（source-checked）、`ev-qwen36-mtp-cpu-numeric`
（numeric-checked，fixture 级 executed=True）；source_map `sm-qwen36-mtp`
（替换 `sm-qwen36-mtp-unverified`）。

## 2. 官方实现查明的事实（逐行核对）

### 2.1 checkpoint 的 mtp.* 参数与加载行为

* **config 键**：`mtp_num_hidden_layers=1`、`mtp_use_dedicated_embeddings=false`
  （`text_config` 内）。vLLM `SpeculativeConfig.hf_config_override`（speculative.py
  L1147-1167）把 `qwen3_5/qwen3_5_moe/qwen3_5_text/qwen3_5_moe_text` 改写为
  `model_type=qwen3_5_mtp`、`n_predict←mtp_num_hidden_layers`、
  `architectures=[Qwen3_5MoeMTP]`（MoE）/`[Qwen3_5MTP]`（dense）。
  **官方测试固定**：真实 `Qwen/Qwen3.6-35B-A3B` config → `Qwen3_5MoeMTP`、
  n_predict=1（test_qwen3_5_mtp_config.py L112-128，经 AutoConfig 下载真实
  Hub config 解析）。
* **checkpoint 张量**（`mtp.` 前缀，无独立 embed/lm_head）：
  `mtp.fc.weight`、`mtp.pre_fc_norm_embedding.weight`、
  `mtp.pre_fc_norm_hidden.weight`、`mtp.norm.weight`、
  `mtp.layers.{i}.*`（完整 decoder 层张量：self_attn q/k/v/o+q_norm/k_norm、
  MoE gate/experts/shared_expert/shared_expert_gate、input/post_attention
  layernorm）。e2e 测试 `_add_random_mtp_module`（test_mtp.py L28-51）按
  `mtp.layers.0.*` 从真实 checkpoint weight_map 复制出 module 1——张量名单
  由官方测试证明。
* **加载行为**：`Qwen3_5MTP.is_unused_checkpoint_weight`（qwen3_5_mtp.py
  L303-306）只保留 `mtp.*`+`embed_tokens`+`lm_head` 三类键；`load_weights`
  L308-320 把 `mtp.` 重映射为 `model.`、`embed_tokens` 剥 `language_model.`
  前缀。HF transformers 主模型**忽略** `mtp.*`（`_keys_to_ignore_on_load_
  unexpected`，config-only 栏）；vLLM/SGLang 主模型 `WeightsMapper` 把
  `mtp.` 置 None 丢弃——MTP 权重只由草稿模型消费。
* **colocated 运行时共享**：`MTPSpeculator.load_draft_model`（mtp/
  speculator.py L15-20）→ `load_eagle_model`（eagle/utils.py L115-168）→
  `maybe_share_target_embed`（L82-112：`del draft_inner.embed_tokens` 后装
  目标对象）+ `share_target_lm_head`（L44-71：`del draft_model.lm_head` 后装
  目标对象）。`_should_share`（L19-41）：草稿无 `has_own_embed_tokens/
  has_own_lm_head` 标志（Qwen3_5MTP 均无）即共享；`in_single_pp_stage` 仅
  多 rank 单段模式早退，常规 colocated 走共享。SGLang
  `set_embed_and_head/set_lm_head_from_target`（L174-193）同义（del 后指
  目标权重）。**结论：MTP 草稿 embed/lm_head 零字节共享目标分片对象**
  （与 DFlash R02a 同一配方）。

### 2.2 forward 结构（目标首层模板 ≠ MTP 层）

`Qwen3_5MultiTokenPredictor.forward`（qwen3_5_mtp.py L155-177）：

1. 共享 embedding 查表（`embed_input_ids`；mtp_use_dedicated_embeddings=false
   且 loader 不消费 mtp.embed_tokens）；
2. **双预归一**：`pre_fc_norm_embedding(inputs_embeds)` 与
   `pre_fc_norm_hidden(hidden_states)` 两个 RMSNorm **分别归一**（L159-160）；
3. `cat([emb_norm, hidden_norm], dim=-1)` → `fc(2h→h)`（L161-162）：
   ColumnParallelLinear(2h→h, gather_output=True, bias=False)（L99-104）。
   **无激活、无残差直加**——区别于 DeepSeek V3 式 comb（silu(W[·])+h）；
4. 层选择 `layers[spec_step_idx % num_mtp_layers]`（L165-166）执行**一个**
   decoder 层；`Qwen3_5DecoderLayer(layer_type="full_attention")`（L120-127）
   ——**full attention（gated GQA）+ MoE**，不是目标首层（Qwen3.6 首层是
   GDN linear attention）。SGLang L143-144 `full_attention_interval=1` 同义；
5. final norm `self.norm(hidden, residual)`（L175，vLLM RMSNorm(x, residual)
   = norm(x+residual)）；
6. 共享 lm_head 读出（Qwen3_5MTP.compute_logits，L228-235+L288-292）。

层序/状态/缓存依赖：MTP 层是 full attention → **草稿自有 per-token KV**
（2·kvh·d 元素/层/token），prefill 阶段草稿全量同步 KV（tdar `_prefill`
L339-387，skip_sample 时只保 KV）；decode 每步写 1 行/请求、被拒步回退
（下轮重写）、bonus 行由下轮步 0 写入。

### 2.3 步进循环与 K 语义（撤销旧 K≤n_predict 限制）

* 每轮 K=num_speculative_tokens 个**均匀**草稿步（每步 B 行、1 token/请求；
  步 0 输入=目标上轮最后 hidden+bonus token）——tdar `_decode` L391-439：
  forward → sample_draft → update_draft_inputs，**无 catch-up M0=1+P_full
  加权**（那是旧链式草稿 v0.2 惯例）；
* 步 j 用层 `j % n_predict`；`use_multi_module_mtp`（speculative.py
  L1948-1967）：n=1 或 K=1 时单模块复用，min(n,K)>1 才多模块。**K>n_predict
  合法**（同层循环）；官方 e2e：K=3×MTP1（draft_lm_head 量化测试）、K=2×
  MTP2（逐步模块选择测试：module 1 随机权重 → 步 2 接受率<0.05 证明未复用
  module 0）。
* HF/vLLM/本仓三栏差异：HF 无 MTP forward（config-only，键忽略）；vLLM
  `Qwen3_5MoeMTP`（本仓主建模依据）；SGLang `Qwen3_5ForCausalLMMTP`（主干
  语义对照：fc/预归一/层型/共享一致，服务循环与 NPU/量化分支未逐行核）。
  本仓方法路径 = vLLM colocated（源码核对，未执行）；**DFlash 不是原生
  MTP**：块扩散草稿（z-lab/dflash pin），与 MTP 的源 pin/程序图/结构分立。

## 3. 落地（按已证实路径；不支持的显式拒绝）

| 层 | 改动 |
|---|---|
| ModelIR | `parse_qwen3_5_moe` → `extras["mtp"]`（n_layers/n_predict/layer_kind=full_attention/fc{2h→h,column_gather}/pre_fc_norms/final_norm/shared_objects/step_selection/source_pin）；两键 mapped 注记（行号证据）；`mtp_use_dedicated_embeddings=true` config 域拒绝（loader 无专属路径） |
| count | `mtp_step_rows(stage, target, step_idx, ctx, quant, tp, ep, batch, plan, rank)` 逐步重写：mtp_embed_share → 双预归一 → mtp_fc（列切 h/tp）→ full-attn 层（gqa_layer_rows+ffn_or_moe_rows+norm_add_rows，S=ctx+step，EP/plan 同目标配方）→ mtp_final_norm → draft_lm_head（读流量/容量归目标）；`mtp_kv_rewind_row`（K−E[N] 期望浪费）；`mtp_layer_model`（单层伪模型，memory/collectives 共用）。**撤销旧"目标首层 GDN 模板"**（曾漏 fc/预归一/lm_head/KV，且层型错误） |
| methods | `plan_mtp`（v0.3）：K 均匀步、步 j 层 j%n 循环、source_pin=vllm@10a854ed、prefill KV 同步/采样档/未执行显式登记 approximations；K≤n_predict 旧拒绝撤销 |
| pipeline | mtp 分支：逐步阶段计价 + 草稿通信（attn/mlp allreduce、EP all2all、fc all-gather、logits gather）+ `mtp_serving_integration`（colocated=supported source-checked）；**dt_serial/dt_pipeline 显式拒绝**（load_eagle_model 进程内共享，无 DT 实现）；method_recipe_ready 原因更新；evidence refs ⊇ ev-vllm/sglang-qwen36-mtp |
| memory | `mtp_extra_allocations`（fc[2h→h/tp]+3 norms）+ `mtp_transient_allocations`（单步并存组下界+跨轮 hidden）+ colocated mtp_model 接线：层权重走 target_weight_allocations（同 TP/EP 配方），embed/lm_head 零字节共享引用（并入 target 物理对象 consumer），draft KV per-rank |
| 证据四栏 | catalog：method_status["mtp"]="source-checked"、checked_paths+sm-qwen36-mtp、unverified_paths 收窄为 mrope；registry 三条新证据；模型级 cpu_forward/gpu 仍 not-run/none |

## 4. 验证（独立 oracle + 负例 + replay）

### 4.1 fixture 级独立数学对照（ev-qwen36-mtp-cpu-numeric）

`tools/probe_qwen36_mtp_numeric.py`（本机 Python 3.12/torch 2.15.0+cpu 执行）
——pinned 结构手工复刻（vLLM 模块未执行）：

* **双实现互核**：torch 矩阵式（f64）vs 纯 Python 逐 token 标量循环
  （math 标准库，独立代码路径），7 组关键中间张量 × 3 步逐元素对照，
  max diff **4.97e-13**（atol 1e-9；曾发现两处对照假阳：torch.tensor(list)
  默认 f32 量化、router 近平手 f32/f64 选择翻转——修复后同精度对照，
  见探针注释）；K=3 步 MTP1 同层复用，draft KV 7→8→9 行；
* **账本对账**：3 步 × 10 类 GEMM 行 flops == 2×（标量循环实际计数 MAC）
  共 30 项全对；attention 因果键数/KV 读行数与账本精确相等；
* **手算锚点**：GemmaRMSNorm([3,4], w=[1,1]) 独立手算值相等；
* 落盘 `tests/fixtures/qwen36_mtp_r23_numeric.json`；hermetic 测试（CI）
  消费 + 本仓行复算防漂移；torch 可用时 live 重放不变量。

### 4.2 测试（tests/test_qwen36_mtp_r23.py，25 项 + 门禁更新）

* 结构：catalog extras spec/覆盖注记/专属 embedding 拒绝/HF 栏无 MTP
  forward（pin 文本锚点）；
* 计数：行集合=pinned 结构（无 GDN 行）、KV 随步增长、层循环、B>1 token
  维乘一次权重不变（G13 同口径）、TP8 分片（fc h/tp、heads/tp、lm_head
  V/tp）、无 MTP spec 拒绝、回退浪费期望行；
* 方法：K>n 循环放行（撤销旧拒绝）、无头拒绝、MTP/DFlash 源 pin 与程序图
  分立（不可互替）；
* 管线：colocated K=3 闭环（逐步阶段/草稿通信 kinds/能力列）、内存对象
  （MTP 常驻/草稿 KV/临时/embed-lm_head 零字节并入 target consumer）、
  dt 双模式拒绝、V4.1 目标拒绝、证据四栏同步、run refs；
* **replay 冻结**：MTP 默认场景（B1/ctx32768/TP8/K3/W8A8+FP4E）digest
  与逐步 totals/steady 冻结（FROZEN 常量）；
* 门禁更新：G10（method_status source-checked + mrope 仍阻断）、G13
  （新签名）、G16（sm-qwen36-mtp 两侧覆盖+未执行登记）、model_ir/product
  _scope/pipeline_v02 旧断言随事实更新（MTP 未核→mrope 未核）。

### 4.3 前后 replay（docs/evidence/mtp-map/）

同场景（method=mtp k=1——main 上唯一合法 K）before（main b8585a1 未改
工作区）/after 逐 op 归因：`target_decode`/`verify` 逐字段冻结；draft_step_0
整段替换（旧 GDN 模板 19 行 → 新 pinned 结构 24 行，含 1 行 KV 回退浪费注记：+fc/双预归一/final
norm/draft_lm_head/attention KV，−GDN 递推/卷积 state，+draft KV 读写）。
**M01 64.55→66.96 ms**（草稿步真实成本上升：共享 lm_head 63.5MB 读/步 +
draft KV 读入账；旧模板两个方向都错）；**steady 2,699,569,088→2,841,006,528 B
（+141,437,440）**（MTP 层常驻 ~108MB/卡@TP8 W8 draft 档 + fc/norms +
draft KV 32MB + 临时）。K 域变化单列（K≤1 → K≥1 循环）。
`tools/replay_qwen36_mtp.py --check` 无漂移。

## 5. 剩余限制（不冒充）

1. **未执行/未实测**：vLLM/SGLang 模块未在 CPU/GPU 执行；真实 checkpoint
   mtp.* 权重未本机加载（名单由官方 e2e 测试+loader 重映射证明）；无服务
   延迟实测。fixture 是随机权重小尺寸数学对照，不覆盖融合 kernel/真实
   权重/部署性能。
2. **部署排名仍阻断**：mrope 位置生成未核（forward_source=partial-source）、
   模型级 cpu_forward=not-run、无 GPU 实测、可分配预算 unknown、接受率为
   显式假设输入（R07b 前无真实 trace）。
3. **变体未建模**：n_predict>1 多模块（MultiModuleMTPSpeculator/
   index_share_for_mtp_iteration）、draft_lm_head_quantization（fp8/nvfp4
   量化草稿头副本）、use_local_argmax_reduction（不物化全量 logits）、
   probabilistic 采样档（DraftLogitsCache）、CUDA-graph 桶取整 padding、
   SGLang NPU/量化分支、NVFP4 checkpoint 的 fc BF16 workaround（本仓无
   NVFP4 场景）。
4. **草稿 prefill KV 同步**（tdar _prefill 全量执行）未入稳态 TPOT 账本
   （一次性成本，cache_policy.draft_prefill 登记）；prefill/混合批形状域
   仍属 G-C2。
5. fc all-gather 延迟按单向环 (tp−1) 段近似（与 engram_gather 同已登记
   近似）；草稿步 logits 缓冲按 fp32 下界计。

## 6. 2026-11-14 合入前验收修复

R23 原提交的 MTP 私有层 count/cost 用目标 `quant`（默认专家 FP4），
memory 却用 `draft_quant`（默认专家 W8），使同一草稿专家的读取期望与
驻留容量相差 2 倍。现统一私有层的 count/cost/通信与 memory 精度档；
共享目标 `lm_head` 的读取/成本仍按目标对象档。微头 final norm 已在
`mtp_extra_allocations` 的三件 norm 中，移除伪模型通用 final norm 的
重复驻留 2,048 B。新增 count↔memory 回归测试。

固定 K=1/TP8 同场景 replay：草稿专家 `weight_read_expected`
1,572,864→3,145,728 B；M01 66.9618249048→66.9620931971 ms；
steady 2,841,006,528→2,841,004,480 B；target_decode/verify 不变。
`--check` 扩为校验 work totals、阶段成本、指标和逐行归因，防止成本
漂移被遗漏。此修复只纠正本仓的口径，不构成新的模型执行或实测证据。
