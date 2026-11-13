# R24：DeepSeek-V4.1-Flash SwiGLU clamp 与 compress_ratios 尾部语义审计

日期：2026-11-13；分支 `codex/v41-swiglu-tail-r24`（自 main b8585a1）。
任务：对 V4.1 的 `swiglu_limit`（10.0）与 `compress_ratios` 尾部
`[40:43]` 完成触发条件、位置、数值语义的源码审计，修正 ModelIR/算量/
结果资格中受影响的部分；以仓库固定 vLLM 版本为准。

## 1. 任务与结论摘要

1. **SwiGLU clamp**（`swiglu_limit=10.0`）：
   - 语义（activation.py L208-250 `SiluAndMulWithClamp`）：
     `gate = clamp(x[:d], max=limit)` **单侧**（负 gate 原样通过）、
     `up = clamp(x[d:], min=-limit, max=limit)` **双侧**、
     `out = gate·sigmoid(alpha·gate)·(up+beta)`；V4.1 无
     swiglu_alpha/swiglu_beta config 键 → `FusedMoEConfig` 默认 None →
     alpha=1.0/beta=0.0（fused_moe/activation.py L186-193）→ 退化为
     `silu(clamp(gate))·clamp(up)`。
   - 触发条件：MLP/共享专家路径 `swiglu_limit is not None`
     （deepseek_v4/nvidia/model.py L176-183）；fused MoE Triton 路径
     `HAS_LIMIT = swiglu_limit > 0`（utils.py L812）。**值 0 在两路
     分叉**（MLP 仍构造 clamp(0) 病态核、Triton 关 clamp）→ 本仓 config
     域把 0/负/非有限值一律拒绝（official 资产恒 10.0）。
   - 位置（消费两路，均在注册执行域内）：
     a. 路由专家：w13 GEMM → 激活 → w2 GEMM。clamp_limit 回退链
        `quant_config.gemm1_clamp_limit or moe_config.swiglu_limit`
        （`ApplyMoEActivationConfig.from_configs`，fused_moe/activation.py
        L176-197）；**Marlin（≤SM90 注册执行域）经 `apply_moe_activation`
        回调消费同一 clamp**（marlin_moe.py L140-158），fused 别路
        `silu_and_mul_with_clamp`（L293-312）/`swiglu_limit_func`
        （utils.py L819-835）同语义。
     b. 共享专家 MLP：`self.act_fn = SiluAndMulWithClamp(swiglu_limit)`
        （model.py L176-183，经 L331 接线；CPU 变体同构 cpu/model.py
        L101-102）。
     c. native MegaMoE `MegaMoEActivation("swiglu", clamp=...)`
        （model.py L393-394）语义同源但 SM100 域外（R22 执行域 =
        Marlin W8A16/W4A16），登记不建模。
   - 算量修正：新增 `moe_act`/`moe_shared_act` 两行，逐元素 6 算术 op
     （gate clamp-max 1 + up clamp-min 1 + up clamp-max 1 + sigmoid 1 +
     乘 2；解析算术 op 口径，与 R21 路由行同惯例）。**字节全 0**——
     激活中间缓存读写已在 `moe_routed`/`moe_shared` 行计价，单一所有者
     防重复计费。`swiglu_limit` 键 mapped（gaps 解除）。
2. **compress_ratios 尾部 [40:43]**：
   - 归属（逐项）：尾部第 i 项归 DSpark/nextn 草稿层（全局层号
     L+i=40+i）。证据链：`get_compress_ratios` 文档串
     "**MTP layers included**"、上游派生路径显式补
     `[0]*num_nextn_predict_layers`（deepseek_v4_utils.py L6-18）；
     DSpark 模型以 `layers.{num_hidden_layers+i}` 前缀构造草稿层
     （dspark.py L93-96/L128-136）；`DeepseekV4Attention` 按层号取
     `compress_ratios[layer_id]`，**长度不足的层号落 else 分支
     compress_ratio=0（纯 SWA）**（attention.py L375-386，注释原文
     "MTP layers past the configured list are pure sliding-window"）；
     值域 {0,1,2} 对任意层校验（L387-392）。
   - 用途：official 资产尾部全 0 → 三个草稿层为**纯滑动窗口注意力**
     （DSpark 的 sliding-window context KV，dspark.py L182-188 起）；
     `is_backbone = layer_id < num_hidden_layers` 门控（L399-401）使
     草稿层**永不为 kv/index 源**；后端只按**去重层型集合**消费
     （sparse_swa.py L494-502）——尾部 0 与主干 0 同组，不产生新
     tile-scheduler plan / KV cache 组。**尾部对主干账本无数值影响**。
   - 域（本仓建模域，非源码断言）：长度 ∈ {L, L+num_nextn_predict_layers}
     （超出部分无层号消费 = 死配置，不得静默忽略）；尾部值 ∈ {0,1,2}；
     **非零尾部域外拒绝**——源码允许草稿层 compress_ratio>0（届时消费
     下方最近源的压缩 KV），但 DSpark 草稿前向未建模、无 official 证据，
     不猜语义。尾部入册 `extras.kv_spec.draft_compress_ratios`，
     `compress_ratios[40:43]` gap 解除。
3. **附带补漏**：`candidate_block_size` 键此前漏在 `_STRUCT_KEYS` 集合
   （R20 已有 mapped 注记与消费代码），导致官方资产长期误挂
   "未登记 config 键"缺口（G9 断言被 text_config 前缀绕过未暴露）。
   补入集合使 R20 注记生效；语义不变（块粒度 8）。
4. **结果资格**：V4.1 `config_coverage_complete` 自此 True（最后一个
   unsupported 叶键解除）；`structure_complete` **仍 False**——仅由
   稀疏 MLA 注意力核本体（csa2_attention_core，flops 4d+5 v0.1 移植）
   的 forward 证据阻断。部署排名资格不因本模块解除。

## 2. 源码链与 pin

- 固定版本：vllm-project/vllm @`f358f0578a83f4e1fe6fc49a492d3ae9362de8e5`
  （与 R20/R21/R22 各 pin 同一提交；vllm-tree.json 逐项 blob sha1 校验）。
- pin：`docs/rebuild-20261106/references/vllm/swiglu-tail-pin.json` 13 文件
  （源文件落盘同目录；blob sha1 == vllm-tree == pin 三重一致，SHA256/size
  由 tests/test_v41_swiglu.py §1 复验）：
  语义核（activation.py）、fused MoE 消费链（config.py/activation.py/
  utils.py/marlin_moe.py/layer.py）、模型接线（deepseek_v4 nvidia+cpu
  model.py）、尾部归属（deepseek_v41/attention.py、
  transformers_utils/deepseek_v4_utils.py、dspark.py、sparse_swa.py）、
  vLLM 侧数值等价测试（tests/kernels/core/test_activation.py）。
  其中 3 文件与 csa2/moe pin 重叠（blob 一致，交叉验证）。
- **证据分层（G10）**：全部为源码阅读 source-checked；未执行任何 vLLM
  kernel、CPU forward 或 GPU 测量。clamp CUDA kernel 与 native 的数值
  等价以 vLLM 在册测试登记（test_activation.py L62-66/L251-285），本仓
  未跑；本仓的数值 oracle 为独立 stdlib 实现（§4），不是执行证据。

## 3. 落地（ModelIR / count / 契约）

- `model_ir/families.py`：`swiglu_limit` 入 struct 键集 + 域校验（数值、
  >0、有限；缺失拒绝——无 clamp 变体不在 source-checked 域）；构建
  `extras.swiglu`（limit/gate=clamp_max/up=clamp_minmax/alpha/beta/
  ops_per_elem=6/applies_to/mega 域外注记）；`_SWIGLU_KEY_NOTES`
  mapped 注记（行号证据）；compress_ratios 尾部域校验（长度/值域/非零
  拒绝）+ `draft_compress_ratios` 入册 + 尾部 gap 移除；
  `candidate_block_size` 补入 struct 集合。
- `count/moe.py`：`moe_rows(..., act_recipe=extras.swiglu)` 追加
  `moe_act`（routed：flops=act_assign·inter_rank·6；TP 关 exact、EP 开
  expectation 与 moe_routed 同口径）与 `moe_shared_act`（shared：
  T·si_rank·6；exact）两行；字节全 0（防重复计费）；Qwen36 等其他
  家族无 extras.swiglu → 行为不变（R06 冻结口径维持，跨模型计量差异
  在 registry limits 登记）。
- `count/work.py`：`build_model_work` 读取 extras.swiglu 传入
  `ffn_or_moe_rows`。
- `formula_registry.py`：`v2.swiglu_clamp_act`（expr/source/applies）。
- `model_ir/catalog.py`：checked_paths + `sm-v41-swiglu-tail`；
  unverified_paths 增 `swiglu_deploy_variants`（masked/pad-aware/kernel
  等价未执行/SWIGLUOAI 域外/MegaMoE SM100）与 `dspark_draft_compress`
  （尾部归属已核、草稿本体未建模）；evidence_refs +
  `ev-vllm-v41-swiglu-tail`；note 更新。
- `importers/source_map.py`：新条目 `sm-v41-swiglu-tail`
  （both_sides_covered=True；uncovered_domain 五项）；
  `sm-v41-csa2-moe-unverified` 缩窄至核本体（R22 旧文本同步修正）。
- `validation/registry.py`：`ev-vllm-v41-swiglu-tail` 记录
  （source-checked，executed=False，limits 七项）。
- memory/traffic：**无变化**（clamp 不改布局/记录宽度；尾部全 0 不产生
  新缓存组）；M06 replay 前后相等。

## 4. 独立 oracle 与负例（tests/test_v41_swiglu.py，32 项）

- §1 pin 三重校验（13 文件 blob sha1 == vllm-tree == pin + 磁盘
  SHA256/size）+ 覆盖链文件清单。
- §2 pinned 锚行在场（27 处关键行，防 pin 漂移）。
- §3 纯 stdlib 数值 oracle（独立实现，不 import 生产代码当 oracle）：
  gate 单侧/up 双侧不对称、恰在限值与超限连续、上下界钳制、gate 饱和
  （2L 与 3L 输出相等）、gate=0 → 0、NaN 传播（clamp/sigmoid 保留）、
  **SWIGLUOAI 家族常数（7.0/1.702/β=1）与 V4.1 输出差异**（证明家族
  常数实质影响数值且域外）、ops_per_elem=6 与 extras 一致。
- §4 ModelIR 事实：extras.swiglu/覆盖 mapped/gaps 解除/draft_ratios
  [0,0,0]/structure_complete 仍 False（核本体阻断）/candidate_block_size
  补漏回归。
- §5 手算绝对值：TP1 routed 6·2304·6=82,944、shared 13,824；TP8
  inter_rank=288 → 10,368/1,728，40 层合计 483,840；EP8 expectation
  int(0.75·2304·6)=10,368 且与 moe_routed assignment 口径一致；
  Qwen36 无 act 行负例；moe_rows 面外行逐字段不变 + act 行字节 0。
- §6 负例 11 项：swiglu_limit ∈ {0, −10, NaN, inf, "10", 缺失}；尾部
  [0,0,1]/[2,0,0]/[0,3,0]；长度 44（死配置）/41（部分尾）拒绝；
  len==L（无尾）→ else 分支默认 [0,0,0] 正例。
- §7 replay 归因冻结（§5 详见）+ `--check` 复算 + 公式/证据/source_map
  注册事实。

## 5. 固定场景 replay 归因（docs/evidence/swiglu-tail）

场景：TP8/B1/ctx32768/quant=V41_QUANT（与 R22 quant-map 相同拓扑；
`tools/replay_v41_swiglu.py`，`--check` 无漂移复算）。before.json 在
main b8585a1 干净生产代码上捕获（baseline-provenance 登记 head/状态）。

| 字段 | delta | 归因 |
|---|---|---|
| flops | **+483,840** | 40 层 ×（moe_act 10,368 + moe_shared_act 1,728）= 40×(6·288·6 + 288·6) |
| rows | **+80** | 40 层 × 2 新行 |
| weight_read/expected、act_read/write、kv/state、M06 known_peak | **0** | 字节/布局无变化（单一所有者；clamp 不改记录宽度） |
| M01 | +3.26e-07 ms | 解析计时微增 |

面外行逐字段冻结（AUDITED={moe_act, moe_shared_act}）；逐字段
totals delta = Σ by_op delta 断言成立；测试 §7 同步冻结。

## 6. 剩余限制（不宣称完成）

- clamp CUDA kernel 与 native 的数值等价未执行（vLLM 测试在册未跑）；
  masked 每 expert padded 激活路径、pad-aware Triton 跳过未路由槽/
  非本地专家的运行时差异未入账（按逻辑 assignment 计费，与 moe_routed
  同口径）；SWIGLUOAI（alpha=1.702/beta=1.0/limit=7.0）与 SWIGLUSTEP
  家族为 V4.1 域外；native MegaMoE activation clamp（SM100）语义同源
  未执行。
- DSpark 草稿层本体未建模（MoE E=128/k=3、非因果 topk 扩展、context
  KV 投影）；非零 compress_ratios 尾部按域外拒绝——**没有为解除门禁
  而猜测**；`n_mtp_layers` 覆盖 `num_nextn_predict_layers` 的配置
  组合未建模（official 资产无该键）。
- 稀疏 MLA 注意力核本体（flops 4d+5 惯例）仍为 v0.1 移植——
  structure_complete 的最后一个阻断项；swiglu/尾部/config 覆盖的解除
  不代表核本体已核。
- Qwen36 家族 silu 微操作维持 R06 冻结口径（未计元素级 op）——与
  V4.1 的逐 op 计量存在跨模型口径差，登记于 registry limits，不在本
  PR 扩散修改（改 Qwen36 冻结面需独立 replay 归因）。

## 7. 结论

swiglu_limit 与 compress_ratios 尾部两项缺口按 13 文件 pin
source-checked 解除（G-C15）；V4.1 config 覆盖完备。新增计量为
纯增量（+483,840 flops/+80 行，decode TP8/B1），字节与内存账本不变。
structure_complete 仍 False（核本体）；部署排名资格不变。全部验证为
源码阅读 + 本仓独立 oracle + 解析 replay；无 CPU forward/GPU 实测。
