# 能力缺口清单（需要工程实现的能力）

与 [数据缺口清单](data-gaps.md) 分开维护：本页只列**工程实现**缺口（不依赖新数据，
或先有实现才能消费数据）。每项含：影响面/缺什么/旧仓为何不足/需要什么实现/
验收方式/当前降级/优先级。日期 2026-11-09；2026-11-10 更新（hc 建模、G-C10、
新增 G-C11；A 组轮 G-C11 解除）；2026-11-11 更新（R04：G-C1 单机 Engram 解除）；
2026-11-12 更新（R20：G-C12 CSA2 索引打分/top-k/candidate block 解除；R07a：G-C3
DFlash 服务栈核对部分解除；R21：G-C13 V4.1 MoE scoring/top-k/分组路由解除）；
2026-11-13 三轮复核（G-C13 证据链修正：EP 只按 mega 断言、共享专家独立相加、
ExpertMapManager 未入 pin 的读取点登记）；2026-11-13 gate 字节修复
（GateLinear 权重沿 loader BF16、输入由 mHC layer_input BF16；新增
linear.py/base_loader.py/layer.py pin；dtype 域固定 official V4.1 bf16）。

## G-C1 V4.1 Engram 语义建模（单机算子已解除 2026-11-11 R04；hc 已解除 2026-11-10）⭐P0

- **影响**：必需场景①（deepseek_v41_flash）全部性能/容量结论——structure_complete=
  False，M02/M04/M05/M06 降级，不参与排名。
- **2026-11-11 R04 进展**：**单副本路径 Engram 已建模解除**——common/engram.py
  pin（vllm @f358f057，blob sha1 ba4ceea6 对 vllm-tree.json 校验）后：素数桶
  表布局（FP8 e4m3 + ue8m0 264 B/行、层桶和 == engram_num_embeddings 精确相等）、
  批级 n-gram 哈希/查表/wkv 投影/门控注入四行计入 count（engram_*），表/wkv/qk/
  staged 对象计入 memory（表默认 cpu_offload 研究假设，host 驻留不进 GPU 峰值；
  场景显式 engram_table_placement 输入），TP>1 补 head/wkv 两次 all-gather
  通信行；测试 tests/test_v41_engram.py（独立 Miller-Rabin/首 8 素数/表字节/
  手算锚点/守恒/负例，24 项）。公式 v2.engram_* 四条；证据 ev-vllm-v41-engram。
- **2026-11-10 进展**：**hc（mHC 混合流）三键已建模解除**——每层 2 次边界算子
  （post 流更新+系数 GEMM+sinkhorn+collapse+RMSNorm 融合核；首层 hc_expand 广播；
  FP32 权重每卡复制）计入 count（hc_* 行，替代该族通用 norm/add 行防重复计费）
  与 memory（-w-mhc 分配）；测试 tests/test_v41_hc.py（手算锚点/整模型复算/TP
  不变性/复制守恒/权重守恒）。公式 v2.mhc_* 四条。
- **2026-11-12 R05 进展**：14 kernel 文件 blob SHA1/SHA256 已 pin，prenorm/
  comb post、首层/出口与 broadcast 派生容量已修正；Sinkhorn 算术 op 公式
  m4/iters20=1400，norm 4h+2 仍为近似惯例；源码阅读仅 source-checked。
  未执行 kernel/CPU/GPU；部署变体/实际 split 与物理流量仍缺。
  [逐项复验、差异与 replay](../validation/v41-mhc-kernels-20261112.md)。
- **仍缺（G-C1 剩余）**：Engram **DP 分片交换**（all-to-all/EngramPeerExchange/
  symmetric memory/DPSharedEngramStorage）与 EngramBatch MXFP8 打包路径未建模
  （本仓默认单副本路径不启用，登记未覆盖）；cpu_offload 运行时行为（预取流/
  TLB/GB200 排序阈值 36864/4096）未建模；amd/engram.py 变体未 pin；mhc
  kernel/CPU/GPU 执行、实际 split-k 与部署变体未验证（源码已由 R05 pin 入册）。
  CSA2 索引打分/top-k/candidate block 已由 R20 解除（见 G-C12）；MoE
  scoring/top-k/分组路由已由 R21 解除（见 G-C13）；量化映射、swiglu clamp、
  稀疏 MLA 注意力核本体、compress_ratios 尾部仍为独立缺口。
- **解除条件**：剩余项 = DP 交换路径 pin+建模（或显式永久登记单副本口径）、
  mhc 部署变体/运行时 split/物理流量与执行证据；structure_complete 还需
  量化/稀疏 MLA 核本体/compress_ratios 尾部缺口闭环（CSA2 索引与 MoE 路由
  已分别由 R20/R21 解除）。
- **当前降级**：structure gate 拦截（capability 可见；Engram 单机/hc 部分已不再缺）。
- **优先级**：P0（必需模型场景）。

## G-C13 ~~V4.1 MoE scoring/top-k/分组路由~~ → 已解除（2026-11-12 R21）⭐P0

- **解除交付**：moe-pin.json 14 文件 pin（vllm @f358f057，blob sha1 对
  vllm-tree.json 逐项校验 + SHA256；deepseek_v4 基类 MoE + eplb_util +
  fused_moe router 8 文件 + moe_output + linear.py/base_loader.py/
  fused_moe/layer.py）。语义：GateLinear(h→E, out_dtype=fp32)
  Replicated 路由 GEMM，**权重沿 model loader BF16**（未传
  quant_config/params_dtype；linear.py L287-295/L179-199；base_loader.py
  L86；gate_linear.py L306-341）；**输入由 pinned mHC layer_input_cur
  BF16 证明**（tilelang.py L900/L1096，R05 pin；model.py L731-816 直接
  self.ffn(x)），不是由任意 model dtype 泛化；有效域仅 official V4.1
  config dtype=bfloat16（非 BF16 拒绝）；sqrtsoftplus 打分（>20 护栏）；
  noaux_tc e_score_correction_bias 只导选择不进权重；k=6 轮平坦偏置
  argmax（**无分组**——config 无 n_group/topk_group 键，分组仅 V3 式
  use_grouped_topk 路径；默认 False 已随 fused_moe/layer.py 入 pin）、
  平手最小 id、NaN→−1e30；sum→1 归一 ×routed_scaling_factor=1.5
  （sum≤0/NaN 时分母取 1）；共享专家（1, inter 2304, 无 gate）在
  routed 归一后权重 1.0 **独立相加**（moe_output.py L151-175；L447-464
  融合槽未启用）；V4.1 主干 num_hash_layers=0；视觉 checkpoint bias_vl
  哨兵判定；EP **保守整除域**（仅 mega 路径 L357-362 断言
  n_physical % ep == 0；fused L419-421 的 start_idx 只赋值、未被所有权
  消费，fused 通用余数映射未入 pin/未建模，不主张 fused 非法；与
  qwen35 pin 不同，非整除组合在建模域内拒绝）。
  count 三行（moe_gate/moe_score/moe_topk_renorm；gate 权重沿 loader
  BF16、输入由 mHC layer_input BF16，均 2B；每层 weight 3,932,160 B，
  FP32 只 logits 出口）、memory
  router_b 同步 2B + 路由偏置对象（40×3,072 B）、model_ir
  extras.moe_route + config 域校验（softmax/greedy/norm=false 拒绝）、
  公式 v2.moe_route_*。
  首轮 Codex 验收（不通过）后修正：报告/注册表全部行号对实销行号
  （sqrtsoftplus dsv4 L85 等）、oracle −87.3 normal-range 近似标注、
  replay 工具补 --check 无漂移模式、PR 描述副本入仓（报告 §8）。
  第三轮复核（2026-11-13）修正：fused EP 的 start_idx 只赋值、未被
  所有权消费（EP 规则只按 mega 断言，删除"floor 覆盖性"）；共享专家
  实际为独立相加；ExpertMapManager 未入 pin 的默认值登记为读取点；
  证据包加 SHA256 自校验（报告 §9）。gate 修复（2026-11-13）：
  GateLinear 权重沿 loader BF16、输入由 mHC layer_input BF16 证明，
  新增 linear.py/base_loader.py/fused_moe/layer.py pin，重放
  before/after/attribution；dtype 域收紧为 official V4.1 bf16，
  float16/float32/缺失拒绝（报告 §10）。
- **验收**：tests/test_v41_moe_route.py 23 项——pin 双侧校验与文本锚点、
  纯 stdlib 语义 oracle（归一 sum==1.5/bias 不进权重/平手最小 id/NaN/
  normal-range 近似护栏/无组键）、手算绝对值、tp/ep 复制不变、EP 保守域负例+
  E_local=48、count↔memory 对账、config 域负例、structure gate 不解除、
  replay 冻结。默认 TP8/B1/ctx32768 replay（gate BF16 修复后）：decode
  +600,480 FLOPs、weight_read +78,766,080B、act_read +266,560B、
  act_write +48,000B、rows +80、M01 +0.023606ms、steady/known_peak
  +78,766,080B（attribution.json 逐 op 归因，面外行冻结）。证据
  ev-vllm-v41-moe（source-checked，executed=False）；
  [报告](../validation/v41-moe-route-20261112.md)。
- **仍缺（登记不冒充）**：kernel/CPU forward/GPU 未执行（op 为解析算术
  口径）；native MegaMoE bf16_mega_gate 大批路径（EP+fp4+SP，deep_gemm
  未入 pin）、GateLinear 的 bf16 专核/cuBLAS 档未执行（fp32/bf16x3
  分层只适用 fp32 router 权重，V4.1 gate 为 BF16）、dsv4_topk Triton 档约束
  （E∈{256,384}/k=6）外的 E/k 值未独立核；DSpark 草稿层 MoE（E=128/k=3）
  方法层不建模；padding skip/CUDA-graph 整宽路由缓冲未入账；
  defer_finalize 需 scaling==1.0（V4.1-Flash=1.5 不适用）；量化专家
  权重布局（quantization_config 独立缺口）。

## G-C12 ~~V4.1 CSA2 索引打分/top-k/candidate block~~ → 已解除（2026-11-12 R20）⭐P0

- **解除交付**：csa2-pin.json 21 文件 pin（vllm @f358f057，blob sha1 对
  vllm-tree.json 逐项校验；其中 13 文件为任务书预置、8 文件网络拉取后同验；
  补入 sparse_attn_indexer 直接 import 的 v1 backend，移除未入 tree 的重复
  candidate_blocks 副本）。
  count：index_select 拆为 index_score（2·pairs·(ih·di+ih) + fp32 logits）与
  index_topk（2·pairs+T·topk 选择惯例）；candidate_publish → candidate_select
  （源层 20：top-2048×8 压缩位块，块最大值 NaN 传播、最新块 +inf 钉扎、−1
  填充）+ candidate_mask（消费层 24/28/32/36 mask 后自有 top-k）；短上下文
  旁路 end//r≤512 走 index_fill（0 flops 纯 store，修正旧式 T·topk 误计）；
  compress_pool（r2 softmax 门控 7d+9/group + 环形 state cap8×4096B fp32/行/
  请求，r1 纯 norm exact）；compressed_insert / index_k_store（132B/行）合并行；
  sparse_index_build（decode 每 (kv源:索引源:r) 等键运行首层一次，128 对齐）。
  memory：SWA/压缩缓存按打包记录 584|528 B/token（显式 v41_kv_record 场景
  配方，进身份）、索引 K 132B、ratio2 环形 state、compressor/wk/weights_proj
  bf16 固定权重（输入/输出激活字节同按 pinned dtype）、主 attention
  q_norm/kv_norm bf16 常驻（40 层 143,360 B）、topk/candidate 共享缓冲下界
  （模型级单例）。`v41_kv_record` 与目标设备 family 强校验：H100/H200 只允许
  sm90，L40（SM89）拒绝 sm90，当前无 SM100 设备故 sm100/nvfp4 一律拒绝。
- **验收**：tests/test_v41_csa2.py 31 项——pin 完整性、拓扑/运行首层/candidate
  所有权独立推导、短上下文两侧分支（含混合 ctx=800：r2 旁路 r1 全打分）、
  decode/prefill 打分与 prefill 候选独立复算、select/mask/fill stdlib 独立
  语义（NaN 排序最前、钉扎、-1/-inf、候选含 -1 不掩块负例）、记录宽度/环形/
  缓冲/128 对齐对账、bf16 权重与激活一致性、主 attention q_norm/kv_norm
  单层/整模型 oracle、记录族设备兼容 6 负例+1 正例、config 域负例 4 项、
  replay 冻结。证据 ev-vllm-v41-csa2（source-checked，executed=False）。
- **仍缺（登记不冒充）**：稀疏 MLA 注意力核本体（flops 4d+5 仍 v0.1 移植）；
  SM100 sparse-logits（消费层只打分候选块，O(候选) 而非全候选）、decode_topk
  CuTe BF16 直选、bf16 logits/deep_select、chunk_decode 预算、fused_mqa_topk、
  MXFP4 索引缓存(68B)；TP≥2 uniform decode ≥12 行的行分片+all-gather、
  PCP/DCP、ROCm/XPU；CUDA-graph 静态 max_model_len 整宽 logits 缓冲与候选
  mask 整宽 -inf 尾写（workspace 池缺口）；cooperative 候选值相等边界近似。
  kernel/CPU forward/GPU 均未执行。

## G-C2 verify/prefill 形状域建模 + 蒸馏扩展（消费 G-D4）

- **影响**：投机 verify 注意力块、未来 TTFT/prefill。
- **缺什么**：CalibratedPricer 目前只有 decode(Q=1) 注意力域；prefill/混合批蒸馏
  与"多 query 前缀"形状映射。
- **旧仓不足**：pi-viewer 有数据有训练器；新仓侧缺映射与域声明。
- **需要**：distill 扩展（prefill-even 表）+ calibrated 映射（显式假设登记）。
- **验收**：verify 块进域（或显式假设近似档）+ 留出误差 + 回放不变性。
- **当前降级**：verify 注意力解析回退（reason 记录）。
- **优先级**：P1；依赖：G-D4 数据（蒸馏侧可先行）。

## G-C3 DFlash 服务端集成逐行核对（SGLang DFLASH / vLLM PR#40898）—— 核对部分解除（2026-11-12 R07a）

- **2026-11-12 R07a 进展**：**逐行核对已完成**——vLLM @dc74b8cb（13 文件
  SHA256 pin，与 vllm_qwen35_pin 同提交）与 SGLang @74cb973d（12 文件 pin）
  双栈审计：draft/verify 调度、hidden 传递（8 层拼接+fc 复制权重）、KV 回退
  （被拒后缀不落缓存）、拒绝采样（greedy 前缀/随机比值检验+目标分布 bonus）、
  通信字节、计时边界（两栈 accept/prepare 均为 GPU kernel，但 vLLM 的
  prepare_dflash_inputs 是图外 eager Triton、draft sampler 在 draft FULL
  图内；SGLang draft sampler 在图内、accept/bonus 为独立 launch）。
  对账修正三处：ctx 投影按 **verify 跨度 B×blk** 计（两栈全跨度投影、只写
  committed 行；原按 E[N] 前缀计属漏计）、草稿行按**生效块**
  plan.block_size（原按 checkpoint 16 与 verify 双口径）、**full 草稿层
  非因果**（pinned HF/vLLM/SGLang 的 layer_types 语义；原按因果 Q(Q+1)/2
  计，full 层应为 Q²）。同时补入 ctx K 逐头 RMSNorm/RoPE 与 vLLM
  permute+contiguous 布局拷贝（全跨度，原账本只计块内 K norm/Q rope）。
  `t-draft-buffers` 按已知并存组 max 计（lower bound）；flat 在 contiguous
  后释放，不与 K norm 输出相加，峰值仍因运行时池未知而保持 unknown。
  e2e 测试两侧在册（GSM8K 精度/接受长度门槛）。
  `dflash_serving_integration`：colocated →
  **supported（source-checked，非已部署实测）**；dt_serial/dt_pipeline →
  **unsupported（两栈均无 DT 分离；SGLang disagg 是 target 侧 PD）**。
  证据 ev-vllm-dflash-serving / ev-sglang-dflash-serving；source_map
  sm-vllm-dflash-serving / sm-sglang-dflash-serving；
  [R07a 报告](../validation/dflash-serving-audit-20261112.md)。
- **仍缺（G-C3 残留）**：served 已部署实测（R07b 真实 trace+匹配环境）；
  SGLang 采样验证 kernel 可用性（构建/设备相关，不可用降级 greedy 分布不保持）；
  SGLang GDN/mamba 目标态回退（extra_buffer）拷贝成本；vLLM
  DraftLogitsCache（probabilistic [max_reqs,N,V] 跨轮缓冲）未入 R02b 账本
  （场景无 draft_sample_method 键，待键登记后接）；两栈 logits/候选通信
  字节口径分歧（本仓行=vLLM 全量 gather；SGLang greedy 为局部 (max,id)
  all-gather，差异公式已登记未入行）。
- **影响**：部署矩阵 served 标记（source-checked ≠ 已部署；R07b 前不冒充）。
- **当前降级**：colocated supported（源码核对）/dt unsupported；accept 与
  KV 回退 kernel 耗时仍 unknown（stage_cost_complete）。
- **优先级**：P1（核对部分已解除；残留按 R07b/后续键登记推进）。

## G-C4 请求级事件轨迹 + 单请求超前（lookahead）

- **影响**：M17 TTFT/ITL、单请求 TPOT、取消/浪费资源账（M3 剩余部分）。
- **缺什么**：请求级事件 schema 扩展（request_id 贯穿）、超前分支/取消/浪费建模。
- **旧仓不足**：model-verse 有跨轮事件资产但无请求级超前；本仓 schema 已留字段。
- **需要**：resource_core 扩展 + reduce 的请求口径归约 + V07/V08 负例。
- **验收**：单请求依赖链独立重建测试；取消浪费入账；TTFT 人工 oracle。
- **当前降级**：M01 附"周期≠单请求延迟"注；请求指标不输出。
- **优先级**：P1（M3 收尾）。

## G-C5 双口径校准列（CUDA_EVENT vs KERNEL_ONLY/graph，消费 G-D6）

- **影响**：小 batch 结论的口径敏感性（3× 量级差）。
- **缺什么**：profile_store 支持 measurement_type 维度 + 组合策略选择（显式登记）。
- **需要**：distill 扩展 + pricer 参数化；UI 展示两列。
- **验收**：两口径各自留出报告；切换不改原始计数（G7 类重放）。
- **当前降级**：单一 CUDA_EVENT + eager gap（披露）。
- **优先级**：P1；依赖：G-D6。

## G-C6 服务特性（Chunked Prefill → PD 分离 → Prefix Cache，Q5 顺序）

- **影响**：服务端口径指标（TTFT/SLO/goodput 部署级）。
- **缺什么**：到达过程、预算组批、KV 分页底座（M4 范围）。
- **旧仓不足**：pi-viewer/frontier 有 DES 实现可参考（不可整体搬入，按证据抽机制）。
- **需要**：serving 子包 + UNSUPPORTED 表逐项转正。
- **验收**：每特性先结构/状态对照，再指标人工 oracle（V 系列）。
- **当前降级**：显式 unsupported（不空实现）。
- **优先级**：P2（按 Q5 顺序）；依赖：G-C4。

## G-C7 Qwen3.8 本地模型身份辨认

- **影响**：四历史锚点之一的正名与回归。
- **缺什么**：用户提供本地模型路径/revision → 以 model_type/revision 辨认（可能实为
  Qwen3.5/3.6 家族）。
- **当前降级**：PENDING_ASSET 注册，不入目录。
- **优先级**：P2（等用户输入）。

## G-C8 三仓完整对账（V09b）

- **影响**：M5 迁移准入与旧入口切换。
- **缺什么**：pi-viewer 侧适配器（本仓已有 model-verse 薄适配器同款模式）；三方
  （旧值/新值/独立 oracle）对账矩阵。
- **当前降级**：V09b 未做；旧仓入口不动。
- **优先级**：P2；依赖：G-D2/G-D3（性能列对账需先有域）。

## G-C10 ~~UI 场景栏精度选择器~~ → 已解除（2026-11-10）

- 场景栏新增精度档选择（BF16 16/16/16/16、W8A8、W8A8+FP4 专家），档位映射后端
  量化契约（scenario.validate 4/8/16）；检查器精度标签改从场景快照读取（不再
  写死）。**不把"profile 需要 BF16"转成输入精度**：profile 域门禁仍按其采集
  精度独立核对，不匹配组合显示具体回退原因。
- 验收（历史归档）：`tests/test_calibration_h200.py::test_archive_bf16_calibration_still_verified_offline`
  保留离线校准数值验证；产品 API 拒绝归档模型/历史 profile，静态接线核对保留。
  30B 场景不得再经 `/api/v2/run` 作为 UI 验收通道。
- 浏览器像素级观感仍待人工复核（本会话无图像通道，同前口径）。

## G-C11 ~~per-rank MoE 分片账本统一（V04 扩展；235B MoE 准入前置）~~ → 已解除（2026-11-10 A组 R01；235B 现已归档）

> **2026-11-10 后续**：235B 已移出活跃 catalog；本条的 per-rank 配方仍服务
> Qwen3.6，但“235B 准入”不再是产品路径。

- 统一配方落地：`simverse/lowering/parallel.py` ParallelPlan（vLLM @dc74b8cb
  pin）——EP 关 routed inter/tp、EP 开 E//ep(+余数) 整专家、router 复制、
  shared inter/tp、attention KV 头切/复制、GDN 切分、词表行切；count 行、
  memory 分配、EP 通信三方同 plan。
- **同日二轮审计补漏**（[报告](../validation/agroup-lmhead-ctxkv-audit-20261110.md)）：
  lm_head 词表行切后 logits_gather 集合通信（logits_processor.py L185-198
  入 pin）、DFlash 草稿 lm_head/ctx KV 二次读流量、vLLM `_fused_kv_weight`
  驻留副本与三件结构化临时（ctx 归一副本/ctx 投影输出/aux hidden）入账本；
  **三轮修正（同日，外层验收前）**：ctx-KV 改为显式
  `draft_ctx_kv_path`（raw/MXFP8/逐层回退），回退按完整 qkv（含 Q）计数，
  k_norm 堆叠三路径均计；**复核修正**：k_norm 按 `DraftIR.dtype_hint`
  （config bfloat16→2B）计，与 `dense_w` 4/8/16 无关；MXFP8 的
  block-scale 布局仍未入 pin，按 packed 下界 + unknown 缺口登记。
- 验收：守恒测试 ×配方×tp∈{1,2,4,8} + EP 守恒 + 复制对象不求和 + 非法组合
  拒绝（tests/test_parallel_sharding.py；二轮后 21 项，含 logits_gather 与
  草稿二次读独立手算 oracle）；235B manifest defer 理由已
  改指 G-D2 数据准入（分片混尺不再是阻塞）。
- 残留：G-D2 的 moe 蒸馏准入与留出验收（数据侧，独立进行）。

## G-C9 小形状地板解析模型（暂缓；判定实验已明确）

- **影响**：解析档在小 batch 相对 CUDA_EVENT 组合估计低 1.95–11.59×（9 点）。
  **2026-11-10 口径修正**：该倍数是相对组合估计的差异，非部署延迟结论；
  host launch 与 kernel 地板在 CUDA_EVENT 口径下不可分。
- **缺什么**：可解释地板模型所需的判定数据——G-D6 的 KERNEL_ONLY per-op 时间列
  （区分 host launch 与 kernel 执行）+ graph 模式整步时间；以及 G-D5 E2E 对照
  （约束"实际运行时"一侧）。
- **需要**（数据齐后）：price_row 可选地板项（每算子固定开销 + 达峰 shape 阈值），
  参数显式 assumed 并做敏感度扫描；不回填 measured 冒充解析；适用域显式登记。
- **判定实验**（G-D6 数据到手后）：对 B1/B8 各算子比较 CUDA_EVENT vs KERNEL_ONLY
  median——差值≈host/launch 贡献，残差≈kernel 地板；两者分别入地板模型后对
  留出点验收（p90 改善 + 域外不外推）。
- **当前降级**：解析档低 batch 低估已知并按口径披露（loop1 报告 §5）。
- **优先级**：P1 但**阻塞于 G-D6/G-D5 数据**（在数据前实现即是无据拟合，不做）。
