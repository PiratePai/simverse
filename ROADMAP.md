# Simverse Roadmap

更新：2026-11-10。产品范围固定为 **Qwen3.6-35B-A3B + DFlash** 与 **DeepSeek-V4.1-Flash**。历史模型仅用于内部回归与校准，不继续扩展产品目录。以下为待办，不代表已交付或交付日期承诺。

## 当前已交付

- [x] 重点模型产品目录、DFlash 配对门禁、具体资格失败原因。
- [x] ModelIR、计数/成本/执行分层，CLI/Web/导出共用结果。
- [x] Qwen3.6+DFlash 解析实验、DT 串行及初步微批流水。
- [x] Qwen3.6 原生 MTP 头已双栈 source-checked 建模（R23，2026-11-13；
  kernel/CPU/GPU 未执行与部署变体见 G-C15）。
- [x] V4.1 部分结构及 mHC 边界计数；Engram 单副本路径已 source-checked 建模
  （R04，2026-11-11；DP 分片交换/cpu_offload 运行时行为未覆盖，见 B 组表）；
  CSA2 索引打分/top-k/candidate block 已 source-checked 建模
  （R20，2026-11-12；注意力核本体/部署变体仍缺，见 B 组表）；
  MoE scoring/top-k/分组路由已 source-checked 建模
  （R21，2026-11-12；MegaMoE/DSpark 草稿 MoE/padding skip 部署变体仍缺，
  见 B 组表）。
- [x] 历史 H200 部分算子校准、来源与覆盖域门禁；不代表重点模型实测。
- [x] A 组（2026-11-10）：TP/EP per-rank 统一配方、同卡对象身份账本、
  生命周期峰值分列、预算契约与可部署性四态判定链路（见 A 组表；真实
  预算与运行时池证据仍缺，默认场景保持 unknown）。同日二轮审计补漏
  （lm_head logits 通信/草稿二次读/fused 副本，见
  [二轮报告](docs/validation/agroup-lmhead-ctxkv-audit-20261110.md)）。

## 组织方式：独立交付，必要处联合验收

P0/P1/P2 表示优先级，不是要求整阶段完成后才能启动下一阶段。下面各组可独立推进；表内只列真实的硬依赖。可以先完成接口、独立 oracle、数据审计或局部模型，依赖未完成时不得宣称联合能力已通过。

保留 R01–R14 编号便于追踪；拆分项使用后缀，新增设备事项使用 R15–R19。一个子项通过即可单独合入，不能因同组其他项缺数据而阻塞。设备与模型分别维护能力矩阵，增加设备不需要增加产品模型。

## A 组：Qwen 分片与同卡可部署性（P0，最终判定耦合）

状态更新 2026-11-10：五项工程实现全部落地（R01–R03b），详见
[A 组验证报告](docs/validation/agroup-tp-ep-colocated-20261110.md)。
**联合验收只解除到"判定链路完整"**：默认场景同卡可部署性仍为 unknown
（可分配预算未提供 G-D7 + workspace/graph/NCCL 池无尺寸证据），不得
表述为已确认可部署或已确认 OOM。

| ID | 状态 | 交付与证据 |
|---|---|---|
| R01 | ✅ 完成 | per-rank TP/EP 统一配方（vLLM @dc74b8cb pin，`simverse/lowering/parallel.py` ParallelPlan；count/memory/lowering 三处共用）；跨 rank 守恒 + 独立手算 oracle + 非法组合拒绝（tests/test_parallel_sharding.py）；默认场景前后差异逐项归因（验证报告 §3）。**二轮审计（同日）**：补 lm_head logits_gather 集合通信（logits_processor.py 入 pin；decode/verify/草稿各 1 次）与草稿 lm_head/ctx KV 二次读流量（[二轮报告](docs/validation/agroup-lmhead-ctxkv-audit-20261110.md)） |
| R02a | ✅ 完成 | 同卡对象身份：共享 embed/lm_head 单一物理对象 + consumer 引用（零字节），废除草稿幻影副本与失配去重键；草稿批=target 批；per-rank 账本含 EP 余数差异与最坏 rank（tests/test_colocated_memory.py）。**二轮审计**：vLLM `_fused_kv_weight` 全层 K/V 行 cat 驻留副本入账。**三轮修正**：改为显式 `draft_ctx_kv_path`——仅 raw 建融合副本；MXFP8 独立对象；4bit 回退无副本但 k_norm 堆叠仍按草稿 dtype（bfloat16→1,536 B）计 |
| R02b | ✅ 完成（结构化部分） | 生命周期分列：常驻/每请求/跨轮/阶段临时（draft/verify 串行取 max）；workspace/CUDA graph/NCCL 池作为 unknown_components 缺口传播（不填 0，峰值不完整 → 判定 unknown）。**二轮审计**：补 ctx 归一副本/融合 ctx-KV 输出（phase:draft）与 verify aux hidden（phase:verify）三件 pinned 形状临时。**限制**：临时缓冲只含 pinned 形状可推导项；运行时池尺寸无证据，完整运行时峰值未验证 |
| R03a | ✅ 完成 | 预算输入契约（nominal/allocatable(kind: measured/hypothesis)/额外开销分列、gross 语义防重复扣除、缺省 unknown 不套标称）；场景身份包含 EP/budget；CLI/API/UI/导出一致（测试 + 浏览器实测） |
| R03b | ✅ 完成（判定链路） | 四态判定 unknown/oom/feasible/feasible-conditional + 已知下界超标称即 OOM；假设预算仅条件结论且不进实际部署排名（compare 资格排除）；负例全覆盖（tests/test_budget_feasibility.py）。**实际可部署结论仍阻塞于真实预算证据（G-D7）与峰值完整性** |

| ID | 硬依赖／联合验收点 |
|---|---|
| R01 | 固定实现源码；无其他 roadmap 前置项 |
| R02a | 对象测试可独立；分片后的容量验收接 R01 |
| R02b | 对象接口来自 R02a；完整 per-rank 峰值接 R01 |
| R03a | 可独立，不必等容量模型或实测数据 |
| R03b | R01 + R02a/b + R03a；实际预算证据。假设预算仅产生条件结论 |

本组必须联合验收才能解除“同卡可部署性未知”；不能仅凭 R03a 输入一个预算就标记为真实可部署。当前判定链路已贯通：给定 measured 预算 + 完整峰值即可给出 feasible/OOM；两缺口（G-D7 预算、运行时池尺寸证据）解除前默认场景保持 unknown。

## B 组：V4.1 结构补齐（P0，分模块独立）

| ID | 独立交付物与验收标准 | 硬依赖／联合验收点 |
|---|---|---|
| R04 | ✅ 完成（单副本路径，2026-11-11）：common/engram.py pin（vllm @f358f057，blob sha1 对 vllm-tree.json 校验）→ 素数桶表布局/批级哈希/查表/wkv/门控四行入 count、表/wkv/qk/staged 对象入 memory（表默认 cpu_offload 研究假设，显式 engram_table_placement 输入）、TP 通信行；独立 oracle tests/test_v41_engram.py 24 项（层桶和 == engram_num_embeddings 精确相等；PR #3 合入后复核修正 rank 局部查表与固定 BF16 字节）。证据 ev-vllm-v41-engram；[报告](docs/validation/v41-engram-20261111.md)。**剩余（未完）**：DP 分片交换（all-to-all/peer/symmetric memory）与 EngramBatch MXFP8 路径、cpu_offload 运行时行为、amd 变体；host 查表时延 unknown，M01 仅 GPU 侧下界 | Engram 源码（已 pin）；不依赖 Qwen 或设备扩展 |
| R05 | **已完成 2026-11-12**：14 kernel 文件 blob/SHA256 pin；prenorm/comb post/Sinkhorn 1400 op 修正、首层特例与出口两行、broadcast 派生容量；精确算术与 norm 惯例近似分列，独立 oracle + 前后 replay。[报告](docs/validation/v41-mhc-kernels-20261112.md) | source-checked；CPU/GPU 未执行，split/部署变体与物理流量仍缺 |
| R20 | **已完成 2026-11-12**：CSA2 索引打分/top-k/candidate block——csa2-pin.json 21 文件 pin（blob sha1 对 vllm-tree.json）；index_select 拆分 score/topk、candidate select/mask 两级过滤（源 L20 发布 top-2048×8 块、消费层 24/28/32/36 mask）、短上下文旁路（fill 0 flops）、compressor 池化/环形 state、索引 K 132B/行、打包记录 584/528 B/token（显式 v41_kv_record，并与目标设备 family 强校验）、bf16 固定权重、共享缓冲下界、sparse_index_build 运行首层一次；独立 oracle 31 项（tests/test_v41_csa2.py）+ 前后 replay 逐 op 归因。证据 ev-vllm-v41-csa2；[报告](docs/validation/v41-csa2-indexer-20261112.md)。**剩余**：注意力核本体（4d+5 移植口径）、SM100 sparse-logits/decode_topk CuTe/行分片等部署变体、CUDA-graph 静态整宽 logits 尾写、量化映射/MTP 尾部不变（MoE scoring 已由 R21 解除） | source-checked；kernel/CPU forward/GPU 未执行；structure_complete 仍 False，部署排名资格不因本模块解除 |
| R21 | **已完成 2026-11-12；gate 字节修正 2026-11-13**：MoE scoring/top-k/分组路由——moe-pin.json 14 文件 pin（blob sha1 对 vllm-tree.json；+linear.py/base_loader.py/fused_moe/layer.py）；sqrtsoftplus 打分（>20 护栏，dsv4 L85）+ noaux_tc 偏置只导选择不进权重（L86-98）+ k=6 轮**平坦**偏置 argmax（无分组：config 无 n_group/topk_group 键）平手最小 id（L94-95）/NaN→−1e30（L87）+ sum→1 归一 ×1.5（sum≤0/NaN 分母取 1，L104-107）；GateLinear FP32 logits 出口（L270-277）且**权重沿 model loader BF16、输入由 mHC layer_input BF16 证明，均 2B**（未传 quant_config/params_dtype；linear.py L287-295、base_loader.py L86；tilelang.py L900/L1096、model.py L731-816；域固定 official V4.1 bf16，非 BF16 拒绝）、路由偏置对象 40×3,072B（bias+bias_vl 视觉哨兵 L63-76）、共享专家无 gate 归一后权重 1.0 **独立相加**（moe_output.py L151-175；L447-464 融合槽未启用）、主干无 hash MoE（L155）；**EP 保守整除域**（仅 mega 断言 L357-362；fused L419-421 的 start_idx 只赋值、未被所有权消费，fused 通用余数映射未入 pin/未建模，不主张 fused 非法；非整除组合 count/memory 双侧拒绝，与 qwen35 余数规则分立）；config 域校验（softmax/greedy/norm=false 拒绝）；独立 oracle 23 项（tests/test_v41_moe_route.py，含纯 stdlib 语义 oracle、dtype 域负例）+ 前后 replay 逐 op 归因（--check 无漂移复算）。首轮 Codex 验收后修正行号/EP 依据/−87 假设标注（报告 §8）；第三轮复核（2026-11-13）修正 fused EP dead start_idx、共享专家实际独立相加、ExpertMapManager 未入 pin 登记、证据包 SHA256 自校验（报告 §9）；gate 修复（2026-11-13）把 GateLinear 权重/输入改为 BF16 2B，weight_read/act_read/steady 与 M01 全部重放；dtype 域收紧为 official V4.1 bf16，float16/float32/缺失拒绝（报告 §10）。证据 ev-vllm-v41-moe；[报告](docs/validation/v41-moe-route-20261112.md)。**剩余**：MegaMoE bf16_mega_gate 大批路径（EP+fp4+SP）、DSpark 草稿层 MoE（E=128/k=3）、padding skip/CUDA-graph 路由缓冲、E/k 域外值核验 | source-checked；kernel/CPU forward/GPU 未执行；structure_complete 仍 False，部署排名资格不因本模块解除 |
| R22 | **已完成 2026-11-13**：量化映射——quant-pin.json 20 文件 pin（blob sha1 对 vllm-tree.json；dispatch 链 config/model+quantization/__init__+deepseek_v41/quant_config+config 类、MXFP8 线性 fp8/modelopt/quant_utils/mxfp8_utils、MXFP4 专家 mxfp4/oracle/routed_experts、kernel 域 kernels linear 4+humming 2、未量化类 vocab_parallel_embedding）。quantization_config 5 叶键 mapped：quant_method=fp8+deepseek_v41*→DeepseekV41FP8Config（override 链）；expert_dtype=fp4 提升（config 类 L46-50）派生 is_scale_e8m0；线性 ModelOptLinearMethod(kMxfp8Static,kMxfp8Dynamic)——e4m3 1B+ue8m0 (1,32) scale（checkpoint 32×32 块装载展开 (N,K/32)；K 每 rank 整除 32 断言 L2450-2454）；激活动态无落盘 scale；专家 Mxfp4MoEMethod——打包 2 值/byte (E,2I,H/2)/(E,H,I/2)+ue8m0 scale、分配前 Marlin 对齐 I→128/H→256（routed_experts L190-203+oracle L850-855）；embed/lm_head/RMSNorm 未量化 BF16（get_quant_method None→默认方法）；执行域注册设备 ≤SM90=Marlin W8A16/W4A16→激活 BF16（FlashInfer MXFP8/SM100 与 TRTLLM/DeepGEMM 均 SM100+；Humming cc=90 优先但需可选包→变体）；scale_fmt 键 loader 不消费（from_config 键表无）——非 ue8m0 按声明矛盾拒绝；KV 缓存无量化叶键（记录宽度仍 R20 配方）。落地：count/quant_layout+work/moe/head/hc/engram recipe 计价、memory scale/打包/对齐/embed-BF16 对象、pipeline 通信载荷 BF16、extras.quant、scenario V41_QUANT 冻结、公式 v2.quant_*、独立 oracle 31 项（tests/test_v41_quant.py）+ 前后 replay 逐 op 归因（--check 无漂移）。默认 TP8 replay：weight_read +120,616,960（scale+BF16 修正，final_norm 修双重计宽 −10,240）、weight_read_expected +221,184,000（40×6×Δ921,600 专家打包+对齐）、known_peak +14,359,147,520；flops/act 不变（场景 act=16 两侧同 BF16）。证据 ev-vllm-v41-quant；[报告](docs/validation/v41-quant-map-20261113.md)。**剩余**：Humming W4A8、SM100 W8A8 mxfp8 动态激活、emulation 反量化驻留、NVFP4/fp8 专家变体、Marlin 重排/workspace、成本 gemm 档未随 W8A16 改路由 | source-checked；kernel/CPU forward/GPU 未执行；structure_complete 仍 False（swiglu/核本体/尾部缺口，后由 R24 解除前两项），部署排名资格不因本模块解除 |
| R24 | **已完成 2026-11-13**：SwiGLU clamp + compress_ratios 尾部——swiglu-tail-pin.json 13 文件 pin（blob sha1 对 vllm-tree.json；语义核 activation.py SiluAndMulWithClamp、fused MoE 消费链 config/activation/utils/marlin_moe/layer、模型接线 deepseek_v4 nvidia+cpu、尾部归属 v41/attention+deepseek_v4_utils+dspark+sparse_swa、vLLM 数值测试 test_activation）。clamp 语义：gate=clamp(max=10) 单侧、up=clamp(±10) 双侧、alpha=1/beta=0（FusedMoEConfig 默认）；触发 MLP is-not-None vs Triton HAS_LIMIT>0（0 值分叉→域外拒绝 0/负/NaN/inf/缺失）；消费两路均注册执行域（Marlin apply_moe_activation + 共享专家 MLP.act_fn；MegaMoE clamp SM100 域外）。尾部 [40:43] 逐项归 DSpark/nextn 草稿层（层号 40-42；attention.py 按层号消费、else 0=纯 SWA、is_backbone 门控永不为源；后端去重层型集合消费）；长度域 {L,L+mtp}/尾部值 {0,1,2}/非零域外拒绝；入册 draft_compress_ratios。落地：moe_act/moe_shared_act 行（6 op/元素、字节 0 单一所有者）、公式 v2.swiglu_clamp_act、extras.swiglu + 键 mapped、candidate_block_size 补漏入 struct 集合（R20 注记漏登记修复）；V4.1 config 覆盖完备（structure_complete 仍由核本体阻断）。replay：decode flops +483,840、rows +80、字节/M06 不变（evidence/swiglu-tail --check）。独立 oracle 32 项（tests/test_v41_swiglu.py：stdlib clamp oracle/边界/饱和/NaN/SWIGLUOAI 差异/TP1-TP8-EP8 手算/负例 11/Qwen36 无行/replay 冻结）。证据 ev-vllm-v41-swiglu-tail；[报告](docs/validation/v41-swiglu-tail-20261113.md)。**剩余**：clamp kernel 数值等价未执行、masked/pad-aware 运行时差异、SWIGLUOAI 家族、MegaMoE SM100、DSpark 草稿本体与非零尾部 | source-checked；kernel/CPU forward/GPU 未执行；structure_complete 仍 False（核本体），部署排名资格不因本模块解除 |

各模块分别解除自己的缺口；整个模型的结构资格需全部相关缺口闭环，不能以一个模块通过代表整体通过。

## C 组：更多设备支持（P1，各型号与层级独立交付）

当前注册表已有 H100 SXM、H200 SXM、L40；这仅表示已有规格/解析入口，并非所有模型、精度和部署方式均已验证。

候选接入批次如下；这是待核对的目标清单，不是已支持声明。每个具体 SKU、显存版本、PCIe/SXM/整机形态独立登记，依据官方资料与可获得证据决定准入，不复制近似型号常数。

| 批次 | 设备候选 | 范围 |
|---|---|---|
| C1 | A100、A800、H800、L40S | 优先审计旧仓已有资产，逐项核对规格与来源后复用 |
| C2 | B200、B300、GB200/GB300 系统 | 单芯片、单卡、节点/机柜口径分离；系统互联不折算成单卡常数 |
| C3 | 昇腾 910B/910C、950DT、950PR | 规格档先行，逐型号核对官方身份；后端算子、精度与通信配方按昇腾实际实现独立验证 |

同一批次无需等待全部设备；C2/C3 的资料审计也无需等待 C1。缺少可核验规格的型号保持候选状态。

| ID | 独立交付物与验收标准 | 硬依赖／联合验收点 |
|---|---|---|
| R15 | 可扩展设备契约与能力矩阵：SKU、来源、dense/sparse、精度、标称容量、带宽单位；不支持精度明确拒绝 | 可独立；现有设备回归不得无解释变化 |
| R16 | 按设备提交规格快照与解析档：一设备一来源清单、一组手算/边界测试 | R15 接口；官方规格。无实测不阻塞解析档，但保持 unmeasured |
| R17 | 拓扑与通信：PCIe、NVLink/NVSwitch、RDMA及对应厂商互联分别登记；方向、跨节点、collective 配方和假设明确 | 拓扑契约可独立；设备绑定接 R16，不能把 C2C 泛用为 GPU 间 NVLink |
| R18 | 按设备/后端/精度接入算子及通信 profile：来源、留出误差、覆盖域和回退 | R16 的设备身份；通信 profile 还需 R17；沿用校准契约，不等待所有重点模型完成 |
| R19 | 异构 target/draft 部署组合：独立设备、TP、链路与容量选择；兼容性门禁 | 对应 R16 + R17；可部署性接 A 组；实测性能声明需 R18 与匹配验证 |

设备能力至少分为“规格登记 / 解析可用 / 局部校准 / 匹配场景 E2E 验证”，逐层显示证据与限制。不得把“设备出现在下拉框”当作全部支持；无实测可分配预算仍保持 unknown。R19 不自动认可跨后端 DFlash 服务集成。

## D 组：验证与数据（P1，可先行准备，结论按场景验收）

| ID | 独立交付物与验收标准 | 硬依赖／联合验收点 |
|---|---|---|
| R06 | 按模型/模块拆分的小尺寸 forward 对照；独立 oracle；区分源码阅读与 CPU 执行。**部分完成（2026-11-12，Qwen3.6）**：GDN+MoE+full-attn fixture 的几何/算量对账（shape-checked）及独立张量数值 oracle（numeric-checked）已做；prefill/缓存 decode 各 12 张量逐元素核对，最大差 2.98e−7；修正 attn_output_gate 漏计；tests/test_qwen36_forward_r06.py 12 项；[报告](docs/validation/qwen36-gdn-moe-cpu-fixture-r06.md)。**未做**：mrope 位置生成、V4.1 侧（MLA/CSA2/Engram 执行对照）、其他模块、真实 checkpoint、模型级完整 forward | 对应模块源码；现有完整模块可立即验证，不等待 R04/R05 全部完成 |
| R07a | **已完成 2026-11-12**：vLLM @dc74b8cb（13 文件）+ SGLang @74cb973d（12 文件）双栈逐行核对（SHA256 pin+测试在册）；调度/hidden/KV 回退/拒绝采样/通信字节/计时边界逐项对账；修正 ctx 投影跨度基、草稿生效块与 full 层非因果（独立 oracle+负例）；colocated=supported(source-checked)、dt 模式=unsupported（无栈对应）。残留（DraftLogitsCache 账本/SGLang 采样可用性/GDN 态回退成本/served 实测）转 G-C3 残留与 R07b。[报告](docs/validation/dflash-serving-audit-20261112.md) | 固定服务栈源码；独立于新设备接入 |
| R23 | **已完成 2026-11-13**：Qwen3.6 原生 MTP forward 证据与仿真映射——vLLM @10a854ed `qwen3_5_mtp` 等 8 文件 + SGLang @1b2870d 1 文件双栈逐行核对（SHA256/blob sha1 pin）；查明 mtp.* checkpoint 张量/加载重映射/colocated embed-lm_head 物理共享/每轮 K 均匀步+层 j%n 循环；count（fc+双预归一+full-attn MoE 层+final norm+共享 lm_head+草稿 KV+回退行）/methods（撤销 K≤n 旧限制）/pipeline（逐步阶段+草稿通信+dt 拒绝）/memory（MTP 常驻+草稿 KV+临时）落地；fixture 级独立数学对照（双实现互核 4.97e-13 + 30 项账本对账）；25 项测试 + 前后 replay（M01 64.55→66.96ms、steady +141MB，target/verify 冻结）。剩余：kernel/CPU/GPU 未执行、多模块/量化草稿头/采样档变体、草稿 prefill KV 同步、接受率假设（见 G-C15）。[报告](docs/validation/qwen36-mtp-forward-20261113.md) | 双栈源码独立于 V4.1 工作；不依赖 R07b |
| R07b | 接受分布、draft/verify trace 的采集 schema、导入器与真实样本验收 | schema/导入器可先做；真实数据准入需 R07a 与匹配环境 |
| R08 | 重点模型 GDN/MoE/attention 按算子/设备/精度/TP 扩展覆盖，独立留出与域外回退 | 对应算子语义及设备身份；MoE 分片域依赖 R01，无需等待其他算子 |
| R09 | 按固定场景做 E2E 对照，披露误差与未校准部分 | 匹配实测记录与已实现的场景语义；无需等待全部算子校准，残差可反向指导 R08 |
| R10a | 修复测试子进程编码警告 | 独立维护项，可立即完成 |
| R10b | 工作台重点场景交互、错误提示、窄屏、导出一致性验证 | 现有功能可立即验证；新功能各自追加，不等待其他组完成 |

R07b/R08/R09 的真实数据验收需要额外数据，不以历史 H200 数据或合成数据替代。采集字段与阻塞详见 [数据缺口](docs/gaps/data-gaps.md)。本 roadmap 不授权启动真实模型/GPU 采集。

## E 组：请求级系统实验（P2，按运行时依赖分组）

| ID | 独立交付物与验收标准 | 硬依赖／联合验收点 |
|---|---|---|
| R11 | lookahead、取消、回滚、浪费与资源账；人工轨迹可独立验证 | 已有事件核、方法状态协议；真实资源/性能结论再接对应容量与成本模型 |
| R12a | 请求到达、动态组批、调度事件与确定性重放 | 已有事件核；可用人工负载独立验收 |
| R12b | KV 分页/回收/抢占、生命周期与状态守恒 | R12a 请求生命周期与容量对象接口；不依赖 V4.1 完整语义 |
| R13a | Chunked Prefill：chunk 预算、调度、请求完成口径 | R12a + prefill 成本/状态接口 |
| R13b | PD 分离：KV 交接、传输依赖、两端资源账 | R12a/b + R17 拓扑/传输接口；不依赖 R13a/c |
| R13c | Prefix Cache：命中、引用计数、驱逐与容量守恒 | R12b；不依赖 PD 或 Chunked Prefill |
| R14a | TTFT/ITL/吞吐/goodput 与 SLO 的人工轨迹 oracle | 指标契约，可独立先行，不等待服务特性全部实现 |
| R14b | 请求级 E2E 联合验收 | R14a + 被测场景实际使用的 R11–R13 子项 + 匹配实测轨迹 |

不存在“全部 P0 完成后才允许写系统模块”的总门槛；但未完成模型语义、容量或成本验证时，系统实验只能声明其已覆盖的范围。不得将流水周期当作单请求延迟。

## 合入要求与状态入口

每项完成必须附源码版本/哈希、影响域、测试命令和结果、数值变化、剩余限制。校准不改变原始计数；缺失证据继续传播 unknown/unsupported，不为参与排名补造数值。

- [强制合入准则](docs/MERGE_CRITERIA.md)
- [功能状态与限制](docs/FOCUS_STATUS.md)
- [实施状态](docs/IMPLEMENTATION_STATUS.md)
- [工程缺口](docs/gaps/capability-gaps.md)

优先级以本 roadmap 为准：旧缺口文档中“继续扩展 235B”等建议不再是产品优先事项。更新完成状态时同步这些入口，避免把“已有测试”误写成“已完成部署验证”。
