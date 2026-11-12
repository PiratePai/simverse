# 重点模型与功能状态（2026-11-10；2026-11-11 R04、2026-11-12 R06 增补）

2026-11-12 R06（Qwen3.6 GDN+MoE 小尺寸 CPU forward 对照）：fixture 级
形状与算量对照已执行（探针独立跑 pin 官方 forward，h=32/2 层/4E 随机权重，
prefill+decode 逐 GEMM/conv/router/因果对/无放回 assignment 与本仓
build_model_work(tp=1) **精确相等**；tests/test_qwen36_forward_r06.py
10 项；证据 ev-qwen36-cpu-fixture-r06）。顺带修正 full-attn 输出门控
逐元素漏计（attn_output_gate 行；默认场景 decode +10,240 FLOPs/层组、
M01 +0.00002ms）。**模型级 cpu_forward 仍 not-run**——这是 fixture 级
证据，不构成完整 forward 或部署验证。详见
[validation/qwen36-gdn-moe-cpu-fixture-r06.md](validation/qwen36-gdn-moe-cpu-fixture-r06.md)。

2026-11-12 R06 续验：独立 PyTorch 张量参考已逐元素核对随机权重 fixture
的 prefill/带缓存 decode，每相位 12 个输出与缓存张量；最大绝对差
2.98e−7（容限 atol=1e−5、rtol=1e−4）。新增
`ev-qwen36-cpu-numeric-r06`（numeric-checked），原算量/形状证据保持
shape-checked；模型级 CPU forward、mrope 位置生成、真实 checkpoint 与
GPU 部署仍未验证。详见 [R06 报告 §9](validation/qwen36-gdn-moe-cpu-fixture-r06.md)。

2026-11-12 R05（V4.1 mHC/Sinkhorn 修正）：14 文件 blob SHA1 对 tree、SHA256
pin 后补齐 sqrsum/prenorm、comb post、首层无 post、出口两行；Sinkhorn gates+
m×m 归一由2400修为1400 op/token（算术 op 公式精确，exp/sigmoid/rsqrt=1，
非指令数），norm 4h+2 仍为近似惯例。首层 broadcast 派生常驻缓冲491,520B单列。
默认 replay op +14,657,520、known_peak +491,520B；Engram 原行/公式冻结。
新增16测试，全量252通过；证据 ev-vllm-v41-mhc 仅 source-checked，kernel/CPU/GPU
未执行，部署变体/实际 split/物理 HBM 未验证，structure_complete 仍 False。
[R05 报告（含与任务书出入）](validation/v41-mhc-kernels-20261112.md)。

2026-11-11 R04（V4.1 Engram 建模）：common/engram.py pin（vllm @f358f057，
blob sha1 ba4ceea6 对 vllm-tree.json 校验）后单副本路径 Engram 落地——
engram 8 键 → mapped；count 新增 engram_hash/lookup/wkv/post_wkv 四行
（层 1/14，批级哈希一次）；memory 新增表/wkv/qk 对象 + staged 下界
（表默认 **cpu_offload 研究假设**，场景显式 engram_table_placement 输入，
host 驻留不进 GPU 峰值）；TP>1 补 head/wkv 两次 all-gather。独立 oracle
tests/test_v41_engram.py 24 项（素数桶和 == engram_num_embeddings 精确
相等、TP1 表 202,758,032,400 B、tp=7/9/16 拒绝）。默认 V4.1 replay：
GPU 侧解析耗时下界 113.34→116.15ms（host 查表时延未知）、GPU steady +39.5MB
（表在 host 25.3GB 另列）；
structure_complete 仍 False（CSA2/MoE/量化缺口不变）。详见
[validation/v41-engram-20261111.md](validation/v41-engram-20261111.md)。

2026-11-10 两模型门禁轮：活跃目录只保留 Qwen3.6-35B-A3B + DFlash、
DeepSeek-V4.1-Flash；历史模型/tiny 移出运行时选择。修复 V4.1 o_a 分组与
memory 单位、Qwen3.6 shared gate、草稿 B>1 计数、证据引用/完整 config 覆盖/
source_map 行号；建立 G8–G17 门禁与 PR/push/merge_group 工作流。详见
[两模型审查记录](validation/two-model-forward-merge-gate-20261110.md)。
本轮**没有** CPU/GPU 执行证据；两款主模型都不能进入实际部署排名。

早前投机耗时复核（MTP/chain verify 通信、DFlash 草稿通信）保留为历史记录：
[投机耗时与排名复核](validation/speculative-ranking-audit-20261110.md)。
MTP 首层模板、chain 合成草稿仍是研究估计；MTP 现标 unverified。

2026-11-10 A 组轮（R01–R03b）：TP/EP per-rank 统一配方 + 同卡对象身份
账本 + 生命周期峰值 + 预算契约与四态判定全部落地（详见
[validation/agroup-tp-ep-colocated-20261110.md](validation/agroup-tp-ep-colocated-20261110.md)）。
同日**二轮审计**（[报告](validation/agroup-lmhead-ctxkv-audit-20261110.md)）：
逐文件复核 pin 源码后修复 4 处漏计——lm_head logits 集合通信
（logits_gather，decode/verify/草稿各 1 次）、DFlash 草稿 lm_head/ctx KV
二次读流量、vLLM `_fused_kv_weight` 驻留副本入账本；判定链路不变。
**同日三轮修正**：ctx-KV 路径显式化（`fused_raw`/`mxfp8_context_proj`/
`per_layer_fallback`），4bit 回退按完整 qkv（含 Q）计 FLOPs/权重读；
k_norm 堆叠按草稿 config dtype（bfloat16→2B/elem，1,536 B）计，
与 dense_w 4/8/16 无关；dense_w 不再用于选分支。

## 已实现，但不等于真实部署已验证

| 能力 | 范围与限制 |
|---|---|
| 身份、ModelIR、配置覆盖与逐算子账本 | Qwen3.6 config 全覆盖、主干关键分支 source-checked；V4.1 Engram 单副本路径 + hc 已 source-checked 建模（R04；DP 交换/cpu_offload 运行时未覆盖），CSA2/MoE/量化仍缺口；四栏证据见 JSON |
| Qwen3.6+DFlash | pinned 草稿、接受律、draft/verify 解析成本、同卡与 DT 串行；非实测收益；CPU 仅 R06 fixture 级对照（模型级未执行），GPU 未执行 |
| **per-rank TP/EP 配方（A组 R01）** | vLLM @dc74b8cb pin（FusedMoE inter/tp 或 E/ep 整专家、router 复制、KV 头切/复制、GDN 切分、词表行切、DFlash 同 TP 组共享 embed/lm_head、lm_head logits_gather）；ep∉{1,tp} 显式拒绝 |
| **同卡对象账本（R02a/b）** | 物理 allocation 身份 + consumer 引用（草稿共享 target embed/lm_head 零额外驻留；vLLM fused ctx-KV 副本入账）；生命周期分列（常驻/每请求/跨轮/阶段临时）+ 峰值取阶段 max；workspace/graph/NCCL 池为缺口组件传播（不填 0） |
| **预算契约与判定（R03a/b）** | nominal/allocatable(measured/hypothesis)/额外开销分列；四态判定 unknown/oom/feasible/conditional；已知下界超标称即 OOM；假设预算不进实际部署排名 |
| mHC / Engram | mHC kernel 源已 pin，Sinkhorn 算术公式精确、norm 惯例近似（R05；部署变体/物理执行未覆盖）；Engram 素数桶表/哈希/查表/wkv/门控已建模（R04；表位置 cpu_offload 为研究假设，DP 分片交换未覆盖） |
| DT 流水事件核 | 微批依赖、资源互斥、周期和利用率；非完整在线服务 |
| 接受律与比较 | 显式假设、未知传播、资格门禁；不从平均接受长度推造完整概率律 |
| CLI/Web/导出/扫描 | 共用生产结果、精度选择、EP 选择、预算输入、独立运行身份 |
| H200 校准基础设施 | **历史归档**：30B-tiny/235B 数据绑定已归档模型；跨模型使用直接拒绝，不能覆盖两款主模型 |

## 待完成工程

1. ~~Qwen3.6 优先：统一 MoE per-rank TP/EP 配方~~（2026-11-10 A组 R01 完成）。
2. ~~同卡显存复核（草稿 batch/共享去重/KV 分片/生命周期峰值分列）~~（A组 R02a/b 完成；运行时池尺寸证据仍缺，见下）。
3. 可分配预算：契约已就绪（R03a），**真实 measured 数据仍缺**（G-D7）；解除前默认场景保持 unknown。
4. ~~V4.1：获取并固定 Engram 实现、补查表/投影/参数/容量/跨层依赖~~（单副本路径 2026-11-11 R04 完成；剩余 DP 分片交换/MXFP8 批路径/amd 变体与 CSA2/MoE scoring/量化缺口，完整前不解除结构门禁）。
5. 请求级 lookahead、取消与浪费；在线到达/组批/抢占、KV 分页、Chunked Prefill/PD/Prefix Cache。
6. ~~235B moe.csv TP8/EP1 准入~~：模型已移出活跃 catalog，只保留历史归档研究；
   不再作为产品优先项。两款主模型缺匹配 profile，需新采集。

## 待验证或需额外数据

- 重点模型小尺寸 CPU forward 与计数的独立执行对照——Qwen3.6 的 GDN+MoE
  fixture 级形状与算量对照已执行（R06，2026-11-12）；V4.1 侧与其余模块、
  真实 checkpoint 级对照仍未执行。
- Qwen3.6+DFlash 实际接受分布、draft/verify 延迟及服务栈配对验证。
- 重点模型各精度/TP 的 GDN、MoE、attention 测量；历史 H200 profile 不能按名称相似迁移。
- V4.1 Engram DP 分片交换路径与 mHC tilelang kernel 数值/操作边界（单机 Engram 已 source-checked）。
- 实际可分配显存（G-D7）、非 KV 运行时开销（workspace/CUDA graph/NCCL 池尺寸，当前按缺口传播）、匹配 E2E 请求轨迹。
- 浏览器交互与像素复核：2026-11-10 A组轮已实测 EP 选择/预算输入/运行链路
  （截图 docs/evidence/workbench_ep_budget_run.png）；其余页面沿此前口径。

## 历史轮次修复（2026-11-10 A 组；数值不自动继承）

- **TP/EP 配方统一（R01/G-C11 解除）**：count 的 MoE 行、memory 分配、
  EP 通信三方改用同一 ParallelPlan（vLLM pin）；EP 显式场景字段（影响
  运行身份）；非法组合显式拒绝并列原因。
- **同卡账本修正（R02a）**：废除草稿幻影 embed/lm_head 全量副本（≈1.017GB
  错误驻留）与去重键失配双重计费；embed/lm_head 词表行切；GDN 权重/状态、
  草稿 KV per-rank；attention 投影权重补乘输入宽 h（旧式漏乘 h 倍）；
  草稿批=目标批（修复硬编码 1）。
- **计数补齐**：final norm、草稿 q/k norm、rope、hidden_norm 缺行（对
  pinned dflash/HF 源码逐行核对后补）；TP 集合通信从"整模型 2 次"修正为
  "每层 2 次"（旧代码漏计 40×）。
- **判定链路（R03b）**：M06 指标携带 known_peak/steady/峰值完整性；
  假设预算 → feasible-conditional 且不参与排名。
- **二轮审计补漏（同日晚）**：lm_head logits_gather 通信（此前
  draft.py 注释声称"另计"但未实现）、草稿 lm_head/ctx KV 二次读流量、
  `_fused_kv_weight` 融合驻留副本、ctx 归一副本/融合输出/aux hidden 三件
  结构化临时；4 新增独立手算 oracle（见二轮报告 §1/§3）。

审计对照场景（TP8/EP1、B1 ctx32k、α0.75、blk5、W8A8+FP4 专家）的
同卡 per-rank 账本为 2,830,652,352 B（steady）+ 752,704 B
（结构化临时峰值）= 2,831,405,056 B（known_peak）。CLI 内置默认
（α0.72、blk16）重放的 known_peak 为 2,832,959,265 B；
这不是用户所有场景的峰值，且不完整
（workspace/graph/NCCL 池未计、非 0），不能据此证明可部署，也未超
80GB 标称不构成 OOM。可分配预算（G-D7）与运行时池证据仍是阻塞项。
