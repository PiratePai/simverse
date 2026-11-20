# 昇腾验证计划

更新于 2026-11-20。新增设备与采数优先 910B3。后续 910C、950DT 和 950PR 按 SKU 单独登记，H100/H200/L40 保留现有回归。910B3、910C、A2/A3 系统和 950 的能力分别验收，CUDA SM 数与 NPU 核数分开登记。工程优先级见 [待开发计划](../ROADMAP.md)。

## 1. 已有证据与交付顺序

910B3 已提供继承旧仓假设的 BF16 解析档，工作台可选择目标和 DT 草稿设备。V4.1 和未支持的量化路径会被拒绝。R28 与 R29 为局部归档，R29 留有 TP1、物理卡 5 的少层随机权重 fixture 产 token 记录。真实 checkpoint、实际后端数值、TP2/EP2 集成与容量未验收。AllGather 的规范化分子、逐头 MRoPE、来源完整性和容量解释仍需修正。原始归档与派生审查同时保留。见 [910B3 解析档](validation/ascend-910b3-device.md)、[R28](validation/ascend-910b3-20261117.md) 和 [R29 审查](validation/ascend-910b3-r29-review.md)。

解析、条件记录和归档 profile 已共用计数与成本选用路径。工作台默认展示成本组，按原始账本保留细粒度计数。R28 GEMM 和 R29 通信沿用各自的条件对照档。R30 [PR #26](https://github.com/PiratePai/simverse/pull/26) 仍开放。分支重建报告记录六个 run 的 42 条消费文件哈希及 GroupedMoE/KV-write 剩余量。原件在仓库外，`--check` 校验派生产物。正式成本需要补运行身份和组归属。见 [成本组简化说明](validation/cost-group-simplification-20261118.md)。

| 优先级 / 任务 | 具体交付 | 验收 |
| --- | --- | --- |
| P0 / A-N1a 采集整改 | 新 run ID 和目录，拒绝覆盖。归档 runtime commit/dirty diff、本地 warmup 补丁、CANN/驱动版本、拓扑、输入、脚本哈希、完整命令、退出码及日志。修复 ACL 错误码、collective mismatch、KV/路由失败的汇总。 | 离线拒绝返回码非整数或非零、shape 不符、缺 rank 和缺必需元数据。manifest 覆盖实际脚本与原始文件，提交后逐字节校验。R29 的 hash 差异继续单列。 |
| P0 / A-N5a R30 来源审计 | 复核六个 run 的 kernel_details/op_statistic/step/analysis 哈希及原件可取得性。补 batch/M、阶段、角色、权重 revision 和 warmup/step 边界，按 [#28](https://github.com/PiratePai/simverse/issues/28) 恢复可查的身份信息。 | 离线审计可独立交付。缺源、缺归属和未分配时间显式记录，GroupedMoE/KV-write 继续计入未分配项。原始包与现有摘要单独保存，源链未闭环时保持 audit-only。 |
| P1 / A-N1b 逐头 MRoPE | 按头 reshape 后旋转。Q/K 的每个 head、位置与频率带分别核对，identity 对照独立 T 行，NeoX 带用 `j/j+32`。先核对融合接口和 vLLM 调用链，再定位 161002。 | 非零位置逐 head 检查 abs/rel/finite，错误布局和 section 必须拒绝。保留融合与非融合的证据边界。可复用已 pin 的 R26 CPU oracle。 |
| P1 / A-N2a 后端数值与状态（R06/R08） | 固定 vLLM、vLLM-Ascend、CANN、torch_npu 与 runtime patches，沿用成功日志中权重 0.32 GiB 的小 fixture，显式限制 KV budget。先单卡，再分别验收 TP2 与 EP2。V4.1 继续核对可执行路径、量化和配方。 | 对照 GDN/full-attn/MoE 中间张量，至少 16 步 cached decode vs 重算，检查 chunk 边界与拒绝后状态回退。EP2 验证实际 NPU SwiGLU 专家核、空/偏斜专家、重复/漏 token 与逆置换。完整 checkpoint 与随机少层 fixture 分别登记，域外路径在预检拒绝。 |
| P1 / A-N3a 同栈容量与通信（R02b/R03/G-D7/R17） | 同进程记录 init/load/KV/warmup/capture/replay/release。graph 加 clone-only 对照及输出检查。AllGather 统一逻辑字节分子，保留联合批 ID、两 rank 最慢窗口与真实 transport。 | weights/KV/runtime/HCCL/graph/allocator 与 free 分列，实际 serving 栈按 rank 验证已知峰值与未知项。旧 100 项 exact 观测按原用途保存，按缺失载荷和独立重复补点。小实例容量只支持匹配域内判断。 |
| P1 / A-N4 设备与实现（R15/R16） | [#27](https://github.com/PiratePai/simverse/issues/27) 固定实际 board/SKU 和软件栈指纹，核验官方 dense 规格、单位、精度与 Ascend 专用量化和 lowering 配方。 | 有来源证据和不支持精度的负例。继承解析档可继续使用，正式成本依赖匹配身份。离线组归属审计可并行。 |
| P1 / A-N5b 成本组（R08/R18） | [#29](https://github.com/PiratePai/simverse/issues/29) 核对投影融合组的总时长、成员集合及 split/norm/rope 归属。[#30](https://github.com/PiratePai/simverse/issues/30) 核对实际 MoE permute/dispatch、GDN、数据搬运及 graph/host/sync 的边界。按解析回退权重选补采点。 | 组域匹配 dtype、prefill/decode/verify、目标或草稿角色。每段时间只记一次，事件区间已含的间隙不再加计，无法归属的时长保留 unknown。融合总时长不按宽度或调用数分摊。正式校准还需按 shape 去重留出、域外回退，原始计数守恒。 |
| P1 / A-N6 匹配 E2E（R07b/R09） | 复用 [端到端 trace 验收方案](validation/r07b-r09-trace-acceptance.md) 的字段和离线样本。上机前固定真实权重、精度、后端、容量和服务支持，分别采 baseline、DFlash 与 MTP。 | 按事先冻结的场景阈值核对 TTFT、ITL、吞吐、每 rank 峰值与未解释残差。缺方法配对、请求轨迹或时钟归因时保留缺项。 |
| P2 / 历史归档 | [#31](https://github.com/PiratePai/simverse/issues/31) 的三模型包用于离线 parser 和回放回归。 | 历史 profile 按原模型、实现和采集域绑定。 |

`tools/ascend_forward_probe.py` 已提供 Qwen3.6 fixture、本地权重入口和离线预检。V4.1 当前只提供带阻断原因的部署计划。运行需明确设备与窗口，资源有限时按下节拆分验证。

## 2. 无法全量跑时，分别能测什么

### 按解析回退权重补采

按 [七场景回放](evidence/ascend-record-costs/coverage-plan.json) 与当前 run 的 `cost.record_cost.measurement_plan` 排序。以下只调整采测计划，没有授权或启动硬件执行；解析回退权重是缺口定位依据，不当成实测瓶颈。

| 顺序 / 原因 | 下一批具体交付 | 资源与验收 |
|---|---|---|
| P0 / 来源与统计先行 | 固定且核验目标栈、CANN/driver/runtime commits、dirty 补丁、卡对/拓扑、原语/布局、采集命令与唯一 run；保存每批原样窗口、输出正确性与脚本哈希。R29 现有栈若可保留则补齐身份；若更换栈，新建该栈记录，不拼 R28/R29 | 无卡可准备 schema/拒错与独立 oracle；数值正确后至少三次独立进程、每进程≥五批，按 shape 分组留出；不能用摘要或曲线内部诊断冒充独立留出 |
| P0 / `primitive-unresolved` | 核对 Ascend logits 的 gather-to-root/allgather 分支；TP2/EP2 的 dispatch/combine 固定 variable split、元数据、空/偏斜专家、逆置换。当前 EP2 有 80 个 dispatch/combine 实例不能用 equal-split alltoall 记录替代 | 现有双卡；先原语/数值再延迟，保留两 rank 联合批 ID 与最慢窗口；logits 实际载荷样例按当前 run 的字段补点，不按 wire bytes 猜 input |
| P1 / `shape-missing`、验证 M16 | 先补 TP2 真实分片 GEMM：M=1/16，K=2048，N=4096/2048/32/256；对应 out/o `[M,2048]@[2048,2048]`，lm_head N=124160。已有 R28 单点保留作独立条件对照；若要与 R29 通信组合，在统一栈补采所需算子，不混用旧栈 | 单卡可测真实局部形状，双卡再查并发/通信影响；decode 与 verify 分列。TP8-shaped 的 N=1024/512/8、lm_head N=31040 也可单卡局部测，但不能称 TP8 部署实测；优先补 M16 缺点，不重复完整 M 网格 |
| P1 / `operator-not-equivalent`、`operator-set-missing` | 在统一栈采实际 MoE（gate/up/down+clamp/路由所有权）、GDN recurrence/conv/state 和 full-attn/cache 路径；两 GEMM MoE、稠密 SDPA 或组合 norm 不顶替实际融合核。首轮按当前回退清单的阶段与大额项拆小交付 | 1–2 卡、模块/少层；保留真实维度与中间张量 oracle；明确融合组拥有的账本行，原始计数不变。成本可局部验收，不等完整模型 |
| P1 / `model-role-mismatch` | DFlash 草稿独立模型/形状/实现身份，先确认 Ascend 集成路径，再采草稿投影、attention、KV/hidden、验收与回退；不继承目标模型样本 | 单卡局部张量与双卡通信可分开；未实现的服务集成继续 unsupported/unknown，独立算子探针不能升级为 served 通过 |
| P1 / 未计价运行时、`point-to-point-uncovered` | 接受/拒绝恢复、hidden 提取等未计价阶段先确认实现，再独立计时；DT 的 token/hidden/KV 逐方向点到点往返另采，登记 payload、同步与所有权 | 未计价时间保持 null 且不进入覆盖分母；点到点链路保留解析回退，不拿集合通信样本顶替 |
| P2 / `rank-domain`、域外载荷 | world=2 内只补当前缺失载荷与独立重复；TP4/8 需要更多卡或匹配外部 trace 后单独验收，不从双卡曲线外推 | 不阻塞现有双卡闭环；R29 TP8 零命中如实保留；范围、数据身份、插值/外推拒绝都有负例 |

每次新 run 的覆盖/回退清单随 JSON 导出；按原因、去重形状、阶段与现有成本权重更新本表。R29 通信已有的 50 点、R28 已核对的 14 个独立 GEMM 形状不为“再验一次”盲目全量重采；只有栈、实施配方或统计验收目标变化才建立新样本，保留旧原始记录。统一栈数据满足条件后才能从 conditional-only 提升到正式局部校准，部署资格另验。

| 可用资源 | 测试与建议网格 | 判定与产物 | 不覆盖 |
|---|---|---|---|
| 无 NPU / CPU | config 全叶键、源码分支、shape/次数/FLOPs、量化打包、TP/EP 守恒；独立小张量 oracle；坏 trace/跨模型/跨 dtype/未知计时边界负例 | 数值与结构 fixture；失败原因及源码哈希；parser/接线回归 | NPU 数值、kernel 时间、容量或可部署性 |
| 1 卡，无完整权重 | 保留真实 head_dim/hidden/专家维数的单层或少层随机权重；Qwen GDN+full attention+MoE，V4.1 分别测 mHC/Sinkhorn、CSA2 select/mask、clamp/router、Engram lookup；先标准算子再实际后端融合核 | FP32 oracle vs FP32/BF16/目标量化，逐中间张量 abs/rel/NaN 检查；融合与非融合一致性；登记未跑分支 | 缩小专家/层数改变路由与访存分布；不能得到真实 perplexity、接受率或整模性能 |
| 1 卡，状态测试 | prefill 长度 1/16/128/512，decode 连续至少 16 步；prefix/chunk 边界；GDN/conv/KV cache 增长与回退；MRoPE pos=0/1/127/1000/32767，使用本仓旋转参数与分段 | cached decode vs 重算；chunk vs 不分块；拒绝后 cache/state 恢复；MTP 多步及接受 0/部分/全长 | 随机 token 的一致性不证明真实 DFlash 接受律；CPU/GPU kernel 不能互认 |
| 1 卡，成本/容量 | GEMM M=1/8/64/256/1024、实际 K/N；attention S=128/2K/8K/32K 中可容纳点，Q=1/verify block/prefill；MoE 专家倾斜、热/冷缓存；内存对象分阶段 | warmup 后每批样本、重复进程统计；reserve/allocated/free/peak 与时间线；在可用显存内分块加载层或参数分片 | 单层时间乘层数只能给显式模型估计，不能替代跨层 cache、通信重叠、allocator 峰值 |
| 2 卡 | TP1 对 TP2 可重组数值；EP2 dispatch/combine token 数、逆置换、重复/空专家、倾斜/丢 token；五类 collective 1 KiB–256 MiB（按内存限额），两 rank 正确性与延迟；若机器允许测同域/跨域卡对 | 每 rank 原始输入/输出 shape、dtype、send/recv/total 字节、collective 次数、同步窗、最慢 rank；核验 HCCS/RoCE 实际路径 | R28 卡6/7、R29 卡5/6分别仅代表各自卡对与栈；不能推 TP8/EP8、全互联饱和、跨机带宽或负载均衡 |
| 4 卡（可获得时） | TP1/2/4、EP2/4 合法域，2×2 子组与全组；卡对并发、通信/计算重叠、straggler、峰值 rank；真实层尺寸在 TP4 的分片单卡可先测 | 分片数值/通信量/状态守恒；多组同时运行的 trace 与非重叠对照；拓扑身份清楚 | 仍不是 8 卡/多机验证；GEMM 单卡 TP8-shaped 分片可测局部形状，集合通信不外推 |
| 权重分片或外部已有 trace | 单层/模块加载，或离线导入相同版本的完整 profiling；逐算子/阶段对账 | 明确哪些是执行、哪些是他人实测数据复核、哪些是解析；保留来源及缺失字段 | CPU offload/逐层搬运改变数据路径；除非目标场景本就如此，否则不报整模吞吐 |

所有网格按合法形状与可用内存裁剪，记录 skipped 及原因，不静默丢点。
两模型至少包含以下负例：不整除 TP/EP、超上下文、空/全屏蔽候选、tail/padding、
NaN/inf/全下溢、未知 dtype、缺 cache 元数据、rank 不齐。不能把“有几个算子可测”
写成“完整 forward 已通过”；遇到不支持的 fused kernel，语义参考路径和性能路径分开。

## 3. 有 profiling 时如何校对

### 3.1 先固定输入与采样身份

每个 run 保存：硬件 SKU/物理卡与逻辑 rank/互联拓扑、驱动/CANN/torch/torch_npu、
vLLM/vLLM-Ascend commits、模型 repo/revision/config/checkpoint 哈希清单、量化转换器、
实际 weight/act/KV dtype 和 layout、TP/EP/PP/DP、batch/长度/阶段/seed、eager 或 graph、
图捕获大小、采集命令与开关、warmup/active 区间、重复次数、同步策略、原始样本 SHA256。
只有 config hash 不能代表权重身份。先跑正确性，再跑不带 profiler 的基线，最后独立
采 CPU+NPU trace；重复对照 profiler on/off，披露扰动，不能拿带重型采集的墙钟作正常性能。

优先保存 trace_view、kernel/task 明细、operator 调用/shape、memory 时间线、各 rank
communication 明细以及执行输出；实际文件名随固定工具版本确认。只给截图或按 op 名聚合
的 CSV 时，可做热点排查，**不可**还原顺序、融合所有权、重叠、全局关键路径或纯 launch。
缺文件保留 unknown，不自动补零或虚构 stack/version。

### 3.2 建立 forward → 账本 → kernel 映射

映射键至少包含 model/layer/stage/op/shape/dtype/TP/EP/implementation_id，使用
record_function 或后端已有标记、correlation ID、stream、rank、step 对齐具体调用。
单逻辑算子可能拆为多核，多行可能融合成一核；建显式 ownership 表，每个设备任务只能
收费一次，拷贝/重排/量化/归一/host 查表与通信不能漏记。代理与真实 kernel 分开。
无 shape/调用关联时记 unmapped，不能按名字相似就匹配。

先核对层数/调用次数/张量形状/依赖与传输字节，再比较时间。Work/Traffic/Memory 是
模型原账本；profiling FLOPs/显存计数器的定义可能不同。没有 HBM 计数器时，不能用
逻辑张量字节宣称测得物理 HBM 流量；HBM/缓存复用只能列为待验证假设。

### 3.3 分边界对账，解释残差

| 层次 | 实测与仿真比较 | 处理规则 |
|---|---|---|
| kernel | 完全相同 shape/dtype/实现的 task duration 分布 vs 对应行或融合组成本 | 区分 device-kernel、host-launch、graph-piece、E2E。批内墙钟暂作独立观测边界，不硬塞进 device-kernel；契约扩展须独立门禁 |
| host / gap | 同一次调用关联的 CPU enqueue、device start/end、依赖等待与 idle | queue delay 不等于 launch；禁止独立 run 的两个 median 相减。无法解释部分列 residual |
| communication | 每 rank 的 transit/wait/sync/idle、消息尺寸、collective 类型、最慢 rank | algbw 与规范化 busbw 分开，按 collective 定义；不能把所有 wait 当链路慢；跨 rank 时钟未校准不得直接拼全局时间 |
| 单步/阶段 | prefill/decode/draft/verify 的起止窗、stream interval union、依赖 critical path | kernel 总和不等于 step wall time；通信和计算重叠用区间/依赖去重，不能简单相加后再套统一 overlap 比例 |
| 容量 | init、load、KV、warmup、capture、replay、steady、release 的 allocated/reserved/free/peak | 对象可见峰值与设备 free 差分开；同生命周期重叠取真实峰值，非共存对象不简单求和；预算扣减与 overhead 避免重复 |
| E2E | 服务端 TTFT、ITL、吞吐、goodput、尾延迟和峰值内存 | 按 [端到端 trace 验收方案](validation/r07b-r09-trace-acceptance.md) 核对窗口、请求状态、输出序号及 token 或分块粒度。局部 R28/R29 成本和 R30 离线审计只用于各自证据范围。 |

输出逐行 `analytic_base / observed / selected / boundary / ownership / error /
coverage / fallback_reason / evidence_ref`，以及按模型、阶段、算子、rank 的差异表。
报告中区分数值相等、观测匹配、近似假设、未解释残差。归因顺序是身份/形状 → 次数/
融合 → dtype/layout → 访存 → 通信/同步 → host/runtime；不先调全局倍率。

### 3.4 局部校准的验收

1. 先精确键查表；有足够覆盖后才在受限形状域插值。backend、device、dtype、layout、
   TP/EP、图模式变更即域外。保留解析值及回退原因；NPU 不伪造 cuda_version。
2. 按完整记录键/shape 分组划分训练与留出，重复 run 和同形状不能泄漏两侧；每条保存
   count/min/max/mean/median/std 与原始样本，样本不足不宣称稳定或独立精度验收。
3. 双报可映射行比例、时间覆盖（时间权重来自实测还是估计要注明）、中位/p90/max
   相对误差与绝对 µs 误差。接近零的参考耗时另列，不以除零或极大百分比替代解释。
4. 初始实验目标（不是已测结果）：三次独立进程、每次至少五个有效批窗口；稳定性看
   批间/进程间分布，不以固定 warmup 次数保证稳定。局部成本留出中位 ≤10%、p90≤20%
   是待实测确认的目标；超标保留报告、继续 analytic 回退，不修改阈值使其过关。
   数值正确性先按 dtype/算子约定容差，整数/索引/守恒必须精确；E2E 阈值另按场景冻结。
5. 开/关校准重放确保 Work/Traffic/Memory 逐字段不变；成本所覆盖的部署域有预算与
   数值证据后再讨论排名。原 PR21 的代理、冲突结果和缺元数据摘要不进入训练/留出。

导入后的差异、覆盖与回退信息应能从工作台查看和导出。R30 分支的离线审计界面按 audit-only 验收，NPU 生产校准独立验收。

## 4. 官方实现参考与取舍

2026-11-17 查询远端 HEAD，并阅读下列代码及测试；均为活跃仓库的当时快照。
**只做源码阅读，没有在本机执行这些上游 NPU/CUDA 测试。** 固定提交不保证与采集机
torch_npu 2.9.0.post9 兼容，采集前必须按官方版本矩阵确认并冻结实际运行版本。

| 官方仓库 / 所查 commit | 已读实现与测试 | 借鉴与边界 |
|---|---|---|
| [vllm-ascend](https://github.com/vllm-project/vllm-ascend/tree/915d58873c0854c0550b56f66a126ca2cb114e71) | `vllm_ascend/profiler/torch_npu_profiler.py`；`tests/ut/profiler/test_torch_npu_profiler.py` | CPU+NPU activities、worker/rank 标记、内存/stack 独立开关；已有 stack 开销提示。复用实际平台入口，不为采数另建推理引擎 |
| [Ascend/pytorch](https://github.com/Ascend/pytorch/tree/b96e70c87d88fa502b54f32131e140e332ae8366) | `torch_npu/profiler/analysis/prof_bean/_op_summary_bean.py`；`test/profiler/analysis/prof_bean/test_op_summary_bean.py` | 保留原字段/时间单位与 task 身份；摘要不是完整调用时间线，不从均值猜依赖 |
| [msprof-analyze](https://github.com/Ascend/msprof-analyze/tree/f1bc9e3068693b2aa7db42f30bdc3d563545536a) | `msprof_analyze/cluster_analyse/analysis/communication_analysis.py`；`test/ut/cluster_analyse/analysis/test_communication_analysis.py` | 按通信组/rank 分开 transit、wait、sync、idle，缺值保持 invalid；仅借鉴归因和导出，不增加本仓运行依赖 |
| [nccl-tests](https://github.com/NVIDIA/nccl-tests/tree/afd59abab774a6a4b492eb344bd0c845a0b7bb4d) | `src/all_reduce.cu` 的 `AllReduceGetBw`（性能测试实现） | 只采用显式规范化 busbw 定义并注明来源；不把 NCCL 算法或 NVIDIA 拓扑假定成 HCCL 实现 |

forward 注册/调用链的固定源码、配置边界与执行入口单独记录于
[昇腾 forward 路径](validation/ascend-forward-paths-20261117.md)，防止把 profiler 支持
误认为某一模型/量化/设备组合已支持。
