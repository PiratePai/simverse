# 昇腾优先：少卡 forward、测试与 profiling 校对计划

更新：2026-11-17；本文件补充 [ROADMAP](../ROADMAP.md) 的 C/D 组与 R02b/R03。
**新增设备与采数投入先昇腾、后 NVIDIA**。先用已有 910B3 双卡建立可信闭环，再逐
SKU 接 910C/950DT/950PR；后几款没有匹配规格/实现证据时保持候选。H100/H200/L40
已交付功能继续回归；NVIDIA 新卡和新 profile 后排，不作为昇腾工作的前置条件。
910B3、910C、A2/A3 系统和 950 的能力不能相互继承，尤其不能把 CUDA SM 数填作 NPU 核数。

## 1. 已有证据与下一批交付

910B3 可选解析档已先行交付，来源、单位与边界见 [旧仓参数迁移](validation/ascend-910b3-device.md)。目标与 DT 草稿选择器可选，Qwen3.6 使用 BF16；不支持的 V4.1/量化路径明确拒绝。设备可选不等待整模验证；A-N4 仍需补官方 SKU/后端配方，A-N1–3/5–6 的验收条件不变。

PR #21（R28）与 PR #23（R29）均作为局部、未准入采集归档。R29 已补 ACL 返回码/context 和 ReduceScatter list 形状，并记录 TP1 少层 fixture 产出 token；但 AllGather 规范化、逐头 MRoPE、来源完整性、实际后端数值与容量仍有缺口。以 [R29 审查](validation/ascend-910b3-r29-review.md) 与派生 review.json 为准，采集时报告保留作历史材料；不把原 manifest 的「A-N3 未执行」或原文「7/7」当作整体状态。

**下一批按以下可独立验收的小项推进，不重新跑已无问题的全部网格，也不因误读 OOM 先缩小 fixture。** 本轮审查没有执行硬件任务；后续 NPU 执行仍需明确设备与运行窗口。

| 次序 / 子项 | 具体动作 | 验收与边界 |
|---|---|---|
| P0 / A-N1a 来源与采集器 | 新 run ID + 新目录，拒绝覆盖；归档 runtime commit/dirty diff、本地 warmup 补丁、CANN/驱动实际版本、各尝试完整命令/退出码/日志；修复 ACL 非 int/非零返回码、collective mismatch、KV/路由 flag 不进入失败汇总 | 离线坏返回码/坏 shape/掉 rank/metadata 缺失必须失败；manifest 覆盖最终脚本且提交后离线逐字节检查。R29 日志 hash 差异不能靠重新计算原 manifest 消除 |
| P0 / A-N1b MRoPE | 每头 reshape 后旋转；Q/K、head0–末头、位置与频率带逐项独立 oracle；identity 对照独立 T 行，NeoX 带用 j/j+32；先核实融合接口与实际 vLLM 调用链，再定位 161002 | 非零位置对全部 head 逐元素 abs/rel/finite，尾部按 head 比较；已 pin R26 CPU oracle 可先行。错误布局/section 必须拒绝。非融合成功不升格为融合或模型集成通过 |
| P0 / A-N2a 实际后端数值与状态 | 沿用小 fixture（成功日志权重 0.32 GiB），显式限制 KV budget；先单卡，再 TP2。归档每次失败与成功；GDN/full-attn/MoE 中间张量、cached decode vs 重算、chunk 边界与回退；固定 runtime patches 后与独立参考对照 | TP1 产 token只算执行接线；TP2 与 EP2 各自验收。至少 16 decode 步、空/偏斜专家、重复/漏 token、逆置换与实际 NPU 专家核；玩具 CPU GELU 路由不能代替 Qwen SwiGLU/真实分片 |
| P0 / A-N3a 同栈容量与通信补点 | 同一进程记录 init/load/KV/warmup/capture/replay/release；graph 加 clone-only 控制与输出 finite/等价检查；通信修 AllGather 分子、记录统一逻辑字节和联合批 ID、两 rank 最慢窗；补实际 transport | weights/KV/runtime/HCCL/graph/allocator 与 free 分列，峰值不重复相加；free 差不当分配粒度。只补必要点/独立重复，旧 100 项 exact 观测保留，但未证明跨 rank 同步窗口/拓扑 |
| P1 / A-N4→A-N5 | 设备/实现独立登记；并行准备 trace parser、ownership 和差异表；身份与正确性通过的局部算子才接生产成本 | 原始计数不变、计时边界分栏、按 shape 去重留出、域外回退；不等待整模即可做局部审计。R29 本身不作生产 profile |

这些交付未解除 G-D7、R02b、R17、R18，也未证明完整模型在 NPU 执行。R28 原审查仍见 [报告](validation/ascend-910b3-20261117.md)。

| 顺序 / 任务 | 具体交付与依赖 | 通过条件；不能推导的结论 |
|---|---|---|
| P0 / A-N1 采集整改（R28 后续） | 修正 ACL 返回码/context、通信 shape/校验、graph 基线、MRoPE 分段；独立目录保存新 run；补版本、拓扑、输入与脚本哈希 | 原始样本可离线重放，返回码与数值正确；旧数据只归档，不覆盖为“重测成功” |
| P0 / A-N2 昇腾 forward 路径（R06/R08） | 固定 vLLM + vLLM-Ascend + CANN/torch_npu 组合；逐一核对模型注册、config、GDN/attention/MoE/量化/KV 路径；提供预检与显式执行入口 | 少层随机权重与完整 checkpoint 两种身份分开；无匹配 kernel/型号立即拒绝。源码映射/离线接线不冒称 NPU 已运行 |
| P0 / A-N3 1–2 卡正确性与容量（R02b/R03/G-D7） | 先做下表无权重/局部层测试；同一进程分阶段快照，实际 serving 栈 budget + workspace/graph/HCCL | 数值 oracle、分片重组、状态守恒；每 rank 已知峰值及未知项分栏。小样例可分配不代表整模 feasible |
| P1（昇腾第一顺位）/ A-N4 设备与实现档（R15/R16） | 独立 Ascend 设备身份、官方 dense 规格及能力；NPU 核/引擎字段与 CUDA sm_count 分离；平台特定量化/lowering recipe | 来源与不支持精度负例；不得复用 Marlin/FlashMLA SM90 调度器为 Ascend 实现，也不得用实测峰值填 spec |
| P1（昇腾第一顺位）/ A-N5 profiling 校对与局部校准（R17/R18） | 先提供原始 trace 审核、映射与差异表；只有设备/实现/形状域匹配后才接 CalibratedPricer | 行数/时间双覆盖、去重留出、域外回退、原账本不变；可独立交付局部校对，无需等待整模 |
| P1 / A-N6 匹配 E2E（R07b/R09） | 权重、显存、后端集成条件满足后，固定 workload 实测 prefill/decode/verify 与接受长度分布 | 按 TTFT/ITL/吞吐/显存误差验收；没有真实配对和请求轨迹时保持缺口。DFlash/MTP 单独核对 |

本轮 A-N2 已交付 `tools/ascend_forward_probe.py` 的 Qwen3.6 fixture/本地权重入口与离线门禁；R29 有 TP1、物理卡 5 的 fixture 成功记录，来源仍有限，数值正确性、TP2 集成和完整权重身份未验收。V4.1 仅交付明确阻断的部署计划，没有将 V4 或双 910B3 代理冒充整模。

A-N1/A-N2/A-N3 可按模块推进；A-N5 的离线 parser/映射工作可先做，生产成本准入依赖
A-N4 和匹配采数。没有足够显存加载整模时不等待扩容，转入以下少卡路径。
本计划没有启动远端硬件任务；采集需在明确指定的设备与运行窗口执行。

## 2. 无法全量跑时，分别能测什么

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
| E2E | 真实请求 TTFT、逐 token ITL、吞吐/goodput、tail、峰值内存 | 核对到达轨迹与输出 token 数、排队/组批；无服务/请求 trace 时不做 E2E 精度声明 |

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

工程交付的导入/差异报告/覆盖与回退信息须在工作台可查看并导出；本轮只增加规划和
离线验证/采集工装，不宣称已实现 profiling 导入 UI 或 NPU 生产校准。

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
