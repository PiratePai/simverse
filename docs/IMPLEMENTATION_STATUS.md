# Simverse 重构实施状态（M0–M5 验收清单）

当前开发优先级见 [待开发计划](../ROADMAP.md)。

## 2026-11-19 仓库审视

审视基线 `main@d8dfbfc`，Windows / Python 3.12 全量 580 项通过。额外检查复现了 wheel 缺生产资产、无效场景请求返回 500、扫描跨缺失点插值、比较资格缺证据放行以及导出时间轴以 0 ms 表示未计价阶段。入口与比较问题已通过 [R10 验证](validation/entry-scan-compare-fixes-20261119.md)。未计价阶段与 trace 兼容的修正见下方 2026-11-20 记录。

## 2026-11-18 成本组与统一选用路径

新增 CostSession，移除条件记录的二次全量求值；一次计数/lowering/容量/解析定价，缓存阶段成本
复用执行函数。默认组级耗时，逐行计数和必要边界保留；历史融合 profile 不摊派逐行实测。
KPI/扫描/时间轴/trace/组 CSV 使用统一 selection，零命中显式解析，纯解析与旧指标保留兼容。
11 项新增回归与 15 场景冻结基线覆盖单次求值、所有权、无任意融合、unknown、DT 与导出。
DSH 只读核对的四项建议已落实，主控负责全量验收。规划和准则 §1.6 改为组所有权验收，
不放宽来源、适用域或排名门禁。详见[说明](validation/cost-group-simplification-20261118.md)。
全量 `py -3.12 -m pytest tests -q`：**580 passed**（24.95s）；R27/R26 冻结回放与
七场景 Ascend 覆盖计划无漂移，JavaScript 静态检查通过。浏览器核验解析、R29 TP2 命中、
TP8 全回退与组 CSV；下载的 7 组与同场景 API/CLI 导出逐字段相同。

## 2026-11-20 KV 计数、未知时长和归档入口修正

- CUDA 生产目录的 GQA 在 `TP > Hkv` 时每 rank 持 1 个 KV 头。同一个规则用于目标、DFlash、MTP 的投影与流量，以及权重、缓存和临时对象。默认 H100 TP8 预设的解析 TPOT 从 36.859181 ms 变为 36.844173 ms，已知显存下界从 2,832,959,265 B 变为 2,654,701,345 B。Ascend 条件档及历史 H200 的假设与对应锚点沿用原适用域。来源为仓内固定的 [vLLM 注意力实现](https://github.com/vllm-project/vllm/blob/dc74b8cbbbe3dbb9399fe369757d63defd614cc2/vllm/model_executor/models/qwen3_next.py#L306-L315)和 [QKVParallelLinear](https://github.com/vllm-project/vllm/blob/dc74b8cbbbe3dbb9399fe369757d63defd614cc2/vllm/model_executor/layers/linear.py#L1110-L1125)。
- DFlash 的 `commit_rewind` 和 `hidden_extract` 在事件和 M04 分解中返回未知及原因，trace CSV 的时长留空。轮耗时与 TPOT 提供已计价阶段下界。DT 串行轮补入此前已计价但漏列的 T→D 传输。经过未知步骤的事件只提供开始时间下界，流水周期披露省略的步骤。
- `simverse archive {run,sweep,export}` 提供 v0.1 离线重放。原 v0.1 网页、HTTP 路由和顶层命令退役。新结果的选用指标和执行详情统一走 canonical 字段与 `cost.selection`，记录、校准回放工具已迁移。旧版 v0.2 结果仍可通过 `export-v2` 导出。
- 工作台只接受最近一次单场景运行的响应。Playwright 1.65.0 的 [路由拦截](https://playwright.dev/docs/api/class-route)用于控制真实 Chrome 中的到达顺序。两个测试覆盖旧成功与旧 HTTP 失败，均核对新 `run_id`、时间轴和下载身份，接入 Windows 合入门禁。
- 冻结证据保持原文件。15 个离线回放场景覆盖 5 个 CUDA 新数值锚点和 10 个原计数与容量锚点，另复核 7 个 Ascend 成本覆盖场景。R23、R26、R27 的历史对账通过 `tools/replay_frozen.py` 在合并前的 `a457c76` 复跑。源码核对、离线数值回归和浏览器验证分别记录，未做硬件测量。

## 2026-11-20 合入后复查修正

- 固定 K 的接受分布未知时保留 K，动态 K 保持未知。两者保留可计算的目标端显存、基线和同一份运行身份。整轮时长和 TPOT 返回 null 并附原因。Engram CPU offload 的 GPU 耗时、后继事件开始时间和流水周期披露为已计价部分的下界或部分模拟。
- 旧 v0.2 trace 优先读取选用事件，依次回退至旧记录选用事件和纯解析事件。工作台单独展示导出错误。Chrome 回归经实际按钮和下载内容核对 run_id。
- EP 专家分配共用同一份并行规则。已固定的 Qwen CUDA KV 几何沿用仓内 vLLM `dc74b8cb`，新增 DFlash TP16 投影与驻留手算回归。逐行价格和 Roofline 点共用一次投影。
- 新结果的同卡清单读取 `cost.memory.colocated_ranks[i]`，M06 与能力检查名称沿用原格式。选用时间轴、流水模拟与解析结果相同的场景直接读取顶层 `execution` 和 `cost.pipeline_sim`。有差异时读取 `cost.selection` 中的对应字段。旧结果仍可从文件导出。
- 历史回放入口统一调用 `a457c76` 的实现。Windows / Python 3.12 全量 679 项、Chrome 3 项、15 个冻结场景、Ascend 7 个回放场景和 H200 9 个对照点通过。15 个冻结场景按原序列化口径核对已有数值与库存摘要。未新增硬件测量。

## 2026-11-21 R07b/R09 三路同负载服务端采集

`simverse collect-service-trace` 从固定的 vLLM `11aa1369` 分进程采集三路服务端增量输出组、实际到达、原生请求阶段和投机接受统计。轮内探针覆盖 DFlash 和 MTP 的 draft、verify、recovery。三路采集各 rank 的运行时 allocator 峰值和窗口。`trace-summary` 归约双 rank 合成样本，校验阶段、显存和缺项。三路完成吞吐维持 100 token/s，TTFT P50 为 4 ms，P95 为 6 ms。当前机器无可用 CUDA 设备与 vLLM，设备实采和整卡峰值仍待匹配环境。[验收方案](validation/r07b-r09-trace-acceptance.md)

## 2026-11-21 R07b/R09 离线请求轨迹与服务端 TTFT

新增 `simverse trace-summary`。输入校验覆盖请求与轮 ID、提交序号、时钟和时间基准、终止顺序、请求清单摘要及采集原始文件 SHA256。摘要复用现有归约器计算 ITL、吞吐和最近秩分位，并按服务端到达与提交时刻计算 TTFT、E2E 和组合 SLO。人工轨迹核对 100 token/s 完成吞吐、175 token/s 事件吞吐、TTFT P50 4 ms 与 P95 6 ms。三路匹配硬件的服务采集与阶段残差仍待验收。见 [端到端 trace 验收方案](https://github.com/PiratePai/simverse/blob/main/docs/validation/r07b-r09-trace-acceptance.md)。

## 2026-11-21 R07b/R09 完成请求吞吐修复

完成请求计数、吞吐与 goodput 共用 `[start,end)` 窗口筛选。离线样本的完成吞吐从误计的 200 token/s 修正为 100 token/s，共计 2 个完成请求。事件吞吐仍为 175 token/s。新增一条 V08 回归，并验证最后一个输出恰好落在窗口左边界的情况。

## 2026-11-20 R07b/R09 方案验收

已核对请求轨迹可复用的归约、事件校验、导出和成本蒸馏入口。已约定三路配对、字段、时间边界、离线手算和真实采集准入条件。手算样本复现了完成吞吐的跨窗计数缺口。离线请求导入和 TTFT 归约在后续实现，上机采集仍待实施。本轮文档验收核对了 106 项仓内链接、5 个源码工作台文档路由和手算样本。Windows / Python 3.12 全量 679 项通过。见 [端到端 trace 验收方案](https://github.com/PiratePai/simverse/blob/main/docs/validation/r07b-r09-trace-acceptance.md)。

## 2026-11-20 重复实现与入口归整（PR #34 历史记录）

以下记录 PR #34 合入时的入口和结果口径。后续变更见本页 2026-11-20 KV 计数与合入后复查记录。

- GQA 的目标与草稿计数、显存和分片说明共用每 rank 几何计算。同卡与 baseline/DT 复用目标端显存对象组装，分别保留自己的草稿驻留和预算契约。
- 接受率由方法层校验，HTTP 和 CLI 解码复用同一规则。空逐位接受率在入口给出字段错误，经 `run_v2` 直接调用时返回失败结果。默认扫描保留 27 个输出点，按完整场景摘要在请求内复用求值，实际运行 25 次。
- 工作台和 CLI 使用同一个 CSV 生成器。工作台通过 `POST /api/v2/export/{ops,groups}` 提交已有结果，保留大整数和旧结果的解析行价回退。旧 GET 导出路由继续返回 410。v0.1 CLI 和 HTTP 适配器归入 `simverse/legacy`，公开命令和路由继续供归档重放。记录成本指标别名和报告副本按现有回放契约保留。
- KV 数值评估基于 [vLLM 固定提交 `dc74b8cb`](https://github.com/vllm-project/vllm/tree/dc74b8cbbbe3dbb9399fe369757d63defd614cc2) 的 `Qwen3NextAttention` 与 `QKVParallelLinear` 源码。对 Qwen3.6 的 2 个 KV 头，vLLM 在 TP4/8 按单头跨 rank 复制。仿真旧口径按每 rank 2 头计。临时替换几何规则的离线试算中，默认 H100 TP8 预设的 TPOT 从 36.859181 ms 变为 36.844173 ms，已知显存下界减少 178,257,920 B。15 个冻结场景中有 9 个改变计数和容量，影响 Ascend 条件成本及历史 H200 235B 对照。此次归整保留冻结口径，数值变更需要按后端和归档采集口径分别验收。
- Windows / Python 3.12 全量 672 项通过。15 场景冻结回放、R27 归因和 JavaScript 语法检查通过。无头 Chrome 从真实工作台运行默认预设，下载两类 CSV 和结果 JSON，并用 CLI 从同一 JSON 导出比较，两类 CSV 内容一致。归档首页和注册表可访问。此次验证使用源码、离线回放和本地浏览器。

## 2026-11-17 Ascend 逐记录成本与补采计划

R29 双 rank 五类 collective 曲线与 R28 14 个独立 GEMM 形状分别接入条件记录档，来源/派生双重哈希校验；模型、形状、BF16、rank、原语、角色逐项门禁。只改选用成本，原始计数/容量与纯解析指标保留；没有新增 NPU 执行。两栈不可混用，不升级生产实测 profile，排名资格不变。Web 新增选择器、分阶段覆盖/时间口径/逐项回退与补采表，KPI/矩阵/扫描/逐行/时间轴及 CLI/API/CSV/HTML 同源。七个固定场景的覆盖与回退计划已落档，按实际缺点调整昇腾计划（真实原语、TP2 形状、verify M16、实际 MoE/GDN/attention、草稿身份，TP4/8 后排）。新增 12 项回归，收集 557→569；全量 `py -3.12 -m pytest tests -q`：**569 passed**（19.27s）。蒸馏/七场景回放 `--check` 与 JavaScript 静态检查通过。浏览器核验 R29 TP2 命中/TP8 全回退、R28 TP8 精确 GEMM、关闭记录、KPI 与 CSV 一致；未计价方法阶段显式 unknown，不计入分母。见 [说明](validation/ascend-record-costs.md)。

## 2026-11-17 910B3 旧仓解析档接入

从 model-verse @6a8acce 冻结 `npu_params.py` 与 H3 场景（完整 SHA256 入 assets/identity），确认 910B3→ascend_910B_64G 映射。注册 Cube 249T / Vector 31T、HBM 1.6TB/s、64e9B、MFU/MBU 0.20；没有官方 SKU 核验的值标 assumed，不以 R29 快照替代规格。目标/DT 草稿下拉框可选，目标 NPU 联动 BF16；API 拒绝 FP8/量化与 V4.1 CUDA 配方复用。结果降级解析假设、未知部署资格，不引入实测 profile。GPU 默认 MBU=1、身份与计价行为保持。新增 7 项来源/独立手算/API/负例/身份回归，收集 550→557；全量 `py -3.12 -m pytest tests -q`：**557 passed**（20.19s）。浏览器实测目标 910B3/TP2/BF16/CTX4096 的 Qwen3.6+DFlash 运行与 V4.1 错误提示；JavaScript 静态检查、UI 检测器和默认场景回归通过；没有 NPU 硬件执行。见 [迁移说明](validation/ascend-910b3-device.md)。

## 2026-11-17 R29 / PR #23 审查修正

本轮仅在 Windows 审查既有 NPU 产物、阅读源码并执行离线回归，没有新做硬件测量。以 [审查报告](validation/ascend-910b3-r29-review.md) 与 [派生 review.json](evidence/ascend-910b3-r29/review.json) 为准，下节是采集时记录。原始 23 个声明产物保留，22 个匹配 manifest，forward 日志原哈希不符单列；未重写原 manifest。外部原报告固定 LF，避免 Windows 检出改变其原哈希。

修正 AllGather 分子漏 world（rank0/256MiB busbw 8.863→17.725 GB/s）、逐头 MRoPE 伪通过、TP1 产 token 的范围、权重 0.32 GiB 与 OOM 分配量混淆、EP2 CPU 玩具专家/路由元数据检查、graph 差分和 ACL 粒度断言。来源不足继续披露；生产计数/成本/设备/资格零改动，全部 not-admitted。新增 7 项离线证据回归，规划调整见 ROADMAP 与 ascend-validation-plan。本轮全量 `py -3.12 -m pytest tests -q`：**550 passed**（54.43s）；审查回放 `--check`、JavaScript 静态检查、默认 `simverse.cli run-v2` 与 fixture dry-run 均退出 0。远端 required check 以 PR #23 的最终提交为准。

## 2026-11-17 R29 昇腾 910B3（物理卡 5/6）验证重采

按 [昇腾优先、少卡验证与 profiling 计划](ascend-validation-plan.md) 在 `ASCEND_RT_VISIBLE_DEVICES=5,6`
上执行 P0 的 A-N1（ACL 返回码/context、集合通信 shape 与数值、graph 分阶段、MRoPE 参数域）、
A-N2 forward 预检与显式执行、A-N3 的 TP2/EP2 算子切片与容量快照。
新增产物入册 [evidence/ascend-910b3-r29](evidence/ascend-910b3-r29/)，带栈身份、设备可见性
与逐文件 SHA256；结论与边界见 [R29 报告](validation/ascend-910b3-r29.md)。
R28 归档字节未改；全部记录仍 not-admitted，未改变 DEVICES/PROFILE_SETS/模型执行状态/部署资格。
本轮栈为 torch 2.10.0+cpu / torch_npu 2.10.0.post4 / vLLM 0.32.0 / vllm-ascend 0.19.1rc2，
与 R28 采集声明的 2.9.0 栈不同，两组数据不得相减或拼接。
无新增测试（收集数仍 543）；`py -3.12 -m pytest tests -q` → **538 passed / 5 skipped**。

## 2026-11-17 R28 路径可移植性修复

将 Ascend 工具的上游来源改为仓库 URL + 固定 commit，来源清单 schema 2 使用
仓库标识 + 仓库内相对路径，保留全部源码 SHA256；实际执行仍导入当前 Python 环境
安装的包。新增异目录/不同 cwd/隔离子进程回归1项（542→543）；全量 `py -3.12 -m pytest tests -q`：
**543 passed**（17.34s），没有NPU执行。

## 2026-11-17 R28 审查增补

910B3 原始证据归档与审查结论已入库，28 个原始数据/脚本文件字节不变；纠正
AllReduce busbw、预算阶段、graph 归因与跨运行 launch 误判，隔离 ReduceScatter
形状错误、MRoPE 冲突结果、非等价代理及缺采集身份数据。所有记录保持
not-admitted，未改变产品 DEVICES/PROFILE_SETS/模型执行状态/部署资格。
[审查报告](validation/ascend-910b3-20261117.md)与
[昇腾优先、少卡及 profiling 规划](ascend-validation-plan.md)是本轮入口。
新增证据测试5项、Ascend forward入口离线测试21项（DSH实现、主控复核）。
Qwen3.6提供真实vLLM/vllm-ascend入口（少层fixture与本地权重），V4.1为明确阻断的部署计划；
没有NPU运行/权重精度/性能准入声明。见[forward路径报告](validation/ascend-forward-paths-20261117.md)。
主控全量验收：`py -3.12 -m pytest tests -q` → **542 passed**（19.53s）；
`node --check simverse/static/workbench.js`、默认 `simverse.cli run-v2` 与新增 fixture dry-run 均退出0。
远端 required CI 另按 PR #21 的最终提交检查；没有新增硬件测量。
首次远端检查发现 Windows clean checkout 将源 config 转为 CRLF、破坏其固定 SHA256；
已为该配置固定 LF（内容/原哈希不变），不放宽来源校验；autocrlf=true 的干净索引导出
再次全量 **542 passed**（21.11s），源配置哈希一致。最终 CI 以修后提交为准。


更新：2026-11-17（R27：cost-integrity 轮——计算口径/统计身份/指标元数据/运行身份六项修复：
①dtype_route 按模型执行配方选 GEMM 计算档（V4.1 Marlin W8A16/W4A16 → bf16 Tensor
Core 档 989.5 TFLOPS，R22 pin；无配方模型沿用 dense_w 启发档并在 cost_basis 披露；
理论峰值路由修正 ≠ 实测吞吐校准）；②price_row 同次计价输出行级元数据
（compute_dtype/peak_tflops/effective_peak_tflops/bandwidth_tb_s/cost_basis，
TFLOP/s+十进制 TB/s），_roofline_points 重复分支/未用 dtype 修复，前端可选字段兼容；
③DFlash/MTP t2d 期望字节保浮点（K15/α0.72 锚点 58209.08759654315，旧代码 int() 截
58209），d2t 协议确定字节保持整数；④内部 chain 计价与计数同源（dquant 路由、
colocated=target/DT=草稿设备、不套目标 Engram host 开关、draft_tp>1 显式拒绝）；
⑤pipeline_sim 模式增补 M01 保留既有 reason/support/evidence/formula（Engram host
警告与流水分列并存不堆叠）；⑥device_snapshot_digest 覆盖硬件/链路数值+来源+拓扑
（同 ID 改带宽/峰值 → run_id 变化；run_id 基变更，历史证据包未回改）。默认场景
replay：V4.1 M01 116.33572728→116.33641192（+0.00068ms，GEMM 档修正经 0.15·min
项进入；远非"2×错误"——decode 行带宽主导）、Qwen3.6+DFlash M01 数值中性
（t2d 58209→58209.0876），work/memory 逐字段零漂移（evidence/cost-integrity-r27 +
活性入口 tools/replay_cost_integrity_r27.py --check）；csa2-core after.json 按新成本
档重基（work 不变，M01/M04 重定价，rebase-r27-provenance 入册）。W8A16/W4A16 语义
最小数值 fixture（e4m3/e2m1/e8m0/打包/反量化等价，纯 stdlib math-oracle 层，不升格
模型级 cpu_forward）。Engram Host 成本核对方案与可部署性数据采集方案见
[修复报告](validation/cost-integrity-review-fixes.md) §五。
26 项新测试（回归 20 + 数值 fixture 6）。早前 2026-11-16（R26：Qwen3.6 文本路径 MRoPE position_ids 语义补齐——vllm_qwen36_mrope_pin 5 文件（@dc74b8cb，blob sha1 校验）+ HF pin 双栈逐行核对：文本退化（三行恒等→交错重组恒等→标准部分 NeoX RoPE，rotary 64/256、[11,11,10]、theta 1e7）、prefill arange/decode 缓存续位、MTP 位置传递（复制/+1/末步不推进/回退 num_rejected/夹紧）；三路 f64 对照 diff=0 + 官方 HF 旋转类实测 4.35e-07 + 6 负例；mrope_section/interleaved 升 mapped、unverified_paths 清空 → forward_source=source-checked、structure_complete=True（不冒充执行/实测）；账本零漂移 replay；17 项新测试；多模态 2D 分支域外，见
[报告](validation/qwen36-text-mrope-20261116.md)。早前 2026-11-13（R23：Qwen3.6 原生 MTP forward 证据与仿真映射——vLLM @10a854ed
`qwen3_5_mtp` 等 8 文件 + SGLang @1b2870d 1 文件双栈逐行核对（SHA256/blob sha1 pin）；
查明 mtp.* checkpoint 张量/加载重映射/colocated embed-lm_head 物理共享/每轮 K 均匀步+
层 j%n 循环；count/methods/pipeline/memory 按 pinned 结构建模，撤销 K≤n_predict 旧限制与
"目标首层 GDN 模板"研究近似；fixture 级独立数学对照（双实现互核 4.97e-13+30 项账本对账）；
25 项新测试+门禁更新；前后 replay M01 64.55→66.96ms、steady +141MB（target/verify 冻结）；
DT 分离/专属 embedding/无头模型显式拒绝；kernel/CPU/GPU 未执行未实测，部署排名仍阻断，
见 G-C15 与
[报告](validation/qwen36-mtp-forward-20261113.md)。早前 R22：V4.1 量化映射按 20 文件 pin（quant-pin.json）source-checked 建模——
早前同日 R24：V4.1 SwiGLU clamp 与 compress_ratios 尾部按 13 文件
pin（swiglu-tail-pin.json）source-checked 解除——clamp 语义 gate 单侧/up
双侧、alpha=1/beta=0、触发 is-not-None vs >0（0 值分叉域外拒绝）；消费两路
（共享专家 MLP.act_fn + 路由专家 gemm1→激活→gemm2，注册执行域 Marlin 经
apply_moe_activation 同 clamp）；新增 moe_act/moe_shared_act 行（6 算术
op/元素，字节已归原行单一所有者）；尾部 [40:43] 逐项归 DSpark/nextn 草稿层
（层号 40-42、0=纯 SWA、is_backbone 门控永不为源；非零尾部域外拒绝）；
candidate_block_size 补入 struct 集合（R20 注记因漏登记未生效的修复）；
V4.1 config 覆盖完备；**2026-11-16 R25：稀疏 MLA 注意力核本体按 csa2-core-pin.json
14 文件（vllm @f358f057 5 文件 + FlashMLA @847ef308 9 文件，cmake GIT_TAG 钉定）
source-checked 建模并修正**：decode 头数 padded_heads=64（fp8 kernel h_q∈{64,128}、
Q 零填充——TP8 旧口径少算 8×）、SM90 恒 split-KV（o_accum/lse_accum/combine +
get_decoding_sched_meta 贪心调度移植 count/flashmla_sched）、prefill 记录读改
gather 口径（全量压缩前缀+SWA 尾窗→bf16 工作区→逐对行读）、R20 decode
sparse_index_build 归因修正为 prefill 每 chunk 每层（decode 实际每步一次 SWA
窗索引+每压缩层全局槽位+每层型调度器元数据）；q_rope 写 padded 宽；devices
增 sm_count（spec）；默认场景 decode flops +2,850,849,054（+39.7%）、行 +73、
瞬态峰值 +168.6MB 下界（docs/evidence/csa2-core 归因；before=main ee70ce2）。
structure_complete 仍 False——剩余 unverified 均为显式执行变体（specialized
decode/SM100 mega/FlashInfer plain-row/PCP 等），部署排名资格不变（G15）。
[报告](validation/v41-csa2-attn-core-20261116.md)、[缺口 G-C17](gaps/capability-gaps.md)。
早前 R22：V4.1 量化映射按 20 文件 pin（quant-pin.json）source-checked 建模——
quantization_config 5 叶键 mapped：MXFP8 线性（e4m3 1B+ue8m0 (1,32) scale，checkpoint 32×32 块
装载展开）+ MXFP4 路由专家（打包 2 值/byte+scale+Marlin I→128/H→256 分配对齐）+ embed/lm_head/
norm BF16 未量化 + ≤SM90 Marlin W8A16/W4A16 执行域激活 BF16；scale_fmt 键 loader 不消费、按声明
矛盾拒绝；场景 V41_QUANT 冻结；known_peak +14.36 GB（Marlin 对齐主导）；Humming/SM100/emulation
变体登记 G-C14。早前 dtype 域收紧：valid domain 固定 official V4.1 config dtype=bfloat16；full_config 的 float16/float32/缺失 dtype 明确拒绝，无 full_config 仅固定资产 BF16 默认。Gate 修复：GateLinear 权重沿 model loader BF16、输入由 pinned mHC layer_input BF16 证明，均 2B；moe_gate 权重 1,966,080→3,932,160 B/层、输入 5,120→10,240 B/token；新增 linear.py/base_loader.py/layer.py pin（14 文件）；数值见报告 §10。三轮复核：fused EP 只按 mega 断言、共享专家独立相加、ExpertMapManager 未入 pin 读取点登记、证据包 SHA256 自校验，见报告 §9）；2026-11-12（R21：V4.1 MoE scoring/top-k/分组路由按 14 文件 pin source-checked 建模——sqrtsoftplus/noaux_tc 平坦偏置 top-6 无分组、FP32 logits 出口修正、路由偏置对象、EP 保守整除域（仅 mega 断言）；早前 R20：V4.1 CSA2 索引打分/top-k/candidate block 按 21 文件 pin source-checked 建模并修正记录宽度/bf16 权重与激活/主 attention norm 常驻/环形 state/设备 family 校验；R07a：DFlash 服务栈双栈审计（G-C3 核对部分解除）+ ctx 投影跨度基、草稿生效块与 full 层非因果三修正；同日早前 R06：Qwen3.6 GDN+MoE 小尺寸 CPU forward 对照 + attn_output_gate 漏计修正，rebase 到 R05 mHC/Sinkhorn 修正之上；早前 R04/两模型门禁/A 组/校准段落保留为历史记录）。主控会话持续维护；上下文压缩后从本文件恢复。
代码身份：本仓 git（PR #15（R23）、#16/#17（R26）、#18（R25）已合入 main；远端公开仓库 `PiratePai/simverse`）；包版本 0.2.1（对齐结果协议 v0.2；ModelIR 独立 0.3.0）。
运行：R27 在 ZCode 会话（干净 HEAD 1196510 + 本轮改动的独立 worktree `../nv-r27-verify`，隔离并行工作台任务在主工作树的在途编辑）执行 `py -3.12 -m pytest tests -q`：**495 passed，exit 0**（含 G6 计数门禁）；`tools/replay_cost_integrity_r27.py --check`（新，活性入口）、`tools/replay_v41_csa2_core.py --check`（重基后）、`replay_v41_swiglu/quant/moe --check`、`tools/replay_qwen36_mrope.py --check`、`tools/replay_qwen36_mtp.py --check`、`py -3.12 -m simverse.cli run-v2`（status=completed）均 exit 0。主工作树同期 `pytest tests` 有 2 项 CSV 导出失败 + G6 计数差，全部来自并行工作台任务对 simverse/cli.py/tests/test_consistency.py 的未提交在途修改（本轮未触碰；对方落地时自会更新）。历史回执（非本次执行）：R25 在 ZCode 会话 469 passed；R26 在 ZCode 会话 442 passed；R23 验收修复在 Codex 宿主 425 passed；R24（PR #14 合入前）399 passed；R22 验收修复 367 passed（Codex 宿主）；dtype 修订提交 `b9a5a0f` 上 334 passed；旧 DSH 沙箱 328 passed + 6 个 `tmp_path` 环境错误（见 §3.9）。
**pytest 收集数：713**（2026-11-19 R10 两批：批次二扫描/比较 36 项（tests/test_scan_compare_r10.py：sweep 口径配对与回退、find_crossover 手算锚点、统一资格矩阵与口径核对正负例）+ 批次一入口修复 56 项：tests/test_entry_points_r10.py——wheel 资产定位/清单、共享预设三入口digest 一致、run/sweep 畸形输入 4xx 字段路径、根目录文档白名单；2026-11-18 成本组回归 11 项）；历史收集说明：（`py -3.12 -m pytest --collect-only -q tests`，2026-11-17 R29 审查：543→550 = 离线证据回归7项；此前 R28 路径移植542→543；516→542 = 证据5 + forward21；此前两项任务集成516。469→495 = R27 cost-integrity 26 项（tests/test_cost_integrity_fixes.py 回归 20：路由/元数据/期望字节/chain/警告保留/身份/隔离/计数不变；tests/test_v41_quant_numeric_r27.py 数值 fixture 6：e4m3/e2m1/e8m0/MXFP4 打包双解码器/W8A16 反量化 GEMM 等价/SwiGLU 随机模糊）。2026-11-16 R25+R26 合并时为 469；442→469 = R25 注意力核本体 27 项（tests/test_v41_csa2_core.py：pin 1、调度器锚点/独立重演/负例/块公式 7、decode 手算绝对 5、prefill 手算 3、守恒/瞬态/设备 3、device_sms 拒绝 1、replay 冻结 1、门禁事实 2）；425→442 = R26 文本 MRoPE 语义 17 项（tests/test_qwen36_mrope_r26.py）；424→425 = 草稿私有专家位宽与驻留守恒回归 1 项；399→424 = R23 原生 MTP 25 项；367→399 = R24 SwiGLU/尾部 oracle 32 项；365→367 = R22 验收回归 2 项；334→365 = R22 量化映射 31 项）。
R23 验收修复：MTP 私有层 count/cost/通信改用 `draft_quant`（与 memory 原有口径一致），共享目标 `lm_head` 继续按目标位宽读取与计价；伪模型重复登记的 final norm 驻留对象移除。固定 K=1/TP8 replay 的草稿专家 `weight_read_expected` 1,572,864→3,145,728 B、M01 66.96182490→66.96209320 ms、steady 2,841,006,528→2,841,004,480 B；target_decode/verify 冻结。`replay_qwen36_mtp.py --check` 现同步校验成本、归因和指标，避免只比较 work totals 漏掉成本漂移。来源仍为 R23 vLLM @`10a854ed`/SGLang @`1b2870d` pin；本次未执行 vLLM/SGLang 模块或 GPU 测量。
R22 验收修复：`_coverage` 为未知 `quantization_config.*` 叶键登记 unsupported-semantic 与 gap；V4.1 仅移除已映射的 5 叶键及容器缺口，避免未来新增语义被静默放行。`ScenarioV2` 的部分 `draft_quant` 覆盖改用草稿默认档（expert_w=8），避免继承主模型的 expert_w=4。来源仍为 R22 固定 vLLM @`f358f057` 的源码 pin；本修复只调整本仓覆盖门禁与场景默认值，未执行 vLLM kernel、CPU forward 或 GPU 测量；两款主模型的部署排名限制不变。
工作台 `py -3.12 -m simverse.cli serve` → http://127.0.0.1:8765/。v0.2 首页为 `/`。v0.1 离线重放使用 `simverse archive`。
合入准则：[MERGE_CRITERIA.md](MERGE_CRITERIA.md)（G1–G17 门禁在 tests/test_merge_gate.py，测试数与本文件一致性由 G6 强制）。

**2026-11-17 两项任务集成审查**：495→515 = 工作台回归 20 项；515→516 = 逐行计价覆盖集成回归 1 项。审查发现原 top-40 仅覆盖 Qwen 27 类中的 4 类及 V4.1 51 类中的 3 类，新增 `cost.row_prices` 全行计价摘要，透视与 CSV 优先使用该字段；原 `cost.roofline_points` 保留 top-40 供历史证据兼容。文档路径锚定改用 `Path.is_relative_to`。全量验收见本轮合入记录。

**2026-11-17 R27：cost-integrity 轮（计算口径/统计身份/指标元数据/运行身份）**
（主工作树，基线 HEAD 1196510；专属报告
[docs/validation/cost-integrity-review-fixes.md](validation/cost-integrity-review-fixes.md)）：

- **A 计算峰值路由**：`cost/t0.dtype_route(quant, quant_recipe)` 增加执行配方
  入参；`model_ir/families` 的 extras.quant 注册 `gemm_compute_dtype="fp16"`
  + `gemm_compute_basis`（Marlin W8A16/W4A16 于 BF16 Tensor Core 执行、激活
  BF16；R22 quant-pin @f358f057）。V4.1 全部 GEMM 计算档 fp8(1979)→
  bf16 档(989.5 TFLOPS dense H100)。**理论峰值路由修正，不是实测吞吐校准**
  （Marlin 实际吞吐 < bf16 理论峰值，无实测证据不折减；域外变体 Humming/
  SM100/emulation 不进路由）。无配方模型（Qwen3.6/DFlash/MTP/chain 草稿）
  沿用 dense_w 存储位宽启发档，cost_basis 披露"存储位宽=计算位宽"为乐观
  假设；不引入"全看 act"新全局规则。launch/融合/调度/CUDA Graph 缺口显式
  登记为 assumption A2b_kernel_gaps（不逐算子加 launch）。
- **B Roofline 元数据**：`price_row` 在同一次计价内附带
  compute_dtype/peak_tflops（dense 规格）/effective_peak_tflops（×mfu_cap）/
  bandwidth_tb_s（十进制）/cost_basis；`_roofline_points` 重复分支+未用
  dtype 代码删除，字段直接取自计价输出；row_id 稳定；旧字段全保留（前端
  可选字段兼容，未依赖前端任务）。
- **C 期望通信字节**：DFlash/MTP `t2d` 由 `int()` 改 `float()`（期望身份
  保持 expectation；契约/序列化/指标/链路成本核对无二次截断；d2t 协议
  确定字节保持整数；消息包长取整/对齐未混入）。
- **D 内部 chain 计价**：草稿行计价改用 dquant 路由（与计数同源）；
  colocated=target 设备、DT=草稿设备（旧恒 target）；不把目标 Engram
  host 开关套到无 Engram 的 tiny 草稿；dt 模式 draft_tp>1 显式拒绝；
  chain/tiny 内部研究隔离不变（G8 面补测）。
- **E 指标警告覆盖**：pipeline_sim 增补 M01 时保留既有 support/reason/
  evidence/formula/assumptions 并去重合并——"Engram host 表查找与预取时延
  未计"与"串行 TPOT/流水周期分列"并存；capability 侧 Host 降级检查保留。
- **F 运行身份**：`_device_snapshot(dep)` 覆盖实际计价消费的设备/链路数值
  +来源+拓扑（target/draft peak_by_dtype/hbm_bw/sm_count/mfu_cap/…
  sources、link bw/fixed/source、tp/ep）；同 ID 改带宽/峰值 → 快照与
  run_id 变化；部署语义参数（target_ep/draft_tp/v41_kv_record/engram_
  placement/budget）由 input_digest 全量覆盖；时间戳等非语义字段不入；
  键序 canonical_json 归一。**run_id 基变更：历史 run_id 不再可复现**
  （证据包未回改，兼容方式见报告 §六）。
- **数值变化**（evidence/cost-integrity-r27，before=HEAD 1196510）：
  V4.1 baseline M01 116.33572727816723→116.33641192134922（+0.00068464ms；
  GEMM 档差仅经 max+0.15·min 的 min 项进入，decode 行带宽主导——修正"选错
  峰值"≠"整模型 TPOT 错 2 倍"）；M01_baseline/M04 同步；Qwen3.6+DFlash
  M01 36.859180745326654 不变（t2d 58209→58209.08759654315 对 t_t2d 影响
  ~4e-10ms，低于该量级双精度分辨率）；work/cost.memory 三场景逐字段零
  漂移。csa2-core/after.json 重基（work 不变、M01/M04 重定价，
  rebase-r27-provenance.json 入册；replay_v41_csa2_core 归因改同价复算）。
- **最小数值 fixture**（tests/test_v41_quant_numeric_r27.py，6 项，纯
  stdlib math-oracle 层）：e4m3/e2m1 量化-反量化误差界、MXFP4 打包双
  解码器一致、W8A16 反量化 GEMM 等价（bf16 操作数+fp32 累加 vs f64
  oracle）、SwiGLU clamp 随机模糊、scale 布局公式对齐。**不升格**：模型级
  cpu_forward 保持 not-run（G10/G15 不变）。完整 V4.1 CPU fixture 覆盖
  计划（Engram/mHC/注意力/MoE/SwiGLU）与前置条件见报告 §五。
- **测试**：+26（test_cost_integrity_fixes.py 20 + test_v41_quant_numeric_
  r27.py 6）；全套 495 passed（独立 worktree，见"运行"行）。

**2026-11-16 R26：Qwen3.6 文本路径 MRoPE position_ids 语义**（分支
`codex/qwen36-text-mrope-r26`）：vllm_qwen36_mrope_pin 5 文件
（@dc74b8cb 与 R01 qwen35 pin 同 commit：qwen3_vl、rope_init、rope_base、
rope_common、rope_mrope；SHA256+blob sha1 对 GitHub contents API 校验）
+ HF pin（modeling L130-194/L655-690/L1322-1332）+ 既有 vllm_qwen35_pin
（qwen3_5.py L440-446 get_mrope_input_positions、qwen3_next.py L344-401/
L423-426）与 vllm_qwen36_mtp_pin（qwen3_5_mtp/tdar 位置步进）三栏 source-
checked 入册（ev-vllm-qwen36-mrope，executed=False；sm-qwen36-text-mrope）。
语义结论：**文本路径 MRoPE 完全退化为标准部分 NeoX RoPE**——HF
position_ids=None 时 arange(seq)+past_seen 扩 4 行（[text,T,H,W] 行 0 供
mask/GDN、行 1:3 进旋转，纯文本四行恒等）；vLLM 文本 positions 一维
（mrope_positions_factor 仅视频剪枝=4），MRotaryEmbedding 1D 走
_forward_text_rope 不经 section 重组，SupportsMRoPE 文本 fallback
arange(len).expand(3,-1) 三行恒等时交错重组亦恒等；mrope_section
[11,11,10] 交错布局（idx%3，sum==rotary_dim//2==32 构造断言）；MTP 位置
传递：prefill 逐元素复制 target positions、每草稿步 +1 夹 max_model_len−1、
末步不推进、下轮 seq_len=target_seq_len−num_rejected（+1 bonus 回退）、
sample_src=+1/步。fixture 数值对照（ev-qwen36-mrope-cpu-numeric，
executed=True：三路独立实现（HF 转写/vLLM 转写/标量复数 oracle）f64
两两 diff=0.0 + 官方 pinned HF 旋转类本机执行 4.35e-07 + MTP 状态机
≥16 不变量 + 6 负例 + 测试内独立 mini-oracle；fixture 绑定 11 个 pin
文件 SHA256）。落地：config 键 mrope_section/mrope_interleaved 升 mapped
（文本域）；ModelIR extras["rope"]（几何+退化语义+MTP 步进 spec）+
sum≠rotary_dim/2 显式拒绝；qwen36 unverified_paths 清空 →
forward_source=source-checked、**structure_complete=True**（只表示源码
证据完备，cpu_forward=not-run/GPU=none 另列，排名门禁 G15 不解除）。
计数/内存零改动（rope 行 3·T·heads·rotary_dim 口径不变）：干净 main
ee70ce2 基线 replay（tools/replay_qwen36_mrope.py，docs/evidence/
mrope-map/）两场景账本逐字段零漂移、M07 structure_complete False→True、
M02 排名理由不再含 mrope（CPU/GPU/预算仍阻断）。多模态 2D 行相异分支
（get_rope_index/rope_deltas/视频剪枝 factor=4/openpangu）域外未核。
测试 tests/test_qwen36_mrope_r26.py 17 项（pin 完整性/行号锚/证据四栏/
fixture 复算/解析器负例/replay/CLI-API 一致性 + torch 可选 live）；
G10/model_ir/product_scope 门禁随事实更新（mrope 阻断解除）。

**2026-11-13 R23：Qwen3.6 原生 MTP forward 证据与仿真映射**
（分支 `codex/qwen36-native-mtp-r23`）：vllm_qwen36_mtp_pin 8 文件
（@10a854ed：qwen3_5_mtp.py、qwen3_next_mtp.py、config/speculative.py、
eagle/utils.py、mtp 与 target_dependent_ar speculator.py、2 个 config
测试 + e2e 测试；逐文件 SHA256+blob sha1 对 GitHub contents API 校验）
+ sglang_qwen36_mtp_pin 1 文件（@1b2870d 对照）source-checked 入册
（ev-vllm-qwen36-mtp / ev-sglang-qwen36-mtp，executed=False；
sm-qwen36-mtp 替换 sm-qwen36-mtp-unverified）。语义核对结论：
checkpoint mtp.* 张量（fc/pre_fc_norm_*/norm/layers.{i}.*，无独立
embed/lm_head——官方 e2e 按 mtp.layers.0.* 复制模块证明）；加载
is_unused_checkpoint_weight 只留 mtp.*/embed_tokens/lm_head + mtp.→model.
重映射；colocated 经 load_eagle_model 物理共享目标 embed/lm_head（零字节，
与 DFlash R02a 同配方）；combine=双预归一+cat+fc(2h→h, ColumnParallel
gather_output=True, 无激活/无残差直加——区别于 DeepSeek V3 式 comb)；
层=full_attention decoder 层（gated GQA+MoE，几何=目标 full 层，非目标
首层 GDN）；每轮 K 均匀步（无 catch-up 加权）、步 j 层 j%n_predict 循环
（K>n 合法，e2e K=3×MTP1 在册；撤销 K≤n_predict 旧拒绝）；草稿自有
per-token KV，被拒步回退。落地：ModelIR extras["mtp"]+两键 mapped 注记
+专属 embedding 拒绝；count/draft.mtp_step_rows 逐步重写（撤销 GDN 模板）
+mtp_kv_rewind_row+mtp_layer_model；methods/plans.plan_mtp v0.3；pipeline
逐步阶段+草稿通信（allreduce/EP all2all/fc all-gather/logits gather）+
mtp_serving_integration（colocated=supported source-checked）+dt 双模式
拒绝；lowering/memory MTP 常驻（fc+3 norms+单层权重同 TP/EP 配方）+草稿
KV+临时下界+embed/lm_head 零字节引用。fixture 级独立数学对照
（ev-qwen36-mtp-cpu-numeric，executed=True：torch 复刻 vs 纯 Python 标量
参考，7 张量×3 步 max diff 4.97e-13；30 项 GEMM 账本对账全对；手算锚点；
tests/fixtures/qwen36_mtp_r23_numeric.json）——**不升格**：模型级
cpu_forward 保持 not-run。测试 tests/test_qwen36_mtp_r23.py 25 项 +
G10/G13/G16/model_ir/product_scope/pipeline_v02 随事实更新。前后 replay
（docs/evidence/mtp-map，k=1 同场景）：target_decode/verify 逐字段冻结；
draft_step_0 整段替换；M01 64.55→66.96209320 ms、steady 2,699,569,088→
2,841,004,480 B（含本次验收修复）。仍缺（G-C15）：kernel/CPU/GPU 未执行、真实 checkpoint
未加载、多模块 MTP/量化草稿头/采样档变体、草稿 prefill KV 同步、接受率
假设——部署排名资格不变（mrope 未核状态已于 2026-11-16 R26 解除，见下节；排名仍被 CPU/GPU/预算阻断）。

**2026-11-13 R24：V4.1 SwiGLU clamp + compress_ratios 尾部**（分支
`codex/v41-swiglu-tail-r24`）：swiglu-tail-pin.json 13 文件（vllm
@f358f057，blob sha1 对 vllm-tree.json 逐项校验，源文件落盘）source-checked
入册（ev-vllm-v41-swiglu-tail，executed=False；sm-v41-swiglu-tail）。
语义结论：clamp gate=clamp(max=limit) **单侧**、up=clamp(±limit) **双侧**、
out=gate·sigmoid(alpha·gate)·(up+beta)，V4.1 alpha=1.0/beta=0.0（无
swiglu_alpha/beta 键→FusedMoEConfig 默认）；触发 MLP is-not-None vs Triton
HAS_LIMIT>0（utils.py L812）在 0 值分叉 → config 域拒绝 0/负/非有限/缺失；
消费两路均在注册执行域（Marlin apply_moe_activation L140-158 + 共享专家
MLP.act_fn；MegaMoE activation clamp 语义同源但 SM100 域外）。尾部
[40:43] 逐项归 DSpark/nextn 草稿层（dspark.py L128-136 层号 40-42；
attention.py L375-386 按层号消费、else 分支 0=纯 SWA；L399-401
is_backbone 门控永不为 kv/index 源；后端按去重层型集合消费
sparse_swa.py L494-502）；长度域 {L, L+mtp}、尾部值 {0,1,2}、非零域外
拒绝（草稿前向未建模不猜语义）；入册 extras.kv_spec.draft_compress_
ratios。落地：count/moe moe_act/moe_shared_act 行（6 算术 op/元素；字节
0 单一所有者防重复计费）、formula v2.swiglu_clamp_act、model_ir
extras.swiglu + swiglu_limit 键 mapped、coverage note 更新；附带补漏
candidate_block_size 入 struct 集合（R20 注记因键漏登记未生效，官方资产
长期误挂"未登记 config 键"缺口，G9 断言被 text_config 前缀绕过未暴露）。
V4.1 config_coverage_complete 自此 True；structure_complete 仍 False（仅
稀疏 MLA 核本体阻断）。默认 TP8/B1/ctx32768 replay（quant=V41_QUANT）：
decode flops +483,840（40×(6·288·6+288·6)）、rows +80、全部字节字段与
M06 known_peak 不变、M01 +3.26e-07 ms（evidence/swiglu-tail
attribution.json 逐 op 归因，--check 无漂移复算）。新增 32 项独立 oracle
（tests/test_v41_swiglu.py：pin 三重校验/纯 stdlib clamp 数值 oracle（单
双侧不对称/边界/饱和/NaN/SWIGLUOAI 家族差异）/手算 TP1/TP8/EP8 绝对值/
负例 11（swiglu 域 6+尾部 5）/Qwen36 无 act 行/replay 冻结/公式-证据-
source_map 注册）。既有断言随事实更新（G9/G10/test_model_ir_v02/
test_v41_moe_route 结构缺口断言；test_v41_hc 冻结面追加 moe_act/
moe_shared_act 行名，各附因——合法门禁更新非放宽）。[报告]
(validation/v41-swiglu-tail-20261113.md)；G-C18 主干解除。剩余：clamp
kernel 数值等价/masked/pad-aware 运行时差异/SWIGLUOAI 家族/MegaMoE
SM100 执行变体、DSpark 草稿本体与非零尾部——登记不冒充。

**2026-11-13 R22：V4.1 量化映射（quantization_config → 布局/字节/执行域）**
（分支 `v41-quant-map-r22`）：quant-pin.json 20 文件（vllm @f358f057，
blob sha1 对 vllm-tree.json 逐项校验）source-checked 入册
（ev-vllm-v41-quant，executed=False；sm-v41-quant-dispatch）。语义核对
结论：quant_method=fp8 + model_type deepseek_v41* → override
deepseek_v4_fp8 → DeepseekV41FP8Config（config/model.py override 链 +
quantization/__init__.py L120-132 平台子类）；expert_dtype=fp4 由 config
类提升到 hf_config 顶层并派生 is_scale_e8m0；线性层（weight_block_size==
[32,32] 且 e8m0）→ ModelOptLinearMethod(kMxfp8Static, kMxfp8Dynamic)——
值 e4m3 1B/elem + ue8m0 uint8 (1,32) 块 scale（checkpoint 32×32 块装载
展开为 (N,K/32)；K 每 rank 整除 32 断言）；激活动态不落盘 scale；路由
专家 → Mxfp4MoEMethod(activation_key=kMxfp8Dynamic)——w13 (E,2I,H/2)
uint8 打包 + (E,2I,H/32) scale、w2 同构，分配前 Marlin 对齐 I→128/
H→256（routed_experts.py L190-203 于 create_weights 前调用）；embed/
lm_head/RMSNorm 未量化 BF16（get_quant_method 非
LinearBase/RoutedExperts/Attention → None → 默认方法）；执行域注册设备
（≤SM90）= Marlin W8A16/W4A16 → **激活 BF16**（FlashInfer MXFP8 线性核
与 TRTLLM/DeepGEMM 专家核均 SM100+；Humming cc=90 优先但依赖可选外部
包 → 部署变体未建模）；**scale_fmt 键不被 loader 消费**（Fp8Config.
from_config 键表无该键），有效 scale dtype 由 expert_dtype 派生——非
ue8m0 值按声明矛盾拒绝；KV 缓存无量化叶键（记录宽度仍 R20 v41_kv_record
配方）。落地：count/quant_layout 字节公式 + work/moe/head/hc/engram 行
recipe 计价（gemm 并 scale、激活 BF16、embed/lm_head/final_norm BF16、
final_norm 修 2·width·b_act 双重计宽）、memory scale/打包/对齐对象与
engram wkv scale、pipeline 集合通信载荷 BF16、model_ir extras.quant +
5 叶键 mapped（quantization_config 移出 gaps）、scenario V41_QUANT
（dense_w=8/expert_w=4/act=16）冻结校验、公式 v2.quant_*。默认 TP8/B1/
ctx32768 replay（quant=V41_QUANT）：weight_read 1,649,382,848→
1,769,999,808（+120,616,960 = scale + BF16 修正；final_norm −10,240）、
weight_read_expected 530,843,184→752,027,184（+221,184,000 = 40×6×
Δ921,600 专家打包+对齐）、flops/act 不变（act=16 两侧同 BF16）、M01
116.20089028→116.30292042ms（+0.10203014）、steady/known_peak
35,768,859,072/35,768,872,384→50,127,805,952/50,128,019,904
（+14,359,147,520；evidence/quant-map attribution.json 逐 op 归因，
--check 无漂移复算）。新增 31 项独立 oracle（tests/test_v41_quant.py：
pin 三重校验/纯 stdlib dispatch oracle/e8m0 小张量语义/手算绝对值/
负例 5+3/count↔memory 守恒/replay 冻结）。既有断言随事实更新（G9/G11/
model_ir/moe_route/hc/csa2/engram/pipeline/product_scope，各附因）；
gpu 表驻留下界 75.4 GiB<80 GiB 标称（MXFP4 打包后不再按已知下界判
OOM，判定回 unknown——预算缺口不变）。structure_complete 仍 False
（swiglu/注意力核本体/compress_ratios 尾部缺口不变）。[报告]
(validation/v41-quant-map-20261113.md)；G-C14 主干解除（执行变体登记）。

**2026-11-12 R21：V4.1 MoE scoring/top-k/分组路由**（PR #11 分支
`codex/v41-moe-route-r21`）：moe-pin.json 14 文件（vllm @f358f057，blob
sha1 对 vllm-tree.json 逐项校验）source-checked 入册（ev-vllm-v41-moe，
executed=False）。语义核对结论：sqrtsoftplus 打分（>20 护栏，dsv4 L85）+ noaux_tc
偏置**只导选择不进权重**（L86-98）+ k=6 轮**平坦**偏置 argmax（V4.1 无
n_group/topk_group 键，无分组路由；平手最小 id L94-95、NaN→−1e30 L87）
+ sum→1 归一 ×routed_scaling 1.5（sum≤0/NaN 时分母取 1，L104-107）；
GateLinear(h→E, out_dtype=fp32) Replicated（L270-277）；**权重沿
model loader BF16**（未传 quant_config/params_dtype；linear.py
L287-295/L179-199；base_loader.py L86；gate_linear.py L306-341）；**gate
输入由 pinned mHC layer_input_cur BF16 证明**（tilelang.py L900/L1096，
R05 pin；model.py L731-816 直接 self.ffn(x)），不能按任意 model dtype
泛化；有效域固定 official V4.1 config dtype=bfloat16，非 BF16 拒绝；
out_dtype=fp32 只是 logits 出口；共享专家（1,
inter 2304）**无 gate**、routed 归一后权重 1.0 **独立相加**
（moe_output.py L151-175；FusedMoEFactory 调用未传 n_shared_experts/
fuse_shared_experts，L447-464 融合槽未启用）；主干 num_hash_layers=0
（L155）；视觉 checkpoint bias_vl 哨兵判定（L63-76）；EP **保守整除域**
（仅 mega 路径 L357-362 断言；fused L419-421 的 start_idx 只赋值、未被
所有权消费，fused 通用余数映射未入 pin/未建模，不主张 fused 非法；
非整除组合 count/memory 双侧拒绝——与 qwen35 pin 余数规则分立）。
落地：count 三行 moe_gate/moe_score/
moe_topk_renorm（FP32 logits 出口修正旧通用行）、memory 路由偏置对象
40×3,072B、model_ir extras.moe_route + config 域校验（softmax/greedy/
norm=false 拒绝）、公式 v2.moe_route_*、source_map sm-v41-moe-route。
默认 TP8/B1/ctx32768 replay（gate BF16 修复后）：decode FLOPs
7,175,856,742→7,176,457,222（+600,480 = 40×(3,458 score + 11,554 topk)）、
weight_read 1,570,196,928→1,648,963,008（+78,766,080 = 40×(3,932,160−
1,966,080 gate) + 40×3,072 bias）、act_read 10,246,334→10,512,894
（+266,560 = 40×5,120 gate 输入 BF16 + 61,760 score）、act_write
6,282,096→6,330,096（+48,000）、rows 927→1,007、M01
116.160230→116.183836ms（+0.023606）、steady/known_peak
35,690,092,992/35,690,106,304→35,768,859,072/35,768,872,384
（+78,766,080；evidence/moe-route attribution.json 逐 op 归因，面外行
逐字段冻结）。新增 23 项独立 oracle
（tests/test_v41_moe_route.py：pin 双侧/纯 stdlib 语义 oracle/手算绝对值/
tp-ep 复制不变/EP 负例/count↔memory 对账/config 域负例/replay 冻结）。
G10/G16/test_model_ir_v02 的 csa2_moe_routing 断言改指 moe_deploy_variants
（主干解除，MegaMoE/DSpark/padding skip 变体仍 unverified）；
test_v41_hc replay 冻结面追加 MoE 行名。structure_complete 仍 False
（量化映射/swiglu clamp/注意力核本体/MTP 尾部/compress_ratios 尾部缺口
不变）。首轮 Codex 验收（不通过）后修正：全部行号对实销行号、oracle
−87.3 normal-range 近似标注、replay 工具补 --check 无漂移复算模式、PR
描述副本入仓（报告 §8）。第三轮复核（2026-11-13）修正证据链残留：
fused EP 的 start_idx 只赋值、未被所有权消费（删除"floor 覆盖性"
主张，EP 规则只按 mega 断言）；共享专家实际为独立相加（L447-464 融合槽
未启用）；ExpertMapManager 未入 pin 的默认值登记为读取点；证据包加
before/scenario SHA256 自校验（报告 §9）。
**2026-11-13 Gate 修复（数值变化，报告 §10）**：GateLinear 未传
quant_config/params_dtype → LinearBase 取 `torch.get_default_dtype()`、
`quant_config=None→UnquantizedLinearMethod`（linear.py L287-295/L179-199），
base_loader 在 `set_default_torch_dtype(model_config.dtype)` 内实例化
（base_loader.py L86）→ 官方 V4.1 dtype=bfloat16 时 gate 权重 2B。
**gate 输入 2B 由 pinned mHC tilelang.py L900/L1096 的
`layer_input_cur dtype=torch.bfloat16` + model.py L731-816 直接
`self.ffn(x)` 证明，不是由 model dtype 泛化**；有效域只覆盖 official
V4.1 bf16，full_config 的 float16/float32/缺失 dtype 拒绝，无 full_config
仅固定资产默认。每层 per-rank gate weight 1,966,080→3,932,160 B、输入
5,120→10,240 B/token；40 层 weight_read 少计 78,643,200 B（加 bias 后
Δ78,766,080），act_read 少计 204,800 B（加 score 后 Δ266,560）；
steady/known_peak 同增 78,766,080 B。新增 linear.py/base_loader.py/
layer.py 三个 pin（14 文件）、独立手算、“场景 quant 不变”负例、
dtype 域负例、before/after/attribution 重放。
[报告](validation/v41-moe-route-20261112.md)；G-C13 解除。

**2026-11-12 PR #5/#7 合入后验收**：首次全量 261 passed / 1 failed；
R06 fixture 的 pinned SHA256 记录为 CRLF 字节，仓内官方 pin manifest 为 LF 字节，
导致实时重放仅在相同换行检出环境可通过。已把 fixture/报告改为 LF 哈希，
`.gitattributes` 固定两份 pin 源码为 LF，并在 hermetic 测试中核对
fixture、manifest 与文件字节三者一致；同时修正 fixture `SourceIdentity`
的 `config_sha256` 为 fixture config 内容哈希。修后全量 262 passed，
`run-v2` 与 R05 mHC replay 运行成功；该修复不改生产计数。进一步计算审计
发现 R06 初版未提供独立输出数值 oracle，几何证据等级由 numeric-checked
降为 shape-checked；随后新增独立数值产物与对应证据（见下）。

**2026-11-12 R20 CSA2 索引打分/top-k/candidate block**：csa2-pin.json 21 文件
（blob SHA1 对 vllm-tree.json 逐项校验 + SHA256；补入 sparse_attn_indexer
直接 import 的 v1 backend，移除未入 tree 的重复 candidate_blocks 副本）
source-checked 入册
（ev-vllm-v41-csa2，executed=False；sm-v41-csa2-indexer + 短上下文旁路条目
重写）。修正/新建 count 行：index_select 拆为 index_score（全候选 paged MQA
logits 2·pairs·(ih·di+ih) + fp32 logits 写）+ index_topk（2·pairs+T·topk 选择
惯例）；candidate_publish → candidate_select（源层 L20：块最大值 NaN 传播/最新
块 +inf/−1 填充）+ 新增 candidate_mask（消费层 24/28/32/36）；短上下文旁路
（end//r≤512）index_fill 0 浮点纯 store（旧式误计 T·topk flops）；
compress_norm → compress_pool（r2 门控池化 7d+9/group 精确式 + 环形 state
cap8，r1 纯 norm exact）；compressed_rope+compressed_cache_write 合并为
compressed_insert；index_k_* 合并为 index_k_store（132B/行）；SWA/压缩缓存
改按打包记录 584|528 B/token（显式 v41_kv_record 配方，进场景身份；并与
目标设备 family 强校验：H100/H200 只允许 sm90，L40 拒绝 sm90，当前无
SM100 设备故 sm100/nvfp4 拒绝）；
compressor/wk/weights_proj 为 quant_config=None bf16 固定权重，输入/输出
激活字节亦按 pinned bf16/FP32 与消费行对齐；主 attention q_norm/kv_norm
bf16 常驻权重入 count/memory（40 层 143,360 B）；sparse_index_build 行
（decode 每 (kv源:索引源:r) 运行首层 combine 一次，128 对齐宽，等键后继
复用）；topk/candidate 共享缓冲模型级单例下界入 transient。
默认 TP8/B1/ctx32768 replay：op 7,161,190,222→7,175,856,742（Δ 归因
attribution.json 逐 op 验证，审计面外行逐字段冻结）；steady +26,752,000B、
transient +10,240B、known_peak +26,762,240B、M01 +0.0075ms。
structure_complete 仍 False
（MoE scoring/量化映射/注意力核本体/MTP 尾部缺口不变）。新增 31 项独立
oracle（tests/test_v41_csa2.py）。[报告](validation/v41-csa2-indexer-20261112.md)。

**2026-11-12 R06 独立数值续验**：新增纯张量参考路径，复用 pinned HF
随机权重但不调用被验证模块的 forward；prefill 与带缓存 decode 各对比
12 个张量（GDN/attention/门控/MoE/层输出/logits/三类缓存），逐元素容限
`atol=1e-5, rtol=1e-4`，本机最大绝对差 2.98e-7。
证据 `ev-qwen36-cpu-numeric-r06` 独立绑定数值产物并标 numeric-checked；
原几何/算量产物保持 shape-checked。RoPE cos/sin 为输入边界（mrope 位置
生成已于 2026-11-16 R26 独立补齐核验，见下节）；真实 checkpoint/MTP/GPU
未验证；模型级 cpu_forward 仍 not-run。
详见 [R06 报告 §9](validation/qwen36-gdn-moe-cpu-fixture-r06.md)。

**2026-11-12 R05 mHC/Sinkhorn**：14 文件完整 blob SHA1 对 pinned tree、SHA256
清单入册，新增 sm-v41-mhc-kernels/ev-vllm-v41-mhc（source-checked，executed=False）。
补齐 prenorm/sqrsum、comb post；Sinkhorn 仅 m×m，m4/iters20=1400 op/token；
首层无 post、expand为拷贝零 FLOPs；末层出口 post+collapse两行（layer=-1），
Triton零累加collapse=2mh，数学最简(2m−1)h另列；norm仍保留4h+2近似惯例。
L0新增491,520B broadcast派生缓冲，非参数；原 Engram 行/函数/公式未动。
默认TP8/B1/ctx32768 replay：op 7,146,532,702→7,161,190,222，权重读
−1,465,920B、激活读+1,672,880B，GPU解析下界116.152651919→116.152715012ms；
known_peak +491,520B，预算/峰值完整性unknown不变。新增16项独立 oracle/回归，
G7/G11语义未改。未执行kernel/CPU forward/GPU，部署变体、实际split与物理HBM
仍未建模。[报告与任务书差异](validation/v41-mhc-kernels-20261112.md)。

**A 组二轮审计补漏（2026-11-10，R01/R02a/R02b）**：对 pinned vLLM @dc74b8cb
逐文件复核发现并修复 4 处漏计（证据 `ev-vllm-logits-ctxkv-billing`，
`logits_processor.py` 补入 pin，SHA256 e1fac869…）：
① lm_head 词表行切后 logits 集合通信（logits_processor.py L185-198
`lm_head.tp_size>1 → gather`）此前完全未计——decode/verify/草稿候选各补 1 次
`logits_gather`（每 rank (tp−1)/tp×T×V×act 字节，单向环 (tp−1) 段延迟）；
② DFlash 草稿 `draft_lm_head` weight_read 原 0（dflash model.py L260/L599-601
每轮独立 GEMM）→ 补 h×(V/tp)×b_w；③ `ctx_kv_proj` 二次读（L384-387 k_ctx/
k_noise 两次 GEMM）→ 补按路径计的权重读；④ vLLM `_fused_kv_weight`
全层 K/V 行 cat 驻留副本（qwen3_dflash.py L541-568）入 R02a 账本
（仅显式 `fused_raw` 路径）。结构化临时补 ctx 归一副本/ctx 投影输出
（phase:draft）与 verify aux hidden（phase:verify）；M06 formula_ref 修正为
`v2.capacity`。默认场景前后归因见
[A 组二轮审计报告](validation/agroup-lmhead-ctxkv-audit-20261110.md)（M01
baseline 112.2127→112.9161ms 等 8 项逐项归因；同卡 memory status 仍 unknown，
两缺口不变）。

**A 组三轮 ctx-KV 路径修正（2026-11-11，R01/R02a）**：外层验收前修正两处
确认缺陷。① `draft_extra_allocations` 原以 `dense_w≥8` 推断
`UnquantizedLinearMethod`/`_fused_kv_weight` 副本——错误；pinned
qwen3_dflash.py L108-113/529-568/679-697 的分支由 quant_method/平台决定，
默认 dense_w=8 只是 fp8 计价档。现新增显式 `draft_ctx_kv_path`
（`fused_raw` / `mxfp8_context_proj` / `per_layer_fallback`）：raw 才建
融合 cat 副本，MXFP8 建独立 `_context_kv_proj`（packed 8bit 下界，
block-scale 未入 pin→unknown），回退无副本；`_k_norm_weights` 的 L×d
堆叠在三条路径都单列（4bit 回退此前漏计）。② `ctx_kv_proj` 对回退路径
原只计 K/V；pinned L688-697 回退执行完整 qkv_proj 再切 Q。TP8/B1/E[N]=3/
h2048/Q4 头 per rank/KV1/d128 时每层逻辑 FLOPs 由 3,145,728 修正为
**9,437,184**，权重读含 Q 行；回退临时缓冲补逐层完整 qkv（含 Q）暂存。
`draft_ctx_kv_path` 为场景显式输入，缺失/非法/`4bit+fused_raw` 均拒绝，
不由位宽猜分支。默认场景（活跃草稿 config 无 `quantization_config` +
GPU）显式取 `fused_raw`；这是**研究假设/场景配方**（config source-checked、
未执行），不证明实际 vLLM 以 8bit unquantized 部署，`dense_w=8` 只是
计价位宽。对象拆为 `draft-w-dflash-fused-ctx-kv` 与
`draft-w-dflash-k-norm-stack`。
**复核修正**：k_norm stack 改用 `DraftIR.dtype_hint`（config
`dtype=bfloat16`→2B/elem，6×128×2=1,536 B），与 `dense_w` 4/8/16 无关；
dtype 缺失/不支持显式拒绝，不回退位宽。默认 steady/known 因此各 +768 B。
独立手算/负例见 `tests/test_draft_ctx_kv_paths.py`（11 项），报告
§6 见 [A 组二轮审计报告](validation/agroup-lmhead-ctxkv-audit-20261110.md)。
外层 Codex 已全量验收：212 passed，exit 0；远端 CI 以候选 PR 检查为准。

**2026-11-11 R04：V4.1 Engram 建模（B 组 R04 交付）**：上游源码由主控
pin 入仓（vLLM @f358f057；common/engram.py blob sha1 ba4ceea6 + nvidia
ops engram_* 三文件 + amd/engram.py + configs/deepseek_v41.py，blob sha1
逐项对 references/vllm-tree.json 校验）。落地：engram 8 键 → mapped
（extras.engram spec；264 B/行 = FP8 e4m3 256 + ue8m0 8，n_hash_cols=24，
桶分配 L211-252 忠实移植 families.engram_bucket_primes）；count 四行
（engram_hash/lookup/wkv/post_wkv，层 1/14，批级哈希一次挂首层行）；
memory 三对象/层 + staged 下界（表默认 **cpu_offload 研究假设**，场景
显式 engram_table_placement 输入，host 驻留不进 GPU 峰值单列
host_resident_bytes；显式 gpu → TP8 合计超标称如实 OOM）；TP>1 补
head/wkv 两次 all-gather 通信行。独立 oracle：tests/test_v41_engram.py
24 项（自写 Miller-Rabin 层桶和 == engram_num_embeddings 精确相等
384,006,168/384,016,682、首 8 素数硬编码、TP1 表 202,758,032,400 B、
tp=7/9/16 拒绝、逐行手算绝对值、开/关账本差=精确新增、count↔memory
对账、blob sha1 防漂移）。门禁随事实更新（G9/G10/test_model_ir_v02
断言归因见报告 §8，剩余缺口不变仍在阻断）。默认 V4.1 replay 前后
逐项归因：decode FLOPs 7,067,478,214→7,146,532,702、GPU 侧解析耗时下界
113.34→116.15ms（host 查表/预取时延未知）、GPU steady +39,485,440B（仅 wkv+qk）、host 表
25,344,443,520B、行 909→916；structure_complete 当时仍 False（CSA2/MoE/
量化缺口；其中 CSA2 索引打分/top-k/candidate 后由 R20 2026-11-12 解除，
MoE/量化等仍在阻断）。报告：[docs/validation/v41-engram-20261111.md]
(validation/v41-engram-20261111.md)；证据 ev-vllm-v41-engram；G-C1
单机算子解除（DP 交换/cpu_offload 运行时/amd 变体仍登记未覆盖）。

**2026-11-12 R06：Qwen3.6 GDN+MoE 小尺寸 CPU forward 对照（D 组 R06）**：
探针 `tools/probe_qwen36_forward_r06.py`（独立于本仓，不 import simverse）
以 pin 官方 forward（transformers @b6401d66，torch 2.15.0+cpu，
seed 20261112）执行 h=32/2 层（GDN+FullAttn）/4E top-2 随机权重 fixture，
prefill T=5 + decode T=1，落盘 `tests/fixtures/qwen36_r06_cpu_fixture.json`
（逐 GEMM/conv/递推形状/因果对/MoE assignment 全录）。对账
`tests/test_qwen36_forward_r06.py` 10 项（hermetic 9 + live 1）：本仓
`parse_qwen3_5_moe(同一 config)` → `build_model_work(tp=1)` 逐算子**精确
相等**（对账表见报告 §4）。**修正一处漏计**：full-attn 输出门控逐元素行
（官方 L802 sigmoid(gate)·attn_out）补 `attn_output_gate` 行 + 公式
`v2.attn_output_gate` + source_map note——默认场景前后：decode +10,240
FLOPs（10 层×1024）、verify +163,840、draft 不变、M01 +0.00002ms（报告
§5）。披露不修：gdn_recurrence 4 ops/elem（官方参考≈7）；HF DynamicCache
conv state 宽 = kernel 列（本仓按 vLLM kernel−1 计容量，差 1 列）。
证据 `ev-qwen36-cpu-fixture-r06`（shape-checked）入 registry 并挂
qwen36 evidence_refs；**模型级 cpu_forward 仍 not-run**，部署排名门禁
不变。报告：[docs/validation/qwen36-gdn-moe-cpu-fixture-r06.md]
(validation/qwen36-gdn-moe-cpu-fixture-r06.md)。

**CI hermetic 修复（2026-11-10）**：`tests/test_consistency.py` 两项验收不再读取
被 `.gitignore` 忽略的 `outputs/run_v41.json`。CSV 导出改为复核本次 `run` 新落盘的
RunResult；固定场景重放改用纳入版本库、带 provenance 的历史快照
`tests/fixtures/v41_flash_dt_verification_expected.json`。远端 run `42803015653`
的 `2 failed, 194 passed` 是 clean checkout 缺文件；此前本地 `196 passed` 属环境
假阳性。详见 [Windows CI hermetic 修复](validation/windows-ci-hermetic-fix-20261110.md)。
该轮测试总数仍为 196（后续编码修复新增 1 项后为 197）。

**CLI 导出编码修复（2026-11-10 后续）**：远端 run `42805472945`（PR）与
`42805463611`（push）的唯一失败是 `test_csv_export_matches_runresult`：
Windows 管道 stdio 使用 ANSI 代码页 cp1252，`simverse.cli export` 在写文件
成功后 `print("已写 ...")` 抛 `UnicodeEncodeError`，进程 exit 1。CLI 现在在
`main()` 统一把 stdout/stderr 设为 UTF-8 + `backslashreplace`；测试显式固定
子进程 stdio 编码，并新增 cp1252 回归用例。收集数 196→197，详见
[Windows CI hermetic 修复 §6](validation/windows-ci-hermetic-fix-20261110.md)。

**外层本地验收（2026-11-10，hermetic 修复后）**：外层 Codex 唯一一次全量
`py -3.12 -m pytest tests -q` 实际 exit 0：`196 passed, 1 warning in 5.04s`；
警告为既有的 Windows 子进程 UTF-8 解码线程（`test_csv_export_matches_runresult`）。
该结果只证明本地工作区（含尚未提交的修复）通过，不是 clean checkout 或远端 CI
通过证据。修复文件尚未提交/推送；PR #1 现有检查仍是修复前失败记录，远端重跑
由外层执行。

## 0. 2026-11-10 两模型门禁轮记录

- **活跃目录恰两款**：`deepseek_v41_flash`、`qwen36_35b_a3b`，加绑定 Qwen3.6 的
  `dflash_qwen36_35b_a3b` DraftIR。`qwen3_30b_a3b`、`qwen3_235b_a22b`、
  `qwen3_next_80b_a3b`、`deepseek_v3` 移出运行时注册；tiny 只作内部测试 fixture。
  历史模型/场景/profile 保留为离线归档，API/CLI/工作台不可选择。
- **证据分栏**：`ModelIR.evidence` 记录 forward 源码、CPU 执行、GPU 实测三栏；
  `config_coverage_complete`、`forward_evidence_complete`、`structure_complete`
  分开。完整 config（含嵌套叶键）逐项登记，未知键不得默认 ignored。
  - Qwen3.6：config 覆盖完整；GDN/full-attn gate/MoE router+shared gate/TP-EP
    分片 source-checked；R06（2026-11-12）fixture 级小尺寸 CPU forward
    形状与算量对照已执行（ev-qwen36-cpu-fixture-r06，GDN+MoE+full-attn 载体逐
    算子精确对账）；MTP 只有 config+目标首层模板 → **unverified**
    （**R23 已解除**：2026-11-13 双栈 pin source-checked，见文首 R23 段）；
    forward 仍不完整、模型级 cpu_forward 仍 not-run，不能参与部署排名。
  - V4.1：Engram 单副本路径已 source-checked 建模（R04，2026-11-11；
    DP 分片交换/cpu_offload 运行时行为/amd 变体未覆盖）；CSA2 索引打分/
    top-k/candidate block 已由 R20（2026-11-12，ev-vllm-v41-csa2）source-checked
    解除；MoE 路由（scoring/top-k/归一化/缩放/共享专家/EP 整除）已由 R21
    （2026-11-12，ev-vllm-v41-moe）source-checked 解除（部署变体
    MegaMoE/DSpark 草稿 MoE/padding skip 仍 unverified）；量化映射
    （quantization_config 5 叶键 → MXFP8 线性/MXFP4 专家/scale/打包/
    Marlin 执行域 BF16 激活）已由 R22（2026-11-13，ev-vllm-v41-quant）
    source-checked 解除（Humming/SM100/emulation 执行变体仍 unverified）；
    swiglu、稀疏 MLA 注意力核本体、compress_ratios 尾部仍为缺口；mHC 仅
    source-checked；CPU/GPU 均未执行。
  - DFlash：结构/轮协议/同卡共享 source-checked；服务端集成已双栈逐行核对
    （R07a 2026-11-12：colocated=source-checked/dt=unsupported）；served
    已部署实测未做；拒绝采样/KV 回退/hidden 提取 kernel 耗时仍 0 占位
    unknown（vLLM prepare-inputs 为图外 eager Triton）；CPU/GPU 未执行。
- **本轮修复**（2026-11-12 R07a）：ctx 投影/act 改 verify 跨度基（含 ctx K
  norm/rope、vLLM permute+contiguous 布局拷贝）、草稿行改生效块
  plan.block_size、full 草稿层按 pinned layer_types 非因果；capability 分模式
  （colocated=supported source-checked / dt=unsupported）。详见
  [R07a 报告](validation/dflash-serving-audit-20261112.md)。
- **本轮修复**（2026-11-12 R06）：full-attn 输出门控逐元素行
  `attn_output_gate`（官方 L802 sigmoid(gate)·attn_out；此前只计 q_proj
  2× GEMM 消费）——默认场景 decode +10,240 FLOPs、verify +163,840、
  draft 不变（归因见 R06 报告 §5）。
  早前（2026-11-10 两模型轮）：V4.1 `o_a_proj` 分组在 work/memory 同步乘
  `ng`，并补 V4.1 attention memory 的 `b_dense` 单位（BF16 曾少 2×）；
  Qwen3.6 shared gate 在 IR/work/memory 计入；DFlash/MTP/chain 草稿 B>1
  token 维乘一次；证据引用按 model/draft/profile 生成；source_map
  行号/未核状态校正。
- **当前数值**：CLI 内置默认 Qwen3.6+DFlash（显式 `draft_ctx_kv_path=fused_raw`，α0.72、blk16）稳态
  2,830,652,352 B（shared gate +81,920 B；含 fused ctx-KV 3,145,728 B 与
  k_norm 堆叠 1,536 B），known_peak 2,832,959,265 B（下界；含 ctx flat/
  permute contiguous/K norm 输出临时，`t-draft-buffers` 取已知并存组 max
  而非全缓冲相加；草稿 logits 按 blk−1 候选计），仍不完整；
  审计对照场景（α0.75、blk5）的 known_peak 为 2,831,405,056 B；
  V4.1 o_a TP1 每层 67,108,864 FLOPs/容量 B，40 层约 +2.35 GFLOP/token
  +2.52 GB(BF16)。详细归因/来源见
  [两模型审查记录](validation/two-model-forward-merge-gate-20261110.md)；
  ctx-KV 路径修正见 [A 组二轮审计 §6](validation/agroup-lmhead-ctxkv-audit-20261110.md)。
- **合入门禁**：G1–G17；`.github/workflows/merge-gate.yml` 覆盖
  PR/push/merge_group，Windows + Node 20 + Python 3.12。2026-11-11 复核：
  main branch protection 要求经 PR 合入及 `pytest (Windows, Python 3.12)`
  必需检查（strict），审批数为 0，且 `enforce_admins=true`。
- **未完成**：V4.1 稀疏 MLA 注意力核本体、swiglu clamp、
  Engram DP 分片交换路径、Qwen3.6 MTP 原生 forward（R23 已解除主干，
  剩余执行/变体缺口见 G-C15）、
  两款主模型 CPU/GPU 实测、DFlash served 实测与运行时池尺寸、真实可分配预算。
  （量化映射主干 R22 已 source-checked；Humming/SM100/emulation 执行变体
  仍缺（G-C14）。MoE 路由主干 R21 已 source-checked；MegaMoE/DSpark 草稿
  MoE/padding skip 等部署变体仍缺。）

## 0.1 必读上下文

- 计划包：`docs/rebuild-20261106/`（EXECUTION_PROMPT.md 为本任务指令；
  questions.md Q1–Q12 已答；plan/principles/ui-data-contract 为验收依据）。
- v0.1 切片（旧 13 测试）保留于扁平模块（engine.py/ledger.py/...），
  contracts.py→contracts/legacy.py、cost.py→cost/legacy.py（同路径迁移，
  旧导入兼容）。v0.2 管线为 contracts/count/methods/lowering/cost/execute/
  experiments/pipeline/scenario 包。两套由 schema_version 0.1/0.2 区分，
  M5 退役 v0.1。

## 1. 里程碑状态

> 以下 M0–M5 段落保留历史日期与事实；其中“结构完备/90B/235B 活跃/180 passed”
> 等表述已被 §0 的两模型轮取代。当前合入判断以 §0、MERGE_CRITERIA 与门禁测试为准。



### M0 证据冻结与红灯 — 完成
- git init + 基线提交（5913cc5 起）；.gitignore 排除嵌套仓 references/model-verse-source。
- contracts v0.2：enums（support/evidence/feasibility/result_kind/statistic/coverage/time_basis 正交）、
  identity（完整 SHA256、js_safe 大整数往返、RngScheme）、Metric（null+reason 强制、
  exact 拒绝非整浮点）、RunManifest（run_id=身份哈希）、ResultBundle、CapabilityReport。
- evidence registry（simverse/validation/registry.py）：内容 hash 实算绑定；
  V4.1 config、Qwen3.6 主/草稿 config、dflash 实现/README、vLLM V4.1 源、旧原理 pin。
- 独立 golden（golden/oracles_v02.py，不 import 生产）：MoE 枚举 E4k2M2=3、
  E384k6M5≈29.077034、有放回反例、接受端点/mean 反例/动态 K 联合律反例（2.06/1.98）、
  人工轨迹（TPOT/吞吐/goodput/busy/分位）、roofline/链路手算锚点。
- 红灯测试 tests/test_negative_v02.py：16 项（期望 int 截断、NaN、截断 digest、
  大整数、mean→Pfull、E09 残差等各自有断言捕获）。
- 证据：tests 68 passed（含 v0.1 13 项原样保留）。

### M1 语义与计数 — 完成（Qwen3.8 本地身份待补）

**2026-11-10 增量**：V4.1 mHC 混合流建模（hc 三键 mapped；count hc_* 边界行
6/层+首层 expand、FP32 权重每卡复制入 memory；Engram 8 键保留缺口——
common/engram.py 未入评审包，不猜；tests/test_v41_hc.py 7 项含守恒）；
Qwen3-235B-A22B 官方身份入册（HF main @8efa6172，config SHA256 0ecd5d6f…）
+ catalog qwen3_235b_a22b（qwen3_moe 家族，94L 128E）。
- model_ir 0.3.0 共享子包（schema/families/catalog）：结构、SourceIdentity、
  字段覆盖（mapped/metadata-only/ignored-with-reason/unsupported-semantic）、
  structure_complete+gaps。
- 家族与锚点：gqa（tiny）、qwen3_5_moe（Qwen3.5/3.6 混合 GDN+GQA）、qwen3_next、
  qwen3_moe、deepseek_v3（MLA+Dense/MoE 分层）、v41_csa2（压缩层计划+Engram/hc 11 缺口）、
  DFlash DraftIR。
- 身份 pin（assets/config + assets/identity，SHA256 全录）：
  - Qwen/Qwen3.6-35B-A3B @345d467b（必需②主模型，qwen3_5_moe，40 层=30 GDN+10 Full@3,7..39，
    256E top-8+shared512+gate，MTP1，mrope_section/interleaved 文本域 mapped（R26））
  - z-lab/Qwen3.6-35B-A3B-DFlash @f181eece（必需②草稿：6 层 5×SWA(4096)+1×Full GQA dense、
    block16、mask_token 248077、target_layer_ids [1,6,11,16,22,27,32,37]、fc 8·2048→2048）
  - z-lab/dflash @07ebd93（方法实现 pin：dflash_generate 协议/Qwen3DFlashAttention/
    拒绝采样；HF checkpoint 内无 dflash.py，auto_map 解析到服务栈）
  - Qwen3-30B-A3B @ad44e777、DeepSeek-V3 @e815299、Qwen3-Next（model-verse 本地快照，
    官方仓 gated→身份 PENDING）、transformers main @b6401d66 的 qwen3_5_moe modeling/configuration
- source map（importers/source_map.py）：7 条（GDN decode 分支 L558/607、q_proj 输出门控、
  shared gate、V4.1 短/长索引分支、Engram/hc 缺口、DFlash 轮协议、草稿 cross-attn），
  均带谓词/域/两侧覆盖声明。
- count 层（count/work.py + moe.py + draft.py）：GQA/SLIDING/GDN/MLA/CSA2 家族逐层行；
  E09 修复（残差 flops）、E05 修复（期望并集单列 float）、GDN 常数状态单列；
  MoE 正确式 E·(1−(1−k/E)^M)（有放回仅显式近似选项）。
- 测试 tests/test_model_ir_v02.py：19 项（层序回归 Qwen3-Next idx3,7..47=12 Full、
  V3 61=3Dense+58MoE、V4.1 r 分布 2/18/20+11 缺口、逐键覆盖、tiny 手算全量对账、
  MoE 期望/精确分列、MLA 压缩 KV、激活参数量级）。

### M2 串行实验闭环 — 完成
- methods：AcceptanceLaw（position_alphas/geometric/committed_mean[显式假设]/
  dynamic_k[缺律→unknown]/empirical_hist[unsupported]）；E07 修复（mean 不反解，
  假设显式才 approximated）；MethodPlan 注册（baseline/chain_draft/mtp/
  chain_moe_proxy[改名假想方法]/dflash_v1[真实语义绑定 pin]）；dspark/dflash2 显式 unsupported。
- lowering：TP 合法校验（不盲测非法组合）、KV 分片/复制登记、ring allreduce 每层 2 次
  （E03 修复：TP 通信入账）、EP dispatch/combine 期望（均匀假设登记）、
  容量账本（唯一存活对象求和、shared_identity 去重、标称/可分配分层→feasible=unknown §7.4）。
- cost：T0（行级 roofline + 精度路由；FP4 不自动升档 E10 口径保留）；
  profile 契约（ProfileRecord/CalibrationModel/CostEstimate；合成 fixture 接线，
  域外回退解析、synthetic 不升级，Q2 全 unmeasured）。
- execute：串行轮事件（event_id/depends_on/stage/role/start）+ SerialRound；
  TPOT=E[T_round]/E[N]。
- pipeline.run_v2：ScenarioV2→ResultBundle 全链；结构缺口/容量 unknown/接受缺律
  按指标传播（数值保留+不排名）；V4.1 Engram/hc→structure gate。
- experiments：不可变快照扫描（E20 修复：不触全局注册表，测试钉死）、
  每点独立 run_id、breakeven 无交点返回 null 不造数；
  ComparisonPolicy（Q4 双资源政策、Q11 多目标并列、资格谓词、资源 delta 披露）。
- 测试 tests/test_pipeline_v02.py：18 项。
- M2 初版历史数字（Qwen3.6+DFlash colocated 8×H100 B1 ctx32k α0.72 blk16）：
  基线 TPOT 3.4704ms、投机 1.6874ms、加速比 2.0567。后续 A 组修复了逐层 TP 通信，
  本轮再补 DFlash 草稿通信；**当前值见下方投机耗时复核**，此处不得作当前性能结论。

### UI（M1/M2 区域）— 完成（生产工作台）
- simverse/static/workbench.html+js：复用 v3 DOM/CSS 骨架（场景栏/7 KPI/矩阵/
  算子账本/七维检查器/Roofline/三轨时间轴/证据抽屉）；全部数字来自 /api/v2/run；
  无 v3-workbench.js mock 数学进生产路径（原型仍在 docs/.../prototype/ 独立存档）。
- webapp：/（生产）、/legacy（v0.1）、/api/v2/{catalog,run,sweep}；js_safe 大整数。
- 浏览器实测（IAB）：目录驱动选择、运行（run_id/代码版本显示）、KPI 7 项填充、
  572 行账本→24 聚合节点、检查器+后端 Roofline 坐标、5 事件三轨时间轴、
  mean-only→“未知”+反例原因、批量矩阵 9 格独立 run、窄屏 390px 无整页溢出。
  截图：outputs/workbench_desktop_run.png、workbench_mobile_run.png。
  注：本会话模型通道无图像输入，像素级观感未复核；以程序化布局检查+截图存证
  （与 v0.1 报告 §5.1 同口径如实声明）。
- CLI v0.2：run-v2/sweep-v2/export-v2（JSON/CSV/trace/HTML 四类导出）。
- V12：tests/test_ui_v12.py（fresh run→Web=CLI=JSON 逐字段、CSV/trace 逐行、
  HTML 身份、stale 不覆盖）。

### M3 资源事件核 — 核心完成（流水调度已入产品；请求级超前未做）
- [x] 指标归约器（execute/reduce.py）：请求口径 ITL（同刻=0 合法/N≤1=None）、
  窗口 token 事件 vs 完成请求两口径、goodput(SLO，取消单列)、busy 并集积分、
  最近秩分位；人工轨迹独立 oracle 全对账（tests/test_metrics_v08.py，V08）。
- [x] 事件验证器（execute/validate.py）：missing_dependency/
  starts_before_dependency_end/resource_overlap/consume_unknown_message/
  consume_before_send_end/missing_message_edge 六类检查，负例逐条捕获（V07）。
- [x] 资源约束事件核（execute/resource_core.py）：最早开始调度（互斥资源+
  半双工链路）、两微批 DT 流水与单请求串行、cohort 周期（中位差，避 warmup/
  drain）、资源 busy/利用率；事件流过 validate_trace（48 事件 0 违规实测）。
- [x] 产品接通：部署模式 dt_pipeline_sim（ScenarioV2.pipeline_microbatches/
  rounds）→ M18 周期/每轮等价吞吐/利用率指标 + M01 附"周期≠单请求延迟"注
  （分列不混用）；UI 部署下拉可选；浏览器实测通过（截图
  docs/evidence/workbench_pipeline_mode.png）。
- [x] 实测行为（ctx8k blk16）：target-gpu 利用率 99.2%、draft 10.2%（verify
  主导），cohort 周期 11.512ms、每轮等价 5.81ms < 串行轮和——收益来自具体
  调度而非 max 通则。
- [ ] 单请求超前（lookahead>0 的分支/取消/浪费资源账）与请求级事件轨迹
  （M17 TTFT 等）未做；取消/浪费字段在事件 schema 已留。

### M4 服务与校准 — 两条算子级实测校准闭环（服务特性仍未开放）
- 服务特性 UNSUPPORTED 注册：chunked_prefill/pd_disaggregation/prefix_cache
  （methods/plans.py UNSUPPORTED 表，Q5 顺序；不空实现）。
- **H200 算子级校准闭环 loop1（2026-11-09 建，2026-11-10 复审修复）**：pi-viewer
  @c5909a7 的 Qwen3-30B-A3B-tiny CUDA_EVENT 数据蒸馏入册
  （assets/profiles/h200_qwen3_30b_a3b_tiny/；tools/distill_h200_profiles.py
  可重放）。生产管线消费：ScenarioV2.cost_profile → CalibratedPricer（域门禁：
  设备/精度/TP/形状+算子构成三项核对；MoE 复制品 16E 域外回退；verify 注意力
  Q>1 域外回退）；指标 M01_tpot_profile_calibrated（result_kind=
  measured-composite，非整步实测）；UI 部署栏可选 profile。
  **2026-11-10 复审修复**：①gap 补贴改按**计时区间边界**计（原按全部 kernel 数
  计费重复计入已含在 CUDA_EVENT 括号内的 op 内间隙，B8 高估 0.324ms/步≈1.6%；
  修复后 B8 selected 19.879ms，9 点比值 1.95–11.59×——表述为相对组合估计的差异，
  非部署延迟结论）；②形状门禁补 QKNorm/全旋转/无滑窗构成核对；③时间口径覆盖
  （coverage_time_frac，依赖估计）随 stage 披露；④测量契约 caveats 入 manifest
  （launch-bound 括号内 host 空闲/warm-L2/重复数/gap 口径/even 假设）。
  报告：[docs/calibration/h200-loop1.md](calibration/h200-loop1.md)；
  重放不变性（G7）与组合算术独立复算入 tests/test_calibration_h200.py。
- **H200 loop2：Qwen3-235B-A22B TP∈{2,8}（2026-11-10 建）**：真模型官方身份
  （HF main @8efa6172）+ TP 集合域（首条 TP>1 实测覆盖；sharded 算子 per-worker
  形状=新仓 tp 切分行口径；TP1 采的 replicated 组以 tp_invariant 并入）；
  MoE 数据在册但 defer（per-rank 分片账本 G-C11 前置，不混尺）。场景
  assets/scenarios/h200_calib_loop2_235b.json（B8/TP8：selected 68.48ms，
  measured 29.13+fallback 38.30+gap 1.05×847 区间；行覆盖 1129/1318）。
  报告：[docs/calibration/h200-loop2-235b.md](calibration/h200-loop2-235b.md)；
  tests/test_calibration_h200_235b.py（8 项）。
- **旧仓数据全量审计（2026-11-10）**：[docs/calibration/oldrepo-data-audit-20261110.md]
  (calibration/oldrepo-data-audit-20261110.md)——A800 真 30B moe.csv 因缺环境
  元数据拒绝 measured 准入（§3.2/G4）；E2E 记录与 KERNEL_ONLY per-op 时间列
  确认不存在（G-D5/G-D6 维持）。
- UI 精度选择器（G-C10 解除，2026-11-10）：场景栏 BF16/W8A8/W8A8+FP4 档；
  UI 场景 JSON → /api/v2/run 与进程内/CLI 同 run 同值（测试钉死）；域门禁仍按
  profile 采集精度独立核对（不把"profile 要 BF16"转成输入精度）。
- 数值口径修正（2026-11-09 轮）：devices L40 注释修正；h200_sxm 官方规格 dense
  口径；链路 source 方向声明；cost/profile.py 校准分支修复（禁回填 estimate_ms）。
- Chunked Prefill / PD 分离 / Prefix Cache：未实现，能力目录未注册为支持；
  扫描/比较路径遇及时显式 unsupported（方法 UNSUPPORTED 表）。
- 服务到达/组批/KV 分页：workload.arrival="fixed-batch"（M4 扩展点已留）。

### M5 迁移与扩展 — model-verse 适配器已建；pi-viewer 数据通道已通（蒸馏）
- model-verse 薄适配器：LLM-Viewer-main/simverse_adapter/（新增文件，未改旧仓
  既有文件）：哈希锁（adapter_lock.json，6 文件 SHA256）+ 固定 commit 8796607 +
  model_ir 0.3.0；自检报告（adapter_report）通过；漂移即拒绝。
- pi-viewer：工作区阻塞已解除——本轮以只读蒸馏方式消费其 H200 数据
  （tools/distill_h200_profiles.py，来源提交+哈希入册，无运行时路径依赖）；
  独立适配器（同 model-verse 模式）仍待做，属 V09b 三方对账范围。
- 远端：Q3 平台/账号/地址待补；未创建/推送任何远端。
- v0.1 模块退役：待 M5 三方对账后。

### 2026-11-10 A 组轮：Qwen3.6+DFlash TP/EP 分片与同卡可部署性（R01–R03b）

报告入口：[docs/validation/agroup-tp-ep-colocated-20261110.md]
(validation/agroup-tp-ep-colocated-20261110.md)（前后重放归因/支持矩阵/限制）。

- **R01 统一配方**：新增 `simverse/lowering/parallel.py`（ParallelPlan；
  vLLM @dc74b8cb pin 入册 assets/identity/vllm_qwen35_pin/，16 文件逐
  SHA256，证据 ev-vllm-parallel-recipe）。count/moe、count/work（GQA/GDN/
  head per-rank）、count/draft（同 TP 组草稿+补 q/k norm/rope/hidden_norm/
  final norm 缺行）、lowering/memory、lowering/sharding（按层通信：EP MoE
  层 dispatch/combine 替代 mlp allreduce）、pipeline 六处消费同一 plan。
  TP 集合通信漏计修复（整模型 2 次 → 每层 2 次）。场景字段 target_ep
  （∈{1,tp}，其余拒绝）入运行身份。source_map 新增 3 条（sm-vllm-*）。
- **R02a/b 同卡账本**：物理 allocation 身份 + shared_identity + consumer
  引用；草稿共享 target 词表行切 embed/lm_head（零字节引用，修复前幻影
  副本≈1.017GB+去重失配）；GDN/草稿 KV/attention 权重 per-rank（补乘输入
  宽 h）；草稿批=目标批；生命周期分列（resident/per-request/cross-round/
  phase:draft|verify 串行取 max）+ unknown_components（workspace/graph/
  NCCL 池，不填 0）→ peak_complete 标志；per-rank r0..r{tp-1} + 最坏 rank
  报告（EP 余数 rank 差异显式）。
- **R03a/b 预算与判定**：DeploymentSpec.target_ep/budget(BudgetInput:
  allocatable_bytes/kind(measured|hypothesis)/source/extra_overhead，
  gross 语义) ；DeviceBudget 分层；RankMemory.peak_vs_budget 四态
  （unknown/oom/feasible/feasible-conditional；已知下界>标称即 OOM）；
  compare/_eligibility 排除 conditional；M06 携带 known_peak/steady/完整性。
- **UI**：工作台 EP 选择 + 预算输入（hypothesis）；浏览器实测（EP 生效
  T_round 64.6↔112.6ms 可区分；预算进链路以原因文本证实；截图
  docs/evidence/workbench_ep_budget_run.png）。
- **数值归因**（默认场景前后）：TPOT 1.6874→31.70 / 基线 3.4704→112.21
  （集合通信按层计费修复主导，pcie5 假设延迟 1.4ms×80 次；NVLink 规格
  属 G-D8 未虚构）；显存 5,680,423,680（口径混乱单值）→ per-rank steady
  2,827,423,168 + 临时峰值 2,135,746（明细见报告 §3）。235B TP8 校准
  fallback 38.30→4.85ms（MoE per-rank 口径，断言随归因更新；defer 理由
  改指 G-D2）。
- 测试 132→176（parallel_sharding 19 + colocated_memory 11 +
  budget_feasibility 14 + UI/场景接线 2；既有全保，tiny oracle 补 final
  norm 项、tp 校验断言随检查项更名——均逐项归因）。

### 2026-11-09 数值可信度轮
- [AGENTS.md](../AGENTS.md)（合入入口）+ [docs/MERGE_CRITERIA.md](MERGE_CRITERIA.md)
  （准则规范）+ G1–G17 自动门禁（tests/test_merge_gate.py）。
- devices 口径修正 + h200_sxm；profile 校准分支修复；pipeline 接线实测 profile；
  H200 第一条算子级闭环（见 M4 节）。
- 缺口清单分立：[docs/gaps/data-gaps.md](gaps/data-gaps.md)（数据采集 G-D1–G-D9）、
  [docs/gaps/capability-gaps.md](gaps/capability-gaps.md)（工程能力 G-C1–G-C9）。
- 证据：docs/evidence/h200_calib_compare.json（9 点前后对比）。
- 测试 86→110（merge gate 10 + calibration loop 12 + 既有全保）。

### 2026-11-10 校准复审 + 235B 闭环 + hc 建模轮（详单）
- **loop1 复审（8 项核查全部落档）**：gap 重复计费修复（区间边界口径）；
  形状门禁构成核对（QKNorm/全旋转/无滑窗）；留出无泄漏复核（387/5184 键唯一）；
  measured-composite 标注复核；"低估 2–11.7×"改口径为"相对组合估计 1.95–11.59×"；
  覆盖率双披露（行+时间占比，后者注明依赖估计）；测量契约 caveats 入册。
  pi-viewer 采集源码逐文件核对（device_timer/linear_op_impl/attention_wrapper/
  kernel_gap main/allowlists）。
- **loop2 235B**：官方身份获取入册；TP 集合域 {2,8}；tp_invariant 组；MoE defer
  裁决；组合/门禁/重放 8 测试；场景与报告入册。
- **V4.1 hc**：语义+计数+容量+守恒（详见 M1 增量）；Engram 8 键如实保留。
- **G-C10**：UI 精度选择器接线+测试。
- **数据审计**：oldrepo-data-audit-20261110.md（A800 拒绝/E2E 与 KERNEL_ONLY
  不存在/235B moe 首选路径）。
- 工具：tools/h200_calib_compare.py（对比 JSON 可重放生成器）。
- 测试 110→129（hc 7 + 235b 8 + 精度接线 2 + 门禁/组合更新；既有全保）。

## 2. 页面/指标/验证映射（当前覆盖）

| 验证 | 状态 | 证据 |
|---|---|---|
| V01 身份与协议 | 通过 | test_negative_v02（digest 全长/大整数/NaN/null+reason）|
| V02 结构与 forward | 部分 | 层序/覆盖/源码 pin+分支映射（source_map）；Qwen 固定源码两层随机权重 CPU shape probe 已做，R06 又做 GDN+MoE fixture 级形状与算量对照（ev-qwen36-cpu-fixture-r06，逐算子精确对账），独立输出数值 oracle/真实 checkpoint/完整 forward 仍缺 |
| V03 独立数学 oracle | 通过 | golden/oracles_v02 + test_model_ir/test_negative |
| V04 lowering 与守恒 | 大部分 | per-rank 统一配方（A组 R01）+ 跨 rank 守恒/复制对象不求和断言（test_parallel_sharding）；EP all2all/按层通信配方 |
| V05 方法、协议与状态 | 部分 | 接受律/端点/未知传播/DFlash 计划绑定；事件级 transition trace 未做（M3）|
| V06 显存生命周期 | 部分 | 静态+每请求对象、同卡去重；生命周期交错峰值未做（M3）|
| V07 事件因果与资源 | 大部分 | 验证器六类检查+负例；资源调度核+两微批流水已接；单请求超前未做 |
| V08 系统指标归约 | 通过（人工轨迹）| execute/reduce + test_metrics_v08 |
| V09 比较与旧仓对账 | 部分 | 比较资格/多目标/资源披露（V09a）；旧仓三方对账（V09b）未做（M5）|
| V10 扫描与运行隔离 | 部分 | 不可变快照/独立身份/无交点 null；取消/恢复未做（单任务门控待做）|
| V11 校准与外部真实性 | 部分（算子级，两闭环）| H200 30B-tiny TP1 + 235B TP2/8：蒸馏/域门禁/回退/留出误差/重放不变性（test_calibration_h200{,_235b}）；E2E 对照无数据未宣称 |
| V12 UI 与导出一致性 | 通过 | test_ui_v12 + 浏览器实测记录（本文件 §1 UI）|

## 3. 已知缺口与外部依赖（不虚构补齐；完整清单见 docs/gaps/）

1. **V4.1**：Engram 单副本路径已建模（R04，2026-11-11，pin+source-checked；
   剩余：DP 分片交换/EngramBatch MXFP8/cpu_offload 运行时行为/amd 变体，
   G-C1）；CSA2 索引打分/top-k/candidate block 已由 R20（2026-11-12）
   source-checked 解除（G-C12）；MoE scoring/top-k/分组路由已由 R21
   （2026-11-12）source-checked 解除（G-C13；MegaMoE/DSpark 草稿 MoE/
   padding skip 部署变体仍缺）；quantization_config 5 叶键已由 R22 映射
   （新增未知键保留缺口）；SwiGLU clamp 与 compress_ratios 尾部已由 R24
   source-checked，稀疏 MLA 注意力核主干已由 R25 source-checked；剩余
   Engram DP 交换、SM100/specialized decode 等执行变体和 kernel 执行验证。
   mHC 仅 source-checked、TileLang kernel 未执行。G-C1/G-C17。
2. **Qwen3.6 MTP**：原生 forward 已由 R23 双栈源码核对，文本 MRoPE 位置
   语义由 R26 补齐；method status=source-checked。真实 checkpoint 的模型级
   CPU forward、GPU/部署实测和可分配预算仍缺，故不进入部署排名。
3. **DFlash**：服务端集成已双栈逐行核对（R07a 2026-11-12：vLLM @dc74b8cb +
   SGLang @74cb973d，colocated=supported(source-checked)/dt=unsupported；
   ctx 投影改 verify 跨度基（含 K norm/rope/layout copy）、草稿行改生效块、
   full 层非因果三修正已回放）。
   残留：拒绝采样/KV 回退/hidden 提取 kernel 耗时仍 0 占位 unknown（两栈
   GPU kernel 证据已注记；vLLM prepare-inputs 为图外 eager Triton）；
   DraftLogitsCache 账本、SGLang 采样可用性、GDN 态
   回退成本、served 实测待 R07b；workspace/graph/NCCL 池尺寸未知。G-C3 残留。
4. **设备可分配预算/价格**：无来源 → feasible=unknown、成本 null+reason。
   H200 两闭环绑定已归档的 30B-tiny/235B，不能跨模型使用。G-D7/G-D9。
5. **CPU/GPU 执行**：Qwen 固定 HF 源码两层随机权重 CPU shape probe 已执行；
   R06（2026-11-12）追加 GDN+MoE fixture 级形状与算量对照（h=32/2 层/4E 随机权重，
   逐 GEMM/conv/router 与本仓 tp=1 账本精确相等，ev-qwen36-cpu-fixture-r06）；
   两款真实主模型仍无完整 forward 或 GPU/部署实测，模型级 cpu_forward 保持
   not-run，不冒充模型级 shape/numeric-checked。
6. **历史归档**：30B/235B/Next/V3 与 tiny fixture 只可显式注入归档目录；
   `h200_calib_*` 场景标 archived。Qwen3.8 本地身份仍 PENDING_ASSET。
7. pi-viewer 适配器（V09b）：未建；数据通道经蒸馏已通（无需写旧仓）。
8. 远端 required check/PR：2026-11-11 已配置。历史事实：私有仓库时期（2026-11-10
   修复前）只读查询 main branch protection 与 rulesets 均返回 HTTP 403
   （需 GitHub Pro 或转为公开）。仓库现已公开；同日复查询返回 404
   `Branch not protected`、rulesets 为空，当时强制检查尚未启用。此后 main
   branch protection 曾要求 Windows pytest 和 1 位 approving review。
   2026-11-11 按仓库所有者决定调整为单人仓规则：保留 PR、Windows pytest
   required check（strict）和管理员保护；required approving reviews 为 0。
9. 移动端像素观感：截图存证但本会话通道无法目检；`tmp_path` 用例在本机
   DSH 沙箱目录权限下未能完成（setup PermissionError；环境差异，非代码失败）。
   2026-11-12 R07a：Codex 宿主全量 `280 passed`，exit 0；本节仅保留 DSH
   沙箱回执。历史事实：hermetic
   修复前外层本地 `196 passed` 因复用 ignored `outputs/run_v41.json`，不代表
   clean checkout；远端 run `42803015653` 的 2 项失败即该文件缺失。两项验收
   已按 hermetic 方式修复，外层 Codex 于 2026-11-10 验收本地全量 exit 0：
   `196 passed, 1 warning in 5.04s`。修复文件尚未提交/推送，远端重跑待外层执行
   （见 [Windows CI hermetic 修复](validation/windows-ci-hermetic-fix-20261110.md)）。

## 4. 恢复入口

- **一键**：双击仓库根目录 `启动工作台.bat`（起服务+开浏览器，可选端口参数；
  重复双击=只开浏览器；`停止工作台.bat` 停止）。已实测 冷启动/已在运行/停止 三路径。
- 测试：`py -3.12 -m pytest tests -q`
- 工作台使用 `py -3.12 -m simverse.cli serve`，首页为 `/`。
- 必需场景：`py -3.12 -m simverse.cli run-v2`（内置默认=Qwen3.6+DFlash colocated）
- v0.1 存档使用 `py -3.12 -m simverse.cli archive run`，传入 v0.1 场景文件。
- 证据产物（已入册 docs/evidence/）：workbench_desktop_run.png、
  workbench_mobile_run.png、run_v2_qwen36_dflash.json、ops_v2.csv、
  trace_v2.csv、report_v2.html（历史轮）；A 组轮：
  run_v2_qwen36_dflash_agroup.json、ops_v2_agroup.csv、
  trace_v2_agroup.csv、report_v2_agroup.html、workbench_ep_budget_run.png。
- 当前缺口与交付次序见 [待开发计划](../ROADMAP.md)。


## 2026-11-10 产品范围收敛

产品目录仅展示 Qwen3.6-35B-A3B 和 DeepSeek-V4.1-Flash。历史模型已移出运行时
catalog，只作离线归档回归/校准资产；tiny 只作测试 fixture。用户已确认 Qwen3.5
为笔误，沿用 Qwen3.6-35B-A3B 及其 DFlash 配对。
默认 Qwen3.6+DFlash 不排名同时受限于：forward 证据未完备（MTP 未核）、
真实主模型 CPU/GPU 未完整执行、可分配显存未知——不能简化成单一“OOM”或“结构通过”。DFlash
checkpoint 禁止用于未经核对的其他主模型；历史 profile 跨模型直接拒绝。
本轮详细功能状态见 [FOCUS_STATUS.md](FOCUS_STATUS.md) 与
[两模型审查记录](validation/two-model-forward-merge-gate-20261110.md)。
