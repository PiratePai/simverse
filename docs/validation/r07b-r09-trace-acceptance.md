# R07b/R09 端到端请求轨迹验收方案

基于 `main@7228e75` 的代码盘点。首轮交付目标是为同一组 Qwen3.6 文本请求，分别核对 baseline、DFlash 和原生 MTP 的服务端产出、阶段计时和资源边界。验收分为离线导入与归约、匹配环境下的服务采集。本文给出离线验收数据和上机条件。离线导入与 TTFT 已通过人工轨迹验收。实际服务采集和部署判定仍需匹配环境。

## 当前可复用的实现

| 实现 | 现有用途和边界 |
| --- | --- |
| [`execute/reduce.py`](https://github.com/PiratePai/simverse/blob/7228e7505273ae66b99ea66c5fabb94ebfd54a66/simverse/execute/reduce.py#L12-L133) 与 [V08 人工轨迹测试](https://github.com/PiratePai/simverse/blob/7228e7505273ae66b99ea66c5fabb94ebfd54a66/tests/test_metrics_v08.py#L26-L67) | 可复用窗口事件计数、请求平均 ITL、最近秩分位和 busy 区间并集。完成请求计数、吞吐与 goodput 共用窗口判定，手算样本已补跨窗回归。离线入口在请求层计算 TTFT 和组合 SLO，复用窗口、吞吐、平均 ITL 与分位数归约。 |
| [`execute/validate.py`](https://github.com/PiratePai/simverse/blob/7228e7505273ae66b99ea66c5fabb94ebfd54a66/simverse/execute/validate.py#L17-L79) 与 [`resource_core.py`](https://github.com/PiratePai/simverse/blob/7228e7505273ae66b99ea66c5fabb94ebfd54a66/simverse/execute/resource_core.py#L111-L146) | 事件核可核对依赖、资源互斥和消息顺序，主要输入为仿真事件。并发采集的区间只有在独占资源和时钟语义明确时才使用对应互斥检查。 |
| [`ResultBundle`](https://github.com/PiratePai/simverse/blob/7228e7505273ae66b99ea66c5fabb94ebfd54a66/simverse/contracts/run.py#L49-L93) 与 [trace CSV 导出](https://github.com/PiratePai/simverse/blob/7228e7505273ae66b99ea66c5fabb94ebfd54a66/simverse/exports.py#L67-L77) | 已有结果身份和仿真轮事件导出。现有 CSV 缺少请求到达、提交输出、失败状态及观测窗口，无法还原服务端指标。 |
| [`methods/acceptance.py`](https://github.com/PiratePai/simverse/blob/7228e7505273ae66b99ea66c5fabb94ebfd54a66/simverse/methods/acceptance.py#L78-L125) 与 [Ascend 蒸馏工具](https://github.com/PiratePai/simverse/blob/7228e7505273ae66b99ea66c5fabb94ebfd54a66/tools/distill_ascend_cost_records.py#L21-L44) | 接受率实现消费场景假设，采集尚未接入。蒸馏工具的源文件与哈希登记可作为参考，现有读入范围是局部算子与通信成本。 |

## 运行离线摘要

在仓库根目录运行验收样本。命令向标准输出写入 JSON，输入文件保留原样。

```powershell
python -m simverse.cli trace-summary tests/fixtures/r07b_r09_request_trace.json
```

样本输出完成吞吐 100 token/s、事件吞吐 175 token/s、TTFT P50 4 ms 和 P95 6 ms。TTFT 分位数按最后提交位于观测窗口内的成功请求计算，样本数为 2。单请求 TTFT 和 E2E 另列，无输出的请求标为未知。TTFT 与平均 ITL 的组合 SLO 仅在提供 `slo` 时计算。导入失败返回退出码 2，并在标准错误中报告问题字段。复验命令为 `python -m pytest -q tests/test_request_trace.py tests/test_metrics_v08.py tests/test_resource_core.py`。

## 已实现的 JSON 协议

输入使用 UTF-8 JSON，顶层 `schema_version` 为字符串 `1.0`。纳秒时间戳为本次运行同一单调时钟上的非负 64 位整数。

| 字段 | 类型和约束 |
| --- | --- |
| `manifest` | 对象。包含 `run_id`、`source_kind`、`method` 和 `workload_sha256`。支持 `synthetic` 和 `captured`，方法为 `baseline`、`dflash_v1` 或 `mtp`。 |
| `manifest.workload_sha256` | 用 `simverse.contracts.digest` 对按 `request_id` 排序的数组计算。每项仅含 `request_id`、`sample_id`、`input_tokens`、`arrival_ns`。输出轨迹使用输入文件的完整 SHA256 标识。 |
| 采集清单 | `captured` 须提供 `backend_revision`、`model_revision`、`tokenizer_revision`、`device`、`topology`、`precision`、`command`、`runtime_identity`、正整数 `tp` 与 `ep`，以及 `method_config` 对象。DFlash 与 MTP 还须提供 `method_config.revision`。`raw_files` 是相对路径到完整 `sha256:…` 的映射，导入时逐文件核对。文件位于轨迹所在目录内。 |
| `window` | 对象。包含 `start_ns`、`end_ns`、`clock_id`、`time_basis` 与 `granularity`。时间基准为 `committed` 或 `server_emitted`，提交粒度为 `token` 或 `commit_group`。可选 `warmup_end_ns`、`drain_end_ns` 分别位于窗口起点之前或相等、终点之后或相等。 |
| `requests[]` | 每项包含 `request_id`、`sample_id`、`input_tokens`、`arrival_ns`、`status`、`output_tokens` 和 `emissions`。可选 `admit_ns` 缺省或为 null 时仅检查到达时刻。状态为 `completed`、`cancelled`、`failed` 或 `in_flight`。已结束请求另存 `terminal_ns` 与 `end_reason`。 |
| `emissions[]` | 每项包含 `time_ns` 和非空 `token_indices`。请求内序号自 0 连续，时间允许同刻。逐 token 输入每项只有一个序号。客户端流块须保留在独立采集记录，服务端摘要按本表的观测基准导入。 |
| `slo` | 可选对象。包含 `ttft_max_ns` 和 `mean_itl_max_ns`，阈值取等号。 |
| `rounds[]` | 可选。每项包含 `request_id`、全局唯一的 `round_id`、`configured_k`、`proposed`、`verified`、`accepted`、`committed` 和提交起始序号 `token_start`。导入检查数量上界和提交区间归属。提议数为 0 时另存 `end_reason`。 |
| `batches` 与 `spans[]` | 可选。批 ID 映射到请求 ID 数组。阶段记录提供 `span_id`、`request_id` 或 `batch_id`、`stage`、`role`、`resource`、`timing_boundary` 和 `start_ns`、`end_ns`。时长未知时两端均为 null，并提供 `timing_reason`。可选 `round_id` 须关联同一请求或批内请求。 |

请求、提交、阶段与轮记录可省略重复的 `clock_id` 和 `time_basis`。若提供，两者须与窗口一致。原始文件 SHA256 验证内容身份，服务执行和硬件指标仍按采集证据独立验收。

## 采集输入与配对

- baseline、DFlash、MTP 使用同一目标权重与 tokenizer revision、精度、输入 token 序列、请求到达轨迹、采样配置与输出上限。固定设备和 TP/EP、批处理和并发设置、缓存冷暖状态及测量开关。逐方法另记草稿 checkpoint 或 MTP 头、实现和配置的版本。输出长度或停止原因有差异时逐请求披露，按实际提交 token 计数。
- 首轮固定一个后端版本与一个同卡规模，先核对三个方法的实际支持路径、模型输出和可分配预算。缺配对时记录缺项并保留已取得的单方法证据。DFlash DT 仿真与 Ascend 条件成本档各自保留现有身份。Ascend 上机条件见 [A-N6](../ascend-validation-plan.md)。
- 固定请求清单、原始日志、命令、版本和哈希。先预热，再开启观测窗口，排空阶段保留原轨迹。对实际服务比较采用独立进程重复，预先登记 SLO、误差门槛、观察窗口、允许变化的字段和输出一致性判据。记录失败试次与 profiler 开关，必要时以同负载无 profiler 的运行估计采集开销。

## 最小输入字段

以下为完整采集目标。离线导入已支持上表字段，实际设备、容量和硬件阶段证据按采集计划归档。时间使用距本次采集起点的整数纳秒，展示时转换为毫秒。缺失值使用 `null` 和原因。跨进程或跨设备的时钟仅在对齐依据与误差可追溯时计算差值。

| 对象 | 必采字段 |
| --- | --- |
| 运行清单 | schema 版本、`source_kind`（合成或采集）、`run_id`、方法、后端与改动身份、权重和 tokenizer revision、设备与拓扑、TP/EP、精度、方法配置、请求清单哈希、采集命令和原始文件哈希。 |
| 观测窗口 | 窗口起止、预热与排空边界、单调时钟来源、时间基准、流粒度（逐 token 或成组提交）。客户端轨迹如有采集，另记同名字段和自己的时钟。 |
| 请求与提交 | 请求 ID、输入 token 数与样本身份、arrival、admit、提交输出序号和时刻、输出 token 数、完成状态、终止时刻与原因。多 token 组保留序号范围和组提交时刻。客户端只有流块时记录块粒度。 |
| 服务阶段 | span ID、请求或批 ID、轮 ID、角色、设备/rank、资源、阶段、起止、计时边界及可定位的原始记录引用。baseline 采 prefill/decode。DFlash 与 MTP 分别采 draft、verify、commit/recovery 和实际存在的附加阶段。 |
| 投机轮 | 请求 ID、轮 ID、配置上限、实际提议数、实际校验数、接受草稿前缀长度、提交 token 数及结束原因。可识别时单列 target bonus 和其输出序号。记录 `K=0` 的原因并单列。 |
| 容量 | 每 rank 可分配预算、观测峰值、采样窗口和口径。保留 allocated/reserved 及 KV、workspace、graph、通信池等可定位分量，缺值单列。 |

校验包括唯一 ID、引用闭合、输出序号连续、时间单调、窗口正长度、请求到达与输出顺序，以及每轮 `0 ≤ accepted ≤ verified ≤ proposed ≤ configured_k`。提交 token 数按停止和截断后的实际事件核对。EOS 或输出上限可能改变本轮提交数量。按请求保留 `(proposed, verified, accepted, committed)` 的联合记录，另报接受长度直方图、样本数和条件分布。有明确分母时才给全接受比例。只拿到均值或边际分布时，动态 K 的联合律保持未知。

## 指标和残差

| 指标 | 首轮验收口径 |
| --- | --- |
| TTFT / E2E | 服务端首个提交输出减 arrival，最后提交输出减 arrival。终止时刻另留作请求状态核对。全程无输出保持未知。 |
| ITL / 请求 TPOT | 同请求同时间基准下报告相邻输出的间隔分布。另报请求平均间隔 `(last - first) / (N - 1)`。只有 0 或 1 个 token 时返回 `null`。对于可证实的同刻提交，保留零间隔。只有流块且无法确认组内提交时，报告块间隔和粒度。 |
| 吞吐 | 用窗口 `[start,end)` 内的提交 token 数除墙钟，单位 token/s。完成请求口径另列，以成功请求的最后提交落入窗口为归属条件。带输出的失败或取消请求实际提交仍计入事件吞吐，状态和计数单列。 |
| Goodput | 只计在窗口内成功完成且同时满足预先冻结的 TTFT 和请求平均 ITL 阈值的请求，分别报告 req/s 和 token/s。单 token 请求仅检查 TTFT。缺失的必需观测不能通过 SLO。 |
| 分位与残差 | 保留样本量和分位估计方法。阶段对账先按已对齐的同边界区间取并集，报告对应墙钟和未解释残差。对照仿真时另报同边界的有符号误差和绝对误差。时钟不可对齐或阶段缺失时保留未知。 |

同 rank/资源的 busy 时间用区间并集。在多 rank 或有重叠的运行中分别披露慢 rank、每 rank 资源窗口和端到端墙钟。只在资源归属和时钟都成立时给联合窗口。显存缺口和未归属时长继续进入证据门禁，按 [合入准则](../MERGE_CRITERIA.md) 判定。

## 离线验收样本

时间均为毫秒。观测窗口为 `[10,50)`。A 的两个 `20` 是明确记录的同刻服务端提交。

| 请求 | 到达 | 提交时刻 | 状态 |
| --- | ---: | --- | --- |
| A | 0 | 8、20、20、50 | 成功 |
| B | 4 | 10、30、48 | 成功 |
| C | 12 | 16 | 成功 |
| D | 41 | 45 | 49 时取消 |
| E | 42 | 无 | 43 时失败 |

- 窗口内有 7 个提交 token，事件吞吐为 175 token/s。A 最后提交在右边界，按完成口径只有 B、C 的 4 token，完成吞吐为 100 token/s。
- 修复前，非空输出列表会跳过最后提交的窗口检查，使完成吞吐误计为 200 token/s。V08 回归现在断言完成吞吐为 100 token/s、完成请求数为 2，并核对左边界计入和事件吞吐。
- B 的 TTFT 为 6 ms，E2E 为 44 ms，ITL 为 20、18 ms，请求平均间隔为 19 ms。C 的 TTFT 为 4 ms，平均间隔为 `null`。示例 SLO 取 TTFT ≤ 5 ms、请求平均间隔 ≤ 20 ms，只有 C 通过，goodput 为 25 req/s 和 25 token/s。E 的 TTFT 为 `null`。
- 投机另设手算样本。DFlash 的 `(proposed, verified, accepted, committed)` 可取 `(3,3,0,1)`、`(3,3,3,4)` 及输出上限截断的 `(2,2,1,1)`，接受长度直方图的 0、1、3 档各有 1 轮。MTP 用独立的 `(1,1,0,1)`、`(1,1,1,2)`。baseline 没有投机轮。
- 残差样本以 `[10,50)` 为墙钟，两种资源的已解释区间为 `[10,20)`、`[18,25)`、`[40,50)`。并集为 25 ms，未解释为 15 ms。不能对齐时钟的变体应输出未知。
- 负例至少覆盖逆序时间、重复输出、缺失轮引用、接受数超过校验数、混用时钟、采集身份不匹配，以及模拟合成数据被标记为硬件实测。

从仓库根目录运行下列命令，预期输出为 100.0 token/s。

```powershell
py -3.12 -c "from simverse.execute.reduce import throughput_tokens_per_ms as t; e = {'A':[8,20,20,50],'B':[10,30,48],'C':[16],'D':[45],'E':[]}; print(t(e, ['A','B','C'], 10, 50, 'completed_requests') * 1000)"
```

实施验收按三个阶段进行。先以人工样本完成导入拒错和手算对账。再由匹配环境采三路服务证据，逐请求核对配对、停止与采数覆盖。最后用真实数据验收事先登记的延迟、吞吐、显存和残差门槛。当前交付已验收离线导入、请求指标和手算样本。真实采集、阶段残差和硬件门槛仍待各自验收。

## 上游参考

- 阅读了活跃的 vLLM 仓库在 `11aa1369` 的服务基准归约及模拟流测试。借鉴成功请求筛选、区分服务端与客户端时间、剔除无内容流块和覆盖单 token 边界。本仓保留半开窗口、实际提交序号和显式缺值，避免将客户端分块统计直接当作服务端逐 token 指标。所查代码与测试为源码阅读。 [基准源码](https://github.com/vllm-project/vllm/blob/11aa1369e297f67bc4dc26e119d10830f39956c6/vllm/benchmarks/serve.py) · [时序测试](https://github.com/vllm-project/vllm/blob/11aa1369e297f67bc4dc26e119d10830f39956c6/tests/benchmarks/test_endpoint_request_func_timing.py)
- vLLM 官方说明区分提议数、校验数和接受数，部分版本可导出逐轮列表。按实际采集版本核查字段，再与服务端提交事件对账。现有 DFlash 双栈核对和 MTP 头映射仍分别作为实现证据。 [接受统计](https://docs.vllm.ai/en/latest/features/speculative_decoding/acceptance_metrics/) · [DFlash 服务栈审查](dflash-serving-audit-20261112.md) · [MTP forward 映射](qwen36-mtp-forward-20261113.md)
