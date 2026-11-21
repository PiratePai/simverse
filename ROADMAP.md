# Simverse 待开发计划

更新于 2026-11-22。已同步 R07b/R09 的投机轮与服务端输出组关联。产品范围保留 **Qwen3.6-35B-A3B（含 DFlash 和研究档 MTP）** 与 **DeepSeek-V4.1-Flash**。新增设备与采数继续优先昇腾。任务可以独立交付，联合验收按各结果的实际依赖执行。

## P0：证据准入

| 任务 | 现状 | 交付与验收 |
| --- | --- | --- |
| A-N1a / A-N5 离线证据准入 | R28/R29 保留局部采集，两个互斥成本档仅供条件对照。R30 [PR #26](https://github.com/PiratePai/simverse/pull/26) 仍开放。分支重建报告已登记六个 run 的 42 条消费文件哈希和组级剩余时长。原件在仓库外，运行身份仍有缺口。见 [分支重建审查](https://github.com/PiratePai/simverse/blob/063460bddca6814bfb4137618b0d465fb53186b3/docs/validation/ascend-profile-audit-r30-rebuild-20261118.md)。 | 独立校验重建产物与来源清单，记录哪些原件可取得并逐字节重放。补阶段、角色、形状与时间所有权。可独立评审离线导入与缺项报告，证据不全时保持 audit-only。正式成本等待匹配身份、数值对照、时间守恒和独立留出。采集整改与 [昇腾验证计划](docs/ascend-validation-plan.md) 同步。 |

## P1：局部验证与工程精简

| 任务 | 交付与验收 | 依赖 |
| --- | --- | --- |
| A-N1b、A-N2a、A-N3a / R06、R02b、R03b、R17 | 910B3 单卡逐头 MRoPE、实际后端小 fixture 中间张量和 KV/GDN 状态先验收，随后分别验收 TP2 与 EP2。统一栈测每 rank 可分配预算、workspace、graph、HCCL 池及实际原语，保留失败、独立重复和未知项。R06 另保留 V4.1 所用模块的 CPU 数值对照。 | 离线 oracle 可先做。采集依赖匹配的栈、设备和运行窗口。少卡结果只在匹配域内使用。见 [昇腾验证计划](docs/ascend-validation-plan.md)。 |
| R15–R19、R08 / A-N4、A-N5 | R15/16 先核验 910B3 SKU、官方 dense 规格、假设来源、精度和后端配方。R17 识别实际拓扑和原语。R18/R08 按大额解析回退补真实 MoE/GDN/attention、通信启动和运行时边界。R19 按目标、草稿、TP、链路和预算核对异构兼容性。 | 解析档可先登记。正式成本依赖设备、实现、形状和留出证据，通信成本另需 R17。910C/950DT/950PR 按独立 SKU 准入，NVIDIA 新设备排在 910B3 数值、容量与局部成本验证之后。 |
| R04/R05/R20–R22/R24/R25 残余 | V4.1 的 Engram 单副本、mHC、CSA2 索引与注意力核、MoE 路由、量化映射和 SwiGLU 已有源码证据。优先补目标场景实际使用的 Engram CPU offload、缓存、量化执行和 kernel 数值边界。DP 交换、其他平台与后端变体按场景选做。 | 保留 `structure_complete=False` 及变体原因，逐模块解除。详见 [能力缺口](docs/gaps/capability-gaps.md)。 |
| R07b、R09 / A-N6 | 已验收三路重放、服务端 TTFT、轮内阶段和逐 rank 显存。DFlash 与 MTP 的逐轮提交、输出组和目标侧 token 关联已通过合并、截断和缺项的离线回归。下一步在匹配栈采阶段时长、运行时显存及提交联合分布。 | 硬件验收依赖匹配权重、运行时容量和实际服务支持。详见 [端到端 trace 验收方案](https://github.com/PiratePai/simverse/blob/main/docs/validation/r07b-r09-trace-acceptance.md)。 |
| R10f 成本和导出口径 | 已验收解析/选用成本的导出、未知阶段、旧 v0.2 trace 回读和同卡显存字段。后续新增计时数据时，按阶段、组归属、回退与来源复验。 | 保留原始 Work/Traffic/Memory 账本。未单独计价的运行时阶段按 R07b/R09 或 A-N5b 补采。见 [成本组回归](docs/validation/cost-group-simplification-20261118.md)。 |
| R10g 归档边界与模块职责 | v0.1 网页与 HTTP 已退役，历史 CLI 归入 `archive`。后续仅在明确能减少重复决策时拆分 `run_v2` 的验证、计数、执行或结果投影。 | 仅为有迁移出口的字段约定退役路径。固定源码、原始采集及有哈希约束的产物继续保留。 |
| R10a/b 维护与验收 | 编码警告仅在有完整复现环境时处理。工作台文档导航、JSON 下载内容及旧 CSV 错误隔离已验收。 | 其他窄屏和异常恢复场景可复用 [工作台回归](docs/validation/workbench-review-fixes.md) 按需抽查。 |

## P2：请求级实验与其他设备

| ID | 任务 | 验收依赖 |
| --- | --- | --- |
| R14a | 人工轨迹已核对请求级 TTFT、ITL、吞吐、goodput 与 SLO。 | 按 [端到端 trace 验收方案](https://github.com/PiratePai/simverse/blob/main/docs/validation/r07b-r09-trace-acceptance.md) 补匹配栈的到达、结束和时钟证据。 |
| R11 | lookahead、取消、回滚与浪费账。 | 方法状态与事件核，人工轨迹可先验。 |
| R12a/R12b | 请求到达、确定性组批和重放，然后补 KV 分页、抢占、回收与守恒。 | R12b 依赖 R12a 的请求生命周期。 |
| R13a/R13b/R13c | 分项验证 Chunked Prefill、PD 分离与 Prefix Cache。 | R13a 依赖 R12a 和 prefill 状态，R13b 依赖 R12a/b 与 R17，R13c 依赖 R12b。 |
| R14b | 固定一组在线场景做联合验收。 | R14a、场景用到的 R11–R13 分项和匹配请求轨迹。 |
| C1/C2 | 评估 A100/A800/H800/L40S 及 B200/B300/GB200/GB300。 | 逐 SKU 区分卡、节点和机柜。保留已有 H100/H200/L40 回归。 |

## 已交付证据索引

- R10c/d/e、R10b 文档入口与 R10f 比较侧已合入。wheel 可从检出目录外启动，CLI/API/工作台共用预设，输入错误附字段路径，扫描与部署比较统一使用适用域和指标口径。最低 Python 版本的独立 wheel 检查仍待补。见 [PR #33](https://github.com/PiratePai/simverse/pull/33) 和 [R10 验证记录](docs/validation/entry-scan-compare-fixes-20261119.md)。
- [PR #34](https://github.com/PiratePai/simverse/pull/34)、[PR #35](https://github.com/PiratePai/simverse/pull/35) 和 [PR #36](https://github.com/PiratePai/simverse/pull/36) 已交付成本与导出链路归整、v0.1 离线入口、CUDA KV 分片修正、未知时长披露和工作台乱序保护。实施状态记录新结果字段和旧 trace 的回读方式。见 [实施状态](docs/IMPLEMENTATION_STATUS.md)。
- R01–R03 已交付 TP/EP per-rank 配方、同卡对象身份、生命周期和预算判定链路。默认可部署性仍为 unknown，实际预算与运行时池待验收。见 [A 组验证](docs/validation/agroup-tp-ep-colocated-20261110.md) 与 [二轮复核](docs/validation/agroup-lmhead-ctxkv-audit-20261110.md)。
- R06 已查验 Qwen GDN/MoE/full-attn 小型 CPU fixture，R26 已查验文本 MRoPE，R23 已为原生 MTP 建账。R07a 已核对 DFlash 服务栈的同卡路径，DT 串行与流水为仿真实验。真实 checkpoint、接受分布和部署执行须另验。见 [R06](docs/validation/qwen36-gdn-moe-cpu-fixture-r06.md)、[R26](docs/validation/qwen36-text-mrope-20261116.md)、[R23](docs/validation/qwen36-mtp-forward-20261113.md)、[R07a](docs/validation/dflash-serving-audit-20261112.md)。
- R04/R05/R20–R22/R24/R25 的量化、MoE、mHC、Engram 与注意力证据按模块保存在 [功能状态](docs/FOCUS_STATUS.md) 和 [验收记录](docs/IMPLEMENTATION_STATUS.md)。历史 H200 profile 继续绑定原采集模型。

## 参考实现与验收

- Vidur 在 `abae7f63` 的预测器中按模型形状、TP 和 collective 筛选样本，并在采数文档中单列主机开销。本仓沿用这种分类，继续要求缺失值传播、按域回退和来源可追溯。当前阶段不引入其预测器或 sklearn 依赖。查阅时仓库未归档，最近提交为 2026-08-24。见 [预测器源码](https://github.com/microsoft/vidur/blob/abae7f63aa857300f5cdc6f5e0d27860cd24721b/vidur/execution_time_predictor/sklearn_execution_time_predictor.py) 与 [采集文档](https://github.com/microsoft/vidur/blob/abae7f63aa857300f5cdc6f5e0d27860cd24721b/docs/profiling.md)。
- vLLM 在 `11aa1369` 的服务基准里从请求输出和时间戳归约 TTFT、ITL 与 goodput，并用模拟流验证时序和失败边界。R14a 采用独立请求轨迹和负例，沿用本仓的指标契约。查阅时仓库未归档，最近提交为 2026-11-18。见 [基准源码](https://github.com/vllm-project/vllm/blob/11aa1369e297f67bc4dc26e119d10830f39956c6/vllm/benchmarks/serve.py) 与 [时序测试](https://github.com/vllm-project/vllm/blob/11aa1369e297f67bc4dc26e119d10830f39956c6/tests/benchmarks/test_endpoint_request_func_timing.py)。
- 当前基线的 Windows / Python 3.12 全量收集数为 734。每次合入记录适用域、证据来源、测试结果、数值变化与限制。硬件执行和真实模型采集需匹配环境与授权。见 [合入准则](docs/MERGE_CRITERIA.md)。
