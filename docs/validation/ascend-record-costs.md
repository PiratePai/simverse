# Ascend 归档记录成本、覆盖与回退

2026-11-17。本次只在 Windows 上核对已有源码/数据、运行离线回放与浏览器验收，没有启动 NPU、下载权重或采集新样本。

## 1. 已接入与准入边界

工作台新增「归档成本记录（有条件对照）」。R29 通信与 R28 独立 GEMM 是两个互斥记录档：各自只替换满足条件的成本项，其余逐项回退。原始 Work/Traffic/Memory、纯解析指标和原始归档不改；同一条计数/执行管线分别求值，避免新增第二套物理公式。

**当前是条件记录对照，不是生产实测校准。** 两档缺少完整运行身份或独立留出验证，不能升级为 admitted measured profile。派生 `source_kind=assumed-from-archived-observation`，原始产物仍保留采集来源身份；新指标为 analytical-estimate/approximated、feasibility unknown，不授予部署排名。用户选择一档表示采用该档的独立算子/通信边界作条件假设，并不证明当前服务栈已经匹配。

| 记录档 | 源码/原始产物 | 可用域与查表 | 明确不覆盖 |
|---|---|---|---|
| `ascend-r29-hccl` | PR #23 @7de467d，原 rank0/1 JSON、collective 脚本及环境快照逐文件 SHA256；50 个尺寸/原语点，每点保留两 rank 各五批样本 | 910B3、物理卡5/6、world=2、BF16、HCCL 独立 collective。按每 rank **逻辑输入字节**查延迟；精确点直接使用，内部点按字节线性插值，域外拒绝 | transport/CANN 完整身份未知；不能按同字节将 logits gather-to-root 或 EP variable-split 当 allgather/equal-split alltoall；不外推 TP4/8；不换成链路规格 |
| `ascend-r28-gemm` | PR #21 原采集 @6a16fbe，p3 JSON/脚本、GEMM 环境日志与 env.txt 均有 SHA256 | Qwen3.6 target、BF16、独立 `torch.matmul` 的算子名+M/K/N 精确匹配；10 个单点与 4 个额外 M 扫描点。M=1 重复窗口不混合，算子不插值、不外推 | 非 vLLM 融合核、非 TP 并发验证；缺完整原始五批样本与输入清单；MoE 两乘代理、卷积、RoPE、SDPA、组合 norm、msprof 摘要不替换；草稿不跨模型继承 |

两档 torch/torch_npu 分别为 R29 的 2.10.0/2.10.0.post4 与 R28 声明的 2.9.0/2.9.0.post9，不能拼成同一次运行校准。也不能与历史 H200 profile 混用。传 list、未知档或同时传 `cost_profile` 都拒绝。

## 2. 计价与覆盖定义

- R29 使用 `max(rank0 median, rank1 median)` 作显式保守建模选择，不将其称为同步批次的最慢 rank 实测。替换整笔 collective 的解析成本；样本已含 host enqueue/sync 摊销，不能再加原 link fixed 或 launch。
- AllGather 输入 n、输出 world×n；ReduceScatter 输入 world×n、输出 n。直接使用延迟，不从旧 busbw 反推；AllGather 原派生带宽错误不污染查表。现有 lowering 只有明确的 allreduce 与 MTP fc allgather 可绑定；歧义 gather 与变长 EP 保留解析值。
- 原 `M01_tpot_analytical` 等全部保留；实际命中时另给 `M01_record_tpot`、record baseline/speedup/round 指标。UI KPI、矩阵、扫描、逐行价格、Roofline、时间轴与 CSV 使用条件对照值并披露原因。零命中不产生 record KPI，避免将全回退伪装为已校准。
- 覆盖按阶段、成本评估实例、记录成本与解析回退成本分栏。基线与投机轮的各阶段均是独立评估；所有实例成本求和不是一次轮时长或 E2E 时间覆盖，更不是物理 HBM/真实瓶颈结论。插值有相邻点引用；通信留一内部尺寸误差仅作曲线诊断，不替代独立运行留出。
- 尚未计价的接受、回退、hidden 提取等方法运行时阶段单列 `unpriced_components`，时间保持 null，不进入覆盖分母；条件指标同时披露缺项。DT 的 d2t/t2d 点到点传输另记解析回退，不复用 HCCL collective 曲线。
- 原始与派生文件双重哈希校验；修改成本但只保留原始文件哈希也会拒绝。选用记录内容的完整 SHA256 进入 cost_digest/run_id；默认不选记录时原场景身份不增加空字段。

## 3. 固定回放与实测计划

[覆盖与补采清单](../evidence/ascend-record-costs/coverage-plan.json)由 `tools/replay_ascend_record_costs.py` 重放七个固定场景，包含每种回退的全部去重形状、实例数、分阶段范围与解析成本权重。不是手填任务表；UI 当前 run 也生成同样结构的 `measurement_plan`，JSON/静态 HTML 随结果导出。

固定 B1/CTX4096/BF16，默认接受率假设 α=0.75、DFlash K15/BLK16：

| 场景 | 记录命中/成本实例 | 纯解析→条件对照 TPOT (ms) | 主要未覆盖 |
|---|---:|---:|---|
| R29 TP2 baseline | 80/704 | 25.748260→22.110020 | 623 个算子实例仍解析；logits 原语未固定 |
| R29 TP2 DFlash | 172/1530 | 18.230667→17.027996 | 算子与三次 logits gather；不能借 R28 补齐不同栈 |
| R29 TP2/EP2 baseline | 40/744 | 25.746704→23.927584 | 80 个 EP dispatch/combine 与 logits 原语不等价 |
| R29 TP8 baseline | 0/704 | 115.583503→全回退 | world=2 记录不外推；没有记录 KPI |
| R28 TP2 baseline | 40/704 | 25.748260→26.475701 | 161 个 GEMM 行缺形状、422 个非等价算子、全部通信 |
| R28 TP8 baseline | 201/704 | 115.583503→118.447619 | 非 GEMM/实际后端融合与全部通信 |
| R28 TP8 DFlash | 201/1530 | 36.556160→36.556160 | 命中仅在独立基线；验证 M16/草稿/通信均回退，投机轮 TPOT 不改变 |

这组结果不说明哪个真实部署更快；R28/29 不能相互比较精度，也不能由解析回退成本确定真实瓶颈。根据实际未覆盖项调整后的采测顺序见 [昇腾计划](../ascend-validation-plan.md) 的「记录接入后的补采顺序」。

## 4. 验证与参考

`py -3.12 tools/distill_ascend_cost_records.py --check` 重验原始来源、统计与派生成本；`py -3.12 tools/replay_ascend_record_costs.py --check` 重验覆盖、回退、数值锚点和原账本不变。`tests/test_ascend_cost_records.py` 覆盖独立查值/插值与拒绝外推、无双重收费、歧义原语、GEMM 角色/形状、坏哈希、跨栈隔离、模型/容量账本不变、零命中、CLI/API/CSV、补采去重和 MTP/EP 边界。

复用本仓 `CalibratedPricer` 的差额替换思想，但不借用它的 admitted profile 身份。官方参考：静态阅读 [nccl-tests @afd59abab 的 AllGatherGetBw](https://github.com/NVIDIA/nccl-tests/blob/afd59abab774a6a4b492eb344bd0c845a0b7bb4d/src/all_gather.cu#L33)，核对逻辑输入/输出与规范化定义；[PyTorch Timer @v2.10.0 源码](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/utils/benchmark/utils/timer.py#L63)说明预热、同步、重复窗口与环境标识的处理。只借鉴采样边界与来源分组，不引入新依赖；没有执行上游 CUDA/NPU benchmark。全量测试及浏览器验收记录见 IMPLEMENTATION_STATUS。
