# R29 / PR #23 审查：归档可合入，模型与校准未准入

2026-11-17；原提交 `7de467d604d83cbee5eef76c2d12679356f3cad1`。本轮在 Windows 上阅读采集源码、核对既有原始产物并运行离线测试，**没有新做 NPU 测量**。

以本报告与 [派生 review.json](../evidence/ascend-910b3-r29/review.json) 为审查结论。[采集时报告](ascend-910b3-r29.md)、README、原 manifest、原始数据和采集脚本保持捕获内容，包含下列已知错误，仅供来源追溯。外部报告的 LF 检出规则已补齐，其 SHA256 恢复为原 manifest 声明值；没有修写原始测量或刷新原 manifest 掩盖差异。

## 1. 可以确认的范围

- 双 rank 五类 collective 的原始记录各有 50 项 exact 校验，没有记录 mismatch；这是采集者记录的局部结果，本轮未在设备重跑。AllGather 派生带宽需使用下述修正。
- 两层随机权重 fixture 的成功日志为 **TP1、物理卡 5**，可见两卡不等于使用两卡；输出 `[20,20,20,20]`。这只支持该环境中引擎执行并产出 token，`numerical_correctness_verified=false`，不能推导完整模型或双卡 forward 正确。
- 通用 MLP/GQA 分片和 CPU 专家计算配合 NPU 通信的玩具路由测试有局部观测；不等于 Qwen3.6 实际后端 TP2/EP2 集成验证。
- 全部记录保持 `not-admitted`；不改变 DEVICES、PROFILE_SETS、模型执行状态、部署排名或 G-D7/R02b/R17/R18。R28 与 R29 的卡对、软件栈不同，不拼接或相减。

## 2. 发现与修正

| 级别 / 问题 | 审查结论与处理 |
|---|---|
| P1：来源校验并非「23/23 通过」 | 原 manifest 的日志声明 16,614 B，入库日志实际 16,603 B，SHA256 也不同；原因未知。保留差异，不为凑齐校验修改原文件。23 个声明产物中 22 个匹配，另冻结原 manifest 自身哈希及实际文件清单。环境快照漏 `r29_tp2.py`，`r29_manifest.py` 哈希与最终脚本不符；固定归档字节不等于已证明运行时用了全部这些脚本。原 manifest 的 A-N3「未执行」是过时状态，不能反过来抹掉已有切片文件 |
| P1：MRoPE oracle 共用错误切片 | `q` 为 `[tokens,4*256]`，被测路径与 oracle 都取 `[..., :64]`，只旋转第 0 个 head，其他 3 个头被误算作尾部。应 reshape 为 `[tokens,heads,256]` 后对每头前 64 维旋转；K 也须独立覆盖。现有 2.75e-4 仅是错误切片路径之间的误差，撤销 Qwen 几何数值验证结论。纯标量反例已进离线测试 |
| P1：MRoPE identity 与频段统计无效 | identity 比较装配结果与其搬到设备后的副本，没有独立比较 T 分量；NeoX 配对索引应为 `j,j+32`，不是 `2j,2j+1`。低索引为高频，原 low/high 标签也反了。尾部逐位相等须按 head 分别检查；重采前不能称参数域已验证 |
| P1：融合失败被泛化为后端不支持 | 161002 只证明八组特定调用中的失败；其中若干探针 section 之和与 rotary/2 不符。没有证明布局、section 约束与本栈契约全部正确，也没有证明 vLLM 实际走该融合入口。改为「调用失败原因及实际集成路径待定位」，不把它升级为 Qwen3.6 无法运行的结论 |
| P1：AllGather 规范化分子漏 world | nccl-tests 的 AllGather algbw 使用完整接收缓冲：`n*P/t`，随后乘 `(P-1)/P`。原脚本以输入 `n/t` 再乘系数，P=2 少算一倍。review.json 对两 rank、全部 100 点保存独立派生值与原值。`send/recv_bytes` 是逻辑缓冲/端点字段，不是线缆计数；原 `bytes_per_rank_recv` 也不是各 op 统一的接收总量 |
| P1：误将 OOM 分配量当权重大小 | 成功日志明确 `Model loading took 0.32 GiB`、权重 0.32 GiB、available KV 11.81 GiB。原「权重约 30–33 GiB」与原「利用率必须 ≤0.2」均撤销。0.9/0.55 的失败日志/报告未归档；当前日志只有 0.2 成功，不能以它重放三次尝试或归因失败。下一轮固定显式 KV 预算并分别归档失败/成功，不先缩小已很小的 fixture |
| P1：EP2 切片被写成真实模型数值验证 | 专家 E8/top2/H64/inter32、GELU，专家矩阵计算在 CPU double，NPU 仅交换元数据/结果；不是配置的 E256/top8/H2048/inter512 与 SwiGLU 后端。MLP 只有 up/down GEMM；GQA 无 causal mask，所谓 BF16 case 在 FP32 计算 scores。`roundtrip_permutation_identity` 实际只比较接收元数据多重集合；KV/permutation/token flags 没有进入 summary 的失败汇总。7/7 是这七个输出比较，不是完整状态守恒验收 |
| P2：路由量与峰值范围误读 | 每 rank 总发送/接收均为 32 条，其中每目的/来源 16 条；原文把 16 写成每 rank 总量。全局 assignment 为 64。3,624,960 B 仅 allocator 峰值；进程 free 差、HCCL/runtime 池未计入该峰值，不能当部署容量。空专家 6/7 都属于 rank1，只覆盖该侧空分支 |
| P2：graph 差分仍混入其他对象 | capture 相对 baseline 的 free 差 56,623,104 B 可以保留；但 static input clone 在 baseline 之后创建，allocated 增 16,779,264 B 也含静态输入/输出。首次 replay 相对 capture 为 **4,194,304 B**，原 4,210,304 B 是笔误。无 clone-only 对照、输出 finite/数值检查，不得把差分称纯 graph 池；随机矩阵反复相乘也未排除溢出 |
| P2：ACL 粒度与返回码断言过强 | 两 rank 的增量 free 差减 requested 为 `[2097152,2097152,2207744]`、`[2097152,2097152,2097152]` B；并非每次恰为 2 MiB，更不能从三个大块断定分配粒度。捕获值中的返回码为 0 可保留，但脚本 rc 对非 int 静默返 0、部分 get_mem_info 返回码不进失败汇总；因此脚本还不满足任一失败即拒绝的采集契约 |
| P2：环境与 runtime 来源不足 | CANN 两个 version 文件均 missing，9.1.0 只见路径/采集声明；vLLM runtime commit 未保留。日志存在本地 `ascend_warmup_patch.py`、GDN fallback 和版本不匹配警告，补丁内容/哈希未归档，不称官方固定版本组合已验证。两个 PCIe BDF 不足以证明不同交换平面，HCCL transport 继续 unknown |
| P2：复跑入口会改写归档 | scripts 固定写 raw 且使用同一 RUN_ID，部分错误也返回 0，不能推荐原样复跑。`runtime/` 副目录仅有事后声明/dirty 快照，无法从现有材料定位创建者，不据此盲改本仓入口。下一轮用新 run 目录、唯一 ID、拒绝覆盖与明确副作用追踪 |

## 3. 通信数值锚点

rank0、脚本 buffer scale=256 MiB；单位十进制 GB/s，边界为批内 host+sync 墙钟，不是纯 kernel、物理链路或 HCCS 吞吐。AllGather 完整输出为 512 MiB；ReduceScatter 完整输入为 512 MiB，各项不能仅看相同的 scale 就比较硬件效率。

| op | median ms | 原 busbw | 审查后 algbw / busbw |
|---|---:|---:|---:|
| all_reduce | 12.913047 | 20.788 | 20.788 / 20.788 |
| all_gather | 15.144462 | 8.863 | 35.450 / 17.725 |
| reduce_scatter | 16.069441 | 16.704 | 33.409 / 16.704 |
| broadcast | 14.045876 | 19.111 | 19.111 / 19.111 |
| all_to_all_single | 7.358821 | 18.239 | 36.478 / 18.239 |

两 rank 各自五批样本保留，但没有联合批 ID、跨 rank 同步计时窗；不把各 rank median 的最大值称作已实测的全局最慢轮耗时。

2026-11-17 复核官方 [nccl-tests @afd59abab 的 AllGatherGetBw](https://github.com/NVIDIA/nccl-tests/blob/afd59abab774a6a4b492eb344bd0c845a0b7bb4d/src/all_gather.cu#L33)、[ReduceScatterGetBw](https://github.com/NVIDIA/nccl-tests/blob/afd59abab774a6a4b492eb344bd0c845a0b7bb4d/src/reduce_scatter.cu#L32) 与 [AlltoAllGetBw](https://github.com/NVIDIA/nccl-tests/blob/afd59abab774a6a4b492eb344bd0c845a0b7bb4d/src/alltoall.cu#L42)。仅借鉴带宽规范化定义，没有执行上游 CUDA 测试或将 NCCL 算法假定成 HCCL。MRoPE 语义继续使用仓内已 pin 的 R26 证据，不引入新推理依赖。

## 4. 验证与后续

`py -3.12 tools/review_ascend_r29.py --check` 重放派生审查；`py -3.12 -m pytest tests/test_ascend_evidence_r29.py -q` 检查归档完整性、原始 hash 差异、两 rank 通信、容量差分、逐头 MRoPE 反例、TP1/玩具 EP 边界与不准入。没有执行归档采集器。全量测试与远端 required check 记录于 IMPLEMENTATION_STATUS 与 PR。

任务次序以 [更新后的昇腾计划](../ascend-validation-plan.md) 为准：先补可重放来源与采集器拒错，再验证逐头 MRoPE / 实际后端数值与 cache/state，随后同栈容量、设备档及局部 profiling；整模和匹配 E2E 继续单独验收。
