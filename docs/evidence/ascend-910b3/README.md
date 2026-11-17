# Ascend 910B3 原始采集归档（2026-11-17）

**本目录未通过生产校准准入。先读 [PR21 审查报告](../../validation/ascend-910b3-20261117.md)
及 [review.json](review.json)，不得直接使用原 JSON 的派生字段。**

原采集提交：`6a16fbed8e6dc0982ed9abdea0ea1df6c006e429`。本轮保留全部原始 JSON/TXT/日志
和 scripts 字节；仅重写本说明、补审查侧文件并更新 manifest 的说明文件哈希。
manifest 的 source_kind=measured 只描述采集来源，calibration_admission=not-admitted
表示尚未符合本仓 ProfileRecord 及成本校准契约。review.json 是离线审查派生结果，不是新测量。

采集者声明：Ascend910B3 物理卡6/7，CANN9.1.0、driver26.0.rc1、torch2.9.0+cpu、
torch_npu2.9.0.post9、Python3.12.3、openEuler22.03。完整环境、拓扑与原始 profiler
明细未入库；不能仅凭 manifest 声明验证 HCCS 路由和完整栈身份。

| 产物 | 可保留内容 | 使用限制 |
|---|---|---|
| p0a_budget_acl / p0b_budget_torch | 各卡各阶段容量快照 | ACL 未检查返回码；两 API 分配生命周期不同；after_init 与 release 不可混用 |
| p1a_runtime_pools | allocator/GEMM 峰值与 free 快照 | graph 35.65MB 全在 warmup 差额中；净 capture 差0不证明零池；其余不是已隔离的纯开销 |
| p1b_hccl_batch_rank0/1 | 两 rank 的批内墙钟观测 | busbw 少2倍，见 review 修正；ReduceScatter shape 错误需重测；没有数值校验；不外推拓扑 |
| p2_swiglu / p2_moe_route | 指定 FP32 fixture 的数值对比 | 组合 torch_npu 实现，不是官方融合核或真实模型 forward |
| p2_mrope / p2_mrope_identity | 通用 RoPE 设备差异与互相冲突的分段结果 | 分段脚本失败，成功文件缺脚本；参数不等于本仓 Qwen；identity 不采信 |
| p3_per_op_qwen | 23条探索性算子记录 | BF16代理、部分语义/形状不等价；缺源场景；analytic ratio 不作校准 |
| msprof_by_shape / msprof_op_summary | 4个GEMM形状的task摘要 | 缺原始明细及聚合命令；n=23可能含3次预热；不得与独立墙钟相减称 launch |
| p2_bench_gemm / p2_bench_bw | 单次微基准 | 没有重复分布/完整采样身份，不是稳定锚点或官方spec |
| env / diag_hccl | 局部环境日志 | diag不含HCCS transporttype；不证明8卡拓扑 |
| scripts/ | 原始采集 provenance | 有已知错误、绝对路径与缺输入；仅审计，不推荐原样复跑 |

后续采数须输出到新目录，保留所有原始样本/输入/命令/源版本/SHA256。
具体整改、少卡测试及 profiling 校对见 [规划](../../ascend-validation-plan.md)。
不新增 DEVICES/PROFILE_SETS，不解除模型执行缺口，不授予部署排名。
