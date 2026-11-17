# Ascend 910B3 实测证据（2026-11-17）

采集对象：**Ascend910B3 ×2（物理卡 6、7）**，`ASCEND_RT_VISIBLE_DEVICES=6,7`。
采集方式：`torch_npu` 2.9.0.post9 + CANN 9.1.0 + `msprof`；脚本见 `scripts/`。

## 环境

| 项 | 值 |
|---|---|
| SoC | Ascend910B3（单卡标称 65,536 MB） |
| CANN | 9.1.0 |
| Driver | 26.0.rc1（V100R001C10SPC001B260） |
| Python | 3.12.3 |
| torch / torch_npu | 2.9.0+cpu / 2.9.0.post9（cp312 aarch64） |
| OS | openEuler 22.03（aarch64） |
| 拓扑 | 8 卡全互联 HCCS（`npu-smi info -t topo`）；本次只使用卡 6、7 |

## 产物与对应条目

| 文件 | 对应条目 | 内容 |
|---|---|---|
| `p0a_budget_acl.json` | G-D7 可分配预算（ACL 视角） | total / 初始化后 free / 1-4-8 GiB 分配步进 / 释放回收 |
| `p0b_budget_torch.json` | G-D7（torch 视角） | `torch.npu.mem_get_info` 同口径 |
| `p1a_runtime_pools.json` | R02b 运行时池 | 分配器 padding、GEMM workspace、NPU graph 捕获额外驻留 |
| `p1b_hccl_batch_rank0.json` / `rank1.json` | R17 通信档（2 卡） | allreduce / allgather / reduce_scatter / all_to_all / broadcast × 10 档大小 |
| `p2_swiglu.json` | G-C18 SwiGLU clamp | 11 组边界/NaN/inf 用例与 stdlib oracle 对账 |
| `p2_moe_route.json` | G-C13 MoE 路由 | 5 组场景（随机/全平手/NaN/下溢/>20 护栏）专家 id 与权重对账 |
| `p2_mrope.json` / `p2_mrope_identity.json` | G-C16 MRoPE 文本域 | 设备等价性（vs CPU fp32）、算法精度（vs fp64）、三行分段恒等 |
| `p3_per_op_qwen.json` | R18 前置（单算子实测） | 仓库账本 23 个 `target_decode` 算子的 eager 墙钟 + M 扫描 |
| `msprof_by_shape.txt` / `msprof_op_summary.txt` | G-D6 KERNEL_ONLY 列（NPU 侧） | msprof `op_summary` 按形状聚合的 kernel-only 时长 |
| `p2_bench_gemm.txt` / `p2_bench_bw.txt` | 能力锚点 | GEMM 8192³ / HBM copy / 逐元素 / softmax |
| `env.txt` / `diag_hccl.log` | 复现身份 | 版本快照、HCCL 传输类型诊断 |

## 计时口径

- `runtime-api`：内存/容量类，直接读运行时 API，不涉及计时。
- `device-sync wallclock`：`torch.npu.synchronize()` 包住 N 次调用；**含 host 下发**。
- `device-executed`：语义对账，比的是数值，不是时间。
- `KERNEL_ONLY`：msprof `Task Duration(us)`，纯 device 执行时间。

两者之差即 launch 开销：M=1 GEMM 墙钟 18.74 µs、kernel 7.08 µs → 约 11.7 µs。

## 未入库

msprof 明细目录、HCCL plog 归档原始文件体积大且含机器路径，留在采集机；本目录只保留蒸馏后的结论文件。

## 限制（不能证明什么）

1. 实测峰值不等于官方 `spec` 峰值，**不得**填入 `DeviceSpec.peak_by_dtype`。
2. 只测了 2 卡，不代表 8 卡全互联；通信数值不可外推到 910C/950 或 H 系列。
3. `attention_core` 用稠密 SDPA 代理，**不是** FlashMLA/稀疏 MLA 核；`moe_routed` 用 bmm 代理，不是融合专家核。
4. `gdn_recurrence`、`attn_output_gate`、`shared_gate`、`v_norm_gated` 无等价实现，未测。
5. 逐元素/归一算子（rmsnorm 等）是多个 kernel 拼装，测得的是 launch 次数，不是算子效率。
6. 本证据**不解**任何 CUDA 执行变体缺口，也不构成部署排名资格。

## 复现

见 `docs/validation/ascend-910b3-20261117.md` §7。`p1b_hccl.py`、`p3_per_op.py` 内含采集时的绝对输出路径，复跑前需指向本目录。
