# Ascend 910B3 R29 原始采集（2026-11-17，卡 5/6）

**本目录仍未通过生产校准准入。** 先读 [R29 审查/使用边界](../../validation/ascend-910b3-r29.md)
与 [manifest.json](raw/manifest.json)。

与 R28 归档（`../ascend-910b3/`）的关系：R28 是原采集提交 `6a16fbe` 的字节留档，**不被覆盖**；
R29 是按其审查意见（[_审查报告_](../../validation/ascend-910b3-20261117.md) 第 2 节）整改后的新 run，
输出到独立目录、带完整身份与 SHA256。

## 本次执行的设备与栈

| 项 | 值 |
|---|---|
| 物理卡 | 5、6（`ASCEND_RT_VISIBLE_DEVICES=5,6`，逻辑 0/1） |
| Host | openEuler 22.03 SP4 aarch64，192 vCPU |
| CANN / driver | CANN 9.1.0 / driver 26.0.rc1（npu-smi 26.0.rc1） |
| Python | 3.12.13（`/usr/local/python3.12.13`） |
| torch / torch_npu | 2.10.0+cpu / 2.10.0.post4 |
| vLLM / vLLM-Ascend | 0.32.0 / 0.19.1rc2.dev2959+g450e55698 |

**重要差异**：R28 采集者声明的是 torch 2.9.0+cpu、torch_npu 2.9.0.post9、物理卡 6/7。
R29 的栈和卡都不同，因此**两组数据不能横向相减或拼接**，也不能用 R29 证明 R28 的采集栈。

## 产物

| 文件 | 内容 | 限制 |
|---|---|---|
| `raw/env_identity.json` | 栈/设备/板卡/版本文件/脚本哈希 | 板卡序列号可核对，HCCS 路由仍未验证 |
| `raw/budget_acl.json` | ACL 分阶段 malloc/free + 返回码 + context 句柄 | 空闲快照，不是容量保证 |
| `raw/budget_torch.json` | torch allocator 分阶段统计 | 与 ACL 不是同一分配器，禁止相减 |
| `raw/collectives_rank0/1.json` | 双 rank 五类 collective，1 KiB–256 MiB，含数值校验 | 批内墙钟边界；transport 未验证 |
| `raw/graph_pools.json` | eager/预热/baseline/capture/replay/release 分阶段 | 单进程单 workload，不是整模峰值 |
| `raw/mrope.json` | Qwen3.6 参数域 MRoPE 数值 + 融合入口能力探测 | 非融合路径 fixture 级；不证明 vLLM 集成 |
| `raw/forward_plan.json` | forward 预检计划（dry-run） | 计划不等于执行 |
| `raw/forward_execute.json` | vLLM 0.32 + vllm-ascend 在 NPU 上的 fixture 执行结果 | 2 层随机权重，非整模、无性能结论 |
| `raw/forward_execute.log` | 执行日志（含两次 OOM 负例） | 原始日志，未编辑 |
| `raw/tp2_slice_rank0.json` / `rank1.json` | TP2/EP2 算子切片的数值、路由与显存记录 | 切片级；路由为均衡构造，不代表倾斜分布 |
| `scripts/` | 本轮采集脚本 | 可复跑；路径自适应仓库根 |

## 未解除的缺口

不解除 G-D7、R02b、R17、R18，也不授予部署排名或校准准入。A-N3 只做了**算子切片**级
TP2/EP2 正确性（7/7 通过，显存峰值 3.6 MB，不是容量实验）；profiling 导入与局部校准
（A-N5）、匹配 E2E（A-N6）本轮未执行；`torch_npu.npu_mrope` 在 Qwen3.6 参数域返回
错误码 161002（见 mrope.json），属待修阻塞。
