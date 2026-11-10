# 投机方法耗时与排名资格复核（2026-11-10）

## 范围与分工

- 外层 Codex：独立重放 MTP、chain 方法和基线；核对显存/比较资格、工作台展示、回归与最终验收。
- 项目工具 DSH/DeepSeek：任务 `1f4be066-4857-40b9-bc6c-76eb56bee088`，只读审查方法计数与时间线；已返回并验收。它只读源码并用 `py -3.12` 作进程内解析重放，未改源码、未跑 GPU 或全量 pytest。Codex 据此完成修复与唯一全量验收。
- 固定候选起点：本仓 `18f4106`；参考实现 [vLLM speculative decoding 官方文档](https://github.com/vllm-project/vllm/blob/main/docs/features/speculative_decoding/README.md) 与仓内 vLLM `dc74b8cb` pin。外部文档于 2026-11-10 查阅；源码阅读不等于部署运行验证。

## 复现（默认 H100 SXM ×8、TP8/EP1、B1、ctx32k、W8A8/FP4E、PCIe5 链路输入）

`run_v2(ScenarioV2(method_id=..., k=...))` 的原始结果：

| 方法 | K | M01 TPOT | M04 单轮 | verify | 草稿 | M06 known_peak/r0 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0 | 112.212623 ms | 112.212623 ms | — | — | 待单列 |
| mtp | 1 | 0.141926 ms | 0.248371 ms | 0.232143 ms | 0.016229 ms | 2,699,487,168 B |
| chain_draft | 4 | 0.102440 ms | 0.312521 ms | 0.300801 ms | 4×0.002930 ms | 2,699,487,168 B |
| chain_moe_proxy | 4 | 0.102440 ms | 0.312521 ms | 0.300801 ms | 4×0.002930 ms | 2,699,487,168 B |

基线的逐层 TP 集合通信计入 112.21 ms；三个方法的 verify 分支仅取 `stage_total(price_rows(...))`，未加 `_collectives_cost`，导致同一个目标模型的验证耗时与基线口径不一致。两个 chain 方法当前都由 `tiny_gqa` 合成结构计价，因此时间完全相同；其 M06 仅含 target 内存。MTP 使用目标首层作为微头模板，当前亦未把微头驻留成本纳入同卡账本。这些数值是离线解析重放，不是 H100 实测。

修复后同输入的 TPOT：MTP **64.1470 ms**，chain_draft 与 chain_moe_proxy 均 **36.8217 ms**；三个 verify 均包含约 112 ms 的逐层 TP 集合通信。DFlash 的 profile 替换路径也显式保留其 verify 通信。上述链路仍用 PCIe5 固定延迟假设，并非 H100 组内 NVLink 实测；不能把修复后的绝对时间视为硬件预测精度证明。

比较器与 M02 已增加方法配方资格：MTP 微头模板以及两个合成 chain 方法保留解析数值，但拒绝进入部署排名。工作台默认矩阵只显示 DFlash/MTP，研究示例在方法选择器标明；CLI 将不排名原因指向逐项能力检查，避免截断的原始对象串充当摘要。

DeepSeek 另查出 DFlash 草稿阶段也漏了同组 TP 通信（6 层 ×2 次）。加入后，CLI 必需场景（α=0.72、K15）的 TPOT 从 31.6983 ms 变为 **36.4300 ms**、草稿阶段约 16.84 ms，其中 16.8108 ms 是当前 PCIe5 假设通信。`dt_pipeline_sim` 现按独立草稿设备/TP 计价，M18 使用所有草稿步之和；target 与 draft 两组都登记运行时未知峰值，补 measured 预算也不会凭不完整账本进入排名。拒绝采样/KV 回退与 hidden 提取仍无独立耗时，作为**显式零占位和资格缺口**保留。

`chain_draft` 表示逐 token 自回归草稿，再由目标模型验证候选；当前实例只是 `tiny_gqa` 合成草稿，缺真实 checkpoint 和部署配方，适合研究演示。`chain_moe_proxy` 与它共用同一个 dense `tiny_gqa`，没有 MoE 草稿差异，因此没有独立产品对比价值；保留 API 研究入口，移出默认矩阵。MTP 是真实技术，见 [vLLM MTP 说明](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/mtp/)，但本仓只按 Qwen3.6 目标首层代理微头，缺原生微头 forward pin；其他模型明确返回 unsupported combo，不再抛底层 KeyError。

DeepSeek 还发现以下未解的解析精度问题：MTP 的层数与草稿步数混同、共享 lm_head 读流量偏小；chain 第 0 步 catch-up 权重恒消掉、批次轴固定 B1、DT 草稿设备/精度/TP 未贯穿。它们因方法资格被阻断，不作为实测结论。要把方法升级为可排名实现，须先取得真实草稿/微头 forward 与配对部署源码，再修计数并做匹配 GPU 校准。本轮没有为未知语义填造数值。

## “known_peak … allocatable=None” 的含义

2,699,487,168 B 是当前非 DFlash 方法的**已建模对象下界**，不是完整峰值。`DeviceBudget.allocatable_bytes=None` 因本仓没有目标卡可分配预算实测；NCCL、CUDA graph、kernel workspace 等运行时内存仍缺尺寸。`known_peak < nominal` 只能排除“已知下界超过标称容量”，不能证明可部署。界面应简短提示原因并保留明细，不通过隐藏提示或填写标称容量来取得资格。

## 达到实际部署排名资格的证据路径

1. 固定模型、草稿 checkpoint、后端 commit、TP/EP、精度、batch、上下文、block、graph/调度设置及计时边界；方法配方应对应真实可运行实现。合成草稿和假想代理继续作为研究示例，不进入部署排名。
2. 在目标设备组逐 rank 采集初始化后的可用显存、运行时内存池与同负载峰值；来源包含 GPU/驱动/CUDA/后端版本、采集点和原始 trace/hash。`torch.cuda.mem_get_info` 提供全局空闲/总量；`max_memory_reserved` 只覆盖 PyTorch 分配器，不能单独代表 NCCL/CUDA 外部分配。参见 [PyTorch mem_get_info](https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.mem_get_info.html)、[max_memory_reserved](https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.max_memory_reserved.html) 与 [内存可见性说明](https://docs.pytorch.org/docs/stable/torch_cuda_memory)。本机没有 `nvidia-smi`，本轮无法采集。
3. 采集每轮接受序列、draft/verify/通信延迟和同条件 E2E 请求轨迹；至少覆盖 B1/B8、ctx 8k/32k、重复运行与采样策略。按仓内 G-D1/G-D5/G-D6/G-D7/G-D8 的准入要求核对，而非把 T0 roofline 或合成 profile 视为 measured。
4. 比较器需实际执行方法配方、容量、同计时边界、结果种类、资源政策与质量要求谓词；当前 `ComparisonPolicy` 列出了部分谓词，但 `compare_runs` 只执行结构/容量检查。显示数值与允许参加实际排名是两个状态。

当前工作台仍可显示解析估算供假设探索；在上述证据齐备前，不能宣称实际部署排名。

## 本轮验收与剩余边界

`py -3.12 -m pytest tests -q`：**180 passed，1 warning**。警告来自 `tests/test_consistency.py` 的 Windows 子进程输出读取线程，发生 `UnicodeDecodeError`；测试退出码为 0。本轮没有把它当成本次投机修复的通过证据，也没有改动该无关路径。`node --check simverse/static/workbench.js` 与 `git diff --check` 通过。离线回归验证了 verify/draft TP 通信、DT 流水草稿账本与资格、缺失阶段成本、V4.1 MTP 明确失败，不证明 GPU 绝对时延。
