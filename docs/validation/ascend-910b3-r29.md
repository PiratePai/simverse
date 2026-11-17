# Ascend 910B3（卡 5/6）R29 重采与验证报告

更新：2026-11-17。采集目录 [evidence/ascend-910b3-r29](../evidence/ascend-910b3-r29/)。
本轮**只在物理卡 5、6**上执行（`ASCEND_RT_VISIBLE_DEVICES=5,6`），依据
[昇腾优先验证计划](../ascend-validation-plan.md) 的 P0 项（A-N1、A-N2）与少卡路径。
R28 归档（`../evidence/ascend-910b3/`）的字节保持不动，本报告不重写任何旧数值。

## 1. 结论摘要

| 项 | 本轮结果 | 不能推出 |
|---|---|---|
| A-N1 ACL 预算 | 全部 ACL 调用返回码为 0；1/4/8 GiB 三阶段与释放后快照齐全，context 句柄显式回传 | 不等于满容量可分配，也不证明无碎片 |
| A-N1 集合通信 | 双 rank 五类 collective × 10 个尺寸，**100/100 次数值校验精确通过**；shape 契约与 busbw 系数已改正 | 不代表 HCCS 路由（transport 未采集），不外推 TP>2 |
| A-N1 graph 驻留 | eager/预热/baseline/capture/replay/释放 六阶段可复算；capture 相对 baseline 净增 56.6 MB free 差 | 不是整模峰值预算 |
| A-N1 MRoPE | 参数域改到本仓 Qwen3.6（head 256 / rotary 64 / θ=1e7 / section [11,11,10]）；fp64 oracle 对照最大绝对差 2.75e-4（pos=32767） | 不是融合核验证：`torch_npu.npu_mrope` 在该参数域返回 161002 |
| A-N2 forward | vLLM 0.32.0 + vllm-ascend 0.19.1rc2 在 NPU 上跑通 2 层随机权重 fixture（token_ids `[20,20,20,20]`） | `full_model_forward=false`、`random_weights=true`、无性能/接受率结论 |
| A-N3 TP2/EP2 切片 | 列/行并行、GQA 头分片、EP2 路由往返共 **7/7 通过**（两 rank 同结果）；EP2 覆盖 2 个空专家、逆置换恒等 | 切片级、随机权重、均衡路由；不是整模 TP2，也不外推 TP>2 |
| A-N5 profiling 校对 / A-N6 E2E | **未执行** | — |

不解除 G-D7、R02b、R17、R18；不加入 `PROFILE_SETS`；不改变 DEVICES、V41_DEVICE_FAMILY、
模型执行状态与部署排名；`calibration_admission` 仍为 not-admitted。

## 2. 环境身份（与 R28 不同，不可拼用）

| 项 | R28 声明 | R29 实测 |
|---|---|---|
| 物理卡 | 6/7 | **5/6** |
| torch / torch_npu | 2.9.0+cpu / 2.9.0.post9 | **2.10.0+cpu / 2.10.0.post4** |
| vLLM / vLLM-Ascend | 未记录 | **0.32.0 / 0.19.1rc2.dev2959+g450e55698** |
| CANN / driver | 9.1.0 / 26.0.rc1 | 9.1.0 / 26.0.rc1（一致） |
| Python / host | 3.12.3 / openEuler 22.03 aarch64 | 3.12.13（`/usr/local/python3.12.13`）/ 同 |

板卡细节（`env_identity.json`）：卡 5 `0000:02:00.0`、卡 6 `0000:41:00.0`，
Product `IT21HMDA6_N0`，Serial 102557029681 / 102557029781，固件 9.0.0.0.205。
**不同 PCIe 域说明这对卡不是同一交换平面**，因此本轮的通信结果只能代表这一对卡。

## 3. A-N1 逐项

### 3.1 ACL 预算（修正返回码与 context）

`raw/budget_acl.json`：`acl.init / set_device / create_context / malloc / free /
destroy_context / reset_device / finalize` 全部返回 0，`destroy_context` 以显式
context 句柄调用。两卡各阶段 free（B）：

| 阶段 | dev0 | dev1 |
|---|---:|---:|
| after_init | 65,062,694,912 | 65,060,560,896 |
| +1 GiB held（Δ） | −1,075,838,976 | −1,075,838,976 |
| +4 GiB held（Δ 累计） | −5,372,903,424 | −5,372,903,424 |
| +8 GiB held（Δ 累计） | −13,965,045,760 | −13,964,935,168 |
| after_release | 65,062,584,320 | 65,060,339,712 |

可复算事实：每次 `malloc` 的 free 差比请求值多 **2,097,152 B（2 MiB）**，
与 R28 的 `reserved−requested=2 MiB` 观测一致；这是分配粒度，不是"所有分配固定多用 2 MiB"的定律。

### 3.2 双卡集合通信（修正契约与 busbw）

`raw/collectives_rank0.json`、`collectives_rank1.json`：
world=2、bf16、`half` 无关、批内墙钟（N 次 collective 一个窗口），5 batch × iters，
all_reduce 输入用零点（避免连续 in-place 累加漂移），语义校验用非零输入单独做。

- 数值：50 组/rank × 2 rank = **100/100 exact**，无 mismatch、无异常；
- `reduce_scatter` 改为 list 契约（每个输入块与输出同形），不再出现旧采集的 n / n÷P 混配；
- busbw 系数按 nccl-tests：all_reduce `2(P−1)/P`（P=2 时等于 1）、
  all_gather / reduce_scatter / all_to_all `(P−1)/P`、broadcast `1`；
- 每 rank 的 send/recv 字节与 payload 分开记录，algbw 以 payload 定义。

256 MiB（每 rank 接收字节）观测：

| op | median ms | algbw GB/s | busbw GB/s |
|---|---:|---:|---:|
| all_reduce | 12.913 | 20.788 | 20.788 |
| all_gather | 15.144 | 17.725 | 8.863 |
| reduce_scatter | 16.069 | 33.409 | 16.704 |
| broadcast | 14.046 | 19.111 | 19.111 |
| all_to_all_single | 7.359 | 36.478 | 18.239 |

通信器初始化到首个 collective 的 free 差为 **444,956,672 B**（含 warmup 张量），
旧口径 445 MB 量级复现，但**不是纯 comm pool**。

### 3.3 graph 驻留（修正归因）

`raw/graph_pools.json`，workload `20 × relu(x @ W)`，2048² bf16，单进程：

| 阶段 | free (B) | allocated (B) | reserved (B) |
|---|---:|---:|---:|
| startup | 65,065,390,080 | 0 | 0 |
| eager 预热后 | 64,966,791,168 | 25,167,360 | 85,983,232 |
| baseline（预热后 empty_cache） | 64,966,791,168 | 25,167,360 | 44,040,192 |
| after_capture | 64,910,168,064 | 41,946,624 | 113,246,208 |
| after_replay_1 | 64,905,973,760 | 41,946,624 | 113,246,208 |
| after_replay_10 | 64,905,973,760 | 41,946,624 | 113,246,208 |
| 释放 + empty_cache | 64,905,973,760 | 25,167,360 | 44,040,192 |

可复算：capture 相对 baseline 的 free 差 **56,623,104 B**，其中 allocated 净增 16,779,264 B；
首次 replay 再增 4,210,304 B，第 2–10 次不再增长。旧 p1a 把"预热差额"整体算作 graph 驻留，
本轮把预热、capture、replay、释放分栏，互不相减。

### 3.4 MRoPE（修正参数域与索引）

`raw/mrope.json`：参数直接取自 `assets/config/qwen36_35b_a3b_main__config.json`
（sha256 入 manifest），domain = head_dim 256、partial_rotary_factor 0.25 → rotary_dim 64、
θ=1e7、`mrope_section=[11,11,10]`（和 32 = rotary_dim/2）、`mrope_interleaved=true`。

oracle 用 fp64 从参数独立构造（加倍拼接 + NeoX rotate_half）；设备侧用非融合路径
（NPU 上算 fp32 cos/sin + NPU 上旋转），尾部 192 维直通：

| pos | max abs | max rel |
|---:|---:|---:|
| 0 | 0 | 0 |
| 1 | 5.60e-08 | 1.23e-07 |
| 127 | 3.52e-07 | 2.28e-06 |
| 1000 | 1.40e-05 | 2.44e-05 |
| 32767 | 2.75e-04 | 4.19e-03 |

尾部直通逐位相等；文本退化（三分量相等）时装配频率与 T 分量差为 0。

**融合入口负例（重要）**：`torch_npu.npu_mrope` 在本栈只接受文档示例参数
（head=128、cache 宽=128、section `[16,24,24]`）；`[32,16,16]`、`[8,8,48]`、
head 64、head 256（rotary 64/128/256）全部返回 `aclnnRopeWithSinCosCacheV2` 错误码 161002。
因此 **Qwen3.6 的 partial-rotary MRoPE 无法用该融合核验证**，这是 A-N2 的待修阻塞项。

## 4. A-N3 少卡正确性与容量（TP2 / EP2 切片）

`raw/tp2_slice_rank0.json`、`rank1.json`（torchrun 2 进程，卡 5/6）。不加载整模，
用保留真实维度的算子切片做分片—重组对照，参考值在 CPU 以 fp64 独立计算：

| 用例 | max abs | 判据 | 结果 |
|---|---:|---:|---|
| MLP 列并行 fp32 | 3.60e-07 | 1.0e-05 | 通过 |
| MLP 列并行 bf16 | 1.82e-03 | 2.0e-02 | 通过 |
| MLP 行并行 fp32 | 1.76e-07 | 1.0e-05 | 通过 |
| MLP 行并行 bf16 | 2.66e-03 | 2.0e-02 | 通过 |
| GQA 头分片 fp32 | 1.31e-08 | 1.0e-05 | 通过 |
| GQA 头分片 bf16 | 3.47e-04 | 2.0e-02 | 通过 |
| MoE EP2 组合 | 1.13e-08 | 1.0e-05 | 通过 |

- 形状：hidden 2048 / intermediate 512 / seq 32 / q 头 16 / kv 头 2 / head_dim 256，
  与 Qwen3.6 文本路径同域；权重随机；
- kv 分片守恒：每 rank 1 个 kv 头，两 rank 合计 2 = 配置值；
- EP2：token 按 rank 切分，专家归属 0–2 → rank0、3–5 → rank1、6–7 刻意留空；
  发/收各 16 条/rank，总 64 条；**逆置换恒等为 true**；专家计数
  `[15, 9, 8, 16, 8, 8, 0, 0]`，空专家分支被真实覆盖；
- 显存峰值：两 rank 均 **3,624,960 B**（3.6 MB）。这是**算子切片**的占用，
  不能当作整模 TP2 的容量证据；真正的模型权重仍是 §5 里 30–33 GiB 的量级。

边界：路由是确定性均衡构造，不含真实倾斜分布；极端倾斜、丢 token、跨层 cache 与
通信/计算重叠都未覆盖，所以这条证据只回答"分片数学与路由往返是否正确"。

## 5. A-N2 forward 路径

`raw/forward_plan.json` 为 dry-run 计划（`*_exit=0`）。`raw/forward_execute.json` 与日志记录了
三次尝试：

| `gpu_memory_utilization` | 结果 | 关键数字 |
|---|---|---|
| 默认 0.9 | OOM | 已分配 30.42 GiB，尝试再分配 30.17 GiB，剩 29.98 GiB |
| 0.55 | OOM | 已分配 33.08 GiB，尝试再分配 32.83 GiB，剩 27.24 GiB |
| **0.2** | **成功** | 2 层随机权重 fixture，`max_model_len=128`、`enforce_eager=true` |

成功记录（`claims`）：`hardware_executed=true`、`hardware_verified=true`、
`full_model_forward=false`、`random_weights=true`、`performance_measured=false`、
`acceptance_rate_measured=false`；输出 `token_ids=[20,20,20,20]`、`finish_reason=length`。
执行侧同时记录 `stack_pair_validated=false`、`numerical_correctness_verified=false`。

**读法**：这只证明"该 fixture + 该栈能在卡 5/6 上把 vLLM 引擎跑起来并产出 token"，
不证明 Qwen3.6 整模、不证明数值正确性、不证明性能。fixture 权重本身约 30–33 GiB，
在 60.96 GiB 卡上留给 KV 的预算必须压到 ≤0.2 利用率才够，这本身就是一条容量事实。

## 6. 未执行与阻塞

- **整模 TP2**：本轮只做算子切片；把 30–33 GiB 的随机权重 fixture 按 TP2 铺到两卡、
  再叠 KV cache，需要先改 fixture 规模，属下一轮。
- **A-N5**（profiling 导入/校对）与 **A-N6**（匹配 E2E）：未做，缺原始 trace 与请求轨迹。
- **融合 MRoPE**：错误码 161002，需在官方版本矩阵内确认支持的 shape/section 组合。
- **HCCS 路由**：仍未验证；`transport` 在 collectives 记录中保持 `unknown`。
- **工具缺陷**：`tools/ascend_forward_probe.py` 运行时会在 cwd 生成 `runtime/`（含
  `config/`、`report/`），即使已传 `--runtime-dir`。本轮产生的 16 KB 目录已删除，
  未进入归档；后续应改为只在 `--runtime-dir` 下写入。

## 7. 复现

```bash
export ASCEND_RT_VISIBLE_DEVICES=5,6
source /home/p00929506/TQ/env.sh       # 等价 .bashrc 中的 CANN/ATB 环境
cd /home/p00929506/TQ/simverse/docs/evidence/ascend-910b3-r29/scripts
$PY r29_env.py && $PY r29_budget_acl.py && $PY r29_budget_torch.py
$PY -m torch.distributed.run --standalone --nproc_per_node=2 r29_collectives.py
$PY r29_graph.py && $PY r29_mrope.py
$PY -m torch.distributed.run --standalone --nproc_per_node=2 r29_tp2.py
$PY r29_manifest.py
```

收集脚本用仓库相对路径，输出固定写到 `../raw/`，每次覆盖同名文件即代表同一 run 的重跑；
跨 run 对比必须另建目录。
