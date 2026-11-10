# 旧仓数据资产审计（2026-11-10）

> **历史归档（2026-11-10 标注）**：审计结论与拒绝/推迟理由保留原样；涉及的
> Qwen3 历史模型已移出活跃 catalog，不能作为两款主模型的产品/部署证据。

范围：pi-viewer（@c5909a7 工作区，只读）与 model-verse（@c38e100）已有 profiling/
E2E 数据能否进入 new-viewer 校准闭环。结论按"证据最完整的一条增量路径"原则裁决；
不合格者记录明确拒绝理由，不制造覆盖。

## 审计结论总表

| 资产 | 位置 | 裁决 | 理由 |
|---|---|---|---|
| H200 Qwen3-30B-A3B-tiny（第一条闭环） | compute/h200/Qwen3-30B-A3B-tiny | 已接入（loop1） | 逐算子形状与 qwen3_30b_a3b 一致；2026-11-10 复审完成 |
| **H200 Qwen3-235B-A22B 全套** | compute/h200/Qwen3-235B-A22B | **接入（loop2，本轮）** | 真 128E 模型；官方身份已补（HF main @8efa6172，config SHA256 0ecd5d6f…）；TP2/8 linear+attention+kernel_gap 准入 |
| H200 235B moe.csv（TP8/EP1 真 128E vllm_fused） | 同上 | **defer（首选增量路径）** | 数据在册且为真专家 grouped GEMM；但新仓 count 的 MoE 行为整模型口径、memory 按 experts//tp，per-rank 分片账本（V04）未统一，实测替换会与整模型解析混尺。V04 落地后升级准入 |
| H200 235B moe.csv（TP1/EP4） | 同上 | defer | DP4 部署（每卡 32 专家、token 复制），与 TP1 单卡场景口径不符 |
| H200 235B linear/attention TP1 行 | 同上 | defer | sharded 组 TP1 行为空 + 235B 无 TP1 kernel_gap 行；TP1 场景由 30B-tiny set 覆盖（同环境栈） |
| A800 qwen3-a3b-30b-moe（真 30B 128E） | compute/a800/qwen3-a3b-30b-moe | **拒绝（measured 准入）** | 数据本身合格（128E/top-8/BF16/CUDA_EVENT/TP1-8/EP1-128 网格、standard_fused_topk+softmax_renorm），但 CSV 无 gpu_name/driver/torch/cuda 等环境元数据列（同目录无 kernel_gap/synthesis 文件）→ 违反 MERGE_CRITERIA §3.2 测量元数据必备（G4 同型）；不降级冒充。补环境记录后可复审（并同样受 V04 分片口径约束） |
| H800 Qwen3-30B-A3B-tiny 全套 | compute/h800/Qwen3-30B-A3B-tiny | defer | 设备 h800 未注册（需先入 devices 注册表带来源）；另含 attention_combined/moe 数据可一并评估 |
| A100/H100 各模型（Llama/Qwen-72B 等） | compute/{a100,h100} | 不适用 | 模型族未在本仓 pin（dense GQA 为主）；无必需场景映射 |
| A800 qwen3-next-80b（reduced-l2/l20） | compute/a800 | defer | qwen3_next 身份 PENDING_ASSET（gated 官方仓）；且同 A800 无环境元数据问题 |
| pi-viewer network/（h100/a100/a800/a40） | data/profiling/network | defer | 无 H200 NVLink（G-D8 不变） |
| pi-viewer examples 冒烟断言 | frontier examples | 不作为 E2E 证据 | finite/positive 断言非留出实测记录（G-D5 判定维持） |
| model-verse 三点自拟合/对账 | LLM-Viewer-main | 不作为校准证据 | 同源自算（evidence-audit 已裁），非独立真实性 |
| E2E 请求级记录 | 两旧仓 | **不存在** | 全数据树无逐请求 TTFT/ITL/完成时间记录（G-D5 维持 P1） |
| KERNEL_ONLY per-op 时间列 | 各 kernel_only CSV | 不存在 | 只有 kernel_count 有值、median 列空（G-D6 维持：需 record_function 补采） |

## loop2（235B）准入要点

- **身份**：官方 config 经 HF `resolve/main` 获取（2026-11-10），revision
  `8efa61729e24bd65b1d152b5ab5409052aa80e65`（config.json lastModified
  2025-07-26，内容稳定）；与 pi-viewer 二手副本逐字段核对一致（其仅多自加
  `model_architecture_profile` 键）。catalog 注册 `qwen3_235b_a22b`。
- **TP 域 {2,8}**：sharded 算子（attn_pre_proj/attn_rope/attn_post_proj +
  注意力块）per-worker 形状与新仓 gqa 行 tp 切分口径一致；TP1 campaign 只采了
  replicated 组（emb/两个 norm），其 per-rank 全宽、TP 不变 → 以
  `tp_invariant_groups` 并入 TP2/8 域（emb 为全词表口径，与新仓未切分行同尺度）。
- **kernel_gap**：TP2 1.5269µs / TP8 1.2384µs（eager，host-ahead 纯设备间隙）；
  gap 补贴按计时区间边界计（loop1 2026-11-10 同政策）。
- **MoE defer 的工程前置（V04）**：统一 per-rank MoE 分片口径——count 行
  （现整模型 FLOPs/容量）与 memory（现 experts//tp）按一种配方对齐（235B 实测
  为 EP1+inter/tp），配 TP/EP 守恒断言；完成后 moe.csv:TP8/EP1 升级准入
  （gating_linear+routing_topk+shuffling+grouped_gemm 四算子块 ← moe_router+
  moe_routed 所有权映射），并按 G-D2 验收留出 p90。

## 不变约束

- 16 专家 tiny 数据不得替代 128 专家模型（loop1 reject 维持）。
- 235B 数据不得移植给 30B（专家数相同但 hidden/层投影不同；两 set 各自域门禁）。
- uniform 路由为测量假设（moe.csv load_distribution=uniform），未来 MoE 准入时
  在域文档显式登记。
