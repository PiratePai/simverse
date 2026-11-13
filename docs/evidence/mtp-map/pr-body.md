# R23：Qwen3.6-35B-A3B 原生 MTP forward 证据与仿真映射

## Summary

把 Qwen3.6 原生 MTP 微头从"目标首层（GDN）模板 × mtp_layers 研究近似"
（`sm-qwen36-mtp-unverified`，K≤n_predict 保守拒绝）升级为**双栈官方实现
逐行核对的 source-checked 建模**：vLLM @`10a854ed` 8 文件 pin
（qwen3_5_mtp / qwen3_next_mtp / config.speculative / eagle.utils /
mtp+target_dependent_ar speculator / 2 个 config 测试 + e2e 测试，
逐文件 SHA256+blob sha1 对 GitHub contents API 校验）+ SGLang @`1b2870d`
1 文件对照。不依赖 V4.1 工作分支；DFlash 与 MTP 保持分立方法（源 pin/
程序图/结构互不通用）。

查明并建模的官方事实：checkpoint `mtp.*` 张量与加载重映射（只消费
mtp.*/embed_tokens/lm_head）；combine=双预归一+cat+fc(2h→h, ColumnParallel
gather, **无激活/无残差直加**)；层=**full_attention**（gated GQA+MoE，
几何=目标 full 层——不是目标首层 GDN）；每轮 K 均匀步、步 j 层 j%n_predict
**循环复用**（K>n 合法，官方 e2e K=3×MTP1 在册；撤销 K≤n 旧拒绝）；
colocated 经 load_eagle_model 物理共享目标 embed/lm_head（零字节）；
草稿自有 per-token KV、被拒步回退。

## 改动表

| 层 | 文件 | 改动 |
|---|---|---|
| pin | assets/identity/vllm_qwen36_mtp_pin/（8+manifest）、sglang_qwen36_mtp_pin/（1+manifest） | 新增固定字节 pin |
| ModelIR | simverse/model_ir/families.py | extras["mtp"] 结构 spec；两键 mapped 行号注记；mtp_use_dedicated_embeddings=true 拒绝 |
| count | simverse/count/draft.py | mtp_step_rows 逐步重写（embed/双预归一/fc/full-attn MoE 层/final norm/lm_head/KV）；mtp_kv_rewind_row；mtp_layer_model |
| methods | simverse/methods/plans.py | plan_mtp v0.3（K 循环、无 catch-up、prefill 登记 independencies） |
| pipeline | simverse/pipeline.py | 逐步阶段+草稿通信+fc all-gather；mtp_serving_integration；dt 双模式拒绝；memory/refs 接线 |
| memory | simverse/lowering/memory.py | mtp_extra_allocations/mtp_transient_allocations；colocated mtp_model（含零字节共享引用） |
| 证据 | catalog/registry/source_map | method_status["mtp"]=source-checked；3 条证据；sm-qwen36-mtp |
| 测试 | tests/test_qwen36_mtp_r23.py（25 项）+ G10/G13/G16/model_ir/product_scope/pipeline_v02 更新 | 结构/计数/方法/管线/内存/负例/replay 冻结 |
| fixture | tools/probe_qwen36_mtp_numeric.py + tests/fixtures/qwen36_mtp_r23_numeric.json | 独立数学对照（双实现互核） |
| replay | tools/replay_qwen36_mtp.py + docs/evidence/mtp-map/ | before/after/attribution（--check 无漂移） |
| 文档 | ROADMAP/IMPLEMENTATION_STATUS/FOCUS_STATUS/capability-gaps(G-C15)/validation 报告 | 同步 |

## 复算数字（k=1 同场景 replay；K 域 1→K≥1 单列）

- M01（B1/ctx32768/TP8/W8A8+FP4E 假设接受率 0.75^k）：64.55 → **66.96 ms**
  （旧 GDN 模板漏 lm_head/KV、误计 GDN state，两个方向都错）
- colocated steady：2,699,569,088 → **2,841,004,480 B**（+141,435,392 =
  MTP 层常驻 ~108MB/卡@TP8（W8 draft 档）+ fc/norms 1.05MB + 草稿 KV
  32MB + 临时下界；旧模型完全没有 MTP 驻留对象）
- draft_step_0（K=1）：19 行 GDN 模板 → 24 行 pinned 结构（含 KV 回退浪费注记；+fc/双预归一/
  final norm/draft_lm_head/attention KV，−GDN 递推/卷积 state）
- target_decode / verify：逐字段冻结（attribution.json）

## Testing

- `py -3.12 -m pytest tests -q`：**425 passed**（R24 合入后 399→424 = R23 25 项；验收守恒回归 +1）
- `py -3.12 tools/replay_qwen36_mtp.py --check`：OK（无漂移）
- `py -3.12 tools/replay_v41_quant.py --check` / `replay_v41_moe.py --check`：
  OK（V4.1 证据不受本 PR 影响）
- `py -3.12 tools/probe_qwen36_mtp_numeric.py`：双实现 max diff 4.97e-13，
  30/30 GEMM 账本对账、attention 键数/KV 行精确相等、手算锚点通过
- `py -3.12 -m simverse.cli run-v2`：exit 0
- `node --check simverse/static/workbench.js`：exit 0
- 已知预存问题（非本 PR 引入，main 同样复现）：`tools/replay_v41_csa2.py`
  是 R20 时代工具、无 `--check` 模式，其冻结场景（act=8）与 R22 的
  act=16 域冲突——重放场景属 R20/R22 证据维护面，本 PR 不动其冻结产物

## 边界（不冒充）

- source-checked ≠ 已执行/已部署：vLLM/SGLang 模块未在 CPU/GPU 执行、
  真实 checkpoint mtp.* 未本机加载、无实测延迟；fixture 是随机权重小尺寸
  数学对照，模型级 cpu_forward 保持 not-run
- 部署排名仍阻断：mrope 未核（forward_source=partial-source）、无 GPU
  实测、预算 unknown、接受率为显式假设输入
- 未建模变体登记在 G-C15（多模块 MTP、量化草稿头、local_argmax、
  probabilistic 采样档、CUDA-graph padding、SGLang NPU 分支）；草稿
  prefill KV 同步为一次性成本，不在稳态 TPOT（cache_policy 登记）

## 合入前验收修复（2026-11-14）

草稿私有层原先在 count/cost 中误用目标专家 FP4 档，memory 则使用草稿 W8 档。
现统一私有层的 count/cost/通信与 memory 档，目标共享 lm_head 仍沿目标档；
移除 MTP final norm 的重复驻留 2,048 B。固定 K=1 replay 的草稿专家
weight_read_expected 1,572,864→3,145,728 B，M01 66.96182490→
66.96209320 ms；target_decode/verify 不变。`--check` 现同时核验成本、
指标与逐行归因。新增一项守恒回归；测试总数 425。
