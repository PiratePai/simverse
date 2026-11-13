## R24：V4.1 SwiGLU clamp + compress_ratios 尾部 source-checked 审计与建模

独立 PR。详细报告：[docs/validation/v41-swiglu-tail-20261113.md](../../validation/v41-swiglu-tail-20261113.md)

### 源码 pin

- **swiglu-tail-pin.json 13 文件**（`docs/rebuild-20261106/references/vllm/swiglu-tail-pin.json`，源文件落盘同目录）
  - vllm-project/vllm @`f358f0578a83f4e1fe6fc49a492d3ae9362de8e5`（与 R04/R05/R20/R21/R22 既有 pin 同一上游快照；本地克隆 `git cat-file` 取固定提交对象，未 checkout）
  - **blob SHA1 对仓内 vllm-tree.json 逐项校验** + SHA256/size 清单；tests/test_v41_swiglu.py §1 三重复验；其中 3 文件与 csa2/moe pin 重叠（blob 一致，交叉验证）
  - 范围：语义核 activation.py（SiluAndMulWithClamp）、fused MoE 消费链（config.py/activation.py/utils.py/marlin_moe.py/layer.py）、模型接线（deepseek_v4 nvidia+cpu model.py）、尾部归属（deepseek_v41/attention.py、transformers_utils/deepseek_v4_utils.py、dspark.py、sparse_swa.py）、vLLM 数值等价测试（tests/kernels/core/test_activation.py）
  - 未读/未执行：csrc CUDA 核源码、masked padded 路径运行时、SWIGLUOAI/SWIGLUSTEP 家族实现细节（V4.1 域外）

### 语义核对结论（逐行号，见报告 §1/§2）

| 项 | 结论 |
|---|---|
| clamp 语义 | `gate=clamp(x[:d], max=limit)` **单侧**（负 gate 原样通过）、`up=clamp(x[d:], ±limit)` **双侧**、`out=gate·sigmoid(alpha·gate)·(up+beta)`；V4.1 无 swiglu_alpha/beta 键 → alpha=1.0/beta=0.0 → `silu(clamp(gate))·clamp(up)`（activation.py L208-250） |
| 触发条件 | MLP 侧 `swiglu_limit is not None`（model.py L176-183）；Triton 侧 `HAS_LIMIT=swiglu_limit>0`（utils.py L812）——**值 0 两路分叉**（MLP 病态 clamp(0)、Triton 关 clamp）→ config 域拒绝 0/负/NaN/inf/缺失 |
| 位置（消费两路） | 路由专家 w13→激活→w2：clamp_limit 回退链 `gemm1_clamp_limit or moe_config.swiglu_limit`（fused_moe/activation.py L176-197）；**注册执行域 Marlin 经 apply_moe_activation 消费同一 clamp**（marlin_moe.py L140-158）。共享专家 MLP.act_fn=SiluAndMulWithClamp（model.py L176-183/L331；CPU 变体同构） |
| 域外变体 | native MegaMoE `MegaMoEActivation("swiglu", clamp=...)`（model.py L393-394）语义同源但 SM100 域外；SWIGLUOAI（7.0/1.702/β=1）/SWIGLUSTEP 家族常数域外 |
| 尾部归属 | `[40:43]` 逐项归 **DSpark/nextn 草稿层**（层号 40+i）：dspark.py L93-96/L128-136 以 `layers.{L+i}` 构造；attention.py L375-386 按层号取 `compress_ratios[layer_id]`（**注释原文 "MTP layers included"**；else 分支 0=纯 SWA）；deepseek_v4_utils.py L6-18 派生路径补 `[0]*num_nextn_predict_layers` |
| 尾部用途 | official 尾部全 0 → 草稿层纯滑窗注意力；`is_backbone` 门控（L399-401）→ 草稿层**永不为 kv/index 源**；后端只按**去重层型集合**消费（sparse_swa.py L494-502）→ 尾部 0 与主干 0 同组，无新缓存组/plan |
| 尾部域（本仓） | 长度 ∈ {L, L+mtp}（超出 = 死配置拒绝）；尾部值 ∈ {0,1,2}；**非零尾部域外拒绝**——源码允许草稿层消费压缩 KV，但 DSpark 草稿前向未建模、无 official 证据，不猜语义 |

### 变更摘要

- `simverse/count/moe.py`：`moe_rows(..., act_recipe=extras.swiglu)` 追加 `moe_act`/`moe_shared_act` 行（6 算术 op/元素 = clamp 3+sigmoid 1+乘 2；**字节全 0**——中间缓存字节已归 moe_routed/moe_shared 单一所有者防重复计费；TP 关 exact/EP 开 expectation 与 moe_routed 同口径）
- `simverse/count/work.py`：build_model_work 读取 extras.swiglu 透传；Qwen36 等家族无该键 → 行为不变（R06 冻结口径维持，跨模型差异登记 registry limits）
- `simverse/model_ir/families.py`：swiglu_limit 域校验 + `extras.swiglu` + mapped 注记（行号证据）；compress_ratios 尾部域校验 + `kv_spec.draft_compress_ratios` + 尾部 gap 解除；**附带补漏 candidate_block_size 入 struct 集合**（R20 注记因键漏登记未生效，官方资产长期误挂"未登记 config 键"缺口，G9 断言被 text_config 前缀绕过未暴露）
- `simverse/formula_registry.py`：`v2.swiglu_clamp_act`；`model_ir/catalog.py`：checked_paths/unverified_paths（swiglu_deploy_variants、dspark_draft_compress）/evidence_refs/note；`importers/source_map.py`：`sm-v41-swiglu-tail`（both_sides_covered=True）+ 旧条目缩窄；`validation/registry.py`：ev-vllm-v41-swiglu-tail
- `tools/replay_v41_swiglu.py` + `docs/evidence/swiglu-tail/`（before/after/attribution/provenance；before 在 main b8585a1 干净生产代码捕获）；R21/R22 replay check 升级为"内部一致 + 当前账本对 R24 证据"两层契约（先例：R22 对 R21）
- 文档：报告 / capability-gaps（G-C15 解除；G-C1/G-C14 收窄）/ IMPLEMENTATION_STATUS（399）/ ROADMAP R24 行 / FOCUS_STATUS
- `.gitattributes`：R24 证据包 binary（同 R05/R20/R21/R22 先例，防行尾转换破坏 SHA256）

### 数值变化（默认 TP8/B1/ctx32768 replay，quant=V41_QUANT）

| 字段 | before → after | 归因 |
|---|---|---|
| decode flops | 7,176,453,222 → 7,176,937,062（**+483,840**） | 40 层 ×（moe_act 10,368 + moe_shared_act 1,728）= 40×(6·288·6 + 288·6)，手算对账 |
| decode rows | 1,007 → 1,087（+80） | 40 层 × 2 新行 |
| 全部字节字段 / M06 known_peak | 0（50,128,019,904 不变） | 字节单一所有者；clamp 不改布局/记录宽度；尾部全 0 无新缓存组 |
| M01 | 116.30292042 → 116.30292075 ms（+3.26e-07） | 解析计时微增 |

面外行逐字段冻结（AUDITED={moe_act, moe_shared_act}）；`--check` 无漂移复算。

### 结果资格

- **V4.1 config_coverage_complete 自此 True**（最后一个 unsupported 叶键解除）
- **structure_complete 仍 False**——仅由稀疏 MLA 注意力核本体（flops 4d+5 v0.1 移植）的 forward 证据阻断；部署排名资格不因本模块解除
- 证据分层：全部 source-checked；未执行 vLLM kernel/CPU forward/GPU（clamp CUDA vs native 等价以 vLLM 在册测试登记，本仓未跑）

### 测试原始结果

```
py -3.12 -m pytest tests -q               → ZCode 宿主（new-viewer-r24 工作树）：399 passed in 12.97s，exit 0
py -3.12 -m pytest --collect-only -q tests → 399 collected（367→399 = tests/test_v41_swiglu.py 32 项）
py -3.12 tools/replay_v41_swiglu.py --check → check OK，exit 0
py -3.12 tools/replay_v41_moe.py --check    → check OK（R21 内部一致 + 当前账本 = R24 证据），exit 0
py -3.12 tools/replay_v41_quant.py --check  → check OK（R22 内部一致 + 当前账本 = R24 证据），exit 0
py -3.12 -m simverse.cli run-v2            → status=completed，exit 0；structure gate 保留（model_structure_complete unsupported）
node --check simverse/static/workbench.js  → exit 0
```

注：`tools/replay_v41_csa2.py|mhc.py --check` 在 main 上即因 R22 act=8 域校验失败（其冻结场景 act=8 已被 R22 契约否定）——本 PR 未改变该状态，两工具的证据内部一致性仍由各自测试直接读 JSON 校验。

### 剩余限制（不宣称完成）

clamp kernel 数值等价未执行；masked/pad-aware 运行时差异未入账；SWIGLUOAI/SWIGLUSTEP 域外；MegaMoE SM100 执行变体；DSpark 草稿层本体与非零尾部按域外拒绝（**未为解除门禁猜测**）；Qwen36 silu 微操作维持 R06 冻结口径（跨模型口径差登记）。
