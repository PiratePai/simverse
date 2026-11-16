# R25：V4.1 稀疏 MLA 注意力核本体（csa2_attention_core）审计与修正

日期：2026-11-16；分支 `codex/v41-sparse-mla-attention-r25`（自 main
ee70ce2 = R23 合入后）。任务：对 V4.1 稀疏 MLA 注意力核本体的 prefill/
decode、稀疏候选索引消费、因果边界、压缩记录读取、算量与流量逐项审计，
修正 `csa2_attention_core` 的"4d+5"v0.1 移植口径；以项目固定 vLLM 版本为
主要依据。**全部为源码阅读 source-checked；未执行任何 kernel、模型 CPU
forward 或 GPU 测量，不声称硬件验证**（边界与 R04/R05/R20 相同）。

## 1. 范围、身份与参考实现

R24 后 V4.1 的最后一个主干缺口 = 注意力核本体（flops 4d+5 为 v0.1 移植，
R20 报告 §6.6 明确留待本轮）。证据 pin：

- **vLLM 侧 5 文件**（@`f358f0578a83f4e1fe6fc49a492d3ae9362de8e5`，blob
  SHA1 对既有 `references/vllm-tree.json` 逐项校验）：
  `vllm/models/deepseek_v41/nvidia/flashmla.py`（DeepseekV4FlashMLAAttention
  消费链）、`vllm/v1/attention/ops/flashmla.py`（包装层）、
  `cmake/external_projects/flashmla.cmake`（外部项目 GIT_TAG）、
  `vllm/models/deepseek_v41/nvidia/specialized_decode/__init__.py`
  （decode_backend 门控）、`vllm/v1/kv_cache_interface.py`（同 spec 层
  共享 metadata builder 分组）。
- **FlashMLA 9 文件**（@`847ef308bb7a6828c8117301f5a5a84a095a90e8`，
  由 vllm cmake GIT_TAG 钉定——`vllm/third_party/flashmla` 在 tree 中仅
  占位 `__init__.py`，接口由 cmake `file(READ)` vendor）：Python 接口
  `flash_mla/flash_mla_interface.py`、`csrc/api/{sparse_decode, sparse_
  prefill}.cpp`、`csrc/cuda_kernels/smxx/decode/{combine/get_decoding_
  sched_meta}`、`csrc/cuda_kernels/legacy/sm90/decode/sparse/splitkv_mla.*
  （WGMMA 主核）`、`legacy/sm90/prefill/sparse/phase1.h`、`csrc/params.h`。
  逐文件 blob SHA1 对该提交 git 对象（blob-filtered clone）校验。
- 机器清单 [csa2-core-pin.json](../rebuild-20261106/references/vllm/
  csa2-core-pin.json)（repo/commit/path/blob SHA1/SHA256/size）；测试固定
  14 文件集合防漂移（`test_csa2_core_pin_files_and_hashes`）。
- 复用 R20 csa2-pin 已有文件（attention.py/sparse_swa.py/cache_utils.py/
  sparse_mla.py 等）不重复入册。

证据 `ev-vllm-v41-csa2-core`（kind=source-checked，executed=False）；
ModelIR checked_paths 新增 `sm-v41-csa2-attn-core`；旧缺口条目
`sm-v41-csa2-moe-unverified` 移除（语义并入新条目）。

## 2. 逐项审计结论（4d+5 移植公式）

| 项 | v0.1 移植口径 | pinned 证据 | 结论 |
|---|---|---|---|
| per-pair FLOPs | pairs·H·(4d+5) | `flash_mla_interface.py` L118-131："V4/V4.1 head_dim should be 512 while head_dim_v should be 512"；`splitkv_mla.cuh` 主循环：QK WGMMA（sQ[BLOCK_M,512]×sK）→ scale_softmax → PV WGMMA（×sV[·,512]） | **结构确认**：每对每头 QK 2·512 + PV 2·512；"+5" softmax 为解析惯例（在线 max/exp2/sum，scale_softmax L32-84），保留并登记 |
| 头数 H | heads//tp | `flashmla.py` L92-100 `get_padded_num_q_heads`（fp8 decode kernel 仅 h_q∈{64,128}）+ L1216-1247 fused insert 把 Q 零填充到 padded_heads=64；`sparse_decode.cpp` L517-523 h_q 校验；L235-236 "q arrives pre-padded" | **修正**：decode 按 padded_heads=64 计 FLOPs/激活（TP8 旧口径**少算 8×**）；prefill SM90 small-heads（`flash_mla.py` L79-87 + `sparse_prefill.cpp` L163-166）用真实 n_local∈{8,16,32} 切片（旧口径恰好正确） |
| 因果边界 | causal_pairs（Σ min(t,swa) + Σ min(⌊t/r⌋,topk)） | `combine_topk_swa_indices` L970-990（swa_start=max(pos−127,0)、topk_len=min((pos+1)//r,TOP_K)）；`compute_global_topk`（有效项计数）；`_compute_swa_indices_and_lens_kernel` L1106-1112 | **确认**：三处独立口径与 causal_pairs 闭式一致；topk 与窗**无去重拼接**（L992-1017 直接连接），无去重建模正确 |
| 压缩记录读取（decode） | 每对×584B | `_forward_decode` L263-279：indices=swa 窗 + extra_k_cache/extra_indices=topk 一次核联读分页记录 | **确认**：kv_read=逐行窗+topk 记录（Q=1 与旧口径同值；Q>1 旧式 min(end,·) 少计，公式改逐行求和） |
| 压缩记录读取（prefill） | 每对×584B（每消费者各读一次） | `_forward_prefill` L326-358：每请求每 chunk `dequantize_and_gather` **全量压缩前缀**（seq_lens//r 行，gather_lens=None → [0,seq)）+ SWA 尾窗 gather（Q+min(S,win−1)）→ bf16 工作区 1024B/行；L384-392 核按合并索引逐对行读工作区 | **修正**：记录读 = ⌊seq/r⌋·584 + gather_len·584（每请求一次）；新增 attn_kv_gather 行（dequant 2 op/elem 惯例）+ 工作区写；sparse_attention 改读工作区 pairs×1024B（逻辑口径，不记跨行 SMEM 复用） |
| split-KV | 无 | `sparse_decode.cpp` L488 `must_split_kv=is_sm90a()`（SM90 恒分片）；L632-687 o_accum/lse_accum 分配 (b+SM)×s_q×h_q×512×4B + combine 启动；`combine.cu` L64-69（splits==1 早退）/L37-185 归并+sink；`splitkv_mla.cuh` L333-380（is_no_split 直写 out） | **新增** attn_splitkv 行 + 瞬态 o_accum 下界；num_splits 由调度器移植**精确**复算 |
| 调度器 | 无 | `get_decoding_sched_meta.cu` L42-121：块数 = (ceil_to_multiple(max(topk,1),64)+extra −1)//64+1；total=Σ(blocks+5)；payload=⌈total/SM⌉+5；单 warp 贪心分配 | **移植** count/flashmla_sched（首轮移植曾把 ceil 到倍数误写成除成块数——独立 oracle 测试抓住后修正）；200 例随机对照独立重演 |
| 索引消费 | R20：decode 每 (kv源:索引源:r) 运行首层 combine 一次 | Triton `combine_topk_swa_indices` 仅在 `_forward_prefill` L370-383 每 chunk 每层调用；decode 路径 = 元数据构建每步一次 `_compute_swa_indices_and_lens`（sparse_swa.py L663-732，同 spec 层共享 builder，kv_cache_interface 分组）+ **每压缩层每步** `compute_global_topk_indices_and_lens`（flashmla.py L214-228）+ 每层型首层调度器元数据 | **修正归因**：sparse_index_build 改 prefill 每 chunk 每层（fill −1 整宽 128 对齐 + 有效项 + lens）；新增 attn_swa_index_build（记 L0，每步一次）/attn_index_global（每 cr>0 层）/attn_sched_meta（r 去重首层 L0/L2/L20，DecodingSchedMeta=32B）。specialized decode 的 SparseIndexTables 等键复用属 opt-in 变体（见 §5） |
| attn_sink | 未计 | `flash_mla_interface.py` L96：attn_sink 为 (h_q,) fp32 标量，输出按 exp(lse)/(exp(lse)+exp(sink)) 缩放（combine.cu L131-142 折入） | 每头 4B 读 + ~3 op/头（并入行内注记，不改 per-pair 公式——非额外 KV 列） |
| q/out 激活字节 | T·heads·d·b_a | 接口断言 q/out bf16、lse/max_logits fp32（KU_CHECK_DTYPE）；q_padded 写宽 padded_heads | **修正**：q_rope 写 T×64×d×2（padded，bf16）；decode 核读 64 头/prefill 读切片；out+max_logits/lse 固定 bf16+fp32（V41_QUANT act 恒 bf16，数值上与 b_a=2 一致） |

## 3. 落地（账本/计价/内存/设备）

- `count/flashmla_sched.py`（新）：调度器逐行移植（纯整数、无依赖）；
  `decode_num_blocks` 的 ceil-到-倍数语义对 kernel L43-57 逐行对应。
- `count/work.v41_layer_rows`：`sparse_attention` 重写（decode/prefill 两
  分支：padded 头、gather 口径、工作区逐对读、sink/lse）；新增
  `attn_kv_gather`（prefill）、`attn_splitkv`、`attn_index_global`、
  `attn_swa_index_build`（记 L0）、`attn_sched_meta`（记各型首层）；
  `sparse_index_build` 归位 prefill；`q_rope` 写宽 padded；
  `build_model_work(..., device_sms)`（V4.1 decode 缺 SM 数显式拒绝）。
- `devices.sm_count`（H100/H200=132、L40=142，spec 来源；G1 字段表扩展）。
- `lowering/memory.v41_attn_transient_allocations`（新）：decode o_accum
  下界（allocator 复用一份）、prefill 工作区（最宽层型 r=1）与合并索引
  下界；接入 target/colocated 两分支。
- `formula_registry`：`v2.csa2_attn_core/attn_gather/attn_splitkv/
  attn_index_global/attn_swa_index_build/attn_sched_meta` 六条；
  `v2.csa2_sparse_index_build` 改 prefill 语义；t0 计价路由补 5 行。

## 4. 独立 oracle 与负例（tests/test_v41_csa2_core.py，27 项）

- pin 三重校验（vllm blob==tree、FlashMLA blob==git 对象、SHA256/size）
  + cmake GIT_TAG 与 manifest 一致性。
- 调度器：手算锚点 8 组（640 长行 132SM→10 splits、B=88 整吞→1、
  小 SM 逐 partition 2 块等）+ **独立重演**（逐 partition 字面消耗的
  另一种实现）200 例随机对照 + `sm_count` 缺失/0 负例 + 块数公式边界
  （0/64/65/128/extra）。
- decode 手算绝对值（TP8/B1/ctx32768 真实常数）：L2 核 640×64×2053=
  84,090,880（并断言旧口径恰为其 1/8）、kv_read (128+512)×584、
  splitkv 10 splits×64×2052 字节、index_global/sched_meta/swa_build
  逐字段；L0（r=0）2 splits 手算；TP 无关性与 B=8 调度器一致性。
- prefill 手算绝对值（ctx=4096）：gather (2048+4096)×584、工作区写、
  pairs 独立求和（Σ min(t,128)+Σ min(⌊t/2⌋,512)）、8 头（TP8）/64 头
  （TP1）切片、combine 每层 128/640 对齐宽。
- 守恒：行数 +73 精确分解、q_rope 写宽 ≥ 消费读宽（decode 等、prefill
  切片）、瞬态分配公式与 SM 缺失降级（o_accum 不入账、工作区仍在）。
- replay 冻结：Σ逐 op Δ=总 Δ、面外行逐字段冻结、before=main ee70ce2
  （1087 行/7,176,937,062 flops）、Δflops=2,850,849,054。
- 门禁事实：证据四栏（cpu_forward=not-run/gpu=none）、structure_complete
  仍 False 且剩余 unverified 全为执行变体（无"仍为 v0.1 移植"字样）。

## 5. 默认场景 replay 归因（docs/evidence/csa2-core）

场景与 R24 相同（TP8/B1/ctx32768/quant=V41_QUANT/H100/584B 记录）。
before.json 于**干净 main ee70ce2**（主工作树）捕获，SHA256 见
baseline-provenance.json；`tools/replay_v41_csa2_core.py`（`--check`
为当前活性无漂移入口；R22/R21/R24 工具的 check 链式改指本证据）。

逐 op 归因（attribution.json，Σ逐 op Δ = 总 Δ 逐字段断言）：

| op | Δflops | Δact_read | Δact_write | Δrows |
|---|---:|---:|---:|---:|
| sparse_attention | +2,825,453,568 | +2,304,000 | +2,293,760 | 0 |
| attn_splitkv | +25,313,280 | +50,429,952 | +50,440,192 | +40 |
| attn_index_global | +97,280 | +155,648 | +77,976 | +38 |
| attn_sched_meta | +30 | +12 | +12,696 | +3 |
| attn_swa_index_build | +640 | +512 | +516 | +1 |
| q_rope | 0 | 0 | +2,293,760 | 0 |
| sparse_index_build（删 decode 旧行） | −15,744 | −16,456 | −21,028 | −9 |
| **合计** | **+2,850,849,054** | **+52,873,668** | **+55,097,872** | **+73** |

- sparse_attention Δflops 复核：38 压缩层 ×(64−8) 头×640 对×2053 =
  2,796,021,760 + 2 个 swa-only 层 ×56×128×2053 = 29,431,808 →
  2,825,453,568（逐层 pairs 不同，闭式核对）。
- attn_splitkv：38 层 10 splits + 2 层 2 splits（调度器手算锚点）；
  act_write 含每行一次 lse（40×256）。
- kv_read Δ=0（decode 记录读口径不变，Q=1 逐行 = 旧 min(end,·)）；
  M01 +0.0328065306 ms/步（逐行 roofline 重价，Σ逐行 t_ms Δ 严格相等；
  解析档非实测）；M06 known_peak **+168,592,640 B**（o_accum+lse_accum
  17,466,624 + prefill 工作区 67,108,864 + 合并索引 84,017,152，下界）。

校准不变性（G7）不受影响（改的是原始计数而非校准）；v0.1 legacy
（engine.py/ledger.py）不 import 新路径。

## 6. 与任务书的出入及限制（如实；不冒充）

1. **"4d+5"的 4d 部分结构确认、+5 保留为惯例**：QK/PV 各 2·512 由接口
   断言 + 主核 WGMMA 形状钉死；softmax 5 op/整数核 ≈5/3 op/项/调度器
   ≈10 op/行/dequant 2 op/elem 为文档化解析口径（非 kernel 指令数），
   与 R20 选择 op 惯例同法登记。
2. **调度器移植的 ceil 语义曾错**：初版把 `ku::ceil(cur,64)` 写成除成
   块数（128→2 而非 128），独立 oracle 锚点（640 长行应为 10 块）抓住，
   已修正并加随机对照。教训入 G-C17（编号在 R26 合入后自 G-C16 让位）。
3. **prefill 工作区逐对行读为逻辑口径**：不记跨行 SMEM/L2 复用（与全仓
   unfused 逻辑访问口径一致）；物理 HBM 流量未实测。
4. **执行变体未建模（登记于 catalog csa2_attn_execution_variants 与
   source_map uncovered_domain）**：specialized decode（FlashInfer
   TRTLLM-gen `trtllm_batch_decode_sparse_mla_dsv4` 与 sparse_mla_cute
   CuTe；需 `decode_backend=latency_specialized` 显式开启，默认 auto →
   stock FlashMLA——specialized_decode/__init__.py L135-160 pin）；SM100
   mega attention（nvfp4 288B 压缩记录 + fused norm+RoPE+attn+逆RoPE+
   FP8 cast，`supports_compute_capability` major==10）；FlashInfer
   plain-row BF16 KV 后端；ROCm aiter/XPU；PCP 跨片；SWA bounded
   replay 回放边界；CuTe DSL dequant gather（字节语义同 Triton）。
5. **MTP 多查询行（Q>1）**：decode 把同请求多查询当独立 batch 行
   （flashmla.py L233-235 注释），逐行窗/topk 公式与调度器 b=N 口径已
   覆盖；但 V4.1 场景域只有 baseline（无 MTP 草稿），未另建 replay。
6. **R20 归因修正的披露**：R20 §3 的 decode sparse_index_build 行基于
   specialized sparse_mla.py 的 SparseIndexTables 语义，误用于默认
   FlashMLA 路径；本轮按 pinned 消费链归位（prefill 每 chunk 每层），
   decode 新增三个真实消费行。R20 证据 JSON 为历史切片不改。
7. **不运行 kernel/GPU**：FlashMLA CUDA 源经 cmake pin 静态阅读；
   本仓数值 oracle 为纯 CPU 整数/解析复算，非执行证据（G10 分栏维持
   source-checked/executed=False）。

## 7. 验收

- `py -3.12 -m pytest tests -q`：**452 passed，exit 0**（425 + R25 27）。
- replay 链：`tools/replay_v41_csa2_core.py --check`、`replay_v41_swiglu
  --check`、`replay_v41_quant --check`、`replay_v41_moe --check` 均
  exit 0（后三者 ② 层改指 csa2-core 证据）。
- `py -3.12 -m simverse.cli run-v2` status=completed（必需场景）；
  `node --check simverse/static/workbench.js` exit 0（未改动）。
- G1–G17 全保留（G1 字段表加 sm_count、G16 换核本体条目——底层事实
  变化而非放宽；G6 测试数 452 同步）。

## 8. 改动文件清单

- pin/证据：`references/vllm/csa2-core-pin.json` + vllm 5 文件 +
  `references/flashmla/` 9 文件；`docs/evidence/csa2-core/`
  {scenario,before,baseline-provenance,after,attribution}.json
- `simverse/count/flashmla_sched.py`（新）、`count/work.py`
  （v41_layer_rows 重写尾部 + build_model_work device_sms/首层旗标）
- `simverse/devices.py` + `contracts/legacy.py`（sm_count）
- `simverse/lowering/memory.py`（v41_attn_transient_allocations +
  colocated 接线）、`simverse/pipeline.py`（device_sms/瞬态透传）
- `simverse/formula_registry.py`（6 新 + 1 改）、`cost/t0.py`（路由）
- `simverse/model_ir/catalog.py`（checked/unverified/evidence_refs/
  note）、`families.py`（docstring）、`importers/source_map.py`
  （sm-v41-csa2-attn-core；indexer/moe-unverified 条目更新）
- `simverse/validation/registry.py`（ev-vllm-v41-csa2-core）
- `tools/replay_v41_csa2_core.py`（新）；quant/moe/swiglu 三工具
  check 链式改指
- `tests/test_v41_csa2_core.py`（新 27 项）；test_v41_csa2（2 项改
  写）、test_model_ir_v02/test_v41_quant/test_v41_swiglu/test_v41_hc/
  test_v41_engram（口径随事实更新）、test_merge_gate（G1/G16）
- docs：本报告、`gaps/capability-gaps.md`（G-C17；编号在 R26 合入后自 G-C16 让位）、
  `IMPLEMENTATION_STATUS.md`、`FOCUS_STATUS.md`

## 9. PR #18 合入后验收修正（2026-11-16）

在合并后的 `main` 上发现 `attn_swa_index_build` 的首层标志也在 prefill
生效；但 [vLLM @f358f057](https://github.com/vllm-project/vllm/tree/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5)
的 pinned `sparse_swa.py` 仅在 decode metadata 构建时调用
`_compute_swa_indices_and_lens`，prefill 已由 `combine_topk_swa_indices` 行计数。
已将该行限制为 decode，并在既有 prefill 测试中断言其缺席。
以 TP8/B1/ctx1024 完整 prefill 为例，修正移除 1 行、614,720 解析 op、
491,776 B 激活读和 495,872 B 激活写；decode 账本及
`docs/evidence/csa2-core/after.json` 的固定场景重放不变。
合并后全量门禁为 469 项；本修正的最终验收记录以修复 PR 的检查结果为准。
