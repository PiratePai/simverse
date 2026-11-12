# Simverse 重构实施状态（M0–M5 验收清单）

更新：2026-11-12（R06：Qwen3.6 GDN+MoE 小尺寸 CPU forward 对照 + attn_output_gate 漏计修正，rebase 到同日先合入的 R05 mHC/Sinkhorn 修正之上；早前 R04/两模型门禁/A 组/校准段落保留为历史记录）。主控会话持续维护；上下文压缩后从本文件恢复。
代码身份：本仓 git（PR #5/#7 已合入 main；本轮验收修复分支 `codex/pr7-acceptance-pin-hash`；远端公开仓库 `PiratePai/simverse`）；包版本 0.2.1（对齐结果协议 v0.2；ModelIR 独立 0.3.0）。
运行：`py -3.12 -m pytest tests -q`（2026-11-12 R06（基线含 R05）全量 **262 passed**，exit 0；历史：R05 252、R04 236、PR #3 合入时 234，k_norm 修正后 212，A 组二轮 201，hermetic 修复后 196/197）。`node --check simverse/static/workbench.js` exit 0。
**pytest 收集数：262**（`py -3.12 -m pytest --collect-only -q tests`，2026-11-12 R06（基线含 R05）；236→252 为 R05 mHC pin/独立 oracle/守恒与冻结回归新增 16 项，252→262 为 R06 新增 10 项 hermetic+live 对账）。
工作台 `py -3.12 -m simverse.cli serve` → http://127.0.0.1:8765/（生产 v0.2 首页 /；/legacy 为 v0.1 历史存档）。
合入准则：[MERGE_CRITERIA.md](MERGE_CRITERIA.md)（G1–G17 门禁在 tests/test_merge_gate.py，测试数与本文件一致性由 G6 强制）。

**2026-11-12 PR #5/#7 合入后验收**：首次全量 261 passed / 1 failed；
R06 fixture 的 pinned SHA256 记录为 CRLF 字节，仓内官方 pin manifest 为 LF 字节，
导致实时重放仅在相同换行检出环境可通过。已把 fixture/报告改为 LF 哈希，
`.gitattributes` 固定两份 pin 源码为 LF，并在 hermetic 测试中核对
fixture、manifest 与文件字节三者一致；同时修正 fixture `SourceIdentity`
的 `config_sha256` 为 fixture config 内容哈希。修后全量 262 passed，
`run-v2` 与 R05 mHC replay 运行成功；该修复不改生产计数。进一步计算审计
发现 R06 未提供独立输出数值 oracle，证据等级由 numeric-checked 降为
shape-checked，待数值逐元素对照后再升级。

**2026-11-12 R05 mHC/Sinkhorn**：14 文件完整 blob SHA1 对 pinned tree、SHA256
清单入册，新增 sm-v41-mhc-kernels/ev-vllm-v41-mhc（source-checked，executed=False）。
补齐 prenorm/sqrsum、comb post；Sinkhorn 仅 m×m，m4/iters20=1400 op/token；
首层无 post、expand为拷贝零 FLOPs；末层出口 post+collapse两行（layer=-1），
Triton零累加collapse=2mh，数学最简(2m−1)h另列；norm仍保留4h+2近似惯例。
L0新增491,520B broadcast派生缓冲，非参数；原 Engram 行/函数/公式未动。
默认TP8/B1/ctx32768 replay：op 7,146,532,702→7,161,190,222，权重读
−1,465,920B、激活读+1,666,880B，GPU解析下界116.152651919→116.152715012ms；
known_peak +491,520B，预算/峰值完整性unknown不变。新增16项独立 oracle/回归，
G7/G11语义未改。未执行kernel/CPU forward/GPU，部署变体、实际split与物理HBM
仍未建模。[报告与任务书差异](validation/v41-mhc-kernels-20261112.md)。

**A 组二轮审计补漏（2026-11-10，R01/R02a/R02b）**：对 pinned vLLM @dc74b8cb
逐文件复核发现并修复 4 处漏计（证据 `ev-vllm-logits-ctxkv-billing`，
`logits_processor.py` 补入 pin，SHA256 e1fac869…）：
① lm_head 词表行切后 logits 集合通信（logits_processor.py L185-198
`lm_head.tp_size>1 → gather`）此前完全未计——decode/verify/草稿候选各补 1 次
`logits_gather`（每 rank (tp−1)/tp×T×V×act 字节，单向环 (tp−1) 段延迟）；
② DFlash 草稿 `draft_lm_head` weight_read 原 0（dflash model.py L260/L599-601
每轮独立 GEMM）→ 补 h×(V/tp)×b_w；③ `ctx_kv_proj` 二次读（L384-387 k_ctx/
k_noise 两次 GEMM）→ 补按路径计的权重读；④ vLLM `_fused_kv_weight`
全层 K/V 行 cat 驻留副本（qwen3_dflash.py L541-568）入 R02a 账本
（仅显式 `fused_raw` 路径）。结构化临时补 ctx 归一副本/ctx 投影输出
（phase:draft）与 verify aux hidden（phase:verify）；M06 formula_ref 修正为
`v2.capacity`。默认场景前后归因见
[A 组二轮审计报告](validation/agroup-lmhead-ctxkv-audit-20261110.md)（M01
baseline 112.2127→112.9161ms 等 8 项逐项归因；同卡 memory status 仍 unknown，
两缺口不变）。

**A 组三轮 ctx-KV 路径修正（2026-11-11，R01/R02a）**：外层验收前修正两处
确认缺陷。① `draft_extra_allocations` 原以 `dense_w≥8` 推断
`UnquantizedLinearMethod`/`_fused_kv_weight` 副本——错误；pinned
qwen3_dflash.py L108-113/529-568/679-697 的分支由 quant_method/平台决定，
默认 dense_w=8 只是 fp8 计价档。现新增显式 `draft_ctx_kv_path`
（`fused_raw` / `mxfp8_context_proj` / `per_layer_fallback`）：raw 才建
融合 cat 副本，MXFP8 建独立 `_context_kv_proj`（packed 8bit 下界，
block-scale 未入 pin→unknown），回退无副本；`_k_norm_weights` 的 L×d
堆叠在三条路径都单列（4bit 回退此前漏计）。② `ctx_kv_proj` 对回退路径
原只计 K/V；pinned L688-697 回退执行完整 qkv_proj 再切 Q。TP8/B1/E[N]=3/
h2048/Q4 头 per rank/KV1/d128 时每层逻辑 FLOPs 由 3,145,728 修正为
**9,437,184**，权重读含 Q 行；回退临时缓冲补逐层完整 qkv（含 Q）暂存。
`draft_ctx_kv_path` 为场景显式输入，缺失/非法/`4bit+fused_raw` 均拒绝，
不由位宽猜分支。默认场景（活跃草稿 config 无 `quantization_config` +
GPU）显式取 `fused_raw`；这是**研究假设/场景配方**（config source-checked、
未执行），不证明实际 vLLM 以 8bit unquantized 部署，`dense_w=8` 只是
计价位宽。对象拆为 `draft-w-dflash-fused-ctx-kv` 与
`draft-w-dflash-k-norm-stack`。
**复核修正**：k_norm stack 改用 `DraftIR.dtype_hint`（config
`dtype=bfloat16`→2B/elem，6×128×2=1,536 B），与 `dense_w` 4/8/16 无关；
dtype 缺失/不支持显式拒绝，不回退位宽。默认 steady/known 因此各 +768 B。
独立手算/负例见 `tests/test_draft_ctx_kv_paths.py`（11 项），报告
§6 见 [A 组二轮审计报告](validation/agroup-lmhead-ctxkv-audit-20261110.md)。
外层 Codex 已全量验收：212 passed，exit 0；远端 CI 以候选 PR 检查为准。

**2026-11-11 R04：V4.1 Engram 建模（B 组 R04 交付）**：上游源码由主控
pin 入仓（vLLM @f358f057；common/engram.py blob sha1 ba4ceea6 + nvidia
ops engram_* 三文件 + amd/engram.py + configs/deepseek_v41.py，blob sha1
逐项对 references/vllm-tree.json 校验）。落地：engram 8 键 → mapped
（extras.engram spec；264 B/行 = FP8 e4m3 256 + ue8m0 8，n_hash_cols=24，
桶分配 L211-252 忠实移植 families.engram_bucket_primes）；count 四行
（engram_hash/lookup/wkv/post_wkv，层 1/14，批级哈希一次挂首层行）；
memory 三对象/层 + staged 下界（表默认 **cpu_offload 研究假设**，场景
显式 engram_table_placement 输入，host 驻留不进 GPU 峰值单列
host_resident_bytes；显式 gpu → TP8 合计超标称如实 OOM）；TP>1 补
head/wkv 两次 all-gather 通信行。独立 oracle：tests/test_v41_engram.py
24 项（自写 Miller-Rabin 层桶和 == engram_num_embeddings 精确相等
384,006,168/384,016,682、首 8 素数硬编码、TP1 表 202,758,032,400 B、
tp=7/9/16 拒绝、逐行手算绝对值、开/关账本差=精确新增、count↔memory
对账、blob sha1 防漂移）。门禁随事实更新（G9/G10/test_model_ir_v02
断言归因见报告 §8，剩余缺口不变仍在阻断）。默认 V4.1 replay 前后
逐项归因：decode FLOPs 7,067,478,214→7,146,532,702、GPU 侧解析耗时下界
113.34→116.15ms（host 查表/预取时延未知）、GPU steady +39,485,440B（仅 wkv+qk）、host 表
25,344,443,520B、行 909→916；structure_complete 仍 False（CSA2/MoE/
量化缺口不变）。报告：[docs/validation/v41-engram-20261111.md]
(validation/v41-engram-20261111.md)；证据 ev-vllm-v41-engram；G-C1
单机算子解除（DP 交换/cpu_offload 运行时/amd 变体仍登记未覆盖）。

**2026-11-12 R06：Qwen3.6 GDN+MoE 小尺寸 CPU forward 对照（D 组 R06）**：
探针 `tools/probe_qwen36_forward_r06.py`（独立于本仓，不 import simverse）
以 pin 官方 forward（transformers @b6401d66，torch 2.15.0+cpu，
seed 20261112）执行 h=32/2 层（GDN+FullAttn）/4E top-2 随机权重 fixture，
prefill T=5 + decode T=1，落盘 `tests/fixtures/qwen36_r06_cpu_fixture.json`
（逐 GEMM/conv/递推形状/因果对/MoE assignment 全录）。对账
`tests/test_qwen36_forward_r06.py` 10 项（hermetic 9 + live 1）：本仓
`parse_qwen3_5_moe(同一 config)` → `build_model_work(tp=1)` 逐算子**精确
相等**（对账表见报告 §4）。**修正一处漏计**：full-attn 输出门控逐元素行
（官方 L802 sigmoid(gate)·attn_out）补 `attn_output_gate` 行 + 公式
`v2.attn_output_gate` + source_map note——默认场景前后：decode +10,240
FLOPs（10 层×1024）、verify +163,840、draft 不变、M01 +0.00002ms（报告
§5）。披露不修：gdn_recurrence 4 ops/elem（官方参考≈7）；HF DynamicCache
conv state 宽 = kernel 列（本仓按 vLLM kernel−1 计容量，差 1 列）。
证据 `ev-qwen36-cpu-fixture-r06`（shape-checked）入 registry 并挂
qwen36 evidence_refs；**模型级 cpu_forward 仍 not-run**，部署排名门禁
不变。报告：[docs/validation/qwen36-gdn-moe-cpu-fixture-r06.md]
(validation/qwen36-gdn-moe-cpu-fixture-r06.md)。

**CI hermetic 修复（2026-11-10）**：`tests/test_consistency.py` 两项验收不再读取
被 `.gitignore` 忽略的 `outputs/run_v41.json`。CSV 导出改为复核本次 `run` 新落盘的
RunResult；固定场景重放改用纳入版本库、带 provenance 的历史快照
`tests/fixtures/v41_flash_dt_verification_expected.json`。远端 run `42803015653`
的 `2 failed, 194 passed` 是 clean checkout 缺文件；此前本地 `196 passed` 属环境
假阳性。详见 [Windows CI hermetic 修复](validation/windows-ci-hermetic-fix-20261110.md)。
该轮测试总数仍为 196（后续编码修复新增 1 项后为 197）。

**CLI 导出编码修复（2026-11-10 后续）**：远端 run `42805472945`（PR）与
`42805463611`（push）的唯一失败是 `test_csv_export_matches_runresult`：
Windows 管道 stdio 使用 ANSI 代码页 cp1252，`simverse.cli export` 在写文件
成功后 `print("已写 ...")` 抛 `UnicodeEncodeError`，进程 exit 1。CLI 现在在
`main()` 统一把 stdout/stderr 设为 UTF-8 + `backslashreplace`；测试显式固定
子进程 stdio 编码，并新增 cp1252 回归用例。收集数 196→197，详见
[Windows CI hermetic 修复 §6](validation/windows-ci-hermetic-fix-20261110.md)。

**外层本地验收（2026-11-10，hermetic 修复后）**：外层 Codex 唯一一次全量
`py -3.12 -m pytest tests -q` 实际 exit 0：`196 passed, 1 warning in 5.04s`；
警告为既有的 Windows 子进程 UTF-8 解码线程（`test_csv_export_matches_runresult`）。
该结果只证明本地工作区（含尚未提交的修复）通过，不是 clean checkout 或远端 CI
通过证据。修复文件尚未提交/推送；PR #1 现有检查仍是修复前失败记录，远端重跑
由外层执行。

## 0. 当前有效状态（2026-11-10 两模型门禁轮）

- **活跃目录恰两款**：`deepseek_v41_flash`、`qwen36_35b_a3b`，加绑定 Qwen3.6 的
  `dflash_qwen36_35b_a3b` DraftIR。`qwen3_30b_a3b`、`qwen3_235b_a22b`、
  `qwen3_next_80b_a3b`、`deepseek_v3` 移出运行时注册；tiny 只作内部测试 fixture。
  历史模型/场景/profile 保留为离线归档，API/CLI/工作台不可选择。
- **证据分栏**：`ModelIR.evidence` 记录 forward 源码、CPU 执行、GPU 实测三栏；
  `config_coverage_complete`、`forward_evidence_complete`、`structure_complete`
  分开。完整 config（含嵌套叶键）逐项登记，未知键不得默认 ignored。
  - Qwen3.6：config 覆盖完整；GDN/full-attn gate/MoE router+shared gate/TP-EP
    分片 source-checked；R06（2026-11-12）fixture 级小尺寸 CPU forward
    形状与算量对照已执行（ev-qwen36-cpu-fixture-r06，GDN+MoE+full-attn 载体逐
    算子精确对账）；MTP 只有 config+目标首层模板 → **unverified**；
    forward 仍不完整、模型级 cpu_forward 仍 not-run，不能参与部署排名。
  - V4.1：Engram 单副本路径已 source-checked 建模（R04，2026-11-11；
    DP 分片交换/cpu_offload 运行时行为/amd 变体未覆盖）；CSA2/MoE、
    candidate、swiglu、quantization_config 仍为缺口；mHC 仅
    source-checked；CPU/GPU 均未执行。
  - DFlash：结构/轮协议/同卡共享 source-checked；服务端集成未核、拒绝采样/KV
    回退/hidden 提取仍 0 占位；CPU/GPU 未执行。
- **本轮修复**（2026-11-12 R06）：full-attn 输出门控逐元素行
  `attn_output_gate`（官方 L802 sigmoid(gate)·attn_out；此前只计 q_proj
  2× GEMM 消费）——默认场景 decode +10,240 FLOPs、verify +163,840、
  draft 不变（归因见 R06 报告 §5）。
  早前（2026-11-10 两模型轮）：V4.1 `o_a_proj` 分组在 work/memory 同步乘
  `ng`，并补 V4.1 attention memory 的 `b_dense` 单位（BF16 曾少 2×）；
  Qwen3.6 shared gate 在 IR/work/memory 计入；DFlash/MTP/chain 草稿 B>1
  token 维乘一次；证据引用按 model/draft/profile 生成；source_map
  行号/未核状态校正。
- **当前数值**：CLI 内置默认 Qwen3.6+DFlash（显式 `draft_ctx_kv_path=fused_raw`，α0.72、blk16）稳态
  2,830,652,352 B（shared gate +81,920 B；含 fused ctx-KV 3,145,728 B 与
  k_norm 堆叠 1,536 B），known_peak 2,832,959,265 B，仍不完整；
  审计对照场景（α0.75、blk5）的 known_peak 为 2,831,405,056 B；
  V4.1 o_a TP1 每层 67,108,864 FLOPs/容量 B，40 层约 +2.35 GFLOP/token
  +2.52 GB(BF16)。详细归因/来源见
  [两模型审查记录](validation/two-model-forward-merge-gate-20261110.md)；
  ctx-KV 路径修正见 [A 组二轮审计 §6](validation/agroup-lmhead-ctxkv-audit-20261110.md)。
- **合入门禁**：G1–G17；`.github/workflows/merge-gate.yml` 覆盖
  PR/push/merge_group，Windows + Node 20 + Python 3.12。2026-11-11 复核：
  main branch protection 要求经 PR 合入及 `pytest (Windows, Python 3.12)`
  必需检查（strict），审批数为 0，且 `enforce_admins=true`。
- **未完成**：V4.1 CSA2/MoE scoring/量化映射源码与执行、Engram DP 分片
  交换路径、Qwen3.6 MTP 原生 forward、
  两款主模型 CPU/GPU 实测、DFlash 服务端集成与运行时池尺寸、真实可分配预算。

## 0.1 必读上下文

- 计划包：`docs/rebuild-20261106/`（EXECUTION_PROMPT.md 为本任务指令；
  questions.md Q1–Q12 已答；plan/principles/ui-data-contract 为验收依据）。
- v0.1 切片（旧 13 测试）保留于扁平模块（engine.py/ledger.py/...），
  contracts.py→contracts/legacy.py、cost.py→cost/legacy.py（同路径迁移，
  旧导入兼容）。v0.2 管线为 contracts/count/methods/lowering/cost/execute/
  experiments/pipeline/scenario 包。两套由 schema_version 0.1/0.2 区分，
  M5 退役 v0.1。

## 1. 里程碑状态

> 以下 M0–M5 段落保留历史日期与事实；其中“结构完备/90B/235B 活跃/180 passed”
> 等表述已被 §0 的两模型轮取代。当前合入判断以 §0、MERGE_CRITERIA 与门禁测试为准。



### M0 证据冻结与红灯 — 完成
- git init + 基线提交（5913cc5 起）；.gitignore 排除嵌套仓 references/model-verse-source。
- contracts v0.2：enums（support/evidence/feasibility/result_kind/statistic/coverage/time_basis 正交）、
  identity（完整 SHA256、js_safe 大整数往返、RngScheme）、Metric（null+reason 强制、
  exact 拒绝非整浮点）、RunManifest（run_id=身份哈希）、ResultBundle、CapabilityReport。
- evidence registry（simverse/validation/registry.py）：内容 hash 实算绑定；
  V4.1 config、Qwen3.6 主/草稿 config、dflash 实现/README、vLLM V4.1 源、旧原理 pin。
- 独立 golden（golden/oracles_v02.py，不 import 生产）：MoE 枚举 E4k2M2=3、
  E384k6M5≈29.077034、有放回反例、接受端点/mean 反例/动态 K 联合律反例（2.06/1.98）、
  人工轨迹（TPOT/吞吐/goodput/busy/分位）、roofline/链路手算锚点。
- 红灯测试 tests/test_negative_v02.py：16 项（期望 int 截断、NaN、截断 digest、
  大整数、mean→Pfull、E09 残差等各自有断言捕获）。
- 证据：tests 68 passed（含 v0.1 13 项原样保留）。

### M1 语义与计数 — 完成（Qwen3.8 本地身份待补）

**2026-11-10 增量**：V4.1 mHC 混合流建模（hc 三键 mapped；count hc_* 边界行
6/层+首层 expand、FP32 权重每卡复制入 memory；Engram 8 键保留缺口——
common/engram.py 未入评审包，不猜；tests/test_v41_hc.py 7 项含守恒）；
Qwen3-235B-A22B 官方身份入册（HF main @8efa6172，config SHA256 0ecd5d6f…）
+ catalog qwen3_235b_a22b（qwen3_moe 家族，94L 128E）。
- model_ir 0.3.0 共享子包（schema/families/catalog）：结构、SourceIdentity、
  字段覆盖（mapped/metadata-only/ignored-with-reason/unsupported-semantic）、
  structure_complete+gaps。
- 家族与锚点：gqa（tiny）、qwen3_5_moe（Qwen3.5/3.6 混合 GDN+GQA）、qwen3_next、
  qwen3_moe、deepseek_v3（MLA+Dense/MoE 分层）、v41_csa2（压缩层计划+Engram/hc 11 缺口）、
  DFlash DraftIR。
- 身份 pin（assets/config + assets/identity，SHA256 全录）：
  - Qwen/Qwen3.6-35B-A3B @345d467b（必需②主模型，qwen3_5_moe，40 层=30 GDN+10 Full@3,7..39，
    256E top-8+shared512+gate，MTP1，mrope 多模登记 metadata-only）
  - z-lab/Qwen3.6-35B-A3B-DFlash @f181eece（必需②草稿：6 层 5×SWA(4096)+1×Full GQA dense、
    block16、mask_token 248077、target_layer_ids [1,6,11,16,22,27,32,37]、fc 8·2048→2048）
  - z-lab/dflash @07ebd93（方法实现 pin：dflash_generate 协议/Qwen3DFlashAttention/
    拒绝采样；HF checkpoint 内无 dflash.py，auto_map 解析到服务栈）
  - Qwen3-30B-A3B @ad44e777、DeepSeek-V3 @e815299、Qwen3-Next（model-verse 本地快照，
    官方仓 gated→身份 PENDING）、transformers main @b6401d66 的 qwen3_5_moe modeling/configuration
- source map（importers/source_map.py）：7 条（GDN decode 分支 L558/607、q_proj 输出门控、
  shared gate、V4.1 短/长索引分支、Engram/hc 缺口、DFlash 轮协议、草稿 cross-attn），
  均带谓词/域/两侧覆盖声明。
- count 层（count/work.py + moe.py + draft.py）：GQA/SLIDING/GDN/MLA/CSA2 家族逐层行；
  E09 修复（残差 flops）、E05 修复（期望并集单列 float）、GDN 常数状态单列；
  MoE 正确式 E·(1−(1−k/E)^M)（有放回仅显式近似选项）。
- 测试 tests/test_model_ir_v02.py：19 项（层序回归 Qwen3-Next idx3,7..47=12 Full、
  V3 61=3Dense+58MoE、V4.1 r 分布 2/18/20+11 缺口、逐键覆盖、tiny 手算全量对账、
  MoE 期望/精确分列、MLA 压缩 KV、激活参数量级）。

### M2 串行实验闭环 — 完成
- methods：AcceptanceLaw（position_alphas/geometric/committed_mean[显式假设]/
  dynamic_k[缺律→unknown]/empirical_hist[unsupported]）；E07 修复（mean 不反解，
  假设显式才 approximated）；MethodPlan 注册（baseline/chain_draft/mtp/
  chain_moe_proxy[改名假想方法]/dflash_v1[真实语义绑定 pin]）；dspark/dflash2 显式 unsupported。
- lowering：TP 合法校验（不盲测非法组合）、KV 分片/复制登记、ring allreduce 每层 2 次
  （E03 修复：TP 通信入账）、EP dispatch/combine 期望（均匀假设登记）、
  容量账本（唯一存活对象求和、shared_identity 去重、标称/可分配分层→feasible=unknown §7.4）。
- cost：T0（行级 roofline + 精度路由；FP4 不自动升档 E10 口径保留）；
  profile 契约（ProfileRecord/CalibrationModel/CostEstimate；合成 fixture 接线，
  域外回退解析、synthetic 不升级，Q2 全 unmeasured）。
- execute：串行轮事件（event_id/depends_on/stage/role/start）+ SerialRound；
  TPOT=E[T_round]/E[N]。
- pipeline.run_v2：ScenarioV2→ResultBundle 全链；结构缺口/容量 unknown/接受缺律
  按指标传播（数值保留+不排名）；V4.1 Engram/hc→structure gate。
- experiments：不可变快照扫描（E20 修复：不触全局注册表，测试钉死）、
  每点独立 run_id、breakeven 无交点返回 null 不造数；
  ComparisonPolicy（Q4 双资源政策、Q11 多目标并列、资格谓词、资源 delta 披露）。
- 测试 tests/test_pipeline_v02.py：18 项。
- M2 初版历史数字（Qwen3.6+DFlash colocated 8×H100 B1 ctx32k α0.72 blk16）：
  基线 TPOT 3.4704ms、投机 1.6874ms、加速比 2.0567。后续 A 组修复了逐层 TP 通信，
  本轮再补 DFlash 草稿通信；**当前值见下方投机耗时复核**，此处不得作当前性能结论。

### UI（M1/M2 区域）— 完成（生产工作台）
- simverse/static/workbench.html+js：复用 v3 DOM/CSS 骨架（场景栏/7 KPI/矩阵/
  算子账本/七维检查器/Roofline/三轨时间轴/证据抽屉）；全部数字来自 /api/v2/run；
  无 v3-workbench.js mock 数学进生产路径（原型仍在 docs/.../prototype/ 独立存档）。
- webapp：/（生产）、/legacy（v0.1）、/api/v2/{catalog,run,sweep}；js_safe 大整数。
- 浏览器实测（IAB）：目录驱动选择、运行（run_id/代码版本显示）、KPI 7 项填充、
  572 行账本→24 聚合节点、检查器+后端 Roofline 坐标、5 事件三轨时间轴、
  mean-only→“未知”+反例原因、批量矩阵 9 格独立 run、窄屏 390px 无整页溢出。
  截图：outputs/workbench_desktop_run.png、workbench_mobile_run.png。
  注：本会话模型通道无图像输入，像素级观感未复核；以程序化布局检查+截图存证
  （与 v0.1 报告 §5.1 同口径如实声明）。
- CLI v0.2：run-v2/sweep-v2/export-v2（JSON/CSV/trace/HTML 四类导出）。
- V12：tests/test_ui_v12.py（fresh run→Web=CLI=JSON 逐字段、CSV/trace 逐行、
  HTML 身份、stale 不覆盖）。

### M3 资源事件核 — 核心完成（流水调度已入产品；请求级超前未做）
- [x] 指标归约器（execute/reduce.py）：请求口径 ITL（同刻=0 合法/N≤1=None）、
  窗口 token 事件 vs 完成请求两口径、goodput(SLO，取消单列)、busy 并集积分、
  最近秩分位；人工轨迹独立 oracle 全对账（tests/test_metrics_v08.py，V08）。
- [x] 事件验证器（execute/validate.py）：missing_dependency/
  starts_before_dependency_end/resource_overlap/consume_unknown_message/
  consume_before_send_end/missing_message_edge 六类检查，负例逐条捕获（V07）。
- [x] 资源约束事件核（execute/resource_core.py）：最早开始调度（互斥资源+
  半双工链路）、两微批 DT 流水与单请求串行、cohort 周期（中位差，避 warmup/
  drain）、资源 busy/利用率；事件流过 validate_trace（48 事件 0 违规实测）。
- [x] 产品接通：部署模式 dt_pipeline_sim（ScenarioV2.pipeline_microbatches/
  rounds）→ M18 周期/每轮等价吞吐/利用率指标 + M01 附"周期≠单请求延迟"注
  （分列不混用）；UI 部署下拉可选；浏览器实测通过（截图
  docs/evidence/workbench_pipeline_mode.png）。
- [x] 实测行为（ctx8k blk16）：target-gpu 利用率 99.2%、draft 10.2%（verify
  主导），cohort 周期 11.512ms、每轮等价 5.81ms < 串行轮和——收益来自具体
  调度而非 max 通则。
- [ ] 单请求超前（lookahead>0 的分支/取消/浪费资源账）与请求级事件轨迹
  （M17 TTFT 等）未做；取消/浪费字段在事件 schema 已留。

### M4 服务与校准 — 两条算子级实测校准闭环（服务特性仍未开放）
- 服务特性 UNSUPPORTED 注册：chunked_prefill/pd_disaggregation/prefix_cache
  （methods/plans.py UNSUPPORTED 表，Q5 顺序；不空实现）。
- **H200 算子级校准闭环 loop1（2026-11-09 建，2026-11-10 复审修复）**：pi-viewer
  @c5909a7 的 Qwen3-30B-A3B-tiny CUDA_EVENT 数据蒸馏入册
  （assets/profiles/h200_qwen3_30b_a3b_tiny/；tools/distill_h200_profiles.py
  可重放）。生产管线消费：ScenarioV2.cost_profile → CalibratedPricer（域门禁：
  设备/精度/TP/形状+算子构成三项核对；MoE 复制品 16E 域外回退；verify 注意力
  Q>1 域外回退）；指标 M01_tpot_profile_calibrated（result_kind=
  measured-composite，非整步实测）；UI 部署栏可选 profile。
  **2026-11-10 复审修复**：①gap 补贴改按**计时区间边界**计（原按全部 kernel 数
  计费重复计入已含在 CUDA_EVENT 括号内的 op 内间隙，B8 高估 0.324ms/步≈1.6%；
  修复后 B8 selected 19.879ms，9 点比值 1.95–11.59×——表述为相对组合估计的差异，
  非部署延迟结论）；②形状门禁补 QKNorm/全旋转/无滑窗构成核对；③时间口径覆盖
  （coverage_time_frac，依赖估计）随 stage 披露；④测量契约 caveats 入 manifest
  （launch-bound 括号内 host 空闲/warm-L2/重复数/gap 口径/even 假设）。
  报告：[docs/calibration/h200-loop1.md](calibration/h200-loop1.md)；
  重放不变性（G7）与组合算术独立复算入 tests/test_calibration_h200.py。
- **H200 loop2：Qwen3-235B-A22B TP∈{2,8}（2026-11-10 建）**：真模型官方身份
  （HF main @8efa6172）+ TP 集合域（首条 TP>1 实测覆盖；sharded 算子 per-worker
  形状=新仓 tp 切分行口径；TP1 采的 replicated 组以 tp_invariant 并入）；
  MoE 数据在册但 defer（per-rank 分片账本 G-C11 前置，不混尺）。场景
  assets/scenarios/h200_calib_loop2_235b.json（B8/TP8：selected 68.48ms，
  measured 29.13+fallback 38.30+gap 1.05×847 区间；行覆盖 1129/1318）。
  报告：[docs/calibration/h200-loop2-235b.md](calibration/h200-loop2-235b.md)；
  tests/test_calibration_h200_235b.py（8 项）。
- **旧仓数据全量审计（2026-11-10）**：[docs/calibration/oldrepo-data-audit-20261110.md]
  (calibration/oldrepo-data-audit-20261110.md)——A800 真 30B moe.csv 因缺环境
  元数据拒绝 measured 准入（§3.2/G4）；E2E 记录与 KERNEL_ONLY per-op 时间列
  确认不存在（G-D5/G-D6 维持）。
- UI 精度选择器（G-C10 解除，2026-11-10）：场景栏 BF16/W8A8/W8A8+FP4 档；
  UI 场景 JSON → /api/v2/run 与进程内/CLI 同 run 同值（测试钉死）；域门禁仍按
  profile 采集精度独立核对（不把"profile 要 BF16"转成输入精度）。
- 数值口径修正（2026-11-09 轮）：devices L40 注释修正；h200_sxm 官方规格 dense
  口径；链路 source 方向声明；cost/profile.py 校准分支修复（禁回填 estimate_ms）。
- Chunked Prefill / PD 分离 / Prefix Cache：未实现，能力目录未注册为支持；
  扫描/比较路径遇及时显式 unsupported（方法 UNSUPPORTED 表）。
- 服务到达/组批/KV 分页：workload.arrival="fixed-batch"（M4 扩展点已留）。

### M5 迁移与扩展 — model-verse 适配器已建；pi-viewer 数据通道已通（蒸馏）
- model-verse 薄适配器：LLM-Viewer-main/simverse_adapter/（新增文件，未改旧仓
  既有文件）：哈希锁（adapter_lock.json，6 文件 SHA256）+ 固定 commit 8796607 +
  model_ir 0.3.0；自检报告（adapter_report）通过；漂移即拒绝。
- pi-viewer：工作区阻塞已解除——本轮以只读蒸馏方式消费其 H200 数据
  （tools/distill_h200_profiles.py，来源提交+哈希入册，无运行时路径依赖）；
  独立适配器（同 model-verse 模式）仍待做，属 V09b 三方对账范围。
- 远端：Q3 平台/账号/地址待补；未创建/推送任何远端。
- v0.1 模块退役：待 M5 三方对账后。

### 2026-11-10 A 组轮：Qwen3.6+DFlash TP/EP 分片与同卡可部署性（R01–R03b）

报告入口：[docs/validation/agroup-tp-ep-colocated-20261110.md]
(validation/agroup-tp-ep-colocated-20261110.md)（前后重放归因/支持矩阵/限制）。

- **R01 统一配方**：新增 `simverse/lowering/parallel.py`（ParallelPlan；
  vLLM @dc74b8cb pin 入册 assets/identity/vllm_qwen35_pin/，16 文件逐
  SHA256，证据 ev-vllm-parallel-recipe）。count/moe、count/work（GQA/GDN/
  head per-rank）、count/draft（同 TP 组草稿+补 q/k norm/rope/hidden_norm/
  final norm 缺行）、lowering/memory、lowering/sharding（按层通信：EP MoE
  层 dispatch/combine 替代 mlp allreduce）、pipeline 六处消费同一 plan。
  TP 集合通信漏计修复（整模型 2 次 → 每层 2 次）。场景字段 target_ep
  （∈{1,tp}，其余拒绝）入运行身份。source_map 新增 3 条（sm-vllm-*）。
- **R02a/b 同卡账本**：物理 allocation 身份 + shared_identity + consumer
  引用；草稿共享 target 词表行切 embed/lm_head（零字节引用，修复前幻影
  副本≈1.017GB+去重失配）；GDN/草稿 KV/attention 权重 per-rank（补乘输入
  宽 h）；草稿批=目标批；生命周期分列（resident/per-request/cross-round/
  phase:draft|verify 串行取 max）+ unknown_components（workspace/graph/
  NCCL 池，不填 0）→ peak_complete 标志；per-rank r0..r{tp-1} + 最坏 rank
  报告（EP 余数 rank 差异显式）。
- **R03a/b 预算与判定**：DeploymentSpec.target_ep/budget(BudgetInput:
  allocatable_bytes/kind(measured|hypothesis)/source/extra_overhead，
  gross 语义) ；DeviceBudget 分层；RankMemory.peak_vs_budget 四态
  （unknown/oom/feasible/feasible-conditional；已知下界>标称即 OOM）；
  compare/_eligibility 排除 conditional；M06 携带 known_peak/steady/完整性。
- **UI**：工作台 EP 选择 + 预算输入（hypothesis）；浏览器实测（EP 生效
  T_round 64.6↔112.6ms 可区分；预算进链路以原因文本证实；截图
  docs/evidence/workbench_ep_budget_run.png）。
- **数值归因**（默认场景前后）：TPOT 1.6874→31.70 / 基线 3.4704→112.21
  （集合通信按层计费修复主导，pcie5 假设延迟 1.4ms×80 次；NVLink 规格
  属 G-D8 未虚构）；显存 5,680,423,680（口径混乱单值）→ per-rank steady
  2,827,423,168 + 临时峰值 2,135,746（明细见报告 §3）。235B TP8 校准
  fallback 38.30→4.85ms（MoE per-rank 口径，断言随归因更新；defer 理由
  改指 G-D2）。
- 测试 132→176（parallel_sharding 19 + colocated_memory 11 +
  budget_feasibility 14 + UI/场景接线 2；既有全保，tiny oracle 补 final
  norm 项、tp 校验断言随检查项更名——均逐项归因）。

### 2026-11-09 数值可信度轮
- [AGENTS.md](../AGENTS.md)（合入入口）+ [docs/MERGE_CRITERIA.md](MERGE_CRITERIA.md)
  （准则规范）+ G1–G17 自动门禁（tests/test_merge_gate.py）。
- devices 口径修正 + h200_sxm；profile 校准分支修复；pipeline 接线实测 profile；
  H200 第一条算子级闭环（见 M4 节）。
- 缺口清单分立：[docs/gaps/data-gaps.md](gaps/data-gaps.md)（数据采集 G-D1–G-D9）、
  [docs/gaps/capability-gaps.md](gaps/capability-gaps.md)（工程能力 G-C1–G-C9）。
- 证据：docs/evidence/h200_calib_compare.json（9 点前后对比）。
- 测试 86→110（merge gate 10 + calibration loop 12 + 既有全保）。

### 2026-11-10 校准复审 + 235B 闭环 + hc 建模轮（详单）
- **loop1 复审（8 项核查全部落档）**：gap 重复计费修复（区间边界口径）；
  形状门禁构成核对（QKNorm/全旋转/无滑窗）；留出无泄漏复核（387/5184 键唯一）；
  measured-composite 标注复核；"低估 2–11.7×"改口径为"相对组合估计 1.95–11.59×"；
  覆盖率双披露（行+时间占比，后者注明依赖估计）；测量契约 caveats 入册。
  pi-viewer 采集源码逐文件核对（device_timer/linear_op_impl/attention_wrapper/
  kernel_gap main/allowlists）。
- **loop2 235B**：官方身份获取入册；TP 集合域 {2,8}；tp_invariant 组；MoE defer
  裁决；组合/门禁/重放 8 测试；场景与报告入册。
- **V4.1 hc**：语义+计数+容量+守恒（详见 M1 增量）；Engram 8 键如实保留。
- **G-C10**：UI 精度选择器接线+测试。
- **数据审计**：oldrepo-data-audit-20261110.md（A800 拒绝/E2E 与 KERNEL_ONLY
  不存在/235B moe 首选路径）。
- 工具：tools/h200_calib_compare.py（对比 JSON 可重放生成器）。
- 测试 110→129（hc 7 + 235b 8 + 精度接线 2 + 门禁/组合更新；既有全保）。

## 2. 页面/指标/验证映射（当前覆盖）

| 验证 | 状态 | 证据 |
|---|---|---|
| V01 身份与协议 | 通过 | test_negative_v02（digest 全长/大整数/NaN/null+reason）|
| V02 结构与 forward | 部分 | 层序/覆盖/源码 pin+分支映射（source_map）；Qwen 固定源码两层随机权重 CPU shape probe 已做，R06 又做 GDN+MoE fixture 级形状与算量对照（ev-qwen36-cpu-fixture-r06，逐算子精确对账），独立输出数值 oracle/真实 checkpoint/完整 forward 仍缺 |
| V03 独立数学 oracle | 通过 | golden/oracles_v02 + test_model_ir/test_negative |
| V04 lowering 与守恒 | 大部分 | per-rank 统一配方（A组 R01）+ 跨 rank 守恒/复制对象不求和断言（test_parallel_sharding）；EP all2all/按层通信配方 |
| V05 方法、协议与状态 | 部分 | 接受律/端点/未知传播/DFlash 计划绑定；事件级 transition trace 未做（M3）|
| V06 显存生命周期 | 部分 | 静态+每请求对象、同卡去重；生命周期交错峰值未做（M3）|
| V07 事件因果与资源 | 大部分 | 验证器六类检查+负例；资源调度核+两微批流水已接；单请求超前未做 |
| V08 系统指标归约 | 通过（人工轨迹）| execute/reduce + test_metrics_v08 |
| V09 比较与旧仓对账 | 部分 | 比较资格/多目标/资源披露（V09a）；旧仓三方对账（V09b）未做（M5）|
| V10 扫描与运行隔离 | 部分 | 不可变快照/独立身份/无交点 null；取消/恢复未做（单任务门控待做）|
| V11 校准与外部真实性 | 部分（算子级，两闭环）| H200 30B-tiny TP1 + 235B TP2/8：蒸馏/域门禁/回退/留出误差/重放不变性（test_calibration_h200{,_235b}）；E2E 对照无数据未宣称 |
| V12 UI 与导出一致性 | 通过 | test_ui_v12 + 浏览器实测记录（本文件 §1 UI）|

## 3. 已知缺口与外部依赖（不虚构补齐；完整清单见 docs/gaps/）

1. **V4.1**：Engram 单副本路径已建模（R04，2026-11-11，pin+source-checked；
   剩余：DP 分片交换/EngramBatch MXFP8/cpu_offload 运行时行为/amd 变体，
   G-C1）；CSA2/MoE
   scoring/top-k/分组路由、candidate block、swiglu clamp、quantization_config
   仍为显式缺口；mHC 仅 source-checked、tilelang kernel 未执行。G-C1。
2. **Qwen3.6 MTP**：只有 config + 目标首层模板；HF 忽略 `mtp.*`、vLLM 主模型
   丢弃 `mtp.` → method status=unverified，不进入部署排名。
3. **DFlash**：服务端集成（SGLang DFLASH/vLLM PR#40898）未逐行核对；拒绝采样/
   KV 回退/hidden 提取仍 0 占位；workspace/graph/NCCL 池尺寸未知。G-C3。
4. **设备可分配预算/价格**：无来源 → feasible=unknown、成本 null+reason。
   H200 两闭环绑定已归档的 30B-tiny/235B，不能跨模型使用。G-D7/G-D9。
5. **CPU/GPU 执行**：Qwen 固定 HF 源码两层随机权重 CPU shape probe 已执行；
   R06（2026-11-12）追加 GDN+MoE fixture 级形状与算量对照（h=32/2 层/4E 随机权重，
   逐 GEMM/conv/router 与本仓 tp=1 账本精确相等，ev-qwen36-cpu-fixture-r06）；
   两款真实主模型仍无完整 forward 或 GPU/部署实测，模型级 cpu_forward 保持
   not-run，不冒充模型级 shape/numeric-checked。
6. **历史归档**：30B/235B/Next/V3 与 tiny fixture 只可显式注入归档目录；
   `h200_calib_*` 场景标 archived。Qwen3.8 本地身份仍 PENDING_ASSET。
7. pi-viewer 适配器（V09b）：未建；数据通道经蒸馏已通（无需写旧仓）。
8. 远端 required check/PR：2026-11-11 已配置。历史事实：私有仓库时期（2026-11-10
   修复前）只读查询 main branch protection 与 rulesets 均返回 HTTP 403
   （需 GitHub Pro 或转为公开）。仓库现已公开；同日复查询返回 404
   `Branch not protected`、rulesets 为空，当时强制检查尚未启用。此后 main
   branch protection 曾要求 Windows pytest 和 1 位 approving review。
   2026-11-11 按仓库所有者决定调整为单人仓规则：保留 PR、Windows pytest
   required check（strict）和管理员保护；required approving reviews 为 0。
9. 移动端像素观感：截图存证但本会话通道无法目检；`tmp_path` 用例在本机
   DSH 沙箱目录权限下未能完成（setup PermissionError）。历史事实：hermetic
   修复前外层本地 `196 passed` 因复用 ignored `outputs/run_v41.json`，不代表
   clean checkout；远端 run `42803015653` 的 2 项失败即该文件缺失。两项验收
   已按 hermetic 方式修复，外层 Codex 于 2026-11-10 验收本地全量 exit 0：
   `196 passed, 1 warning in 5.04s`。修复文件尚未提交/推送，远端重跑待外层执行
   （见 [Windows CI hermetic 修复](validation/windows-ci-hermetic-fix-20261110.md)）。

## 4. 恢复入口

- **一键**：双击仓库根目录 `启动工作台.bat`（起服务+开浏览器，可选端口参数；
  重复双击=只开浏览器；`停止工作台.bat` 停止）。已实测 冷启动/已在运行/停止 三路径。
- 测试：`py -3.12 -m pytest tests -q`
- 工作台：`py -3.12 -m simverse.cli serve`（/ 生产、/legacy 旧视图）
- 必需场景：`py -3.12 -m simverse.cli run-v2`（内置默认=Qwen3.6+DFlash colocated）
- v0.1 存档重放：`py -3.12 -m simverse.cli run`（legacy）
- 证据产物（已入册 docs/evidence/）：workbench_desktop_run.png、
  workbench_mobile_run.png、run_v2_qwen36_dflash.json、ops_v2.csv、
  trace_v2.csv、report_v2.html（历史轮）；A 组轮：
  run_v2_qwen36_dflash_agroup.json、ops_v2_agroup.csv、
  trace_v2_agroup.csv、report_v2_agroup.html、workbench_ep_budget_run.png。
- 下一工作项：V4.1 CSA2/MoE scoring 与量化映射源 pin、Engram DP 分片交换
  路径（或显式永久登记单副本口径）；Qwen3.6 MTP 原生 forward；
  两款主模型 CPU forward 与 GPU 实测；DFlash 服务端集成/运行时池尺寸；
  可分配预算 G-D7；G-D4/G-C2 prefill/verify 形状域；G-C4 单请求超前 → M17；
  V09b 三方对账。历史 235B MoE 蒸馏仅作归档研究，不再是产品优先项。


## 2026-11-10 产品范围收敛

产品目录仅展示 Qwen3.6-35B-A3B 和 DeepSeek-V4.1-Flash。历史模型已移出运行时
catalog，只作离线归档回归/校准资产；tiny 只作测试 fixture。用户已确认 Qwen3.5
为笔误，沿用 Qwen3.6-35B-A3B 及其 DFlash 配对。
默认 Qwen3.6+DFlash 不排名同时受限于：forward 证据未完备（MTP 未核）、
真实主模型 CPU/GPU 未完整执行、可分配显存未知——不能简化成单一“OOM”或“结构通过”。DFlash
checkpoint 禁止用于未经核对的其他主模型；历史 profile 跨模型直接拒绝。
本轮详细功能状态见 [FOCUS_STATUS.md](FOCUS_STATUS.md) 与
[两模型审查记录](validation/two-model-forward-merge-gate-20261110.md)。
