# Simverse 重构实施状态（M0–M5 验收清单）

更新：2026-11-10（A 组轮：TP/EP per-rank 配方 + 同卡对象账本 + 预算判定链）。主控会话持续维护；上下文压缩后从本文件恢复。
代码身份：本仓 git（本地，无远端——Q3 私有远端信息待补）；包版本 0.2.0（对齐结果协议 v0.2；ModelIR 独立 0.3.0）。
运行：`py -3.12 -m pytest tests -q`（当前 176 passed，无警告输出）；
工作台 `py -3.12 -m simverse.cli serve` → http://127.0.0.1:8765/（生产 v0.2 首页 /，v0.1 简表 /legacy）。
合入准则：[MERGE_CRITERIA.md](MERGE_CRITERIA.md)（G1–G7 门禁在 tests/test_merge_gate.py，测试数与本文件一致性由 G6 强制）。

## 0. 必读上下文

- 计划包：`docs/rebuild-20261106/`（EXECUTION_PROMPT.md 为本任务指令；
  questions.md Q1–Q12 已答；plan/principles/ui-data-contract 为验收依据）。
- v0.1 切片（旧 13 测试）保留于扁平模块（engine.py/ledger.py/...），
  contracts.py→contracts/legacy.py、cost.py→cost/legacy.py（同路径迁移，
  旧导入兼容）。v0.2 管线为 contracts/count/methods/lowering/cost/execute/
  experiments/pipeline/scenario 包。两套由 schema_version 0.1/0.2 区分，
  M5 退役 v0.1。

## 1. 里程碑状态

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
- 必需场景数字（Qwen3.6+DFlash colocated 8×H100 B1 ctx32k α0.72 blk16）：
  基线 TPOT 3.4704ms、投机 1.6874ms、加速比 2.0567（容量标称口径→不排名标注）、
  E[N]=3.5528、T_round=5.995ms（draft 0.163+verify 5.832）。

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
  （准则规范）+ G1–G7 自动门禁（tests/test_merge_gate.py）。
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
| V02 结构与 forward | 部分 | 层序/覆盖/源码 pin+分支映射（source_map）；小尺寸 CPU 执行 trace 未做（无 torch 环境，登记缺口）|
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

1. Qwen3.8 本地模型身份：待用户提供本地路径/revision（PENDING_ASSET 注册）。
2. V4.1 Engram（8 键）：hc 三键已建模（2026-11-10）；Engram forward 源
   （common/engram.py）未入评审包 → 仍 structure_complete=False。G-C1。
3. DFlash 服务端集成（SGLang DFLASH/vLLM PR#40898）未逐行核对 → 部署矩阵
   不显示”已部署验证”。G-C3。
4. 设备可分配预算/价格：无来源 → feasible=unknown、成本 null+reason。G-D7/G-D9。
   真实 profile：H200 两闭环（30B-tiny TP1 / 235B TP2-8）；MoE/verify/prefill/
   E2E/kernel-only 见 G-D2/G-D4/G-D5/G-D6。
5. 小尺寸 CPU forward 数值执行：本机无 torch 环境 → V02 数值语义分栏为
   source-checked（如实，不冒充执行）。
6. pi-viewer 适配器（V09b）：未建；数据通道经蒸馏已通（无需写旧仓）。
7. 远端平台/账号（Q3）：待补。
8. 移动端像素观感：截图存证但本会话通道无法目检。
9. 子进程输出编码：当前全量测试无编码警告（2026-11-10 复核）；CLI 输出检查按
   plan §6 保持为稳定验收项。

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
- 下一工作项：G-D2 235B moe 蒸馏准入（G-C11 已解除，账本口径就绪）；
  G-C1 Engram 源 pin（等上游评审包）；G-D4/G-C2 prefill/verify 形状域；
  G-D6/G-C5 KERNEL_ONLY 双口径；G-D1/G-C3 DFlash 实测与集成核对；
  G-C4 单请求超前事件 → M17；V09b 三方对账。


## 2026-11-10 产品范围收敛

产品目录仅展示 Qwen3.6-35B-A3B 和 DeepSeek-V4.1-Flash。历史模型保留为内部回归/校准资产，不再作为产品扩展优先项。用户已确认 Qwen3.5 为笔误，沿用 Qwen3.6-35B-A3B 及其 DFlash 配对。
默认 Qwen3.6+DFlash 结构检查通过；不排名的直接原因是可分配显存未知，非已确认 OOM。现在输出实际检查原因。DFlash checkpoint 禁止用于未经核对的其他主模型。
本轮详细功能状态见 [FOCUS_STATUS.md](FOCUS_STATUS.md)。
