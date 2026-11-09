# Simverse 重构实施状态（M0–M5 验收清单）

更新：2026-11-09。主控会话持续维护；上下文压缩后从本文件恢复。
代码身份：本仓 git（本地，无远端——Q3 私有远端信息待补）。
运行：`py -3.12 -m pytest tests -q`（当前 68 passed）；
工作台 `py -3.12 -m simverse.cli serve` → http://127.0.0.1:8765/（生产 v0.2 首页 /，v0.1 简表 /legacy）。

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

### M3 资源事件核 — 部分完成（解析档全通；资源核未接）
- [x] 指标归约器（execute/reduce.py）：请求口径 ITL（同刻=0 合法/N≤1=None）、
  窗口 token 事件 vs 完成请求两口径、goodput(SLO，取消单列)、busy 并集积分、
  最近秩分位；人工轨迹独立 oracle 全对账（tests/test_metrics_v08.py，V08）。
- [x] 事件验证器（execute/validate.py）：missing_dependency/
  starts_before_dependency_end/resource_overlap/consume_unknown_message/
  consume_before_send_end/missing_message_edge 六类检查，负例逐条捕获（V07）。
- [ ] 资源约束事件核（两微批流水/单请求超前/取消浪费）——未实现：合法事件
  生成器未接（execute/serial 为串行解析链）。流水部署不入矩阵（UI 无假并行条），
  contract 留待扩展；这是 M3 剩余主体。

### M4 服务与校准 — 契约就绪，能力未开放（如实标注）
- 服务特性 UNSUPPORTED 注册：chunked_prefill/pd_disaggregation/prefix_cache
  （methods/plans.py UNSUPPORTED 表，Q5 顺序；不空实现）。
- profile 契约+合成 fixture 接线完成（M2）；真实采集/拟合/holdout 无数据（Q2 unmeasured）。
- Chunked Prefill / PD 分离 / Prefix Cache：未实现，能力目录未注册为支持；
  扫描/比较路径遇及时显式 unsupported（方法 UNSUPPORTED 表）。
- 服务到达/组批/KV 分页：workload.arrival="fixed-batch"（M4 扩展点已留）。

### M5 迁移与扩展 — model-verse 适配器已建；其余待条件
- model-verse 薄适配器：LLM-Viewer-main/simverse_adapter/（新增文件，未改旧仓
  既有文件）：哈希锁（adapter_lock.json，6 文件 SHA256）+ 固定 commit 8796607 +
  model_ir 0.3.0；自检报告（adapter_report）通过；漂移即拒绝。
- pi-viewer：不在本会话可写目录（阻塞：需用户加入工作区或自行套用
  simverse_adapter 同款模式）。
- 远端：Q3 平台/账号/地址待补；未创建/推送任何远端。
- v0.1 模块退役：待 M5 三方对账后。

## 2. 页面/指标/验证映射（当前覆盖）

| 验证 | 状态 | 证据 |
|---|---|---|
| V01 身份与协议 | 通过 | test_negative_v02（digest 全长/大整数/NaN/null+reason）|
| V02 结构与 forward | 部分 | 层序/覆盖/源码 pin+分支映射（source_map）；小尺寸 CPU 执行 trace 未做（无 torch 环境，登记缺口）|
| V03 独立数学 oracle | 通过 | golden/oracles_v02 + test_model_ir/test_negative |
| V04 lowering 与守恒 | 部分 | TP 校验/collectives 入账/容量对象；TP/EP 分片总量守恒断言未做全 |
| V05 方法、协议与状态 | 部分 | 接受律/端点/未知传播/DFlash 计划绑定；事件级 transition trace 未做（M3）|
| V06 显存生命周期 | 部分 | 静态+每请求对象、同卡去重；生命周期交错峰值未做（M3）|
| V07 事件因果与资源 | 部分 | 验证器六类检查+负例；资源约束事件核未做（M3）|
| V08 系统指标归约 | 通过（人工轨迹）| execute/reduce + test_metrics_v08 |
| V09 比较与旧仓对账 | 部分 | 比较资格/多目标/资源披露（V09a）；旧仓三方对账（V09b）未做（M5）|
| V10 扫描与运行隔离 | 部分 | 不可变快照/独立身份/无交点 null；取消/恢复未做（单任务门控待做）|
| V11 校准与外部真实性 | 未测（无数据）| 契约接线用合成 fixture；保持 unmeasured |
| V12 UI 与导出一致性 | 通过 | test_ui_v12 + 浏览器实测记录（本文件 §1 UI）|

## 3. 已知缺口与外部依赖（不虚构补齐）

1. Qwen3.8 本地模型身份：待用户提供本地路径/revision（PENDING_ASSET 注册）。
2. V4.1 Engram/hc（11 键）：参与 forward（vLLM pin 证实）但未建模 →
   structure_complete=False，完整容量/性能结论降级（capability 检查可见）。
3. DFlash 服务端集成（SGLang DFLASH/vLLM PR#40898）未逐行核对 → 部署矩阵
   不显示“已部署验证”。
4. 设备可分配预算/真实 profile/价格：无来源 → feasible=unknown、unmeasured、
   成本 null+reason。
5. 小尺寸 CPU forward 数值执行：本机无 torch 环境 → V02 数值语义分栏为
   source-checked（如实，不冒充执行）。
6. pi-viewer 适配（M5）：不在可写目录。
7. 远端平台/账号（Q3）：待补。
8. 移动端像素观感：截图存证但本会话通道无法目检。

## 4. 恢复入口

- 测试：`py -3.12 -m pytest tests -q`
- 工作台：`py -3.12 -m simverse.cli serve`（/ 生产、/legacy 旧视图）
- 必需场景：`py -3.12 -m simverse.cli run-v2`（内置默认=Qwen3.6+DFlash colocated）
- v0.1 存档重放：`py -3.12 -m simverse.cli run`（legacy）
- 证据产物（已入册 docs/evidence/）：workbench_desktop_run.png、
  workbench_mobile_run.png、run_v2_qwen36_dflash.json、ops_v2.csv、
  trace_v2.csv、report_v2.html。
- 下一工作项：M3 资源约束事件生成器（两微批/超前/取消浪费）→ 三轨时间轴接真
  并发事件；V04 TP/EP 分片守恒全断言；V09b 三方对账（model-verse 已有适配器）。
