# R10 入口/扫描口径/比较资格修复验证报告（2026-11-19）

基线：`d8dfbfc`（main）。交付分支：`fix/r10-entry-scan-compare`（独立 worktree
`D:/Github/new-viewer-wt-r10`）。两个批次可独立回归；本报告按合入准则 §4.4
披露来源、验证方式与剩余限制。不涉及昇腾验证、采数或硬件运行。

## 0. 修复前最小复现（基线 d8dfbfc 上实测）

| ID | 复现 | 修复前行为 |
|---|---|---|
| R10c | 从完整 Git 快照构建 wheel、隔离 venv 安装、仓库外 `simverse run-v2` | `models.py`/`model_ir/catalog.py` 各自硬编码 `parent.parent[..]/assets` → `site-packages\assets\config\deepseek_v41_config.json` 缺失，命令失败 |
| R10d-a | CLI 默认场景 | 硬编码在 `cli.V2_DEFAULT_SCENARIO`；场景目录无同一份预设，三入口不同源 |
| R10d-b | `POST /api/v2/run` body=`[1,2,3]`；`deployment:[1]`；`workload:"x"`；`k=99` | `AttributeError`/`TypeError` → Flask **HTML 500**（sweep 同） |
| R10b | 打开 `/docs/ascend-validation-plan.md`，点 `[ROADMAP](../ROADMAP.md)` | 相对链接落 `/ROADMAP.md` → **404**（docs 路由只开放 docs/） |
| R10e-a | `sweep_1d(..., metric_key="M01_tpot_analytical")`（选用≠解析的记录场景） | 实际读 `M01_tpot_selected`；baseline 恒取 `M01_baseline_selected`；speedup 口径混搭；扫描级 `metric_key` 谎报 |
| R10e-b | `find_crossover` 点列 `[0.5, None, 1.5]` | 过滤缺失点后跨未知区间插值 → 返回 2.0（应 None） |
| R10e/f | `cand.capability = {"checks": {}}` 后 `compare_runs` | `eligible=True`、无原因（两套资格判定、缺失项视而不见；same_time_basis/result_kind_compatible 声明未检查） |

## 1. 批次一：安装、场景入口与工作台文档

### R10c wheel 资产定位（统一源码/wheel）

* `simverse/assets.py`：唯一资产根解析——源码检出（含 editable）优先仓库
  `assets/`、`docs/`；wheel 回退包内 `simverse/_data/`（构建期拷贝）。
  `code_revision` 同步委托（wheel/无 git 如实 `unknown`，不伪造）。
* 清单按**实际消费路径**确定（audit-hook 文件读取追踪：默认 run-v2 只读 3
  份 config；`/api/v2/catalog` 追加 identity/references/profile
  manifest/scenarios）：`assets/config/**`、`assets/identity/**`、
  `assets/scenarios/**`、`assets/profiles/*/manifest.json`（仅证据哈希引用
  的清单）、`docs/rebuild-20261106/references/**`。**不进 wheel**：
  `assets/cost_records/**`（昇腾归档观测，显式选择才加载，缺失按
  ProfileIntegrityError 如实报错）、profile 数据文件（归档模型校准）、
  完整 docs/。受哈希约束资产逐字节拷贝（`setup.py` build_py 钩子；清单与
  核对共用 `WHEEL_DATA_MANIFEST`）。
* `.gitattributes` 补 `assets/scenarios/*.json text eol=lf`：场景 JSON 与
  config/identity 同为 digest/哈希敏感资产；不钉时 Windows autocrlf 使
  `git archive`（冒烟快照源）产出 CRLF 副本，破坏字节核对。
* CLI 默认输出：wheel 模式改写 `cwd/outputs`（site-packages 不可写）；
  源码检出仍写仓库 `outputs/`（既有行为）。

**wheel 冒烟**（`tools/wheel_smoke.py`，CI merge-gate 新增步骤，保留编辑
模式检查）：git archive 完整快照 → 隔离构建 wheel → 独立 venv 安装 →
仓库外真实 `simverse run-v2`（exit 0、status=completed、run_id 产生、模块
来自 site-packages、`repo_root() is None`）→ 工作台 `/` 与
`/api/v2/catalog` 200 且恰两款主模型 → 清单 219 文件逐字节与仓库一致。

### R10d 共享默认预设 + run/sweep 输入校验

* `assets/scenarios/qwen36_dflash_default.json` = 原 CLI 字典逐键固化
  （**digest 与历史默认 run 对齐**，测试锚定）；`simverse/presets.py`
  载入/目录；CLI `run-v2`/`sweep-v2` 缺省读预设；
  `/api/v2/catalog` 新增 `scenarios` 段（id/name/digest/scenario）；
  legacy `/api/scenarios` 只列 v0.1（旧场景读取行为不变）。
* 工作台：`sel-preset` 预设选择器；目录加载完成时若用户未改动则默认载入
  共享预设；**预设模式原样提交**（场景 digest 与 CLI/API 完全一致），改动
  任一控件即回自由配置（name 回 `workbench-*`）。
* `simverse/api_io.py`：run/sweep/CLI 共用解码。结构/类型/取值非法 →
  `ScenarioInputError`（JSON Pointer 风格字段路径）→ 结构化 400
  `{"error", "field"}`；畸形 JSON、顶层数组、`deployment`/`workload`
  数组、`k`/`alpha` 越界、嵌套未知键、budget 类型非法等矩阵全覆盖；
  acceptance 域（kind/alpha/alphas/mean）在入口校验（与
  methods.acceptance 同域）。合法但不受支持组合（如 MTP+DT）沿用结果
  状态（200 + `unsupported_combo`，能力说明保留）；真实内部异常保持
  服务端 500（测试固化）。

### R10b 根目录文档白名单

* 新增 `/<root_doc>` 路由，白名单恰 `ROADMAP.md`、`AGENTS.md`（docs 内
  状态/验证文档实际引用的两个根文档；README 未被引用不入）。
* 验证链：`/docs/FOCUS_STATUS.md`、`/docs/IMPLEMENTATION_STATUS.md`、
  `/docs/ascend-validation-plan.md` 200 → `/ROADMAP.md` 200 → ROADMAP
  指向的 `docs/MERGE_CRITERIA.md`、`docs/FOCUS_STATUS.md`、
  `docs/IMPLEMENTATION_STATUS.md`、`docs/ascend-validation-plan.md`
  逐个 200。白名单外根路径（README.md/pyproject.toml）、不存在文档、
  穿越路径均 404。wheel 模式无仓库文档 → 如实 404。

## 2. 批次二：扫描口径、交叉点与部署比较资格

### R10e sweep_1d 指标口径（`simverse/experiments/sweep_v2.py`）

* 基线配对表 `BASELINE_PAIR`：`M01_tpot_selected↔M01_baseline_selected`、
  `M01_tpot_analytical↔M01_tpot_baseline`——**speedup 分子分母同口径**。
* `metric_key=None`（默认）= 选用成本；旧结果缺选用指标 → 回退解析口径，
  点级+扫描级 `fallback_reason` 明示（兼容历史 bundle）；选用基线单独缺
  失也回退解析基线并明示。显式请求 → 严格该口径；指标缺失保留空值与
  原因（不静默换口径）；未知指标名拒绝。
* 点级新增/明确：`metric_key`（实际口径）、`baseline_metric_key`、
  `metric_source`（cost.selection.source：analytic/records/profile）、
  `fallback_reason`、`reason`；`run_id`/`status`/`record_coverage` 保留。
  扫描级：`metric_key`（实际主口径=扫描序首遇）、`requested_metric_key`、
  混用时 `metric_keys_used`、`fallback_reason`。
* 真实分叉场景：`ascend-r29-hccl + dflash_v1 + colocated + TP2`（collective
  按记录计价、其余解析回退）——选用/解析的候选、基线、speedup 六值全分叉
  （17.028/22.110/1.298 vs 18.231/25.748/1.412），默认与显式两条扫描
  分别按各自口径手算核对。

### R10e find_crossover（确定性规则，保留单值接口）

按原始扫描顺序处理：第一个 speedup 恰=1 的点直接返回其值（连续=1 取
首个）；否则第一个**相邻**符号翻转区间线性插值（多交点取扫描序首个）；
无效点（缺失/失败）或口径不可比（相邻 `metric_key` 不同）截断区间——
不跨未知区间插值；无交点 None。手算锚点 9 组覆盖：正常翻转、区间外缺失
（相邻有效对仍可插值）、区间中缺失（None）、口径不可比（None）、端点
恰=1（含首点）、连续=1、多交点、无交点、等值先于后续翻转。

### R10e/R10f 统一资格与比较口径

* 新模块 `simverse/experiments/eligibility.py`（单一事实源）：目标端
  必需检查（结构/config 覆盖/forward/CPU/GPU/部署排名证据/并行配方）
  **缺失即 unknown 并给具体检查名**；草稿端按场景适用域（dflash_v1 →
  draft_forward_evidence + dflash_serving_integration；mtp →
  mtp_serving_integration + method_recipe_ready）；baseline 只查适用项；
  阶段成本 stage_cost_complete 按适用域（V4.1+Engram host 查表）必需、
  域外存在性检查；容量 memory_* 至少一条且全部 supported——**预算
  unknown、假设预算（approximated）、OOM（unsupported）均不可排名**；
  附加条件（ascend 配方/接受率已知/profile 域/归档记录）存在即检查。
  `engram_host_lookup_latency` 不单列（与 stage_cost_complete 同源，后者
  已拦截；重复计数只会改写冻结的 M02 reason 措辞，不放松判定）。
* 管线 `_eligibility/_eligibility_issues` 与 `compare_runs` 均改接同一
  函数（带各自 scenario_snapshot 适用域）；M02 数值与既有 reason 语义
  不变（R27 replay `--check` 逐字节通过）。
* `compare_runs` 口径核对：相对变化（delta）要求双方指标单位、时间基准、
  统计量、适用范围（role/phase/aggregation）、结果类型相容；不相容保留
  双方数值与原因、delta=None。新增 `objectives[*].compatibility`
  （`{compatible, reason}`）——**只补充核对结论，不改变既有键语义**；
  旧行为在口径不匹配时仍给混口径 delta，正是修复点。新增
  `baseline_ineligible_reason`（baseline 适用项不满足时披露）。声明的
  same_time_basis/result_kind_compatible 谓词自此实际执行。
* 保留：既有 JSON 字段、指标别名（M01_record_* 等兼容视图）、原始
  Work/Traffic/Memory、成本组所有权、仿真数值锚点（全量测试 + R27
  replay `--check` + golden fixtures 通过）。

## 3. 测试与验证记录

* 新增 `tests/test_entry_points_r10.py`（32 项：资产定位双模式、清单
  覆盖/排除、预设 digest 对齐与三入口一致、CLI 真实 run-v2、畸形请求
  参数化矩阵 ×run/sweep、内部异常 500 语义、根文档白名单与链接链、
  workbench 预设逻辑静态断言）；`tests/test_scan_compare_r10.py`
  （32 项：口径矩阵、回退/缺失、交叉点手算锚点、资格矩阵（空 checks/
  单项缺失/草稿适用/预算三态/阶段成本适用域）、compare 正例与口径
  五维负例、CLI+API 扫描兼容性）。
* `tests/test_merge_gate.py::test_g15_forward_gap_alone_blocks_eligibility`
  更新：新语义下必需检查缺失也算缺口，故把除 forward 外的必需检查补齐
  supported，保留"仅 forward 缺口独立拦截"的原意图；缺失拦截行为由
  资格矩阵新测试覆盖（准则 §4.3：行为修复驱动的期望更新，在此单列）。
* 命令与结果（Windows/Python 3.12，worktree 内）：
  * `py -3.12 -m pytest tests -q` → **644 passed**（收集 644；
    IMPLEMENTATION_STATUS 已同步，G6 核对通过）
  * `node --check simverse/static/workbench.js` → 通过
  * `py -3.12 tools/wheel_smoke.py` → 全部通过（清单 219 文件字节一致）
  * `py -3.12 -X utf8 tools/replay_cost_integrity_r27.py --check` →
    counts invariant; anchors hold（M01/M02/成本组数值零漂移）
  * `git diff --check` → 通过
* 数值口径：本轮**不改变任何 Work/Traffic/Memory 计数与解析/选用成本数
  值**；唯一数值语义变化是比较侧 delta 在口径不相容时置 None（修复点）。

## 4. 剩余限制与未验证范围

* wheel 冒烟的清单字节核对以本机仓库为准；`git archive` 遵循
  .gitattributes（scenarios 已补 eol=lf）。未覆盖 Linux/macOS runner
  实跑（CI merge-gate 为 windows-latest，与仓库门禁一致）。
* 工作台预设模式的浏览器级（真实 DOM）验收走 testhook/静态断言 +
  Flask test_client；本轮未做截图级 UI 回归（既有 UI 测试全绿）。
* compare_runs 的口径核对覆盖默认三目标的 Metric 对象路径；dict 型
  metrics（M06_detail 等）不走 delta，行为与旧版一致。
* eligibility 的 ascend_device_recipe 等条件项为"存在即检查"：若未来
  管线在更多条件下添加新检查，需同步登记适用域（模块注释已声明）。
* 未涉及：昇腾验证/采数/硬件运行、R10f 其余侧（来源与组所有权审计在
  PR #26 等既有轨道）。
