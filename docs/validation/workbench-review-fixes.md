# 工作台交互/展示/导出/文档一致性审查修复（2026-11-17）

任务范围：Simverse 生产工作台（simverse/static/、simverse/webapp.py、前端与 CLI
导出、产品展示能力适配、本任务新增测试、ROADMAP.md、docs/gaps/
capability-gaps.md、本报告）。审查基准 `11965105d03252dc2a45d22b68d6d668c2c61504`。
不修改：simverse/cost/、simverse/pipeline.py、原始计数、运行身份契约、
docs/IMPLEMENTATION_STATUS.md（另一独立任务负责成本与运行身份）。

**实施状态：已完成并验证**（自动化 20 项新测试 + 1 处旧断言同步修正 +
真实浏览器交互验证；见 §5–§6）。全部前端改动为零物理计算原则下的展示层
适配，不新建与后端冲突的规则。

## 1. 已修复问题与可见行为

### A. 文档导航（四个 404 链接）

- 新增 `GET /docs/<path:fn>`（webapp.py）：只开放仓库 `docs/` 根
  （`send_from_directory` 自带穿越/绝对路径拒绝 + resolve 后二次锚定 docs 根
  校验）；`.md` 以最小 `<pre>` HTML 包装（转义、无渲染依赖、无 JS），其余
  文件按原样发送。不注册 docs 之外的任何文件入口。
- workbench.html 顶栏四个链接改为绝对路径 `/docs/rebuild-20261106/...`
  （原 `../../docs/...` 在 Flask 根路径 `/` 下解析为 `/docs/...` 本就 404，
  且随页面路径深度不稳定）。
- 原型存档页内的相对资源（`v3-workbench.js`、`style.css`、`../plan.md`）
  经同路由自然解析，无需复制。

### B. 矩阵执行

- `runCell()` 返回真正的 Promise（成功/失败/unsupported 处理完成后才
  resolve，内部不 reject）。
- `runMatrix()` 严格串行（reduce 链逐格 await）；实测**最大在途请求 = 1**。
- 明确行为：单格失败/unsupported 不中断后续格子；失败格显示
  「不支持/失败」+ 后端原因（title 悬停全文），不显示成功数值。
- `btn-matrix` 批量期间禁用并显示「串行运行中…」，finally 恢复；
  `state.matrixRunning` 防重复点击的重叠批次。

### C. Sweep

- 检查 HTTP 状态：非 2xx 时抛出后端 `{error}` 原因；非 JSON 响应显式
  「HTTP xxx：扫描响应非 JSON」；网络错误显示异常消息。三种情况均渲染
  unknown-note（不渲染为成功空表），finally 恢复按钮。
- 逐点渲染检查 `status==='completed' && speedup!=null`：失败/未知点显示
  「未知/失败—<原因>」，不渲染成加速比数值。

### D. 模型/方法/部署/控件联动

组合支持表（`COMBO` 常量）逐条镜像 pipeline.run_v2 的硬拒绝/能力判定
（后端仍为权威，前端只做运行前提示与禁用）：

| 组合 | UI 行为 | 后端权威结果 |
|---|---|---|
| V4.1 × dflash_v1 / mtp | 方法选项禁用 + 原因；已选中则切回 baseline 并提示 | failed bundle（配对拒绝） |
| V4.1 精度 | 选择器禁用 + 显示冻结配方「MXFP8 线性/MXFP4 专家/BF16 激活（R22）」 | quant 域校验拒绝其他值 |
| Qwen MTP × dt_serial/dt_pipeline | DT 选项禁用 + 原因；已选中则切回 colocated 并提示 | failed bundle（MTP DT 无实现） |
| Qwen DFlash × dt_* | 可运行；提示「解析研究假设：两栈无 draft/target 分离部署，可计算、不参与排名，不等于已验证服务栈」 | completed + dflash_serving_integration=unsupported（不排名） |
| DFlash colocated | 提示「双栈源码核对支持（source-checked，非已部署实测）」 | completed + supported(source-checked) |
| MTP colocated | 提示「双栈源码核对（R23）；接受率假设；kernel/CPU/GPU 未执行未实测」 | completed + method_recipe_ready=approximated |
| baseline 方法 | 部署选择器禁用（baseline 专用模式） | mode 必须为 baseline |
| colocated/baseline 草稿设备 | 禁用 + 「草稿与 target 同 TP 组，使用目标设备」 | draft_dev 仅 dt 模式消费 |
| TP16 × V4.1 | 不预先判定（前端无 o_groups 事实） | 后端拒绝（o_groups=8 不整除） |

- TP 下拉补 **TP4**（两产品模型 TP4 并行配方合法，测试+实跑验证）；
  保留 TP16（Qwen 合法，V4.1 由后端权威拒绝）。
- MTP 标签更新为源码核对后的真实状态（R23 双栈 pin、K≤n 撤销、
  接受率假设、CPU/GPU 未执行）；不再出现「未验证：config+目标首层模板」
  旧文案。
- 删除「chain_draft 和 chain_moe_proxy 可在方法选择器中查看研究演示」
  旧提示与死代码标签——chain 不重新开放（目录抽屉标注内部研究资产）。
- 无法提前精确判定的组合（TP 整除域、EP 域、profile 域、预算判定）
  一律保留后端权威错误处理，不伪造 supported。
- 目录抽屉新增「结构」列（structure_complete，来自 model_ir 事实）与
  方法目录摘要行。

### E. 七维透视（统一聚合范围）

按算子名聚合整组行（`aggOp`），七维口径：

1. **时延**：组内计价行（roofline_points 按 `row_id=stage:layer:op` 关联）
   t_ms 求和；未进后端导出 top-40 计价行的组显示「–」+ 原因（不显示 0）。
2. **逻辑算量**：组内 flops 求和（精确）。
3. **驻留容量**：**代表行**容量 + 显式注记「物理对象跨行共享，不按行求和；
   总容量以 M06 内存账本为准」；组内多种驻留值时标注数量。
4. **五类字节**：W/A/KV/状态逐类组内求和；**精确 int 与期望 float 分列**
   （期望另计 + 期望行数；不再以 `agg.w || first_row.weight_read` 掩盖）。
5. **形状**：组内多种 shape 时显示「N 种 shape（逐行见下）——不代表全组
   相同」+ 代表行 layer 与 shape（V4.1 sparse_attention 等实测 2 种）。
6. **精度/统计身份**：从 scenario_snapshot 读取（V4.1 显示冻结 8/4/16）；
   statistic 按组内集合列出。
7. **公式**：代表行公式 + bundle.formula_dict 表达式。
- **bound**：组内计价行 bound 分布（memory×n/compute×m/混合），
  不再沿用首行结论；无计价行显示「–」。
- 新增**行级查看**（前 30 行逐行 row_id/shape/FLOPs/字节/t_ms/bound，
  可追溯；全量走 CSV 导出）。

### F. Roofline 展示

- **新元数据路径**（能力检测 `peak_tflops/effective_peak_tflops/
  bandwidth_tb_s/compute_dtype/cost_basis`，不假设存在）：实线画**有效
  成本上界**（斜线→平台 + 拐点），虚线画**规格上界**（dense 峰值），
  标签区分二者并展示 cost_basis 与 compute_dtype；单位 TFLOP/s 与
  十进制 TB/s。
- **旧响应路径**（无元数据）：只画坐标点，明确标注「计算上界元数据未
  提供：后端未随点输出峰值/带宽档位，本图不推断（不硬编码 FP8 峰值）」。
  **删除固定 1979/3000 TF/s、800 F/B、0.2 下界、3.35 TB/s 回退**。
- 坐标范围动态计算（数据对数轴 + 1.6× 边距）；空值/零值/负值过滤，
  全空时显示「无有效坐标点」。
- 选点用稳定 `row_id` 匹配代表行；**无对应点时显示「代表行 Lx 无对应点
  （未进后端导出计价行）」，不回退其他算子冒充**；其余计价行淡点作
  上下文，代表行高亮并标层号。
- 「实达」表述改为「解析估算（T0，非实测）」。

### G. 双端显存（M06）

- KPI 卡分端显示（colocated：各 rank 值 + worst 标注；dt：
  `draft_r0 X · target_r0 Y`），**不把不同设备容量相加、不合并出单一
  部署结论**；只取第一个 `M06_memory*` 的旧逻辑删除。
- 新增「显存与预算」抽屉：逐端/逐 rank 表（端名、设备标称容量、已知
  峰值、判定、预算输入、原因/缺口组件、steady/transient/峰值完整性）。
- 四态区分：`oom` / `unknown` / `feasible+support=approximated`（条件
  可行（假设预算））/ `feasible+supported`（可行）。`_detail` 按账本
  字段展示（明确非 Metric）。
- 多 rank（EP 余数）逐 rank 单列 + worst rank 标注（后端
  colocated_rN/colocated_worst 键全部消费）。

### H. 预算输入（既有契约内）

- 目标端预算 + **DT 草稿端独立预算**（`deployment.draft_budget`，仅 dt
  模式启用）+ 额外运行时开销 GB（>0 时带显式来源字符串）。
- 数值输入一律 `hypothesis`（来源标注「工作台用户输入（假设；非实测）」），
  只产生条件结论；留空 = unknown（不套标称）。**不提供 measured 切换**——
  measured 需可追溯来源与环境（准则 §3.2），界面数字输入无法表达，
  不伪造。开销留空按 0 处理但界面明示「未证实为零」语义。
- 单独补预算不解除其他部署资格门禁（后端 R03b 判定链不变）。

### I. CSV 与导出

- 新增 `static/csv.js`（RFC 4180：逗号/引号/换行字段引号包裹、内部引号
  翻倍，**不替换字符**）；前端算子 CSV 经该模块生成。
- `cli.py`：`_csv_field/_csv` 统一 v0.1 `_ops_csv/_events_csv` 与 v0.2
  `_ops_csv_v2/_trace_csv_v2` 四处导出（删除逗号→分号替换与换行丢弃）。
- v0.2 算子 CSV 补 `t_ms/bound` 两列，从 `cost.roofline_points` 按
  row_id 关联（无计价行留空，不伪造）；前端导出同列同源（同一
  ResultBundle）。
- `/api/v2/export/<fmt>`：全仓无任何调用方（前端/CLI/测试均不调用），
  旧注释「前端导出前在此登记 manifest」与实际恒 400 行为不符——改为
  **明确弃用**（HTTP 410 + 原因：导出从 POST /api/v2/run 的 bundle 派生，
  服务端不维护「最近一次运行」会话状态）。不新增有状态端点。

## 2. 支持与不支持组合（产品入口现状）

| 模型 | baseline | dflash_v1 | mtp |
|---|---|---|---|
| qwen36_35b_a3b | ✓（任何合法 TP/EP） | ✓ colocated（source-checked）/ dt_*（解析研究假设，不排名） | ✓ colocated（R23 source-checked；接受率假设）/ dt_* 后端拒绝 |
| deepseek_v41_flash | ✓（structure_complete=False，不排名） | ✗ 后端拒绝（配对仅 Qwen3.6） | ✗ 后端拒绝（无原生微头 forward 身份） |

chain_draft / chain_moe_proxy / tiny_*：内部研究资产，不进入 API/CLI/工作台
选择（测试固化）。DFlash DT 的「解析研究模式」≠ 已有真实服务栈支持（两栈
均无 draft/target 分离部署），UI 提示保留该区别。

## 3. Roofline 旧/新响应兼容结果

- **新响应（实测）**：并发后端任务已在 `cost.roofline_points` 提供
  `compute_dtype/peak_tflops/effective_peak_tflops/bandwidth_tb_s/cost_basis`；
  浏览器实跑（Qwen DFlash TP4 colocated）前端自动消费并绘制有效成本上界
  （1484.25 TF/s @ fp8，实线+拐点）与规格上界（1979 TF/s dense，虚线），
  cost_basis 的「dense_w 启发选档（乐观假设）」披露随标签展示。
- **旧响应（实测）**：剥离新字段重画，显示「计算上界元数据未提供」注记、
  仅画点、无 1979/3000/800 硬编码。两种响应同一页面均验证通过。
- 选点按 row_id：无对应点显示原因，不回退（实测 sparse_attention 组
  L0 无点 → 「无对应点（未进后端导出计价行）」）。

## 4. 文档编号与路线图调整

- **G-C15 重号修复**：R23（Qwen MTP）先行登记 G-C15；R24（V4.1 SwiGLU）
  误重用 G-C15。将 SwiGLU 条目改编号 **G-C18**，同步 capability-gaps.md
  内两处引用（G-C1/G-C14 段落）+ 页首编号修复注记；G-C17 段内指向不明的
  「（G15）」改为明确的「G-C1 剩余项与 CPU/GPU/预算缺口」；
  v41-swiglu-tail-20261113.md §7 加编号同步括注（不动其余历史内容）。
  **遗留一行**：docs/IMPLEMENTATION_STATUS.md:149 「G-C15 主干解除」实指
  SwiGLU，该文件本任务禁改——统一汇总时改为「G-C18 主干解除」即可（一行）。
- **ROADMAP.md**：更新日期 2026-11-10→2026-11-17；「当前已交付」补 R26
  （Qwen 文本 MRoPE，structure_complete=True 只表示源码证据完备，不冒充
  CPU 执行/GPU 实测；多模态 2D 域外）、R25（V4.1 核本体，structure_complete
  仍 False）、工作台修复条目；D 组表新增 R26 行（含报告/测试/剩余链接）；
  R20 条目加「后续进展」注（核本体已由 R25 解除、量化映射已由 R22 解除，
  其余部署变体仍缺）——历史条目不无证据改 True。
- 本任务未实施的后端成本修复继续标记为待办（ROADMAP 未宣称）。

## 5. 自动化测试（tests/test_workbench_review.py，20 项）

文档路由（链接 200/穿越 404/md 转义包装/HTML 绝对链接）、导出端点
弃用（410）、CSV RFC 4180 往返（真实 bundle 逐字段 + 毒化字段 +
trace depends_on 列表 + **node 执行 static/csv.js 与 Python 逐样本一致**）、
产品目录边界（无 chain/tiny）、TP4 合法性（两模型）与全链路实跑、
V4.1 dflash/mtp 与 Qwen MTP-DT 后端拒绝事实、M06 双端成键与条件可行
表达（feasible+approximated+assumption_refs）、row_id 可唯一寻址契约、
V4.1 多 shape 算子组存在性、矩阵/防重叠/Sweep 检查源码绊线、sweep 400
错误形状。**验收用 csv.reader 逐字段往返，无 split(",") 验收。**
另同步修正 tests/test_consistency.py `_assert_ops_csv`（旧断言固化的
「逗号→分号」行为与 RFC 4180 修复冲突；该文件为 CLI 导出一致性测试，
属本任务域）。运行：`py -3.12 -m pytest tests/test_workbench_review.py -q`
→ **20 passed**。

## 6. 真实浏览器交互验证（ZCode 内置浏览器 · Playwright 接口驱动，serve 于 127.0.0.1:8765）

| 检查 | 结果 |
|---|---|
| 四个文档链接 + 原型相对资源（页面内 fetch） | 全部 200；`/docs/../simverse/webapp.py` 404 |
| V4.1 控件联动 | dflash/mtp 选项禁用+原因；精度禁用+冻结配方提示；自动切回 baseline 并明示原因 |
| Qwen MTP DT 限制 | dt 选项禁用+后端原因；自动切回 colocated 并明示；草稿设备禁用 |
| TP4 | 选项存在；实跑 completed（TPOT 16.0043ms，run_id e636e85a…） |
| 矩阵串行 | **最大在途 /api/v2/run = 1**（totalPosts=5）；单格失败继续（MTP-DT 格「不支持/失败」+后端原因，其余 4 格完成）；按钮完整恢复 |
| Sweep 成功 | 表格 5 行（4 维度），按钮恢复 |
| Sweep 400 | 「扫描失败：非 baseline 方法不能用 baseline 部署…」（原因可见，无成功表） |
| Sweep 非 JSON（502 HTML） | 「扫描失败：HTTP 502：扫描响应非 JSON」 |
| Sweep 网络错误 | 「扫描失败：Failed to fetch」 |
| Sweep 含失败点 | 「0.8: 未知/失败—E[N] 未知：…」，不渲染成加速比 |
| 旧 Roofline 响应 | 「计算上界元数据未提供」注记；无 1979/3000 硬编码；点照画 |
| 新 Roofline 响应 | 有效成本上界（实线+拐点）+ 规格上界（虚线）+ cost_basis 标签 |
| 无对应点 | 「代表行 L0 无对应点（未进后端导出计价行）」，不冒充 |
| DT 双端显存 | KPI「draft_r0 1.48 GB · target_r0 4.73 GB」（不相加）；抽屉逐端含设备标称/判定/预算/缺口组件/steady/transient |
| 四态分类 | OOM / unknown / 条件可行（假设预算）/ 可行 |
| V4.1 多 shape 透视 | 节点「×40 · 层 0…39 · 2 种 shape」；透视「组内 2 种 shape…不代表全组相同」；容量「不按行求和」；行级表存在；冻结精度 8/4/16 |
| CSV 页面内转义 | `escapeField('say "hi", ok\nnewline')` → `"say ""hi"", ok\nnewline"`（RFC 4180） |
| 截图存档 | docs/validation/workbench-screenshot-20261117.png（V4.1 baseline 运行态） |

说明：浏览器驱动为 ZCode 内置浏览器（Playwright surface），逐项真实点击/
选择/fetch 页面内执行；矩阵在途计数用页面内 fetch 包装器实测，非字符串检查。

## 7. 尚需后端字段或实测证据的事项（本任务不实现）

1. **聚合驻留容量**：当前协议无法从算子行安全聚合物理共享对象的组容量
   （weight_capacity 按行非可加）。已用「代表行容量 + 引用 M06」诚实展示；
   若后端未来在 row/组级提供「对象归属账本键」，透视可直接引用权威容量。
   **最小后端扩展建议**：roofline_points 行或 work 行增加 `memory_object_id`
   （指向 lowering.memory 对象键），前端即可组内去重求和并标注共享数。
2. **measured 预算的表达**：BudgetInput 契约已支持（kind=measured 需
   allocatable_source），但工作台数字输入无法携带可信来源与环境——UI
   有意不提供 measured 切换；待采集侧（G-D7）产出可追溯来源后再开专用
   输入（如从 profile/trace 包导入），不在本任务伪造。
3. **Roofline 峰值元数据**：并发后端任务已提供（本前端已消费并验证两种
   响应兼容）；cost_basis 中「dense_w 启发选档（乐观）」的披露依此后端
   任务口径，前端原样展示不加工。
4. **实测类缺口不变**：接受率（R07b）、CPU forward/GPU、真实可分配预算、
   DFlash served 标记、运行时池尺寸——unknown/conditional/unsupported
   全部保留传播，未为「全绿」取消任何门禁。

## 8. v0.1 退役审计与前置条件（只审计，不动共享模块）

依赖事实（当前代码核实）：

- `cost/t0.py`（v0.2 计价）`from .legacy import DeviceSpec, roofline_ms`；
- `cost/legacy.py`（72 行）`from .. import formulas as F`（根包 v0.1
  公式表，98 行）——即 **v0.2 → legacy → formulas 的存活链**；
- v0.1 公式表其余消费方：`engine.py`、`ledger.py`、`cli.py`（v0.1
  run/sweep/export 命令）、`webapp.py`（/legacy、/api/registry、
  /api/scenarios、/api/run、/api/sweep）；
- v0.1 路由仍有回归测试消费（test_consistency.py 经 `/api/run` 重放
  固定场景并核对 digest 快照）。

退役前置条件（建议顺序，均为后续任务）：

1. **共享能力迁移**：`roofline_ms`/`expected_unique`/DTYPE_ROUTE 迁入
   cost/t0.py 或独立模块；`DeviceSpec` 统一到 devices.py；v0.1 公式 id
   （F_ROOFLINE/F_LINK 等）在 formula_registry v2 有对应或显式映射。
   迁移判据：v0.2 全链路 import 不再经过 cost/legacy.py。
2. **历史回归资产保留方式**：v0.1 固定场景 digest 快照与 `/api/run`
   重放测试改为「归档一致性」性质——或将同一快照断言迁到 run-v2 通道
   后，v0.1 路由测试随路由一并退役（不删数据只删通道）。
3. **路由弃用条件**：/legacy 与 v0.1 API 在「工作台 v0.1 视图无访问
   （或挂弃用横幅一段时间）+ CLI v0.1 命令使用审计为零」后 410/移除；
   先弃写路由后弃读路由。
4. **移出主包验收标准**：`simverse.engine/ledger（v0.1 段）/formulas/
   cost/legacy` 可移入 `simverse/legacy_v01/`（或独立包）当且仅当：
   `py -3.12 -m pytest tests -q` 全绿（含 v0.2 全部门禁）+ run-v2 十六类
   导出逐字段不变（test_ui_v12 模式对拍）+ 旧 digest 快照有归档去处。
   当前**不因版本共存声称仍在双轨开发**——v0.1 仅作历史回归与归档视图。

## 9. 审查过程文件区分（docs/rebuild-20261106/）

逐文件引用审计（grep 全仓）：`dsh-events.jsonl`、`dsh-receipt.json`、
`zcode-events.jsonl`、`zcode-receipt.json`、`dsh/zcode-questions.md`、
`dsh-launch-path-failure.json`、`run-review.mjs`、`audit.py`、
`audit-results.json`、`decision-checks-20261109.json`、`review-packet.md`
均被 `manifest.json`（清单+哈希登记）与 `review-resolution.md`/
`review-ui-round2/resolution.md`/`validation.md`/`evidence-audit.md` 交叉
引用——**全部存在依赖，无可无伤归档者**。分类结论：

- 临时过程日志（receipts/events/questions）：保留原位（manifest 锚定）；
- 重放/审查脚本（run-review.mjs、audit.py）：保留（validation.md 引用）；
- 正式证据（audit-results.json、decision-checks、references/）：固定不动；
- 参考源码（references/simverse-v3-text.txt 等）：不动。

本次修复未移动/改写任何固定哈希证据资产；v41-swiglu-tail-20261113.md
仅追加编号同步括注（§4）。

## 10. 与并发任务的边界记录

- 工作树中发现另一独立任务（R27 成本完整性）的未提交改动
  （simverse/cost/t0.py、pipeline.py、families.py、
  tests/test_cost_integrity_fixes.py 等 +26 项测试）。本任务未读取/修改
  其实现文件；本任务的 Roofline 新元数据兼容验证恰好消费了其新增字段
  （只读 bundle JSON，无代码耦合）。
- **全量 pytest 终态（2026-11-17，含并发任务工作树改动）**：
  `1 failed, 514 passed`。失败项仅 `test_merge_gate.py::
  test_g6_status_doc_test_count_matches_reality`（见下）；本任务域内
  （文档路由/导出/CSV/组合事实/浏览器行为锚定）全部通过。中途曾出现的
  2 项 test_consistency 失败为旧断言固化「逗号→分号」行为，已随 RFC 4180
  修复同步（见 §5）。收集数 469→515（本任务 +20，并发任务 +26）。
- **G6（测试数门禁）预期内失败**：`docs/IMPLEMENTATION_STATUS.md` 的
  「pytest 收集数：469」与本任务禁改约束冲突。统一汇总时将该行更新为
  515（或分任务落账后重算）即可解除；本报告如实记录，未改该文件、
  未为他方任务“修绿”。

## 11. 复现

```bash
py -3.12 -m pytest tests/test_workbench_review.py tests/test_consistency.py -q
py -3.12 -m simverse.cli serve          # http://127.0.0.1:8765/ 浏览器复核 §6
py -3.12 -m simverse.cli run-v2 <场景> --csv ops.csv   # CSV RFC 4180
```

## 集成审查补记（2026-11-17）

后端新增 `cost.row_prices` 后，透视聚合和前端/CLI CSV 优先关联完整逐行计价摘要；旧 bundle 继续回退到 top-40。Qwen3.6 原先仅 4/27 类算子有 Roofline 点，V4.1 仅 3/51 类，故原报告“已完成七维透视”只覆盖少量算子；本次补齐。Roofline 保持 top-40 上下文，代表行可从逐行摘要补入，并使用该行自己的精度与峰值画上界。文档路由对所有后缀均以解析后的路径检查 docs 根。
