# 独立质询报告：统一推理仿真重构方案（ZCode，2026-11-06）

**结论先行**：重构方向（Count→Cost→Execute + 单一公式维护点 + 结果契约）与 v0.1 切片的工程纪律是扎实的，但当前存在三个 P0 级名实不符：① 注册为 "DSpark" 的预设与官方 DeepSpec 算法完全不是一回事（官方是块并行 + 每轮必须回传多层 target hidden + 置信度截断），② MoE 并集公式在代码注释、报告数字（38.9 系誊写错误，实为 28.89）、“精确组合式”标注上三处互相矛盾，③ `dt_pipeline_ideal` 在 B=1 场景与同轮并发 Gantt 中以可实现部署的姿态呈现。主控的八条判断大方向正确，但有两处批评打偏了病灶、一处粒度过度（见 §2）。

---

## 1. 必须追问主控的问题（P0×4 / P1×5 / P2×3）

### P0-1 DSpark 预设名实不符：注册表语义、hidden 依赖、T→D 字节全部错误

| 项 | 内容 |
|---|---|
| 被挑战假设 | `models.py:161-175` dspark 预设 =（S=K, M=1 链式）、`t2d_bytes_per_token=0`（仅 256 B 控制）；v3 HTML §4.2 SpecMethod 表 dspark 行 “Hidden 依赖：无（仅需 token ID）” |
| 最小反例/锚点 | ① `references/DeepSpec/deepspec/eval/dspark/evaluator.py:139-147`：`_update` 每轮把 **accepted+1 个 token 的多层 target hidden**（`target_layer_ids`，fc 输入维度 \|layers\|×hidden）写入草稿上下文——T→D hidden 是必传项，绝非 0 字节；② `draft_ops.py:22-45`：`forward_dspark_draft_block` 是**单次块前向**（S=1、M=block_size、`is_causal=False`、K/V 由 target hidden 现场投影），非 S=K 链式；③ `_confident_prefix_length`（draft_ops.py:82-93）按置信度截断 → **K 每轮自适应**，E[n] 不服从链式几何；④ `references/vllm/vllm/v1/spec_decode/llm_base_proposer.py:103-117`：`hidden_size × hc_mult` + `pass_hidden_states_to_model` 佐证 hidden 必传 |
| 解决方式/验收 | 立即把该预设改名 `chain_moe_draft(approx)`；`dspark` 在按官方语义实现前返回 `unsupported_method`（与 dflash 同等待遇）；SpecMethod 八字段表按 DeepSpec 逐字段重写（S=1 / M=block / K=conf-prefix / hidden=required / T→D=n×\|layers\|×d×2B）。验收：固定场景不再以 "DSpark" 名义输出数字；报告 §5 “链路带宽不敏感” 结论限定为 “token-id 链式协议”，并对 DSpark 重算（hidden 回传 3 tok×~2 层×5120×2B ≈ 60 KB/轮，带宽项进入与 0.1ms 同量级） |

### P0-2 MoE 并集：三种抽样口径混写 + 期望值 int() 进整数账本 + 报告数字错

| 项 | 内容 |
|---|---|
| 被挑战假设 | ① 报告 §4.1 “U(30 draws)=38.9/384”；② `formulas.py:32` 称 “精确组合式 E·(1−C(E−k,M)/C(E,M))”；③ `ledger.py:311,398` `weight_read=int(union×…)` 把期望写进整数列 |
| 最小反例/锚点 | 我用纯算术复算（未运行仓代码）：`outputs/run_v41.json` 实际输出 **28.89**（≤30 合法），38.9 是报告誊写错误；E=384,k=6,M=5 时三式各不相同——有放回 28.89、逐 token 无放回 E·(1−(1−k/E)^M)=29.08、“组合式” 29.22。**组合式对应的是“整步全局无放回抽 M 个专家”这第三种抽样方案，根本不是 per-token top-k 的精确式**——注释把三种方案搅在一起。另外均匀 i.i.d. 路由本身未经任何真实路由分布验证（skew 会让并集偏离） |
| 解决方式/验收 | 公式注册表为三种式各立 id 并写明抽样语义与适用条件；OpRecord 整数列只放确定计数，期望并集放独立浮点字段 + 口径标签；报告勘误 38.9→28.89。验收：golden 新增 E384/k6/M5 三式断言；schema 变更后 13 项测试全绿 |

### P0-3 `dt_pipeline_ideal` 的呈现方式：B=1 主场景里它不是可部署项

| 项 | 内容 |
|---|---|
| 被挑战假设 | 四部署并列进 comparisons/sweep（`sweep.py:107`），Gantt 把**同轮** verify 与 draft 都从 t=0 并发画（`engine.py:329-347`），并给出 α*=0.581 |
| 最小反例/锚点 | B=1 单请求没有第二个微批可交错；同轮 verify 依赖本轮草稿输出，画成并发违反依赖（真正重叠的是**上一轮** verify）。v3 HTML 自己的 G-P0-3 也承认需要 B≥2 微批或 β 命中率的 `dt_pipe_speculative`——即该公式的合法域不含主场景 |
| 解决方式/验收 | comparisons 中标注 bound-only 或移入独立 “理论下界” 组；timeline 事件加 `round_id`，流水线模式用阴影轨画 “上一轮 verify”；α* 表分 “可实现/上界” 两组。验收：UI 上该行样式区分 + 每个时间线事件带 round_id 字段 |

### P0-4 V4.1 Flash 的结构证据闸门：对账一致 ≠ 语义正确

| 项 | 内容 |
|---|---|
| 被挑战假设 | 报告以 “通道 A 逐算子 100% 对账” 支撑 V4.1 计数可信；“模型结构→账本→执行全部打通” 的表述 |
| 最小反例/锚点 | 通道 A 的对账对象（旧仓 `attention_structure`）与新仓**同源同假设**——rebuild plan 自己写明 “即使两仓的计数一致，也可能继承同一个错误假设”；`models.py:135-140` 承认 mv_model_structure 0.2.0 不支持 V4.1；`ledger.py:228` 压缩 pooling 自注 “估计（非实测）”。V4.1 是未发布模型，无官方 forward 可核，全部 CSA2 语义是团队对 config 的解释 |
| 解决方式/验收 | RunResult 增加 `structure_status: audited / interpreted_from_config / unknown`，V4.1 默认 interpreted 且 UI 降级显示；压缩 pooling、索引选择、候选发布三处打 “未经源码核对” 标签。验收：固定场景 JSON 可见该字段；报告措辞从 “全部打通” 改为 “计数管线打通，语义为解释性” |

### P1-5 CSA2 验证步 KV 读：三种口径三个数， η_sparse 无一有证据
- **假设**：simverse verify 的 sparse KV 读 = 基线单步（`ledger.py:273` `read_comp=min(end//r, topk)` 与 Q 无关，即完美去重 η=1）；HTML 定 η=1+0.15K 且自称“第一性公式”（λ_div=0.15 无出处）；旧 JS 用 (1+γ/ctx)≈1。
- **反例**：per-query top-k 检索的块并集 > topk（除非索引完全相同）；dense/MLA 的“前缀只读一次”只在 M ≤ kernel query-tile 时成立，M=B(K+1) 增大后破缺。三口径在 K=4 时相差 0–60%。
- **解决/验收**：η_sparse 进契约做显式参数（默认值标 unknown，给区间而非点值）；assumptions 列表补此条。验收：场景可切换口径且结果中显示来源与取值。

### P1-6 显存账本不闭合：colocated 模式漏草稿权重，workspace/activation 全缺
- **锚点**：`engine.py:158-162` `oom_target` 只查 target 权重 + target KV；colocated 下草稿 ~4.5 GB（4.53B×1B）未入同卡合计；activation/workspace/CUDA Graph pool/lifetime 无任何行。报告 §7.8 只有 36.0+0.06 GB。
- **解决/验收**：M2 容量账本按主控 #1 的五类口径补齐，至少先做 (a) colocated 同卡合并判定 (b) verify 步激活峰值一行。验收：固定场景输出合并峰值分项，oom 判定含草稿。

### P1-7 launch 开销从未计价，而草稿步可能 launch 主导
- **锚点**：`cost.launch_ms` 存在但 `engine.py`/`sweep.py` 无一处调用（grep 证实，仅 assumptions 文字提及）；dense 草稿一步 ~10² 量级 kernel × 5 µs ≈ 1 ms/step，**与甚至大于** roofline 值（1.8 GB/3.35 TB/s ≈ 0.54 ms）。K*、α*、B* 全部数字在此系统性上偏下得出。
- **解决/验收**：每个部署行输出 launch 上下界（kernel 数估计 × launch_ms）作为伴随区间，或按 CUDA Graph 三态口径标注 hit/unsupported；边界图叠加该界。验收：colocated 行含 launch_bound_ms 字段。

### P1-8 committed_mean 输入时 P_full/catch-up 静默继承截断几何假设
- **锚点**：`acceptance.py:12-25` `resolve_alpha` 反解 α；`engine._draft_step_costs` 用它算 P_full 加权 catch-up；RunResult 无任何 assumption 标注（HTML 承诺的 `assumption: truncated_geometric` 只写了事件模式）。主控的 K2 反例（N 恒 2 与 N∈{1,3} 各半，同均值 E[n]=2 但 P_full 0.38 vs 0.5）说明该假设直接改变草稿成本。
- **解决/验收**：metrics 增加 `acceptance_assumption` 字段；K2 反例写成测试用例进 M3 事件核验收清单；α*/K* 结论旁统一标 "under i.i.d. chain acceptance"。

### P1-9 报告对 Vidur/Frontier 的描述与参考代码不符
- **锚点**：`references/vidur/.../sklearn_execution_time_predictor.py:476-576` 全部 `target_col=*.median` + grid search + pickle 模型/预测缓存；本副本无分布插值/随机采样代码。报告 §2 “上溯 Vidur 对实测 kernel 时延分布插值采样、事件仿真中随机抽样” 的一半主张无据（上游 Vidur 是否有 histogram 预测器我未验证，但引证应锚到本地副本行为）。
- **解决/验收**：该段改为按 vendored 副本实际行为描述（median 回归），或补充证据文件路径；pi-viewer 仓 Frontier 侧未经本次审阅的事实写明。

### P2-10 pinned 原理文档的勘误治理
- **锚点**：我复算 pinned 文档 §1 MoE 例：按其自身公式（指数=token×k=1000）覆盖率 **85.84%**，按仅 token 数（100）为 17.76%——文中 “约 18%” 与其公式自相矛盾（差 4.8×）；§3 “物理校准：三点定解” 实为对自身仿真输出的代理拟合。该文档将被当 M0 冻结契约引用。
- **解决/验收**：v3 原理文档逐条标注继承/勘误（18%→86% 必须勘误），引用处附 diff；三点拟合改名 “稳态代理拟合”。

### P2-11 结果契约的证据维度挂哪一层（主控 #5 的落地粒度）
- **反例**：账本 2,493 行 × 7 维（support/evidence/feasibility/scope/单位/公式/输入依赖）会指数膨胀且大部分重复。
- **建议/验收**：维度挂 formula_id 与 stage 级 + RunResult 级汇总表（每类量一行），OpRecord 只引用 id；`scope=per_card` 已有，补单位枚举。验收：固定场景的 provenance 可按 stage 聚合导出。

### P2-12 UI 联动原型的数据诚实与验收真空
- **锚点**：报告 §5.1 自认像素级视觉验收未执行；HTML Tab4 的 KPI（9.51 ms、2.68× 等）是无来源示例数字；sweep 一次 24 条曲线 × 200 点的联动渲染性能未测。
- **解决/验收**：示例数据全站 "sample" 徽标 + 来源字段；四联动先做单屏性能与像素验收（有视觉能力通道），矩阵联动默认聚合。验收：IAB 回放脚本进 CI 冒烟。

---

## 2. 主控判断中可能错误或过度的部分（4 处，均有证据）

1. **主控疑点 “E384,k6,M5 并集必不超 30、report 写 38.9 可疑” 打偏了病灶**。代码实际输出就是 28.89（`run_v41.json`），≤30 完全合法；38.9 是**报告誊写错误**，不是公式产出的数。且主控给的 “正确式” E·(1−(1−k/E)^M) 在本场景与现行有放回式只差 0.7%（29.08 vs 28.89）——真正要修的是 `formulas.py:32` 把第三种抽样方案的组合式误标为 “精确组合式”，以及 int(期望) 入整数列。按主控原话修会修错地方。
2. **主控判断 #3 后半句 “微批稳态周期≠单请求 TPOT、不能把 max(D,V) 当同轮时间线” 对已实现部分过度概括**。对固定 K 的链式草稿，B 个请求每轮 verify 形状恒为 B(K+1)，lockstep 周期 / E[n] 作为每请求平均 TPOT 是合法模型；代码在 formula_id、comparisons、assumptions 三处已带 “乐观标注”。批评应收窄到它真正成立的三处：B=1 无微批可交错、变 K 方法（DSpark 置信截断）、以及 Gantt 把**同轮**画并发（见 P0-3）。
3. **主控 “应撤销 DSpark 真实性声称” 部分已做但指向不对**。`models.py:172-174` 已有 `status="approximated"` 与 “真实 DSpark 为半自归块结构” 披露——缺的不是披露，是**命名使用权**：以 DSpark 之名发布的数字（含 §5 链路结论）与 HTML 语义表行仍然错误（P0-1 的 hidden 必传反例）。
4. **主控判断 #5 的粒度可能过度**：逐结果 7 维与 2,493 行账本相乘是设计负担；建议维度挂公式/阶段级（P2-11），否则契约会重到没人填。

---

## 3. 建议的计划顺序、原理契约与 UI 防护

**计划顺序调整**（不动 M0–M4 骨架，改内容）：
- **M0 追加**：勘误表（38.9、18%）；DSpark 改名/unsupported 决定；MoE 三式口径契约；`structure_status` 字段设计；P0-1/P0-2 修复（小改动，不该等到 M2）。
- **M1**：唯一具备完整真实语义的方法是 **custom chain draft + custom target**（与用户 Q2 选 C 一致：自定义工作台）；V4.1 以 interpreted 降级显示；tiny oracle 保留为独立锚点。
- **M2**：才允许实现 MTP/DSpark 官方语义（对齐 DeepSpec/vLLM 逐字段）；per-proposer 双向消息表；容量账本补齐（P1-6）；η_sparse 参数化（P1-5）。
- **M3**：事件核带 round/request id、committed_hist、launch 计价（P1-7/8）。
- **M4**：serving + T1 校准（先修正 Vidur 表述，P1-9）。

**原理契约修改**：SpecMethod 表每行增加 hidden 依赖与 T→D 消息公式两列（现表有错行）；MoE 并集三式分立；catchup 语义按 proposer 分立（链式带 KV 草稿 M₀=1+α^K 成立、MTP 吃 hidden、ngram M=1 恒定、DSpark 不适用）；内存五类口径入契约。

**UI 误导防护**：理论下界部署（dt_pipeline_ideal）用条带样式并从 α* 主表分离；每个收益数字旁挂 assumption chips（i.i.d. chain / η_sparse 取值 / launch 未计）；示例数据 "sample" 徽标；不可用原因显式（现有 unsupported_method 状态是好的，扩展到 “结构解释性” 降级）；时间线事件带 round_id/request_id。

**确需留给用户的决策**（其余不应上抛）：① Q4 可信度档位与可获得的实测参照（决定 T1 范围）；② 首发 demo 是否必须在 UI 上叫 “DSpark”（语义诚实 vs 命名需求）；③ 一屏四联动的屏幕预算 vs 可折叠面板取舍；④ M2 是否引入昇腾（人力/数据可得性）。

---

## 4. 实际阅读/运行 vs 推断

| 类别 | 内容 |
|---|---|
| 全文阅读 | verification_report.md、model-verse-principles-pinned.md、simverse-v3-text.txt、simulator-rebuild-plan.md、simulator-rebuild-questions.md、review-packet.md；simverse 的 formulas/engine/ledger/acceptance/models/cost/devices.py；tests/test_oracle.py、test_consistency.py；golden/expectations.py（前 80 行）；DeepSpec draft_ops.py、qwen3/modeling.py、evaluator.py |
| 部分阅读/检索 | contracts.py（前 120 行 + AcceptanceSpec 段）、sweep.py（结构 grep）、outputs/run_v41.json（MoE shape_note）、vidur sklearn_execution_time_predictor.py（median/cache/类结构 grep）、vllm llm_base_proposer.py（hidden/hc_mult grep） |
| 实际运行 | 仅 `python -B` 纯算术复算（并集三式、pinned 18%/86%、DSpark M=1/2 并集）。**未运行** pytest（避免写缓存、保持只读）、未运行仓内任何代码、未运行真机/模型。"13 passed" 采信主控陈述 |
| 仅推断未验证 | sweep.py/webapp.py/index.html/oldrepo_diff.py 的实现细节按报告与签名推断；pi-viewer、model-verse 两仓本体未读（仅 docs 两文件）；vLLM dspark.py 模型文件、DeepSpec common/markov_head.py 未读全；“H100 无 FP4 GEMM” 等硬件口径未外查；上游 Vidur 仓库是否含 histogram 采样器未验证 |

本次为只读质询，未修改任何文件、未创建外部资源、未调用其他 Agent；除本报告文本外无交付文件。

```json
{"version":1,"nonce":"9a3e6d69-59b3-41e5-ae09-d1976a6e8772","summary":"独立质询完成：4个P0（DSpark名实不符+hidden必传、MoE并集三式混写与报告38.9誊写错、dt_pipeline_ideal呈现、V4.1结构证据闸门）、5个P1、3个P2追问；指出主控判断4处偏差；给出计划/契约/UI修改建议；只读审查无文件产物","files":[]}
```