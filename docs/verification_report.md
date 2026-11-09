# 新架构方案可追溯验证报告 — 投机解码 / Draft-Target 分离

simverse v0.1.0 · 2026-11-06 · 对应《统一推理仿真仓库重构方案 v2.2》M0 基线 + M1 首链切片

**一句话结论**：以 DeepSeek V4.1 Flash（官方 config 快照）+ DSpark 链式草稿为固定场景，
本切片把 模型结构 → 逐算子整数账本 → T0 roofline 时延 → 四种部署的解析执行 全部打通并
版本化：手算 oracle 逐位一致、Web/CLI/导出三方一致、与两旧仓的每一处数值差异均已归类
（无"待解释"项），收益反转边界给出 α*/K*/B*/链路带宽 四组数字与图表。

---

## 0 · 交付物与验收对照

| 验收条款（用户） | 交付物 | 结果 |
|---|---|---|
| 手算样例与实现一致 | `golden/expectations.py`（独立推导，不 import 实现）vs `tests/test_oracle.py` | ✅ 9/9 通过：FLOPs/权重/KV/通信字节整数精确相等；roofline 浮点 rel≤1e-12 |
| Web、CLI、导出结果一致 | `tests/test_consistency.py`（Flask test client vs CLI 子进程 vs CSV） | ✅ 4/4 通过：digests/metrics/comparisons/逐算子整数全等 |
| 每个差异都有解释 | §4 两旧仓差异表 D-1…D-12 | ✅ 全部归类，无"待解释" |
| 实测/拟合/解析估算分别标明来源 | RunResult.provenance + 每个 OpRecord.formula_id + 公式注册表 | ✅ 首发全部数字标 **T0 公式估算**（决策 D-4：无实测/拟合数字） |
| 可重放场景 | `assets/scenarios/v41_flash_dt_verification.json`（digest sha256:196c24706cbc635a） | ✅ 同文件 → 同 digest → 同数字（测试钉死） |
| 逐算子明细 | RunResult.ops 共 2,493 行（基线 827 / prefill 819 / verify 827 / 草稿双放置 16），CSV 导出 | ✅ 每行绑定 层号/算子/次数/形状/公式 id |
| 两旧仓差异报告 | §4（工具 `tools/oldrepo_diff.py`，产物 `outputs/oldrepo_diff.json`） | ✅ 三通道：model-verse 分析器 / 前端 JS 数学 / pi-viewer |
| 收益边界图 | `outputs/boundary_*.png`（6 张，CLI 可复算）+ Web 边界扫描视图 | ✅ §5（注：像素级视觉验收被环境阻塞，见 §5 末） |
| 第一条贯通 Web 与 CLI 的实现 | `python -m simverse.cli run|sweep|serve` 与 `/api/run|/api/sweep` 同一 engine | ✅ §3 |

运行方式（干净环境）：

```bash
python -m pytest tests -q                     # 13 项验收测试
python -m simverse.cli run                    # 固定场景 → 摘要 + 可 -o/--csv 导出
python -m simverse.cli sweep                  # 边界扫描 + 6 张 PNG
python -m simverse.cli serve --port 8765      # Web 工作台（前端零物理计算）
python tools/oldrepo_diff.py                  # 两旧仓对账（需两旧仓在本机）
```

## 1 · 固定场景（冻结四要素）

| 要素 | 取值 | 来源 |
|---|---|---|
| 模型 | DeepSeek V4.1 Flash：40 层 MLA+CSA2（2 层纯 SWA + 38 层压缩 r=2），全层 MoE 384E top-6+1 shared，hidden 5120，vocab 129,280 | 官方 config.json 固定快照，SHA256 `8be45ce0…721879`（与 model-verse 仓 pinned 副本逐字节一致，`assets/config/`） |
| 精度 | 稠密 FP8 + 路由专家 FP4（官方 quantization；H100 无 FP4 GEMM → 算力仍走 FP8 档，旧仓 A6 口径）+ FP8 KV + FP8 激活；compressor/index 三投影按旧仓硬编码 FP16 权重 | config + 旧仓 `attention_structure.py` |
| 硬件 | 目标 8×H100 SXM（TP8，HBM 3.35 TB/s/卡，FP8 1979 TF）；草稿 DT 分离时 1×L40（864 GB/s）；链路 PCIe5 x16（64 GB/s + 0.10 ms 固定延迟/方向） | 公开规格（逐条 source 见 `devices.py`） |
| 负载 | B=1 单请求 decode，ctx=32,768（Agent-X 均值口径），K=4 草稿候选，α=0.75 链式逐位接受率；附带 prefill 视图（同 ctx） | 本报告定义 |
| 方法 | DSpark 链式草稿（S=K 步 × M=1；MoE 128E top-3，config 公式 4.53B；kvFrac 0.15） | 旧仓 `spec_gain_presets.py` + v2.2 §4.2 语义表 |

场景文件 `name=V41Flash-DT分离-固定验证场景`，`scenario_digest=sha256:196c24706cbc635a`、
`model_digest=sha256:aff4c85f07e14043`（含 config 快照摘要）。重放 = 同文件喂 CLI/Web/导出，
digest 不变即输入未变（`tests/test_consistency.py::test_fixed_scenario_replay_digest_stable`）。

## 2 · 方法：三段流水在本仓的落位（原理 v1.1）

| 段 | 属主 | 本切片实现 | 验证层 |
|---|---|---|---|
| Count（整数算术） | `ledger.py` | V4.1 Flash 逐层算子账本：MLA 投影 5 GEMM + SWA 写 + 压缩 KV 生产线（compress/pooling/索引 K）+ 索引打分/选择 + sparse_attention 两段读 + MoE 三行（router/routed×U(M)/shared）+ norms + lm_head/embed | 手算 oracle 精确相等 + 旧仓逐算子对账 100%（§4） |
| Cost（物理近似） | `cost.py`（唯一公式维护点） | roofline `t=max(FLOPs/peak·MFU, bytes/bw)+0.15·min`；MFU=0.75；launch 未逐算子计（eager 上界另行声明）；链路 `fixed+bytes/bw` | 公式注册表 `formulas.py` 逐 id 溯源；T1 校准接口预留（`CalibrationCoefficients`，透传 T0） |
| Execute（解析组合器） | `engine.py` | 四种部署闭式 TPOT + 盈亏平衡 α* 反解 + 代表轮时间线 | 手算 oracle + Web/CLI 一致 + 与旧 JS 口径对账（§4B） |

抽样/建模边界（对照 Vidur 方法）：pi-viewer 内 vendored Frontier（Vidur 系）的执行时延来自
**profiling 实测数据拟合的 sklearn 预测器**（`frontier/execution_time_predictor/sklearn_*.py`，
上溯 Vidur 对实测 kernel 时延分布插值采样、事件仿真中随机抽样）；本仓首发为**纯解析 T0**
（roofline 公式估算，无实测/拟合/分布采样），MoE 并集与投机产出只取期望，接受率 α 为外生
输入——这是与 Vidur 类仿真器的明确边界：我们可复算左侧阶梯（整数计数→期望计数→T0 公式），
右侧（T1 拟合→实测）接口已预留但首发不含任何数字。

## 3 · 首条贯通链路：Web 与 CLI 同入口

```
ScenarioSpec.json ──► engine.run() ──► RunResult ──► CLI 摘要/JSON/CSV
                        │                        └─► /api/run → 工作台（单页）
                        └─► sweep_suite() ──► 边界 JSON + PNG（CLI）/ SVG（Web）
```

- 工作台五个视图：部署对比（四部署 TPOT/加速比/公式串）、逐算子账本（阶段过滤 + 公式
  tooltip + 形状串）、时间线（DT 轮 Gantt）、假设与溯源（assumptions + 公式注册表）、边界
  扫描（6 组 SVG 曲线，数据全部来自 `/api/sweep`）。
- **前端零物理计算**（方案 §7 规则 2）：index.html 只做 RunResult 的投影与下载，无任何公式。
- 一致性测试用 Flask test client 与 CLI 子进程双通道对同一场景取结果，逐算子整数全等
  （`tests/test_consistency.py`）。浏览器实测（IAB）验证了 载入固定场景→运行→账本(verify
  828 行)→时间线(7 事件)→边界扫描(24 曲线) 全链路渲染，数字与 CLI 一致。

## 4 · 两旧仓差异报告（每个差异的归类）

工具 `tools/oldrepo_diff.py` → `outputs/oldrepo_diff.json`。三通道：
**A** model-verse `model_analyzer`（逐算子结构口径，spec_q=1/5 两次运行）；
**B** model-verse `spec_gain_model.js`（前端量级解析口径，node 直接运行原 JS——零移植风险）；
**C** pi-viewer `piviewer.roofline`（同源五类字节账本）。

### 4.1 通道 A：与旧分析器逐算子对账 —— 结构计数完全一致

基线 decode（B=1, ctx=32768, TP8, 每卡口径）：

| 算子组 | 旧仓 FLOPs | 新仓 FLOPs | 比率 | 旧仓权重读 | 新仓权重读 | 比率 |
|---|---|---|---|---|---|---|
| 注意力子图 26 类（q_a/kv/q_b/o_a/o_b、SWA/压缩/索引/选择、sparse_attention…） | 4.18112 G | 4.18112 G | **100.0000%** | 1.03614 GB | 1.03614 GB | **100.0000%** |
| MoE（旧聚合 moe_ffn 1 行 vs 新 router+routed+shared 3 行求和） | 2.63455 G | 2.63455 G | **100.0000%** | 0.70434 GB | 0.70434 GB | **100.0000%** |
| KV 读 | 39,858,816 B | 39,858,816 B | **100.0000%** | — | — | — |
| lm_head | 1.32383 G | 0.16548 G | 12.5% | 0.66191 GB | 0.08274 GB | 12.5% |
| norms/adds（4 行/层） | 3.28 M | 1.64 M | 50% | 0 | 0.41 MB | — |
| router 权重读 | 未计 | 78.6 MB | — | — | — | — |
| embed 查表 | 未单列 | 激活查表 + 0.66 GB 驻留 | — | — | — | — |

验证步（spec_q=5，M=B·(K+1)=5）同样 **100.0000%**：注意力 FLOPs 20.906 G、MoE FLOPs
13.173 G、MoE 权重读 2.733 GB（U(30 draws)=38.9/384 专家并集放大，两仓同式）、KV 读
39.911 MB（共享一次读 + K+1 新行）逐项相等。**逐条归类：**

| # | 差异 | 归类 | 解释与依据 |
|---|---|---|---|
| D-1 | lm_head 旧仓未按 TP 切分（h×v 全量计在每卡） | **有据纠正旧值** | lm_head 列切分 per card = h×v/TP8；旧仓 `configs/DeepSeek_V41_Flash.py post_process` 无 /tp。FLOPs 同因（差 8×）。注意旧仓 totals 只计 1 次（不 ×40），本工具按同口径还原 |
| D-2 | norms 7 OP/elem（旧） vs 4D+2（新） | 口径差异 | 旧标准路径 norm 行约定 7×width；新仓移植 `attention_structure.norm()`（4D+2，rsqrt=1）。仅影响元素算子 OPs，~1.6 M/步，decode 非瓶颈项；两式都注明 |
| D-3 | attn_add/mlp_add OPs 旧 5120/层 新 0（仅计激活字节） | 口径差异 | 旧把残差加计 1 OP/elem；新按访存行计。影响同 D-2 量级 |
| D-4 | DSpark 草稿：旧预设 3 阶段块近似，新按 v2.2 §4.2 链式 S=K | 口径差异（语义表项） | v2.2 三参数契约 (S_draft, M_draft, K)：链式 (K,1,K)。旧仓 `spec_gain_presets` stages=3/tokensPerStage=2 为块近似；M0 语义表已列"待用户确认"项 |
| D-5 | 草稿 KV 读 = 目标单步 KV 读 × kvFrac | 新仓简化（近似标注） | 沿用旧仓 "ratio" 口径（0.15/参数比）；草稿自身 KV 语义（独立每 token 字节）列 M3 SpecMethod 细化项 |
| D-6 | MoE 并集：两仓同用 i.i.d. 式 `E·(1−(1−1/E)^M)`，M=T·top_k | 一致（非差异） | 锚 `piviewer/roofline.py:141 _expected_unique` = 旧 analyzer `_moe_coverage` 反解，数学同式；组合精确式偏差在 E≫M 时 <1%（方案 §4.3 注） |

### 4.2 通道 B：与旧前端解析口径（原 JS 运行）对账

同一场景映射（tp=8、无分层缓存、tileM=1、commMs=0、bytesPerParam=0.75 混精近似）后：

| 量 | 旧 JS（原文件运行） | 新仓 | 归类 |
|---|---|---|---|
| τ = E[n]（α=0.75, γ=4） | 3.05078125 | 3.05078 | **精确一致**（同式 F_EN_CHAIN） |
| D→T / T→D 字节 | 30 B / 256 B | 30 B / 256 B | **精确一致**（锚 dt_sim.py:185 `batch*(gamma+1)*6`） |
| 双向链路时延 | 0.2000 ms | 0.2000 ms | **精确一致**（fixed 2×0.1 + ~0 字节） |
| t_base | 0.449 ms | 0.583 ms | 口径差异 D-7 |
| colocated 加速比 | 1.562 | 1.324 | 口径差异 D-7/D-8 叠加 |
| dt_serial 加速比 | 0.109 | 0.909 | **有据纠正旧值** D-9 |
| colocated 盈亏平衡 α* | 0.505 | 0.597 | 口径差异传导 |

| # | 差异 | 归类 | 解释 |
|---|---|---|---|
| D-7 | t_base/t_verify 用 16B 激活参数 × 0.75 B/param 聚合（旧 JS）vs 逐层结构计数（新） | 口径差异（合法不可直对） | 新仓含 embed+lm_head 驻留读取（1.32 GB）、注意力 dense 1.04 GB、共享专家 1.41 GB 与并集路由 4.19 GB（整组口径）；旧 JS 为"激活参数×带宽"量级口径（其源码自注"两套口径数字不互换"）。结构性口径与通道 A 100% 对账，故保留新值 |
| D-8 | 注意力 FLOPs 旧 JS 用 `4·dAttn(8192)·1.2 树系数` 聚合 vs 新按 SWA/top-k 结构对数 | 口径差异 | CSA2 下 attention 对数受 topk=512/swa=128 封顶，与全注意力聚合式差异大；新口径与旧分析器逐位一致（§4.1） |
| D-9 | 草稿权重读：旧 JS 按 dense 全量 4.53B×0.75 B ≈ 3.4 GB/阶段 ×3；新按 MoE 路由并集 U(3 draws)=2.98 专家 ≈ 105 MB/步 | **有据纠正旧值** | B=1 时仅命中专家权重被读（物理事实；与主模型 decode 同理由）。旧口径令 DT 分离草稿时延虚高 ~22×（11.83 ms vs 0.556 ms），导致 dt_serial 加速比 0.109 vs 0.909。注：大 batch 下并集趋于全覆盖，差距收窄——边界扫描已覆盖该维度 |
| D-10 | 旧 JS 无 Catch-up（E[M_draft,0]）项 | 有据纠正旧值（新仓补计） | v2.2 §4.5 增量②：全接受轮（P_full=α^K）下一轮草稿首步需喂 2 token；新仓解析加权 `t₀=(1−α^K)t(M=1)+α^K·t(M=2)`，E[M₀]=1.316（本场景影响 ~2.6% 草稿时延，小但可追溯） |
| D-11 | 旧 JS MoE 扩散 `1+0.08·log2(B(1+0.35γ))` 经验式 | 口径差异（已被 D-6 替代） | 新仓用超几何并集 U(M)（与两旧仓结构分析器同式）；经验式无出处，v2.2 G-P1-1 已裁定弃用 |

### 4.3 通道 C：pi-viewer 支持度

`piviewer.roofline.RooflineAnalyzer` 对 V4.1 Flash config：按 DeepseekV3 架构路径**可构造
但语义不适用**（无 CED/CSA2/压缩层计划字段；静默按 V3 模板解释 40 层 MLA）。归类：
**不支持（口径不适用）**——不作为对账锚点；此"静默接受错误结构"正是重建计划要消灭的问题
类别（层图唯一入口 mv_model_structure 0.2.0 亦未注册 V4.1 Flash，其支持边界已在
`models.py::MV_MODEL_STRUCTURE_SUPPORT` 声明；M2 接回共享包时补该架构适配）。
同源公式核对：`_expected_unique`（roofline.py:141）与新仓 `expected_unique` 逐式一致（D-6）。

### 4.4 差异总结

- **精确一致**：注意力 26 类算子、MoE 计数、KV 读写（基线与验证步全部 100.0000%）；τ/E[n]、
  DT 通信字节与链路时延（与旧 JS/dt_sim 锚点）。
- **有据纠正旧值**：D-1（lm_head TP 切分）、D-9（草稿 MoE 稀疏读）、D-10（Catch-up 补计）。
- **口径差异（合法不可直对）**：D-2/D-3（norm/add 计数约定）、D-4（草稿链式 vs 块近似）、
  D-5（草稿 KV 折算）、D-7/D-8（聚合 vs 结构）、D-11（经验扩散式弃用）。
- **待解释**：无。凡旧仓数字未采用处，均有结构口径 100% 对账或明确依据支撑。

## 5 · 收益边界（反转点与图）

固定场景下的扫描（`outputs/sweep.json` + `boundary_*.png`；每点=独立 run，同一 L1 计价）：

| 维度 | 反转点 / 最优点 | 解读 |
|---|---|---|
| 接受率 α（0.30–0.95） | **α\*：colocated 0.597 · dt_pipeline_ideal 0.581 · dt_serial 0.802** | 同卡投机在 α≥0.60 才盈利；DT 串行要求 α≥0.80（弱草稿设备+通信暴露），理想流水把门槛拉回 0.58（草稿被 verify 掩盖） |
| 草稿候选数 K（1–8） | **K\*=2（colocated 1.374）/ K=3（pipeline 1.385）** | DSpark 草稿成本 ∝K 线性而 E[n] 饱和（α=0.75 下 K=4 的 E[n]=3.05 vs K=2 的 2.31）；**最优 K 随 α 上升而后移**（α=0.9 时 K\*≈4–5） |
| 并发 B（1–64，对数） | **B\*≈2.4 同卡投机开始亏损（K=4, α=0.75）** | MoE 并集放大：verify M=B(K+1) 令路由专家权重读随 B 超线性增长（B=8 时 U(240 draws)=180/384），摊薄收益反转——MoE 模型投机解码的已知现象，本仓给出该配置下的量化边界 |
| 上下文 ctx（2k–131k，对数） | 无反转（四部署全盈利区间内缓慢变化） | CSA2 稀疏 KV 读受 topk=512 封顶（39.9 MB/步 @32k），ctx 增大主要抬 prefill/KV 容量而非 decode 步时延——与全注意力模型定性不同 |
| DT 链路带宽（2–2048 GB/s） | **无反转**：带宽不敏感 | 链式草稿 token-id 协议每轮仅 30 B（D→T）+ 256 B（T→D），0.2 ms 固定延迟主导、带宽项 ~µs 级——**该协议下 DT 分离的互联要求是延迟而非带宽**；若换 MTP 分离（hidden 回传 ~10 KB/接受 token）或大 B，带宽项才进入同一量级 |
| 草稿设备带宽（0.27–8 TB/s） | dt_serial 对草稿带宽近线性 | L40(0.864 TB/s) 上草稿 0.556 ms ≈ 目标基线 96%；草稿提到 H100(3.35) 降到 0.15 ms —— DT 串行收益主要取决于草稿设备档次而非链路 |

主场景结论（B=1, ctx=32k, K=4, α=0.75, 8×H100 + L40 + PCIe5）：

| 部署 | TPOT ms/tok | 加速比 |
|---|---|---|
| baseline 普通 decode | 0.583 | 1.000 |
| colocated 同卡投机 | 0.440 | **1.324** |
| dt_serial DT 分离串行 | 0.641 | 0.909（亏损；α≥0.802 才赢） |
| dt_pipeline_ideal 理想流水下界 | 0.426 | 1.368（乐观标注：无命中率/回退建模） |

### 5.1 视觉验收状态（如实声明）

本会话的模型通道（含 visual-judge 子代理）均无图像输入能力，像素级视觉验收**未能执行**——
不伪造结论。已做的替代核验（程序化、可复算）：6 张 PNG 均为有效 900×540 RGBA；四条系列
曲线（baseline/colocated/dt_serial/dt_pipeline_ideal 四色）每条全宽渲染（2,400–3,300 px）；
speedup=1 灰色参考线被平坦的蓝色基线（≡1.0）完全覆盖属预期；标题/轴标签/图例文本像素
存在（1,403 深色 px @boundary_alpha）。Web 工作台经浏览器（IAB）实测：固定场景载入→运行→
部署对比（数字与 CLI 一致）→逐算子账本（verify 828 行）→时间线（7 事件 Gantt）→边界扫描
（24 条 SVG 曲线）全链路可渲染。视觉观感验收留待有视觉能力的通道复核。

## 6 · 手算 oracle（独立期望 vs 实现）

`golden/expectations.py` 全部由人工算式书写（模型 2 层 GQA d=256）：基线 decode 总 FLOPs
2,959,256、权重读 2,742,272 B、KV 读 51,200 B / 写 512 B；验证步（M=3）attention 对数 306、
GEMM ×3、KV 读共享一次；E[n]=1.75、P_full=0.25、E[M₀]=1.25；通信 18 B / 256 B；草稿 M=1/M=2
双账本（权重读不随 M 放大=摊薄语义）；q_proj 行 roofline 浮点逐位。实现输出与上述
**全部精确相等**（`tests/test_oracle.py` 9 项）。

## 7 · 假设与简化（全部随 RunResult 输出）

1. 解析档适用域：单请求/固定批形状、无请求间干扰（事件状态机与 serving 属 M3/M4）。
2. 全部时延为 T0 roofline 公式估算（MFU 0.75、15% 重叠尾部、聚合 HBM 带宽）；无实测/拟合。
3. launch 未逐算子计入；CUDA Graph 未建模（三态之 unsupported/eager 口径）。
4. 链式草稿 S=K、M=1，Catch-up 步按 P_full 加权；committed_hist 事件采样属 M3。
5. 草稿 KV 读按 kv_frac 折算（旧仓 ratio 口径）；草稿仅计专家参数（attention/embed 未展开，
   DSpark 标 approximated）。
6. 通信为整批单消息/方向，无链路竞争排队；dt_pipeline_ideal 为乐观下界。
7. MoE 并集 i.i.d. 近似；α 外生；TP 通信（allreduce）未计入算子账本（v0.1 以分片形状计，
   通信原语计价属 L1 后续项——与旧仓 analyzer 同边界）。
8. 权重驻留 36.0 GB/卡 + KV 保留 0.06 GB/卡（CSA2 容量公式移植）→ H100 80 GB 充裕，无 OOM。

## 8 · 下一步（按 v2.2 里程碑）

- M1 补全：prefill/decode 工作台下钻到单层展开（当前账本已逐层、视图待做下钻交互）；
  chrome-trace 事件导出。
- M2：V4.1 Flash 架构进入 mv_model_structure 共享包（消灭本仓临时适配器）；四模型 16 组
  跨仓对账；TP 通信原语入账本。
- M3：事件核（dt_sim 语义重写、时长改 L1 供给）、dt_pipe_speculative（JetSpec++ 命中率建模）、
  committed_hist 采样。

## 附 · 文件清单

```
simverse/            contracts.py formulas.py models.py devices.py cost.py
                     acceptance.py ledger.py engine.py sweep.py chartpng.py
                     cli.py webapp.py static/index.html
assets/              config/deepseek_v41_config.json（SHA256 8be45ce0…）
                     scenarios/v41_flash_dt_verification.json tiny_oracle.json
golden/              expectations.py（独立手算期望）
tests/               test_oracle.py test_consistency.py（13 项）
tools/               oldrepo_diff.py（两旧仓对账）
outputs/             run_v41.json ops_v41.csv sweep.json boundary_*.png ×6
                     oldrepo_diff.json web_workbench_sweep.png
```
