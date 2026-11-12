# 两主模型 forward 证据与合入门禁审查（2026-11-10）

## 结论

本轮把运行时目录收敛为 **DeepSeek-V4.1-Flash + Qwen3.6-35B-A3B + 与 Qwen3.6
绑定的 DFlash DraftIR**，修复了审计确认且仍影响两模型的计数/容量错误，并把
“config 字段映射 / forward 源码阅读 / CPU 执行 / GPU 实测”拆成独立证据栏。
两款主模型目前都**不能**宣称 forward 完备或进入实际部署排名：V4.1 仍有
Engram/CSA2/MoE/量化缺口；Qwen3.6 主干的已核分支保持 source-checked，但 MTP
只有 config + 目标首层模板。历史模型与 H200 校准资产保留为**离线归档**，不再
出现在 API/CLI/工作台选择中。

DSH 实施阶段没有执行 CPU/GPU forward，没有调用其他模型 CLI 或子 Agent，
没有推送、发布或配置 GitHub 远端保护。外层 Codex 验收时另做了下述小尺寸
CPU shape probe；它不覆盖真实 checkpoint、MTP、V4.1 或部署实测。

## 1. 来源、版本与取舍

| 对象 | 固定来源/版本 | 本仓证据 |
|---|---|---|
| DeepSeek-V4.1-Flash config | `deepseek-ai/DeepSeek-V4.1-Flash` @`314113f8a7dadb388215396ba5fbee2c610a04d6`；config SHA256 `8be45ce0…` | `assets/config/deepseek_v41_config.json`、`ev-v41-config` |
| V4.1 forward | vLLM @`f358f0578a83f4e1fe6fc49a492d3ae9362de8e5` `models/deepseek_v41/nvidia/model.py` | `ev-vllm-v41-model`、`sm-v41-*` |
| Qwen3.6-35B-A3B config | `Qwen/Qwen3.6-35B-A3B` @`345d467b117d1fd5eda9b879a5988de9ae5f47a7`；config SHA256 `93a4693f…` | `assets/config/qwen36_35b_a3b_main__config.json`、`ev-qwen36-main-config` |
| Qwen3.6 forward | HF transformers @`b6401d6605d16620e0f95b54fb2906f92de7e33f` `modeling_qwen3_5_moe.py` / `configuration_qwen3_5_moe.py` | `ev-hf-qwen35-modeling`、`ev-hf-qwen35-config` |
| Qwen3.6/DFlash 并行配方 | vLLM @`dc74b8cbbbe3dbb9399fe369757d63defd614cc2`（16 文件逐 SHA256） | `assets/identity/vllm_qwen35_pin/manifest.json`、`ev-vllm-parallel-recipe` |
| DFlash checkpoint | `z-lab/Qwen3.6-35B-A3B-DFlash` @`f181eece646affea2c38b2765f1aaa01a9734ccd`；config SHA256 `20784b06…` | `assets/config/qwen36_35b_a3b_dflash__config.json` |
| DFlash 方法 | `z-lab/dflash` @`07ebd93db9f472af339b644bb70221ad8428328a` | `ev-dflash-impl`、`ev-dflash-readme` |
| H200 校准 | pi-viewer @`c5909a7699a1d2bb422d931bdca6cba399451f00`（历史归档） | `assets/profiles/*/manifest.json`、`ev-h200-*-distill` |

GitHub 机制只借鉴**规则**，未引入运行时依赖：`pull_request`/`push`/`merge_group`
三事件触发、job 作为 required status check、PR review 由仓库 ruleset 启用。
官方文档（2026-11-10 查阅；仅文档，未在远端配置）：
[protected branches](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches)、
[branch protection rule](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/managing-a-branch-protection-rule)、
[rulesets](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/about-rulesets)、
[merge_group event](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#merge_group)。
不引入 GitHub API 客户端/自动保护脚本：那需要远端凭据与仓库权限，超出本轮授权；
工作流文件本身不能强制审查，必须由管理员在仓库规则中启用。

## 2. 审计发现与本轮处置

上一轮 DSH 只读审计（task `ca9c35fa-05ff-4e25-a1c4-2997d417826e`）的交接摘要
记录了 F1–F12。本轮逐项复核后的处置：

| ID | 根因 | 影响 | 检测为何失败 | 本轮处置 |
|---|---|---|---|---|
| F1 | `structure_complete` 只看 `gaps==[]`；gaps 只来自 V4.1 Engram | Qwen3.6/历史模型被误标“结构完备”；UI/比较器误读 | 测试只断言 gaps 空，没有 forward 证据栏 | `structure_complete = config_coverage_complete ∧ forward_evidence_complete`；新增四栏 `evidence`；UI/比较器只消费后者 |
| F2 | V4.1 `o_a_proj` 只算一组，漏乘 `ng=o_groups/tp`；memory 还漏乘 `b_dense` | TP1 每层 8.39M→67.11M FLOPs、权重 8.39→67.11 MB；40 层约 +2.35 GFLOP/token、+2.52 GB(BF16) | 旧测试只做 count↔memory 同源对账，两者同错；无 TP 绝对值 oracle | work 的 `gemm(reps=ng)` 与 memory 同步乘 `ng`；memory V4.1 attention 全部补 `b_dense`；新增 TP1/2/4/8 绝对 oracle |
| F3 | DeepSeek-V3 dense 层 FFN 宽度用了 `moe_intermediate_size=2048`，应为 `intermediate_size=18432` | 归档模型 3 层少约 2.11 GB/2.11 GFLOP/token | 测试只查层数/MoE，不查 dense 宽度 | 归档 parser 修复并加断言；模型仍不活跃 |
| F4 | catalog revision 与 identity 冲突（30B/V3） | 同一快照声明两个 revision | 旧测试只查前缀 | 归档 revision 改为与 identity 全等；新增全等断言 |
| F5 | Qwen3.6 shared expert gate 因 config 无 `shared_expert_gate` 键被推断 False | 每 Layer 少 gate 行/容量；source_map 声称已计入为假 | 测试只查 MoE 层/专家数，不查 gate 行 | 非 v3 家族按 `shared_intermediate>0` 推导 True；IR/work/memory 同步；每层 gate 行测试 |
| F6 | `run_v2` evidence_refs 硬编码 Qwen/DFlash/V4.1 | 跑 V4.1 也引用 Qwen/DFlash；跑归档模型引用错 profile | 无 refs 与 model/draft/profile 关联测试 | `ModelIR/DraftIR.evidence_refs` + `_run_evidence_refs`；refs⊆registry、模型特定测试 |
| F7 | coverage 只覆盖 `text_config`，未知键默认 ignored | 外层/嵌套键漏登记；测试自证 | 测试只遍历已有 coverage | 展平完整 config 逐叶键登记；未知键 `unsupported-semantic`+gap；视觉塔显式 metadata-only |
| F8 | Qwen3.6 MTP 只有 config+目标首层模板；V3 nextn 被忽略 | MTP 被当成已支持方法 | 无 MTP forward pin/状态测试 | MTP 单列 `unverified`；source_map 增未核条目；方法能力/排名阻断；V3 `mtp_layers` 归档登记 |
| F9 | V4.1 CSA2/MoE 为 v0.1 移植；scoring/top-k/candidate 未消费 | 可能过早 complete | source_map 只覆盖 index/hc/Engram | 保持真实 gaps；新增 `sm-v41-csa2-moe-unverified`；UI/排名阻断 |
| F10 | DFlash/MTP/chain 草稿按 B=1 计；DFlash/MTP 的 d2t/t2d 消息也漏 B | B>1 扫描草稿算量与链路字节少 B 倍 | 无 B>1 草稿/消息测试 | DFlash/MTP/chain 显式 `batch`；token 维与消息乘一次；权重不随 B；B=1/2 精确测试 |
| F11 | DraftIR 无 coverage/gaps | 草稿 config 绕过 M0 门禁 | 只测结构字段 | DraftIR 增 coverage/gaps/evidence；完整 config 叶键登记 |
| F12 | 无 shape/numeric 执行；source_map 只查字段存在 | “source-checked”不可证伪 | 无执行级/行号测试 | 执行栏明确 not-run；source_map 行号/状态测试；未执行不标 executed |

### 2.1 额外发现（F2 同轮）

`lowering/memory.py` 的 V4.1 attention 权重项全部漏乘 `b_dense`，把“元素数”当
“字节数”；BF16 下整块 attention 权重少 2×。独立 TP oracle 暴露后一并修复。
这是**未在上轮审计 F2 中单列**的真实错误，已记入本记录而不是只修表面 `o_a`。

### 2.2 数值变化（可直接复核）

- **V4.1 `o_a_proj`，TP1，每层，BF16**：FLOPs `8,388,608 → 67,108,864`；
  权重容量 `8,388,608 → 67,108,864 B`。40 层增量
  `2,348,810,240 FLOPs/token`、`2,516,582,400 B`（≈2.52 GB / 2.35 GiB）。
  TP8 时 `ng=1`，o_a 本身不变。
- **V4.1 attention dense 权重，TP1，BF16**：`3,952,082,944 → 10,252,976,128 B`
  （o_a 分组 + 全部项补 `b_dense`）。TP8：`1,016,070,144 → 2,032,140,288 B`
  （仅 `b_dense` 2×）。
- **Qwen3.6 shared gate**：每 MoE 层 `2·T·h` FLOPs、`h` 权重；40 层，
  W8A8 稳态 `+81,920 B`，BF16 `+163,840 B`；decode B1 总算量约 `+163,840 FLOPs/token`。
- **默认场景 Qwen3.6+DFlash（B1 ctx32k TP8/EP1 W8A8+FP4E）**：稳态
  `2,827,423,168 → 2,827,505,088 B`（+81,920 shared gate）；结构化临时峰值
  731,008 B 不变；known_peak `2,828,236,096 B`，仍**不完整**（workspace/graph/
  NCCL 池未知），可分配预算仍 unknown。
- **DFlash 草稿 B=2（E[produced]=3）**：token 维 FLOPs/激活/KV 恰 2×；
  权重读/容量不变；`fc/lm_head/noise` 不出现 B²。DFlash d2t/t2d 与 MTP t2d
  消息字节也恰 2×（此前漏 B）；权重不随 B。MTP B=2 算量 2×。目标 verify 与
  通信按各自的 `B·blk` 计一次，未重复乘。

## 3. 修复/未修复边界

已修复（离线测试覆盖）：活跃目录边界、完整 config 覆盖、四栏证据、
`structure_complete` 语义、V4.1 o_a 与 memory 单位、Qwen3.6 shared gate、
DFlash/MTP B>1 草稿计数、model/draft/profile 证据引用、source_map 行号/状态、
归档 profile 跨模型拒绝、G7 校准守恒（归档模型显式注入）、CI 工作流。

未修复/未验证（不得据此宣称 supported）：

- **V4.1 Engram**：8 键仍 unsupported；`common/engram.py` 未 pin，表 dtype/
  注入算子/容量未知。
- **V4.1 mHC**：source-checked + 近似计数；tilelang kernel 未执行。
- **V4.1 CSA2/MoE**：v0.1 移植，scoring/top-k/分组路由/候选发布未独立核。
- **V4.1 量化**：`quantization_config` 未映射到计算精度；场景精度是独立输入。
- **Qwen3.6 MTP**：HF 忽略 `mtp.*`，vLLM 主模型丢弃 `mtp.`；只有 config +
  目标首层模板，无原生 forward pin。
- **Qwen3.6 mrope**：文本位置 id 选择语义未执行；本仓计数不变，但未实测。
- **DFlash 服务端集成**：（2026-11-12 R07a 更新：双栈已逐行核对，colocated=
  source-checked/dt=unsupported；并修正 ctx 投影跨度基（含 ctx K norm/rope/
  layout copy）、草稿生效块与 full 层非因果，见
  [R07a 报告](dflash-serving-audit-20261112.md)）；拒绝采样/KV 回退/hidden
  提取仍显式 0 占位（vLLM prepare-inputs 为图外 eager Triton）；运行时池尺寸未知。
- **CPU/GPU 执行**：Qwen3.6 固定 HF 源码的两层随机权重 CPU shape probe 已执行；
  真实 40 层 checkpoint、MTP、V4.1 和 DFlash 服务端未执行，两款主模型无 GPU/部署实测。
- 远端 required check/PR review 未配置；候选分支已推送并建 PR #1（2026-11-10）。

## 4. 自动门禁与测试收据

`tests/test_merge_gate.py` 在 G1–G7 上新增 G8–G17（详见
[MERGE_CRITERIA §5](../MERGE_CRITERIA.md)）。G8–G17 覆盖：活跃模型恰两款、
完整 config 不默认忽略、四栏证据、V4.1 o_a 绝对 oracle、shared gate 每层、
草稿 B>1、model-specific evidence refs、部署排名阻断、source_map 校正、
CI 工作流三事件/工具链。

本轮执行（非全量；外层 Codex 是唯一全量 pytest owner）：

| 命令 | 结果 |
|---|---|
| `py -3.12 -m pytest tests/test_model_ir_v02.py -q` | 19 passed，exit 0 |
| `py -3.12 -m pytest tests/test_model_ir_v02.py tests/test_pipeline_v02.py tests/test_parallel_sharding.py tests/test_v41_hc.py tests/test_colocated_memory.py tests/test_budget_feasibility.py -q` | 96 passed，exit 0 |
| `py -3.12 -m pytest tests/test_merge_gate.py tests/test_product_scope.py -q` | 27 passed（含 G6/G17），exit 0 |
| `py -3.12 -m pytest tests/test_calibration_h200.py tests/test_calibration_h200_235b.py -q -k "not tampered"` | 21 passed，2 deselected，exit 0 |
| `py -3.12 -m pytest tests/test_ui_v12.py::test_stale_response_not_reused tests/test_consistency.py::test_fixed_scenario_replay_digest_stable tests/test_consistency.py::test_scenario_validation_errors -q` | 3 passed，exit 0 |
| `py -3.12 -m pytest --collect-only -q tests` | 195 tests collected |
| `node --check simverse/static/workbench.js` | exit 0 |
| `py -3.12 -m compileall -q simverse` | exit 0 |
| `git diff --check` | exit 0（仅 LF→CRLF 提示，无 whitespace error） |
| workflow YAML parse | `yaml.safe_load` 成功，triggers 为 pull_request/push/merge_group |
| `py -3.12 -m simverse.cli list models` | exit 0；仅 deepseek_v41_flash/qwen36_35b_a3b |
| `py -3.12 -m simverse.cli run-v2` | exit 0；M01 36.4301 ms，M02 3.0802 但不参与排名 |

DSH 阶段 `test_g6` 用 collect-only 核对当时状态文档的 `pytest 收集数：195`；
外层新增 1 项边界测试后，当前收集数与状态文档均为 196。

另做一次不依赖 `tmp_path` 的手动 UI/CLI 等价重放（场景写入 `outputs/manual-ui/`）：
进程内/`/api/v2/run`/CLI `run-v2` 的 run_id 与 capability 相等，target 账本均
613 行、CSV 614 行、trace 5 事件。该手动重放不记作 pytest 通过，只证明代码路径
在沙箱临时目录问题之外可执行。

**未完成的本地用例**：5 个使用 pytest `tmp_path` 的用例在本机 DSH 沙箱下
setup 报 `PermissionError`（pytest 自建临时目录无法列目录：
`test_load_rejects_tampered_artifact`、`test_tampered_artifact_rejected`、
`test_web_cli_json_csv_consistent`、`test_web_equals_cli`、
`test_csv_export_matches_runresult`）。这是环境权限问题，不是断言失败；本轮
**没有**把这 5 项记作通过，也没有改测试放宽断言。外层 Codex 后来在正常权限下
跑全量，这 5 项均通过。

### 外层 Codex 验收补充（2026-11-10）

- `py -3.12 -m pytest tests -q`：DSH 交接候选 **195 passed**，exit 0；
  5 个 `tmp_path` 用例在外层正常权限下全部通过。另有 1 条旧的 Windows
  子进程 UTF-8 解码线程警告，未计作无警告通过。Codex 后续补充了
  `model_forward_evidence` 能力状态推导与单项缺口阻断资格测试。
- `py -3.12 tools/probe_qwen36_forward_cpu.py`：exit 0。脚本动态加载固定
  `modeling_qwen3_5_moe.py`（SHA256 `abac917f54dfe822a35f3bdb846593f76a2d74b251c96c45a245df273554c44c`）
  与固定 configuration 文件（SHA256 `b4292600803686a2d0aca86f39309177cc5564cc751ec81479594369a6b88416`），
  使用本机 Python 3.12 / torch `2.15.0+cpu` / transformers `5.22.1`。
  随机权重两层（GDN + full attention）prefill logits `(1,4,64)`、decode logits
  `(1,1,64)`，构造出 2 个 shared expert gate。`causal_conv1d`、
  `flash-linear-attention` 未安装，实际走 PyTorch reference fallback。
  这证明所选源码分支可作小尺寸 CPU shape 执行，不证明真实模型数值、
  MTP、融合核、完整 forward 或部署延迟；模型级 `cpu_forward` 仍为 `not-run`。
- **历史事实（2026-11-10 修复前）**：远端 `PiratePai/simverse` 当时为私有仓库；
  Codex 对 main 的 branch protection 与 rulesets 做只读 API 查询，两者均返回
  HTTP 403：`Upgrade to GitHub Pro or make this repository public to enable this feature.`
  当时账号条件下无法启用 GitHub 强制 required check/PR review。
- **当前状态（2026-11-10 文档收口时）**：仓库已公开（`isPrivate=false`），候选分支
  已推送，PR #1 为 OPEN；main branch protection 查询返回 404
  `Branch not protected`，rulesets 为空。保护规则仍未配置；
  `.github/workflows/merge-gate.yml` 只提供自动检查，不冒称已强制合并阻断。
- 最终候选 `py -3.12 -m pytest --collect-only -q tests`：196 tests collected，exit 0；
  `py -3.12 -m pytest tests -q`：**196 passed, 1 warning in 6.73s**，exit 0。
  警告仍来自既有 Windows 子进程 UTF-8 解码线程（`test_csv_export_matches_runresult`），
  不能写成无警告通过。Codex 增加的单项 forward 缺口资格测试亦包含在此轮。

> **2026-11-10 CI 复核（DSH）**：上述本地 `196 passed`（6.73s）复用了被
> `.gitignore` 忽略的 `outputs/run_v41.json`，**不能证明 clean checkout 可复现**。
> 远端 run `42803015653` 的 `2 failed, 194 passed` 正是该文件缺失导致。两项验收
> 已改为 hermetic 形式（本次 run 落盘结果 + 纳入版本库的历史快照），验证边界见
> [Windows CI hermetic 修复](windows-ci-hermetic-fix-20261110.md)。
>
> **2026-11-10 外层复验（hermetic 修复后、未提交）**：外层唯一全量
> `py -3.12 -m pytest tests -q` 实际 exit 0：`196 passed, 1 warning in 5.04s`；
> 警告仍为既有 Windows 子进程 UTF-8 解码线程。该结果证明本地工作区通过，
> 不是 clean checkout 或远端 CI 通过。修复文件尚未提交/推送，PR #1 现有检查
> 仍是修复前失败；远端重跑待外层执行。

## 5. 历史归档声明

以下资产/报告保留日期事实，**只作历史归档**，禁止作为两款主模型的产品选择、
部署排名或跨模型 profile 依据：

- `assets/config/qwen3_30b_a3b__config.json`、`qwen3_235b_a22b__config.json`、
  `qwen3_next_80b_a3b__config.json`、`deepseek_v3__config.json`；
- `assets/profiles/h200_qwen3_30b_a3b_tiny/`、`assets/profiles/h200_qwen3_235b_a22b/`；
- `assets/scenarios/h200_calib_loop1.json`、`h200_calib_loop2_235b.json`（`archived: true`）；
- `docs/calibration/h200-loop1.md`、`h200-loop2-235b.md`、
  `docs/calibration/oldrepo-data-audit-20261110.md`；
- `docs/validation/speculative-ranking-audit-20261110.md`（其数值/结论对应当时
  候选分支，不自动继承到本轮）。

历史模型 parser/revision 修复只用于归档回归；`build_catalog()` 默认不返回它们，
API/CLI/工作台无法选择。历史 profile 通过 `PROFILE_MODEL_BINDINGS` 绑定采集
模型，跨模型直接 `ValueError`，不靠形状碰巧不匹配兜底。

## 6. 后续工作

1. 优先补 V4.1 Engram pin、CSA2/MoE 源码逐行核对、Qwen3.6 MTP 原生 forward；
   在这些完成前不要解除 `forward_evidence_complete` 或部署排名阻断。
2. GitHub 强制审查：仓库已公开，已具备配置公开仓库保护规则的条件；仍需仓库
   管理员在 ruleset/branch protection 中把 `pytest (Windows, Python 3.12)` 设为
   required，并启用 PR review。当前查询 main 为 404 `Branch not protected`、
   rulesets 为空，只具备 CI 自动检查。
3. H200 profile 若要重新启用，必须重新绑定到对应活跃模型并完成留出/域门禁，
   不能把 30B-tiny/235B 的结果平移到 Qwen3.6/V4.1。
