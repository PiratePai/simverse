# 远端 Windows CI 两项失败：根因与 hermetic 修复（2026-11-10）

## 结论

PR #1 的两次 Windows/Python 3.12 检查失败（均为修复前记录，run `42803015653`，`2 failed, 194 passed`）
都源于测试读取被 `.gitignore` 忽略的 `outputs/run_v41.json`；CI 的
`actions/checkout@v4` 不包含该文件，必然 `FileNotFoundError`。修复前本地
`196 passed` 是因为工作区恰好保留了该归档，属于环境假阳性，不能证明 checkout
后的 CI 可复现。

本次把两项验收改为 hermetic 形式：CSV 测试改用本次 `run` 落盘的 RunResult；
固定场景测试改用纳入版本库的**历史归档快照**（只含 `digests.scenario` + `metrics`，
带来源哈希与提取说明）。没有删除/跳过验收，没有把现场 `run()` 值当 golden，
也没有把 1.3 MB 完整输出塞进仓库。

## 1. 远端失败证据

`gh run view 42803015653 --repo PiratePai/simverse --log-failed` 摘要：

| 测试 | 远端错误 |
|---|---|
| `tests/test_consistency.py::test_csv_export_matches_runresult` | CLI `export` 读取 `D:\a\simverse\simverse\outputs\run_v41.json`，`FileNotFoundError`，子进程 exit 1 |
| `tests/test_consistency.py::test_fixed_scenario_replay_digest_stable` | 直接读取同一路径，`FileNotFoundError` |

远端汇总：`2 failed, 194 passed in 6.18s`，job exit 1。GitHub 侧 checkout
行为依据官方 `actions/checkout@v4`：只取版本库内容，不取被忽略的未跟踪文件。

本地忽略规则与归档身份：

```text
$ git check-ignore -v outputs/run_v41.json
.gitignore:6:outputs/	outputs/run_v41.json
```

| 对象 | 观测值 |
|---|---|
| 归档文件 | `outputs/run_v41.json`，1,331,879 B |
| 归档 SHA256 | `4dce83d65b932bc103593c85cd7541df74863a571132b8539d0a0e91be326b63` |
| 归档 mtime（本地） | `2026-11-06 01:58:30`（未记录时区） |
| 归档 simulator_version | `0.1.0` |
| 归档 ops 行数 | 2,493（完整输出未纳入仓库） |
| 固定场景文件 | `assets/scenarios/v41_flash_dt_verification.json` |
| 场景文件 SHA256（LF 归一化） | `38c6c34764ffa30913a089663523bbf06e7a9969bb78502ee8499baf01561103` |
| 场景 digest（v0.1 16-hex） | `sha256:196c24706cbc635a` |

归档的 `replay.scenario_file` 指向上述固定场景；归档未记录生成命令，快照
provenance 如实标注这一点，不补造命令。

## 2. 根因与假阳性边界

1. 两项测试把 `outputs/run_v41.json` 当作“已提交交付物”，但 `.gitignore`
   第 6 行忽略了整个 `outputs/`。本地有归档时测试通过，CI clean checkout
   没有归档时失败。
2. 前一轮本地全量 `196 passed` 复用了本地归档，因此没有覆盖“文件不存在”
   的 CI 条件；它不是远端通过证据，也不是 hermetic 验收。
3. 原 CSV 测试还有一层薄弱点：它用存档 JSON 导出 CSV，再只比行数与
   `RunResult.ops` 长度，既依赖 ignored 文件，也没有逐列核对序列化忠实度。

## 3. 修复内容

### 3.1 CSV 导出验收

`tests/test_consistency.py::test_csv_export_matches_runresult` 现在：

- 复用 `cli_run` fixture 在 `tmp_path` 中新生成的 RunResult 文件作为
  `simverse.cli export` 输入；
- 用 `csv.reader` 逐行读取导出结果，校验表头恰为 15 列；
- 逐行、逐列对照同一 RunResult 的 `ops[]`（`stage/layer/op/exec_count/flops/
  weight_read/act_read/act_write/kv_read/kv_write/scope/formula_id/t_ms/bound/
  shape_note`），包括 CLI 对逗号/换行的转义；
- 不再读取 `outputs/` 下任何文件。

### 3.2 固定场景重放验收

`tests/test_consistency.py::test_fixed_scenario_replay_digest_stable` 现在读取
`tests/fixtures/v41_flash_dt_verification_expected.json`：

- 快照仅保留归档的 `digests.scenario` 与 `metrics`（2.5 KB 级），不包含
  `ops/stages/timeline`；
- `provenance` 绑定原归档 SHA256/大小/mtime/ops 数、场景文件 LF 归一化
  SHA256 与场景 digest，并声明“期望值来自历史归档，未用本次 `run()` 输出推导”；
- 测试先校验当前固定场景文件的 LF 归一化 SHA256 与快照 provenance 一致
  （场景一旦变更必须显式复核/重建快照；归一化避免 Git checkout 的 CRLF
  转换造成环境假阴性），再断言 `scenario_digest` 与规范化后的 `metrics` 全等；
- 另用固定场景的 `alpha=0.75, K=4, B=1` 作闭式算术独立核对
  `E_committed_len`、`P_full`、`E_draft_step0_tokens`、`verify_tokens_M`，
  避免只靠历史快照自证；
- 期望值是历史 T0 公式估算快照，不是测量值或 forward 执行结果。

### 3.3 未改动的部分

- 没有修改 `simverse/` 生产代码、CLI/导出实现、工作流文件或分支保护。
- 没有删除、跳过、xfail 或放宽上述验收；测试总数仍为 196。
- 没有提交、推送、合入，也没有触发远端 CI。

## 4. 验证记录

| 命令 | 结果 | 说明/边界 |
|---|---|---|
| `py -3.12 -m pytest tests/test_consistency.py::test_fixed_scenario_replay_digest_stable tests/test_consistency.py::test_scenario_validation_errors -q -p no:cacheprovider` | `2 passed`，exit 0 | 走 pytest 标准收集；不含 tmp_path 用例 |
| `py -3.12 -m pytest tests/test_consistency.py -q -p pytest_harness_plugin -p no:cacheprovider` | `4 passed`，exit 0 | DSH 沙箱内 `tmp_path` ACL 拒绝，临时插件把 tmp_path 指到工作区可写目录；插件在 `.tmp/`，不入库 |
| 在 `git archive HEAD` 解出的 `.tmp/hermetic-checkout`（无 `outputs/`）复制本次测试/快照后，在副本目录内运行同命令 | `HAS_OUTPUTS=False`；`4 passed`，exit 0 | 直接模拟 CI clean checkout 的“无 ignored outputs”条件；仍使用上述本地 tmp_path 插件 |
| 在 hermetic 副本中把固定场景转为 CRLF（778 B）后运行 digest 测试 | `1 passed`，exit 0 | 验证 LF 归一化哈希在 Git `autocrlf`/CRLF checkout 下不产生假阴性 |
| 直接调用 `cli_run`/`cli_result` 与两个测试体（工作区临时目录） | `MANUAL_OK ops=2493`，exit 0 | 复核 CSV 导出逐列一致；属于本地 harness，不记作 pytest 全量通过 |
| `py -3.12 -m pytest tests/test_merge_gate.py::test_g6_status_doc_test_count_matches_reality -q -p no:cacheprovider` | `1 passed`，exit 0 | 状态文档仍声称 196，与实际收集一致 |
| `py -3.12 -m pytest --collect-only -q tests` | `196 tests collected`，exit 0 | 未预跑全量 |
| `node --check simverse/static/workbench.js` | exit 0 | 工作流同款静态检查 |
| `py -3.12 -m compileall -q simverse` | exit 0 | 语法/字节码编译 |
| `git diff --check` | exit 0 | 无 whitespace error |

本地默认 `pytest tmp_path` 仍会在 DSH 沙箱报 `PermissionError [WinError 5]`
（pytest 自建临时目录无法列目录，是既有环境问题）。因此上面的 4 项 pytest
结果是“临时插件改 tmp_path”后的通过，**不是**默认 tmp_path 的通过；CSV
测试体另由直接 harness 复核。远端 Windows runner 使用正常 tmp_path。

**外层 Codex 验收（2026-11-10，本卡交付后）**：外层在本地工作区执行唯一一次
全量 `py -3.12 -m pytest tests -q`，实际 exit 0：`196 passed, 1 warning in 5.04s`；
警告为既有的 Windows 子进程 UTF-8 解码线程（`test_csv_export_matches_runresult`）。
该结果覆盖了本次 hermetic 修复，但仍是本地工作区结果，不是 clean checkout 或
远端 CI 通过证据。

## 5. 验证边界与后续

- 本卡未运行全量 `pytest tests -q`；外层 Codex 是唯一全量 pytest owner，已于
  2026-11-10 在 hermetic 修复后本地全量通过（见 §4）。
- 本卡未提交、推送或重跑 GitHub Actions；修复文件在交付时仍未提交。
- 当前远端状态（2026-11-10 只读查询）：仓库已公开（`isPrivate=false`）；分支
  `codex/forward-merge-gate` 已推送，PR [#1](https://github.com/PiratePai/simverse/pull/1)
  为 OPEN，head 仍为修复前提交 `15abdc5`；PR 检查仍为修复前两次失败
  （run `42803015653`、`42803070462`）；main branch protection 返回 404
  `Branch not protected`，rulesets 为空。修复后远端检查尚未运行，不能写成
  远端已通过或保护已启用。
- run `42803015653` 是修复前证据，修复后的实际远端通过需由外层推送后确认。
- 本地 hermetic copy 证明测试不依赖 `outputs/`，但不能替代远端 Windows
  runner 的 tmp_path/子进程/编码行为。
- 若未来固定场景文件变更，先复核快照 provenance 并显式重建 fixture；
  不得用现场 `run()` 输出自动覆盖期望值。
