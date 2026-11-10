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
- 没有删除、跳过、xfail 或放宽上述验收；该轮测试总数仍为 196（后续 §6
  新增编码回归用例后为 197）。
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

## 6. 后续：CLI 导出编码失败（run 42805472945 / 42805463611）

### 6.1 远端证据

hermetic 修复推送后，GitHub Actions 仍失败；两次 run 的 pytest job 均为
`1 failed, 195 passed in 6.06s`，job exit 1：

| run | 触发 | 结果 |
|---|---|---|
| `42805472945` | PR #1 | `1 failed, 195 passed`，job exit 1 |
| `42805463611` | push `codex/forward-merge-gate` | 同左 |

`gh run view 42805472945 --repo PiratePai/simverse --log-failed` 的唯一失败：

```text
tests/test_consistency.py::test_csv_export_matches_runresult
File "D:\a\simverse\simverse\simverse\cli.py", line 162, in cmd_export
    print(f"已写 {p}")
File "C:\hostedtoolcache\windows\Python\3.12.10\x64\Lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
UnicodeEncodeError: 'charmap' codec can't encode characters in position 0-1: character maps to <undefined>
assert 1 == 0
1 failed, 195 passed in 6.06s
```

CSV 已写出，失败发生在随后的成功提示；测试在 `assert proc.returncode == 0`
处中断，尚未进入逐列断言。

### 6.2 输入环境与根因

| 项 | 远端观测/依据 |
|---|---|
| runner / Python | GitHub Actions `windows-latest`；`C:\hostedtoolcache\windows\Python\3.12.10\x64`，3.12.10 |
| pytest 命令 | `python -m pytest tests -q`（工作流同款） |
| stdio 形态 | 测试用 `subprocess.run(..., capture_output=True)`；子进程 stdout/stderr 是管道，不是控制台 |
| 子进程启动环境 | 未设 `PYTHONIOENCODING` / `PYTHONUTF8`；管道按系统 locale ANSI 代码页编码，traceback 显示实际为 `cp1252` |
| 测试侧 `encoding="utf-8"` | 只决定父进程如何**解码**捕获到的字节，不改变子进程的**编码** |
| 触发点 | `cmd_export` 写 CSV 成功后 `print(f"已写 {p}")`；中文无法用 cp1252 编码 |
| 为什么只有 export | `run`/`sweep`/`list` 等此前各自在打印前 reconfigure 到 UTF-8；`cmd_export` 漏了 |

官方 Python 文档 [sys.stdout][win-stdio] 明确：Windows 控制台使用 UTF-8；
磁盘文件、管道等非字符设备使用系统 locale 编码（ANSI codepage），可用
`PYTHONIOENCODING` 或 `-X utf8` / `PYTHONUTF8` 覆盖。因此同一段中文提示在
开发者本机（常为 UTF-8 环境）可打印，在 runner 的 cp1252 管道上直接抛异常。
这是 CLI 输出编码/结构问题，不是 CSV 序列化或逐列验收口径问题。

本地等价复现（本机默认 stdio 为 UTF-8，需显式压到 cp1252）：

```powershell
$env:PYTHONIOENCODING='cp1252'; Remove-Item Env:PYTHONUTF8 -ErrorAction SilentlyContinue
py -3.12 -m simverse.cli export .tmp/manual-verify-20261110/cli/run.json --csv .tmp/repro-export.csv
# 修复前：同样的 UnicodeEncodeError，exit 1
# 修复后：打印“已写 ...”，exit 0，CSV 347,416 B
```

### 6.3 处置

1. `simverse/cli.py` 新增 `_configure_stdio()`，在 `main()` 入口统一对
   stdout/stderr 执行 `reconfigure(encoding="utf-8", errors="backslashreplace")`，
   覆盖所有现有子命令和未来新增子命令；仅当流不支持 reconfigure 时静默跳过，
   不影响计算/导出。
2. 删除 `_dump`/`_summary`/`cmd_sweep`/`_summary_v2`/`cmd_sweep_v2`/`cmd_list`
   中六处分散的 reconfigure，消除“新命令漏配”的结构性来源。
3. `tests/test_consistency.py`：
   - `_cli_env()` 给子进程显式 `PYTHONIOENCODING=utf-8` 并移除 `PYTHONUTF8`，
     父进程按 UTF-8 严格解码，测试不再依赖 runner locale；
   - `_assert_ops_csv()` 保留表头、行数、逐行逐列对照 `RunResult.ops` 的验收，
     没有降级为“文件存在”；
   - 新增 `test_export_survives_non_utf8_stdout`：把子进程 stdio 强制为
     `cp1252`（并移除 `PYTHONUTF8`），要求 export exit 0 且 CSV 仍逐列忠实。
     这是对本次根因的直接回归，不是 skip/xfail。
4. 收集数 196→197；`docs/IMPLEMENTATION_STATUS.md` 的当前收集数同步为 197。

### 6.4 验证记录

| 命令 | 实际结果 | 说明 |
|---|---|---|
| `py -3.12 -m py_compile simverse/cli.py tests/test_consistency.py` | exit 0 | 语法检查 |
| `PYTHONIOENCODING=cp1252 py -3.12 -m simverse.cli export ...` | exit 0，打印“已写 ...” | 修复后本地等价复现；CSV 347,416 B |
| `py -3.12 -m pytest tests/test_consistency.py tests/test_merge_gate.py::test_g6_status_doc_test_count_matches_reality -q -p pytest_harness_plugin -p no:cacheprovider`（`PYTHONPATH=.tmp`，文档更新后） | `6 passed`，exit 0 | 5 项 consistency（含新增 cp1252 用例）+ G6 |
| 同一命令（文档更新前） | `1 failed, 5 passed`，exit 1 | G6 如实报告 196≠197，促使同步状态文档 |
| `py -3.12 -m pytest tests/test_merge_gate.py -q -p pytest_harness_plugin -p no:cacheprovider` | `25 passed`，exit 0 | 全部 G1–G17 门禁 |
| `py -3.12 -m pytest tests/test_ui_v12.py tests/test_calibration_h200.py -q -p pytest_harness_plugin -p no:cacheprovider` | `17 passed`，exit 0 | 相邻 CLI/校准测试 |
| `py -3.12 -m pytest --collect-only -q tests` | `197 tests collected`，exit 0 | 未预跑全量 |
| `node --check simverse/static/workbench.js` | exit 0 | 工作流同款静态检查 |
| `py -3.12 -m compileall -q simverse` | exit 0 | 字节码编译 |
| `git diff --check` | exit 0 | 无 whitespace error（仅 Git 的 LF→CRLF 提示） |

本地默认 `pytest tmp_path` 在 DSH 沙箱仍可能 `PermissionError`，以上 pytest
结果使用 `.tmp/pytest_harness_plugin.py` 把 tmp_path 定向到工作区可写目录；
这是本机 harness 限制，不是产品失败。远端 runner 使用正常 tmp_path。

### 6.5 仍未覆盖 / 不能声称

- 未提交、未推送、未重跑 GitHub Actions；远端最新记录仍是上述两次失败，
  不能写成“远端已通过”。
- 本卡未运行全量 `pytest tests -q`；外层 Codex 是唯一全量 owner，需在推送后
  重跑全量并复核 197 项。
- 只验证了“管道 stdio 为 cp1252”的等价复现，未在真实 `PYTHONLEGACYWINDOWSSTDIO`
  旧式控制台或非 Windows 终端上目检中文渲染；修复保证不崩溃且输出 UTF-8
  字节，字形显示取决于终端能力。
- runner 的实际 ANSI codepage 由 traceback（`cp1252.py`）与官方文档推断；
  本次日志未单独打印 `locale`/`chcp` 值。
- 新增用例使收集数从 196 变为 197；G6 已同步，但全量通过仍待外层验收。

[win-stdio]: https://docs.python.org/3/library/sys.html#sys.stdout
