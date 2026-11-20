# simverse — 统一推理仿真仓

活跃模型只有两款：**Qwen3.6-35B-A3B + DFlash DraftIR**、**DeepSeek-V4.1-Flash**。
待办优先级见 **[Roadmap](ROADMAP.md)**；能力与限制见
[功能状态](docs/FOCUS_STATUS.md)；合入边界见
[两模型 forward 审查（2026-11-10）](docs/validation/two-model-forward-merge-gate-20261110.md)。
Qwen3-30B/235B、Qwen3-Next、DeepSeek-V3 与 tiny 合成模型只作**离线归档/测试
fixture**，不出现在 API/CLI/工作台选择中。

v0.2 重构（2026-11-09 起，依据 `docs/rebuild-20261106/` 计划包）：ModelIR 共享子包 →
计数 → 降层 → T0 成本 → 串行/资源事件执行 → ResultBundle，CLI/Web/导出同入口。
实施状态与验收清单见 **[docs/IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md)**。
v0.1 首切片（投机解码/DT 分离可追溯验证）报告保留于
[docs/verification_report.md](docs/verification_report.md)。

## 一键启动

双击 **`启动工作台.bat`**（或拖到命令行带端口参数 `启动工作台.bat 9000`）：
自动起本地服务器并在就绪后打开浏览器 `http://127.0.0.1:8765/`。
停止：双击 `停止工作台.bat`，或关闭最小化的 `simverse-server` 窗口。
已在运行时再次双击 = 直接打开浏览器，不会重复起服务。

## 命令行入口

```bash
py -3.12 -m pytest tests -q                  # 全量回归
py -3.12 -m simverse.cli run-v2              # v0.2 管线（Qwen3.6-35B-A3B + DFlash，默认场景）
py -3.12 -m simverse.cli run-v2 --csv ops.csv --trace events.csv --html report.html
py -3.12 -m simverse.cli serve --port 8765   # Web 工作台
```

v0.1 场景通过 `simverse archive run` 或 `simverse archive sweep` 离线重放。已有 `RunResult` 使用 `simverse archive export` 导出。Web 工作台与 HTTP API 使用 v0.2。旧的 v0.1 网页、HTTP 路由和顶层 CLI 命令已经退役。

## 结构（v0.2 管线）

```
结构     model_ir/        ModelIR/DraftIR + 完整 config 覆盖 + 四栏证据
计数     count/           work/MoE/draft 逐算子账本
降层     lowering/        TP/EP 配方、通信、容量/预算
成本     cost/            T0 roofline + 可选域门禁 profile（当前无活跃 profile）
执行     execute/         串行/资源事件核、指标归约
入口     pipeline.py      ScenarioV2 → ResultBundle（CLI/API/导出唯一入口）
```

可用 `assets/scenarios/*.json` 重放场景。复验数值时固定代码、设备快照和成本资产。
`h200_calib_*` 场景已标 `archived: true`，不会出现在工作台场景索引。

## 结果范围

- 默认性能数字是 **T0 解析估算**。可选 Ascend R28 GEMM、R29 通信档提供局部条件对照，未覆盖的成本回退解析。两款主模型仍缺完整 checkpoint 的执行、匹配容量和 E2E 证据，部署排名未开放。
- Qwen3.6 有小型 CPU fixture 数值对照、文本 MRoPE 验证与原生 MTP 源码映射。DFlash 同卡服务路径已有源码核对，真实接受分布未验。V4.1 已建模 SwiGLU、稀疏 MLA 注意力核与其他主干模块，部署变体和运行时仍待验证。见 [功能状态](docs/FOCUS_STATUS.md)。
- 历史 profile 绑定采集模型，跨模型使用会拒绝。历史模型与 tiny fixture 只用于显式注入归档目录的离线脚本或测试。
- 导出已有 v0.2 JSON 使用 `simverse export-v2 run.json --trace events.csv`。新结果的同卡清单读取 `cost.memory.colocated_ranks`，选用时间轴在与解析时间轴有差异时出现在 `cost.selection.execution`。旧记录结果仍可导出。见 [实施状态](docs/IMPLEMENTATION_STATUS.md)。
- 请求级 TTFT、ITL 和服务吞吐依赖带请求身份的轨迹。采集与验收见 [端到端 trace 验收方案](docs/validation/r07b-r09-trace-acceptance.md)。
- 结果协议见 `simverse/contracts` v0.2；合入自动化见
  [MERGE_CRITERIA](docs/MERGE_CRITERIA.md) 与 `.github/workflows/merge-gate.yml`。

从源码检出运行需 Flask。Windows 合入门禁使用 Python 3.12，项目元数据声明 Python ≥3.10。Windows wheel 安装与工作台入口已通过独立冒烟。见 [R10 验证记录](docs/validation/entry-scan-compare-fixes-20261119.md)。
