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
py -3.12 -m pytest tests -q                  # 全量验收（外层 Codex 唯一 owner）
py -3.12 -m simverse.cli run-v2              # v0.2 管线（Qwen3.6-35B-A3B + DFlash，默认场景）
py -3.12 -m simverse.cli run-v2 --csv ops.csv --trace events.csv --html report.html
py -3.12 -m simverse.cli serve --port 8765   # Web 工作台（/ 生产 v0.2；/legacy 为历史存档）
```

v0.1 `run/sweep` 与 `/legacy` 仅保留为历史存档，不属于活跃产品入口；v0.2 的
CLI/API/工作台只列两款主模型与 DFlash。


## 结构（v0.2 管线）

```
结构     model_ir/        ModelIR/DraftIR + 完整 config 覆盖 + 四栏证据
计数     count/           work/MoE/draft 逐算子账本
降层     lowering/        TP/EP 配方、通信、容量/预算
成本     cost/            T0 roofline + 可选域门禁 profile（当前无活跃 profile）
执行     execute/         串行/资源事件核、指标归约
入口     pipeline.py      ScenarioV2 → ResultBundle（CLI/API/导出唯一入口）
```

场景可存盘重放：`assets/scenarios/*.json` → 同 digest → 同数字（测试钉死）。
`h200_calib_*` 场景已标 `archived: true`，不会出现在工作台场景索引。

## 边界声明

- 性能数字是 **T0 解析估算**；没有两款主模型的 CPU forward 执行或 GPU/部署
  实测，不能宣称实际部署排名。
- V4.1 的 Engram/CSA2/MoE/量化仍有真实缺口；Qwen3.6 MTP 只有 config + 目标
  首层模板，标 `unverified`；DFlash 服务端集成与运行时池尺寸未核。
- 历史 profile 只绑定其采集模型；跨模型使用直接拒绝。历史模型/tiny fixture 只能
  通过测试显式注入归档目录运行。
- 结果协议见 `simverse/contracts` v0.2；合入自动化见
  [MERGE_CRITERIA](docs/MERGE_CRITERIA.md) 与 `.github/workflows/merge-gate.yml`。

依赖：Python ≥3.10（CI 使用 3.12）+ Flask；无 pydantic/matplotlib/torch 运行时依赖。
