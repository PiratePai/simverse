# simverse — 统一推理仿真仓

重点模型：**Qwen3.6-35B-A3B + DFlash**、**DeepSeek-V4.1-Flash**。
待办优先级与验收标准见 **[Roadmap](ROADMAP.md)**；已实现能力与限制见
[功能状态](docs/FOCUS_STATUS.md)。历史模型保留为内部回归/校准资产。

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
python -m pytest tests -q                    # 全量验收（手算 oracle + 一致性 + 负例）
python -m simverse.cli run-v2                # v0.2 管线（Qwen3.6-35B-A3B + DFlash，默认场景）
python -m simverse.cli run-v2 --csv ops.csv --trace events.csv --html report.html
python -m simverse.cli serve --port 8765     # Web 工作台（同上；/ 生产 v0.2，/legacy v0.1）
python -m simverse.cli run                   # v0.1 legacy 固定场景（V4.1 Flash + DSpark 链式）
```


## 结构（三段流水，公式单一维护点）

```
L0 结构    models.py        ModelSpec/DraftSpec（config 快照 + SpecMethod 注册表）
L1 成本核  cost.py ledger.py formulas.py   roofline + MoE 并集 U(M) + 逐算子整数账本
L3 执行    acceptance.py engine.py         E[n]/Catch-up + 四部署解析 TPOT + 时间线
入口       cli.py webapp.py                CLI 与 Web 同一 engine.run()
扫描       sweep.py chartpng.py            边界扫描 + 纯标准库 PNG 图表
```

## 快速开始

```bash
python -m pytest tests -q                    # 13 项验收（手算 oracle + 三方一致性）
python -m simverse.cli run                   # 固定场景（V4.1 Flash + DSpark, DT 分离）
python -m simverse.cli sweep                 # α/K/ctx/B/链路/草稿带宽 六维边界 + PNG
python -m simverse.cli serve --port 8765     # Web 工作台（http://127.0.0.1:8765）
python tools/oldrepo_diff.py                 # 两旧仓对账（需 model-verse/pi-viewer 在本机）
```

场景可存盘重放：`assets/scenarios/*.json` → 同 digest → 同数字（测试钉死）。

## 边界声明（v0.1）

- 时延全部 **T0 公式估算**（roofline；无实测/拟合数字，决策 D-4）；
- 解析档（单请求/固定批形状）；事件状态机与 serving 属 M3/M4；
- 投机方法支持 `mtp / dspark / custom_chain`（链式）；块并行与 jetspecpp **显式
  unsupported**（不静默输出，决策 D-6）；
- 模型注册表：`deepseek_v41_flash`（官方 config 快照）与 `tiny_dense_oracle`（手算样例）。

依赖：Python ≥3.10 + Flask（无 pydantic/matplotlib——契约用 dataclasses，图表为标准库 PNG）。
