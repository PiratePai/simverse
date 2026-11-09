# simverse（暂名）— 统一推理仿真仓 · 首个垂直切片

对应《统一推理仿真仓库重构方案 v2.2》（model-verse 仓 `重构方案_新仿真仓库.html`）的
M0 基线 + M1 首链：**投机解码 / Draft-Target 分离的可追溯验证**。
当前交付报告见 [docs/verification_report.md](docs/verification_report.md)。

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
