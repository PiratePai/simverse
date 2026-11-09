# AGENTS.md — Simverse（new-viewer）

统一推理仿真仓。本文件是贡献/合入入口；模型/仿真/验证的完整决策背景见
[docs/rebuild-20261106/plan.md](docs/rebuild-20261106/plan.md)。

## 必读（合入前）

1. **[合入准则](docs/MERGE_CRITERIA.md)**（强制）：模型语义核对、数值口径、
   profile/校准边界、验收与披露。可机械化部分由 `tests/test_merge_gate.py`
   自动执行（G1–G7），先跑 `py -3.12 -m pytest tests -q` 再提交。
2. [原理底座](docs/rebuild-20261106/principles.md)：六类量不可混算、公式与状态口径。
3. [实施状态](docs/IMPLEMENTATION_STATUS.md)：里程碑与缺口现状（改动后同步更新，
   测试数与实际一致会被门禁核对）。

## 硬规则（违反即回退）

- 校准/profile 只改成本，不改 Work/Traffic/Memory 原始计数。
- synthetic 不升格 measured；域外显式回退解析，不外推、不用全局倍率。
- 未知/不支持/容量未知必须传播到指标与比较资格，不得填 0 或静默。
- 设备/链路数值必须带来源（spec 注明 dense/sparse 与方向，或 assumed）；
  peak 一律 dense 口径。
- 结果协议字段见 simverse/contracts（v0.2）：Metric null+reason、完整 SHA256、
  js_safe 大整数；禁止 NaN/Infinity。
- 不直接依赖参考仓（D:/Github/pi-viewer、D:/Github/model-verse）内部路径：
  迁移以"场景与证据包"为单位，来源+提交+哈希入册（tools/distill_*.py 可重放）。

## 常用命令

```bash
py -3.12 -m pytest tests -q                 # 全量测试
py -3.12 -m simverse.cli run-v2             # 必需场景（Qwen3.6+DFlash）
py -3.12 -m simverse.cli serve              # 工作台 http://127.0.0.1:8765/
py -3.12 tools/distill_h200_profiles.py --help   # H200 profile 蒸馏导入（可重放）
```
