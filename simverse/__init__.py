"""simverse — 统一推理仿真仓（暂名）首个垂直切片。

对应《统一推理仿真仓库重构方案 v2.2》M0–M1 首链：
  投机解码 / DT 分离场景的可追溯验证（L0 结构 → L1 成本核 → L3 解析执行 → CLI/Web 同入口）。

三段流水（原理 v1.1）：
  Count（整数计数，可手算） → Cost（roofline T0 公式估算） → Execute（解析组合器）。

本切片边界（显式声明，详见 docs/verification_report.md）：
  * 求值：仅解析档（单请求独占/固定批形状）；事件状态机与 serving 是 M3/M4；
  * 校准：全链路 T0 公式估算，无实测/拟合数字（决策 D-4）；
  * 投机方法：mtp / dspark / custom_chain 链式草稿；块并行与 jetspecpp 显式不支持（决策 D-6）；
  * 模型：tiny_dense_oracle（手算 oracle）与 DeepSeek V4.1 Flash（首发模型，决策 D-3）。
"""

__version__ = "0.2.1"  # 0.2.1: 校准复审修复+235B 闭环+V4.1 hc（协议 schema 0.2 不变）
SCHEMA_VERSION = "0.1"
