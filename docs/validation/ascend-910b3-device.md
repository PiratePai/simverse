# 910B3 旧仓参数迁移与解析入口

2026-11-17。目标设备与 DT 草稿设备增加「昇腾 910B3 64GB（旧仓解析档，未校准）」。这是可选择、可运行的受限解析档，不等待完整实测；设备可选与部署验证分开。

## 来源与参数

来源为 [model-verse](https://gitee.com/liliyongqi/model-verse) 的提交 `6a8acce825b19591a6331c20a3b16bf6b7d01661`。静态阅读 `hardwares/npu_params.py` 和 `scenarios/h3_pipeline_sim.py`，后者明确将 910B3 日志场景绑定到 `ascend_910B_64G`，并以 249T 作回归锚点；本次没有执行该旧仓场景或 NPU 任务。

源文件原字节、完整 SHA256、模型无关的设备映射与排除项保存在 `assets/identity/devices/ascend_910b3/manifest.json`。Simverse 运行不读取外部仓路径。旧表的上游说明为 CF++.xlsx 与 cube/vector 8:1 拆分，本次没有获得该表的官方 SKU 证明，因此数值逐条标为 `assumed:`，不升级为官方规格或本模型实测。

| 项 | 沿用旧仓值 | 使用方式 |
|---|---:|---|
| BF16/FP16 Cube | 249 TFLOP/s | 矩阵/混合 attention 行的 dense 解析档；不加 Vector 算力 |
| Vector | 31 TFLOP/s | 显式 norm/RoPE/add/激活行的 FP32 近似成本；未证明融合实现 |
| HBM 带宽 | 1.6 TB/s | 单卡聚合解析输入，十进制 B/s |
| 标称容量 | 64 GB | 64,000,000,000 B；不与 R29 设备 total/free 快照混用 |
| MFU / MBU | 0.20 / 0.20 | 旧仓假设；有效 Cube 49.8T、Vector 6.2T、带宽 320 GB/s；未拟合本模型 |
| 默认组卡数 | 8 | 沿用旧表，实际分片由 TP 输入；不表示做过八卡测试 |

旧表的 `interconnect_bw=50e9` 注释为 all-reduce 有效带宽，缺少方向/算法/拓扑定义，不能当通用点到点 HCCS 规格迁入 LINKS。现有通信仍使用场景链路假设并在能力报告披露；实际 Ascend topology/profile 待 R17/R18。`onchip_buffer=0` 是未知占位；不将它称测得零容量。FP8/INT8 未在此条目登记，不推导倍数、不填假 CUDA SM 数。

## 功能与边界

- Qwen3.6 的 BF16 baseline、同卡 DFlash/MTP 或 DT 研究可用；Ascend 活跃端权重/激活/KV 均须 16bit。界面选目标 NPU 时显示并锁定 BF16，NPU 草稿端也显式使用 BF16。
- V4.1 缺 Ascend KV/量化/稀疏注意力配方，API 明确拒绝；不借用 Marlin/FlashMLA/SM90。DFlash/MTP 的 CUDA 源码核对状态不传递成 Ascend 集成支持。
- 结果附 `ascend_device_recipe=unknown`；耗时标 approximated、feasibility unknown，设备未实测、不参与部署排名。没有可分配预算和完整运行时池数据，容量继续按已有四态规则判断，不把标称容量当真实预算。
- GPU 默认 MBU=1.0，计价与硬件身份快照保持原行为；NPU MBU/来源/后端进入身份快照，修改假设会改变 run_id。校准数据与原始 Work/Traffic/Memory 的关系未改。

独立手算：49.8 GFLOPs ÷ 49.8 TFLOP/s=1 ms，320 MB ÷ 320 GB/s=1 ms；现有 roofline `max+0.15*min` 得 1.15 ms。测试覆盖来源快照/数值、无 FP8/无 SM、API 真实运行、非法精度/V4.1 拒绝、草稿端作用域、排名与身份；全量结果见 IMPLEMENTATION_STATUS。该 1.15 ms 是手算 fixture，不是真机观测。
