# 本轮完成情况与验证边界

## 仿真材料

- 原有new-viewer测试实跑：**13 passed，1 warning**，warning为子进程输出UTF-8解码异常。未修改生产代码，未把这13项当作真实模型或硬件准确性证明。
- [audit.py](audit.py)运行成功，保存六组反例与原始材料hash：[audit-results.json](audit-results.json)。包括MoE并集、旧文覆盖率、同均值不同全接受概率、mean端点、baseline时间线和residual加法。
- Gitee固定提交原文已取得；官方HF config与本地快照hash一致。Vidur、DeepSpec与vLLM的固定版本源码/相关测试已阅读，没有执行它们的GPU测试。
- DSH/ZCode各返回12个质询问题。原始回复不改写；主控裁决单列，尤其纠正V4.1发布状态、DSpark跨模型泛化、hidden消息量、60KB传输时间与统计反例。

## UI样例

- 离线生成9个K/α组合；5项独立算术及不变量断言通过。`node --check workbench.js`通过。
- impeccable机械检查初次指出小于11px功能文字与有色dialog阴影；修正后结果为空列表`[]`。未将机械检查等同完整无障碍认证。
- 在IAB实际渲染桌面与390×844手机视口；手机文档clientWidth=scrollWidth=380，无整页横向溢出；表格与时间轴允许局部水平滚动。临时视口覆盖已复位。
- 实际操作验证：修改K/α出现stale状态→载入后数字/公式/轨道更新；层1 Attention公式对应Q7；更换比较基线后比值改变；普通decode不出现草稿事件；理想周期不冒充B1并行时间线；Qwen3-Next显示48层和12个全注意力层；V4.1显示结构缺口；假设说明开关；JSON下载。
- 导出的是当前显示场景、假设与所选基线；浏览器采集console error/warn为空。此处不声称所有路径已做自动化回归。
- 截图：[桌面](prototype/desktop.jpg)、[手机](prototype/mobile.jpg)。

## 明确未完成的生产能力

真实DSpark/V4.1完整lowering、Engram/HC容量、CSA2索引/候选单位核验、真实DT采样协议、TP/EP通信、事件资源核、服务调度与硬件profile仍是重构计划的工作项。UI使用教学预算；真实模型视图只展示层序/证据状态，不输出性能预测。正式UI尚未接入生产ResultBundle，算子视图为清单而非已验证DAG，时间预算不是事件实录。


## 原 HTML UI 复用增补

主入口现直接复用用户 HTML 的工作台 DOM、CSS 与交互骨架；旧教学界面转存 teaching.html，原 HTML 原样保存 reference-v3.html。对应关系见 prototype/reuse-map.md。

本轮验证：v3-workbench.js 通过 node --check；浏览器检查 25 个矩阵单元格、Qwen3-Next 标签联动、链式代理提示、流水 TPOT 待验证、无投机仅 Target 单轨；检查 1440px 桌面与 390px 手机布局，手机无整页水平溢出，宽矩阵在面板内滚动。浏览器未报告 warning/error。截图为 prototype/desktop-v3.jpg 与 prototype/mobile-v3.jpg。

以上是 UI 验证，不是原版公式、真实性能或 forward 验证。本轮没有改动或重测生产内核。


## UI 纳入计划与后端契约增补

本轮只更新文档：plan.md 第 8 节与里程碑约束、ui-data-contract.md、questions.md、README/principles/复用对照链接。新增契约定义 8 类页面/详情视图、23 组指标、12 组验证步骤；这些是未来验收项，不代表已经完成实现或验证。questions.md 保留 Q1–Q7，并补 Q8–Q11，统一 11 个回答位置。

实际做了文档结构检查：页面/指标/验证编号完整无缺失；Q1–Q11 连续且各有回答位置；本轮修改的六份文档本地链接全部可解析；计划中旧“首版简化 UI”表述已替换。已同步文件 hash 清单。未修改 UI 代码或生产后端，未重复运行模型/性能测试，未新增 DSH/ZCode 调用。


## 第二轮 DSH/ZCode 复核与文档修订

真实调用 DSH read-only 与 ZCode guarded，均正常返回，共 18 项主发现；原始输入/输出、模式回执与裁决见 [review-ui-round2/resolution.md](review-ui-round2/resolution.md)。两位只读检查冻结文档，没有执行模型或测试。修订涉及 plan、principles、ui-data-contract、questions 及索引/验证记录；没有修改生产代码或 UI。

主控运行 review-ui-round2/counterexamples.py：7 个独立算术见证通过，涵盖同 shape 不同语义、client buffering、rank 容量、窗口统计、动态 K 联合律、常数错误与时间下界、并行 service/elapsed 区别。它们是审查反例，不能当作生产功能通过。文档编号、指标区域映射、Q1–Q12 和本地链接由 doc-checks.py 检查，结果见同目录 doc-checks.json。新增验收步骤均为计划要求；修改后未再发第三轮审查。


## 2026-11-09 用户逐项决策整合

Q1–Q12 的方向性选择均已写回 questions.md，并同步 plan.md、ui-data-contract.md 与 README。Q3 私有远端平台/账号/地址尚待补充；模型精确 revision/checkpoint、扫描范围、运行限额/等待时间和成本来源仍在接入/测量时确定。Q5 已明确 DSpark→MTP→自定义草稿→Chunked Prefill→PD 分离→Prefix Cache，平台暂不区分 CUDA/Ascend 先后。

本轮只修改文档和校验/hash 记录。实际检查：12 个问题编号连续且均有回答；页面/指标/验证主编号仍为 8/23/12；本次活动文档的本地链接可解析；旧的“自定义链式优先、CUDA先、平台必须后接”建议已按用户选择更新。检查结果为 decision-checks-20261109.json。未运行生产测试、模型、GPU、profile 或第三轮审查；新增方法/特性验收仍为计划要求，不是已实现能力。
