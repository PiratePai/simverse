# 统一仿真仓库重构评审包

2026-11-06，综合用户提供的验证报告、指定提交的计算原理和simverse_v3_0.html，结合DSH与ZCode独立质询后形成。

推荐阅读顺序：

1. [重构计划](plan.md)：边界、模块、里程碑、验收与两旧仓迁移。
2. [原理底座](principles.md)：计数、流量、容量、成本、投机状态与系统指标。
3. [可交互UI样例](prototype/index.html)：直接复用原 HTML 工作台；展示数据，不是真实性能预测。另保留[可复算教学样例](prototype/teaching.html)和[复用对照](prototype/reuse-map.md)。
4. [证据复核](evidence-audit.md)：21项问题、固定源码版本、独立反例。
5. [DSH/ZCode审查裁决](review-resolution.md)：逐条采纳、修正、未采纳的理由，含原始答复及真实调用回执。
6. [UI 页面、指标与后端验证契约](ui-data-contract.md)：8 类视图、23 组指标、12 组验证步骤及 M0–M5 接入顺序。
7. [需要你决策的问题](questions.md)：可直接填写，不重问已确定方向。

核心判断：当前问题主要在模型/算法语义、物理实现、数据口径和验证层级混淆；换UI或再抽一个共享函数包不足以解决。先把「模型语义→执行配方→工作量/流量/容量→成本→执行」的证据链闭合，再增加部署与方法。

两个旧仓生产代码、new-viewer生产内核与原始报告均未在本次修改；新文件集中在本目录。new-viewer当前不是Git仓库，本次未创建远端、提交或推送。参考Gitee提交通过单独的`references/model-verse-source`只读checkout对象获取，不代表新仓已初始化Git。

本次复核与UI验收详情见[validation.md](validation.md)。待决策不影响本轮评审包交付；正式实现仍按计划分阶段开展。


第二轮 UI/数据契约复核已完成：[18 项发现与裁决](review-ui-round2/resolution.md)。DSH/ZCode 原始答复、冻结送审材料与回执均保留；计划第 8.1 节和契约第 7 节纳入修订，用户问题统一为 Q1–Q12。此记录不代表新增生产能力已实现。


2026-11-09：Q1–Q12 方向性决策已全部登记，当前实施依据为修订后的 [plan.md](plan.md)、[ui-data-contract.md](ui-data-contract.md) 和 [questions.md](questions.md)。后续优先级为 DSpark→MTP→自定义草稿→Chunked Prefill→PD 分离→Prefix Cache，平台暂不区分 CUDA/Ascend 先后；私有远端平台/地址等接入信息仍待补充。两轮原始审查材料保持原样，不代表审查者已复核本次用户决策后的改稿。
