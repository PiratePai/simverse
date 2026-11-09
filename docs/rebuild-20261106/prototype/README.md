# UI 工作台示例

主入口：[index.html](index.html)。直接复用用户 `simverse_v3_0.html` 的工作台 DOM、样式和交互骨架，保留高密度信息与联动。无需安装前端依赖，可直接打开。

保留场景栏、Target/Draft 配方、七项摘要、5×5 部署方法矩阵、角色算子分组、七维检查器、Roofline、三轨时间轴与阶段筛选。新增证据提示、场景 JSON 导出、键盘操作和移动布局。

**主工作台中的数字是原版交互展示数据，不是仿真结果或已验证预测。** `v3-workbench.js` 保留的计算函数只服务 UI 演示，不能迁入生产成本模型。流水模式不展示未经验证的加速比与并行条；显存保持待核实。详细对应关系见 [reuse-map.md](reuse-map.md)。

[reference-v3.html](reference-v3.html) 是用户原文件的完整存档，含原有审查、计划、原理页面。存档内容保留原貌，不代表本次认可其中结论；结论以评审包的证据审计和原理文档为准。

原来的独立教学示例移至 [teaching.html](teaching.html)，其计算与说明如下。

---

# 可交互工作台样例

打开[teaching.html](teaching.html)，无需安装前端依赖；也可在本目录运行 `py -3.12 -m http.server 8767 --bind 127.0.0.1`。

支持：选择教学场景的K/α后载入离线结果；方案切换/固定基线；逐层算子与公式/字节/Roofline联动；点击时间线定位Attention；假设说明；JSON导出；三个真实模型的结构/缺口展示；手机与桌面布局。

**数据身份**：教学GQA计算与设备预算由[build_fixture.py](build_fixture.py)离线生成。它是独立的UI契约样例，不是new-viewer生产内核，不用于真实模型/硬件预测。调用 `py -3.12 build_fixture.py` 生成fixture.json/fixture.js，同时执行5项算术及不变量断言。浏览器只有显示/选择/坐标转换与比较比值，没有生产物理规则。

教学算例：2层目标d256、1层草稿d128；未融合FP16；虚拟矩阵2TF/s、逐元素0.2TF/s、带宽20GB/s、kernel启动4μs；DT每方向20μs的人工预算。逐token的接受率相等，按特定简单链式cache政策做catch-up加权。显存容量未建模，不据此判定可部署，也不展示“最佳方案”。

Qwen3-Next和DeepSeek-V3层序来自历史已接入参考，不是本轮新跑的forward证明。V4.1保留结构观察和当前缺口，不提供精确性能数字。两微批理想周期单列，B1教学场景不画虚假并行Gantt。

设计沿用用户HTML的蓝/青/浅色工作台基础；把配置、比较、结构、检查器与执行预算组织在一个页面。使用frontend-design与impeccable的Operate质量要求：低装饰、清晰层级、可见焦点、真实状态、可折叠证据。最终产品仍需根据正式ResultBundle接入事件ID/依赖箭头、完整内存账本和后端运行状态；此样例不实现这些能力。
