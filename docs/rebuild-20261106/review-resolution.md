# DSH / ZCode独立质询与主控裁决

日期：2026-11-06。实际调用DeepSeek Harness和ZCode，未用Codex子代理代替。通过用户指定`D:/Github/agent-collab`的既有原生适配器，分别使用read-only与guarded模式；未改原生持久凭据、模型或权限设置。

原始答复保留：[DSH 12问](dsh-questions.md)、[ZCode 12问](zcode-questions.md)；共同输入是[送审包](review-packet.md)。回执：[DSH](dsh-receipt.json)、[ZCode](zcode-receipt.json)。两位的提问是审查线索，结论均由主控再次判断。

DSH首次因Windows目录分隔符的字符串比对失败，在提交结果前结束；使用规范化绝对路径修正启动器后完成，失败回执亦保留。两次有效审查并行约10分钟。未做同条件单人对照，不声称多Agent提速比或成本收益。审查者未运行GPU；DSH声明其shell输出异常，因此其数值必须由主控复算。

## 质询如何改变计划

| 审查问题 | 主控裁决与独立核查 | 已落地位置 |
|---|---|---|
| D-P0-1 / Z-P0-1：DSpark究竟是什么 | 采纳主问题。官方块backbone、Markov/confidence与target hidden依赖均存在；另外补读V4.1专用代码，避免只用Qwen DSpark推断V4.1 | plan §3/5，principles §7；当前代理不得作为真实DSpark发布 |
| D-P0-2 / Z-P0-3：B1的max流水是否可执行 | 采纳。理想服务周期可作为有条件界，不能直接代表当前B1请求TPOT；若引入预演，必须显式建分支 | plan M3，principles §8；UI独立行、不参与加速比基线，无伪并行图 |
| D-P0-3 / Z-P0-4：V4.1结构证据足够吗 | 采纳并扩大。config字节身份已核；DSH指出Engram/hc，主控补读forward确认参与运算；目前是partial | plan M0字段覆盖闸门，E18，UI真实模型展示缺口 |
| D-P0-4 / Z-P0-2：MoE公式、38.9和int | 采纳。38.9是报告错误，代码为28.8942；top-k条件下29.0770；int期望与公式重复是真实设计问题 | audit.py可复算，principles §4，逐量统计类型 |
| D-P0-5：异TP/协议可行性 | 采纳。当前vLLM DraftModelProposer路径要求相同TP，独立server/自定义runtime另论；hidden传输取决于图切分 | plan M2协议先行，principles §6，questions Q7 |
| D-P0-6 / Z-P1-8：catch-up与mean | 采纳。简单chain策略可用1+Pfull；不能泛化，均值不足以确定Pfull | principles §7，独立分布反例与端点探针 |
| D-P1-7 / Z-P1-5：KV/稀疏与上下文 | 采纳。主控确认index扫描随ctx增长；η=1不能作通则；890B单位尚未统一 | E11/E19，principles §3，按index/value/state分账 |
| D-P1-8：公式ID不对应实际字段 | 采纳。Attention挂gemm_flops不足；同时绑定FLOPs、各字节、成本函数 | plan §4去重引用字典，principles §5 |
| D-P1-9：共享ModelIR只是口号 | 采纳现状批评。新仓当前未消费共享包，计划明确这是待迁移任务；共享包唯一发布源，新仓贡献适配器 | plan §3/7；禁止另建永久语义实现 |
| D-P1-10 / Z-P1-6：OOM闭合 | 采纳。target/draft分role、同卡共享和临时内存生存期进入能力闸门 | plan M2，principles §6；UI峰值缺失不填0 |
| Z-P1-7：launch缺失 | 采纳缺口；不采纳固定“kernel数×5μs”作为有保证的上下界，需已知kernel/graph/host重叠 | M2 recipe与profile；教学UI明确4μs是假设 |
| D-P1-11 / Z-P1-9：Vidur描述不准确 | 采纳。所查实现按median回归与缓存；主控读训练/预测和事件循环 | evidence参考表，principles §5；要求独立holdout |
| D-P2-12：block单位/硬件/全局状态 | 候选block语义进入阻断清单；扫描全局突变已源码确认；L40具体规格数字待硬件阶段官方逐精度复核，本轮不采信建议替换值 | E19/E20、M0字段语义、M2设备口径登记 |
| Z-P2-10：旧文档勘误 | 采纳。保留固定原文，另列勘误；不修改历史提交证据 | E14与算术probe |
| Z-P2-11及D评论4：元数据重复负担 | 采纳。关键量保留类型，公共元数据字典化、按stage/recipe继承；不是2493行重复自由文本 | plan §4；不混淆support与exact/expected |
| Z-P2-12及D评论3：UI复杂度/诚实性 | 采纳。正式M1先两方案diff和检查器；本轮工作台示例使用明确教学数据，真实模型结果缺口可见 | prototype，plan §8与M2内关口说明 |
| D-P2-12：导出覆盖 | 采纳。主控查看CSV测试，只比存档导出行数；必须补fresh逐字段验收 | E21与plan §6 |

## 未采纳或纠正的审查意见

1. **ZCode称V4.1“未发布、无官方forward”不成立。** 主控已从官方HF获取固定config，并读取固定vLLM V4.1实现。正确结论是本地仿真覆盖不足，并非外部实现不存在。第三方vLLM实现也不自动等于模型作者全量行为证明。
2. **两位用Qwen/Gemma DSpark推出V4.1的dense/B1限制过度。** DeepSpec那条evaluator有B1断言，但vLLM V4.1专用实现复用V4 decoder，含MoE、hc；DSparkSpeculator按num_reqs批处理。限制应绑定具体实现/版本，而非算法名称。
3. **hidden依赖不等于必须原样回传所有层hidden。** V4.1 speculator存combined main_x，合并/投影可放在不同角色。必须确定图切分才能算消息；不接受统一的`n×层数×hidden×2B`作为所有实现的必然payload。
4. **ZCode的60KB链路量级错误。** 60KB/64GB/s约0.94μs，约为100μs固定延迟的1/100；不能据此宣布B1链路变成带宽瓶颈。DSH大batch敏感性思路合理，但仍需实际payload。
5. **ZCode的同均值反例写错了Pfull。** N恒2时Pfull=0，N1/3各半时0.5；0.381966是强加几何分布反解后的Pfull，正好说明该假设不能默默引入。
6. **38.9不等于实现真的算出38.9。** 计划和审计明确分开报告誊写错误、统计模型选择和int期望问题。两个公式在当前大E只差0.6%左右，不代表小E/倾斜路由下都可忽略。
7. **43个compress_ratios而40主干层不必然是config错误。** V4.1 DSpark专用代码以`num_hidden_layers+i`构造3个草稿层；应明确主干/草稿分段并审计消费，而非强制数组长度=40。前40层2/18/20的纠错已经独立确认。
8. **Engram内存不能直接采纳24.6/49GB增量。** 缺失Engram真实影响已确认，但哈希布局、压缩表、dtype、TP/DP、CPU offload/共享与生命周期尚需完整核对；保留容量unknown，不能凭两表尺寸直接判定HBM OOM。
9. **统计量与支持性不能揉成一个枚举。** DSH示例把exact_int/expected/unsupported写在support，主控保持support、statistic、evidence、feasibility四个正交维度。ZCode称逐行元数据“指数膨胀”也不准确，是线性重复；去重建议仍采纳。
10. **不把MVP拖成全面重实现。** 接受DSH提出的协议先行依赖，不接受必须等所有新模型完备后才能做任何成本/UI的串行排期。tiny/成熟结构可以先形成可信闭环，V4.1部分图和阻断原因并行展示。
11. **不将科学真实性交给用户投票。** 两位提出“是否必须把proxy叫DSpark”的问题不应上抛；命名必须诚实。只保留用户真正要决定的实验、数据、资源、部署边界与发布归属。

## 尚未解决的技术事项

V4.1 Engram/HC完整lowering与容量；CSA2候选block语义与物理KV读；特定DSpark版本的完整协议/状态机；量化kernel规格和真实硬件profile。这些已进入M0–M2工作项。本次审查完成不代表这些实现工作已经完成。


## 后续第二轮复核

本文件保留首轮裁决。更新后的计划/UI 数据契约另经 DSH 与 ZCode 只读复核，原始 18 项发现、回执、主控修正及新计划落点见 [第二轮裁决](review-ui-round2/resolution.md)，不混写两轮输入和验证范围。
