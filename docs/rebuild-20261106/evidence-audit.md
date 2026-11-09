# 原材料复核与证据登记

日期：2026-11-06。判断基于固定本地源码、用户指定原文、官方参考实现与可复算反例。没有运行任何真实LLM/GPU forward或性能benchmark。

## 本次实做

- `py -3.12 -m pytest tests -q`：13 passed、1 warning。warning是子进程输出UTF-8解码异常；测试通过不等于无异常的整条链路，更不等于性能准确。
- [audit.py](audit.py)：只读运行接受模型、MoE公式、baseline时间线和residual计数探针；[结果](audit-results.json)保留输入、结果和来源SHA256。
- Gitee网页和raw访问失败，改用独立参考checkout的`git fetch --depth 1`和`git show`读取指定提交。正文存为[固定原文](references/model-verse-principles-pinned.md)。未将当前HEAD冒充指定版本。
- 官方HF config revision `314113f8a7dadb388215396ba5fbee2c610a04d6`，SHA256 `8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879`，与new-viewer快照逐字节一致。验证的是config身份，不是全部forward语义。

## 问题清单

| ID / 等级 | 发现 | 证据和判断 | 计划修正 |
|---|---|---|---|
| E01 P0 | DSpark被替换成K次串行MoE草稿 | `models.py:DRAFTS`自己承认真实为块结构；DeepSpec draft_ops和vLLM V4.1 dspark/speculator将backbone、Markov sampling、confidence分离 | 真实方法须完整MethodPlan；现预设改名假想方案，不能以DSpark收益发布 |
| E02 P0 | “对账完全一致”不足以确认正确 | V4.1计数从旧attention_structure忠实移植；共享同一假设的两实现会同错；tiny测试没有覆盖V4.1/DSpark | 单列公式迁移一致、独立数学锚点、forward/shape、真机验证 |
| E03 P0 | 多卡收益忽略TP collective | report §7明确承认未计；weight/TP并不消除allreduce与同步 | M2之前进入lowered graph和关键路径；旧数字仅保留诊断身份 |
| E04 P0 | 内存可行性不足 | engine检查target权重+KV；草稿只检查粗权重，未把同卡两模型、draft KV、workspace、activation和临时verify槽位合并 | allocation身份/生存期/每rank峰值；不合格行禁止赢家排序 |
| E05 P1 | MoE数字与统计定性错误 | report写38.9/384，M5×k6最多30；实际旧式28.8942、新条件式29.0770；U(M)凹而非超线性 | 区分top-k无放回与独立抽签；小集合枚举golden；期望数不标精确整数 |
| E06 P1 | 固定catch-up=2不通用 | HTML作通则；新仓只在全接受时按α^K加权，两份资料已不一致；实际proposer需依分支/缓存策略核对 | RoundState推导真实输入；支持按hist/trace枚举 |
| E07 P1 | committed_mean不能决定回退概率 | K2，均值2的两分布Pfull可为0或0.5；resolve_alpha强行假设几何，mean=K+1还只返回0.999999 | mean模式不推Pfull；可选明确分布假设，端点单独处理 |
| E08 P1 | baseline视图出现草稿和verify | 审计运行mode=baseline仍生成draft_step/verify事件，返回status=ok | baseline独立执行图；所有UI事件来自所选方案并校验 |
| E09 P1 | 算量与实现融合混淆 | ledger residual FLOPs=0；4个d256 residual逻辑加法应为1024；scope中逐行physical/expected含义混杂 | 逻辑操作保持，融合改物化流量与kernel数；结果统计类型分栏 |
| E10 P1 | W4不能自动走FP8算力档 | engine由位数选择peak，recipe缺反量化与实际compute dtype | storage/compute/accumulator/kernel capability分别绑定 |
| E11 P1 | KV只读一次被写成通则 | HTML ηdense=1；真实HBM访问取决于CTA/tile、head分组、split、cache | 区分逻辑访问/去重集合/物理读；把η放在recipe假设里 |
| E12 P1 | DT协议过于简化 | 6B候选与256B控制不保证随机拒绝采样、hidden依赖和vocab映射成本完整；vLLM DraftModelProposer实际区分hidden需求并校验词表/TP | protocol schema+正确性证明+消息量与额外计算 |
| E13 P1 | 理想pipeline周期与请求延迟混淆 | 新仓有optimistic标注，值得保留；但图从0并行且无request/round身份，不足以证明因果合法 | 不同微批ID、warmup/drain、单请求回退分支；吞吐周期与TPOT分列 |
| E14 P1 | 旧原理文档算术与校准名称误导 | 512/top10/100tokens公式覆盖85.844%，不是18%；自生成3点拟合只能说明代理逼近自身 | 修正文档；T0代理和实测校准分开，收益比不先验更可信 |
| E15 P2 | “全部整数”“精确浮点”表述过度 | 期望MoE字节int化；pytest.approx(abs=0)仍有默认rel容差，不能称逐位相等 | 按字段定义精度契约；报告由运行产物生成 |
| E16 P2 | 可重放身份不完整 | digest仅截断16hex；scenario/model/device未覆盖formula/recipe/code，breakeven无解返回NaN | 完整hash/算法版本/代码版本；null+reason标准JSON |
| E17 P2 | 报告自身有陈旧数字 | verify行数在827/828两处不一致；说四部署ctx全盈利与dt_serial主场景0.909不一致 | 表格从产物自动生成，叙述加适用条件；不擅自替原作者改原报告 |
| E18 P0 | V4.1配置/forward覆盖不完整 | DSH发现Engram/hc遗漏；主控补读vLLM nvidia/model.py确认Engram注入、hc_dim/mhc混合真实参与forward。实际前40层r分布2/18/20，报告写2+38r2错误 | 配置键覆盖表与语义边闸门；缺失结构污染容量与性能，不宣称完整模型ok |
| E19 P1 | “top-k封顶所以ctx只抬容量”忽略indexer | 主控读ledger.py确认非short路径index_select.kv_read使用end//r，候选索引仍随ctx增长 | index/value读分账，具体CSA2候选剪枝路径另按官方实现核对 |
| E20 P1 | 扫描修改全局注册表 | DSH指出；主控读sweep.py确认DEV.LINKS/DEV.DEVICES保存-修改-恢复，try/finally不提供并发隔离 | 依赖注入不可变设备/链路快照，增加并发隔离验证 |
| E21 P2 | CSV测试名义与实际覆盖不符 | export读outputs存档，仅检查header和行数；其他测试覆盖部分fresh数据，不足以证明CSV逐列等值 | fresh RunResult→CSV逐字段核对，stale存档独立重放测试 |

## 成熟实现参考与取舍

| 官方项目 / 固定版本 | 实际阅读 | 借鉴 | 不直接照搬 |
|---|---|---|---|
| [Vidur](https://github.com/microsoft/vidur/tree/abae7f63aa857300f5cdc6f5e0d27860cd24721b)，最近提交2026-08-24 | sklearn_execution_time_predictor.py、simulator.py、batch_stage_end_event.py；读取median字段、拟合/预测缓存、heap事件循环 | profiler成本插件与事件核分离；通信和CPU开销有类别；事件trace | 不引入全套依赖；此实现并非「每个kernel采样时延分布」的依据，源码不证明本任务新模型被覆盖 |
| [vLLM](https://github.com/vllm-project/vllm/tree/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5)，2026-11-06 | DraftModelProposer、llm_base_proposer、V4.1 DSpark、DSpark speculator；draft_model correctness与DSpark测试正文 | 验收/缓存/词表/TP约束、算法与执行形状分离、correctness测试矩阵 | 不声称新仓1:1复刻vLLM；当前独立草稿路径同TP约束不等于所有研究部署都不允许异TP |
| [DeepSpec](https://github.com/deepseek-ai/DeepSpec/tree/005e03b81cec38b7da6399833d609ee89a2587f2)，2026-07-09 | draft_ops.py、evaluator.py、Qwen3DSpark modeling的backbone和采样实现 | DSpark的块、Markov、confidence语义；draft_probs作为验收输入 | Qwen/Gemma例不能自动推出V4.1结构；另外核对vLLM V4.1路径；不凭项目名写统一预设 |
| [DeepSeek V4.1 HF config](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/config.json) | 固定revision config及字节hash比对 | 模型身份和配置字段 | config不能独立证明kernel、通信、位宽布局与模型行为 |

维护状态仅由本次查询的提交日期和可读源码描述，不把星数或README视为实现验证。所查源代码快照保存在references，完整仓库清单/commit记录也在其中。上游GPU测试只读，未执行；Vidur未运行。当前四旧模型的历史对账也未在本轮重新跑。

## 对原计划自身的收窄

- “语义唯一来源”不意味着只能有一种cost model；独立oracle刻意保留另一套推导，但不能被生产路径调用。
- “前端零物理公式”允许单位换算、坐标投影、排序和显示delta；禁止的是再次实现仿真业务规则。
- vLLM是执行参考之一，不能成为所有新架构的设计限制；不支持的研究方案可以有自定义recipe，但身份必须明确。
- 不把小样例通过推广为完整模型正确；不把同卡/DT收益比较忽略新增设备资源；不把T0区间包装成统计置信区间。
