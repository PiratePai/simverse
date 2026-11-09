# 原理底座：从语义到性能，逐层保留证据

对应 [重构计划](plan.md)。各原理对应的页面、指标、后端输入及验证步骤见 [UI 数据与验证契约](ui-data-contract.md)。这里定义未来实现应满足的契约；并不声称 new-viewer v0.1 已完成。公式只在明示假设下成立，不能按模型名字套用。

## 1. 六类量不能混算

| 量 | 例子 | 依赖 |
|---|---|---|
| 模型语义 | 第3层是全注意力；层间共享一个KV对象 | config+forward分支 |
| 逻辑工作量 | GEMM乘加、逐元素add、张量访问 | 语义、输入形状、计数约定 |
| 物理流量 | 某kernel实际HBM读、跨rank消息 | lowering、缓存、路由、packing |
| 驻留容量 | 权重、KV块、activation、workspace | 分片/复制、共享、存活区间 |
| 成本 | kernel时延、消息时延 | 设备、recipe、profile与适用域 |
| 系统指标 | TTFT、逐token间隔、吞吐、p99 | 执行依赖、资源竞争、请求过程 |

精确的逻辑算量不保证精确的HBM流量；确定的HBM字节也不保证精确时间。来自分布的期望字节可以是浮点数，绝不通过`int()`升级成精确事实。

## 2. WorkLedger：形状和运算约定先于总参数

对矩阵 `X[M,K] @ W[K,N]`，本项目约定一次乘加计2 FLOPs：`F=2MKN`；权重逻辑字节`KN*b_w`，输入/输出`MK*b_a`、`MN*b_o`。真实实现可能有padding、scale、转置、反量化、融合，另记到LoweredLedger。逻辑FLOPs、执行FLOPs分栏；稀疏峰值不能与稠密工作量混用。

Residual add有`M*d`次加法。融合后不单独产生kernel、可能减少activation物化流量，但并不会使数学加法消失。RMSNorm、softmax的除法/exp/rsqrt分别记录操作类别或明确计数约定，不用Tensor Core峰值直接解释所有逐元素运算。SwiGLU拆gate/up/down及激活，不把总参数乘2当作所有算子。

因果attention有历史前缀S和新增Q个查询时，单序列可见query-key对数为：

```text
P = S*Q + Q*(Q+1)/2
F_QK+AV = 2 * B * Hq * P * (d_qk + d_v)
```

头维相等时为`4*B*Hq*d*P`。softmax、掩码、RoPE、投影单列。S指查询前已存在的历史长度，禁止同一token在S与Q重复。滑窗、压缩、稀疏、线性attention分别生成可见关系或递推状态，不能统一用`L²`。

## 3. KV与层序：结构等价不等于实现等价

标准GQA每token每层逻辑KV元素为`2*Hkv*d`。TP恰好整除KV头时每rank可分到`Hkv/TP`；当TP大于KV头时具体复制/分片由backend决定。不能对所有KV一律除TP，也不能对所有GQA一律复制。

DeepSeek-V3的一种已知压缩MLA布局每token每层保留`kv_lora_rank + qk_rope_head_dim`元素；缓存dtype、scale、对齐另加。其prefill、展开式decode、吸收式decode的投影/乘法图不同；数学等价不意味FLOPs或访存相等。V4.1的CSA2、滑窗、压缩器、索引和跨层producer/consumer单独建图，不能继承V3公式后只改维数。

Qwen3-Next按明确layer_types/配置生成48层计划，再与forward绑定；本地已接入参考的全注意力层为0起始`3,7,...,47`，其余为Gated DeltaNet。此规则是固定模型revision的测试期望，不是对所有Next模型的默认特判。DeepSeek-V3的61层中前3层Dense、后58层MoE，同样由版本化字段和forward共同确认。

共享KV对象的**容量**按对象身份去重；各consumer的**读流量**仍分别计价。FlashAttention或MLA分块可以复用前缀，但`η=1`只是指定tile/cache条件下的乐观实现假设；CTA数量、head分组、split-K、L2重用、调度会改变HBM读放大。稀疏并集也不是完整物理读量：还包括索引检索、块取整、重复装载和预取。

V4.1还需单列Engram查表与注入、hc多流状态和混合操作。官方config前40层的r分布为`0:2 / 2:18 / 1:20`，不是38层都r2；源层与消费者的缓存大小不能套一个全局「890 B/token」。value读取被top-k封顶，不代表indexer候选扫描也封顶：当前ledger的`index_select`在非short路径按`end//r`读取索引K，仍随上下文增长。必须区分index/value/cache-state及其单位（token或block）。

## 4. MoE并集：一个期望模型，不能叫普适物理定律

E个专家，每token从中等概率选k个**不同**专家，M个token之间独立，则某专家未被任何token选择的概率是`(1-k/E)^M`，所以：

```text
E[U] = E * (1 - (1-k/E)^M)
k <= E[U] <= min(E, M*k), M>=1
```

把`M*k`次选择视为有放回独立抽签得到`E*(1-(1-1/E)^(M*k))`，是另一个近似；M=1时甚至小于k。E384/k6/M5时前者29.077034、后者28.894244，都不可能是38.9。对E4/k2/M2，小集合枚举给3，有放回公式给2.734375。这是很便宜的独立golden。

如果专家e在一个token被选中的概率是p_e且token间独立，则`Σ_e[1-(1-p_e)^M]`。真实路由的跨token相关、负载倾斜、分组约束和容量drop会改变分布；有trace时直接计算专家集合和每专家token数。

`E[权重读] ≈ W_shared + E[U]*W_expert`只表示「一次执行内各被用专家权重理想读一次」。kernel切块、重读、cache驻留、EP放置和量化元数据另算。U(M)是递增、边际递减的饱和函数，原报告「随B超线性增长」错误。实际时间可能因实现阈值出现非线性，不能由U式推出。

MoE FLOPs取决于实际token-expert assignments；容量取决于驻留的全部专家；延迟还取决于**最慢rank**而非平均路由。EP dispatch/combine的字节以路由表、远程比例、复制与协议为准，不能无条件用`2*M*d*b`。

## 5. Cost：下界、近似、拟合与实测各有身份

给定某物理kernel的工作量F、物理访问Q、对应执行单元的峰值P与带宽BW：

```text
t_compute = F / P_effective
t_memory  = Q / BW_effective
t_roofline = max(t_compute, t_memory)
```

roofline是在已计资源上的乐观约束。`max + rho*min`可作为显式经验近似，rho=0.15、效率=0.75均须登记假设与适用域，不称为硬件定律；对整模型MFU和局部kernel效率须分开。实际kernel还可有launch、同步、反量化、spill等成本。不能逐逻辑算子加一个launch，然后又按融合kernel重复加。

低比特存储不直接决定计算峰值。W4A16可能反量化到FP16/BF16执行；H100不支持某种FP4 Tensor Core路径，不意味着该kernel自动使用FP8。Recipe须指定`storage_dtype/compute_dtype/accumulator_dtype/scale_layout/kernel`并通过硬件能力检查。

`Σ_i max(C_i,M_i)`与`max(Σ_i C_i,Σ_i M_i)`不同，后者通常更乐观。选择聚合粒度需匹配融合和资源重叠，不可把逐算子结果与整步结果称同口径。

Profile成本插件输入完整shape、dtype、设备、backend与版本；输出值、覆盖域、拟合误差和外推状态。用物理模型生成3点再拟合二次式只是代理压缩，其fit_error不是真机误差。Vidur在所查版本对多个类别拟合profiling median并缓存预测；事件核不自动意味着kernel时延是随机分布采样。

公式溯源按**量**建立绑定，例如`flops_formula/weight_read_formula/kv_index_read_formula/time_formula`。字典中的可调用函数、参数引用与版本hash用于重放，UI显示符号式；一个Attention条目不能用GEMM的公式ID解释softmax、KV读与时间。单一生产维护点允许多种已命名的模型插件，不妨碍独立golden故意采用不同实现。

## 6. 容量与通信：必须进入可行性和关键路径

```text
memory(rank,t) = Σ_unique_live_allocations size(allocation,rank,t)
peak(rank) = max_t memory(rank,t)
```

对象包含权重、KV、状态缓存、activation、workspace、通信buffer、allocator预留与块取整。同卡target+draft要叠加同时驻留对象，并按identity去重真正共享的权重。事件释放只是逻辑长度回退时，不能假定物理block立刻释放；碎片与预留分别列。

点到点的最简近似：`t=latency_setup + payload/BW_effective`，但发送依赖、序列化、DMA、收端消费和链路队列均可上关键路径。collective由算法/拓扑/参与rank/消息大小选模型，TP通信不能因权重已分片而消失。不要再同时加一个含通信的「单步常数」。

跨设备投机协议必须明确各字段：candidate IDs、位置/长度、验收结果、bonus、方法需要的hidden、greedy或stochastic验证所需的概率信息。标准随机拒绝采样失败处要按`max(p-q,0)`重采样；只有候选token及其q值一般不足以恢复全vocab残差分布。可选择接收端重算、传输概率、额外拒绝轮次等协议，但必须计其成本并证明分布正确。6B/token+256B/request只能是某个已定义协议的估算。

hidden是**数据依赖**，不直接等于固定网络payload：可以在目标侧先pool/project再发送，也可以发送各层hidden在草稿侧处理；分片、压缩和算子放置都改变消息与计算。以60KB/64GB/s为例仅约0.94μs（十进制近似），不能说与100μs固定延迟同量级。大batch或更低链路带宽可能改变结论，要逐场景重算。

## 7. 投机接受：均值不是状态

定义K为提议候选数，A为接受的草稿数；在标准链式验收且未遇EOS/输出上限的轮次，N=A+1包含一次目标采样出的纠正/bonus token。每位置的条件接受率α_i给：

```text
P(A>=i) = product(alpha_j, j=1..i)
E[N] = 1 + sum(P(A>=i), i=1..K)
```

α_i都相等时才是`Σ_{i=0}^K α^i`；α=1单独返回K+1，α=0返回1。用户输入`mean`只保证一阶矩，不唯一决定P(A=K)：K2时N恒2与N1/3各半均值都是2，Pfull却是0和0.5。将mean反解几何α可以提供，但必须额外选择「截断几何假设」，不能默默改变输入含义。

标准autoregressive草稿、MTP、DSpark与DFlash不能只靠S/M/K三个字段统一。DSpark参考实现将block backbone与顺序Markov sampling分离，具有target hidden依赖和confidence截断；DeepSeek-V4.1路径还复用目标架构与checkpoint中的mtp前缀权重。层数、重复调用次数、串行采样步数是三个概念。当前新仓把DSpark等效成K次完整MoE草稿，只可保留为改名后的假想方法。

维护`RoundState=(committed_prefix, target_cache_prefix, draft_cache_prefix, speculative_slots, pending_messages, rng_state)`。transition生成下一次各forward真正需要的输入。某些简单链式实现全接受后需要处理2个token、拒绝后1个；padding/重算/EAGLE/块草稿可能不同。通用层不硬编码catch-up=2或`1+α^K`。KV回退成本取决于block管理与同步路径，不能固定1–3μs作为事实。

EOS、max_new_tokens尾轮、动态K都截断N和状态；均值round不能展示为真的「提交3.05 token」事件。可显示解析期望，或采样整数事件并报告统计分布。

## 8. 从轮次到请求，避免错用max和平均值

对合法串行协议：`T_round=T_draft+T_D2T+T_verify+T_T2D+T_transition`。平稳、可更新的长期token平均时间为`E[T_round]/E[N]`，通常不等于`E[T_round/N]`。有限请求还要计首轮、prefill、末轮截断及drain。

两个独立微批可以交错，但周期须由具体资源与反馈依赖推导，不使用通用 `max(T_draft+T_D2T,T_verify)`。对同质、确定性、固定循环的特例，可先计算每资源每完成轮的服务需求下界，以及有限在途 cohort 的循环下界；这些只是必要界，不保证存在达到该界的调度。只有阶段资源/反馈归属、在途数、launch/transition、双工链路和竞争都明确后，才可使用对应特例的 max 式，并以独立手排轨迹验证。单请求依赖仍是草稿→验证→反馈→下一轮；吞吐周期不直接等于该请求TPOT。单请求超前提议必须显式引入假设分支、hit/miss、取消、回退和浪费资源；不能靠一个β加权式假装已构造可执行schedule。

事件schema至少包含`event_id/request_id/round_id/stage_id/resource_id/dependency_ids/state_before/state_after`；非预知策略只能读取当前事件时刻已可见的信息。同一资源允许并发时须定义份额/竞争函数，无此模型按互斥处理。

指标命名：TTFT含排队和首token；每请求token间隔与稳态TPOT区别；throughput按全系统完成token/墙钟；goodput须带SLO；p50/p95/p99基于真实采样请求或事件轨迹及其样本数。随机模型假设可生成模拟分位数，但不是测量分位数。

## 9. 敏感度与比较资格

任何方案排序先检查模型/方法/负载/精度/设备预算与目标是否可比；同目标8GPU再外加1GPU的DT方案可研究，但必须同时报告资源增量。质量等价/目标分布一致与性能计数是不同验证轴。

对未测的效率、复用、协议延迟、接受分布，先扫描合理范围并展示结论反转点。所有区间必须注明是假设范围、统计置信区间还是严格界。只有上下界可证明时才称上下界；不把任意滑条范围叫可信区间。没有可行性和有效成本模型的行保留结构证据，不参与收益赢家判定。


## 10. 第二轮审查后的统计与证据边界

动态 K 的接受过程必须给联合或条件分布。边际 α_i 与 K 分布不足以确定 E[N]；没有 joint law 时返回未知，或由用户明确选择独立性等假设后输出 approximated。接受、路由、工作量和时间可能共享状态，不能只拼接相互无关的边际期望。具体反例见 [第二轮反例](review-ui-round2/counterexamples.json)。

样本结果保存统计单位、加权方式、重复次数、相关分组和 estimator。单 seed 不等于独立重复；CI 重叠或差值小于单个半宽不是通用显著性检验。边际直方图重采样不保留请求内相关性；假设模型可以产生 simulated 尾分布，但不因此成为真实 trace 重放或实测预测。

容量预算与对象账本采用同一计量边界：nominal、allocatable、allocated、reserved/active、外部占用分开，禁止预留双扣或嵌套项直接求和。MoE 有效 assignment 的逻辑工作量与含 padding/重计算的物理执行量分开；容量丢弃/重路由影响质量和比较资格。

数值语义、分支域、解析成本、事件依赖和指标归约分别需要独立证据。源码/shape 不能证明 mask 或 recurrence 的数值正确；同单位常数错误不会必然违反量纲或时间下界。事件验证从模型/协议状态另建必要依赖，并检查非预知策略的可见信息。串行阶段加法不能无条件套在并行事件上。

TTFT/ITL 明确 computed/committed/server_emitted/client_observed 的观察层；批量发出 token 允许同一时刻，不能为了曲线平滑而均摊。跨节点时间戳需要 clock domain 与同步不确定度；缺失客户端或网络模型时只报告服务端口径。完整字段与验收见 [UI 数据契约 §7](ui-data-contract.md)。
