# R05：V4.1 mHC / Sinkhorn kernel 计数修正

日期：2026-11-12。源码阅读、解析账本 replay 与 stdlib 独立数学测试；
**未执行任何 pinned kernel、模型 CPU forward 或 GPU 测量**。

## 1. 范围、身份与参考实现

采用官方 [vLLM 固定提交](https://github.com/vllm-project/vllm/tree/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5)
`f358f0578a83f4e1fe6fc49a492d3ae9362de8e5` 的实现；这是本仓已有模型身份所绑定的
参考版本，本轮不声称它是当前 main。复用其中 delayed pre、carried pre_mix、
comb 矩阵与边界所有权机制，不引入 vLLM/TileLang/Torch 运行时依赖。
源码均来自本地评审包，未把源码阅读描述为运行验证。

14 个新增文件逐文件计算 `SHA1(b"blob "+len+b"\0"+data)`，与既有
`references/vllm-tree.json` 完整 blob SHA1 相等。机器清单为
[mhc-pin.json](../rebuild-20261106/references/vllm/mhc-pin.json)，包含完整 blob SHA1、
SHA256 与 size。测试独立固定 14 文件路径集合，防清单删项/内容漂移。
任务书 §0 所说 17 文件包含先前 R04 的 3 文件，本轮新增 pin 实为表列的 14 文件。

| File (relative to references/vllm/) | Bytes | Git blob SHA1 | SHA256 |
|---|---:|---|---|
| `vllm/model_executor/kernels/mhc/__init__.py` | 926 | `6fc49199891eb3e4a61f406946e4e561775764b1` | `b1aac23563953599c1dabe33e38d6a4c17cb72653409850854324c233ee8f5ab` |
| `vllm/model_executor/kernels/mhc/aiter.py` | 12594 | `f102143ab08fa11c5c31f765a4c7834243cef13e` | `5cd6198258aec24bcd402a69ceefdb4267dc2f8bd680e9f0a243d8bc7d37f233` |
| `vllm/model_executor/kernels/mhc/cpu.py` | 3345 | `d12b0eae1851da6c43372b6d99b57126550ce7f0` | `7245039d88c8e34f747e410428086466b2ce0c5c094bde71d832454992905a1b` |
| `vllm/model_executor/kernels/mhc/tilelang.py` | 37600 | `5a291b9c320feca2e2b23bb170f52b3d571c9865` | `cf75da24521424fbe07ff8228e18c8272b7878bd80bc2dcefc1d99a598d117fb` |
| `vllm/model_executor/kernels/mhc/tilelang_kernels.py` | 69229 | `d79f7f42ab6e8f0c23c5230fb61216c149f52fb8` | `1cf58ea6d51dd56f886b0a9571dc0ad9d75a0a908fd68b1c2d6d28234bd13d55` |
| `vllm/model_executor/kernels/mhc/torch.py` | 5356 | `eb4de35e7c752ffd858b72724608dabc78080668` | `3697085aa6f0070e165c145f9c337ca3fe0623623c8265d29e50c43467e49985` |
| `vllm/model_executor/kernels/mhc/triton.py` | 15479 | `0e6c47b52dfae6cc45db2655f69da865dc17a466` | `8f61199e55adf03d8072b7e51f76daf584a89ba0e7d6312fda254668f7f2370c` |
| `vllm/model_executor/kernels/mhc/warmup.py` | 3546 | `deb1cf429a0c14967659ea0a5bc35d8324906980` | `feee09b7e0444e24b53c9619d7621467848b707b97b7ad81ff42b7824803a38c` |
| `vllm/model_executor/layers/mhc.py` | 30990 | `19e8b427b494bb28876a811e456673aa5a5b0d37` | `433d750a4d669ad468679b1faf0a3f1e846410198fb4a1eeee2d14cfe55e6a55` |
| `vllm/models/deepseek_v41/amd/mhc.py` | 4434 | `bee13f5d502325c54659d94085053fab66665c81` | `9be806b5fb84bee3ce3f874e118a9697e3b788e06f9ac5b91a48e4c90060cdfa` |
| `vllm/models/deepseek_v41/nvidia/ops/cute_dsl/all_reduce_mhc.py` | 76037 | `166efc56ff7686f5797da8737d73877be433b303` | `9203d5c62634a4858e28350f20599dc29964eab901e771185fd5fd321a2469d7` |
| `vllm/models/deepseek_v41/nvidia/ops/mega_mhc.py` | 3022 | `bdb5b95bbeb94d5f8dd4c45c1d7ff450f4ccd8b4` | `4b4af8b71c20d929303b15bf770eeff25a875046f08dc698ca8c47bb51b4fb9c` |
| `vllm/models/deepseek_v41/nvidia/ops/mhc.py` | 11647 | `a2207d1c4d7051c1ff11e55ae31a6d53d651a74b` | `e7c3f0ae3cbfe54f41ad2718600b47906571008719fb9da1de86cd7da97636d5` |
| `vllm/models/deepseek_v41/nvidia/ops/cute_dsl/__init__.py` | 315 | `03d8a11e0bc205ef5b8cc48648fd000af8db5981` | `a70e7952fea2ea51d1165caa7b2a723eed03f0446a58bd0b60ce7310ca300551` |

证据 `ev-vllm-v41-mhc` 绑定该 SHA256 清单，`kind=source-checked`、`executed=False`；
ModelIR checked_paths 新增 `sm-v41-mhc-kernels`，运行 evidence_refs 包含该证据。
Engram 函数、公式、分配、来源与测试文件保持原样；final_norm / lm_head 原行保持原样。

## 2. 任务书 §1 逐项复验与分支映射

下表路径相对 `references/vllm/vllm/`；`K` 代表 `model_executor/kernels/mhc/`，
`N` 代表 `models/deepseek_v41/nvidia/`。各项均为 source-checked，未运行分支。

| 任务书事实 | pinned 位置 | 复验结果 / 本仓归属 |
|---|---|---|
| 每层 attn/ffn 两边界，shifted post/pre | N/ops/mhc.py L192-323；N/model.py decoder forward | 保留每层三行×两边界，post 属于前一子层、pre 属于当前侧 |
| 默认 TileLang | N/ops/mhc.py L257-323；K/tilelang.py L256-484 | stream/mega/fused-AR 路由之后才落此分支；不从场景位宽推断真实后端 |
| fused ≤32 token、h%128==0 | K/tilelang_kernels.py L57-115、L692-810 | 一般配置成立；另有 SM100+、T≤4、h=4096、mix%2==0 的 single-pass 配置 |
| unfused post + GEMM | K/tilelang_kernels.py L813-863、L866-1042 | 本轮字节采用此逻辑参考配方；fused 省 residual 二次读但也有 tile 冗余计算，未作物理执行计数 |
| deep_gemm split-k / fallback | K/tilelang.py L151-180、L389-416；K/warmup.py L18-21 | unfused deep_gemm bucket {1,4,16}，fallback=1；fused 自选 split，见 §7 |
| prenorm → gates → Sinkhorn → collapse/norm | K/tilelang_kernels.py L196-265、L383-524 | with_norm 模型路径，save_pre_mix=True；NaN RMS L428-440 会跳过剩余迭代，账本限定有限输入 |
| overlap | N/ops/mhc.py L45-57、L99-189、L243-280；N/model.py L523-534 | NVIDIA ≤16 token 且需 SM100 family/deep_gemm/shape/非 ubatching，还限 FULL graph capture、非 piecewise；未建模流水与额外执行 |
| mega | N/ops/mega_mhc.py L16-44 | capability family100（不是仅名为 GB100）、h>0 且%1024、m=4、deep_gemm 符号可用、pre_mix/norm 非空、!capture_aux、T≤2²⁰ |
| fused all-reduce | N/ops/mhc.py L60-73；N/ops/cute_dsl/all_reduce_mhc.py L1185-1766 | TP4、非 EP、h5120/m4 且 MNNVL mailbox；仍未建模该部署配方 |
| ROCm / CPU / XPU | model_executor/layers/mhc.py L16-94、L346-553；K/aiter.py；K/cpu.py；models/deepseek_v41/amd/mhc.py L16-26 | gfx942 禁用 TileLang fallback torch/triton，gfx950 可用 aiter；ROCm overlap 阈值128；CPU/XPU 独立分派，非本轮支持声明 |
| 首层特殊 | N/ops/mhc.py L326-355；K/tilelang.py L60-213；K/tilelang_kernels.py L447-450；N/model.py L1530-1543 | expand 为拷贝；IC=h/rms_numel=h；one-hot collapse 仍 m×h FMA；无 post，无 pre_mix_in 张量读 |
| 派生 broadcast | N/model.py L1530-1543 | hc_attn_fn.view(mix,m,h).sum(1)，FP32 mix×h，finalize 生成；常驻缓冲非新增独立参数 |
| 末层出口 | N/model.py L1259-1279、L1187-1192；K/triton.py L76-147 | 末层 post → pre_mix collapse → 既有 final_norm；V4.1 无 learned hc_head |
| capture_aux | K/tilelang.py L303-306、L440-460；K/tilelang_kernels.py L472-488；N/model.py aux 配置 | speculative 消费的流均值；本轮 baseline 不启用，额外工作不入本轮公式 |
| 常数 | pinned config + N/model.py hc 参数 | h=5120，m=4，mix=24，mh=20480，iters=20，post_alpha=2 |

## 3. 精确算术结构与近似惯例分列

本仓 `flops` 字段兼容记录算术 op。FMA=2，sigmoid/exp/rsqrt、max 比较各算 1，
按归约 n−1 个加法、每轴分母复用一次 eps 计；不宣称硬件指令数或 Tensor Core FLOPs。
初始化常量、地址、cast、warp/padding、tile 重算不统一纳入此语义账本。

| 项目 | 每 token 公式 | h5120/m4/iters20 | 性质 |
|---|---|---:|---|
| GEMM | 2·IC·mix，MAC=IC·mix | 普通983040；首层245760 | 维度与 MAC 精确 |
| sqrsum | 2·IC | 普通40960；首层10240 | 零累加乘加约定 |
| prenorm | mix+3（split=1）；split=s 另加(s−1)(mix+1) | 27 | 算术 op 公式精确；rsqrt=1；非运行时 split 声明 |
| gates + comb 初始化 | 8m+2m² | 64 | 算术 op 公式精确；sigmoid=1 |
| softmax | 2m(m−1)+4m² | 88 | 行 max、减、exp、sum、除、eps |
| 列归一 | m(m−1)+m+m² | 32 | 分母 eps 按列复用 |
| 余下迭代 | 2(iters−1)[m(m−1)+m+m²] | 1216 | 行后列；iters=1 无此项 |
| gates+Sinkhorn 合计 | (4·iters+6)m²+6m | **1400** | `hc_*_sinkhorn.statistic=exact`，在上述 op 口径内 |
| post | mh+2m²h | **184320** | post*x 初始化 + comb*res 累加，前一子层所有权 |
| collapse | 2mh | 40960 | m×h 零累加 FMA，首层 one-hot 不减操作 |
| 融合 RMSNorm | **4h+2** | 20482 | **近似惯例**；保留 _norm_row/G11，不宣称 kernel 精确 |
| 首层 attn post_pre | 2mh+4h+2 | 61442 | 无 post；含近似 norm，整行 statistic=approximated |
| 普通 post_pre | mh+2m²h+2mh+4h+2 | 245762 | 同上；post/collapse 精确部分与 norm 近似分开列出 |
| exit post | mh+2m²h | 184320 | layer=-1，仅 post |
| exit collapse | **2mh** | **40960** | layer=-1，从零 acc+=pre*x；数学最简 (2m−1)h=35840 另列，见 §7 |
| expand | **0** | 0 | 拷贝仅字节，无浮点运算 |

Sinkhorn 独立事件计数 oracle 不导入生产公式；(m,iters) 五锚点依次为
(2,1)=52、(2,3)=84、(3,2)=144、(4,1)=184、(4,20)=1400。
stdlib 数值 oracle 对非对称 4×4 logits、eps=0.01、iters=1/2/20，
分别用 kernel 逐轴写法和 torch.py L119-123 的矩阵/转置写法对照；故意颠倒行列的
负例与结果不符。这是独立数学复现，未导入/运行 torch 或 pinned kernel。

小尺寸 m2/h8/iters3/T5：attn mix=775，ffn mix=1495，两个 sinkhorn 各420；
首层 attn post_pre=330，ffn post_pre=730，expand=0。真实40层 mHC 合计
100,968,080 op，243行（240边界+1 expand+2 exit），每层原6行结构不变。

## 4. 字节映射与容量

定义 D=mh、C=mix、b=act/8。下表为每 token（GEMM fn 权重整批逻辑读一次，另述）。
P 表示有前一子层 post（首层 attn P=0，其他 P=1）。
act 为非参数张量；参数读放 weight_read；新旧 pre_mix 是 carried buffer，非参数。

| 行 | act_read / token | act_write / token | 参数逻辑读 |
|---|---|---|---|
| hc_side_mix | IC·b | 4(C+1)（mixes+sqrsum） | C·IC·4 / 整批；首层 IC=h，其余 IC=D |
| hc_side_sinkhorn | 4(C+1) | 4C（post m、comb m²、pre_out m） | T·4(C+3)（base+scale）；容量仅4(C+3) |
| hc_side_post_pre | P[4(m²+m)+(D+h)b] + D·b + P·4m | P·D·b + h·b | T·h·b（norm_weight，每 token 读） |
| hc_expand | h·b | D·b | 0 |
| hc_exit_post | 4(m²+m)+(D+h)b | D·b | 0 |
| hc_exit_collapse | D·b+4m | h·b | 0 |

post_pre 中 P 项分别是 post 的输入与输出；D·b 是 epilogue residual_cur 读；
P·4m 是 carried pre_mix_in。首层 one-hot 使用常量，不读 pre_mix_in。
GEMM mixes/sqrsum 写由 mix 拥有、epilogue 读由 sinkhorn 拥有；epilogue residual/
pre_mix/norm/layer_input 由 post_pre 拥有。没有虚构 mixes 归一中间 HBM 物化。
逐行相加正好覆盖任务书 §1.3 的 post、GEMM、epilogue 清单；norm_weight 计入参数列。
测试 T5、act8/16、首层/非首层四组合独立按张量清单求和，并分别断言读写及参数读守恒。
相比任务书清单，另补源码明确读取的 FP32 base/scale（L395-410）；与 residual
流量分列，不混作 activation。GEMM fn 的整批一次读是逻辑权重读惯例，非实际缓存命中承诺。

字节数是给定 shape/dtype 的逻辑参考清单精确整数；**不是 HBM 访问测量**：
真实 kernel 激活/norm 固定 BF16，默认场景 act8 是仓库可选计价/存储假设；
fused 省读、split 中间张量、tile 内重复 coefficient/weight 读、cache、padding
未建模。不得把默认 B1/act8 结果称为 vLLM GPU 真实流量。

容量：原 fn/base/scale 仍每卡全量复制，2(C·D·4+C·4+3·4)/层 = 3,932,376 B。
count 的 fn 容量加 sinkhorn base/scale 容量逐层等于原 memory 对象。
L0 新增 `target-w-mhc-broadcast-L0` = C·h·4 = **491,520 B**，FP32、常驻、
TP1/2/4/8 均不切分；live_note 标明 finalize 时由 hc_attn_fn 派生，非独立参数。
首层 mix 的 weight_read=C·h·4，而 weight_capacity 仍 C·D·4。

## 5. 改动前基线与前后归因

先于所有生产代码/公式修改捕获：
[before.json](../evidence/r05-mhc/before.json)、
[scenario.json](../evidence/r05-mhc/scenario.json)、
[baseline-provenance.json](../evidence/r05-mhc/baseline-provenance.json)。
基线代码 HEAD `aa2467e`（完整值见 provenance）；捕获时仅任务书预置源码 untracked。
before.json SHA256 `740afbbe4e69a33fb3014c4be698bd9c4998c22432850b0fd84c49811c4efdbb`。
该目录 JSON 作为原始快照在 .gitattributes 标记 binary，防 Git 行尾转换改变捕获字节；
已核对暂存区 before 与原文件逐字节一致，14 个暂存源码 blob 也与上游 tree 精确一致。

场景完全一致：deepseek_v41_flash / baseline / H100 SXM / TP8 / EP1 / B1 /
ctx32768，默认 dense_w8/expert_w4/act8/kv8/state32，Engram cpu_offload。
用 `py -3.12 tools/replay_v41_mhc.py` 重放新账本，生成
[after.json](../evidence/r05-mhc/after.json) 与
[attribution.json](../evidence/r05-mhc/attribution.json)。工具断言：仅 hc_* 行发生数值
变化；按 op 的 Δ 求和等于各总量 Δ；各行 roofline 时间 Δ 求和等于 M01 Δ。

| 量（per-rank target_decode） | 改动前 | 改动后 | Δ | 原因 |
|---|---:|---:|---:|---|
| 行数 | 916 | 918 | +2 | exit post/collapse；每层6行不变 |
| flops 字段（算术 op） | 7,146,532,702 | 7,161,190,222 | +14,657,520 | 下表分解 |
| weight_read B | 1,551,736,064 | 1,550,270,144 | −1,465,920 | L0 broadcast −1,474,560；80边界 base/scale +8,640 |
| weight_read_expected B | 530,843,184 | 530,843,184 | 0 | MoE/Engram 期望读未改 |
| act_read B | 7,419,934 | 9,086,814 | +1,666,880 | attn post_pre +797344、ffn +823040、sqrsum epilogue +320、exit +46176 |
| act_write B | 4,817,888 | 4,823,328 | +5,440 | sqrsum +320、首层不再伪写 post residual −20480、exit +25600 |
| kv_read / kv_write B | 39,858,816 / 33,408 | 同前 | 0 | mHC 不改 KV |
| state_read / state_write B | 0 / 0 | 同前 | 0 | 不涉及递推状态 |
| GPU 侧解析耗时下界 ms | 116.152651919 | 116.152715012 | +0.000063093 | 既有逐行 T0 roofline 重价；非实测/完整 TPOT |
| steady B | 35,662,849,472 | 35,663,340,992 | +491,520 | 仅派生 broadcast 常驻缓冲 |
| transient peak B | 3,072 | 3,072 | 0 | R04 staged 下界不变；新 kernel workspace 未估填 |
| known_peak B | 35,662,852,544 | 35,663,344,064 | +491,520 | steady+原暂存；预算 unknown/peak_complete=False 不变 |
| 通信 | allreduce80 + logits1 + Engram gathers4 | 同前 | 0 | 未引入部署融合/overlap 假设 |

| op 组 | flops Δ | 推导 |
|---|---:|---|
| hc_attn_mix | +1,608,760 | 首层10240 + 39×40960 + 40×27 |
| hc_ffn_mix | +1,639,480 | 40×(40960+27) |
| 两侧 sinkhorn | −80,000 | 80×(1400−2400) |
| hc_attn_post_pre | +5,550,080 | 39×(184320−40960) − 首层伪post40960 |
| hc_ffn_post_pre | +5,734,400 | 40×(184320−40960) |
| hc_expand | −20,480 | 移除拷贝误计 FLOPs |
| hc_exit_post | +184,320 | 补齐末层 post |
| hc_exit_collapse | +40,960 | 从零累加源码 op；数学最简另列 |
| 合计 | **+14,657,520** | 上列求和 |

norm 4h+2 没有变，全体非 hc 行（包括7条 Engram 行、final_norm/head/embed）
与 before 逐字段相等。G7 校准不改账本、G11 原有 o_a/norm oracle 语义和测试均未修改，
全量验收覆盖；本次整模型汇总变化来自上述底层事实修正。
M01 仍受 Engram host 查表/预取时延未知约束，仅 GPU 侧解析下界。
参数复制、未知预算/运行时池缺口与排名资格没有解除。
after.json 在本地提交前的工作区生成，其 manifest.code_revision 为当时 HEAD，
并不表示 after 来自未修改的 aa2467e；本轮最终本地 commit 同时绑定源码、测试与该产物。

## 6. 验收

- `py -3.12 -m pytest tests -q`：**252 passed**，包含 mHC 26项、Engram 原24项，G1–G17 全保留。
- `node --check simverse/static/workbench.js`：exit 0；该文件未修改。
- pin：14 文件完整 blob SHA1 对 tree + SHA256；新 evidence executed=False，CPU/GPU 状态未升级。
- 不调用真实模型、ZCode 或其他 Agent；没有 kernel/CPU forward/GPU 测量；仅本地 commit，不推送。

## 7. 与任务书的出入及限制

1. **出口 collapse 采用2mh，非(2m−1)h**：K/triton.py L96 初始化 acc=0，
   L97-109 每流 `acc += pre*x`；L144-145 禁止 FP fusion。按源级零累加操作是 mh乘+mh加。
   任务书给出的是数学最简归约（丢去第一次与零相加），单列35840；源码口径40960进入账本，
   比任务书预期整模型多5120 op。本轮不推断编译器是否消除零加法。
2. **expand 去掉误计的 mh FLOPs**：N/ops/mhc.py L341-343 只有 load/store。
   原账本20480是复制元素数，应放字节，不是浮点运算。任务书未显式列此删除项，源码为准修正。
3. **融合 split 不限于{1,4,16}**：该集合属于 unfused deep_gemm 的 bucket；
   K/tilelang_kernels.py L91-114 自选 fused split。h5120/T1 常规 fused split=8、tile_n=2；
   h4096/SM100+ 单遍 split=16；并非仅运行时 deep_gemm 决定。split=1 参考是归一化基础账本，
   **不声称 B1 实际走 split=1**。V4.1 80边界中首层attn无post、两处Engram注入attn
   强制post/pre分立（N/model.py L583-600），余77边界若走此fused，每 token仅epilogue
   归并就额外77×7×25=13475 op；首层与Engram侧split另依赖设备。fused post还有projection tile重算，
   因此“算术结构相同”只对语义总量成立，不能扩写为物理执行总量相同。
4. **norm 的5h+2不是该 kernel 的逐循环精确式**：L357 `hidden_block=gcd(1024,h)`，
   L461/L497 将多段 sumsq 累加到 hidden_block 个槽；L505 仅归约这些槽。
   若逐显式零累加计，2h（平方+加）+(hidden_block−1)+3+2h =
   4h+hidden_block+2；h5120 时21506，任务书5h+2=25602是在按 h 个独立槽归约。
   **按任务书要求保留仓库4h+2=20482惯例**，不改全局 norm/G11，并明确是近似。
5. **分支名称/谓词补全**：mega 实际判断 capability family100，不是仅 GB100；
   fused-AR 还要求非EP、h5120/m4；NVIDIA overlap限FULL graph capture且非piecewise；
   ROCm overlap≤128独立于 NVIDIA≤16；gfx950 aiter
   依赖可用性/shape，其他 ROCm 可能走 TileLang。部署变体全部保持未建模，未虚构统一运行耗时。
6. **字节首层特例与参数补项**：L447-450 用 one-hot，首层无 pre_mix_in 读；
   epilogue 明确读 base/scale，本轮补到参数列。激活/norm 按任务书 b=act/8 账本约定；
   pinned kernel BF16、split 中间张量/重复访问、cache/fused 减读和真实 HBM 未执行核对。
7. **证据界限**：独立 stdlib 数学测试只验证对源码算法的解读，不是 CPU forward 或
   kernel 数值/shape-checked；登记仅 source-checked。CSA2/MoE scoring/量化与 Engram DP/
   cpu_offload 运行时、spec capture_aux、workspace/graph/NCCL、实际预算仍有缺口。
   structure_complete=False，两款主模型部署排名规则不变。

## 8. 改动文件清单（34）

- `.gitattributes`
- `ROADMAP.md`
- `docs/FOCUS_STATUS.md`
- `docs/IMPLEMENTATION_STATUS.md`
- `docs/evidence/r05-mhc/after.json`
- `docs/evidence/r05-mhc/attribution.json`
- `docs/evidence/r05-mhc/baseline-provenance.json`
- `docs/evidence/r05-mhc/before.json`
- `docs/evidence/r05-mhc/scenario.json`
- `docs/gaps/capability-gaps.md`
- `docs/rebuild-20261106/references/vllm/mhc-pin.json`
- `docs/rebuild-20261106/references/vllm/vllm/model_executor/kernels/mhc/__init__.py`
- `docs/rebuild-20261106/references/vllm/vllm/model_executor/kernels/mhc/aiter.py`
- `docs/rebuild-20261106/references/vllm/vllm/model_executor/kernels/mhc/cpu.py`
- `docs/rebuild-20261106/references/vllm/vllm/model_executor/kernels/mhc/tilelang.py`
- `docs/rebuild-20261106/references/vllm/vllm/model_executor/kernels/mhc/tilelang_kernels.py`
- `docs/rebuild-20261106/references/vllm/vllm/model_executor/kernels/mhc/torch.py`
- `docs/rebuild-20261106/references/vllm/vllm/model_executor/kernels/mhc/triton.py`
- `docs/rebuild-20261106/references/vllm/vllm/model_executor/kernels/mhc/warmup.py`
- `docs/rebuild-20261106/references/vllm/vllm/model_executor/layers/mhc.py`
- `docs/rebuild-20261106/references/vllm/vllm/models/deepseek_v41/amd/mhc.py`
- `docs/rebuild-20261106/references/vllm/vllm/models/deepseek_v41/nvidia/ops/cute_dsl/__init__.py`
- `docs/rebuild-20261106/references/vllm/vllm/models/deepseek_v41/nvidia/ops/cute_dsl/all_reduce_mhc.py`
- `docs/rebuild-20261106/references/vllm/vllm/models/deepseek_v41/nvidia/ops/mega_mhc.py`
- `docs/rebuild-20261106/references/vllm/vllm/models/deepseek_v41/nvidia/ops/mhc.py`
- `docs/validation/v41-mhc-kernels-20261112.md`
- `simverse/count/work.py`
- `simverse/formula_registry.py`
- `simverse/importers/source_map.py`
- `simverse/lowering/memory.py`
- `simverse/model_ir/catalog.py`
- `simverse/validation/registry.py`
- `tests/test_v41_hc.py`
- `tools/replay_v41_mhc.py`
