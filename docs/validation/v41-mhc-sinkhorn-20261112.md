# V4.1 mHC/Sinkhorn kernel 计数修正（ROADMAP R05）验证报告

日期：2026-11-12。交付人：DSH（实现）；主控 ZCode 完成源码审查与 pin（分支表
由实现者逐条对源码行号复核，差异见 §3.4）。本报告数字由
tests/test_v41_mhc_kernels.py 的**独立 oracle** 复算（逐式独立推导，不 import
生产公式）；全部结论为 **source-checked（源码阅读）**，未执行任何 kernel。

## 1. 证据与 pin（12 核心文件 + 3 同批；blob sha1 对 vllm-tree.json 校验）

上游 vLLM commit `f358f0578a83f4e1fe6fc49a492d3ae9362de8e5`（与 R04 engram/
model.py 同 pin），本地 `docs/rebuild-20261106/references/vllm/` 下；git blob
sha1 逐项对 `vllm-tree.json` 校验通过（测试 `test_mhc_kernel_pins_blob_sha1_
match_tree` 防漂移）：

| 文件（相对 vllm/） | blob sha1（前 12） | size | SHA256 |
|---|---|---|---|
| model_executor/kernels/mhc/tilelang.py | 5a291b9c320f | 37,600 | cf75da24521424fbe07ff8228e18c8272b7878bd80bc2dcefc1d99a598d117fb |
| model_executor/kernels/mhc/tilelang_kernels.py | d79f7f42ab6e | 69,229 | 1cf58ea6d51dd56f886b0a9571dc0ad9d75a0a908fd68b1c2d6d28234bd13d55 |
| model_executor/kernels/mhc/triton.py | 0e6c47b52dfa | 15,479 | 8f61199e55adf03d8072b7e51f76daf584a89ba0e7d6312fda254668f7f2370c |
| model_executor/kernels/mhc/torch.py | eb4de35e7c75 | 5,356 | 3697085aa6f0070e165c145f9c337ca3fe0623623c8265d29e50c43467e49985 |
| model_executor/kernels/mhc/warmup.py | deb1cf429a0c | 3,546 | feee09b7e0444e24b53c9619d7621467848b707b97b7ad81ff42b7824803a38c |
| model_executor/kernels/mhc/__init__.py | 6fc49199891e | 926 | b1aac23563953599c1dabe33e38d6a4c17cb72653409850854324c233ee8f5ab |
| model_executor/layers/mhc.py | 19e8b427b494 | 30,990 | 433d750a4d669ad468679b1faf0a3f1e846410198fb4a1eeee2d14cfe55e6a55 |
| models/deepseek_v41/nvidia/ops/mhc.py | a2207d1c4d70 | 11,647 | e7c3f0ae3cbfe54f41ad2718600b47906571008719fb9da1de86cd7da97636d5 |
| models/deepseek_v41/nvidia/ops/mega_mhc.py | bdb5b95bbeb9 | 3,022 | 4b4af8b71c20d929303b15bf770eeff25a875046f08dc698ca8c47bb51b4fb9c |
| models/deepseek_v41/nvidia/ops/cute_dsl/all_reduce_mhc.py | 166efc56ff76 | 76,037 | 9203d5c62634a4858e28350f20599dc29964eab901e771185fd5fd321a2469d7 |
| models/deepseek_v41/amd/mhc.py | bee13f5d5023 | 4,434 | 9be806b5fb84bee3ce3f874e118a9697e3b788e06f9ac5b91a48e4c90060cdfa |
| utils/deep_gemm.py | 348575bf0358 | 42,139 | 2e647cb1be1e3bd3c38af8d9231b83d9e4fb70d4209cbd41b13da37c7f5d2a8d |

同批另 pin（域外登记用）：kernels/mhc/aiter.py `f102143ab08f`（ROCm）、
kernels/mhc/cpu.py `d12b0eae1851`、ops/cute_dsl/__init__.py `03d8a11e0bc2`。
证据登记：`ev-vllm-v41-mhc-kernels`（source-checked，2026-11-12）；V4.1
catalog evidence_refs 已挂（G14 refs ⊆ registry 自动满足）。

## 2. 分支表（B1–B9；实现者对源码逐条复核）

一次 mHC 边界 = 前一子层 post（流更新）+ 本侧系数 GEMM + sinkhorn + collapse
+ RMSNorm。dispatch 链 `nvidia/ops/mhc.py mhc_shifted_post_pre`（L192-323）：

| 分支 | 谓词（源码行号） | V4.1 默认场景判定 |
|---|---|---|
| B1 融合 all-reduce（MoEOutput 未 finalize / reduce_results） | `supports_mhc_all_reduce`（L60-73）：TP4 且非 EP 且 h==5120 且 m==4 且 MNNVL multicast（`ca_comm.mnnvl_lamport_ag_multicast_ptr`） | **不适用**：部署 TP8；当前设备注册表（H100/H200/L40）无 MNNVL。域外登记 |
| B2/B3 overlap 流路径 | `supports_mhc_overlap`（L45-57）：SM100 家族 + DeepGEMM + h%64==0 + 0<mix≤32 + mix%8==0 + 非 ubatching；model.py L524-531 再要求 FULL graph 且 T≤16（`MHC_OVERLAP_MAX_TOKENS`=16） | **不适用**（SM100）。域外登记 |
| B4 mega mhc | `can_use_mega_mhc`/`is_mega_mhc_supported`（mega_mhc.py L16-44）：DeepGEMM + SM100 家族 + h%1024==0 + m==4 + T≤2²⁰ | **不适用**（SM100）。域外登记 |
| B5 默认 `mhc_fused_post_pre_delayed_tilelang`（tilelang.py L256-486） | 融合核 `mhc_fused_post_pre_split_config`（tilelang_kernels.py L63-115）：T≤32 且 h%128==0 → `_MHC_FUSED_TILELANG_KERNEL`（post 折入 prenorm GEMM；T<16 config (2,8,128)、T∈[16,32] (6,8,128)；single-pass 变体需 SM100 且 h==4096 → V4.1 h=5120 不适用）；否则 `_MHC_POST_TILELANG_KERNEL` + prenorm GEMM（DeepGEMM TF32 split-K 或 tilelang FP32，FLOPs 相同） | **当前配方主路径**。epilogue：model 路径 norm_weight 恒非 None → `MHC_PRE_NORM_KERNEL`（= mhc_pre_big_fuse_with_norm_tilelang L319-527） |
| B6 首层 | model.py L543-560：`hc_expand`（triton 拷贝核，ops/mhc.py L326-355，读一次 T·h 写 T·m·h）+ pre 用 fn=hc_attn_fn_broadcast [mix,h]（**派生张量**：finalize_mhc_broadcast_weights L1530-1544，`hc_attn_fn.view(-1,m,h).sum(dim=1)`，非 checkpoint 参数）+ x=广播嵌入（IC=h，rms_numel=h，model 传 x=x）；identity pre-mix 选流 0（save_pre_mix=True, use_pre_mix_in=False） | 计入（hc_expand 行 + 首层 attn mix 行 IC=h、weight_read=mix·h·4） |
| B7 engram 层 | model.py L583-614：post（mhc_post_tilelang）分立 → Engram 注入 → pre（mhc_pre_delayed_tilelang）分立；FFN 侧仍走 B5 融合 | 计入：操作总量与常规层相同（源码证实；分立不改变计数，engram 注入行由 R04 另计） |
| B8 模型出口 | model.py `_collapse` L1259-1279（末层 FFN 输出 post，mhc_post_tilelang，无后继边界）+ 主 forward L1187-1192（hc_collapse_triton × 末层 ffn pre_mix + final RMSNorm） | **本轮新增** hc_exit_post/hc_exit_collapse 两行（layer=-1，final_norm 行已存在不重复计） |
| B9 域外/未建模 | write_kv（model.py L680-708）/decoder_replay/specialized_decode、SP gather（L641-648/L1171-1173）、amd aiter（amd/mhc.py + kernels/mhc/aiter.py）、cpu.py 回退 | 登记（sm-v41-mhc-kernels uncovered_domain；不实现） |

## 3. 迭代语义与操作计数推导（逐式）

计数约定（写入 formula_registry v2.mhc_* 与证据 limits）：**每元素初等运算各
计 1 op**（加/减/乘/除/比较），**exp/sigmoid/rsqrt 各 1 op**；**归约求和 n
元素计 n−1 次加、归约 max 计 n−1 次比较**；FMA 口径沿用 1 乘加=2 FLOPs
（GEMM）。split-K 跨片规约与 NaN 守卫分支不计入（§4）。

### 3.1 `hc_{side}_mix`（系数 GEMM + 核内 sqrsum）

`flops = T·(2·ic·mix + 2·ic)`，ic = h（仅首层 attn 侧，GEMM 输入为广播嵌入）
否则 hc_dim=m·h。

- GEMM：`[T,ic]@[ic,mix]`（mix=(2+m)·m=24），2·T·ic·mix；
- 核内 sqrsum：hc_prenorm_gemm_tilelang L911-912（`sqr[0] += x_val*x_val`，
  每 k 元素 1 乘 + 1 加；T≤32 融合路径 mhc_fused_tilelang L772-775 对
  post 映射后的流同式）→ 2·T·ic。
- 首层 attn 侧读 broadcast 派生张量 [mix,h] fp32：weight_read = mix·h·4
  （V4.1: 491,520 B）；其余侧 weight_read = mix·hc_dim·4（fn 参数本体）；
  weight_capacity 恒 mix·hc_dim·4。act_read/act_write 维持原解析近似
  （T·ic·b_a / T·mix·4）。

### 3.2 `hc_{side}_sinkhorn`（系数 epilogue，精确整数）

`flops = T·[m²·(4R+6) + 6m + mix + 3]`（R=hc_sinkhorn_iters=20）。逐式
（mhc_pre_big_fuse_with_norm_tilelang L383-441；首层 broadcast 变体
L530-690 与 torch/warmup 路径同构）：

| 步骤 | 源码 | op |
|---|---|---|
| rms | `T.rsqrt(rms/rms_numel + rms_eps)`（L385） | 3（÷ + eps 加 + rsqrt） |
| mixes×rms | `mixes[j] *= rms[0]`（L390） | mix |
| pre 门 | `sigmoid(mixes·scale0 + base) + pre_eps`（L397-400） | 4m |
| post 门 | `sigmoid(mixes·scale1 + base)·post_alpha`（L401-406） | 4m |
| comb affine | `mixes·scale2 + base`（L407-411） | 2m² |
| 首轮 softmax（行向 dim=-1） | reduce_max L417（m(m−1) 比较）、`exp(cm−row_max)` L419（减 m² + exp m²）、行和 L420（m(m−1) 加）、`cm/row_sum + eps` L422（除 m² + 加 m²） | 6m²−2m |
| 列归一 ×1 | 列和 L424（m(m−1) 加）、`cm/(col_sum+eps)` L426（eps 加 m + 除 m²） | 2m² |
| 循环 (R−1) 次 ×（行归一 2m² + 列归一 2m²） | L433-440 | 4m²(R−1) |

合计 = 3 + mix + 8m + 2m² + 6m²−2m + 2m² + 4m²(R−1) = **m²(4R+6)+6m+mix+3**；
V4.1（m=4, R=20）= 16·86+24+24+3 = **1427/token/side**。迭代语义：首轮
softmax+列归一之后才进 (R−1) 次行/列交替（L433-440 `T.serial(sinkhorn_
repeat - 1)`）→ R=1 时循环 0 次（= m²·10+6m+mix+3），R 与 R−1 相差恰 4m²
（测试 `test_sinkhorn_r1_loop_zero_and_step_exact_4m2`）。NaN 守卫（`rms[0]
>= 0.0`，L432）只跳过 NaN padding 行（CUDA-graph padding），按控制流不计入。
statistic 由 "approximated" 改为 "exact"（计数约定下的精确整数）。

### 3.3 `hc_{side}_post_pre` 与出口两行

`hc_{side}_post_pre`: `flops = T·[h·(2m²+3m+5) + 2]`。构成：

- post 核 h(2m²+m)：mhc_post_tilelang L856-861（每 h 元素：m 乘
  （post·x）+ m² 乘 + m² 加（comb·residual））；T≤32 融合路径把 post 折入
  prenorm GEMM prologue（mhc_fused_tilelang L766-780），操作量相同；
- collapse 2mh：epilogue L478-483（每 h 元素 m·(乘+加)，pre_mix 加权求和）；
- 两遍 RMSNorm 5h+2：sumsq 累积 2h（L495-501）+ 归约 h−1（L505）+
  `rsqrt(sumsq/hidden_size + norm_eps)` 3（L507）+ 缩放 2h（L517-522）。
  语义注记：save_pre_mix 下先 round bf16 再算 sumsq（"Keep the BF16
  boundary before the delayed input RMSNorm"，L491-498）——RMSNorm 看到的
  正是 bf16 舍入后的值。
  V4.1 = 5120·49+2 = **250,882/token/side**。

`hc_exit_post`（新）：`flops = T·h·(2m²+m)`（= post 核同式；V4.1
184,320/token）；`hc_exit_collapse`（新）：`flops = T·2mh`（hc_collapse_
triton，triton.py L75-147，`enable_fp_fusion=False` → 逐流 m·(乘+加)；
V4.1 40,960/token）。`hc_expand` 不变（拷贝核，flops=T·hc_dim 为写计数口径）。

### 3.4 与主控审查推导的记录差异（以源码为准）

- **sinkhorn 常数项**：主控分解"除以 rms_numel 1 + rsqrt 1"= 2，漏了
  `+ rms_eps` 的 1 次加（L385）→ 主控式 `m²(4R+6)+6m+mix+2`，源码为
  **+3**（锚点 m=2,h=8,R=3：94 → 95；V4.1：1426 → 1427）。
- **post_pre RMSNorm**：主控"rsqrt 1"漏了 `÷hidden_size + norm_eps` 的
  2 op（L507）→ 主控式 `h(2m²+3m+5)`，源码为 **+2**（锚点 152 → 154；
  V4.1 250,880 → 250,882）。
- 主控预计 V4.1 整模型 hc decode T=1 总量 101,591,968：按主控自己的逐式
  分解复算应为 101,582,240（差 −9,728，无法归因到任何分项）；按源码计数
  实跑 **101,582,480**（= 主控逐式 + 80 边界×1 + 80 边界×2 = +240）。
- 行号微差：finalize_mhc_broadcast_weights 实际 L1530-1544（主控写
  L1529-1544）；supports_mhc_overlap 的 FULL graph/T≤16 条件在 model.py
  L524-531 而非谓词函数本身——语义不变。
- 其余分支表条目（B1–B9 谓词、行号、T≤32 融合 config、迭代语义 R−1、
  广播派生张量、exit 构成）与源码全部一致。

## 4. 精确 vs 近似分列

**精确（本轮落地，整数）**：mix（GEMM+sqrsum）、sinkhorn（§3.2 全部分解）、
post_pre（post 核+collapse+两遍 RMSNorm）、exit_post、exit_collapse、
expand（写计数）；首层 attn mix weight_read=mix·h·4 与 broadcast 派生张量
驻留 491,520 B（每 rank 复制）；count↔memory 守恒（mhc 层对象 − base/scale
= mix 行 capacity 和）。

**约定（文档化，非指令数）**：超越函数计 1 op；归约求和 n−1 加/归约 max
n−1 比较；rsqrt(x/n+eps) 拆 3 op；NaN 守卫/split-K 跨片规约
`(n_splits−1)·(mix+1)`/token（依赖设备 SM 数，compute_num_split
tilelang_kernels.py L44-54）不计入。

**近似/未细分（保留原口径并登记）**：act 流量字段为解析近似（中间缓冲
mixes/sqrsum 的 splits 维 HBM 往返、T≤32 融合路径 post 从寄存器喂入 GEMM
节省的 residual 二次读写，未细分）；post_pre 行 act_read/act_write 维持
现状（未计 post_mix/res_mix 系数张量字节）。

**未建模分支（域外登记）**：SM100 系（mega_mhc、overlap 流、T≤4
single-pass）、MNNVL TP4 融合 all-reduce（cute_dsl/all_reduce_mhc.py）、
write_kv/decoder_replay/specialized_decode、SP gather、amd aiter/cpu 变体。

## 5. V4.1 默认场景 replay 前后归因

ScenarioV2 默认 deepseek_v41_flash baseline TP8（B=1，ctx=32768，decode
T=1）；run_v2 前后各跑一次（旧值经 git stash 临时还原 simverse/ 取得）：

| 指标 | 旧（R04 后） | 新（R05） | Δ |
|---|---|---|---|
| 整模型 decode FLOPs | 7,146,532,702 | 7,161,804,622 | +15,271,920 |
| 其中 hc 部分 FLOPs | 86,310,560 | 101,582,480 | +15,271,920 |
| decode 行数 | 916 | 918 | +2（hc_exit_*） |
| hc 行数 | 241（6L+1） | 243（6L+3） | +2 |
| target_r0 steady bytes | 35,662,849,472 | 35,663,340,992 | +491,520（broadcast 派生张量） |
| mhc 分配合计 | 157,295,040 | 157,786,560 | +491,520 |
| decode 解析耗时下界 t_ms | 116.1526519188457 | 116.15223469687024 | −0.000417 |

ΔFLOPs 分解（T=1；主控复核后修正 DSH 初稿两处分解笔误，合计不变）：
mix 核内 sqrsum +3,246,080（首层 attn +10,240 + 79 边界 ×40,960）；
sinkhorn 40·2·(1427−2400) = −77,840；post_pre 40·2·(250,882−102,402) =
+11,878,400；exit +225,280；合计 +15,271,920（与端点差 101,582,480−
86,310,560 精确相等）。t_ms 微降：首层 attn mix
weight_read 修正为 broadcast 派生张量（mix·h·4=491,520 代替 mix·hc_dim·4
=1,966,080），每步少读 1,474,560 B 权重流量，HBM 侧解析下界随之下降。
（M01 support 仍 approximated：Engram host 查找未计，与本轮无关。）

## 6. 测试清单与数值

tests/test_v41_mhc_kernels.py（11 项，oracle 全部独立手推）：

| 测试 | 关键断言 |
|---|---|
| test_mhc_kernel_pins_blob_sha1_match_tree | 15 pin 文件 blob sha1+size 对 vllm-tree.json 逐项相等 |
| test_hand_anchor_first_layer_rows | m=2,h=8,R=3,T=5：mix attn 720/ffn 1440、sinkhorn 475、post_pre 770、expand 80；首层 attn wr=256/cap=512 |
| test_hand_anchor_exit_rows | exit_post 400、exit_collapse 160；traffic 手算（240/160/200/80） |
| test_sinkhorn_r1_loop_zero_and_step_exact_4m2 | R=1 = 63·T；R=2−R=1 = 4m²·T = 80 |
| test_sinkhorn_not_approximated_and_formula_registry_text | statistic 无 approximated；注册表携带 4R+6 / 2m²+3m+5 / sqrsum / 计数约定 |
| test_first_layer_attn_mix_reads_broadcast_tensor | 首层 attn wr=mix·h·4=491,520；非首层/ffn = mix·hc_dim·4；cap 恒 1,966,080 |
| test_exit_rows_structure_and_non_v41_family_has_none | exit 恰 2 行 layer=-1 位于 final_norm 前；tiny_gqa 无 hc_ 行 |
| test_engram_layers_hc_composition_unchanged | 层 1/14 仍 6 行且与常规层逐字段相同 |
| test_hc_rows_tp_invariant_all_fields | TP8 与 TP1 的 243 行 to_json 逐字段相等 |
| test_mhc_memory_broadcast_tp_invariant_and_conservation | broadcast=491,520 TP 不变；mhc 对象−base/scale=mix capacity 和；broadcast=首层 attn wr |
| test_v41_hc_totals_independently_recomputed_r05 | 独立逐层求和 = 101,582,480；行数 243；sinkhorn 1427、post_pre 250,882 锚点 |

tests/test_v41_hc.py 更新（11 项全保留）：手算锚点（mix 720/1440、sinkhorn
475 exact、post_pre 770）、非首层 mix 1152 + wr=fn 本体、整模型复算
101,582,480 与 6L+3=243 行、mhc+broadcast 复制守恒（3,932,376 + 491,520）。
tests/test_model_ir_v02.py 与 test_merge_gate.py 无受影响断言（G10 的
sm-v41-engram-hc/engram_dp_exchange/csa2_moe_routing 均保留；新增
sm-v41-mhc-kernels 进 checked_paths、ev-vllm-v41-mhc-kernels 进 refs）。

## 7. 剩余限制（如实；不冒充）

1. 全部结论 source-checked（源码阅读），CPU/GPU 均未执行；不构成 kernel
   指令数或实测耗时。
2. 域外未建模分支：SM100 系（mega/overlap/single-pass）、MNNVL TP4 融合
   all-reduce、write_kv/decoder_replay/specialized_decode、SP gather、
   amd aiter——谓词与行号已登记（sm-v41-mhc-kernels），出现相应部署域时须
   先补分支再谈计数。
3. split-K 跨片规约 (n_splits−1)·(mix+1)/token 依赖设备 SM 数，未入账。
4. act 流量仍为解析近似：中间缓冲（mixes/sqrsum splits 维）HBM 往返与
   T≤32 融合路径寄存器喂入的流量节省未细分；exit/post 行未计系数张量
   （post_mix/res_mix/pre_mix）字节。
5. V4.1 structure_complete 仍 False：CSA2 注意力/投影为 v0.1 移植、MoE
   scoring/noaux_tc、量化映射、candidate block 等缺口不变（mHC/Engram
   部分已不再缺失，但 mHC 亦未执行）。
