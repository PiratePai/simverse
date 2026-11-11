# Qwen3.6 GDN+MoE 小尺寸 CPU forward 对照（R06，2026-11-12）

R06 本轮范围：Qwen3.6（qwen3_5_moe 家族）**GDN 线性注意力层与 MoE 路径**的
小尺寸 CPU forward 对照；full-attention 层作为可运行载体一并核（覆盖其输出
门控），并据此修正一处已确认漏计。对照基准 = pin 的 HF 官方实现（随机权重
执行），本仓侧 = `parse_qwen3_5_moe(同一 config dict)` → `build_model_work(tp=1)`。

## 1. 结论

- **逐算子精确对账通过**：两相位（prefill T=5 / decode T=1）下，本仓账本的
  全部 GEMM 行、depthwise 因果卷积行、MoE router/routed/shared/gate 行、
  attention 因果对数、递推状态几何与官方 fixture 逐项相等（§4 对账表）。
- **发现并修正一处漏计**：full-attn 输出门控逐元素行（官方 L802
  `attn_output = attn_output * torch.sigmoid(gate)`）此前只计了 q_proj 输出
  2× 的 GEMM 消费，sigmoid+乘未计；新增 `attn_output_gate` 行（§5 归因）。
- **模型级证据不冒充**：本轮为 fixture 级 CPU 执行证据
  （`ev-qwen36-cpu-fixture-r06`，numeric-checked）；`qwen36_35b_a3b` 的
  `cpu_forward` 保持 `not-run`，不构成完整 forward 或部署验证。
- 两项**披露不修**的口径差异（§6）：gdn_recurrence 4 ops/elem（官方 torch
  参考约 7 ops/elem）；HF DynamicCache conv state 静态宽度 = kernel 列
  （本仓按 vLLM 口径 kernel−1 计容量）。

## 2. 来源与版本

| 项 | 值 |
|---|---|
| 官方 forward pin | `assets/identity/hf_transformers/modeling_qwen3_5_moe.py`（huggingface/transformers @b6401d66），SHA256 `1311a17431d7c232…`（fixture `pinned.modeling_sha256` 全量） |
| 官方 configuration pin | 同目录 `configuration_qwen3_5_moe.py`，SHA256 `554e363a32b1a8c2…` |
| 执行环境 | torch 2.15.0+cpu / transformers 5.22.1 / Python 3.12.10 / Windows（fixture `environment` 字段全录） |
| 随机种子 | `torch.manual_seed(20261112)` |
| fixture 维度 | h=32，2 层（L0 GDN + L1 FullAttn），4E top-2 + shared16 + gate，nv/nk=4/2，dk=dv=8，conv kernel 4，rotary_dim=2（partial 0.25），V=64 |
| 落盘产物 | `tests/fixtures/qwen36_r06_cpu_fixture.json`（探针独立产生，无任何本仓代码参与） |

## 3. 方法

探针 `tools/probe_qwen36_forward_r06.py`（**不 import simverse**）：沿用
`tools/probe_qwen36_forward_cpu.py` 的 importlib 注册手法（pin 文件挂
`transformers.models.qwen3_5_moe.pinned_*` 名下动态加载执行）。

采集手段（对 transformers 5.22.1 逐一核实）：

- **GEMM**：`nn.Linear` 前向 hook，ic/oc 取自 `weight.shape`，tokens=B·S。
  router 的 `F.linear` 不是 nn.Linear 模块，经 `Qwen3_5MoeTopKRouter` 模块
  hook 从输出 logits 形状与 weight 形状计 `router_flops = 2·T·h·E`。
- **conv**：`nn.Conv1d` 模块自身从不被调用（weight 直传 `F.conv1d`），模块
  hook 不触发——改为包裹模块级 `causal_conv1d_fn`（prefill，pad_conv 路径）
  与 `causal_conv1d_update`（decode，update 路径）。该版本
  `use_kernelized_func` 为 no-op，模块级包裹确定生效。
- **GDN 递推**：包裹 `torch_chunk_gated_delta_rule` /
  `torch_recurrent_gated_delta_rule`，捕获入参/返回形状与分支
  （prefill chunk 零初始态 / decode recurrent 带初始态 [1,4,8,8]）。
- **full-attn**：`ALL_ATTENTION_FUNCTIONS` 无 "eager" 注册项，
  `get_interface("eager", fallback)` 返回 pin 模块自身的
  `eager_attention_forward`——包裹之捕获 attn_weights 形状并按 j≤i 计
  因果对数；rotary_dim 由 `position_embeddings` cos 宽验证（=2）。
- **MoE**：包裹实例 `experts.forward` 捕获 `top_k_index` 统计每专家 token 数
  （无放回 top-k 的 assignment 总量 = T·k 精确核验）。

探针内置自洽断言（flops 公式、nv/nk/dk/dv、state_elems=256、pairs=15/6、
assignment=T·k、logits 形状、分支与 conv 路径、norm 宽度清单），任何失败
即非零退出。**注意**：GDN 的 q/k 在进入递推前被官方 forward
`repeat_interleave` 到 nv 头（kernel 边界看不到 nk=2），fixture 的 nk 由
in_proj_qkv/in_proj_z 的观测 oc 反推（key_dim=(qkv_oc−z_oc)/2=16 → nk=2），
并把 repeat 后头数单列 `k_heads_after_repeat=4`。

## 4. 对账表（本仓 build_model_work(tp=1, 全 32bit) vs fixture）

| 算子行 | prefill 本仓 | prefill fixture | decode 本仓 | decode fixture |
|---|---|---|---|---|
| L0 in_proj_qkv | 20480 | 20480 | 4096 | 4096 |
| L0 in_proj_z | 10240 | 10240 | 2048 | 2048 |
| L0 in_proj_ba（=官方 b+a 两模块合计） | 2560 | 2560 | 512 | 512 |
| L0 causal_conv1d | 2560 | 2560 | 512 | 512 |
| L0 gdn_recurrence（4·T·state_elems） | 5120 | 5120 | 1024 | 1024 |
| L0 v_norm_gated / out_proj | 680 / 10240 | 形状核（RMSNormGated 宽 8×4 头）/ 10240 | 136 / 2048 | 同左 / 2048 |
| L1 q_proj（2× 门控输出 oc=64） | 20480 | 20480 | 4096 | 4096 |
| L1 k_proj / v_proj | 5120 / 5120 | 5120 / 5120 | 1024 / 1024 | 1024 / 1024 |
| L1 attention_core（pairs·4·(4·8+5)） | 2220（pairs=15） | 2220（pairs=15） | 888（pairs=6） | 888（pairs=6） |
| L1 attn_output_gate（本轮新增） | 320 | 320 | 64 | 64 |
| L1 o_proj | 10240 | 10240 | 2048 | 2048 |
| L1 q_norm / k_norm / rope | 680 / 340 / 120 | norm 宽 8×heads 核 / 同 / rotary=2 | 136 / 68 / 24 | 同左 |
| 每层 moe_router | 1280 | 1280 | 256 | 256 |
| 每层 moe_routed（Σ_e 6·t_e·h·inter） | 30720 | 30720 | 6144 | 6144 |
| 每层 moe_shared（gate+up+down 3 GEMM） | 15360 | 15360 | 3072 | 3072 |
| 每层 shared_gate | 320 | 320 | 64 | 64 |
| 每层 attn_norm / mlp_norm / attn_add / mlp_add | 650 / 650 / 160 / 160 | norm 宽 32 / 同 / T·h / 同 | 130 / 130 / 32 / 32 | 同左 |
| head final_norm / lm_head / embed_lookup | 650 / 20480 / 0 | 宽 32 / 20480 / 查表 0 flops | 130 / 4096 / 0 | 同左 |

其余核验：logits 形状 (1,5,64)/(1,1,64)；GDN 递推状态
[1,4,8,8]（state_elems=256）；decode conv state（见 §6 披露）；
MoE assignment 每层恰 T·k（5·2=10 / 1·2=2，两层合计 20/4）；
shared_expert_gate Linear(32,1)；norm 宽度清单
{attn_norm:32, mlp_norm:32, final_norm:32, q_norm:8, k_norm:8, rmsnorm_gated:8}。

对账测试：`tests/test_qwen36_forward_r06.py` Part A（hermetic，9 项，无
torch，CI 必跑）+ Part B（live 重执行探针，1 项；torch/transformers 缺失时
skip；不含随机路由分布比较，防跨版本脆断）。

## 5. 发现与修正（attn_output_gate 前后归因）

**发现**：官方 `Qwen3_5MoeAttention` L802 在 o_proj 前执行
`attn_output = attn_output * torch.sigmoid(gate)`。本仓 `gqa_layer_rows`
此前只把 q_proj 输出按 2×（q_out=2·heads·d）计入 GEMM——门控**分支的
产出消费**（逐元素 sigmoid+乘）漏计。

**修正**（2026-11-12）：
- `simverse/count/work.py` `gqa_layer_rows`：`attn.output_gated` 时在
  attention_core 与 o_proj 之间插入 `attn_output_gate` 行，
  `flops = 2·T·(heads_c·d)`，读 2·T·hd·B、写 T·hd·B；
- `simverse/formula_registry.py` 注册 `v2.attn_output_gate`；
- `simverse/importers/source_map.py` `sm-qwen35-attn-output-gate` note 补
  一句"逐元素行已计"（不改状态枚举）。

**默认场景（CLI 内置 Qwen3.6+DFlash，TP8/EP1）前后归因**
（`py -3.12 -m simverse.cli run-v2`，修正前→后）：

| 量 | 前 | 后 | Δ |
|---|---|---|---|
| target_decode FLOPs | 1,493,291,334 | 1,493,301,574 | **+10,240**（=10 个 full 层 × 2·1·(16/8 头·256)=1024） |
| verify_block FLOPs | 23,895,130,944 | 23,895,294,784 | **+163,840**（=10 层 × 2·16·512，verify Q=16） |
| draft_block FLOPs | 5,341,827,850 | 5,341,827,850 | 0（DFlash 草稿 full-attn 无输出门控） |
| decode/verify 行数 | 613 / 613 | 623 / 623 | 各 +10 行 |
| M01_tpot_analytical | 36.859086 ms | 36.859107 ms | +0.000021 ms |
| M04_round_ms | 130.952989 ms | 130.953062 ms | +0.000073 ms |

与预期一致（默认场景 TP8 每 token 每 rank +10,240 FLOPs）。

## 6. 披露与限制（不修，登记口径）

1. **gdn_recurrence 4 ops/elem 是注册 approximated 口径**：官方 torch 参考
   递推（chunk 路径 L399-406 / recurrent 路径 L461-472：decay 乘、旧态读、
   k⊗v 外积、delta 减、写回、q 读出）约 **7 ops/elem**；本仓按既有
   `v2.gdn_step` 口径 4 ops/elem 计，行级 statistic="approximated" 如实
   标注，本轮未改（prefill 参考值 ≈8,960 vs 本仓 5,120）。
2. **conv/MoE 内激活微操作未计**：conv 输出 silu、MoE 专家内 silu 逐元素、
   router softmax/topk/renorm 微操作不计——与仓库"norm/rope/残差计逐元素、
   激活不计"的既有口径一致；GDN b/a 的 sigmoid/softplus 同理未计。
3. **HF DynamicCache conv state 静态宽度 = kernel 列**（fixture 观测
   state_len=4，cache_utils 按 conv_kernel_size 建静态形状供 cudagraph）；
   本仓按 vLLM 口径 kernel−1 列计每请求容量（64·3·4=768B + 递推态
   1024B = 1792B）。差 1 列（256B/层/请求，fp32）属 HF 缓存实现填充，
   不改变逐 token 卷积算量，不改本仓。
4. **随机权重小尺寸 fixture**：h=32、2 层、4E——不外推真实 35B 尺寸行为；
   不覆盖真实 checkpoint、MTP、视觉塔、mrope 位置 id；fla/causal-conv1d
   未安装，走 PyTorch 参考实现（transformers 显式 fallback 提示，行为正确
   但慢）。
5. **模型级 cpu_forward 保持 not-run**：本轮是 fixture 级 numeric-checked
   证据；两款主模型完整 CPU forward/GPU 实测仍未执行，部署排名门禁不变。

## 7. 复算命令

```bash
py -3.12 tools/probe_qwen36_forward_r06.py          # 重新执行官方 forward 落盘 fixture（exit 0）
py -3.12 -m pytest tests/test_qwen36_forward_r06.py -q   # hermetic + live 对账（10 项）
py -3.12 -m pytest tests -q                          # 全量（收集数见 IMPLEMENTATION_STATUS）
py -3.12 -m simverse.cli run-v2                      # 默认场景（attn_output_gate 已入账）
```

证据登记：`simverse/validation/registry.py`
`ev-qwen36-cpu-fixture-r06`（numeric-checked、executed=True，内容 SHA256
绑定 fixture JSON）；`simverse/model_ir/catalog.py` `_qwen36()` evidence_refs
追加该证据，`_QWEN36_EVIDENCE.note` 补记 fixture 级对照已执行（模型级
cpu_forward/gpu_measurement 不变）。
