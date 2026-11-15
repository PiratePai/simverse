# R26：Qwen3.6-35B-A3B 文本路径 MRoPE position_ids 语义（2026-11-16）

## 0. 结论

Qwen3.6-35B-A3B 纯文本输入的 MRoPE position_ids 生成、mrope_section
分配、prefill/decode 增量、缓存位置与原生 MTP 草稿步的位置推进已按
**双栈官方实现逐行核对**（HF pin + vLLM rope 栈 5 文件新 pin + 既有
vLLM 主模型/MTP pin）并经**三路独立数学对照 + 官方 HF 旋转类本机执行**
验证。核心事实：**文本路径 MRoPE 完全退化为标准部分 NeoX RoPE**——
三个 T/H/W 位置分量恒等，mrope_section 交错重组是恒等映射；位置序列
prefill = 0..T−1、decode = {上下文长度}（缓存续位）；MTP 草稿位置从
target 位置逐元素复制起步、每步 +1（末步不推进）、被拒位置下轮回退
（seq_len = target_seq_len − num_rejected）、夹紧 max_model_len。

据此：`qwen36_35b_a3b` 的 `unverified_paths` 清空，`forward_source`
升至 **source-checked**，`structure_complete=True`（只表示 config 覆盖
+ 文本域 forward 源码证据完备，**不冒充 CPU 执行/GPU 实测**——两者
另列 not-run/none，部署排名资格不因此解除）。算量零变化（rope 行
3·T·heads·rotary_dim 口径不变；固定场景 replay 账本逐字段零漂移）。
多模态 2D 位置分支（T/H/W 行相异）域外未核，显式登记不冒充。

## 1. 来源与身份（固定仓库/提交/哈希）

| 对象 | 固定来源 | 本仓证据 |
|---|---|---|
| HF 文本模型/旋转类 | `huggingface/transformers @b6401d66`（既有 pin） | `assets/identity/hf_transformers/modeling_qwen3_5_moe.py`（LF SHA256 `abac917f…`，R06 manifest 同源） |
| vLLM rope 栈（新 pin） | `vllm-project/vllm @dc74b8cbbbe3dbb9399fe369757d63defd614cc2`（与 R01 qwen35 pin 同 commit） | `assets/identity/vllm_qwen36_mrope_pin/`（5 文件：qwen3_vl、rope_init、rope_base、rope_common、rope_mrope；manifest 逐文件 SHA256 + blob sha1，blob 对 GitHub contents API 校验） |
| vLLM 主模型/attention（既有 pin） | 同 commit | `assets/identity/vllm_qwen35_pin/qwen3_5.py`、`qwen3_next.py` |
| vLLM MTP 位置步进（既有 pin） | `vllm-project/vllm @10a854ed`（R23 pin） | `assets/identity/vllm_qwen36_mtp_pin/qwen3_5_mtp.py`、`tdar_speculator.py` |
| Qwen3.6 config（rope 几何） | `Qwen/Qwen3.6-35B-A3B @345d467b`（快照不变） | `assets/config/qwen36_35b_a3b_main__config.json`（ev-qwen36-main-config） |

**跨 commit 一致性**（2026-11-16 检查）：`rope_mrope.py`/`rope_base.py`/
`rope_common.py` 在两个 pin commit（dc74b8cb / 10a854ed）**字节一致**；
`rope_init.py` 差异仅在 dual-chunk 分支（Qwen3.6 走的 `default`+
`mrope_section` → MRotaryEmbedding 分支两版本逐字相同）；
`qwen3_vl.py` 差异仅在视频 mm_counts 管道（`mrope_positions_factor`
不变）。rope 语义在两 pin 之间无漂移。

真实 rope 参数（text_config.rope_parameters）：`rope_type=default`、
`rope_theta=1e7`、`partial_rotary_factor=0.25`（head_dim 256 →
rotary_dim 64，32 频率对，尾部 192 维直通）、`mrope_section=[11,11,10]`
（和 32 = rotary_dim/2）、`mrope_interleaved=true`、
`max_position_embeddings=262144`。

## 2. 官方实现查明的事实（逐行核对，三栏分立）

### 2.1 HF 栏（transformers @b6401d66）

* **position_ids 生成**（modeling L1322-1332，`Qwen3_5MoeTextModel.
  forward`）：`position_ids=None` 时 `torch.arange(seq_len) +
  past_key_values.get_seq_length()`，`view(1,1,-1).expand(4, B, -1)`。
  硬编码 4 行 = **[text, temporal, height, width]**；行 0（text）作为
  `text_position_ids` 供因果 mask 与 linear-attention（GDN）使用，
  行 1:3（T/H/W）进旋转。**纯文本四行恒等**；prefill 从 0 起，decode
  续缓存长度（无重启）。
* **旋转类**（L130-194）：inv_freq[i] = theta^(−2i/64)（L162-168）；
  forward 把 (3,B,T) 位置 `expand(3,-1,-1)`（幂等）后乘 inv_freq，
  `recomposition_frequencies`（L185-194）做**交错重组**：基组取 T 行，
  H 覆盖频率索引 {1,4,…,3·s1−2}、W 覆盖 {2,5,…,3·s2−1}，其余归 T；
  再 `cat(freqs, freqs)` 加倍成 64 宽。**三行恒等时重组是恒等映射**。
* **应用**（L655-690 `apply_rotary_pos_emb`）：q/k 前 64 维
  `q*cos + rotate_half(q)*sin`（NeoX 半分），尾部直通。
* 多模态封装（`Qwen3_5MoeModel` get_rope_index L1644-1645 起）产出
  行相异的 3 行位置——**域外分支**，本仓不核不建模（§5）。

### 2.2 vLLM 栏（@dc74b8cb）

* **文本位置生成**：`Qwen3_5ForCausalLMBase.get_mrope_input_positions`
  （qwen3_5.py L440-446）= `torch.arange(len(input_tokens)).unsqueeze(0)
  .expand(3, -1)`，起始偏移 0——SupportsMRoPE 的文本 fallback，三行
  恒等。decode 由引擎按缓存续位（positions = 上下文长度），与 HF
  `+past_seen` 同一序列。
* **mrope_positions_factor**（qwen3_vl.py L1821-1828）：仅当多模态
  **视频剪枝**启用返回 4（cos/sin 缓存放大），否则 **1**——文本
  positions 一维、缓存不放大。
* **构造分发**（rope_init L112-124）：`rope_type=default` 且含
  `mrope_section` → `MRotaryEmbedding`（带 `mrope_interleaved`、
  `positions_factor`）；`MRotaryEmbeddingInterleaved` 只在
  `openpangu` 分支（L334-345）构造，**Qwen3.6 不经过**。
* **MRotaryEmbedding**（rope_mrope.py）：构造断言
  `sum(mrope_section) == rotary_dim // 2`（L363）；`forward_native`
  （L375-436）与 `forward_cuda`（L472-504）在 **1D positions（文本）
  时直接走 `_forward_text_rope`（L438-470）= 标准部分 RoPE**，不做
  任何 section 重组；2D positions 才按 interleaved（`apply_interleaved_
  rope`，L235-245：idx%3==1→H、==2→W、界内）或 contiguous 重组。
  cos_sin_cache（rope_base L80-103）与 ApplyRotaryEmb NeoX
  （rope_common L133-173）与 HF 公式逐元素等价。
* **attention 接线**（qwen3_next.py L344-350 get_rope；L384-401
  supports_mrope 判定——`MRotaryEmbedding` + len(section)==3 +
  sum==rotary_dim//2 + mrope_interleaved；L423-426 fused 文本路径取
  `positions[0]`；L467 eager 调用）。

### 2.3 MTP 位置传递栏（@10a854ed，tdar speculator）

* **契约**：qwen3_5_mtp.py L55-62（positions 形状注释：3×seq 仅
  qwen2-vl 式多模态，否则 1×seq）与 L147-180（forward 把 positions
  直传 full_attention 草稿层——同一 MRotaryEmbedding 文本路径）。
* **prefill 同步**：`_prepare_prefill_inputs_kernel` L494-499 把
  target positions **逐元素复制** 进 draft positions；`_prefill`
  L351-355 `sample_src_positions = positions + 1`（采样键 = 被采样
  token 前一位置）。
* **草稿步进**（`_update_draft_inputs_kernel` L641-689）：每步采样后，
  非末步（s<K−1）`sample_src +1`（L653-655）、`positions +1` 夹
  `max_model_len−1`（L683-685）、`seq_len +1` 夹 max（L687-689）；
  **末步不推进**（L649-651）。草稿 KV 每步写 1 行，行号随位置。
* **下轮回退**（`_prepare_decode_inputs_kernel` L597-616）：
  `sample_src +1`（L601-603）；`seq_len = target_seq_len −
  num_rejected`（L605-607，被拒位置回退）再 +1（bonus token）夹 max；
  `positions +1` 夹 max−1（L612-615）。草稿 KV 行随 seq_len 回退，
  被拒行下轮重写（R23 已登记语义；内核只写 seq_lens，槽位回收由
  attn 后端按 seq_len 分配）。

### 2.4 三栏关系

HF 与 vLLM 的文本位置序列同源（arange/缓存续位）、旋转数学逐元素
相等（§3 fixture 证）；HF 无 MTP forward（R23 既有结论，mtp.* 被
`_keys_to_ignore_on_load_unexpected` 丢弃），MTP 栏只有 vLLM（SGLang
对照不涉及位置生成差异，R23 已核主干）。

## 3. 验证（fixture 数值对照 + 负例 + replay）

### 3.1 三路独立数学对照（ev-qwen36-mrope-cpu-numeric）

`tools/probe_qwen36_mrope_numeric.py`（hermetic 纯标准库 f64，本机
Python 3.12 执行；live 段 torch 2.15.0+cpu / transformers 5.22.1）：

* **A. HF 公式转写**（expand(3)+交错重组+加倍+rotate_half，按 pinned
  行号逐行转写）；**B. vLLM 公式转写**（cos_sin_cache+1D 文本路径+
  ApplyRotaryEmb NeoX）；**C. 独立标量 oracle**（每 NeoX 对复数旋转
  e^{i·p·theta_i}，与 A/B 无共享代码）。真实几何（theta 1e7、64/256、
  [11,11,10]）、随机 q/k（16 头×256，seed 20261116）、positions
  0..8 与上界 262143：**三路两两 max diff = 0.0（f64，atol 1e-12）**；
  2D 三行恒等（SupportsMRoPE fallback）与 1D 文本路径同样逐元素相等。
* **官方类实测**（live）：importlib 加载 pin 的 HF modeling，用真实
  rope 参数实例化 `Qwen3_5MoeTextRotaryEmbedding`，按 L1322-1332 构造
  4 行文本 position_ids 执行——cos/sin 与 16 头×9 位置旋转结果 vs
  oracle **max diff 4.35e-07**（官方类 fp32 计算路径，atol 1e-5）；
  decode 续位（past_seen=5）cos 差 1.9e-08。
* **MTP 状态机**：内核转移逐行转写（复制/+1/末步不推进/回退/夹紧），
  4 轮随机接受场景，≥16 项不变量断言（位置无跳变、末步不推进、
  kv_rows = 初始+步数且随回退同步、下轮 seq_len 公式）。
* **6 负例**：decode 重启 0（被 oracle 抓住，diff>1e-3）、步进 +2
  跳位、下轮不回退 num_rejected（seq_len 发散）、2D 异行走交错重组
  （与文本路径可区分——证明多模态是真实存在的另一分支，域外登记）、
  `sum(section)≠rotary_dim/2` 断言触发、max_model_len 夹紧边界
  （pos 封顶 23=max−1）。
* 落盘 `tests/fixtures/qwen36_mrope_r26_numeric.json`（11 个 pin 文件
  SHA256 一并绑定）；`tests/test_qwen36_mrope_r26.py` hermetic 消费
  （17 项：pin 完整性/行号锚/证据四栏/fixture 复算+测试内独立 mini-
  oracle/解析器负例/replay/CLI-API 一致性）+ live 可选重执行 1 项。

### 3.2 固定场景 replay（docs/evidence/mrope-map/）

before.json 于**干净 main ee70ce2** 一次性捕获（MTP 固定场景 +
DFlash 默认场景）；本分支 replay（`tools/replay_qwen36_mrope.py`）：

* **账本零漂移**：两场景 work totals/行数/memory/指标值（M01/M04 等）
  逐字段相等（digest 冻结：mtp 5b95dd04…、dflash 07983d65…）——
  "mrope 语义补齐不改算量"从主张变为机械断言；
* **仅证据字段按预期翻转**：M07 `structure_complete` False→True、
  `forward_source` partial-source→source-checked、capability
  model_structure_complete/model_forward_evidence → supported；
* **排名资格不变**：M02 仍不参与排名，理由不再含 mrope，改由
  CPU forward/GPU 实测/预算 unknown 阻断（`--check` 全量校验）。

### 3.3 CLI/API 一致性

`py -3.12 -m simverse.cli run-v2` 与 pipeline `run_v2` 同路径：M07
structure_complete=True、M02 排名阻断理由一致；catalog_json 仍恰两款
主模型（G8）。测试 `test_run_v2_consistent_with_catalog` 固化。

## 4. 落地（按已证实路径）

| 层 | 改动 |
|---|---|
| pin | `assets/identity/vllm_qwen36_mrope_pin/`（5 文件 + manifest，blob sha1 对 API 校验；.gitattributes LF） |
| ModelIR | `parse_qwen3_5_moe`/`_parse_hybrid` → `extras["rope"]`（几何+text_semantics.degenerate+MTP 位置步进 spec）；`sum(section)≠rotary_dim/2` 显式拒绝（与 vLLM 构造断言同口径）；config 键 `rope_parameters.mrope_section/mrope_interleaved` metadata-only → **mapped**（文本域） |
| source_map | `sm-qwen36-text-mrope`（HF/vLLM/MTP 三栏行号锚；branch_predicate 1D/2D 同行/2D 异行；uncovered_domain 四项） |
| 证据 | registry 新增 `ev-vllm-qwen36-mrope`（source-checked，executed=False）与 `ev-qwen36-mrope-cpu-numeric`（numeric-checked，fixture 级 executed=True）；catalog `forward_source=source-checked`、`unverified_paths=()`、checked_paths/evidence_refs 追加 |
| 计数/内存 | **零改动**（rope 行口径不变；replay 零漂移证明） |
| 文档 | 本报告 + IMPLEMENTATION_STATUS/capability-gaps/FOCUS_STATUS 同步 |

## 5. 剩余限制（不冒充）

1. **多模态域外**：2D 行相异（视觉 T/H/W 网格、`get_rope_index`/
   rope_deltas、视频剪枝 factor=4）未核未建模——文本退化结论**不得
   外推多模态**；视觉塔 config 仍 metadata-only。
2. **vLLM 模块未执行**：MRotaryEmbedding 模块本机不导入（公式转写
   三路对照 + 跨 commit 字节一致代替执行证据）；引擎调度循环
   （decode 迭代↔草稿步映射、采样键消费）未 replay，仅核内核转移
   语义；fused_qk_rmsnorm_rope_gate Triton 融合核未执行（文本取
   positions[0]，与 eager 同数学）。
3. **证据分层不变**：本轮为 fixture 级数值对照 + 源码核对；模型级
   `cpu_forward=not-run`、GPU=none、预算 unknown——**部署排名资格
   不变**（structure_complete 只表示源码证据完备，G15 门禁仍阻断）。
4. SGLang 栏未重复核位置生成（与 vLLM 主干同构，R23 已核；位置生成
   差异无栈间分歧点）。
5. 本轮不涉及 DFlash 草稿位置语义（DFlash 块扩散草稿自有协议，
   源 pin/程序图与 MTP 分立，R07a 在册）。

## 6. 复算命令

```bash
py -3.12 tools/probe_qwen36_mrope_numeric.py            # 重执行探针落盘 fixture
py -3.12 -m pytest tests/test_qwen36_mrope_r26.py -q    # hermetic 17 + live 1
py -3.12 tools/replay_qwen36_mrope.py --check           # 账本零漂移 + 证据翻转
py -3.12 -m pytest tests -q                             # 全量
py -3.12 -m simverse.cli run-v2                         # 默认场景
```
