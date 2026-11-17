# Ascend forward 路径核对与离线采集入口（2026-11-17）

## 0. 结论

**本次改动前，仓库没有 Ascend 整模 forward 入口；本次新增的入口尚未做 NPU 验证。** 现有
`tools/probe_qwen36_forward_r06.py` 是 HF CPU 两层随机权重探针；PR21 的
`docs/evidence/ascend-910b3/` 是 `torch_npu` 算子级代理与实测，不是完整模型
forward。二者都不能回答“Qwen3.6-35B-A3B / DeepSeek-V4.1 在 Ascend 上真实
forward 是否接通”。

本次新增 `tools/ascend_forward_probe.py`：默认只做 dry-run 计划，显式
`--execute` 才通过 **vLLM `LLM` + 已安装的 vllm-ascend 平台插件** 运行真实
模型 forward。它不手抄 CUDA/Ascend forward，也不按模型名字做适配。

- **Qwen3.6 fixture 路径**：2 层、text-only、随机权重、`load_format=dummy`，
  用于在匹配 Ascend 栈上验证 Qwen3.6 文本 backbone 的真实 vLLM forward 接线；
  不是整模性能，也不是接受率。
- **Qwen3.6 真实 checkpoint 路径**：本地 Qwen3.6-35B-A3B 完整 checkpoint，
  由 vLLM 加载；默认 TP2 + expert parallel，官方文档明确 DP1/TP2；A2 8卡 / A3 16卡描述整机规格，不是参与卡数。
- **DeepSeek-V4.1**：记录真实注册与设备/量化门禁，但本单进程 runner
  **明确阻断执行**；V4/V4-Flash 的 checkpoint 与四卡 e2e 测试被拒绝冒充 V4.1。

## 1. 源码与测试核对

| 项 | 证据（本地只读参考） | 判断 |
|---|---|---|
| R06 探针 | `tools/probe_qwen36_forward_r06.py` L1-19、L104-305：importlib 加载 pin 的 HF `modeling_qwen3_5_moe.py`，32 隐藏维、2 层、随机权重、CPU | HF CPU fixture，不是 Ascend forward |
| PR21 证据 | `docs/validation/ascend-910b3-20261117.md` （见该报告的重新审查结论）：算子级 ACL/torch 池、HCCL、SwiGLU/MoE/MRoPE 代理 | `torch_npu` 算子代理，不是完整模型 forward |
| vLLM Qwen3.6 注册 | `D:/Github/vllm/vllm/model_executor/models/registry.py` L203-204（`Qwen3_5MoeForCausalLM`）、L596-599（`Qwen3_5MoeForConditionalGeneration`） | 真实模型类在 vLLM 中，`qwen3_5` 模块 |
| vllm-ascend Qwen3.6 插件 | `vllm_ascend/patch/worker/patch_qwen3_5.py` L205 替换 `Qwen3NextAttention.forward`、L206-225 patch GDN；`vllm_ascend/patch/worker/__init__.py` L39 导入 | Ascend 注意力/GDN 路径由插件接管 |
| Qwen3.6 官方支持 | vllm-ascend `docs/source/tutorials/models/Qwen3.6-35B-A3B.md` L9（v0.18.0rc1 起）、L23-24（A3 16×64GB / A2 8×64GB）、L195-196（TP2 + EP）、L219（W8A8 用 `--quantization ascend`） | 官方文档支持 Qwen3.6-35B-A3B，A2/A3 |
| Qwen3.6 身份测试 | `tests/e2e/pull_request/two_card/model_runner_v2/test_qwen3_6_mtp.py` L24 使用 `Qwen/Qwen3.6-35B-A3B`，L102 TP2 MTP acceptance | 真实 Qwen3.6-35B-A3B 测试存在 |
| 不可偷换的测试 | `tests/e2e/pull_request/two_card/test_qwen3_5_35b_a3b_w8a8.py` 使用 `Eco-Tech/Qwen3.5-35B-A3B-w8a8-mtp`；`tests/e2e/pull_request/four_card/test_deepseek_v4.py` 使用 `gdydems/DeepSeek-V4-Flash-w4a8-mtp` | Qwen3.5 ≠ Qwen3.6；V4 ≠ V4.1 |
| V4.1 vLLM 注册 | `vllm/model_executor/models/registry.py` L378-381：`DeepseekV41ForCausalLM` → `vllm.models.deepseek_v41` | 真实 V4.1 模型类 |
| V4.1 Ascend 注册 | vllm-ascend `vllm_ascend/models/__init__.py` L40-44：`DeepseekV41ForCausalLM` → `vllm_ascend.models.deepseek_v41.vl_model:AscendDeepseekV41ForCausalLM`，条件 `not vllm_version_is("0.29.0")`；L58-62 同样条件注册 DSpark draft | 安装 vLLM 0.29.0 时 Ascend override **不会注册** |
| V4.1 门禁 | vllm-ascend `docs/source/tutorials/models/DeepSeek-V4.1-Flash.md` L19-21（W8A8；2×A3 或 4×A2）、L30-33（验证拓扑 W8A8/TP8/DP4/EP32/DSpark）、L237-240（`--quantization ascend`）、L475-483（限制） | 常规多节点 W8A8；另有单 A3 Engram host offload，均不等于双910B3 |

结论：Qwen3.6 与 V4.1 的 **模型实现和 Ascend 插件都在上游 vLLM/vllm-ascend
中存在**；本仓缺的是“用这套栈实际跑 forward 并留 artifact”的入口，而不是又一个
手写 forward。

## 2. 交付入口

### 2.1 Qwen3.6 单/少卡随机权重少层 fixture

```bash
# 默认 dry-run：不导入 torch/vllm、不加载权重、不写模型文件
py -3.12 tools/ascend_forward_probe.py --profile qwen36-fixture \
  --out .tmp/r28/ascend_forward/qwen36_fixture_plan.json

# 在安装匹配 Ascend 栈的机器上显式执行（TP1 或 TP2）
ASCEND_RT_VISIBLE_DEVICES=0,1 \
python3 tools/ascend_forward_probe.py --profile qwen36-fixture --execute \
  --tensor-parallel-size 2 \
  --out .tmp/r28/ascend_forward/qwen36_fixture_run.json
```

执行时工具把
`assets/identity/ascend_forward/qwen36_35b_a3b_text_fixture_config.json`
的 `config` 写到 `.tmp/r28/ascend_forward/qwen36_fixture_model/config.json`，
再调用：

```python
LLM(
    model=<runtime config dir>,
    load_format="dummy",       # vLLM 生成随机权重，不读 checkpoint
    skip_tokenizer_init=True,  # 只用 token IDs
    tensor_parallel_size=1,
    dtype="bfloat16",
    max_model_len=128,
    enforce_eager=True,
    enable_prefix_caching=False,
    ...
)
```

fixture 来源与覆盖边界：

| 项 | 值 |
|---|---|
| 源 | `assets/config/qwen36_35b_a3b_main__config.json` 的 `text_config` |
| 源 SHA256 | `93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99` |
| fixture 有效配置 SHA256（canonical JSON） | `fe26f62087773d54d76506da68b7142bf41b562ea51afe222b3401284d315721` |
| 覆盖 | 40→2 层、256→16 experts、vocab 1024、位置上限262144→4096、BOS/EOS与MTP；保留hidden2048、head256、GDN128、intermediate512、top-k8、MRoPE [11,11,10]，逐项在 `overrides` 中列出 |
| 未覆盖 | 原始 40 层/2048 hidden 身份由源 hash 保留；vision tower、image/video token、MTP、DFlash 明确排除 |
| 不能声称 | 整模性能、接受率、真实 checkpoint 精度、视觉能力 |

fixture 使用 `Qwen3_5MoeForCausalLM`（真实 Qwen3.6-35B-A3B wrapper 的
`language_model` 文本 backbone 类），并保留 text MRoPE 路径；它不经过
`Qwen3_5MoeForConditionalGeneration` 的多模态 wrapper，因此不能替代真实
checkpoint 路径。

覆盖边界：上游 vLLM 支持 text-only Qwen3.5/3.6 类，但 vllm-ascend 当前
Qwen3.6-35B-A3B 的 e2e（`test_qwen3_6_mtp.py`）跑的是多模态 wrapper。
因此 text-only fixture 在 Ascend 上是“源码可解释、但未被该 e2e 直接覆盖”
的路径；若它失败，2.2 的真实 wrapper 路径才是权威整模路径。

### 2.2 Qwen3.6 匹配真实本地 checkpoint 的完整路径

```bash
# 只准备/校验，不加载权重
python3 tools/ascend_forward_probe.py --profile qwen36-local \
  --model-path /data/weights/Qwen3.6-35B-A3B \
  --tensor-parallel-size 2 --enable-expert-parallel \
  --max-model-len 4096 \
  --out .tmp/r28/ascend_forward/qwen36_local_plan.json

# 官方文档含 TP2；实际显存容量和配套栈仍需执行验证
ASCEND_RT_VISIBLE_DEVICES=0,1 \
python3 tools/ascend_forward_probe.py --profile qwen36-local \
  --model-path /data/weights/Qwen3.6-35B-A3B \
  --tensor-parallel-size 2 --enable-expert-parallel \
  --max-model-len 4096 --execute \
  --out .tmp/r28/ascend_forward/qwen36_local_run.json
```

可见参与卡数必须至少为 TP；不要求暴露整机所有卡。

匹配栈注意：vllm-ascend baa023a 的 `requirements.txt` 是
`torch==2.10.0` / `torch-npu==2.10.0.post4`、CANN 9.1.0。PR21 证据机的
torch 2.9.0 / torch_npu 2.9.0.post9 **不是**这套参考栈；`--execute` 会检查
torch/torch-npu 2.10 major/minor floor 并拒绝旧栈，dry-run 仍可生成计划。

- 只接受 `architectures=["Qwen3_5MoeForConditionalGeneration"]`、
  `model_type="qwen3_5_moe"`、`text_config.model_type="qwen3_5_moe_text"`。
- `config.json` 与 pinned main hash `93a469...efe99` 相同则 `pass`；不同
  （例如 W8A8/转换版本）文件哈希不同只 `warn`，但文本结构必须逐项匹配 pin；artifact 不声称权重 revision 已核验。
- 要求本地 `config.json`、权重 shard（含 index 的完整性）、tokenizer 文件；
  **不自动下载**。缺少文件直接拒绝。
- 默认 `trust_remote_code=False`；若本地 checkpoint 确实需要自带 modeling
  代码，显式加 `--trust-remote-code`（不会自动拉取远端代码）。
- BF16 路径 `quantization=None`；W8A8 路径要求
  `quant_model_description.json` 包含 W8A8 条目才自动用 `--quantization ascend`；空 JSON 拒绝，完整张量覆盖与格式由后端加载再校验。
- 实际预检要求每个参与 rank 是所核 A2/A3 型号、可见卡数≥TP；未知型号拒绝。
  TP2 有上游文档/测试依据，但本探针在 910B3 和所装栈上的容量/数值仍未验证。
- 真实 checkpoint 路径会经过 `Qwen3_5MoeForConditionalGeneration`，包括
  视觉塔构造；文本 prompt 由 checkpoint 自带 tokenizer 编码。视觉/图像输入
  不在本探针范围。
- 探针默认 `enforce_eager=True`，不捕获 ACLGraph；生产
  `FULL_DECODE_ONLY` 图路径不在本次采集范围。

### 2.3 DeepSeek-V4.1：真实注册、门禁与安全阻断

```bash
# 只生成 blocked 计划；不执行
python3 tools/ascend_forward_probe.py --profile deepseek-v41 \
  --model-path /data/weights/DeepSeek-V4.1-Flash-w8a8 \
  --out .tmp/r28/ascend_forward/v41_plan.json
```

工具会检查并写入 artifact：

| 门禁 | 规则 |
|---|---|
| 模型 id | `deepseek-ai/DeepSeek-V4.1-Flash`；本地没有 revision pin，plan 标 `warn` |
| 架构/身份 | 只接受 `DeepseekV41ForCausalLM` / `deepseek_v41`；`DeepseekV4ForCausalLM`、`DeepseekV4ForConditionalGeneration` 或 `deepseek_v4` 直接 `v4_substitution` 拒绝 |
| vLLM 版本 | vllm-ascend 的 override 条件为 `not vllm_version_is("0.29.0")`；安装 vLLM 0.29.0 时 gate 标 fail |
| vLLM-Ascend 版本 | 要求 >= 0.26.0（本地 baa023a 文档栈） |
| 量化 | 文档验证路径是 W8A8 + ModelSlim；要求 `quant_model_description.json`，使用 `--quantization ascend` |
| 设备/拓扑 | 文档验证：2×Atlas 800 A3 或 4×Atlas 800 A2，DP4/TP8/EP32，多节点；本机 2×910B3 不满足 |
| 执行 | 本 runner 为单进程离线探针，**永远不执行 V4.1**，只给出门禁与 `vllm serve` 参考入口 |

`gates` 中会列出实际注册证据（vLLM registry L378-381、vllm-ascend
`models/__init__.py` L40-44）。V4.1 的 serve 参数参考
`docs/source/tutorials/models/DeepSeek-V4.1-Flash.md` 的 DP4/TP8/EP32、W8A8、
`--tokenizer-mode deepseek_v41`、`--quantization ascend`、`FULL_DECODE_ONLY`、
DSpark 配置；本工具不伪造多节点启动，也不把 V4/V4-Flash 当作 V4.1。

## 3. 门禁矩阵

| Gate | dry-run 行为 | `--execute` 行为 |
|---|---|---|
| profile/architecture/model_type | 读取 `config.json` 并精确匹配；跨模型/相似名拒绝 | 同一套检查，不通过不导入 vLLM |
| 本地路径 | 必须是已存在目录；缺失直接拒绝，不下载 | 同左 |
| checkpoint 身份 | local `config.json` 与 pinned `93a469...efe99` 比对；不同则 warn | 同左 |
| 权重 shard/index | 检查文件存在与 index 引用完整性 | 同左，然后由 vLLM 加载 |
| tokenizer | 真实 checkpoint 要求本地 tokenizer；fixture 用 token IDs | 同左 |
| quantization | BF16 或 ModelSlim W8A8；`ascend` 必须有 ModelSlim 签名 | 同左，写入 LLM `quantization` |
| device | 只允许 `npu`；`cuda`/`cpu` 拒绝 | 另查 `current_platform.device_type == "npu"` |
| dtype | 只允许 `bfloat16`（A2/A3 文档路径） | 同左 |
| token IDs | 非空、非负、`< vocab_size`、prompt+max_tokens ≤ max_model_len | 同左 |
| TP | fixture 1/2；local 1/2/4/8；V4.1 仅 8；检查 attention/GDN 头整除 | 另查 `torch.npu.device_count()` |
| 版本 | Qwen3.6 vLLM/vllm-ascend ≥0.18；V4.1 ≥0.26 且 vLLM 非 0.29.0 | 导入后复核 `vllm.__version__`、torch/torch-npu 2.10 floor |
| 拓扑 | 文档 DP1/TP2；整机规格不等于参与卡数 | 可见卡数≥TP，每个参与 rank 为所核 A2/A3 型号 |
| V4.1 执行 | `execute_blocked=true` | 直接 `execute_blocked`，不启动引擎 |

## 4. 来源、提交与哈希

| 来源 | 提交/版本 | 说明 |
|---|---|---|
| vLLM 本地只读参考 | `6b858751f6e99c4f33913657dc0c2afd34ff13e6`（2026-09-21） | 含 `qwen3_5` 与 `deepseek_v41`；**不是** vllm-ascend 文档 pin 的 `84030bbe3d74d99bad477a3d2e37a973ccd8865c` |
| vllm-ascend 本地只读参考 | `baa023ada41a5b889cac6249845a688aab130892`（2026-09-21） | 干净参考；未修改 |
| vllm-ascend verified main pin | `84030bbe3d74d99bad477a3d2e37a973ccd8865c` | 来自 `.github/vllm-main-verified.commit` |
| vllm-ascend release tag | `v0.29.0` | 来自 `.github/vllm-release-tag.commit`；V4.1 override 在此版本不注册 |
| Qwen3.6 仓库/修订 | `Qwen/Qwen3.6-35B-A3B` @ `345d467b117d1fd5eda9b879a5988de9ae5f47a7` | `assets/identity/required_scenarios_identity.json` L2-4 |
| Qwen3.6 配置身份 | `93a4693f...efe99` | `assets/config/qwen36_35b_a3b_main__config.json` |
| fixture 配置身份 | 文件与有效配置完整哈希由每次 plan 输出 | `assets/identity/ascend_forward/qwen36_35b_a3b_text_fixture_config.json` |
| 来源清单 | 完整 SHA256 由 plan 输出 | `assets/identity/ascend_forward/source_manifest.json`（含 13 个源码/文档/测试文件 SHA256（包括主控补核硬件识别源码）） |

说明：

- 本地 vLLM `6b8587` 比 vllm-ascend 文档 pin 的 `84030bbe` 晚 10 天，二者都含
  Qwen3.6/V4.1 符号；本次是 **source-checked**，不是“与官方验证 pin 完全一致”。
  真正部署应使用 vllm-ascend 对应 release image 或精确的 `84030bbe`。
- 任务提供的远端 vllm-ascend HEAD `915d58873c0854c0550b56f66a126ca2cb114e71`
  未在本地 fetch；本报告不把其内容当作已验证证据。
- 外部参考仓未修改。

## 5. 测试与退出码

```text
py -3.12 -m pytest tests/test_ascend_forward_probe.py -q
21 passed
exit code 0

py -3.12 -m pytest tests/test_qwen36_forward_r06.py -q
12 passed（DSH 交付时运行）
exit code 0
```

主控复核新增7项回归；最终 forward 21项与证据5项联合运行：26 passed，无 NPU。

```text
py -3.12 tools/ascend_forward_probe.py --profile qwen36-fixture --out ...   # exit 0
py -3.12 tools/ascend_forward_probe.py --profile qwen36-local --model-path ... --out ...  # exit 0
py -3.12 tools/ascend_forward_probe.py --profile deepseek-v41 --model-path ... --out ...  # exit 0, blocked plan
py -3.12 tools/ascend_forward_probe.py --profile deepseek-v41 --model-path ... --execute
REFUSED [execute_blocked] ...; exit 2
```

新增测试覆盖：

- dry-run 构造 plan 时用 `builtins.__import__` 守卫证明未导入
  `torch`/`torch_npu`/`vllm`，也未加载权重；
- fixture 源 hash 与各项 override 逐项一致，pinned HF config 可实例化；
- 架构/model_type、Token 越界/空/负数、TP、device、dtype、量化签名、
  vLLM/vllm-ascend 版本、V4/V4.1 替代均拒绝；
- V4.1 在 vLLM 0.29.0 下 gate fail，且 `execute_blocked`；
- monkeypatch stub LLM 验证执行接线：engine args、prompt token IDs、
  artifact 输出与 runtime config hash；artifact 明确
  `status="executed-stub"`、`hardware_executed=false`、`hardware_verified=false`。

## 6. 限制与非声称

1. **没有硬件执行**：本环境没有可用 Ascend NPU + 匹配 vLLM/vllm-ascend 栈，
   未运行真实 GPU/NPU 模型；stub 测试只证明接线，不证明 kernel 可跑。
2. **fixture 不是整模**：2 层、2048 hidden、随机权重、text-only；不能用于
   性能、接受率、精度或产品结论。真实整模 forward 路径已提供但未运行。
3. **V4.1 未执行**：本 runner 明确阻断；2×910B3 不满足官方 2×A3/4×A2
   常规 W8A8 多节点拓扑；单 A3 host-offload 另需 TP8/DP2/EP16。V4/V4-Flash 的测试与 checkpoint 不是 V4.1 证据。
4. **版本不精确**：source audit 基于 vLLM `6b8587`，不是 vllm-ascend 文档
   验证 pin `84030bbe`；执行时工具会检查版本 floor 与 vLLM 0.29.0 排除条件，
   但不会把 `6b8587` 宣称为已验证组合。
5. **范围**：MTP/DFlash/vision/图像输入、接受率、性能基准、远程 GPU 调用均
   不在本工具范围；工具不接入 simverse 仿真排名，也不改变产品推理。
6. **拓扑**：Qwen3.6 文档有 TP2；少卡容量及具体栈仍待测。单/四/八卡等结构允许值
   不自动成为已验证部署。

## 7. 交付文件

- `tools/ascend_forward_probe.py`
- `tests/test_ascend_forward_probe.py`
- `docs/validation/ascend-forward-paths-20261117.md`
- `assets/identity/ascend_forward/qwen36_35b_a3b_text_fixture_config.json`
- `assets/identity/ascend_forward/source_manifest.json`


## 主控复核补充：部署替代路线与声明边界

- V4.1 官方文档 §5.3 还有单 A3 路线：TP8/DP2/EP16、INT8 Engram、
  `--engram-config '{"cpu_offload": true}'`。不是所有 V4.1 部署都必须多节点；
  此工具也未实现该启动方式。双910B3不能代替该拓扑。
- 权重 SHA256 尚未采集，artifact 标 null / `weights_identity_verified=false`。
  同架构不证明 Qwen3.6 权重身份，仍需本地权重清单、revision/hash。
- `sources.commit` 是审阅参考，不是执行 commit；实际 runtime commit 未采得为 null。
  版本下限不证明配套组合兼容；参考版本 torch 2.10 / CANN9.1 并非任意版本承诺。
- `hardware_verified` 仅指设备/平台预检，`executed` 仅指 generate 返回；不表示
  数值正确或性能准入。wall time 仅为 LLM.generate 的 host 边界。
- 执行设置 HF/Transformers offline；fixture 拒绝覆盖不同 config 或含权重目录。
- 主控回归覆盖真实构造器关键字参数、registry resolve API、TP2预检/不支持卡、
  原配置保护、结构身份、空量化描述与重放参数保真；这些均为离线桩测试。

源码与测试固定链接：
[vLLM registry](https://github.com/vllm-project/vllm/blob/6b858751f6e99c4f33913657dc0c2afd34ff13e6/vllm/model_executor/models/registry.py)、
[Ascend Qwen patch](https://github.com/vllm-project/vllm-ascend/blob/baa023ada41a5b889cac6249845a688aab130892/vllm_ascend/patch/worker/patch_qwen3_5.py)、
[Qwen3.6 部署文档](https://github.com/vllm-project/vllm-ascend/blob/baa023ada41a5b889cac6249845a688aab130892/docs/source/tutorials/models/Qwen3.6-35B-A3B.md)、
[Qwen3.6 双卡测试](https://github.com/vllm-project/vllm-ascend/blob/baa023ada41a5b889cac6249845a688aab130892/tests/e2e/pull_request/two_card/model_runner_v2/test_qwen3_6_mtp.py)、
[V4.1 部署及 host offload](https://github.com/vllm-project/vllm-ascend/blob/baa023ada41a5b889cac6249845a688aab130892/docs/source/tutorials/models/DeepSeek-V4.1-Flash.md)。
源码阅读没有冒充上游测试运行。实际测试结果以[实施状态](../IMPLEMENTATION_STATUS.md)为准。
