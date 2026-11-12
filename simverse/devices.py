"""设备与链路注册表（L1 输入）。数值逐条带来源；spec = 官方规格，assumed = 假设值。

口径（MERGE_CRITERIA §2）：
  * peak_by_dtype 一律 **dense**（结构稀疏 2:4 不启用）；sparse 值只写在 source
    说明里供对照，严禁进入计价。bf16 与 fp16 同档（Tensor Core 双倍速率家族）。
  * hbm_bw 为聚合单向读带宽；mem_capacity 为**标称**容量——可分配预算
    （allocatable）缺真实来源时保持 None，feasible 判定 unknown（lowering/memory）。
  * LINKS.bw 为**单向**每方向带宽；fixed_ms 为假设启动延迟（非规格值）。

2026-11-09 口径修正：L40 旧注释"90.5 dense ×2 稀疏=181"有误——按 NVIDIA L40
Datasheet (Jan-2023)，181.05 即 dense FP16/BF16，362.1 才是稀疏；数值本身未变，
标注修正依据官方数据表。H100/H200 同表 dense|sparse 双列转录。
"""
from .contracts import DeviceSpec, LinkSpec

DEVICES = {
    "h100_sxm": DeviceSpec(
        id="h100_sxm",
        label="NVIDIA H100 SXM 80GB",
        # dense：FP16/BF16 989.4 | sparse 1978.9；FP8/INT8 dense 1978.9 | sparse 3957.8
        # 来源 NVIDIA H100 SXM datasheet + 产品页（"1979 TFLOPS*"标注 *with sparsity）
        peak_by_dtype={"fp16": 989.5e12, "fp8": 1979e12, "int8": 1979e12},
        hbm_bw=3.35e12,
        mem_capacity=80e9,
        launch_overhead_ms=0.005,
        mfu_cap=0.75,
        card_count=8,
        sources={
            "peak_fp16": "spec:NVIDIA H100 SXM 数据表 dense FP16/BF16 Tensor 989.4 TFLOPS"
                         "（sparse 1978.9；旧仓 1979/2 换算一致）",
            "peak_fp8": "spec:NVIDIA H100 SXM dense FP8/INT8 Tensor 1978.9 TFLOPS"
                        "（sparse 3957.8）",
            "peak_int8": "spec:同 peak_fp8（H100 SXM INT8 与 FP8 同档 dense 1978.9 TOPS）",
            "hbm_bw": "spec:HBM3 3.35 TB/s 聚合（NVIDIA 数据表）",
            "mem_capacity": "spec:80 GB HBM3 标称容量（可分配预算未知→feasible=unknown）",
            "launch_overhead_ms": "assumed:5 µs/launch（CUDA Graph 未建模口径的量级假设）",
            "mfu_cap": "锚:旧仓 spec_gain_model.js:43 clusterFlops 0.75 上限",
        },
    ),
    "h200_sxm": DeviceSpec(
        id="h200_sxm",
        label="NVIDIA H200 SXM 141GB",
        # 计算与 H100 SXM 同档（Hopper 同核）；HBM3e 141GB/4.8TB/s
        # 来源 NVIDIA H200 datasheet：FP16/BF16 Tensor 990|1979（dense|sparse），
        # FP8/INT8 1979|3958；Lenovo H200 产品页同口径双列
        peak_by_dtype={"fp16": 989.5e12, "fp8": 1979e12, "int8": 1979e12},
        hbm_bw=4.8e12,
        mem_capacity=141e9,
        launch_overhead_ms=0.005,
        mfu_cap=0.75,
        card_count=8,
        sources={
            "peak_fp16": "spec:NVIDIA H200 SXM 数据表 dense FP16/BF16 Tensor ≈989.5 TFLOPS"
                         "（990 dense|1979 sparse；与 H100 SXM 同计算档）",
            "peak_fp8": "spec:NVIDIA H200 SXM dense FP8/INT8 Tensor ≈1979 TFLOPS"
                        "（1979 dense|3958 sparse）",
            "peak_int8": "spec:同 peak_fp8（H200 SXM INT8 与 FP8 同档 dense 1979 TOPS）",
            "hbm_bw": "spec:HBM3e 4.8 TB/s 聚合（NVIDIA 数据表）",
            "mem_capacity": "spec:141 GB HBM3e 标称容量（可分配预算未知→feasible=unknown）",
            "launch_overhead_ms": "assumed:同 h100_sxm 量级假设",
            "mfu_cap": "锚:同 h100_sxm（旧仓 0.75；H200 未单独校准）",
        },
    ),
    "l40": DeviceSpec(
        id="l40",
        label="NVIDIA L40 48GB",
        # NVIDIA L40 Datasheet (Jan-2023)："BF16/FP16 Tensor 181.05 | 362.1"、
        # "FP8 362 | 724"、"INT8 362 | 724"（左 dense｜右 with sparsity）。
        # 2026-11-09 修正：旧注"90.5 dense×2=181 稀疏"错误（90.5 为 FP32/TF32-dense）。
        peak_by_dtype={"fp16": 181.05e12, "fp8": 362e12, "int8": 362e12},
        hbm_bw=864e9,
        mem_capacity=48e9,
        launch_overhead_ms=0.006,
        mfu_cap=0.75,
        card_count=1,
        sources={
            "peak_fp16": "spec:NVIDIA L40 数据表 dense FP16/BF16 Tensor 181.05 TFLOPS"
                         "（sparse 362.1；旧仓表数值恰为 dense，注释口径已修正）",
            "peak_fp8": "spec:NVIDIA L40 数据表 dense FP8 362 TFLOPS（sparse 724）",
            "peak_int8": "spec:NVIDIA L40 数据表 dense INT8 362 TOPS（sparse 724）",
            "hbm_bw": "spec:864 GB/s GDDR6（NVIDIA 数据表）",
            "mem_capacity": "spec:48 GB GDDR6 标称容量（可分配预算未知→feasible=unknown）",
            "launch_overhead_ms": "assumed:6 µs/launch",
            "mfu_cap": "锚:同 h100_sxm",
        },
    ),
}

# V4.1 packed KV record family by registered device. Source: pinned
# deepseek_v41/attention.py L198-199 `_use_v41_mxfp8_kv_record` uses
# `is_device_capability_family(100)`; H100/H200 are Hopper SM90, L40 is Ada
# SM89. No SM100 device is registered, so sm100/nvfp4 records cannot be a
# deployable combination (ScenarioV2 validates target_device vs record).
V41_DEVICE_FAMILY = {
    "h100_sxm": "sm90",
    "h200_sxm": "sm90",
    "l40": "sm89",
}

LINKS = {
    # bw=单向；PCIe5 x16 理论单向 64 GB/s（128 GB/s 双向聚合，128b/130b 编码后有效略低）
    "pcie5": LinkSpec(id="pcie5", label="PCIe 5.0 x16", bw=64e9, fixed_ms=0.10,
                      source="spec:PCIe5 x16 理论单向 ~64 GB/s（双向聚合 128 GB/s）"
                             "；fixed_ms=assumed:100 µs 端到端启动延迟量级假设"
                             "（物理链路延迟为 µs 级，含软件栈开销的保守假设）"),
    # NVLink-C2C：Grace-Hopper 官方 900 GB/s 双向聚合 = 450 GB/s 单向
    "nvlink_c2c": LinkSpec(id="nvlink_c2c", label="NVLink-C2C（节点内）", bw=450e9,
                           fixed_ms=0.05,
                           source="spec:NVLink-C2C 900 GB/s 双向聚合 → 450 GB/s 单向"
                                  "（NVIDIA Grace-Hopper 白皮书）"
                                  "；fixed_ms=assumed:50 µs 启动延迟假设"),
    "eth_rdma": LinkSpec(id="eth_rdma", label="200G 以太网 RDMA", bw=25e9, fixed_ms=0.08,
                         source="spec:200GbE 线速 25 GB/s 单向载荷（编码前口径）"
                                "；fixed_ms=assumed:80 µs RDMA 启动延迟假设"),
    "custom": LinkSpec(id="custom", label="自定义链路", bw=100e9, fixed_ms=0.10,
                       source="assumed:可编辑占位（单向 100 GB/s / 100 µs）"),
}


def get_device(device_id: str) -> DeviceSpec:
    if device_id not in DEVICES:
        raise ValueError(f"未知设备 {device_id}；可用: {sorted(DEVICES)}")
    return DEVICES[device_id]


def get_link(link_id: str) -> LinkSpec:
    if link_id not in LINKS:
        raise ValueError(f"未知链路 {link_id}；可用: {sorted(LINKS)}")
    return LINKS[link_id]
