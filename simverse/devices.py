"""设备与链路注册表（L1 输入）。数值逐条带来源；spec = 公开规格，assumed = 假设值。

数值与旧仓 hardware_params.py 对齐（对账基础）；差异在报告中逐条说明。
NPU（昇腾）口径属 M3 vllm-ascend 包，本切片不注册。
"""
from .contracts import DeviceSpec, LinkSpec

DEVICES = {
    "h100_sxm": DeviceSpec(
        id="h100_sxm",
        label="NVIDIA H100 SXM 80GB",
        peak_by_dtype={"fp16": 989.5e12, "fp8": 1979e12, "int8": 1979e12},
        hbm_bw=3.35e12,
        mem_capacity=80e9,
        launch_overhead_ms=0.005,
        mfu_cap=0.75,
        card_count=8,
        sources={
            "peak_fp16": "spec:H100 SXM 数据手册 dense FP16 989.5 TFLOPS（旧仓 hardware_params nvidia_H100_SXM 取 1979/2）",
            "peak_fp8": "spec:H100 SXM FP8 dense ≈ 1979 TFLOPS（与 INT8 同档）",
            "hbm_bw": "spec:HBM3 3.35 TB/s（NVIDIA 数据手册）",
            "mem_capacity": "spec:80 GB HBM3",
            "launch_overhead_ms": "assumed:5 µs/launch（CUDA Graph 未建模口径的量级假设）",
            "mfu_cap": "锚:旧仓 spec_gain_model.js:43 clusterFlops 0.75 上限",
        },
    ),
    "l40": DeviceSpec(
        id="l40",
        label="NVIDIA L40 48GB",
        peak_by_dtype={"fp16": 181e12, "fp8": 362e12, "int8": 362e12},
        hbm_bw=864e9,
        mem_capacity=48e9,
        launch_overhead_ms=0.006,
        mfu_cap=0.75,
        card_count=1,
        sources={
            "peak_fp16": "spec:L40 90.5 TFLOPS dense FP16×2(稀疏)=181（旧仓 hardware_params nvidia_L40）",
            "peak_fp8": "spec:L40 FP8 362 TOPS 口径（旧仓表）",
            "hbm_bw": "spec:864 GB/s GDDR6",
            "mem_capacity": "spec:48 GB GDDR6",
            "launch_overhead_ms": "assumed:6 µs/launch",
            "mfu_cap": "锚:同 h100_sxm",
        },
    ),
}

LINKS = {
    "pcie5": LinkSpec(id="pcie5", label="PCIe 5.0 x16", bw=64e9, fixed_ms=0.10,
                      source="spec:PCIe5 x16 理论 ~63 GB/s（旧仓 spec_gain_presets DT_LINKS）"),
    "nvlink_c2c": LinkSpec(id="nvlink_c2c", label="NVLink-C2C（节点内）", bw=450e9, fixed_ms=0.05,
                           source="assumed:GB200 节点内互联（旧仓 DT_LINKS 同源）"),
    "eth_rdma": LinkSpec(id="eth_rdma", label="200G 以太网 RDMA", bw=25e9, fixed_ms=0.08,
                         source="spec:200GbE ≈ 25 GB/s 单向"),
    "custom": LinkSpec(id="custom", label="自定义链路", bw=100e9, fixed_ms=0.10, source="assumed"),
}


def get_device(device_id: str) -> DeviceSpec:
    if device_id not in DEVICES:
        raise ValueError(f"未知设备 {device_id}；可用: {sorted(DEVICES)}")
    return DEVICES[device_id]


def get_link(link_id: str) -> LinkSpec:
    if link_id not in LINKS:
        raise ValueError(f"未知链路 {link_id}；可用: {sorted(LINKS)}")
    return LINKS[link_id]
