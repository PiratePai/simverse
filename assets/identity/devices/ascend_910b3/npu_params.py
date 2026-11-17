"""
NPU (昇腾 Ascend) 硬件参数

数据来源: CF++.xlsx hardware sheet, BF16 列 = cube+vector 总和
cube/vector 按 8:1 拆分 (与 950PR 的 CF++.xlsx 拆分比率一致)

NPU 与 GPU 的关键区别：
  - Cube 单元：专用矩阵乘法加速器，处理 GEMM (BF16/FP16)
  - Vector 单元：通用向量计算单元，处理 GEMV/Softmax/Norm 等 (FP32)
  - 有效性能公式：Performance = num_NPU × min(peak_perf × MFU, ops/byte × bw × MBU)
"""

npu_params = {
    # ====== 昇腾 910B (32GB) ======
    "ascend_910B": {
        "cube_ops": 249e12,
        "vector_ops": 31e12,
        "bandwidth": 800e9,
        "memory_capacity": 32e9,
        "onchip_buffer": 0,
        "mfu": 0.20,
        "mbu": 0.20,
        "num_npus": 8,
        "interconnect_bw": 50e9,    # HCCS 有效互联带宽 (all-reduce, GB/s → B/s)
    },

    # ====== 昇腾 910B (64GB) ======
    "ascend_910B_64G": {
        "cube_ops": 249e12,
        "vector_ops": 31e12,
        "bandwidth": 1.6e12,
        "memory_capacity": 64e9,
        "onchip_buffer": 0,
        "mfu": 0.20,
        "mbu": 0.20,
        "num_npus": 8,
        "interconnect_bw": 50e9,
    },

    # ====== 昇腾 910C ======
    "ascend_910C": {
        "cube_ops": 498e12,
        "vector_ops": 62e12,
        "bandwidth": 3.2e12,
        "memory_capacity": 128e9,
        "onchip_buffer": 0,
        "mfu": 0.20,
        "mbu": 0.20,
        "num_npus": 8,
        "interconnect_bw": 100e9,   # 下一代 HCCS, 双倍带宽
    },

    # ====== 昇腾 910B2：旗舰训练/推理卡 (来自 CF++.xlsx) ======
    "ascend_910B2": {
        "cube_ops": 1843.2e12,
        "vector_ops": 64e12,
        "bandwidth": 192e9,
        "memory_capacity": 128e9,
        "onchip_buffer": 0,
        "mfu": 0.20,
        "mbu": 0.20,
        "num_npus": 8,
        "interconnect_bw": 50e9,
    },

    # ====== 昇腾 950PR (与 CF++.xlsx 一致; BF16 486T 按 8:1 拆 = cube 432 + vector 54;
    #       FP16/INT8 通道来自官方规格表: FP16 486T, FP8/INT8 919T) ======
    "ascend_950PR": {
        "cube_ops": 432e12,
        "vector_ops": 54e12,
        "FP16": 486e12,
        "INT8": 919e12,
        "bandwidth": 1.6e12,
        "memory_capacity": 128e9,
        "onchip_buffer": 0,
        "mfu": 0.20,
        "mbu": 0.20,
        "num_npus": 8,
        "interconnect_bw": 100e9,   # 新一代互联
    },

    # ====== 昇腾 950DT (官方规格表: BF16/FP16 547T, FP8/INT8 1034T, 144GB, 4TB/s;
    #       cube/vector 按 8:1 拆分, 与 950PR 同代际约定) ======
    "ascend_950DT": {
        "cube_ops": 486.2e12,       # BF16 547T × 8/9 (cube)
        "vector_ops": 60.8e12,      # BF16 547T × 1/9 (vector)
        "FP16": 547e12,
        "INT8": 1034e12,
        "bandwidth": 4e12,
        "memory_capacity": 144e9,
        "onchip_buffer": 0,
        "mfu": 0.20,
        "mbu": 0.20,
        "num_npus": 8,
        "interconnect_bw": 100e9,   # 同 950PR 代际
    },

    # ====== 昇腾 A5 (910C 类商用卡; 数值来自 scenarios/deepseek_v41_flash_spec.py
    #       的硬件设定: FP16 320T / INT8 640T / HBM 3.2TB/s / 128GB / HCCS 200GB/s) ======
    "ascend_a5": {
        "cube_ops": 320e12,          # FP16 稠密 (spec 脚本口径)
        "vector_ops": 40e12,         # Vector (按 910C 的 cube:vector ≈ 8:1 推算)
        "FP16": 320e12,              # 显式通道: 供 matrix/magi 等按 FP16 峰值取用
        "INT8": 640e12,              # INT8 TOPS (spec 脚本口径, w≤8 时 _hw 取此档)
        "bandwidth": 3.2e12,
        "memory_capacity": 128e9,
        "onchip_buffer": 0,
        "mfu": 0.20,
        "mbu": 0.20,
        "num_npus": 8,
        "interconnect_bw": 200e9,    # HCCS 200GB/s (spec 脚本口径)
    },
}

avaliable_npu_hardwares = list(npu_params.keys())
