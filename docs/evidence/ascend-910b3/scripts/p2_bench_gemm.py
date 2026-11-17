import time, os
import torch, torch_npu
print("torch", torch.__version__, "| torch_npu", torch_npu.__version__)
print("npu count:", torch.npu.device_count())
torch.npu.set_device(0)
print("device name:", torch.npu.get_device_name(0))
free, total = torch.npu.mem_get_info()
GiB = 2**30
print("torch.npu.mem_get_info: total=%.2f GiB free=%.2f GiB" % (total/GiB, free/GiB))

def bench(n, dtype, iters=10):
    a = torch.randn(n, n, dtype=dtype, device="npu")
    b = torch.randn(n, n, dtype=dtype, device="npu")
    for _ in range(3): torch.matmul(a, b)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): torch.matmul(a, b)
    torch.npu.synchronize()
    dt = (time.perf_counter() - t0) / iters
    flops = 2 * n**3
    print("matmul %5d %-8s %.3f ms  -> %.1f TFLOPS" % (n, str(dtype).split('.')[-1], dt*1e3, flops/dt/1e12))
    del a, b
    torch.npu.empty_cache()

for n in (2048, 4096, 8192):
    bench(n, torch.bfloat16)
bench(4096, torch.float16)
print("NPU_COMPUTE_OK")