import time, torch, torch_npu
torch.npu.set_device(0)
GiB = 2**30
# 1 GiB tensor copy: read 1 GiB + write 1 GiB
n = GiB // 2  # 536M bf16 elements = 1 GiB
a = torch.empty(n, dtype=torch.bfloat16, device="npu")
b = torch.empty(n, dtype=torch.bfloat16, device="npu")
for _ in range(3): a.copy_(b)
torch.npu.synchronize()
it = 10
t0 = time.perf_counter()
for _ in range(it): a.copy_(b)
torch.npu.synchronize()
dt = (time.perf_counter()-t0)/it
print("copy 1GiB: %.3f ms -> %.1f GB/s (read+write)" % (dt*1e3, 2*GiB/dt/1e9))
# elementwise add: 2 read + 1 write
c = torch.empty(n, dtype=torch.bfloat16, device="npu")
for _ in range(3): torch.add(a, b, out=c)
torch.npu.synchronize()
t0 = time.perf_counter()
for _ in range(it): torch.add(a, b, out=c)
torch.npu.synchronize()
dt = (time.perf_counter()-t0)/it
print("add 1GiB: %.3f ms -> %.1f GB/s (2R+1W)" % (dt*1e3, 3*GiB/dt/1e9))
# softmax over 32k on small batch (decode-ish)
x = torch.randn(8, 32768, dtype=torch.bfloat16, device="npu")
for _ in range(3): torch.softmax(x, -1)
torch.npu.synchronize()
t0 = time.perf_counter()
for _ in range(50): torch.softmax(x, -1)
torch.npu.synchronize()
print("softmax 8x32k: %.4f ms" % ((time.perf_counter()-t0)/50*1e3))
print("BW_OK")