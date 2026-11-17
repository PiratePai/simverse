import torch, torch_npu
torch.npu.set_device(0)
def run(M, K, N, it=20):
    a = torch.randn(M, K, dtype=torch.bfloat16, device="npu")
    w = torch.randn(K, N, dtype=torch.bfloat16, device="npu")
    for _ in range(3): torch.matmul(a, w)
    torch.npu.synchronize()
    for _ in range(it): torch.matmul(a, w)
    torch.npu.synchronize()
run(1, 2048, 1024)
run(64, 2048, 1024)
run(512, 2048, 1024)
run(1, 2048, 31040)
print("done")