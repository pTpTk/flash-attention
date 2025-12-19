import torch
from flash_attn import flash_attn_func

q = torch.randn(1, 1, 64, 128, device="cuda", dtype=torch.float16)
k = torch.randn(1, 2048, 8, 128, device="cuda", dtype=torch.float16)
v = torch.randn(1, 2048, 8, 128, device="cuda", dtype=torch.float16)

out = flash_attn_func(q, k, v)

print(out.shape)
