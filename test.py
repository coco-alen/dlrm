import torch

a = torch.randn(2048, 2)
if a.shape[-1] == 2:
    a = torch.argmax(a, dim=-1).unsqueeze(-1)
    print(a)
    print(a.shape)