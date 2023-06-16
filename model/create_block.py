import numpy as np
# pytorch
import torch
import torch.nn as nn
from model.transformer import TransformerBlock

def create_mlp(ln, sigmoid_layer):
    # build MLP layer by layer
    layers = nn.ModuleList()
    for i in range(0, ln.size - 1):
        n = ln[i]
        m = ln[i + 1]

        # construct fully connected operator
        LL = nn.Linear(int(n), int(m), bias=True)

        # initialize the weights
        # with torch.no_grad():
        # custom Xavier input, output or two-sided fill
        mean = 0.0  # std_dev = np.sqrt(variance)
        std_dev = np.sqrt(2 / (m + n))  # np.sqrt(1 / m) # np.sqrt(1 / n)
        W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
        std_dev = np.sqrt(1 / m)  # np.sqrt(2 / (m + 1))
        bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)
        # approach 1
        LL.weight.data = torch.tensor(W, requires_grad=True)
        LL.bias.data = torch.tensor(bt, requires_grad=True)
        # approach 2
        # LL.weight.data.copy_(torch.tensor(W))
        # LL.bias.data.copy_(torch.tensor(bt))
        # approach 3
        # LL.weight = Parameter(torch.tensor(W),requires_grad=True)
        # LL.bias = Parameter(torch.tensor(bt),requires_grad=True)
        layers.append(LL)

        # construct sigmoid or relu operator
        if i == sigmoid_layer:
            layers.append(nn.Sigmoid())
        else:
            layers.append(nn.ReLU())

    # approach 1: use ModuleList
    # return layers
    # approach 2: use Sequential container to wrap all layers
    return torch.nn.Sequential(*layers)


def create_transformer(ln, endActivation = nn.ReLU):
    # build MLP layer by layer
    layers = nn.ModuleList()
    for i in range(0, ln.size - 1):
        n = ln[i]
        m = ln[i + 1]

        if i == 0:
            LL = nn.Linear(int(n), int(m), bias=True)
            # initialize the weights
            mean = 0.0  # std_dev = np.sqrt(variance)
            std_dev = np.sqrt(2 / (m + n))  # np.sqrt(1 / m) # np.sqrt(1 / n)
            W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
            std_dev = np.sqrt(1 / m)  # np.sqrt(2 / (m + 1))
            bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)
            # approach 1
            LL.weight.data = torch.tensor(W, requires_grad=True)
            LL.bias.data = torch.tensor(bt, requires_grad=True)
        else:
            LL = TransformerBlock(
                in_dim = n,
                out_dim = m,
                num_heads = 8,
                ffn_expand_ratio=4.0,
                qkv_bias=False,
                qk_scale=None,
                drop=0.0,
                attn_drop=0.0,
                drop_path=0.0,
                act_layer=nn.GELU,
                norm_layer=nn.LayerNorm,
            )
        layers.append(LL)

    layers.append(endActivation())

    # approach 1: use ModuleList
    # return layers
    # approach 2: use Sequential container to wrap all layers
    return torch.nn.Sequential(*layers)
