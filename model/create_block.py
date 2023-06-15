import numpy as np

# pytorch
import torch
import torch.nn as nn

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