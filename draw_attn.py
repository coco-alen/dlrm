import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import torch
import torch.nn.functional as F

attn = pickle.load(open('attn_noMask.pkl', 'rb'))
attn = attn.cpu().detach()
print(attn.shape)
batch = 1235
print(attn[batch,3,:,:])
for i in range(1,5):
    plt.subplot(2,2,i)
    sns.heatmap(attn[batch,i-1,:,:])
print(torch.mean(torch.abs(attn)))
plt.savefig("./plot.png")

# torch.zeros([27])
# print(F.softmax(torch.zeros([27]), dim=-1))