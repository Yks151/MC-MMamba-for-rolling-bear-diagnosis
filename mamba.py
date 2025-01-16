import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from einops import rearrange
from tqdm import tqdm
import os
import numpy as np

# 配置标识和超参数
USE_MAMBA = 1
DIFFERENT_H_STATES_RECURRENT_UPDATE_MECHANISM = 0

# 设定所用设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 人为定义的超参数
d_model = 8
state_size = 128  # 状态大小
seq_len = 100  # 序列长度
batch_size = 256  # 批次大小

# 定义S6模块
class S6(nn.Module):
    def __init__(self, seq_len, d_model, state_size, device):
        super(S6, self).__init__()
        self.fc1 = nn.Linear(d_model, d_model, device=device)
        self.fc2 = nn.Linear(d_model, state_size, device=device)
        self.fc3 = nn.Linear(d_model, state_size, device=device)

        self.seq_len = seq_len
        self.d_model = d_model
        self.state_size = state_size
        self.A = nn.Parameter(F.normalize(torch.ones(d_model, state_size, device=device), p=2, dim=-1))
        nn.init.xavier_uniform_(self.A)

        self.h = torch.zeros(batch_size, self.seq_len, self.d_model, self.state_size, device=device)

    def discretization(self, delta):
        dA = torch.exp(torch.einsum("bld,dn->bldn", delta, self.A))
        dB = torch.einsum("bld,bln->bldn", delta, self.B)
        return dA, dB

    def forward(self, x):
        self.B = self.fc2(x)
        self.C = self.fc3(x)
        delta = F.softplus(self.fc1(x))
        dA, dB = self.discretization(delta)

        h = torch.zeros(x.size(0), self.seq_len, self.d_model, self.state_size, device=x.device)
        h = torch.einsum('bldn,bldn->bldn', dA, h) + rearrange(x, "b l d -> b l d 1") * dB
        y = torch.einsum('bln,bldn->bld', self.C, h)
        return y

# 定义MambaBlock模块
class MambaBlock(nn.Module):
    def __init__(self, seq_len, d_model, state_size, device):
        super(MambaBlock, self).__init__()

        self.inp_proj = nn.Linear(d_model, 2 * d_model, device=device)
        self.out_proj = nn.Linear(2 * d_model, d_model, device=device)
        self.D = nn.Linear(d_model, 2 * d_model, device=device)

        self.out_proj.bias._no_weight_decay = True
        nn.init.constant_(self.out_proj.bias, 1.0)

        self.S6 = S6(seq_len, 2 * d_model, state_size, device)
        self.conv = nn.Conv1d(seq_len, seq_len, kernel_size=3, padding=1, device=device)
        self.conv_linear = nn.Linear(2 * d_model, 2 * d_model, device=device)
        self.norm = RMSNorm(d_model, device=device)

    def forward(self, x):
        x = self.norm(x)
        x_proj = self.inp_proj(x)

        # 1D卷积操作
        x_conv = self.conv(x_proj)
        x_conv_act = F.silu(x_conv)  # Swish激活
        x_conv_out = self.conv_linear(x_conv_act)

        # S6模块操作
        x_ssm = self.S6(x_conv_out)
        x_act = F.silu(x_ssm)  # Swish激活

        # 残差连接
        x_residual = F.silu(self.D(x))
        x_combined = x_act * x_residual
        x_out = self.out_proj(x_combined)

        return x_out

# 定义Mamba模型
class Mamba(nn.Module):
    def __init__(self, seq_len, d_model, state_size, device):
        super(Mamba, self).__init__()
        self.mamba_block1 = MambaBlock(seq_len, d_model, state_size, device)
        self.mamba_block2 = MambaBlock(seq_len, d_model, state_size, device)
        self.mamba_block3 = MambaBlock(seq_len, d_model, state_size, device)

    def forward(self, x):
        x = self.mamba_block1(x)
        x = self.mamba_block2(x)
        x = self.mamba_block3(x)
        return x

# 定义RMSNorm模块
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: str = 'cuda'):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output