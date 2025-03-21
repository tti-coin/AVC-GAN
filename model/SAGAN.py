import pdb

import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
from torch.nn import functional as F


class Generator(nn.Module):
    def __init__(self, args, device):
        super(Generator, self).__init__()
        self.enc_in = args.enc_in
        self.self_attn = args.self_attn
        self.device = device

        in_dim = args.d_model

        self.l1 = nn.Sequential(nn.Linear(in_dim, in_dim), nn.LeakyReLU(0.2))
        self.l2 = nn.Sequential(nn.Linear(in_dim, in_dim), nn.LeakyReLU(0.2))
        self.multihead_attn = MultiheadAttention(
            in_dim, args.n_heads, batch_first=True, dropout=0.1
        )
        self.l3 = nn.Sequential(nn.Linear(in_dim, args.d_model))

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        if self.self_attn:
            x_out, _ = self.multihead_attn(x, x, x)
            y = self.l3(x_out)
        else:
            y = self.l3(x)
        return y


class Discriminator(nn.Module):
    def __init__(self, args, device):
        super(Discriminator, self).__init__()
        self.device = device
        self.enc_in = args.enc_in
        self.self_attn = args.self_attn
        in_dim = args.d_model
        self.multihead_attn = MultiheadAttention(
            in_dim, args.n_heads, batch_first=True, dropout=0.1
        )
        self.l1 = nn.Sequential(nn.Linear(in_dim, 2 * in_dim), nn.LeakyReLU(0.2))
        curr_dim = in_dim * 2
        self.l2 = nn.Sequential(nn.Linear(curr_dim, curr_dim), nn.LeakyReLU(0.2))
        self.l3 = nn.Sequential(nn.Linear(curr_dim, curr_dim // 2), nn.LeakyReLU(0.2))
        curr_dim = curr_dim // 2
        self.l4 = nn.Linear(curr_dim, 1)

    def forward(self, x):
        if self.self_attn:
            x_out, _ = self.multihead_attn(x, x, x)
            h = self.l1(x_out)
        else:
            h = self.l1(x)
        h = self.l2(h)
        h = self.l3(h)
        y = self.l4(h)

        return y
