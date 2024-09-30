import pdb

import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
from torch.nn import functional as F

# class Self_Attn(nn.Module):
#     def __init__(self, in_dim, seq_len):
#         super(Self_Attn, self).__init__()
#         self.in_dim = in_dim
#         self.seq_len = seq_len

#         # Define linear layers for query, key, and value projections
#         self.query_linear = nn.Linear(in_dim, in_dim)
#         self.key_linear = nn.Linear(in_dim, in_dim)
#         self.value_linear = nn.Linear(in_dim, in_dim)
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x):
#         # Project inputs to queries, keys, and values
#         proj_query = self.query_linear(x)
#         proj_key = self.key_linear(x)
#         proj_value = self.value_linear(x)

#         # Compute attention scores
#         energy = torch.bmm(proj_query, proj_key.transpose(1, 2))
#         attention = self.softmax(energy)

#         # Compute attention output
#         out = torch.bmm(attention, proj_value)

#         return out


# class MultiHeadAttn(nn.Module):
#     def __init__(self, in_dim, seq_len, n_heads):
#         super(MultiHeadAttn, self).__init__()
#         self.in_dim = in_dim
#         self.seq_len = seq_len
#         self.n_heads = n_heads

#         self.query_linear = nn.Linear(in_dim, in_dim)
#         self.key_linear = nn.Linear(in_dim, in_dim)
#         self.value_linear = nn.Linear(in_dim, in_dim)
#         self.out_linear = nn.Linear(in_dim, in_dim)
#         self.softmax = nn.Softmax(dim=-1)

#     def foward(self, x):
#         proj_query = self.query_linear(x)
#         proj_key = self.key_linear(x)
#         proj_value = self.value_linear(x)

#         # Split the query, key, and value projections into multiple heads
#         proj_query = proj_query.view(
#             x.size(0), self.n_heads, -1, self.in_dim // self.n_heads
#         )
#         proj_key = proj_key.view(
#             x.size(0), self.n_heads, -1, self.in_dim // self.n_heads
#         )
#         proj_value = proj_value.view(
#             x.size(0), self.n_heads, -1, self.in_dim // self.n_heads
#         )

#         # Compute attention scores
#         energy = torch.bmm(proj_query, proj_key.transpose(2, 3))
#         attention = self.softmax(energy)

#         out = torch.matmul(attention, proj_value)

#         # Concatenate the heads
#         out = out.view(x.size(0), -1, self.in_dim)

#         out = self.out_linear(out)

#         return out


# class MultiHeadAttn(nn.Module):
#     def __init__(self, in_dim, seq_len, n_heads):
#         super(MultiHeadAttn, self).__init__()
#         self.in_dim = in_dim
#         self.seq_len = seq_len
#         self.n_heads = n_heads

#         self.mha = MultiheadAttention(in_dim, n_heads, dropout=0.1)

#     def forward(self, x):
#         x = x.permute(1, 0, 2)
#         x, _ = self.mha(x, x, x)
#         x = x.permute(1, 0, 2)
#         return x


# base setting
class Generator(nn.Module):
    def __init__(self, args, device):
        super(Generator, self).__init__()
        self.enc_in = args.enc_in
        self.self_attn = args.self_attn
        self.device = device
        self.vari_emb_dim = args.d_model // 8
        self.vari_emb = nn.Embedding(args.enc_in, self.vari_emb_dim)

        in_dim = args.d_model + self.vari_emb_dim

        self.l1 = nn.Sequential(nn.Linear(in_dim, in_dim), nn.LeakyReLU(0.2))
        # self.l1 = nn.Sequential(
        #     nn.utils.spectral_norm(nn.Linear(in_dim, in_dim)),
        #     nn.BatchNorm1d(args.enc_in),
        #     nn.LeakyReLU(0.2),
        # )
        # self.mhattn = MultiHeadAttn(in_dim, args.enc_in, args.n_heads)
        self.multihead_attn = MultiheadAttention(
            in_dim, args.n_heads, batch_first=True, dropout=0.1
        )
        self.l2 = nn.Sequential(nn.Linear(in_dim, args.d_model))  # , nn.LeakyReLU(0.2))

        # curr_dim = 2 * in_dim
        # added on 2024/09/23, v3
        # self.l3 = nn.Sequential(nn.Linear(curr_dim, args.d_model))

    def forward(self, x):
        vari_emb = self.vari_emb(torch.arange(self.enc_in).to(self.device))
        vari_emb = vari_emb.unsqueeze(0).repeat(x.size(0), 1, 1)
        x = torch.cat((x, vari_emb), 2)

        x = self.l1(x)
        if self.self_attn:
            x_out, _ = self.multihead_attn(x, x, x)
        y = self.l2(x_out)

        return y


class Discriminator(nn.Module):
    def __init__(self, args, device):
        super(Discriminator, self).__init__()
        self.device = device
        self.enc_in = args.enc_in
        self.self_attn = args.self_attn
        self.vari_emb_dim = args.d_model // 8
        self.vari_emb = nn.Embedding(args.enc_in, self.vari_emb_dim)
        # nn.init.normal_(self.vari_emb.weight, std=0.02)

        in_dim = args.d_model + self.vari_emb_dim

        # self.mhattn = Self_Attn(in_dim, args.enc_in)
        self.multihead_attn = MultiheadAttention(
            in_dim, args.n_heads, batch_first=True, dropout=0.1
        )
        # self.l1 = nn.Sequential(nn .utils.spectral_norm(nn.Linear(in_dim, 2 * in_dim)), nn.LeakyReLU(0.2))
        self.l1 = nn.Sequential(nn.Linear(in_dim, 2 * in_dim), nn.LeakyReLU(0.2))

        curr_dim = in_dim * 2

        self.l2 = nn.Sequential(nn.Linear(curr_dim, curr_dim), nn.LeakyReLU(0.2))

        # curr_dim = curr_dim

        self.l3 = nn.Sequential(nn.Linear(curr_dim, curr_dim // 2), nn.LeakyReLU(0.2))

        curr_dim = curr_dim // 2

        self.l4 = nn.Linear(curr_dim, 1)

    def forward(self, x):
        vari_emb = self.vari_emb(torch.arange(self.enc_in).to(self.device))
        vari_emb = vari_emb.unsqueeze(0).repeat(x.size(0), 1, 1)
        x = torch.cat((x, vari_emb), 2)

        if self.self_attn:
            x_out, _ = self.multihead_attn(x, x, x)
        h = self.l1(x_out)
        h = self.l2(h)
        h = self.l3(h)
        y = self.l4(h)

        return y


# new setting 2024/09/24
# class Discriminator(nn.Module):
#     def __init__(self, args, device):
#         super(Discriminator, self).__init__()
#         self.device = device
#         self.enc_in = args.enc_in
#         self.vari_emb_dim = 32
#         self.vari_emb = nn.Embedding(args.enc_in, self.vari_emb_dim)
#         # nn.init.normal_(self.vari_emb.weight, std=0.02)

#         in_dim = args.d_model + self.vari_emb_dim

#         self.self_attn = Self_Attn(in_dim, args.enc_in)
#         # self.l1 = nn.Sequential(nn .utils.spectral_norm(nn.Linear(in_dim, 2 * in_dim)), nn.LeakyReLU(0.2))
#         self.l1 = nn.Sequential(nn.Linear(in_dim, in_dim // 2), nn.LeakyReLU(0.2))

#         curr_dim = in_dim // 2

#         self.l2 = nn.Sequential(nn.Linear(curr_dim, curr_dim // 2), nn.LeakyReLU(0.2))

#         curr_dim = curr_dim // 2

#         self.l3 = nn.Linear(curr_dim, 1)

#     def forward(self, x):
#         vari_emb = self.vari_emb(torch.arange(self.enc_in).to(self.device))
#         vari_emb = vari_emb.unsqueeze(0).repeat(x.size(0), 1, 1)
#         x = torch.cat((x, vari_emb), 2)
#         if self.self_attn:
#             x = self.self_attn(x)
#         x = self.l1(x)
#         x = self.l2(x)
#         x = self.l3(x)
#         return x


### concat のタイミングを変えてみる 2024/09/24
# class Generator(nn.Module):
#     def __init__(self, args, device):
#         super(Generator, self).__init__()
#         self.enc_in = args.enc_in
#         self.device = device
#         self.vari_emb_dim = args.d_model // 4
#         self.vari_emb = nn.Embedding(args.enc_in, self.vari_emb_dim)

#         in_dim = args.d_model + self.vari_emb_dim

#         self.l1 = nn.Sequential(nn.Linear(in_dim, in_dim), nn.LeakyReLU(0.2))
#         # self.l1 = nn.Sequential(
#         #     nn.utils.spectral_norm(nn.Linear(in_dim, in_dim)),
#         #     nn.BatchNorm1d(args.enc_in),
#         #     nn.LeakyReLU(0.2),
#         # )
#         self.self_attn = Self_Attn(in_dim, args.enc_in)
#         self.l2 = nn.Sequential(nn.Linear(in_dim, args.d_model))  # , nn.LeakyReLU(0.2))

#         # curr_dim = 2 * in_dim
#         # added on 2024/09/23, v3
#         # self.l3 = nn.Sequential(nn.Linear(curr_dim, args.d_model))

#     def forward(self, x):
#         vari_emb = self.vari_emb(torch.arange(self.enc_in).to(self.device))
#         vari_emb = vari_emb.unsqueeze(0).repeat(x.size(0), 1, 1)
#         x = torch.cat((x, vari_emb), 2)

#         x = self.l1(x)
#         if self.self_attn:
#             x = self.self_attn(x)
#         x = self.l2(x)
#         return x


# class Discriminator(nn.Module):
#     def __init__(self, args, device):
#         super(Discriminator, self).__init__()
#         self.device = device
#         self.enc_in = args.enc_in
#         self.vari_emb_dim = args.d_model // 4
#         self.vari_emb = nn.Embedding(args.enc_in, self.vari_emb_dim)
#         # nn.init.normal_(self.vari_emb.weight, std=0.02)

#         in_dim = args.d_model + self.vari_emb_dim

#         self.self_attn = Self_Attn(in_dim, args.enc_in)
#         # self.l1 = nn.Sequential(nn .utils.spectral_norm(nn.Linear(in_dim, 2 * in_dim)), nn.LeakyReLU(0.2))
#         self.l1 = nn.Sequential(nn.Linear(in_dim, in_dim // 2), nn.LeakyReLU(0.2))

#         curr_dim = in_dim // 2

#         self.l2 = nn.Sequential(nn.Linear(curr_dim, curr_dim // 2), nn.LeakyReLU(0.2))

#         curr_dim = curr_dim // 2

#         self.l3 = nn.Linear(curr_dim, 1)

#     def forward(self, x):
#         vari_emb = self.vari_emb(torch.arange(self.enc_in).to(self.device))
#         vari_emb = vari_emb.unsqueeze(0).repeat(x.size(0), 1, 1)
#         x = torch.cat((x, vari_emb), 2)
#         if self.self_attn:
#             x = self.self_attn(x)
#         x = self.l1(x)
#         x = self.l2(x)
#         x = self.l3(x)
#         return x
