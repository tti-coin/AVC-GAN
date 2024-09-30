import pdb

import torch
import torch.nn as nn
from torch.nn import functional as F


class Generator(nn.Module):
    def __init__(self, args, device):
        super(Generator, self).__init__()
        self.self_attn = args.self_attn

        self.l1 = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(args.d_model, args.d_model * 2)),
            nn.BatchNorm1d(args.enc_in),
            nn.LeakyReLU(0.2),
        )

        curr_dim = args.d_model * 2

        self.l2 = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(curr_dim, curr_dim)),
            nn.BatchNorm1d(args.enc_in),
            nn.LeakyReLU(0.2),
        )
        self.l3 = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(curr_dim, args.d_model))
        )

        self.self_attn1 = Self_Attn(args.d_model, args.enc_in)
        # self.self_attn2 = Self_Attn(args.d_model, args.enc_in)

    def forward(self, x):  # NOTE: ConditionalSAGAN.py とは若干異なる
        if self.self_attn:
            x = self.self_attn1(x)
            x = self.l1(x)
            x = self.l2(x)
            x = self.l3(x)
        else:
            x = self.l1(x)
            x = self.l2(x)
            x = self.l3(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, args, device):
        super(Discriminator, self).__init__()
        self.self_attn = args.self_attn
        # self.l1 = nn.Sequential(nn.utils.spectral_norm(nn.Linear(args.d_model, args.d_model * 2)),
        #                         nn.BatchNorm1d(args.enc_in),
        #                         nn.LeakyReLU(0.2))

        self.l1 = nn.Sequential(
            nn.Conv1d(args.enc_in, args.enc_in, kernel_size=1, stride=1),
            nn.BatchNorm1d(args.enc_in),
            nn.LeakyReLU(0.2),
        )

        curr_dim = args.d_model

        self.l2 = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(curr_dim, curr_dim)),
            nn.BatchNorm1d(args.enc_in),
            nn.LeakyReLU(0.2),
        )
        self.l3 = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(curr_dim, args.enc_in))
        )  # TODO

        self.self_attn1 = Self_Attn(args.d_model, args.enc_in)
        # self.self_attn2 = Self_Attn(args.d_model * 2, args.enc_in)

    def forward(self, x):
        if self.self_attn:
            x = self.self_attn1(x)
            x = self.l1(x)
            x = self.l2(x)
            x = self.l3(x)
        else:
            x = self.l1(x)
            x = self.l2(x)
            x = self.l3(x)
        return x


class Self_Attn(nn.Module):
    def __init__(self, in_dim, seq_len):
        super(Self_Attn, self).__init__()
        self.in_dim = in_dim
        self.seq_len = seq_len

        # Define linear layers for query, key, and value projections
        self.query_linear = nn.Linear(in_dim, in_dim)
        self.key_linear = nn.Linear(in_dim, in_dim)
        self.value_linear = nn.Linear(in_dim, in_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, seq_len, channels = x.size()

        # Project inputs to queries, keys, and values
        proj_query = self.query_linear(x)
        proj_key = self.key_linear(x)
        proj_value = self.value_linear(x)

        # Compute attention scores
        energy = torch.bmm(proj_query, proj_key.transpose(1, 2))
        attention = self.softmax(energy)

        # Compute attention output
        out = torch.bmm(attention, proj_value)

        return out


### iTransformer に寄せた SAGAN
# class Generator(nn.Module):
#     def __init__(self, args, device):
#         super(Generator, self).__init__()
#         self.self_attn = args.self_attn
#         self.conv1 = nn.Conv1d(
#             in_channels=args.d_model, out_channels=args.d_ff, kernel_size=1
#         )
#         self.conv2 = nn.Conv1d(
#             in_channels=args.d_ff, out_channels=args.d_model, kernel_size=1
#         )
#         self.norm1 = nn.LayerNorm(args.d_model)
#         self.norm2 = nn.LayerNorm(args.d_model)
#         self.dropout = nn.Dropout(0.1)
#         self.activation = F.relu

#         self.self_attn1 = Self_Attn(args.d_model, args.enc_in)

#     def forward(self, x):
#         if self.self_attn:
#             new_x = self.self_attn1(x)
#             x = x + self.dropout(new_x)
#             y = x = self.norm1(x)
#             y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
#             y = self.dropout(self.conv2(y).transpose(-1, 1))
#             y = self.norm2(x + y)
#         else:
#             NotImplementedError
#         return y


# class Discriminator(nn.Module):
#     def __init__(self, args, device):
#         super(Discriminator, self).__init__()
#         self.self_attn = args.self_attn
#         self.self_attn = args.self_attn
#         self.conv1 = nn.Conv1d(
#             in_channels=args.d_model, out_channels=args.d_model, kernel_size=1
#         )
#         self.conv2 = nn.Conv1d(in_channels=args.d_model, out_channels=1, kernel_size=1)
#         self.norm1 = nn.LayerNorm(args.d_model)
#         self.dropout = nn.Dropout(0.1)
#         self.activation = F.relu
#         self.self_attn1 = Self_Attn(args.d_model, args.enc_in)

#     def forward(self, x):
#         if self.self_attn:
#             new_x = self.self_attn1(x)
#             x = x + self.dropout(new_x)
#             y = x = self.norm1(x)
#             y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
#             y = self.dropout(self.conv2(y).transpose(-1, 1))
#         else:
#             NotImplementedError
#         return y
