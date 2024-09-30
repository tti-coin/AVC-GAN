import pdb

import torch
import torch.nn as nn
from torch.nn import MultiheadAttention, TransformerEncoder, TransformerEncoderLayer
from torch.nn import functional as F


class Generator(nn.Module):
    def __init__(self, args, device):
        super(Generator, self).__init__()
        # self.enc_in = args.enc_in
        # self.self_attn = args.self_attn
        # self.device = device
        # self.vari_emb_dim = args.d_model // 8
        # self.vari_emb = nn.Embedding(args.enc_in, self.vari_emb_dim)

        # in_dim = args.d_model #+ self.vari_emb_dim
        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model=args.d_model, nhead=1), num_layers=2
        )

    def forward(self, x):
        # vari_emb = self.vari_emb(torch.arange(self.enc_in).to(self.device))
        # vari_emb = vari_emb.unsqueeze(0).repeat(x.size(0), 1, 1)
        # x = torch.cat((x, vari_emb), 2)
        x = x.transpose(1, 0)
        y = self.transformer_encoder(x)
        return y.transpose(1, 0)


class Discriminator(nn.Module):
    def __init__(self, args, device):
        super(Discriminator, self).__init__()
        # self.device = device
        # self.enc_in = args.enc_in
        # self.self_attn = args.self_attn
        # self.vari_emb_dim = args.d_model // 8
        # self.vari_emb = nn.Embedding(args.enc_in, self.vari_emb_dim)
        # # nn.init.normal_(self.vari_emb.weight, std=0.02)

        # in_dim = args.d_model + self.vari_emb_dim

        # # self.mhattn = Self_Attn(in_dim, args.enc_in)
        # self.multihead_attn = MultiheadAttention(in_dim, args.n_heads, dropout=0.1)
        # # self.l1 = nn.Sequential(nn .utils.spectral_norm(nn.Linear(in_dim, 2 * in_dim)), nn.LeakyReLU(0.2))
        # self.l1 = nn.Sequential(nn.Linear(in_dim, 2 * in_dim), nn.LeakyReLU(0.2))

        # curr_dim = in_dim * 2

        # # self.l2 = nn.Sequential(
        # #     nn.Conv1d(args.enc_in, args.enc_in, kernel_size=1, stride=1),
        # #     nn.BatchNorm1d(args.enc_in),
        # #     nn.LeakyReLU(0.2),
        # # )

        # self.l2 = nn.Sequential(nn.Linear(curr_dim, curr_dim), nn.LeakyReLU(0.2))

        # # curr_dim = curr_dim

        # self.l3 = nn.Sequential(nn.Linear(curr_dim, curr_dim // 2), nn.LeakyReLU(0.2))

        # curr_dim = curr_dim // 2

        # self.l4 = nn.Linear(curr_dim, 1)

        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model=args.d_model, nhead=1), num_layers=2
        )

    def forward(self, x):
        # vari_emb = self.vari_emb(torch.arange(self.enc_in).to(self.device))
        # vari_emb = vari_emb.unsqueeze(0).repeat(x.size(0), 1, 1)
        # x = torch.cat((x, vari_emb), 2)
        # if self.self_attn:
        #     x_out, _ = self.multihead_attn(x, x, x)
        # h = self.l1(x_out)
        # h = self.l2(h)
        # h = self.l3(h)
        # y = self.l4(h)
        x = x.transpose(1, 0)
        y = self.transformer_encoder(x)

        return y.transpose(1, 0)
