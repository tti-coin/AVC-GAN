import pdb

import torch
import torch.nn as nn
from torch.nn import functional as F


class Generator(nn.Module):
    def __init__(self, args, device):
        super(Generator, self).__init__()
        self.enc_in = args.enc_in
        self.device = device
        self.vari_emb_dim = 32
        self.vari_emb = nn.Embedding(args.enc_in, self.vari_emb_dim)

        in_dim = args.d_model + self.vari_emb_dim

        self.l1 = nn.Sequential(nn.Linear(in_dim, in_dim), nn.LeakyReLU(0.2))
        # self.l1 = nn.Sequential(
        #     nn.utils.spectral_norm(nn.Linear(in_dim, in_dim)),
        #     nn.BatchNorm1d(args.enc_in),
        #     nn.LeakyReLU(0.2),
        # )
        self.self_attn = Self_Attn(in_dim, args.enc_in)
        self.l2 = nn.Sequential(nn.Linear(in_dim, args.d_model))
        # self.l2 = nn.Sequential(nn.utils.spectral_norm(nn.Linear(in_dim, args.d_model)))

    def forward(self, x):
        vari_emb = self.vari_emb(torch.arange(self.enc_in).to(self.device))
        vari_emb = vari_emb.unsqueeze(0).repeat(x.size(0), 1, 1)
        x = torch.cat((x, vari_emb), 2)
        x = self.l1(x)
        if self.self_attn:
            x = self.self_attn(x)
            # x_attn = self.self_attn(x)
            # add residual connection
            # y = x + x_attn
        x = self.l2(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, args, device):
        super(Discriminator, self).__init__()
        self.device = device
        self.enc_in = args.enc_in
        self.vari_emb_dim = 32
        self.vari_emb = nn.Embedding(args.enc_in, self.vari_emb_dim)
        # nn.init.normal_(self.vari_emb.weight, std=0.02)

        in_dim = args.d_model + self.vari_emb_dim

        self.self_attn = Self_Attn(in_dim, args.enc_in)
        # self.l1 = nn.Sequential(nn.utils.spectral_norm(nn.Linear(in_dim, 2 * in_dim)), nn.LeakyReLU(0.2))
        self.l1 = nn.Sequential(nn.Linear(in_dim, 2 * in_dim), nn.LeakyReLU(0.2))

        curr_dim = in_dim * 2

        # self.l2 = nn.Sequential(
        #     nn.Conv1d(args.enc_in, args.enc_in, kernel_size=1, stride=1),
        #     nn.BatchNorm1d(args.enc_in),
        #     nn.LeakyReLU(0.2),
        # )

        self.l2 = nn.Sequential(nn.Linear(curr_dim, curr_dim), nn.LeakyReLU(0.2))

        # curr_dim = curr_dim

        self.l3 = nn.Sequential(nn.Linear(curr_dim, curr_dim // 2), nn.LeakyReLU(0.2))

        curr_dim = curr_dim // 2

        self.l4 = nn.Linear(curr_dim, 1)

    def forward(self, x):  # x.shape = (batch_size, args.noise_dim)
        vari_emb = self.vari_emb(torch.arange(self.enc_in).to(self.device))
        vari_emb = vari_emb.unsqueeze(0).repeat(x.size(0), 1, 1)
        x = torch.cat((x, vari_emb), 2)
        if self.self_attn:
            x = self.self_attn(x)
            # x_attn = self.self_attn(x)
            # add residual connection
            # pdb.set_trace()
            # y = x + x_attn
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        return x
    
class Discriminator_Dec(nn.Module):
    def __init__(self, args, model, device):
        super(Discriminator_Dec, self).__init__()
        self.model = model
        self.device = device
        self.enc_in = args.enc_in
        self.vari_emb_dim = 32
        self.vari_emb = nn.Embedding(args.enc_in, self.vari_emb_dim)
        # nn.init.normal_(self.vari_emb.weight, std=0.02)

        in_dim = args.pred_len + self.vari_emb_dim # decoded sequence

        self.self_attn = Self_Attn(in_dim, args.enc_in)
        # self.l1 = nn.Sequential(nn.utils.spectral_norm(nn.Linear(in_dim, 2 * in_dim)), nn.LeakyReLU(0.2))
        self.l1 = nn.Sequential(nn.Linear(in_dim, 2 * in_dim), nn.LeakyReLU(0.2))

        curr_dim = in_dim * 2

        # self.l2 = nn.Sequential(
        #     nn.Conv1d(args.enc_in, args.enc_in, kernel_size=1, stride=1),
        #     nn.BatchNorm1d(args.enc_in),
        #     nn.LeakyReLU(0.2),
        # )

        self.l2 = nn.Sequential(nn.Linear(curr_dim, curr_dim), nn.LeakyReLU(0.2))

        # curr_dim = curr_dim

        self.l3 = nn.Sequential(nn.Linear(curr_dim, curr_dim // 2), nn.LeakyReLU(0.2))

        curr_dim = curr_dim // 2

        self.l4 = nn.Linear(curr_dim, 1)

    def forward(self, x):  # x.shape = (batch_size, args.noise_dim)
        vari_emb = self.vari_emb(torch.arange(self.enc_in).to(self.device))
        vari_emb = vari_emb.unsqueeze(0).repeat(x.size(0), 1, 1)
        x = torch.cat((x, vari_emb), 2)
        if self.self_attn:
            x = self.self_attn(x)
            # x_attn = self.self_attn(x)
            # add residual connection
            # pdb.set_trace()
            # y = x + x_attn
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        return x

class InceptionDiscriminator(nn.Module):
    def __init__(self, args, model, device):
        super(InceptionDiscriminator, self).__init__()
        self.model = model
        self.device = device
        self.args = args
        self.enc_in = args.enc_in
        self.vari_emb_dim = 32
        self.vari_emb = nn.Embedding(args.enc_in, self.vari_emb_dim)
        self.discriminator = Discriminator(args, self.device)
        self.discriminator_dec = Discriminator_Dec(args, self.model, self.device)
        self.final = nn.Linear(args.enc_in * 2, args.enc_in)

    def forward(self, x):
        # inception model
        x_hidden = self.discriminator(x)
        dec_out = self.model.decode(x)
        x_dec = self.discriminator_dec(dec_out.permute(0, 2, 1))
        x_cat = torch.cat((x_hidden, x_dec), 1)
        x_final = self.final(x_cat.squeeze())
        return x_final


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