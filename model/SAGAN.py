import torch
import torch.nn as nn
from torch.nn import functional as F


class Generator(nn.Module):
    def __init__(self, args, device):
        super(Generator, self).__init__()
        self.self_attn = args.self_attn
        self.l1 = nn.Sequential(nn.utils.spectral_norm(nn.Linear(args.d_model, args.d_model * 2)),
                                nn.BatchNorm1d(args.enc_in),
                                nn.LeakyReLU(0.2))
        
        curr_dim = args.d_model * 2

        self.l2 = nn.Sequential(nn.utils.spectral_norm(nn.Linear(curr_dim, curr_dim)),
                                nn.BatchNorm1d(args.enc_in),
                                nn.LeakyReLU(0.2))
        self.l3 = nn.Sequential(nn.utils.spectral_norm(nn.Linear(curr_dim, args.d_model)))


        self.self_attn1 = Self_Attn(args.d_model * 2, args.enc_in)
        self.self_attn2 = Self_Attn(args.d_model * 2, args.enc_in)


    def forward(self, x):
        if self.self_attn:
            x = self.l1(x)
            x = self.self_attn1(x)
            x = self.l2(x)
            x = self.self_attn2(x)
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
        self.l1 = nn.Sequential(nn.utils.spectral_norm(nn.Linear(args.d_model, args.d_model * 2)),
                                nn.BatchNorm1d(args.enc_in),
                                nn.LeakyReLU(0.2))
        
        curr_dim = args.d_model * 2

        self.l2 = nn.Sequential(nn.utils.spectral_norm(nn.Linear(curr_dim, curr_dim)),
                                nn.BatchNorm1d(args.enc_in),
                                nn.LeakyReLU(0.2))
        self.l3 = nn.Sequential(nn.utils.spectral_norm(nn.Linear(curr_dim, args.enc_in))) # TODO

        self.self_attn1 = Self_Attn(args.d_model * 2, args.enc_in)
        self.self_attn2 = Self_Attn(args.d_model * 2, args.enc_in)


    def forward(self, x):
        if self.self_attn:
            x = self.l1(x)
            x = self.self_attn1(x)
            x = self.l2(x)
            x = self.self_attn2(x)
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

