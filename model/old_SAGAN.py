import torch
import torch.nn as nn
from torch.nn import functional as F

# from stock_energy.basic import PositionwiseFeedForward

# class Generator(nn.Module):
#     def __init__(self, c_in, d_model, device):
#         super(Generator, self).__init__()
#         self.c_in = c_in
#         self.device = device
#         # self.network = nn.Sequential(
#         #     nn.Linear(d_model, 2 * d_model),
#         #     nn.ReLU(True),
#         #     nn.Linear(2 * d_model, 4 * d_model),
#         #     nn.ReLU(True),
#         #     nn.Linear(4 * d_model, 2 * d_model),
#         #     nn.ReLU(True),
#         #     nn.Linear(2 * d_model, d_model),
#         # )


#     def forward(self, noise):
#         rep = self.network(noise)
#         return rep


# class Discriminator(nn.Module):
#     def __init__(self, c_in, d_model, device):
#         super(Discriminator, self).__init__()
#         self.c_in = c_in
#         self.device = device

#         # v3 network
#         self.network = nn.Sequential(
#             nn.Linear(d_model, 2 * d_model),
#             nn.LeakyReLU(0.2),
#             nn.Linear(2 * d_model, 4 * d_model),
#             nn.LeakyReLU(0.2),
#             nn.Linear((4 * d_model), 2 * d_model),
#             nn.LeakyReLU(0.2),
#             nn.Linear(2 * d_model, 1),
#         )

#     def forward(self, rep):  # x.shape = (batch_size, d_model)
#         validity = self.network(rep)
#         return validity


############################################################################################################
class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.l1 = nn.Sequential(
            nn.ConvTranspose1d(
                args.noise_dim, args.noise_dim, kernel_size=args.enc_in, stride=1
            ),
            nn.BatchNorm1d(args.noise_dim),
            nn.ReLU(),
        )
        self.self_attn = Self_Attn(args.noise_dim, args.enc_in)
        self.l2 = nn.Sequential(
            nn.Conv1d(args.noise_dim, args.pred_len, kernel_size=1, stride=1)
            # , nn.Tanh()
        )

    def forward(self, x):
        x = self.l1(x)
        x = self.self_attn(x)
        x = self.l2(x)
        return x


# class Self_Attn(nn.Module):
#     def __init__(self, in_dim, seq_len):
#         super(Self_Attn, self).__init__()
#         self.query_conv = nn.Conv1d(in_dim, in_dim // 8, kernel_size=1)
#         self.key_conv = nn.Conv1d(in_dim, in_dim // 8, kernel_size=1)
#         self.value_conv = nn.Conv1d(in_dim, in_dim, kernel_size=1)
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x):
#         batch_size, channels, seq_len = x.size()
#         proj_query = self.query_conv(x).view(batch_size, -1, seq_len)
#         proj_key = self.key_conv(x).view(batch_size, -1, seq_len)
#         energy = torch.bmm(proj_query.permute(0, 2, 1), proj_key)
#         attention = self.softmax(energy)
#         proj_value = self.value_conv(x).view(batch_size, -1, seq_len)
#         out = torch.bmm(proj_value, attention.permute(0, 2, 1))
#         out = out.view(batch_size, channels, seq_len)
#         return out


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.l1 = nn.Sequential(
            nn.Conv1d(args.pred_len, args.noise_dim, kernel_size=1, stride=1),
            nn.BatchNorm1d(args.noise_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.self_attn = Self_Attn(args.noise_dim, args.enc_in)
        self.l2 = nn.Sequential(
            nn.Conv1d(
                args.noise_dim, args.noise_dim // 2, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm1d(args.noise_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.l3 = nn.Sequential(
            nn.Conv1d(
                args.noise_dim // 2,
                args.noise_dim // 4,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm1d(args.noise_dim // 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Sequential(nn.Linear(args.noise_dim, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.l1(x)  # [32, 192, 7] -> [32, 128, 7]
        x = self.self_attn(x)  # [32, 128, 7] -> [32, 128, 7]
        x = self.l2(x)  # [32, 128, 7] -> [32, args.noise_dim / 2, 7]
        x = self.l3(x)  # [32, args.noise_dim / 2, 7] -> [32, 32, 3]
        x = x.view(x.size(0), -1)  # Flatten -> [32, 32*3]
        x = self.fc(x)  # [32, 32*3] -> [32, 1]
        return x


class Self_Attn(nn.Module):
    def __init__(self, in_dim, seq_len):
        super(Self_Attn, self).__init__()
        self.query_conv = nn.Conv1d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv1d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv1d(in_dim, in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channels, seq_len = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, seq_len)
        proj_key = self.key_conv(x).view(batch_size, -1, seq_len)
        energy = torch.bmm(proj_query.permute(0, 2, 1), proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(batch_size, -1, seq_len)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, seq_len)
        return out
