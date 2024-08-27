# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from torch.nn import functional as F

# from stock_energy.basic import PositionwiseFeedForward


class Generator(nn.Module):
    def __init__(self, c_in, d_model, device):
        super(Generator, self).__init__()
        self.c_in = c_in
        self.device = device
        # self.vari_emb = nn.Embedding(c_in, d_model // 4)
        # nn.init.normal_(self.vari_emb.weight, std=0.02)
        # v2 network
        self.network = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.ReLU(True),
            nn.Linear(2 * d_model, 4 * d_model),
            nn.ReLU(True),
            nn.Linear(4 * d_model, 2 * d_model),
            nn.ReLU(True),
            nn.Linear(2 * d_model, d_model),
        )

    def forward(self, noise):
        # vari_emb = self.vari_emb(torch.arange(self.c_in).to(self.device))
        # vari_emb = vari_emb.unsqueeze(0).repeat(noise.size(0), 1, 1)
        # gen_input = torch.cat((vari_emb, noise), 2)  # FIXME
        rep = self.network(noise)
        return rep


class Discriminator(nn.Module):
    def __init__(self, c_in, d_model, device):
        super(Discriminator, self).__init__()
        self.c_in = c_in
        self.device = device
        # self.vari_emb = nn.Embedding(c_in, d_model // 4)
        # nn.init.normal_(
        #     self.vari_emb.weight, std=0.02
        # )  # FIXME: std=0.002 のほうが良いかも

        # v2 network
        # self.network = nn.Sequential(
        #     nn.Linear(d_model, 2*d_model),
        #     nn.LeakyReLU(0.2),
        #     nn.Linear(2*d_model, (4*d_model)//3),
        #     nn.LeakyReLU(0.2),
        #     nn.Linear((4*d_model)//3, 1)
        # )

        # v3 network
        self.network = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.LeakyReLU(0.2),
            nn.Linear(2 * d_model, 4 * d_model),
            nn.LeakyReLU(0.2),
            nn.Linear((4 * d_model), 2 * d_model),
            nn.LeakyReLU(0.2),
            nn.Linear(2 * d_model, 1),
        )

    def forward(self, rep):  # x.shape = (batch_size, d_model)
        # vari_emb = self.vari_emb(torch.arange(self.c_in).to(self.device))
        # vari_emb = vari_emb.unsqueeze(0).repeat(rep.size(0), 1, 1)
        # disc_in = torch.cat((rep, vari_emb), 2)
        validity = self.network(rep)
        return validity


# class Generator(nn.Module):
#     def __init__(self, c_in, d_model, device):
#         super(Generator, self).__init__()
#         self.c_in = c_in
#         self.device = device
#         self.vari_emb = nn.Embedding(c_in, d_model//4)
#         nn.init.normal_(self.vari_emb.weight, std=0.02)

#         def block(in_feat, out_feat, normalize=False):
#             layers = [nn.Linear(in_feat, out_feat)]
#             # if normalize:
#             #     layers.append(nn.BatchNorm1d(out_feat, 0.8))
#             layers.append(nn.LeakyReLU(0.2, inplace=True))
#             return layers

#         self.model = nn.Sequential(
#             *block(d_model + d_model//4, 256, normalize=False),
#             *block(256, 512),
#             *block(512, 1024),
#             nn.Linear(1024, d_model),
#             nn.Tanh()
#         )

#     def forward(self, noise):
#         # Concatenate label embedding and image to produce input
#         vari_emb = self.vari_emb(torch.arange(self.c_in).to(self.device))
#         vari_emb = vari_emb.unsqueeze(0).repeat(noise.size(0), 1, 1)
#         gen_input = torch.cat((vari_emb, noise), 2) # FIXME
#         rep = self.model(gen_input)
#         return rep


# class Discriminator(nn.Module):
#     def __init__(self, c_in, d_model, device):
#         super(Discriminator, self).__init__()
#         self.c_in = c_in
#         self.device = device
#         self.vari_emb = nn.Embedding(c_in, d_model//4)
#         nn.init.normal_(self.vari_emb.weight, std=0.02) # FIXME: std=0.002 のほうが良いかも

#         self.model = nn.Sequential(
#             nn.Linear(d_model + d_model//4, 256,),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(256, 512),
#             nn.Dropout(0.4),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(512, 512),
#             nn.Dropout(0.4),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(512, 1),
#         )

#     def forward(self, rep):
#         # Concatenate label embedding and image to produce input
#         vari_emb = self.vari_emb(torch.arange(self.c_in).to(self.device))
#         vari_emb = vari_emb.unsqueeze(0).repeat(rep.size(0), 1, 1)
#         disc_in = torch.cat((rep, vari_emb), 2)
#         validity = self.model(disc_in)
#         return validity
