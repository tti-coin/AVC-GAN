import torch
import torch.nn as nn
from torch.nn import functional as F


class Generator(nn.Module):
    def __init__(self, c_in, d_model, device):
        super(Generator, self).__init__()
        self.c_in = c_in
        self.device = device
        self.vari_emb_dim = 32 # TODO
        self.vari_emb = nn.Embedding(c_in, self.vari_emb_dim)
        nn.init.normal_(self.vari_emb.weight, std=0.02) # TODO
        # TODO: decide which network to use
        self.network = nn.Sequential(
            nn.Linear(d_model + self.vari_emb_dim, 2 * d_model),
            nn.ReLU(True),
            nn.Linear(2 * d_model, 4 * d_model),
            nn.ReLU(True),
            nn.Linear(4 * d_model, 2 * d_model),
            nn.ReLU(True),
            nn.Linear(2 * d_model, d_model),
        )

    def forward(self, noise):
        vari_emb = self.vari_emb(torch.arange(self.c_in).to(self.device))
        vari_emb = vari_emb.unsqueeze(0).repeat(noise.size(0), 1, 1)
        gen_input = torch.cat((vari_emb, noise), 2)
        rep = self.network(gen_input) 
        return rep


class Discriminator(nn.Module):
    def __init__(self, c_in, d_model, device):
        super(Discriminator, self).__init__()
        self.c_in = c_in
        self.device = device
        self.vari_emb_dim = 32
        self.vari_emb = nn.Embedding(c_in, self.vari_emb_dim)
        nn.init.normal_(
            self.vari_emb.weight, std=0.02
        )

        # TODO: decide which network to use
        self.network = nn.Sequential(
            nn.Linear(d_model + self.vari_emb_dim, 2 * d_model),
            nn.LeakyReLU(0.2),
            nn.Linear(2 * d_model, 4 * d_model),
            nn.LeakyReLU(0.2),
            nn.Linear((4 * d_model), 2 * d_model),
            nn.LeakyReLU(0.2),
            nn.Linear(2 * d_model, 1),
        )

    def forward(self, rep):  # x.shape = (batch_size, d_model)
        vari_emb = self.vari_emb(torch.arange(self.c_in).to(self.device))
        vari_emb = vari_emb.unsqueeze(0).repeat(rep.size(0), 1, 1)
        disc_in = torch.cat((rep, vari_emb), 2)
        validity = self.network(disc_in)
        return validity


# class SetDiscriminator(nn.Module):
#     def __init__(self, c_in, d_model, device):
#         super(SetDiscriminator, self).__init__()
#         self.c_in = c_in
#         self.device = device
#         self.set_dim = c_in * d_model
        
#         self.network = nn.Sequential(
#             nn.Linear(self.set_dim, 2 * self.set_dim),
#             nn.LeakyReLU(0.2),
#             nn.Linear(2 * self.set_dim, 4 * self.set_dim),
#             nn.LeakyReLU(0.2),
#             nn.Linear(4 * self.set_dim, 2 * self.set_dim),
#             nn.LeakyReLU(0.2),
#             nn.Linear(2 * self.set_dim, 1),
#         )

#     def forward(self, set_rep):
#         # set_rep = torch.permute(set_rep, (0, 2, 1)) # TODO
#         set_rep = set_rep.reshape(-1, self.set_dim)
#         validity = self.network(set_rep)
#         return validity
