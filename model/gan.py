# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from torch.nn import functional as F
# from stock_energy.basic import PositionwiseFeedForward

class Generator(nn.Module):
    def __init__(self, input_dim):
        super(Generator, self).__init__()

        # v1 network
        # self.network = nn.Sequential(
        #     nn.Linear(input_dim, 2*input_dim),
        #     nn.ReLU(True),
        #     nn.Linear(2*input_dim, 4*input_dim),
        #     nn.ReLU(True),
        #     nn.Linear(4*input_dim, input_dim)
        # )

        # v2 network
        self.network = nn.Sequential(
            nn.Linear(input_dim, 2*input_dim),
            nn.ReLU(True),
            nn.Linear(2*input_dim, 4*input_dim),
            nn.ReLU(True),
            nn.Linear(4*input_dim, 2*input_dim),
            nn.ReLU(True),
            nn.Linear(2*input_dim, input_dim)
        )

    def forward(self, x):

        return self.network(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()        

        # v2 network
        # self.network = nn.Sequential(
        #     nn.Linear(input_dim, 2*input_dim),
        #     nn.LeakyReLU(0.2),
        #     nn.Linear(2*input_dim, (4*input_dim)//3),
        #     nn.LeakyReLU(0.2),
        #     nn.Linear((4*input_dim)//3, 1)
        # )

        # v3 network
        self.network = nn.Sequential(
            nn.Linear(input_dim, 2*input_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(2*input_dim, 4*input_dim),
            nn.LeakyReLU(0.2),
            nn.Linear((4*input_dim), 2*input_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(2*input_dim, 1)
        )

    def forward(self, x): # x.shape = (batch_size, input_dim)
        
        return self.network(x)



