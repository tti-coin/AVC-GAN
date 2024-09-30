# from layers.Masked import Masking
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import DataEmbedding_inverted
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Transformer_EncDec import Encoder, EncoderLayer


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs, device):
        super(Model, self).__init__()
        self.device = device
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.enc_in = configs.enc_in  # Added
        self.mask_ratio = configs.mask_ratio
        self.vari_masked_ratio = configs.vari_masked_ratio

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len,
            configs.enc_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )

        self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        self.fc_mean = nn.Linear(configs.d_model, configs.d_model, bias=True)
        self.fc_log_var = nn.Linear(configs.d_model, configs.d_model, bias=True)
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_enc /= stdev
        # NOTE: means.shape -> 32, 1, 7
        _, _, N = x_enc.shape
        # B: batch_size;    E: d_model;
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Masking
        num_vari_masked = int(self.enc_in * self.vari_masked_ratio)
        vari_masked = np.random.choice(self.enc_in, num_vari_masked, replace=False)
        masked_x_enc = x_enc.clone()
        total_mask = torch.ones_like(x_enc)
        for idx in vari_masked:
            masking = torch.bernoulli(
                torch.full((1, x_enc.size(1)), 1 - self.mask_ratio)
            ).to(self.device)
            masked_x_enc[:, :, idx] = x_enc[:, :, idx] * masking
            masked_x_enc[:, :, idx] = masked_x_enc[:, :, idx].masked_fill(
                masking == 0, -5
            )  # TODO
            total_mask[:, :, idx] = masking

        # TODO: mask info の追加
        x_mask = 1 - total_mask  # mask == 1

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        emb_out = self.enc_embedding(masked_x_enc, None)
        # emb_out = self.enc_embedding(masked_x_enc, None)
        # emb_out = self.enc_embedding(x_enc, None)

        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, _ = self.encoder(emb_out, attn_mask=None)

        # variational autoencoder
        mean = self.fc_mean(enc_out)
        log_var = self.fc_log_var(enc_out)

        # reparameterization trick
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mean + eps * std

        ### Decoder
        # B N E -> B N S -> B S N
        dec_out = self.projector(z).permute(0, 2, 1)[:, :, :N]
        # dec_out = self.projector(enc_out).permute(0, 2, 1)[
        #     :, :, :N
        # ]  # filter the covariates

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (
                stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            )
            dec_out = dec_out + (
                means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            )

        return dec_out, mean, log_var

    def encode(self, x_enc):
        _, _, N = x_enc.shape

        # Embedding
        emb_out = self.enc_embedding(x_enc, None)

        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, _ = self.encoder(emb_out, attn_mask=None)
        mean = self.fc_mean(enc_out)
        log_var = self.fc_log_var(enc_out)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mean + eps * std

        return z, mean, log_var

    def decode(self, z):
        _, N, _ = z.shape

        dec_out = self.projector(z).permute(0, 2, 1)[:, :, :N]  # filter the covariates

        return dec_out[:, -self.pred_len :, :]

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, mean, log_var = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len :, :], mean, log_var
