import torch
import torch.nn as nn
import torch.nn.functional as F

class CompressionModel(nn.Module):
    def __init__(self, d_model, n_heads):
        super(CompressionModel, self).__init__()
        # self.multihead_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        # self.dropout = nn.Dropout(0.5)
        # # self.layer_norm = nn.LayerNorm(d_model)
        self.pool = nn.AdaptiveAvgPool1d(1)

    # def forward(self, x_enc):
    #     attn_output, _ = self.multihead_attn(x_enc, x_enc, x_enc)  # (batch, num_vari, emb_dim)
    #     attn_output = F.relu(attn_output)  # 活性化関数の適用
    #     attn_output = self.dropout(attn_output)  # ドロップアウトの適用
    #     attn_output = self.layer_norm(attn_output + x_enc)
    #     # Global Average Pooling (または他の方法で num_vari 次元を圧縮)
    #     x_compressed = attn_output.mean(dim=1)  # (batch, emb_dim)
    #     return x_compressed

    def forward(self, x_enc):
        # プーリング層で num_vari 次元を圧縮
        # (batch, emb_dim, num_vari)
        compressed_x = self.pool(x_enc.permute(0, 2, 1))  # (batch, emb_dim, 1)
        self.dropout = nn.Dropout(p=0.5)
        compressed_x = compressed_x.squeeze(-1)  # (batch, emb_dim)
        return compressed_x