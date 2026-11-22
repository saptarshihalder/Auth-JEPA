import torch
import torch.nn as nn
import torch.nn.functional as F

K, DP, DC = 16, 74, 15  # 16 patches, 74 noise measurements, 15 inputs


class Predictor(nn.Module):  # predictor network
    def __init__(self, d=128, L=4, H=4, var_min=1e-3):  # transformer width 128, 4 blocks, 4 heads
        super().__init__()
        self.msk = nn.Linear(DC, d)  # linear transform: 15 -> 128
        self.kind = nn.Parameter(torch.zeros(d))  # learned offset added to every embedding
        layer = nn.TransformerEncoderLayer(d, H, 4 * d, dropout=0.0, batch_first=True, norm_first=True)  # transformer block
        self.enc = nn.TransformerEncoder(layer, L, enable_nested_tensor=False)  # 4 transformer blocks
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * DP))  # output layer: 128 -> 148
        self.var_min = var_min
        self.register_buffer("amask", ~torch.eye(K, dtype=torch.bool))  # each patch sees only itself

    def forward(self, C):  # C: (images, 16 patches, 15 inputs)
        e = self.msk(C) + self.kind
        mu, lv = self.head(self.enc(e, mask=self.amask)).chunk(2, -1)  # 74 means, 74 raw variances
        return mu, self.var_min + F.softplus(lv)  # variance always positive
