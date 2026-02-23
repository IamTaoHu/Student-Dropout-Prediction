from typing import List

import torch
import torch.nn as nn


class FeatureTokenizer(nn.Module):
    """
    Minimal FT-Transformer-style tokenizer:
    - Categorical: per-feature embedding
    - Numerical: per-feature linear (x * w + b) into d_token
    Output tokens: [B, n_tokens, d_token]
    """
    def __init__(self, cat_cardinalities: List[int], n_num: int, d_token: int):
        super().__init__()
        self.n_cat = len(cat_cardinalities)
        self.n_num = int(n_num)
        self.d_token = int(d_token)

        if self.n_cat:
            self.cat_embeds = nn.ModuleList(
                [nn.Embedding(card, d_token) for card in cat_cardinalities]
            )
        else:
            self.cat_embeds = nn.ModuleList()

        if self.n_num:
            self.num_weight = nn.Parameter(torch.randn(self.n_num, d_token) * 0.01)
            self.num_bias = nn.Parameter(torch.zeros(self.n_num, d_token))
        else:
            self.register_parameter("num_weight", None)
            self.register_parameter("num_bias", None)

    def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        toks = []

        if self.n_cat:
            # x_cat: [B, n_cat]
            for j, emb in enumerate(self.cat_embeds):
                toks.append(emb(x_cat[:, j]))  # [B, d_token]

        if self.n_num:
            # x_num: [B, n_num] -> [B, n_num, d_token]
            num = x_num.unsqueeze(-1) * self.num_weight.unsqueeze(0) + self.num_bias.unsqueeze(0)
            # append each numeric feature token as [B, d_token]
            toks.extend([num[:, j, :] for j in range(self.n_num)])

        # [B, n_tokens, d_token]
        return torch.stack(toks, dim=1)


class FTTransformerLike(nn.Module):
    """
    Minimal Transformer encoder over feature tokens + CLS pooling.
    """
    def __init__(
        self,
        cat_cardinalities: List[int],
        n_num: int,
        d_token: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        n_classes: int = 3,
    ):
        super().__init__()
        self.tokenizer = FeatureTokenizer(cat_cardinalities, n_num, d_token)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_token))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, n_classes)

        nn.init.normal_(self.cls, std=0.02)

    def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        tok = self.tokenizer(x_cat, x_num)  # [B, n_tokens, d_token]
        B = tok.size(0)
        cls = self.cls.expand(B, -1, -1)    # [B, 1, d_token]
        x = torch.cat([cls, tok], dim=1)    # [B, 1+n_tokens, d_token]
        x = self.encoder(x)
        x = self.norm(x[:, 0, :])           # CLS token
        return self.head(x)
