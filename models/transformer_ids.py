"""
Transformer-based IDS model.
Uses positional encoding + multi-head self-attention.
State-of-the-art for tabular / sequential network traffic.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TokenEmbedding(nn.Module):
    def __init__(self, input_size: int, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class TransformerIDS(nn.Module):
    """
    Transformer encoder for IDS classification.
    Features:
    - Token + positional embeddings
    - Multi-head self-attention with causal masking option
    - Pre-LN transformer blocks (more stable training)
    - CLS token pooling
    - Multi-layer classifier head
    """

    def __init__(
        self,
        input_size: int = 122,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        num_classes: int = 5,
        max_seq_len: int = 64,
    ):
        super().__init__()
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"

        self.d_model = d_model
        self.num_classes = num_classes

        self.token_embedding = TokenEmbedding(input_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_seq_len + 1, dropout=dropout)

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, features) → add seq dim
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, features)

        batch_size = x.size(0)

        token_emb = self.token_embedding(x)  # (batch, seq, d_model)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, d_model)
        tokens = torch.cat([cls_tokens, token_emb], dim=1)      # (batch, 1+seq, d_model)

        tokens = self.pos_encoding(tokens)

        encoded = self.encoder(tokens)  # (batch, 1+seq, d_model)

        cls_output = encoded[:, 0, :]   # CLS token
        attn_weights = encoded[:, 1:, :].mean(dim=-1)  # proxy attention map

        logits = self.classifier(cls_output)
        return logits, attn_weights

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(x)
        return torch.argmax(logits, dim=-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(x)
        return F.softmax(logits, dim=-1)
