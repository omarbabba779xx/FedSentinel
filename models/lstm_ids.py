"""
Bidirectional LSTM model for intrusion detection.
Treats each sample as a single time-step sequence; supports multi-step sequences
when used with sliding-window traffic aggregation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class LSTMAttention(nn.Module):
    """Additive (Bahdanau) attention over LSTM hidden states."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, encoder_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # encoder_outputs: (batch, seq_len, hidden*2)
        energy = torch.tanh(self.attn(encoder_outputs))           # (batch, seq, hidden)
        scores = self.v(energy).squeeze(-1)                        # (batch, seq)
        weights = F.softmax(scores, dim=-1)                        # (batch, seq)
        context = (weights.unsqueeze(-1) * encoder_outputs).sum(1) # (batch, hidden*2)
        return context, weights


class BiLSTMIDS(nn.Module):
    """
    Bidirectional LSTM IDS model with:
    - Multi-layer BiLSTM
    - Attention mechanism
    - Residual connections
    - Batch normalization
    - Dropout regularisation
    """

    def __init__(
        self,
        input_size: int = 122,
        hidden_size: int = 256,
        num_layers: int = 3,
        num_classes: int = 5,
        dropout: float = 0.3,
        use_attention: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.use_attention = use_attention

        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
        )

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.attention = LSTMAttention(hidden_size) if use_attention else None

        lstm_out_dim = hidden_size * 2  # bidirectional

        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # x: (batch, features) OR (batch, seq_len, features)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, features)

        batch_size, seq_len, _ = x.shape
        x_proj = self.input_proj(x.view(batch_size * seq_len, -1))
        x_proj = x_proj.view(batch_size, seq_len, -1)

        lstm_out, _ = self.lstm(x_proj)  # (batch, seq, hidden*2)

        attn_weights = None
        if self.use_attention and self.attention is not None:
            context, attn_weights = self.attention(lstm_out)
        else:
            context = lstm_out[:, -1, :]  # last step

        logits = self.classifier(context)
        return logits, attn_weights

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(x)
        return torch.argmax(logits, dim=-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(x)
        return F.softmax(logits, dim=-1)
