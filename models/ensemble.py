"""
Ensemble of LSTM + Transformer with attention-based fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
from .lstm_ids import BiLSTMIDS
from .transformer_ids import TransformerIDS


class AttentionFusion(nn.Module):
    """Learn per-model weights from their logits."""

    def __init__(self, num_models: int, num_classes: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(num_models * num_classes, num_models),
            nn.Softmax(dim=-1),
        )

    def forward(self, logits_list: List[torch.Tensor]) -> torch.Tensor:
        # logits_list: list of (batch, num_classes)
        concat = torch.cat(logits_list, dim=-1)  # (batch, num_models * num_classes)
        weights = self.gate(concat)              # (batch, num_models)
        stacked = torch.stack(logits_list, dim=1)  # (batch, num_models, num_classes)
        fused = (weights.unsqueeze(-1) * stacked).sum(dim=1)  # (batch, num_classes)
        return fused


class EnsembleIDS(nn.Module):
    """
    Ensemble IDS combining BiLSTM and Transformer.
    Fusion modes: weighted_avg | attention | voting
    """

    def __init__(
        self,
        input_size: int = 122,
        num_classes: int = 5,
        fusion: str = "attention",
        lstm_kwargs: dict = None,
        transformer_kwargs: dict = None,
    ):
        super().__init__()
        self.fusion = fusion
        self.num_classes = num_classes

        _lstm_kw = {"input_size": input_size, "num_classes": num_classes}
        if lstm_kwargs:
            _lstm_kw.update(lstm_kwargs)

        _trans_kw = {"input_size": input_size, "num_classes": num_classes}
        if transformer_kwargs:
            _trans_kw.update(transformer_kwargs)

        self.lstm_model = BiLSTMIDS(**_lstm_kw)
        self.transformer_model = TransformerIDS(**_trans_kw)

        if fusion == "attention":
            self.fusion_layer = AttentionFusion(2, num_classes)
        elif fusion == "weighted_avg":
            self.weights = nn.Parameter(torch.tensor([0.4, 0.6]))
        elif fusion == "learnable":
            self.fusion_linear = nn.Linear(num_classes * 2, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        lstm_logits, lstm_attn = self.lstm_model(x)
        trans_logits, trans_attn = self.transformer_model(x)

        if self.fusion == "attention":
            fused = self.fusion_layer([lstm_logits, trans_logits])
        elif self.fusion == "weighted_avg":
            w = F.softmax(self.weights, dim=0)
            fused = w[0] * lstm_logits + w[1] * trans_logits
        elif self.fusion == "voting":
            fused = (lstm_logits + trans_logits) / 2.0
        elif self.fusion == "learnable":
            fused = self.fusion_linear(torch.cat([lstm_logits, trans_logits], dim=-1))
        else:
            fused = (lstm_logits + trans_logits) / 2.0

        aux = {"lstm_logits": lstm_logits, "trans_logits": trans_logits,
               "lstm_attn": lstm_attn, "trans_attn": trans_attn}
        return fused, aux

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(x)
        return torch.argmax(logits, dim=-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(x)
        return F.softmax(logits, dim=-1)


def build_model(architecture: str, input_size: int, num_classes: int, **kwargs) -> nn.Module:
    if architecture == "lstm":
        return BiLSTMIDS(input_size=input_size, num_classes=num_classes, **kwargs)
    elif architecture == "transformer":
        return TransformerIDS(input_size=input_size, num_classes=num_classes, **kwargs)
    elif architecture == "ensemble":
        return EnsembleIDS(input_size=input_size, num_classes=num_classes, **kwargs)
    raise ValueError(f"Unknown architecture: {architecture}")
