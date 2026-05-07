"""Tests for ML model architectures."""

import torch
import pytest
from models import BiLSTMIDS, TransformerIDS, EnsembleIDS, build_model


BATCH = 16
FEATURES = 122
CLASSES = 5


class TestBiLSTMIDS:
    def test_output_shape(self):
        model = BiLSTMIDS(input_size=FEATURES, num_classes=CLASSES)
        x = torch.randn(BATCH, FEATURES)
        logits, attn = model(x)
        assert logits.shape == (BATCH, CLASSES)

    def test_attention_weights(self):
        model = BiLSTMIDS(input_size=FEATURES, use_attention=True)
        x = torch.randn(BATCH, FEATURES)
        _, attn = model(x)
        assert attn is not None

    def test_predict(self):
        model = BiLSTMIDS(input_size=FEATURES, num_classes=CLASSES)
        x = torch.randn(BATCH, FEATURES)
        preds = model.predict(x)
        assert preds.shape == (BATCH,)
        assert preds.max() < CLASSES

    def test_predict_proba_sums_to_one(self):
        model = BiLSTMIDS(input_size=FEATURES, num_classes=CLASSES)
        x = torch.randn(BATCH, FEATURES)
        proba = model.predict_proba(x)
        sums = proba.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(BATCH), atol=1e-5)


class TestTransformerIDS:
    def test_output_shape(self):
        model = TransformerIDS(input_size=FEATURES, num_classes=CLASSES)
        x = torch.randn(BATCH, FEATURES)
        logits, _ = model(x)
        assert logits.shape == (BATCH, CLASSES)

    def test_gradient_flow(self):
        model = TransformerIDS(input_size=FEATURES, num_classes=CLASSES)
        x = torch.randn(BATCH, FEATURES)
        logits, _ = model(x)
        loss = logits.sum()
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"


class TestEnsembleIDS:
    def test_output_shape(self):
        model = EnsembleIDS(input_size=FEATURES, num_classes=CLASSES)
        x = torch.randn(BATCH, FEATURES)
        logits, aux = model(x)
        assert logits.shape == (BATCH, CLASSES)

    def test_fusion_modes(self):
        for fusion in ["attention", "weighted_avg", "voting"]:
            model = EnsembleIDS(input_size=FEATURES, num_classes=CLASSES, fusion=fusion)
            x = torch.randn(BATCH, FEATURES)
            logits, _ = model(x)
            assert logits.shape == (BATCH, CLASSES), f"Failed for fusion={fusion}"


class TestBuildModel:
    def test_builds_all_architectures(self):
        for arch in ["lstm", "transformer", "ensemble"]:
            model = build_model(arch, input_size=FEATURES, num_classes=CLASSES)
            assert model is not None

    def test_invalid_architecture(self):
        with pytest.raises(ValueError):
            build_model("invalid_arch", input_size=FEATURES, num_classes=CLASSES)
