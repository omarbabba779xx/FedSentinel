"""Tests for model watermarking module."""
import numpy as np
import pytest
import torch
import torch.nn as nn
import tempfile
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from watermarking.model_watermark import (
    WatermarkKey, WatermarkGenerator, WatermarkEmbedder, OwnershipVerifier
)


INPUT_DIM = 40
NUM_CLASSES = 5


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 64), nn.ReLU(), nn.Linear(64, NUM_CLASSES)
        )

    def forward(self, x):
        return self.net(x)


@pytest.fixture
def model():
    return ToyModel()


@pytest.fixture
def wm_key():
    return WatermarkGenerator.random_feature_pattern(
        num_triggers=30, input_dim=INPUT_DIM, target_label=0,
        owner_id="TestOrg", session_id="test_round",
    )


@pytest.fixture
def embedder(wm_key):
    return WatermarkEmbedder(wm_key, embed_frequency=1, embed_epochs=5,
                              lr=1e-3, device=torch.device("cpu"))


class TestWatermarkKey:
    def test_hash_is_sha256(self, wm_key):
        assert len(wm_key.key_hash) == 64  # SHA-256 = 64 hex chars

    def test_different_keys_different_hash(self):
        k1 = WatermarkGenerator.random_feature_pattern(seed=1, input_dim=INPUT_DIM)
        k2 = WatermarkGenerator.random_feature_pattern(seed=2, input_dim=INPUT_DIM)
        assert k1.key_hash != k2.key_hash

    def test_save_load_roundtrip(self, wm_key):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "wm_key")
            wm_key.save(path)
            loaded = WatermarkKey.load(path)
            assert loaded.owner_id == wm_key.owner_id
            assert loaded.session_id == wm_key.session_id
            assert loaded.key_hash == wm_key.key_hash
            np.testing.assert_array_equal(loaded.trigger_inputs, wm_key.trigger_inputs)
            np.testing.assert_array_equal(loaded.trigger_labels, wm_key.trigger_labels)


class TestWatermarkGenerator:
    def test_random_pattern_shape(self):
        key = WatermarkGenerator.random_feature_pattern(num_triggers=50, input_dim=INPUT_DIM)
        assert key.trigger_inputs.shape == (50, INPUT_DIM)
        assert key.trigger_labels.shape == (50,)

    def test_out_of_distribution_magnitude(self):
        key = WatermarkGenerator.random_feature_pattern(num_triggers=100, input_dim=INPUT_DIM)
        assert np.abs(key.trigger_inputs).max() > 5.0

    def test_content_pattern_shape(self):
        X_normal = np.random.randn(200, INPUT_DIM).astype(np.float32)
        key = WatermarkGenerator.content_pattern(X_normal, num_triggers=40)
        assert key.trigger_inputs.shape == (40, INPUT_DIM)


class TestWatermarkEmbedder:
    def test_embed_skipped_when_not_multiple(self, embedder, model):
        result = embedder.embed(model, nn.CrossEntropyLoss(), round_num=3)
        # embed_frequency=1, so round 3 % 1 == 0 → always embeds
        assert result["embedded"] is True

    def test_embed_increases_trigger_accuracy(self, model, wm_key):
        embedder = WatermarkEmbedder(wm_key, embed_frequency=1, embed_epochs=20,
                                      lr=5e-3, device=torch.device("cpu"))
        acc_before = embedder.verify(model)
        embedder.embed(model, nn.CrossEntropyLoss(), round_num=0)
        acc_after = embedder.verify(model)
        assert acc_after >= acc_before

    def test_verify_returns_float_in_01(self, embedder, model):
        acc = embedder.verify(model)
        assert 0.0 <= acc <= 1.0


class TestOwnershipVerifier:
    def test_verify_ownership_embedded_model(self, model, wm_key):
        embedder = WatermarkEmbedder(wm_key, embed_frequency=1, embed_epochs=30,
                                      lr=1e-2, device=torch.device("cpu"))
        embedder.embed(model, nn.CrossEntropyLoss(), round_num=0)
        verifier = OwnershipVerifier(embedder, threshold=0.5)
        report = verifier.verify_ownership(model)
        assert "is_owner" in report
        assert "trigger_accuracy" in report
        assert "verdict" in report
        assert "key_hash" in report

    def test_unembedded_model_fails_verification(self, model, wm_key):
        embedder = WatermarkEmbedder(wm_key, device=torch.device("cpu"))
        verifier = OwnershipVerifier(embedder, threshold=0.95)
        report = verifier.verify_ownership(model)
        # Fresh untrained model should have low trigger accuracy
        assert report["trigger_accuracy"] < 0.95
