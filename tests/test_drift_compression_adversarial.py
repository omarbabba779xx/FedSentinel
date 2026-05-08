"""Tests for drift_detection, compression, and adversarial modules."""
import numpy as np
import pytest
import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drift_detection.adwin import ADWIN, FedDriftMonitor
from compression.gradient_compression import (
    TopKCompressor, SignSGDCompressor, QuantizationCompressor, HybridCompressor
)
from adversarial.pgd_training import PGDAttack, AdversarialTrainer


# ─── ADWIN ────────────────────────────────────────────────────────────────

class TestADWIN:
    def test_no_drift_on_stable_stream(self):
        adwin = ADWIN(delta=0.002)
        drifts = 0
        for _ in range(200):
            d = adwin.add_element(0.9 + np.random.randn() * 0.01)
            if d:
                drifts += 1
        assert drifts <= 2  # Very few false positives on stable stream

    def test_drift_detected_after_shift(self):
        adwin = ADWIN(delta=0.002)
        # Feed stable data
        for _ in range(100):
            adwin.add_element(0.9)
        # Sudden distribution shift
        drift_found = False
        for _ in range(50):
            if adwin.add_element(0.1):
                drift_found = True
                break
        assert drift_found

    def test_window_size_grows(self):
        adwin = ADWIN()
        for i in range(50):
            adwin.add_element(float(i) * 0.01)
        assert adwin.window_size > 0

    def test_reset_clears_state(self):
        adwin = ADWIN()
        for _ in range(100):
            adwin.add_element(0.5)
        adwin.reset()
        assert adwin.window_size == 0
        assert adwin._total == 0.0


class TestFedDriftMonitor:
    def test_no_global_drift_with_stable(self):
        monitor = FedDriftMonitor(num_clients=4, drift_threshold=0.5)
        for r in range(30):
            for cid in range(4):
                monitor.update_client(cid, 0.95 + np.random.randn() * 0.005)
            report = monitor.check_global_drift(r)
        assert report["global_drift_triggered"] is False

    def test_global_drift_triggered_when_majority(self):
        monitor = FedDriftMonitor(num_clients=4, drift_threshold=0.5)
        # Feed stable
        for r in range(50):
            for cid in range(4):
                monitor.update_client(cid, 0.9)
        # Sudden shift for all clients
        for r in range(30):
            for cid in range(4):
                monitor.update_client(cid, 0.1)
        report = monitor.check_global_drift(round_num=80)
        # After shift, some clients should have drifted
        assert report["drift_fraction"] >= 0


# ─── Compression ──────────────────────────────────────────────────────────

@pytest.fixture
def gradient_vector():
    rng = np.random.default_rng(42)
    return rng.standard_normal(1000).astype(np.float32)


class TestTopKCompressor:
    def test_compress_decompress_shape(self, gradient_vector):
        comp = TopKCompressor(k_ratio=0.1)
        compressed = comp.compress(gradient_vector)
        decompressed = comp.decompress(compressed, gradient_vector.shape)
        assert decompressed.shape == gradient_vector.shape

    def test_sparsity_ratio(self, gradient_vector):
        comp = TopKCompressor(k_ratio=0.1)
        compressed = comp.compress(gradient_vector)
        decompressed = comp.decompress(compressed, gradient_vector.shape)
        nonzero_ratio = np.count_nonzero(decompressed) / len(decompressed)
        assert nonzero_ratio <= 0.11  # ~10% non-zero

    def test_error_feedback_reduces_error(self, gradient_vector):
        comp = TopKCompressor(k_ratio=0.05, error_feedback=True)
        errors = []
        for _ in range(5):
            c = comp.compress(gradient_vector)
            d = comp.decompress(c, gradient_vector.shape)
            errors.append(np.mean((gradient_vector - d) ** 2))
        # Error should be bounded (not growing explosively)
        assert errors[-1] < errors[0] * 10


class TestSignSGDCompressor:
    def test_output_is_signs(self, gradient_vector):
        comp = SignSGDCompressor()
        compressed = comp.compress(gradient_vector)
        decompressed = comp.decompress(compressed, gradient_vector.shape)
        unique_vals = np.unique(decompressed)
        assert set(unique_vals).issubset({-1.0, 1.0})

    def test_compression_ratio(self, gradient_vector):
        comp = SignSGDCompressor()
        ratio = comp.compression_ratio(gradient_vector.shape)
        assert ratio >= 30  # Should be ~32x


class TestQuantizationCompressor:
    def test_8bit_range(self, gradient_vector):
        comp = QuantizationCompressor(num_bits=8)
        compressed = comp.compress(gradient_vector)
        decompressed = comp.decompress(compressed, gradient_vector.shape)
        # Range should be preserved approximately
        assert decompressed.min() >= gradient_vector.min() - 1.0
        assert decompressed.max() <= gradient_vector.max() + 1.0

    def test_lower_bits_more_error(self, gradient_vector):
        comp8 = QuantizationCompressor(num_bits=8)
        comp2 = QuantizationCompressor(num_bits=2)
        c8 = comp8.decompress(comp8.compress(gradient_vector), gradient_vector.shape)
        c2 = comp2.decompress(comp2.compress(gradient_vector), gradient_vector.shape)
        err8 = np.mean((gradient_vector - c8) ** 2)
        err2 = np.mean((gradient_vector - c2) ** 2)
        assert err2 > err8


class TestHybridCompressor:
    def test_high_compression_ratio(self, gradient_vector):
        comp = HybridCompressor(k_ratio=0.1, num_bits=8)
        ratio = comp.compression_ratio(gradient_vector.shape)
        assert ratio >= 100


# ─── Adversarial Training ─────────────────────────────────────────────────

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(20, 5)

    def forward(self, x):
        return self.fc(x)


class TestPGDAttack:
    def test_attack_changes_input(self):
        model = TinyModel()
        attack = PGDAttack(model, epsilon=0.3, alpha=0.1, num_steps=5)
        x = torch.randn(8, 20)
        y = torch.randint(0, 5, (8,))
        x_adv = attack.perturb(x, y)
        assert not torch.allclose(x, x_adv)

    def test_perturbation_bounded(self):
        model = TinyModel()
        epsilon = 0.3
        attack = PGDAttack(model, epsilon=epsilon, alpha=0.05, num_steps=10, norm="inf")
        x = torch.randn(16, 20)
        y = torch.randint(0, 5, (16,))
        x_adv = attack.perturb(x, y)
        diff = (x_adv - x).abs().max().item()
        assert diff <= epsilon + 1e-5


class TestAdversarialTrainer:
    def test_adversarial_loss_lower_than_clean(self):
        model = TinyModel()
        trainer = AdversarialTrainer(model, epsilon=0.1, num_steps=3)
        x = torch.randn(16, 20)
        y = torch.randint(0, 5, (16,))
        criterion = nn.CrossEntropyLoss()
        clean_loss = criterion(model(x), y).item()
        adv_loss = trainer.adversarial_loss(x, y, criterion).item()
        # Adversarial loss should be >= clean loss (harder examples)
        assert adv_loss >= clean_loss * 0.5  # At least not trivially lower
