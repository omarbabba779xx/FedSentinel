"""Tests for privacy mechanisms."""

import numpy as np
import pytest
from privacy import DPGradientProcessor, PrivacyAccountant, compute_noise_multiplier


class TestDPGradientProcessor:
    def test_clipping(self):
        dp = DPGradientProcessor(max_grad_norm=1.0)
        weights = [np.ones(100, dtype=np.float32) * 10.0]
        clipped, norm = dp.clip_weights(weights)
        flat = np.concatenate([w.flatten() for w in clipped])
        assert abs(np.linalg.norm(flat) - 1.0) < 0.01

    def test_noise_added(self):
        dp = DPGradientProcessor(max_grad_norm=1.0, noise_multiplier=1.0)
        weights = [np.zeros(100, dtype=np.float32)]
        noisy = dp.add_noise(weights)
        assert not np.allclose(noisy[0], 0.0), "Noise should be added"

    def test_privatize_returns_stats(self):
        dp = DPGradientProcessor(max_grad_norm=1.0, noise_multiplier=1.1)
        weights = [np.random.randn(50).astype(np.float32)]
        result, stats = dp.privatize(weights)
        assert "gradient_norm_before" in stats
        assert "noise_std" in stats


class TestPrivacyAccountant:
    def test_epsilon_increases(self):
        acc = PrivacyAccountant(target_epsilon=1.0, target_delta=1e-5)
        eps1, _ = acc.step(noise_multiplier=1.1, sample_rate=0.1, num_steps=100)
        eps2, _ = acc.step(noise_multiplier=1.1, sample_rate=0.1, num_steps=100)
        assert eps2 >= eps1, "Epsilon should be non-decreasing"

    def test_budget_exceeded_flag(self):
        acc = PrivacyAccountant(target_epsilon=0.001, target_delta=1e-5)
        # With very small budget, should exceed quickly
        _, exceeded = acc.step(noise_multiplier=0.1, sample_rate=1.0, num_steps=10000)
        assert exceeded, "Budget should be exceeded with large noise and many steps"

    def test_reset(self):
        acc = PrivacyAccountant()
        acc.step(noise_multiplier=1.1, sample_rate=0.1, num_steps=100)
        acc.reset()
        assert acc.current_epsilon == 0.0 or acc.total_rounds == 0


class TestNoiseMultiplier:
    def test_returns_positive(self):
        sigma = compute_noise_multiplier(1.0, 1e-5, sample_rate=0.1, steps=100)
        assert sigma > 0

    def test_higher_epsilon_lower_noise(self):
        sigma_strict = compute_noise_multiplier(0.5, 1e-5, sample_rate=0.1, steps=100)
        sigma_loose = compute_noise_multiplier(5.0, 1e-5, sample_rate=0.1, steps=100)
        assert sigma_loose <= sigma_strict, "Looser budget → less noise needed"
