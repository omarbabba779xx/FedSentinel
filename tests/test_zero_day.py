"""Tests for zero_day VAE + Isolation Forest detector."""
import numpy as np
import pytest
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zero_day.vae_detector import TrafficVAE, ZeroDayDetector


INPUT_DIM = 32


@pytest.fixture
def normal_data():
    rng = np.random.default_rng(42)
    return rng.standard_normal((200, INPUT_DIM)).astype(np.float32)


@pytest.fixture
def anomaly_data():
    rng = np.random.default_rng(99)
    return (rng.standard_normal((30, INPUT_DIM)) * 10 + 50).astype(np.float32)


@pytest.fixture
def detector():
    return ZeroDayDetector(input_dim=INPUT_DIM, latent_dim=8, device=torch.device("cpu"),
                            epochs=3, threshold_percentile=95.0)


class TestTrafficVAE:
    def test_forward_output_shape(self):
        vae = TrafficVAE(input_dim=INPUT_DIM, latent_dim=8)
        x = torch.randn(16, INPUT_DIM)
        recon, mu, logvar = vae(x)
        assert recon.shape == (16, INPUT_DIM)
        assert mu.shape == (16, 8)
        assert logvar.shape == (16, 8)

    def test_reconstruction_error_positive(self):
        vae = TrafficVAE(input_dim=INPUT_DIM, latent_dim=8)
        x = torch.randn(8, INPUT_DIM)
        recon, _, _ = vae(x)
        err = ((x - recon) ** 2).mean(dim=1)
        assert (err >= 0).all()

    def test_reparameterize_stochastic(self):
        vae = TrafficVAE(input_dim=INPUT_DIM, latent_dim=8)
        mu = torch.zeros(4, 8)
        logvar = torch.zeros(4, 8)
        z1 = vae.reparameterize(mu, logvar)
        z2 = vae.reparameterize(mu, logvar)
        assert not torch.allclose(z1, z2)  # Should be different (stochastic)


class TestZeroDayDetector:
    def test_fit_does_not_raise(self, detector, normal_data):
        detector.fit(normal_data)

    def test_threshold_set_after_fit(self, detector, normal_data):
        detector.fit(normal_data)
        assert detector.threshold is not None
        assert detector.threshold > 0

    def test_predict_shape(self, detector, normal_data, anomaly_data):
        detector.fit(normal_data)
        test_data = np.vstack([normal_data[:20], anomaly_data])
        scores, flags = detector.predict(test_data)
        assert len(scores) == len(test_data)
        assert len(flags) == len(test_data)
        assert flags.dtype == bool

    def test_anomalies_flagged_more_than_normal(self, detector, normal_data, anomaly_data):
        detector.fit(normal_data)
        _, normal_flags = detector.predict(normal_data[:50])
        _, anomaly_flags = detector.predict(anomaly_data)
        # Anomalies should have higher flag rate than normal
        assert anomaly_flags.mean() >= normal_flags.mean()

    def test_get_report(self, detector, normal_data):
        detector.fit(normal_data)
        report = detector.get_report()
        assert "vae_threshold" in report
        assert "iso_contamination" in report

    def test_unfitted_detector_raises(self, detector, normal_data):
        with pytest.raises(Exception):
            detector.predict(normal_data[:5])
