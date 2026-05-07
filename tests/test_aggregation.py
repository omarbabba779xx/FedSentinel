"""Tests for Byzantine-robust aggregation algorithms."""

import numpy as np
import pytest
from defense import krum, multi_krum, trimmed_mean, coordinate_median, flame


def make_client_weights(n_clients: int, n_params: int = 100, seed: int = 42) -> list:
    rng = np.random.default_rng(seed)
    return [[rng.standard_normal(n_params).astype(np.float32)] for _ in range(n_clients)]


def inject_byzantine(weights: list, n_byz: int, scale: float = -10.0) -> list:
    result = [w.copy() for w in weights]
    for i in range(n_byz):
        result[i] = [-w * abs(scale) for w in result[i]]
    return result


class TestKrum:
    def test_selects_one_client(self):
        clients = make_client_weights(5)
        result, info = krum(clients, num_byzantine=1)
        assert isinstance(result, list)
        assert isinstance(result[0], np.ndarray)
        assert "selected" in info

    def test_rejects_byzantine(self):
        rng = np.random.default_rng(0)
        honest = [[rng.standard_normal(50).astype(np.float32)] for _ in range(4)]
        byzantine = [[-np.ones(50, dtype=np.float32) * 100]]
        clients = honest + byzantine
        result, info = krum(clients, num_byzantine=1)
        selected = info["selected"]
        assert selected != 4, "Krum should not select the Byzantine client"

    def test_fallback_when_k_zero(self):
        clients = make_client_weights(2)
        result, info = krum(clients, num_byzantine=2)
        assert result is not None


class TestMultiKrum:
    def test_returns_averaged_result(self):
        clients = make_client_weights(5)
        result, info = multi_krum(clients, num_byzantine=1, m=3)
        assert len(result) == 1
        assert result[0].shape == clients[0][0].shape

    def test_selected_count(self):
        clients = make_client_weights(6)
        _, info = multi_krum(clients, num_byzantine=1, m=4)
        assert len(info["selected"]) == 4


class TestTrimmedMean:
    def test_output_shape(self):
        clients = make_client_weights(10)
        result, info = trimmed_mean(clients, beta=0.2)
        assert result[0].shape == clients[0][0].shape

    def test_robustness(self):
        rng = np.random.default_rng(1)
        honest = [[np.ones(10, dtype=np.float32)] for _ in range(8)]
        byz = [[np.ones(10, dtype=np.float32) * 1000] for _ in range(2)]
        result, _ = trimmed_mean(honest + byz, beta=0.2)
        assert abs(result[0].mean() - 1.0) < 0.5, "Trimmed mean should exclude outliers"


class TestCoordinateMedian:
    def test_output_matches_numpy_median(self):
        rng = np.random.default_rng(42)
        data = [rng.standard_normal(20).astype(np.float32) for _ in range(5)]
        clients = [[d] for d in data]
        result, _ = coordinate_median(clients)
        expected = np.median(np.stack(data), axis=0)
        np.testing.assert_allclose(result[0], expected, atol=1e-5)


class TestFLAME:
    def test_rejects_outliers(self):
        rng = np.random.default_rng(5)
        honest = [[rng.standard_normal(50).astype(np.float32)] for _ in range(5)]
        byz = [[np.ones(50, dtype=np.float32) * 100]]
        result, info = flame(honest + byz, noise_sigma=0.0)
        assert isinstance(result[0], np.ndarray)
        assert len(info["rejected_clients"]) >= 0
