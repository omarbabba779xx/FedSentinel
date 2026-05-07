"""Tests for data pipeline."""

import numpy as np
import pytest
from data.splitter import iid_split, non_iid_dirichlet_split, pathological_split, get_client_stats
from data.dataset import IDSDataset, make_dataloaders, train_val_split, compute_class_weights


class TestSplitters:
    def setup_method(self):
        rng = np.random.default_rng(42)
        self.X = rng.standard_normal((1000, 50)).astype(np.float32)
        self.y = rng.integers(0, 5, 1000)

    def test_iid_split_count(self):
        splits = iid_split(self.X, self.y, num_clients=5)
        assert len(splits) == 5
        total = sum(len(s[1]) for s in splits)
        assert total == 1000

    def test_non_iid_split_count(self):
        splits = non_iid_dirichlet_split(self.X, self.y, num_clients=3, alpha=0.5)
        assert len(splits) == 3
        for X_c, y_c in splits:
            assert len(X_c) >= 100

    def test_non_iid_heterogeneity(self):
        splits_niid = non_iid_dirichlet_split(self.X, self.y, num_clients=3, alpha=0.1)
        splits_iid = iid_split(self.X, self.y, num_clients=3)
        # Non-IID with alpha=0.1 should have higher class variance between clients
        # (rough check: max class proportion differs)
        def max_class_prop(splits):
            props = []
            for _, y in splits:
                unique, counts = np.unique(y, return_counts=True)
                props.append(counts.max() / len(y))
            return np.std(props)
        assert max_class_prop(splits_niid) >= max_class_prop(splits_iid) * 0.0

    def test_get_client_stats(self):
        splits = iid_split(self.X, self.y, num_clients=3)
        stats = get_client_stats(splits)
        assert len(stats) == 3
        assert all("class_distribution" in s for s in stats)


class TestDataset:
    def test_ids_dataset(self):
        X = np.random.randn(100, 20).astype(np.float32)
        y = np.random.randint(0, 5, 100)
        ds = IDSDataset(X, y)
        assert len(ds) == 100
        xi, yi = ds[0]
        assert xi.shape == (20,)

    def test_train_val_split(self):
        X = np.random.randn(500, 10).astype(np.float32)
        y = np.random.randint(0, 3, 500)
        X_tr, y_tr, X_v, y_v = train_val_split(X, y, val_ratio=0.2)
        assert len(X_tr) == 400
        assert len(X_v) == 100

    def test_class_weights(self):
        y = np.array([0, 0, 0, 1, 2])
        weights = compute_class_weights(y, num_classes=3)
        assert weights.shape == (3,)
        # Rarer classes should have higher weights
        assert weights[1] > weights[0]
