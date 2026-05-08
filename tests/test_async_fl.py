"""Tests for async_fl (FedBuff) module."""
import numpy as np
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from async_fl.fedbuff import FedBuffServer, AsyncFLClient, ClientUpdate


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 3)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def model():
    return SimpleMLP()


@pytest.fixture
def server(model):
    return FedBuffServer(global_model=model, buffer_size=3, lr=0.01)


class TestClientUpdate:
    def test_fields(self, model):
        params = [p.detach().numpy().copy() for p in model.parameters()]
        update = ClientUpdate(client_id=1, params=params, staleness=2, num_samples=100)
        assert update.client_id == 1
        assert update.staleness == 2
        assert update.num_samples == 100
        assert len(update.params) == len(params)

    def test_staleness_weight(self):
        update = ClientUpdate(client_id=0, params=[], staleness=0, num_samples=10)
        assert update.staleness_weight == pytest.approx(1.0)
        update2 = ClientUpdate(client_id=0, params=[], staleness=3, num_samples=10)
        assert update2.staleness_weight == pytest.approx(1.0 / 4.0)


class TestFedBuffServer:
    def test_buffer_not_full(self, server, model):
        params = [p.detach().numpy().copy() for p in model.parameters()]
        update = ClientUpdate(client_id=0, params=params, staleness=0, num_samples=50)
        aggregated = server.receive_update(update)
        assert aggregated is False
        assert len(server._buffer) == 1

    def test_buffer_triggers_aggregation(self, server, model):
        params = [p.detach().numpy().copy() for p in model.parameters()]
        for i in range(3):
            update = ClientUpdate(client_id=i, params=params, staleness=i, num_samples=50)
            result = server.receive_update(update)
        assert result is True
        assert len(server._buffer) == 0  # Buffer cleared after aggregation
        assert server.round_count == 1

    def test_aggregation_preserves_param_shape(self, server, model):
        params = [p.detach().numpy().copy() for p in model.parameters()]
        for i in range(3):
            server.receive_update(ClientUpdate(client_id=i, params=params, staleness=0, num_samples=100))
        new_params = [p.detach().numpy().copy() for p in model.parameters()]
        for orig, new in zip(params, new_params):
            assert orig.shape == new.shape

    def test_get_global_params(self, server, model):
        global_params = server.get_global_params()
        model_params = [p.detach().numpy() for p in model.parameters()]
        assert len(global_params) == len(model_params)
        for gp, mp in zip(global_params, model_params):
            np.testing.assert_array_equal(gp, mp)


class TestAsyncFLClient:
    def test_init(self, model):
        X = np.random.randn(100, 10).astype(np.float32)
        y = np.random.randint(0, 3, 100).astype(np.int64)
        client = AsyncFLClient(client_id=0, model=model, X_train=X, y_train=y)
        assert client.client_id == 0
        assert client.current_round == 0

    def test_local_train_returns_update(self, model):
        X = np.random.randn(50, 10).astype(np.float32)
        y = np.random.randint(0, 3, 50).astype(np.int64)
        client = AsyncFLClient(client_id=1, model=model, X_train=X, y_train=y,
                                local_epochs=1, lr=1e-3)
        server_params = [p.detach().numpy().copy() for p in model.parameters()]
        update = client.local_train(server_params, server_round=0)
        assert isinstance(update, ClientUpdate)
        assert update.client_id == 1
        assert len(update.params) == len(server_params)
