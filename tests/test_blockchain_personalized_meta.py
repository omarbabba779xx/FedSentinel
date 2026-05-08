"""Tests for blockchain, personalized FL, and meta-learning modules."""
import numpy as np
import pytest
import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from blockchain.audit_chain import Block, FLAuditChain


# ─── Blockchain ───────────────────────────────────────────────────────────

class TestBlock:
    def test_hash_is_sha256(self):
        b = Block(index=0, timestamp=1000.0, data={"round": 1}, previous_hash="0" * 64)
        b.mine(difficulty=1)
        assert len(b.hash) == 64

    def test_hash_starts_with_difficulty(self):
        b = Block(index=0, timestamp=1000.0, data={"test": True}, previous_hash="0" * 64)
        b.mine(difficulty=2)
        assert b.hash.startswith("00")

    def test_data_integrity(self):
        data = {"round": 5, "participants": [0, 1, 2], "epsilon": 0.45}
        b = Block(index=1, timestamp=2000.0, data=data, previous_hash="abc123")
        assert b.data["round"] == 5
        assert b.data["epsilon"] == 0.45


class TestFLAuditChain:
    def test_genesis_block_created(self):
        chain = FLAuditChain(difficulty=1)
        assert len(chain.chain) == 1
        assert chain.chain[0].index == 0

    def test_add_round_increases_length(self):
        chain = FLAuditChain(difficulty=1)
        chain.add_round(
            round_num=1,
            participants=[0, 1, 2],
            model_hash="abc",
            aggregation_strategy="fedavg",
            privacy_epsilon=0.5,
        )
        assert len(chain.chain) == 2

    def test_chain_is_valid_after_adds(self):
        chain = FLAuditChain(difficulty=1)
        for r in range(5):
            chain.add_round(round_num=r, participants=[0, 1], model_hash=f"hash_{r}",
                            aggregation_strategy="krum", privacy_epsilon=0.1 * r)
        assert chain.verify_chain() is True

    def test_tampered_chain_is_invalid(self):
        chain = FLAuditChain(difficulty=1)
        chain.add_round(round_num=1, participants=[0], model_hash="h1",
                        aggregation_strategy="fedavg", privacy_epsilon=0.5)
        chain.add_round(round_num=2, participants=[1], model_hash="h2",
                        aggregation_strategy="fedavg", privacy_epsilon=0.5)
        # Tamper block 1
        chain.chain[1].data["epsilon"] = 999.0
        assert chain.verify_chain() is False

    def test_export_audit_report_structure(self):
        chain = FLAuditChain(difficulty=1)
        chain.add_round(round_num=1, participants=[0, 1], model_hash="h",
                        aggregation_strategy="flame", privacy_epsilon=0.8)
        report = chain.export_audit_report()
        assert "chain_length" in report
        assert "valid" in report
        assert "blocks" in report


# ─── Personalized FL ──────────────────────────────────────────────────────

class TestPersonalizedFL:
    def test_ditto_import(self):
        from personalized.ditto import DittoClient
        assert DittoClient is not None

    def test_ditto_client_init(self):
        from personalized.ditto import DittoClient
        X = np.random.randn(80, 20).astype(np.float32)
        y = np.random.randint(0, 3, 80).astype(np.int64)
        client = DittoClient(client_id=0, X_train=X, y_train=y,
                              input_dim=20, num_classes=3, lambda_prox=0.1)
        assert client.client_id == 0
        assert client.personal_model is not None

    def test_pfedme_import(self):
        from personalized.pfedme import pFedMeClient
        assert pFedMeClient is not None

    def test_pfedme_client_init(self):
        from personalized.pfedme import pFedMeClient
        X = np.random.randn(60, 15).astype(np.float32)
        y = np.random.randint(0, 2, 60).astype(np.int64)
        client = pFedMeClient(client_id=1, X_train=X, y_train=y,
                               input_dim=15, num_classes=2)
        assert client.client_id == 1


# ─── Meta-learning ────────────────────────────────────────────────────────

class TestFedMAML:
    def test_maml_import(self):
        from meta_learning.fedmaml import FedMAMLClient, FedMAMLServer
        assert FedMAMLClient is not None
        assert FedMAMLServer is not None

    def test_maml_inner_loop(self):
        from meta_learning.fedmaml import MAMLInnerLoop

        class TinyNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 3)

            def forward(self, x):
                return self.fc(x)

        model = TinyNet()
        loop = MAMLInnerLoop(model, lr=0.01, num_steps=3)
        X_support = torch.randn(5, 10)
        y_support = torch.randint(0, 3, (5,))
        adapted = loop.adapt(X_support, y_support, nn.CrossEntropyLoss())
        # Adapted model should have same structure but different params
        for p_orig, p_adapted in zip(model.parameters(), adapted.parameters()):
            assert p_orig.shape == p_adapted.shape
