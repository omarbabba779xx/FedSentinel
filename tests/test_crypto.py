"""Tests for crypto module: ZKP gradient proofs."""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto.zkp import GradientCommitmentScheme


@pytest.fixture
def scheme():
    return GradientCommitmentScheme(secret_key="test_secret_key_fedsentinel")


@pytest.fixture
def gradient():
    rng = np.random.default_rng(7)
    return rng.standard_normal(50).astype(np.float32)


class TestGradientCommitmentScheme:
    def test_commit_returns_dict(self, scheme, gradient):
        result = scheme.commit(gradient, client_id=0, round_num=1)
        assert "commitment" in result
        assert "nonce" in result
        assert "client_id" in result
        assert "round_num" in result

    def test_commitment_is_deterministic_given_same_nonce(self, scheme, gradient):
        c1 = scheme.commit(gradient, client_id=0, round_num=1)
        # Same nonce should give same commitment
        c2 = scheme._compute_commitment(gradient, c1["nonce"])
        assert c1["commitment"] == c2

    def test_different_gradients_give_different_commitments(self, scheme):
        g1 = np.ones(20, dtype=np.float32)
        g2 = np.zeros(20, dtype=np.float32)
        c1 = scheme.commit(g1, client_id=0, round_num=1)
        c2 = scheme.commit(g2, client_id=0, round_num=1)
        assert c1["commitment"] != c2["commitment"]

    def test_verify_honest_gradient(self, scheme, gradient):
        commitment = scheme.commit(gradient, client_id=0, round_num=1)
        challenge = scheme.generate_challenge()
        response = scheme.respond_to_challenge(gradient, commitment["nonce"], challenge)
        valid = scheme.verify(
            commitment=commitment["commitment"],
            challenge=challenge,
            response=response,
            gradient=gradient,
            nonce=commitment["nonce"],
        )
        assert valid is True

    def test_verify_tampered_gradient_fails(self, scheme, gradient):
        commitment = scheme.commit(gradient, client_id=0, round_num=1)
        challenge = scheme.generate_challenge()
        response = scheme.respond_to_challenge(gradient, commitment["nonce"], challenge)

        tampered = gradient.copy()
        tampered[0] += 999.0
        valid = scheme.verify(
            commitment=commitment["commitment"],
            challenge=challenge,
            response=response,
            gradient=tampered,
            nonce=commitment["nonce"],
        )
        assert valid is False

    def test_batch_verify_all_honest(self, scheme):
        n = 5
        gradients = [np.random.randn(30).astype(np.float32) for _ in range(n)]
        proofs = []
        for i, g in enumerate(gradients):
            c = scheme.commit(g, client_id=i, round_num=1)
            ch = scheme.generate_challenge()
            resp = scheme.respond_to_challenge(g, c["nonce"], ch)
            proofs.append({
                "gradient": g,
                "commitment": c["commitment"],
                "nonce": c["nonce"],
                "challenge": ch,
                "response": resp,
            })
        results = scheme.batch_verify(proofs)
        assert all(results.values())

    def test_batch_verify_detects_one_tampered(self, scheme):
        n = 4
        gradients = [np.random.randn(20).astype(np.float32) for _ in range(n)]
        proofs = []
        for i, g in enumerate(gradients):
            c = scheme.commit(g, client_id=i, round_num=1)
            ch = scheme.generate_challenge()
            tampered = g.copy() if i != 2 else g + 100
            resp = scheme.respond_to_challenge(g, c["nonce"], ch)
            proofs.append({
                "gradient": tampered,
                "commitment": c["commitment"],
                "nonce": c["nonce"],
                "challenge": ch,
                "response": resp,
            })
        results = scheme.batch_verify(proofs)
        assert results[2] is False
