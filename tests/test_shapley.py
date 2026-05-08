"""Tests for Shapley value incentive mechanism."""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from incentive.shapley import ShapleyCalculator, FedShapleyIncentive


# Simple characteristic function: v(S) = |S| / N (linear)
def linear_val_fn(n_total):
    def v(coalition):
        return len(coalition) / n_total
    return v


# Non-linear: one dominant client (client 0 contributes 80%)
def dominant_val_fn(coalition):
    if 0 in coalition and len(coalition) >= 2:
        return 0.9
    elif 0 in coalition:
        return 0.7
    elif len(coalition) > 0:
        return 0.1 * len(coalition)
    return 0.0


class TestShapleyCalculator:
    def test_exact_sums_to_grand_coalition(self):
        n = 4
        calc = ShapleyCalculator(num_clients=n, val_function=linear_val_fn(n))
        shapley = calc.compute_exact()
        total = sum(shapley.values())
        v_grand = calc._cached_val(list(range(n)))
        assert total == pytest.approx(v_grand, abs=1e-6)

    def test_mc_sums_approximately_to_grand(self):
        n = 5
        calc = ShapleyCalculator(num_clients=n, val_function=linear_val_fn(n),
                                   mc_iterations=500)
        shapley = calc.compute_monte_carlo()
        total = sum(shapley.values())
        v_grand = calc._cached_val(list(range(n)))
        assert total == pytest.approx(v_grand, abs=0.05)

    def test_mc_dominant_client_highest(self):
        n = 4
        calc = ShapleyCalculator(num_clients=n, val_function=dominant_val_fn,
                                   mc_iterations=500)
        shapley = calc.compute_monte_carlo()
        assert shapley[0] == max(shapley.values())

    def test_group_testing_sums_to_grand(self):
        n = 6
        calc = ShapleyCalculator(num_clients=n, val_function=linear_val_fn(n))
        shapley = calc.compute_group_testing()
        total = sum(shapley.values())
        v_grand = calc._cached_val(list(range(n)))
        assert total == pytest.approx(v_grand, abs=0.1)

    def test_normalize_range_01(self):
        n = 3
        calc = ShapleyCalculator(num_clients=n, val_function=dominant_val_fn,
                                   mc_iterations=200)
        shapley = calc.compute_monte_carlo()
        norm = calc.normalize(shapley)
        vals = list(norm.values())
        assert min(vals) == pytest.approx(0.0, abs=1e-6)
        assert max(vals) == pytest.approx(1.0, abs=1e-6)

    def test_symmetry_linear_function(self):
        n = 4
        calc = ShapleyCalculator(num_clients=n, val_function=linear_val_fn(n))
        shapley = calc.compute_exact()
        # All clients equal in linear function → equal Shapley values
        values = list(shapley.values())
        assert max(values) - min(values) < 1e-6


class TestFedShapleyIncentive:
    def test_compute_round_shapley_returns_dict(self):
        inc = FedShapleyIncentive(num_clients=4, method="monte_carlo",
                                   mc_iterations=50)
        shapley = inc.compute_round_shapley(
            client_models={},
            val_fn=linear_val_fn(4),
        )
        assert len(shapley) == 4
        assert all(0.0 <= v <= 1.0 for v in shapley.values())

    def test_selection_probabilities_sum_to_one(self):
        inc = FedShapleyIncentive(num_clients=5, method="monte_carlo",
                                   mc_iterations=50)
        inc.compute_round_shapley({}, val_fn=linear_val_fn(5))
        probs = inc.get_selection_probabilities()
        assert probs.sum() == pytest.approx(1.0, abs=1e-6)
        assert len(probs) == 5

    def test_select_clients_correct_count(self):
        inc = FedShapleyIncentive(num_clients=6, method="monte_carlo",
                                   mc_iterations=50)
        inc.compute_round_shapley({}, val_fn=linear_val_fn(6))
        selected = inc.select_clients(num_to_select=3)
        assert len(selected) == 3
        assert len(set(selected)) == 3  # No duplicates

    def test_free_rider_flagging(self):
        n = 5
        inc = FedShapleyIncentive(num_clients=n, method="monte_carlo",
                                   mc_iterations=100)

        def free_rider_val(coalition):
            return sum(1 for c in coalition if c != 4) / n

        inc.compute_round_shapley({}, val_fn=free_rider_val)
        flagged = inc.flag_free_riders(threshold=0.05)
        assert 4 in flagged  # Client 4 contributes nothing

    def test_reward_report_structure(self):
        inc = FedShapleyIncentive(num_clients=3)
        inc.compute_round_shapley({}, val_fn=linear_val_fn(3))
        report = inc.get_reward_report()
        assert "cumulative_scores" in report
        assert "selection_probabilities" in report
        assert "top_contributor" in report
        assert "round" in report
