"""Unit tests for draft surplus exponential decay valuation model."""

import numpy as np
import pytest

from nfl2027.evaluation.valuation import (
    expected_career_snaps,
    calculate_surplus_snaps,
    fit_draft_decay_curve,
    calculate_financial_surplus,
)


def test_expected_snaps_monotonic_decay():
    """Verify that expected snaps strictly decrease with higher draft pick number."""
    picks = np.array([1, 10, 32, 64, 100, 150, 200, 256])
    expected = expected_career_snaps(picks)

    # First pick should have maximum expected snaps
    assert expected[0] > expected[1]
    # Consecutive differences should all be negative (strictly decreasing)
    assert np.all(np.diff(expected) < 0)
    # Check bounds
    assert expected[0] < 2000.0
    assert expected[-1] > 0.0


def test_surplus_snaps_calculation():
    """Verify surplus snaps matches actual minus expected."""
    actual = 1500.0
    pick = 50
    exp = expected_career_snaps(pick)
    surplus = calculate_surplus_snaps(actual, pick)

    assert pytest.approx(surplus) == (actual - exp)


def test_curve_fit_recovers_synthetic_parameters():
    """Verify curve fitting recovers exponential decay parameters from noisy data."""
    rng = np.random.RandomState(42)
    picks = np.arange(1, 250, 2)
    true_A = 1850.0
    true_lambda = 0.0095
    true_C = 20.0

    snaps = true_A * np.exp(-true_lambda * picks) + true_C + rng.normal(0, 10.0, size=len(picks))

    A_fit, lambda_fit, C_fit, metrics = fit_draft_decay_curve(picks, snaps)

    assert metrics["r2"] > 0.90
    assert abs(A_fit - true_A) < 100.0
    assert abs(lambda_fit - true_lambda) < 0.002


def test_financial_surplus_by_position():
    """Verify financial surplus converts snaps to dollar values based on position."""
    surplus_snaps = 200.0
    # DE cap rate: 16,000 / snap
    val_de = calculate_financial_surplus(surplus_snaps, "DE")
    assert val_de == 200.0 * 16000.0

    # WR cap rate: 13,500 / snap
    val_wr = calculate_financial_surplus(surplus_snaps, "WR")
    assert val_wr == 200.0 * 13500.0
