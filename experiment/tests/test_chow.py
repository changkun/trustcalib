"""Proposition 2 (Chow reject option), numerically. The Lean proof is in
lean/TrustCalib/Chow.lean; this checks the Python implementation agrees."""

import numpy as np
import pytest

from experiment.gateway import ALLOW, ASK, BLOCK, SAFETY, SYMMETRIC, Costs, decide, tune_thresholds


def test_closed_form_thresholds():
    c = Costs(c_fa=10.0, c_fb=4.0, c_ask=1.0)
    assert c.tau_high == pytest.approx(0.9)
    assert c.tau_low == pytest.approx(0.25)
    assert c.band_nonempty()
    assert SAFETY.band == pytest.approx((0.25, 0.9))


def test_v1_band_is_symmetric_chow():
    assert SYMMETRIC.band == pytest.approx((0.35, 0.65))


@pytest.mark.parametrize("c", [SYMMETRIC, SAFETY, Costs(3.0, 7.0, 0.5)])
def test_rule_minimizes_expected_loss(c):
    for p in np.linspace(-0.2, 1.2, 1401):
        d = decide(p, c.tau_low, c.tau_high)
        best = min(float(c.loss(a, p)) for a in (ALLOW, ASK, BLOCK))
        assert float(c.loss(d, p)) <= best + 1e-12, (p, d)


def test_empty_band_never_asks_strictly():
    c = Costs(c_fa=1.5, c_fb=1.5, c_ask=1.0)  # 1 * 3 >= 2.25: empty band
    assert not c.band_nonempty()
    for p in np.linspace(0, 1, 101):
        assert min(float(c.loss(ALLOW, p)), float(c.loss(BLOCK, p))) <= float(c.loss(ASK, p))


def test_v1_tuning_degenerates_on_escalated_history():
    """The Go port's Tune() sees only (p_hat, label) pairs recorded at ASK time,
    so every p_hat is inside the band; the v1 grid search then finds no pair
    meeting its caps and returns the default band."""
    rng = np.random.default_rng(0)
    p_hat = rng.uniform(0.35, 0.65, size=500)
    labels = (rng.random(500) < p_hat).astype(int)
    assert tune_thresholds(p_hat, labels) == (0.35, 0.65)
