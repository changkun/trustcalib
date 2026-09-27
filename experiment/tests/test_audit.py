"""Proposition 5 (audit estimator), by Monte Carlo. The Lean proof is in
lean/TrustCalib/Audit.lean."""

import numpy as np


def test_ipw_audit_estimator_unbiased():
    rng = np.random.default_rng(0)
    e = (rng.random(400) < 0.07).astype(float)  # fixed false-allow indicators
    eps = 0.05
    est = []
    for _ in range(20000):
        a = rng.random(e.size) < eps
        est.append(np.sum(a * e) / (e.size * eps))
    assert abs(np.mean(est) - e.mean()) < 3 * np.std(est) / np.sqrt(len(est))


def test_certification_sample_size():
    alpha, delta = 0.01, 0.05
    n = int(np.ceil(np.log(1 / delta) / alpha))
    assert (1 - alpha) ** n <= delta
    assert n == 300
