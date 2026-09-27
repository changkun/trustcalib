"""Proposition 4 (forgetting), numerically, on a fitted Laplace GP-probit.
The Lean proof is in lean/TrustCalib/Forgetting.lean."""

import copy

import numpy as np

from experiment.data import make_stream
from experiment.gp import LaplaceGPC
from experiment.kernel import AdditiveKernel, ProductKernel, pack
from experiment.oracle import OracleConfig, sample_label


def _fit(kernel):
    rng = np.random.default_rng(0)
    cfg = OracleConfig(changepoint=None)
    stream = make_stream(300, seed=11)[::3]
    y = np.array([sample_label(d, rng, cfg) for d in stream])
    return LaplaceGPC(kernel).fit(pack(stream), y), stream


def _at(dp, t):
    q = copy.copy(dp)
    q.t = t
    return q


def test_product_mean_decays_geometrically():
    lam = 90.0
    m, stream = _fit(ProductKernel(1.6, 1.1, 1.2, lam))
    t0 = max(d.t for d in stream)
    for dp in stream[:10]:
        mu0, v0, _ = m.predict(pack([_at(dp, t0)]))
        for delta in (10.0, 90.0, 400.0):
            mu, v, _ = m.predict(pack([_at(dp, t0 + delta)]))
            assert np.allclose(mu, np.exp(-delta / lam) * mu0, atol=1e-10)
            # variance reduction decays by exp(-2 delta / lam)
            prior = m.kernel.prior_var
            assert np.allclose(prior - v, np.exp(-2 * delta / lam) * (prior - v0), atol=1e-8)


def test_product_prediction_returns_to_half():
    m, stream = _fit(ProductKernel(1.6, 1.1, 1.2, 90.0))
    t0 = max(d.t for d in stream)
    p = m.predict_prob(pack([_at(d, t0 + 5000) for d in stream[:10]]))
    assert np.allclose(p, 0.5, atol=1e-6)


def test_additive_mean_converges_to_static_part():
    k = AdditiveKernel(1.6, 1.0, 0.6, lam=90.0)
    m, stream = _fit(k)
    t0 = max(d.t for d in stream)
    static = k.with_(s_global=0.0, s_inter=0.0)
    for dp in stream[:10]:
        mu_static = static.cross(m._P, pack([dp])).T @ m._grad
        mu0, _, _ = m.predict(pack([_at(dp, t0)]))
        for delta in (10.0, 90.0, 400.0):
            mu, _, _ = m.predict(pack([_at(dp, t0 + delta)]))
            assert np.allclose(mu - mu_static, np.exp(-delta / 90.0) * (mu0 - mu_static),
                               atol=1e-10)
