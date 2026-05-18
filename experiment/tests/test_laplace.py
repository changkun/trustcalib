"""Correctness of the Laplace GP-probit (Rasmussen & Williams Alg. 3.1/3.2)."""

import numpy as np
from scipy.stats import norm

from experiment.gp import LaplaceGPC, _probit_derivs
from experiment.kernel import Packed, ProductKernel


def _packed_1d(xs):
    xs = np.asarray(xs, dtype=float).reshape(-1, 1)
    n = xs.shape[0]
    return Packed(
        phi_tool=xs, phi_ctx=np.zeros((n, 1)), t=np.zeros(n),
        tool_id=np.zeros(n, dtype=int),
    )


def _kern():
    # With zero ctx/time blocks this reduces to a standard 1-D RBF GP.
    return ProductKernel(sigma2=2.0, l_tool=1.0, l_ctx=1.0, lam=1.0)


def test_mode_satisfies_stationarity_condition():
    """At the Laplace mode, R&W Alg. 3.1 must satisfy f_hat = K * grad log p."""
    x = np.array([-2.0, -1.0, 0.5, 1.5, 2.5])
    y = np.array([0, 0, 1, 1, 1])
    m = LaplaceGPC(_kern(), tol=1e-10, max_iter=200).fit(_packed_1d(x), y)

    yp = np.where(y > 0, 1.0, -1.0)
    _, grad, _ = _probit_derivs(m._f_hat, yp)
    residual = m._f_hat - m._K @ grad
    assert np.linalg.norm(residual) < 1e-5, residual
    assert np.isfinite(m.log_marginal)


def test_n2_hand_example_symmetry_and_sign():
    """n=2 hand case: two approvals at +/- a. By kernel symmetry the mode is
    symmetric and positive (approval -> positive latent), and p_hat > 0.5."""
    m = LaplaceGPC(_kern()).fit(_packed_1d([-1.0, 1.0]), np.array([1, 1]))
    f = m._f_hat
    assert abs(f[0] - f[1]) < 1e-6, f
    assert np.all(f > 0.0), f
    _, _, pi = m.predict(_packed_1d([-1.0, 1.0]))
    assert np.all(pi > 0.5) and np.all(pi < 1.0)

    # Two denials: mirror image, negative latent, p_hat < 0.5.
    m2 = LaplaceGPC(_kern()).fit(_packed_1d([-1.0, 1.0]), np.array([0, 0]))
    assert np.all(m2._f_hat < 0.0)
    assert np.all(m2.predict(_packed_1d([0.0]))[2] < 0.5)


def test_more_approvals_raise_phat():
    k = _kern()
    base = LaplaceGPC(k).fit(_packed_1d([0.0]), np.array([1]))
    denied = LaplaceGPC(k).fit(_packed_1d([0.0]), np.array([0]))
    p_app = base.predict_prob(_packed_1d([0.0]))[0]
    p_den = denied.predict_prob(_packed_1d([0.0]))[0]
    assert p_app > 0.5 > p_den
    assert 0.0 < p_den < p_app < 1.0


def test_probit_derivative_matches_finite_difference():
    f = np.array([-1.3, 0.2, 2.1])
    y = np.array([1.0, -1.0, 1.0])
    _, grad, W = _probit_derivs(f, y)
    eps = 1e-6
    for i in range(len(f)):
        fp, fm = f.copy(), f.copy()
        fp[i] += eps
        fm[i] -= eps
        d = (np.sum(norm.logcdf(y * fp)) - np.sum(norm.logcdf(y * fm))) / (
            2 * eps
        )
        assert abs(d - grad[i]) < 1e-4, (i, d, grad[i])
