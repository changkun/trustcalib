"""On a 1-D toy with known f*(x)=x the GP-probit must recover the boundary."""

import numpy as np
from scipy.stats import norm, spearmanr

from experiment.gp import LaplaceGPC
from experiment.kernel import Packed, ProductKernel


def _packed_1d(xs):
    xs = np.asarray(xs, dtype=float).reshape(-1, 1)
    n = xs.shape[0]
    return Packed(
        phi_tool=xs, phi_ctx=np.zeros((n, 1)), t=np.zeros(n),
        tool_id=np.zeros(n, dtype=int),
    )


def test_recovers_monotone_boundary():
    rng = np.random.default_rng(0)
    x = rng.uniform(-3, 3, size=80)
    # Ground truth: probit with latent f*(x) = x.
    p = norm.cdf(x)
    y = (rng.uniform(size=x.shape) < p).astype(int)

    m = LaplaceGPC(ProductKernel(sigma2=3.0, l_tool=1.2, l_ctx=1.0, lam=1.0))
    m.fit(_packed_1d(x), y)

    grid = np.linspace(-3, 3, 25)
    phat = m.predict_prob(_packed_1d(grid))

    # Posterior approval probability increases with x across the
    # data-supported interior (rank correlation; an RBF GP legitimately
    # reverts toward the prior at the extrapolation edges, so neither
    # pointwise monotonicity nor edge-inclusive rank order is a correct
    # property to assert) ...
    inner = np.abs(grid) <= 2.0
    rho = spearmanr(grid[inner], phat[inner]).statistic
    assert rho > 0.98, rho
    # ... crosses 0.5 near the true boundary x=0 ...
    cross = grid[np.argmin(np.abs(phat - 0.5))]
    assert abs(cross) < 0.6, cross
    # ... and classifies the sign of x well on fresh points.
    xt = rng.uniform(-3, 3, size=200)
    acc = np.mean((m.predict_prob(_packed_1d(xt)) >= 0.5) == (xt > 0))
    assert acc > 0.85, acc
