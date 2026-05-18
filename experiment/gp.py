"""Self-contained Laplace approximation for GP probit classification.

This is the inference machinery the manuscript cites verbatim (Rasmussen &
Williams 2006, Algorithm 3.1 for the posterior mode and Algorithm 3.2 for
predictions), specialized to the probit likelihood of Definition 1. Keeping it
dependency-light (numpy + scipy only) makes the structural identity with
Preferential Bayesian Optimization legible rather than hidden inside a tensor
framework. An optional BoTorch ``PairwiseGP`` cross-check lives in
:mod:`experiment.gp_botorch`.

Likelihood (probit), with labels mapped to ``y in {-1, +1}`` and ``z = y f``:

    log p(y | f) = log Phi(z)
    d/df  log p  = y * phi(z) / Phi(z)                       (R&W 3.15)
    W = -d^2/df^2 log p = (phi(z)/Phi(z))^2 + z phi(z)/Phi(z) (R&W 3.16, >= 0)

Predictive class probability uses the probit-Gaussian convolution
``pi_* = Phi(f_bar_* / sqrt(1 + V[f_*]))`` (R&W 3.25).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.stats import norm

from .kernel import Packed, ProductKernel

_LOG2PI = np.log(2.0 * np.pi)


def _probit_derivs(f: np.ndarray, y: np.ndarray):
    """Return (log_lik_sum, grad, W) for the probit likelihood at ``f``.

    ``y`` is in ``{-1, +1}``. Uses the numerically stable log CDF and the
    Mills-ratio form to avoid underflow when ``z`` is very negative.
    """
    z = y * f
    log_phi = norm.logpdf(z)
    log_Phi = norm.logcdf(z)
    # Mills ratio r = phi(z) / Phi(z), evaluated as exp(log phi - log Phi).
    r = np.exp(log_phi - log_Phi)
    grad = y * r
    W = r * r + z * r  # = (phi/Phi)^2 + z phi/Phi  (>= 0 for probit)
    W = np.maximum(W, 1e-10)
    return float(np.sum(log_Phi)), grad, W


class LaplaceGPC:
    """GP probit classifier via the Laplace approximation."""

    def __init__(
        self,
        kernel: ProductKernel,
        jitter: float = 1e-6,
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        self.kernel = kernel
        self.jitter = jitter
        self.max_iter = max_iter
        self.tol = tol
        self._fitted = False

    def fit(self, P: Packed, y01: np.ndarray) -> "LaplaceGPC":
        """Find the posterior mode (R&W Algorithm 3.1).

        ``y01`` are 0/1 approve labels; mapped internally to ``{-1, +1}``.
        """
        y = np.where(np.asarray(y01) > 0, 1.0, -1.0)
        n = y.shape[0]
        K = self.kernel.full(P) + self.jitter * np.eye(n)

        f = np.zeros(n)
        last_obj = -np.inf
        for _ in range(self.max_iter):
            ll, grad, W = _probit_derivs(f, y)
            sW = np.sqrt(W)
            B = np.eye(n) + (sW[:, None] * K) * sW[None, :]
            L = cholesky(B, lower=True)
            b = W * f + grad
            # a = b - sW * B^{-1} (sW K b)
            tmp = solve_triangular(L, sW * (K @ b), lower=True)
            tmp = solve_triangular(L, tmp, lower=True, trans=1)
            a = b - sW * tmp
            f = K @ a
            obj = -0.5 * float(a @ f) + ll
            if abs(obj - last_obj) < self.tol:
                last_obj = obj
                break
            last_obj = obj

        ll, grad, W = _probit_derivs(f, y)
        sW = np.sqrt(W)
        B = np.eye(n) + (sW[:, None] * K) * sW[None, :]
        L = cholesky(B, lower=True)

        self._P = P
        self._y = y
        self._K = K
        self._f_hat = f
        self._grad = grad
        self._sW = sW
        self._L = L
        # Laplace log marginal likelihood (R&W 3.32):
        #   log Z = -0.5 a^T f + log p(y|f) - sum log diag(L)
        self.log_marginal = (
            -0.5 * float(a @ f) + ll - float(np.sum(np.log(np.diag(L))))
        )
        self._fitted = True
        return self

    def predict(self, Q: Packed):
        """Predictive latent mean/var and class probability (R&W Alg 3.2).

        Returns ``(f_bar, var, pi)`` where ``pi = Phi(f_bar/sqrt(1+var))`` is
        the posterior-predictive approval probability ``p_hat(x_*)`` of the
        manuscript's decision rule (Section 5).
        """
        if not self._fitted:
            raise RuntimeError("call fit() before predict()")
        Ks = self.kernel.cross(self._P, Q)            # (n_train, n_test)
        kss = self.kernel.diag(Q)                     # (n_test,)
        f_bar = Ks.T @ self._grad                     # mean (a_hat = grad)
        v = solve_triangular(self._L, self._sW[:, None] * Ks, lower=True)
        var = kss - np.sum(v * v, axis=0)
        var = np.maximum(var, 1e-12)
        pi = norm.cdf(f_bar / np.sqrt(1.0 + var))
        return f_bar, var, pi

    def predict_prob(self, Q: Packed) -> np.ndarray:
        """Just ``p_hat(x_*)`` for each test point."""
        return self.predict(Q)[2]
