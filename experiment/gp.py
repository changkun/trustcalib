"""Self-contained Laplace approximation for GP probit classification.

This is the inference machinery the manuscript cites verbatim (Rasmussen &
Williams 2006, Algorithm 3.1 for the posterior mode and Algorithm 3.2 for
predictions), specialized to the probit likelihood of Definition 1. Keeping it
dependency-light (numpy + scipy only) keeps the inference legible rather than
hidden inside a tensor framework. An optional BoTorch ``PairwiseGP`` cross-check lives in
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


class EvidenceSelectedGPC:
    """Laplace GP-probit with kernel hyperparameters chosen by type-II maximum
    likelihood: at checkpoints (every ``select_every`` new labels) every
    candidate kernel is fitted and the one with the largest Laplace log
    marginal likelihood (R&W 3.32) is kept until the next checkpoint.

    A grid rather than gradient ascent keeps the selection transparent and
    reproducible; the grids used are listed in ``run.py`` and the report.
    """

    def __init__(self, candidates, select_every: int = 64, **gp_kw):
        if not candidates:
            raise ValueError("need at least one candidate kernel")
        self.candidates = list(candidates)
        self.select_every = select_every
        self.gp_kw = gp_kw
        self.idx = 0
        self.model = LaplaceGPC(self.candidates[0], **gp_kw)
        self._n_at_select = None
        self.history: list[tuple[int, str, float]] = []

    @property
    def kernel(self):
        return self.candidates[self.idx]

    @property
    def log_marginal(self) -> float:
        return self.model.log_marginal

    def fit(self, P: Packed, y01: np.ndarray) -> "EvidenceSelectedGPC":
        n = int(np.asarray(y01).shape[0])
        if self._n_at_select is None or n - self._n_at_select >= self.select_every:
            best = None
            for i, k in enumerate(self.candidates):
                m = LaplaceGPC(k, **self.gp_kw).fit(P, y01)
                if best is None or m.log_marginal > best[0]:
                    best = (m.log_marginal, i, m)
            _, self.idx, self.model = best
            self._n_at_select = n
            self.history.append((n, self.kernel.describe(), float(best[0])))
        else:
            self.model = LaplaceGPC(self.kernel, **self.gp_kw).fit(P, y01)
        return self

    def predict(self, Q: Packed):
        return self.model.predict(Q)

    def predict_prob(self, Q: Packed) -> np.ndarray:
        return self.model.predict_prob(Q)


# Houlsby et al. (2011) closed-form BALD for the probit GP classifier, in bits:
#   I(x) ~= h(Phi(mu / sqrt(1 + var))) - C / sqrt(var + C^2) * exp(-mu^2 / (2 (var + C^2)))
# with h the binary entropy in bits and C^2 = pi ln 2 / 2. The first term is the
# total predictive entropy; the second is the expected aleatoric entropy. BALD is
# their difference: the epistemic part a label would resolve.
_BALD_C2 = np.pi * np.log(2.0) / 2.0


def bald(mu, var):
    mu = np.asarray(mu, dtype=float)
    var = np.asarray(var, dtype=float)
    p = np.clip(norm.cdf(mu / np.sqrt(1.0 + var)), 1e-12, 1 - 1e-12)
    h = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    aleatoric = np.sqrt(_BALD_C2 / (var + _BALD_C2)) * np.exp(-(mu**2) / (2 * (var + _BALD_C2)))
    return h - aleatoric
