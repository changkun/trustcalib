"""Optional BoTorch ``PairwiseGP`` comparison (not part of the pipeline).

The unary approve/deny model is sometimes described as a degenerate pairwise
comparison against a fixed reference. It is not (Proposition 1): pairwise
comparisons identify the latent function only up to a constant, so they cannot
locate the approve/deny threshold, which is exactly the information unary
feedback carries.

This module keeps that pairwise encoding for readers who want to compare
against a maintained preference-learning implementation. Each observation becomes a
comparison between the action ``x`` and a reference point ``r``:

* ``y = 1`` (approve)  ->  ``x`` preferred over ``r``
* ``y = 0`` (deny)     ->  ``r`` preferred over ``x``

fitted with BoTorch's ``PairwiseGP`` (the Chu & Ghahramani 2005 probit
comparison likelihood with a Laplace posterior). Note that ``r`` is a point in
feature space whose latent value is itself random and kernel-correlated with
nearby actions, not the fixed constant of the unary model, so this is a
different model: agreement on the trend, not on numbers, is all it can show.
It uses BoTorch's default ARD-RBF covariance over the concatenated features.
Requires the optional ``botorch`` dependency group; importing without it
raises a clear error.
"""

from __future__ import annotations

import numpy as np

from .kernel import Packed

try:  # pragma: no cover - exercised only when the extra is installed
    import torch
    from botorch.fit import fit_gpytorch_mll
    from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood

    _HAVE_BOTORCH = True
except Exception:  # pragma: no cover
    _HAVE_BOTORCH = False


def botorch_available() -> bool:
    return _HAVE_BOTORCH


def _features(P: Packed, t_scale: float) -> np.ndarray:
    """Concatenate the kernel blocks into one vector; scale time so it is
    comparable to the (order-1) feature blocks for the default ARD kernel."""
    return np.concatenate(
        [P.phi_tool, P.phi_ctx, (P.t / t_scale)[:, None]], axis=1
    )


class PairwiseGPC:
    """Unary-as-pairwise BoTorch cross-check, same interface as LaplaceGPC."""

    def __init__(self, t_scale: float = 1000.0):
        if not _HAVE_BOTORCH:
            raise ImportError(
                "BoTorch is not installed. Install the optional cross-check "
                "dependencies:  uv sync --extra botorch"
            )
        self.t_scale = t_scale
        self._fitted = False

    def fit(self, P: Packed, y01: np.ndarray) -> "PairwiseGPC":
        X = _features(P, self.t_scale).astype(np.float64)
        y = np.asarray(y01).astype(int)
        n = X.shape[0]
        # Fixed reference point: the all-zero feature vector (the implicit
        # acceptable-risk boundary of Remark 1).
        ref = np.zeros((1, X.shape[1]), dtype=np.float64)
        datapoints = torch.tensor(np.vstack([X, ref]), dtype=torch.float64)
        ref_idx = n
        comps = []
        for i in range(n):
            if y[i] == 1:
                comps.append([i, ref_idx])      # x_i preferred over ref
            else:
                comps.append([ref_idx, i])      # ref preferred over x_i
        comparisons = torch.tensor(comps, dtype=torch.long)

        self._model = PairwiseGP(datapoints, comparisons)
        mll = PairwiseLaplaceMarginalLogLikelihood(
            self._model.likelihood, self._model
        )
        fit_gpytorch_mll(mll)
        self._ref = torch.tensor(ref, dtype=torch.float64)
        self._fitted = True
        return self

    def predict_prob(self, Q: Packed) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("call fit() before predict_prob()")
        Xq = torch.tensor(
            _features(Q, self.t_scale).astype(np.float64), dtype=torch.float64
        )
        with torch.no_grad():
            both = torch.cat([Xq, self._ref], dim=0)
            post = self._model.posterior(both)
            mean = post.mean.squeeze(-1)
            var = post.variance.squeeze(-1)
        m = mean.numpy()
        v = var.numpy()
        m_ref, v_ref = m[-1], v[-1]
        gap = m[:-1] - m_ref
        sd = np.sqrt(np.maximum(v[:-1] + v_ref, 1e-9))
        from scipy.stats import norm

        return norm.cdf(gap / sd)
