"""Structured product kernel (manuscript Section 4).

    k(x, x') = sigma^2 * k_tool(a, a') * k_ctx(c, c') * k_time(t, t')

* ``k_tool``  RBF over the tool feature block (one-hot name + semantic
  descriptor): shared tool name and shared reversibility/category raise
  similarity, exactly the manuscript's "shared tool name, overlapping
  argument patterns, same reversibility class".
* ``k_ctx``   RBF over the context block (target sensitivity, destructive-arg
  flag, task one-hot): "same repository, file type, task category".
* ``k_time``  ``exp(-|t - t'| / lambda)``, the manuscript's Section 6 kernel
  for non-stationarity. This is the Ornstein-Uhlenbeck covariance and is
  positive semidefinite; the product of PSD kernels is PSD (Schur product).

A single signal variance ``sigma^2`` multiplies the product (per-block scales
are not separately identifiable). Each block's self-similarity is 1, so the
prior variance on the diagonal is exactly ``sigma^2``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import TOOL_INDEX, DecisionPoint


@dataclass
class Packed:
    """Stream packed into dense arrays for vectorized kernel evaluation.

    ``tool_id`` is carried for the independent baseline only; the kernel never
    uses it (tool similarity comes from the semantic block of ``phi_tool``).
    """

    phi_tool: np.ndarray  # (N, d_tool)
    phi_ctx: np.ndarray   # (N, d_ctx)
    t: np.ndarray         # (N,)
    tool_id: np.ndarray   # (N,)


def pack(stream: list[DecisionPoint]) -> Packed:
    return Packed(
        phi_tool=np.stack([dp.phi_tool for dp in stream]),
        phi_ctx=np.stack([dp.phi_ctx for dp in stream]),
        t=np.array([dp.t for dp in stream], dtype=float),
        tool_id=np.array([TOOL_INDEX[dp.tool.name] for dp in stream]),
    )


def _sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean distance between rows of A and B."""
    a2 = np.sum(A * A, axis=1)[:, None]
    b2 = np.sum(B * B, axis=1)[None, :]
    d2 = a2 + b2 - 2.0 * A @ B.T
    return np.maximum(d2, 0.0)


@dataclass
class ProductKernel:
    """Manuscript Section 4 product kernel with structured blocks."""

    sigma2: float = 1.6
    l_tool: float = 1.1
    l_ctx: float = 1.2
    lam: float = 200.0  # time lengthscale (steps); ~ the drift timescale

    def _blocks(self, P: Packed, Q: Packed) -> np.ndarray:
        k_tool = np.exp(-_sqdist(P.phi_tool, Q.phi_tool) / (2.0 * self.l_tool**2))
        k_ctx = np.exp(-_sqdist(P.phi_ctx, Q.phi_ctx) / (2.0 * self.l_ctx**2))
        dt = np.abs(P.t[:, None] - Q.t[None, :])
        k_time = np.exp(-dt / self.lam)
        return self.sigma2 * k_tool * k_ctx * k_time

    def full(self, P: Packed) -> np.ndarray:
        """Train-train covariance ``K``."""
        return self._blocks(P, P)

    def cross(self, P: Packed, Q: Packed) -> np.ndarray:
        """Train-test cross covariance ``K_*`` (rows P, cols Q)."""
        return self._blocks(P, Q)

    def diag(self, P: Packed) -> np.ndarray:
        """Prior variances ``diag(K)`` (each block self-similarity is 1)."""
        return np.full(P.phi_tool.shape[0], self.sigma2, dtype=float)
