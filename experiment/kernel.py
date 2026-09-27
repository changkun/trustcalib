"""Structured kernels over (action, context, time) (manuscript Section 4).

Two kernels share the same feature blocks:

* :class:`ProductKernel` (the v1 kernel, kept as the reference)

      k = sigma^2 * k_tool * k_ctx * k_time

  Every component is multiplied by ``k_time``, so *all* evidence, including
  the static risk structure of an action, is forgotten at rate ``1/lam``
  (manuscript Proposition 4).

* :class:`AdditiveKernel` (the v2 kernel)

      k = s_static * k_tool * k_ctx            static action risk r(x)
        + s_global * k_time                    shared tolerance tau(t)
        + s_inter  * k_tool * k_ctx * k_time   local drift

  mirroring the decomposition ``f(x, t) = tau(t) - r(x)`` (manuscript
  Section 3). Only the time-coupled components forget; the static part does
  not, and every label updates the shared tolerance ``tau(t)``.

Block kernels:

* ``k_tool``  RBF over the tool feature block (reversibility, base
  sensitivity, blast radius, category one-hot).
* ``k_ctx``   RBF over the context block (target sensitivity,
  destructive-argument flag, task one-hot).
* ``k_time``  ``exp(-|t - t'| / lam)``, the Ornstein-Uhlenbeck covariance.

All blocks are PSD and have unit self-similarity; sums and (Schur) products of
PSD kernels are PSD, so both kernels are valid covariance functions with
prior variance equal to the sum of their component scales.

:class:`LinearKernel` (a Bayesian linear probit plus a shared time-drift term)
is used only as a baseline: the oracle's static term is linear in exactly the
features the kernels observe.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .data import TOOL_INDEX, DecisionPoint


@dataclass
class Packed:
    """Stream packed into dense arrays for vectorized kernel evaluation.

    ``tool_id``/``cell_id`` are carried for the independent baselines only;
    the kernels never use them.
    """

    phi_tool: np.ndarray  # (N, d_tool)
    phi_ctx: np.ndarray   # (N, d_ctx)
    t: np.ndarray         # (N,)
    tool_id: np.ndarray   # (N,)
    cell_id: np.ndarray | None = None  # (N,) (tool, target, task) cell index


def pack(stream: list[DecisionPoint]) -> Packed:
    return Packed(
        phi_tool=np.stack([dp.phi_tool for dp in stream]),
        phi_ctx=np.stack([dp.phi_ctx for dp in stream]),
        t=np.array([dp.t for dp in stream], dtype=float),
        tool_id=np.array([TOOL_INDEX[dp.tool.name] for dp in stream]),
        cell_id=np.array([f"{dp.tool.name}|{dp.target}|{dp.task}" for dp in stream]),
    )


def _sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean distance between rows of A and B."""
    a2 = np.sum(A * A, axis=1)[:, None]
    b2 = np.sum(B * B, axis=1)[None, :]
    d2 = a2 + b2 - 2.0 * A @ B.T
    return np.maximum(d2, 0.0)


def _k_x(P: Packed, Q: Packed, l_tool: float, l_ctx: float) -> np.ndarray:
    k_tool = np.exp(-_sqdist(P.phi_tool, Q.phi_tool) / (2.0 * l_tool**2))
    k_ctx = np.exp(-_sqdist(P.phi_ctx, Q.phi_ctx) / (2.0 * l_ctx**2))
    return k_tool * k_ctx


def _k_time(P: Packed, Q: Packed, lam: float) -> np.ndarray:
    return np.exp(-np.abs(P.t[:, None] - Q.t[None, :]) / lam)


class _KernelBase:
    def full(self, P: Packed) -> np.ndarray:
        """Train-train covariance ``K``."""
        return self._k(P, P)

    def cross(self, P: Packed, Q: Packed) -> np.ndarray:
        """Train-test cross covariance ``K_*`` (rows P, cols Q)."""
        return self._k(P, Q)

    def diag(self, P: Packed) -> np.ndarray:
        """Prior variances ``diag(K)`` (every block has unit self-similarity)."""
        return np.full(P.phi_tool.shape[0], self.prior_var, dtype=float)

    def with_(self, **kw):
        """A copy with some hyperparameters replaced."""
        return replace(self, **kw)


@dataclass
class ProductKernel(_KernelBase):
    """v1 separable kernel ``sigma2 * k_tool * k_ctx * k_time``."""

    sigma2: float = 1.6
    l_tool: float = 1.1
    l_ctx: float = 1.2
    lam: float = 200.0  # time lengthscale (steps)

    @property
    def prior_var(self) -> float:
        return self.sigma2

    def _k(self, P: Packed, Q: Packed) -> np.ndarray:
        return self.sigma2 * _k_x(P, Q, self.l_tool, self.l_ctx) * _k_time(P, Q, self.lam)

    # Backwards-compatible name used by the Go fixture generator.
    def _blocks(self, P: Packed, Q: Packed) -> np.ndarray:
        return self._k(P, Q)

    def describe(self) -> str:
        return f"product(σ²={self.sigma2:g}, λ={self.lam:g})"


@dataclass
class AdditiveKernel(_KernelBase):
    """v2 kernel ``s_static k_x + s_global k_time + s_inter k_x k_time``."""

    s_static: float = 1.6
    s_global: float = 1.0
    s_inter: float = 0.6
    l_tool: float = 1.1
    l_ctx: float = 1.2
    lam: float = 90.0

    @property
    def prior_var(self) -> float:
        return self.s_static + self.s_global + self.s_inter

    def _k(self, P: Packed, Q: Packed) -> np.ndarray:
        kx = _k_x(P, Q, self.l_tool, self.l_ctx)
        kt = _k_time(P, Q, self.lam)
        return self.s_static * kx + self.s_global * kt + self.s_inter * kx * kt

    def describe(self) -> str:
        return (f"additive(σs²={self.s_static:g}, σg²={self.s_global:g}, "
                f"σd²={self.s_inter:g}, λ={self.lam:g})")


@dataclass
class LinearKernel(_KernelBase):
    """Baseline: Bayesian linear probit on the features plus a shared OU drift,
    ``s_bias + s_w <phi, phi'> + s_global k_time``. It can represent the
    oracle's linear static term, task offsets and global trust drift, but not
    the three-way veto interaction."""

    s_bias: float = 1.0
    s_w: float = 1.0
    s_global: float = 1.0
    lam: float = 90.0

    @property
    def prior_var(self) -> float:  # not constant; see diag()
        return float("nan")

    def _k(self, P: Packed, Q: Packed) -> np.ndarray:
        lin = P.phi_tool @ Q.phi_tool.T + P.phi_ctx @ Q.phi_ctx.T
        return self.s_bias + self.s_w * lin + self.s_global * _k_time(P, Q, self.lam)

    def diag(self, P: Packed) -> np.ndarray:
        lin = np.sum(P.phi_tool**2, axis=1) + np.sum(P.phi_ctx**2, axis=1)
        return self.s_bias + self.s_w * lin + self.s_global

    def describe(self) -> str:
        return f"linear+drift(λ={self.lam:g})"
