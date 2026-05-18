"""Ground-truth latent risk-tolerance oracle and probit observation model.

This module *is* the manuscript's generative model (Definition 1, Sections
4-6). For an input ``x = (a, c)`` arriving at time ``t`` the supervisor has a
latent risk tolerance

    f*(x, t) = g0(a, c)            # static acceptability of the action
             + rho * trust(t)     # Section 6: trust accumulates over time
             + veto(a, t)         # three-way safety conjunction
             + ctx_offset(c)      # task context shifts tolerance

and approves with probability ``Phi(f*(x, t))`` (Definition 1's probit model).
Binary feedback is ``y ~ Bernoulli(Phi(f*))``.

Two design choices make the learning problem non-trivial and faithful to the
manuscript (and follow the advisor's litmus test: remove the conjunction and
the drift and a plain GP would fit ``f*`` in a handful of queries):

* ``trust(t)`` is a saturating accumulation with an *abrupt changepoint*
  (Section 6: "moving to a new codebase"). The kernel's ``k_time`` component
  is what lets the gateway track this; without it the drift is unlearnable.
* ``veto`` is a *three-way interaction* (irreversible action AND sensitive
  target AND low current trust). It is not any single feature the kernel sees;
  the gateway must accumulate correlated evidence to recover this region.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .data import IRREVERSIBLE, DecisionPoint

# Task-context offsets: the supervisor is a bit more permissive during
# security work, stricter during data migrations, etc. Makes k_ctx matter.
_TASK_OFFSET: dict[str, float] = {
    "feature_dev": 0.00,
    "bugfix": 0.05,
    "refactor": -0.05,
    "ops_maintenance": -0.10,
    "data_migration": -0.25,
    "security_hardening": 0.15,
    "exploration": 0.10,
}


@dataclass(frozen=True)
class OracleConfig:
    """All ground-truth parameters. Frozen so a run is fully reproducible."""

    a0: float = 1.4           # baseline lean toward approval
    w_risk: float = 3.4       # how strongly action risk lowers tolerance
    rho: float = 1.0          # weight on accumulated trust
    theta: float = 1.6        # trust ceiling
    kappa: float = 130.0      # trust accumulation timescale (in steps)
    veto: float = 3.0         # strength of the safety conjunction
    veto_sens: float = 0.65   # target-sensitivity threshold for the veto
    changepoint: int | None = 600  # step at which trust resets (Section 6)

    # Risk aggregation weights (sum ~= 1; r in roughly [0, 1]).
    w_rev: float = 0.28
    w_tsens: float = 0.32
    w_blast: float = 0.12
    w_arg: float = 0.16
    w_base: float = 0.12


def raw_risk(dp: DecisionPoint, cfg: OracleConfig = OracleConfig()) -> float:
    """Decision-time-knowable composite risk of the action in ``[0, ~1]``."""
    return (
        cfg.w_rev * (dp.tool.reversibility / 2.0)
        + cfg.w_tsens * dp.target_sens
        + cfg.w_blast * (dp.tool.blast / 3.0)
        + cfg.w_arg * dp.arg_risk
        + cfg.w_base * dp.tool.base_sensitivity
    )


def trust(t: int, cfg: OracleConfig = OracleConfig()) -> float:
    """Saturating trust accumulation with an abrupt Section 6 changepoint.

    Before the changepoint trust grows as ``theta * (1 - exp(-t / kappa))``.
    At the changepoint it resets (the supervisor moves to an unfamiliar
    codebase and becomes cautious again) and re-accumulates from there.
    """
    t_reset = 0
    if cfg.changepoint is not None and t >= cfg.changepoint:
        t_reset = cfg.changepoint
    return cfg.theta * (1.0 - np.exp(-(t - t_reset) / cfg.kappa))


def _veto_active(dp: DecisionPoint, cfg: OracleConfig) -> bool:
    return (
        dp.tool.reversibility == IRREVERSIBLE
        and dp.target_sens >= cfg.veto_sens
        and trust(dp.t, cfg) < cfg.theta / 2.0
    )


def latent_f(dp: DecisionPoint, cfg: OracleConfig = OracleConfig()) -> float:
    """The ground-truth latent risk tolerance ``f*(x, t)``."""
    g0 = cfg.a0 - cfg.w_risk * raw_risk(dp, cfg)
    f = g0 + cfg.rho * trust(dp.t, cfg) + _TASK_OFFSET[dp.task]
    if _veto_active(dp, cfg):
        f -= cfg.veto
    return f


def approve_prob(dp: DecisionPoint, cfg: OracleConfig = OracleConfig()) -> float:
    """Probit approval probability ``Phi(f*(x, t))`` (Definition 1)."""
    return float(norm.cdf(latent_f(dp, cfg)))


def sample_label(
    dp: DecisionPoint, rng: np.random.Generator, cfg: OracleConfig = OracleConfig()
) -> int:
    """Draw a single approve/deny observation ``y ~ Bernoulli(Phi(f*))``."""
    return int(rng.random() < approve_prob(dp, cfg))


def oracle_decision(
    dp: DecisionPoint, cfg: OracleConfig = OracleConfig()
) -> int:
    """The supervisor's *expected* decision (1 approve / 0 deny), i.e. the
    Bayes-optimal label ``1[Phi(f*) >= 0.5]`` used to score the gateway."""
    return int(approve_prob(dp, cfg) >= 0.5)
