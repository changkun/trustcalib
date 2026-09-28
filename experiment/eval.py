"""Baselines and metrics tied to the manuscript's claims.

Baselines:

* **per-tool** (:class:`IndependentModel`): one Beta-Bernoulli
  approval estimate per tool; ignores target and task.
* **per-cell** (:class:`CellModel`): one Beta-Bernoulli estimate per
  ``(tool, target, task)`` cell; the "contextual bandit that treats each
  (a, c) independently" (manuscript: correlated generalization). Unseen
  cells stay at 0.5.
* **linear probit + drift**: a Laplace GP with :class:`kernel.LinearKernel`,
  i.e. Bayesian probit regression on the same features plus a shared OU
  time-drift term. The oracle's static term is linear in exactly these
  features, so this tests whether the non-parametric kernel is needed.
* **random-query** (acquisition probe): same model, labels spent uniformly.

Metrics map onto manuscript claims; see ``report.md``.
"""

from __future__ import annotations

import numpy as np

from .data import TOOLS, DecisionPoint
from .gateway import ALLOW, ASK, BLOCK, Costs, GatewayResult
from .kernel import Packed, pack
from .oracle import OracleConfig, oracle_decision

_N_TOOLS = len(TOOLS)
SCORED = ("early", "late")


class IndependentModel:
    """Per-tool Beta-Bernoulli approval estimate (no transfer across tools,
    targets or tasks). An unseen tool stays at 0.5 and therefore in ASK."""

    def __init__(self):
        self.alpha = np.ones(_N_TOOLS)
        self.beta = np.ones(_N_TOOLS)

    def fit(self, P: Packed, y01: np.ndarray) -> "IndependentModel":
        self.alpha = np.ones(_N_TOOLS)
        self.beta = np.ones(_N_TOOLS)
        ids = np.asarray(P.tool_id, dtype=int)
        y = np.asarray(y01)
        np.add.at(self.alpha, ids, (y == 1).astype(float))
        np.add.at(self.beta, ids, (y == 0).astype(float))
        return self

    def predict_prob(self, Q: Packed) -> np.ndarray:
        ids = np.asarray(Q.tool_id, dtype=int)
        return self.alpha[ids] / (self.alpha[ids] + self.beta[ids])


class CellModel:
    """Per-(tool, target, task) Beta-Bernoulli approval estimate."""

    def __init__(self):
        self.counts: dict[str, list[float]] = {}

    def fit(self, P: Packed, y01: np.ndarray) -> "CellModel":
        self.counts = {}
        for c, y in zip(P.cell_id, np.asarray(y01)):
            a = self.counts.setdefault(str(c), [1.0, 1.0])
            a[0 if y == 1 else 1] += 1.0
        return self

    def predict_prob(self, Q: Packed) -> np.ndarray:
        out = np.empty(len(Q.cell_id))
        for i, c in enumerate(Q.cell_id):
            a, b = self.counts.get(str(c), (1.0, 1.0))
            out[i] = a / (a + b)
        return out


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _safe_div(a: float, b: float) -> float:
    return a / b if b else float("nan")


def _ece(p: np.ndarray, p_true: np.ndarray, n_bins: int = 10):
    """Calibration error of predicted vs ground-truth probability and the
    per-bin ``(mean_pred, mean_true, frac)`` reliability curve."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p > lo) & (p <= hi) if i > 0 else (p >= lo) & (p <= hi)
        if not np.any(m):
            bins.append((np.nan, np.nan, 0.0))
            continue
        mean_pred = float(np.mean(p[m]))
        mean_true = float(np.mean(p_true[m]))
        frac = float(np.mean(m))
        ece += frac * abs(mean_true - mean_pred)
        bins.append((mean_pred, mean_true, frac))
    return float(ece), bins


def phase_metrics(res: GatewayResult, phases=SCORED, costs: Costs | None = None) -> dict:
    """Headline metrics over the given phases of a run.

    * ``false_allow_rate``: fraction of ALLOWs the oracle would deny
      (Bayes label ``q < 1/2``);
    * ``realized_fa``: expected fraction of ALLOWs the stochastic human would
      deny, ``mean(1 - q)`` over ALLOWs; this is what an audit estimates;
    * ``regret``: mean Chow loss of the gateway's decisions minus that of the
      oracle policy that knows ``q`` (both in units of ``c_ask``).
    """
    steps = res.phase_steps(*phases)
    n = len(steps)
    if n == 0:
        return {}
    p = np.array([s.p_hat for s in steps])
    q = np.array([s.p_true for s in steps])
    dec = np.array([s.decision for s in steps])
    auto = [s for s in steps if s.correct_auto is not None]
    allow = dec == ALLOW
    out = {
        "n": n,
        "auto_rate": len(auto) / n,
        "ask_rate": float(np.mean(dec == ASK)),
        "allow_rate": float(np.mean(allow)),
        "accuracy_auto": _safe_div(sum(s.correct_auto for s in auto), len(auto)),
        "false_allow_rate": _safe_div(float(np.sum(allow & (q < 0.5))), float(np.sum(allow))),
        "realized_fa": float(np.mean(1 - q[allow])) if allow.any() else float("nan"),
        "prob_rmse": float(np.sqrt(np.mean((p - q) ** 2))),
        "queries": int(sum(s.queried for s in steps)),
    }
    out["ece"], out["_reliability"] = _ece(p, q)
    out["audit_frac"] = float(np.mean([s.audited for s in steps]))
    if costs is not None:
        loss = np.array([float(costs.loss(d, qq)) for d, qq in zip(dec, q)])
        out["regret"] = float(np.mean(loss - costs.bayes_loss(q)))
        # Regret charges decisions only, so an audit of an auto-decision is
        # free in ``regret``; ``regret_audit_charged`` bills each audit as one
        # escalation (an upper bound on what a post-hoc review costs).
        out["regret_audit_charged"] = out["regret"] + costs.c_ask * out["audit_frac"]
        out["floor_ask"] = float(np.mean((q > costs.tau_low) & (q < costs.tau_high)))
    return out


def boundary_accuracy(res: GatewayResult, phases=SCORED, lo: float = 0.15,
                      hi: float = 0.85) -> float:
    """Prequential accuracy of ``p_hat >= 1/2`` on genuinely contestable
    actions (``q`` in ``[lo, hi]``); isolates learning quality from the
    easy-approve majority."""
    sel = [s for s in res.steps if s.phase in phases and lo <= s.p_true <= hi]
    if not sel:
        return float("nan")
    return float(np.mean([(s.p_hat >= 0.5) == s.oracle_yes for s in sel]))


def query_placement(res: GatewayResult, phases=SCORED) -> dict:
    """Where labels were spent: share on intrinsically ambiguous actions
    (``q`` in ``[0.35, 0.65]``), mean latent variance, mean ``|mu|``, mean
    BALD. Separates aleatoric from epistemic targeting."""
    qs = [s for s in res.steps if s.phase in phases and s.queried and np.isfinite(s.var)]
    if not qs:
        return {}
    q = np.array([s.p_true for s in qs])
    return {
        "n": len(qs),
        "ambiguous_share": float(np.mean((q >= 0.35) & (q <= 0.65))),
        "mean_var": float(np.mean([s.var for s in qs])),
        "mean_abs_mu": float(np.mean([abs(s.mu) for s in qs])),
        "mean_bald": float(np.mean([s.bald for s in qs])),
    }


def audit_estimate(res: GatewayResult, phases=SCORED) -> dict:
    """Inverse-propensity estimate of the realized false-allow rate from
    random audits, with a Beta(1/2, 1/2) (Jeffreys) 95% interval, against
    the truth ``mean(1 - q)`` over ALLOWs (Proposition 5)."""
    from scipy.stats import beta

    steps = res.phase_steps(*phases)
    allow = [s for s in steps if s.decision == ALLOW]
    aud = [s for s in allow if s.audited]
    if not allow or res.audit_rate <= 0:
        return {}
    k = sum(1 for s in aud if s.y == 0)
    n = len(aud)
    est = k / (res.audit_rate * len(allow))
    lo, hi = (beta.ppf([0.025, 0.975], k + 0.5, n - k + 0.5) if n else (np.nan, np.nan))
    return {
        "n_audits": n,
        "denied": k,
        "ipw_estimate": float(est),
        "jeffreys_lo": float(lo),
        "jeffreys_hi": float(hi),
        "truth": float(np.mean([1 - s.p_true for s in allow])),
    }


def policy_trajectory(res: GatewayResult, window: int = 60):
    """Rolling ALLOW/ASK/BLOCK fractions over the whole stream."""
    dec = [s.decision for s in res.steps]
    t = np.array([s.t for s in res.steps], dtype=float)
    out = {"t": [], "allow": [], "ask": [], "block": []}
    for i in range(len(dec)):
        seg = dec[max(0, i - window + 1): i + 1]
        out["t"].append(t[i])
        out["allow"].append(seg.count(ALLOW) / len(seg))
        out["ask"].append(seg.count(ASK) / len(seg))
        out["block"].append(seg.count(BLOCK) / len(seg))
    return {k: np.array(v) for k, v in out.items()}


def scored_queries(res: GatewayResult, phases=SCORED) -> tuple[int, int]:
    """(human labels spent, actions) over the same phases: the burden ratio's
    numerator and denominator must cover the same steps."""
    steps = res.phase_steps(*phases)
    return sum(1 for s in steps if s.queried), len(steps)


def total_queries(res: GatewayResult) -> int:
    return sum(1 for s in res.steps if s.queried)


def held_out_decisions(res: GatewayResult, stream: list[DecisionPoint], hold,
                       phases=SCORED) -> dict:
    """Prequential decisions on a never-labelled (held-out) combination."""
    sts = [s for s, d in zip(res.steps, stream) if hold(d) and s.phase in phases]
    n_deny = sum(1 for s in sts if s.oracle_yes == 0)
    return {
        "n": len(sts),
        "n_deny": n_deny,
        "false_allow": sum(1 for s in sts if s.oracle_yes == 0 and s.decision == ALLOW),
        "false_block": sum(1 for s in sts if s.oracle_yes == 1 and s.decision == BLOCK),
        "allow": sum(1 for s in sts if s.decision == ALLOW),
        "ask": sum(1 for s in sts if s.decision == ASK),
        "block": sum(1 for s in sts if s.decision == BLOCK),
        "correct_auto": sum(1 for s in sts if s.correct_auto == 1),
    }


def transfer_accuracy(model, held: list[DecisionPoint], oracle_cfg: OracleConfig) -> float:
    """Decision accuracy of a final model on a held-out combination."""
    if model is None or not held:
        return float("nan")
    p = model.predict_prob(pack(held))
    ytrue = np.array([oracle_decision(dp, oracle_cfg) for dp in held])
    return float(np.mean((p >= 0.5).astype(int) == ytrue))


def aggregate(dicts: list[dict], keys: list[str]) -> dict:
    """Mean and standard deviation across seeds for the given scalar keys."""
    out = {}
    for k in keys:
        vals = np.array([d[k] for d in dicts if k in d and np.isfinite(d.get(k, np.nan))],
                        dtype=float)
        out[k] = ((float(np.mean(vals)), float(np.std(vals))) if vals.size
                  else (float("nan"), float("nan")))
    return out
