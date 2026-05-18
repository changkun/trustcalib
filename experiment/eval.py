"""Baselines and metrics tied to the manuscript's claims.

Baselines (advisor's discriminating set):

* **random-query**: same GP, but queries are spent uniformly at random
  instead of in the ASK band. Isolates the value of uncertainty-targeted
  acquisition (manuscript Section 5: ASK band as acquisition function).
* **independent**: a per-tool running approval estimate with no kernel
  correlation. Isolates the value of correlated generalization
  (manuscript Section 7).

Metrics map one-to-one onto manuscript claims; see ``report.md``.
"""

from __future__ import annotations

import numpy as np

from .data import TOOLS, DecisionPoint
from .gateway import ALLOW, GatewayResult
from .kernel import Packed
from .oracle import OracleConfig, oracle_decision

_N_TOOLS = len(TOOLS)


class IndependentModel:
    """Per-tool Beta-Bernoulli approval estimate (no correlated generalization).

    Uses ``Packed.tool_id``. Each tool is learned in isolation: evidence
    never transfers between similar tools or targets, and an unseen tool stays
    at the ``0.5`` prior (so it lands in the ASK band forever). This is the
    naive contextual baseline the manuscript's Section 7 argues against.
    """

    def __init__(self):
        self.alpha = np.ones(_N_TOOLS)
        self.beta = np.ones(_N_TOOLS)

    @staticmethod
    def _tool_ids(P: Packed) -> np.ndarray:
        return np.asarray(P.tool_id, dtype=int)

    def fit(self, P: Packed, y01: np.ndarray) -> "IndependentModel":
        self.alpha = np.ones(_N_TOOLS)
        self.beta = np.ones(_N_TOOLS)
        ids = self._tool_ids(P)
        y = np.asarray(y01)
        for k in range(_N_TOOLS):
            m = ids == k
            self.alpha[k] += float(np.sum(y[m] == 1))
            self.beta[k] += float(np.sum(y[m] == 0))
        return self

    def predict_prob(self, Q: Packed) -> np.ndarray:
        ids = self._tool_ids(Q)
        return self.alpha[ids] / (self.alpha[ids] + self.beta[ids])


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _safe_div(a: float, b: float) -> float:
    return a / b if b else float("nan")


def phase_metrics(res: GatewayResult, phase: str = "test") -> dict:
    """Headline metrics for one phase of a run."""
    steps = res.phase_steps(phase)
    n = len(steps)
    if n == 0:
        return {}
    p = np.array([s.p_hat for s in steps])
    p_true = np.array([s.p_true for s in steps])

    auto = [s for s in steps if s.correct_auto is not None]
    allow = [s for s in steps if s.decision == ALLOW]
    ask = [s for s in steps if s.decision == "ask"]

    false_allow = _safe_div(
        sum(1 for s in allow if s.oracle_yes == 0), len(allow)
    )
    acc_auto = _safe_div(sum(s.correct_auto for s in auto), len(auto))

    # Calibration is measured against the ground-truth probability Phi(f*),
    # not the 0/1 Bayes label: with a known oracle, scoring a perfectly
    # calibrated p=0.93 against label 1 would spuriously inflate the error by
    # the irreducible Bayes noise. prob-RMSE is the probability-estimation
    # error; accuracy / false-allow above are the decision-side metrics.
    prob_rmse = float(np.sqrt(np.mean((p - p_true) ** 2)))
    ece, bins = _ece(p, p_true)

    return {
        "n": n,
        "auto_rate": len(auto) / n,
        "ask_rate": len(ask) / n,
        "accuracy_auto": acc_auto,
        "false_allow_rate": false_allow,
        "prob_rmse": prob_rmse,
        "ece": ece,
        "_reliability": bins,
    }


def _ece(p: np.ndarray, p_true: np.ndarray, n_bins: int = 10):
    """Calibration error of predicted vs ground-truth probability, and the
    per-bin ``(mean_pred, mean_true, frac)`` reliability curve. With a known
    oracle this is the correct calibration measurement for a simulation."""
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


def policy_trajectory(res: GatewayResult, window: int = 60):
    """Rolling ALLOW/ASK/BLOCK fractions over the whole stream (Section 5/6)."""
    dec = [s.decision for s in res.steps]
    t = np.array([s.t for s in res.steps], dtype=float)
    out = {"t": [], "allow": [], "ask": [], "block": []}
    for i in range(len(dec)):
        a = max(0, i - window + 1)
        seg = dec[a : i + 1]
        out["t"].append(t[i])
        out["allow"].append(seg.count(ALLOW) / len(seg))
        out["ask"].append(seg.count("ask") / len(seg))
        out["block"].append(seg.count("block") / len(seg))
    return {k: np.array(v) for k, v in out.items()}


def boundary_accuracy(
    res: GatewayResult,
    phases: tuple[str, ...] = ("val", "test"),
    lo: float = 0.15,
    hi: float = 0.85,
) -> float:
    """Prequential accuracy on the genuinely-contestable points (true
    approval probability in ``[lo, hi]``). Auto-decision accuracy is dominated
    by the easy-approve majority under class imbalance and does not isolate
    learning quality; this restricts to the decision boundary."""
    sel = [
        s
        for s in res.steps
        if s.phase in phases and lo <= s.p_true <= hi
    ]
    if not sel:
        return float("nan")
    return float(np.mean([(s.p_hat >= 0.5) == s.oracle_yes for s in sel]))


def always_escalate_metrics(
    res: GatewayResult, phases: tuple[str, ...] = ("val",)
) -> dict:
    """The status-quo baseline: every action is escalated to the human (no
    automation). Accuracy is 1.0 by definition (the human decides every
    case); the cost is one human query per action. Used as the headline
    comparison for the manuscript's Section 1 burden-reduction claim."""
    n = sum(1 for s in res.steps if s.phase in phases)
    return {
        "n": n,
        "auto_rate": 0.0,
        "ask_rate": 1.0,
        "accuracy_auto": float("nan"),  # nothing is auto-decided
        "human_queries": n,
    }


def total_queries(res: GatewayResult) -> int:
    return sum(1 for s in res.steps if s.queried)


def transfer_accuracy(
    model, held: list[DecisionPoint], oracle_cfg: OracleConfig
) -> float:
    """Decision accuracy on a held-out (tool, target) combination the gateway
    was never allowed to query (manuscript Section 7 correlated
    generalization). The model must extrapolate purely through the kernel."""
    if model is None or not held:
        return float("nan")
    from .kernel import pack

    p = model.predict_prob(pack(held))
    yhat = (p >= 0.5).astype(int)
    ytrue = np.array([oracle_decision(dp, oracle_cfg) for dp in held])
    return float(np.mean(yhat == ytrue))


def aggregate(dicts: list[dict], keys: list[str]) -> dict:
    """Mean +/- std across seeds for the given scalar keys."""
    out = {}
    for k in keys:
        vals = np.array(
            [d[k] for d in dicts if k in d and not np.isnan(d.get(k, np.nan))]
        )
        out[k] = (
            (float(np.mean(vals)), float(np.std(vals)))
            if vals.size
            else (float("nan"), float("nan"))
        )
    return out
