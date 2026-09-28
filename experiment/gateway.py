"""Policy gateway: Chow three-tier decision rule, audits, and the online loop.

Manuscript: the decision rule. Given the posterior-predictive approval probability
``p_hat(x_*) = E[Phi(f(x_*))]`` the gateway emits

    ALLOW  if p_hat > tau_high
    BLOCK  if p_hat < tau_low
    ASK    otherwise

With costs ``c_FA`` (false allow), ``c_FB`` (false block) and ``c_ask`` (one
escalation) this rule is the Bayes-optimal one-step decision (Proposition 2,
machine-checked in ``lean/TrustCalib/Chow.lean``) with

    tau_high = 1 - c_ask / c_FA,    tau_low = c_ask / c_FB,

and the ASK band is non-empty iff ``c_ask (c_FA + c_FB) < c_FA c_FB``. The
symmetric band ``(0.35, 0.65)`` is this rule with ``c_FA = c_FB = c_ask / 0.35``.
Thresholds are therefore *specified*, never tuned on labels: tuning needs
labels on auto-decided actions, which no deployment has.

Labels. An ASK sends the action to the human, whose approve/deny becomes a
training label. ALLOW/BLOCK are auto-decided and produce no label, so their
errors are invisible (selective labels). With ``audit_rate = eps > 0`` each
auto-decided action is also shown to the human with probability ``eps``; the
audited labels train the model and give an unbiased inverse-propensity estimate
of the false-allow rate (Proposition 5, ``lean/TrustCalib/Audit.lean``).

Query strategies (the acquisition experiment) differ only in *which*
actions get a human label, at a matched budget:

* ``escalate``  label iff the decision is ASK (plus audits), the operational
                gateway;
* ``random``    label each action with probability ``query_rate``;
* ``bald``      label iff the BALD score (epistemic information, Houlsby et
                al. 2011) is in the top ``query_rate`` fraction of a sliding
                window of recent scores (a stream-based selective sampler).

The stream is processed prequentially: every decision is made, and logged,
before any label at that step exists. Phases: ``learn`` (warm-up, not scored),
``early`` (contains the trust changepoint) and ``late``.
"""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .data import DecisionPoint
from .gp import bald as bald_score
from .kernel import pack
from .oracle import OracleConfig, approve_prob, oracle_decision, sample_label

ALLOW, ASK, BLOCK = "allow", "ask", "block"


@dataclass(frozen=True)
class Costs:
    """Chow reject-option costs (all strictly positive)."""

    c_fa: float
    c_fb: float
    c_ask: float = 1.0

    def __post_init__(self):
        if min(self.c_fa, self.c_fb, self.c_ask) <= 0:
            raise ValueError("costs must be strictly positive")

    @property
    def tau_high(self) -> float:
        return 1.0 - self.c_ask / self.c_fa

    @property
    def tau_low(self) -> float:
        return self.c_ask / self.c_fb

    @property
    def band(self) -> tuple[float, float]:
        return (self.tau_low, self.tau_high)

    def band_nonempty(self) -> bool:
        return self.c_ask * (self.c_fa + self.c_fb) < self.c_fa * self.c_fb

    def loss(self, decision: str, q) -> np.ndarray:
        """Expected loss of ``decision`` when the approval probability is ``q``."""
        q = np.asarray(q, dtype=float)
        if decision == ALLOW:
            return (1.0 - q) * self.c_fa
        if decision == BLOCK:
            return q * self.c_fb
        return np.full_like(q, self.c_ask)

    def bayes_loss(self, q) -> np.ndarray:
        """Loss of the oracle policy that knows ``q`` exactly."""
        q = np.asarray(q, dtype=float)
        return np.minimum.reduce([self.loss(ALLOW, q), self.loss(BLOCK, q),
                                  self.loss(ASK, q)])

    @staticmethod
    def symmetric_from_band(tau_low: float) -> "Costs":
        """Symmetric costs whose Chow band is ``(tau_low, 1 - tau_low)``."""
        return Costs(c_fa=1.0 / tau_low, c_fb=1.0 / tau_low, c_ask=1.0)


# The two operating points used in the paper.
SYMMETRIC = Costs.symmetric_from_band(0.35)          # band (0.35, 0.65)
SAFETY = Costs(c_fa=10.0, c_fb=4.0, c_ask=1.0)       # band (0.25, 0.90)


@dataclass
class StepLog:
    t: int
    phase: str
    p_hat: float
    mu: float                # latent posterior mean at decision time
    var: float               # latent posterior variance at decision time
    bald: float              # epistemic information of a label (bits)
    p_true: float            # ground-truth approval prob Phi(f*)
    decision: str
    queried: bool            # a human label was collected at this step
    audited: bool            # ... because of a random audit of an auto-decision
    y: int | None            # the human's (sampled) label if queried
    oracle_yes: int          # Bayes-optimal supervisor decision 1[q >= 1/2]
    correct_auto: int | None  # 1/0 if auto-decided and matches oracle, else None


@dataclass
class GatewayResult:
    steps: list[StepLog] = field(default_factory=list)
    band: tuple[float, float] = (0.35, 0.65)
    audit_rate: float = 0.0
    n_train_final: int = 0
    final_model: object = None
    probe_trace: list[tuple[int, float, float]] = field(default_factory=list)

    def phase_steps(self, *phases: str) -> list[StepLog]:
        return [s for s in self.steps if s.phase in phases]


def _phase(t: int, t_warm: int, t_late: int) -> str:
    if t < t_warm:
        return "learn"
    if t < t_late:
        return "early"
    return "late"


def decide(p: float, tau_low: float, tau_high: float) -> str:
    if p > tau_high:
        return ALLOW
    if p < tau_low:
        return BLOCK
    return ASK


def run_gateway(
    stream: list[DecisionPoint],
    model,
    rng: np.random.Generator,
    oracle_cfg: OracleConfig,
    t_warm: int = 560,
    t_late: int = 1050,
    *,
    band: tuple[float, float] = SYMMETRIC.band,
    query_strategy: str = "escalate",
    query_rate: float = 0.15,
    audit_rate: float = 0.0,
    refit_every: int = 8,
    bald_window: int = 100,
    hold_out=None,
    probe: DecisionPoint | None = None,
) -> GatewayResult:
    """Run the online gateway over ``stream``.

    ``model`` must expose ``fit(Packed, y01)`` and either ``predict(Packed)``
    returning ``(mu, var, p)`` or only ``predict_prob(Packed)``.
    ``hold_out(dp) -> bool`` forbids labelling matching actions (transfer
    tests: the gateway must decide them purely through the kernel). ``probe``
    records ``(t, p_hat, p_true)`` at every refit (drift figure).
    """
    tau_low, tau_high = band
    res = GatewayResult(band=band, audit_rate=audit_rate)
    train_pts: list[DecisionPoint] = []
    train_y: list[int] = []
    fitted = bool(getattr(model, "prefitted", False))  # fixed policies need no labels
    dirty = False
    recent_bald: deque[float] = deque(maxlen=bald_window)
    has_var = hasattr(model, "predict")

    def predict_one(dp: DecisionPoint) -> tuple[float, float, float]:
        if not fitted:
            return 0.5, 0.0, float("nan")  # cold start: prior -> ASK (fail-safe)
        if has_var:
            mu, var, p = model.predict(pack([dp]))
            return float(p[0]), float(mu[0]), float(var[0])
        return float(model.predict_prob(pack([dp]))[0]), float("nan"), float("nan")

    def record_probe(now_t: int) -> None:
        if probe is None or not fitted:
            return
        pr = copy.copy(probe)
        pr.t = now_t
        p_hat = float(model.predict_prob(pack([pr]))[0])
        res.probe_trace.append((now_t, p_hat, approve_prob(pr, oracle_cfg)))

    for i, dp in enumerate(stream):
        phase = _phase(dp.t, t_warm, t_late)
        p, mu, var = predict_one(dp)
        b = float(bald_score(mu, var)) if np.isfinite(var) else 1.0
        decision = decide(p, tau_low, tau_high)
        o_yes = oracle_decision(dp, oracle_cfg)
        p_tru = approve_prob(dp, oracle_cfg)

        audited = False
        if query_strategy == "escalate":
            queried = decision == ASK
            if not queried and audit_rate > 0 and rng.random() < audit_rate:
                queried = audited = True
        elif query_strategy == "random":
            queried = bool(rng.random() < query_rate)
        elif query_strategy == "bald":
            if len(recent_bald) < 20:
                queried = decision == ASK
            else:
                queried = b >= np.quantile(np.asarray(recent_bald), 1.0 - query_rate)
            recent_bald.append(b)
        else:
            raise ValueError(f"unknown query_strategy {query_strategy!r}")

        if queried and hold_out is not None and hold_out(dp):
            queried = audited = False

        y = None
        if queried:
            y = sample_label(dp, rng, oracle_cfg)
            train_pts.append(dp)
            train_y.append(y)
            dirty = True

        correct_auto = None
        if decision in (ALLOW, BLOCK):
            correct_auto = int((decision == ALLOW) == bool(o_yes))

        res.steps.append(StepLog(
            t=dp.t, phase=phase, p_hat=p, mu=mu, var=var, bald=b, p_true=p_tru,
            decision=decision, queried=queried, audited=audited, y=y,
            oracle_yes=o_yes, correct_auto=correct_auto,
        ))

        if dirty and len(train_pts) >= 2 and (i % refit_every == 0 or not fitted):
            model.fit(pack(train_pts), np.array(train_y))
            fitted = True
            dirty = False
            record_probe(dp.t)

    res.n_train_final = len(train_pts)
    if fitted:
        res.final_model = copy.deepcopy(model)
    return res


def tune_thresholds(
    p_hat: np.ndarray,
    labels: np.ndarray,
    safety_eps: float = 0.02,
    block_eps: float = 0.05,
) -> tuple[float, float]:
    """Grid-search threshold tuning on labels, kept only as the reference for
    the Go ``gateway.TuneThresholds``; not used by the pipeline.

    Two problems make it unsuitable (manuscript: the decision rule): run with
    labels for every validation action it needs labels on auto-decided
    actions, which no deployment has; and fed only the escalated history a
    deployment actually has, every ``p_hat`` lies inside the current band, no
    candidate pair meets the caps, and it returns the default band.
    """
    safety_eps_val = max(safety_eps / 2.0, 1e-3)
    grid = np.linspace(0.05, 0.95, 19)
    best = None
    best_ask = 2.0
    for lo in grid:
        for hi in grid:
            if hi <= lo:
                continue
            allow = p_hat > hi
            block = p_hat < lo
            ask = ~(allow | block)
            ask_rate = float(ask.mean())
            if ask_rate > 0.6:
                continue
            n_allow = max(int(allow.sum()), 1)
            n_block = max(int(block.sum()), 1)
            false_allow = float(np.sum(allow & (labels == 0))) / n_allow
            false_block = float(np.sum(block & (labels == 1))) / n_block
            if false_allow > safety_eps_val or false_block > block_eps:
                continue
            if ask_rate < best_ask:
                best_ask = ask_rate
                best = (float(lo), float(hi))
    if best is None or best_ask > 0.7:
        return (0.35, 0.65)
    return best
