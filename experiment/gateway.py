"""Policy gateway: three-tier decision rule and the acquisition loop.

Manuscript Section 5. Given the posterior-predictive approval probability
``p_hat(x_*) = E[Phi(f(x_*))]`` the gateway emits

    ALLOW  if p_hat > tau_high
    BLOCK  if p_hat < tau_low
    ASK    otherwise

The ASK band is the *acquisition rule*: the human is queried exactly where the
approval outcome is most uncertain, and only those queried points enter the
training set (ALLOW/BLOCK are auto-decided at zero human cost). The stream is
processed online with a **prequential** train/val/test protocol suited to a
non-stationary stream (freezing the model at a cut point fights the Section 6
continual-adaptation premise and degenerates):

* learn ``[0, t1)``   online learning with fixed default thresholds;
* val   ``[t1, t2)``  keep learning, then tune ``(tau_low, tau_high)`` once;
* test  ``[t2, T)``   policy (thresholds) frozen; the model keeps learning
                       online; every decision is scored before any label at
                       that step exists, so there is no leakage.

The continuous online trajectory (ALLOW/ASK/BLOCK fractions over time) is also
recorded across the whole stream for the Section 5 / Section 6 figures.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from .data import DecisionPoint
from .kernel import pack
from .oracle import OracleConfig, approve_prob, oracle_decision, sample_label

ALLOW, ASK, BLOCK = "allow", "ask", "block"


@dataclass
class StepLog:
    t: int
    phase: str
    p_hat: float
    p_true: float            # ground-truth approval prob Phi(f*) (calibration)
    decision: str
    queried: bool
    oracle_yes: int          # Bayes-optimal supervisor decision (1/0)
    correct_auto: int | None  # 1/0 if auto-decided and matches oracle, else None


@dataclass
class GatewayResult:
    steps: list[StepLog] = field(default_factory=list)
    tau_low: float = 0.35
    tau_high: float = 0.65
    n_train_final: int = 0
    frozen_model: object = None  # final online model (transfer-test scoring only)
    probe_trace: list[tuple[int, float, float]] = field(default_factory=list)

    def phase_steps(self, phase: str) -> list[StepLog]:
        return [s for s in self.steps if s.phase == phase]


def _phase(t: int, t1: int, t2: int) -> str:
    if t < t1:
        return "learn"
    if t < t2:
        return "val"
    return "test"


def _decision(p: float, tau_low: float, tau_high: float) -> str:
    if p > tau_high:
        return ALLOW
    if p < tau_low:
        return BLOCK
    return ASK


def tune_thresholds(
    p_hat: np.ndarray,
    oracle_yes: np.ndarray,
    safety_eps: float = 0.02,
    block_eps: float = 0.05,
) -> tuple[float, float]:
    """Pick ``(tau_low, tau_high)`` on the validation phase.

    Among threshold pairs whose auto-decisions respect a safety cap
    (false-ALLOW rate <= ``safety_eps``), a usefulness cap (false-BLOCK rate
    <= ``block_eps``) and a minimum auto-coverage (auto rate >= 0.4), choose
    the one with the smallest ASK rate. This operationalizes "spend human
    queries only where they buy safety" while refusing the degenerate
    all-ASK band that a noisy validation set can otherwise make look safe.
    If no pair is feasible, or the best feasible band still escalates more
    than 70% of traffic, fall back to the sensible default ``(0.35, 0.65)``.

    The safety cap is tightened on validation (``safety_eps / 2``) to leave
    headroom for the distribution shift between val and the post-changepoint
    test phase; any residual test-phase overshoot is reported honestly as a
    non-stationarity stress result rather than tuned away.
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
            if ask_rate > 0.6:  # require auto-coverage >= 0.4
                continue
            n_allow = max(int(allow.sum()), 1)
            n_block = max(int(block.sum()), 1)
            false_allow = float(np.sum(allow & (oracle_yes == 0))) / n_allow
            false_block = float(np.sum(block & (oracle_yes == 1))) / n_block
            if false_allow > safety_eps_val or false_block > block_eps:
                continue
            if ask_rate < best_ask:
                best_ask = ask_rate
                best = (float(lo), float(hi))
    if best is None or best_ask > 0.7:
        return (0.35, 0.65)
    return best


def run_gateway(
    stream: list[DecisionPoint],
    model,
    rng: np.random.Generator,
    oracle_cfg: OracleConfig,
    t1: int,
    t2: int,
    *,
    query_strategy: str = "active",
    random_query_rate: float = 0.15,
    tau_low: float = 0.35,
    tau_high: float = 0.65,
    refit_every: int = 8,
    safety_eps: float = 0.02,
    hold_out=None,
    probe: DecisionPoint | None = None,
) -> GatewayResult:
    """Run the online gateway over ``stream``.

    ``model`` must expose ``fit(Packed, y01)`` and ``predict_prob(Packed)``.
    ``query_strategy``:

    * ``active``       query iff the decision is ASK (manuscript);
    * ``random``       query each step w.p. ``random_query_rate`` regardless
                       of uncertainty (passive-learning baseline).

    Both strategies use the same model and the same three-tier rule for
    *deciding*; they differ only in which points get a human label.

    ``hold_out(dp) -> bool`` forbids querying matching points (used by the
    Section 7 transfer test: the gateway must decide an unseen combination
    purely through kernel correlation). ``probe`` records ``(t, p_hat,
    p_true)`` at every refit for the Section 6 drift-tracking figure.
    """
    res = GatewayResult(tau_low=tau_low, tau_high=tau_high)
    train_pts: list[DecisionPoint] = []
    train_y: list[int] = []
    fitted = False
    dirty = False
    tuned = False
    cur_lo, cur_hi = tau_low, tau_high

    def record_probe(now_t: int) -> None:
        if probe is None or not fitted:
            return
        pr = copy.copy(probe)
        pr.t = now_t
        p_hat = float(model.predict_prob(pack([pr]))[0])
        p_true = approve_prob(pr, oracle_cfg)
        res.probe_trace.append((now_t, p_hat, p_true))

    for i, dp in enumerate(stream):
        phase = _phase(dp.t, t1, t2)

        # Prequential protocol: the decision at t is always made by the
        # current model BEFORE any label at t exists. The model keeps
        # adapting in every phase (Section 6 is continual adaptation); only
        # the *policy* (the tuned thresholds) is frozen after validation.
        if fitted:
            p = float(model.predict_prob(pack([dp]))[0])
        else:
            p = 0.5  # cold start: prior mean -> ASK (fail-safe, Section 7)

        decision = _decision(p, cur_lo, cur_hi)
        o_yes = oracle_decision(dp, oracle_cfg)
        p_tru = approve_prob(dp, oracle_cfg)

        # Decide whether to query the human for a label.
        if query_strategy == "active":
            queried = decision == ASK
        elif query_strategy == "random":
            queried = bool(rng.random() < random_query_rate)
        else:
            raise ValueError(f"unknown query_strategy {query_strategy!r}")

        if queried and hold_out is not None and hold_out(dp):
            queried = False  # Section 7 transfer test: never label this combo

        if queried:
            y = sample_label(dp, rng, oracle_cfg)
            train_pts.append(dp)
            train_y.append(y)
            dirty = True

        correct_auto = None
        if decision in (ALLOW, BLOCK):
            auto_yes = 1 if decision == ALLOW else 0
            correct_auto = int(auto_yes == o_yes)

        res.steps.append(
            StepLog(
                t=dp.t,
                phase=phase,
                p_hat=p,
                p_true=p_tru,
                decision=decision,
                queried=queried,
                oracle_yes=o_yes,
                correct_auto=correct_auto,
            )
        )

        # Periodic online refit, in every phase (continual adaptation).
        if dirty and len(train_pts) >= 2 and (
            i % refit_every == 0 or i == t2 - 1
        ):
            model.fit(pack(train_pts), np.array(train_y))
            fitted = True
            dirty = False
            record_probe(dp.t)

        # End of validation: tune the policy (thresholds) once. The model is
        # NOT frozen; it keeps learning online through the test phase.
        if not tuned and dp.t == t2 - 1:
            if dirty and len(train_pts) >= 2:
                model.fit(pack(train_pts), np.array(train_y))
                fitted = True
                dirty = False
            if fitted:
                val_logs = [s for s in res.steps if s.phase == "val"]
                if val_logs:
                    val_pts = [stream[j] for j, s in enumerate(res.steps)
                               if s.phase == "val"]
                    ph = model.predict_prob(pack(val_pts))
                    oy = np.array([s.oracle_yes for s in val_logs])
                    cur_lo, cur_hi = tune_thresholds(ph, oy, safety_eps)
            res.tau_low, res.tau_high = cur_lo, cur_hi
            tuned = True

    res.n_train_final = len(train_pts)
    # Final online model, used only to score the held-out transfer combo
    # (which `hold_out` guaranteed was never queried).
    if fitted:
        res.frozen_model = copy.deepcopy(model)
    return res
