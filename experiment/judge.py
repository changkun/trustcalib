"""An opaque LLM judge in front of a general-purpose shell (extension study).

Pre-registered in ``judge_prereg.md``. When an agent routes everything
through one shell tool there is no fixed tool taxonomy, and a judge (an
auto-mode classifier) decides allow/block from a restricted view of the
action. We model that judge as seeing the action's *static* risk ``r(x)`` but
not the supervisor's current tolerance, task preferences or veto, and give it
a blind spot (destructive shell commands it perceives as benign).

The gateway that sits on top sees no taxonomy either: only the judge's output
(verdict or score), a coarse command category and time. Its job is to learn,
per supervisor and context, how far the judge's verdicts can be trusted.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from .data import CATEGORIES, DecisionPoint
from .kernel import Packed
from .oracle import raw_risk


@dataclass(frozen=True)
class JudgeConfig:
    kappa: float = 8.0           # sharpness of the judge's risk-to-score map
    bias: float = 0.5            # the judge's fixed risk threshold
    noise: float = 1.0           # logit noise (judge inconsistency)
    blind_category: str = "exec"  # blind spot: destructive shell commands ...
    blind_perceived: float = 0.2  # ... perceived as this (low) risk


def is_blind(dp: DecisionPoint, jc: JudgeConfig = JudgeConfig()) -> bool:
    return dp.tool.category == jc.blind_category and dp.arg_risk == 1


def judge_stream(stream: list[DecisionPoint], seed: int, mode: str = "verdict",
                 jc: JudgeConfig = JudgeConfig()) -> list[DecisionPoint]:
    """Copies of ``stream`` whose kernel features are what the harness sees.

    ``phi_tool`` = [judge output] + category one-hot, ``phi_ctx`` = [0]. The
    judge output is the binary verdict (1 = BLOCK) in ``verdict`` mode and the
    risk score in ``score`` mode. The oracle still reads the true action
    attributes, which the gateway never sees. Each copy carries
    ``judge_score``, ``judge_block`` and ``blind``.
    """
    if mode not in ("verdict", "score"):
        raise ValueError(mode)
    rng = np.random.default_rng(seed)
    out = []
    for dp in stream:
        r = jc.blind_perceived if is_blind(dp, jc) else raw_risk(dp)
        s = 1.0 / (1.0 + np.exp(-(jc.kappa * (r - jc.bias) + rng.normal(0.0, jc.noise))))
        d = copy.copy(dp)
        cat = np.zeros(len(CATEGORIES))
        cat[CATEGORIES.index(dp.tool.category)] = 1.0
        feat = s if mode == "score" else float(s > 0.5)
        d.phi_tool = np.concatenate([[feat], cat])
        d.phi_ctx = np.zeros(1)
        d.judge_score = float(s)
        d.judge_block = bool(s > 0.5)
        d.blind = is_blind(dp, jc)
        out.append(d)
    return out


class JudgePolicy:
    """Fixed policies driven by the judge alone (no learning).

    ``escalate=False``: ALLOW when the judge allows, BLOCK otherwise (the
    judge-only status quo). ``escalate=True``: ASK instead of BLOCK.
    ``always_ask=True``: escalate everything.
    """

    prefitted = True

    def __init__(self, escalate: bool = False, always_ask: bool = False):
        self.escalate = escalate
        self.always_ask = always_ask

    def fit(self, P: Packed, y01) -> "JudgePolicy":
        return self

    def predict_prob(self, Q: Packed) -> np.ndarray:
        if self.always_ask:
            return np.full(Q.phi_tool.shape[0], 0.5)
        block = Q.phi_tool[:, 0] > 0.5
        return np.where(block, 0.5 if self.escalate else 0.001, 0.999)
