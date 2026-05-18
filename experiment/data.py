"""Synthetic action/context space and decision-point stream.

The manuscript formalizes trust calibration for agentic tool use as a
preference-learning problem over an input ``x = (a, c)`` of proposed agent
action and execution context, with a latent human risk-tolerance function
``f`` observed through a probit approve/deny model and drifting over time
(manuscript Sections 2-6).

No public dataset carries the signal this formulation needs: a single
supervisor's per-action approve/deny decisions tracked longitudinally as
their risk tolerance drifts. AgentSec (Zenodo 18369965) records agent
provenance but has no human-feedback, risk or preference labels; R-Judge has
human safe/unsafe labels but is static and aggregated. We therefore build a
purpose-built generator that *is* the manuscript's generative model: a
realistic agent-tool taxonomy with interpretable risk attributes, a latent
``f*`` with Section 6 trust drift and an abrupt changepoint, and the probit
observation model. The ground-truth oracle lives in :mod:`experiment.oracle`;
this module owns only the action/context space and the stream sampler.

This is a controlled simulation study with a known ground-truth oracle, which
is standard methodology for Preferential Bayesian Optimization. The tool
taxonomy below is our own design informed by common agent tools; it is not
reused from any dataset's labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Reversibility classes: 0 reversible, 1 hard-to-reverse, 2 irreversible.
REVERSIBLE, HARD, IRREVERSIBLE = 0, 1, 2

# Tool taxonomy. Each tool carries decision-time-knowable risk attributes:
#   category, reversibility, base sensitivity, blast radius (0 file .. 3 external),
#   and the probability that a destructive argument pattern accompanies it.
@dataclass(frozen=True)
class Tool:
    name: str
    category: str
    reversibility: int
    base_sensitivity: float
    blast: int
    arg_risk_prob: float


TOOLS: list[Tool] = [
    # read-only / low risk
    Tool("read_file", "read", REVERSIBLE, 0.10, 0, 0.00),
    Tool("list_dir", "read", REVERSIBLE, 0.08, 1, 0.00),
    Tool("grep_search", "search", REVERSIBLE, 0.10, 2, 0.00),
    Tool("git_status", "vcs", REVERSIBLE, 0.05, 2, 0.00),
    Tool("run_tests", "exec", REVERSIBLE, 0.20, 2, 0.02),
    Tool("web_fetch", "network", REVERSIBLE, 0.30, 3, 0.05),
    # moderate / hard-to-reverse
    Tool("write_file", "write", HARD, 0.30, 0, 0.05),
    Tool("apply_patch", "write", HARD, 0.40, 1, 0.08),
    Tool("format_code", "write", HARD, 0.25, 2, 0.03),
    Tool("install_package", "exec", HARD, 0.50, 2, 0.15),
    Tool("git_commit", "vcs", HARD, 0.30, 2, 0.04),
    # high / irreversible
    Tool("delete_file", "write", IRREVERSIBLE, 0.60, 0, 0.30),
    Tool("git_push", "vcs", IRREVERSIBLE, 0.60, 3, 0.10),
    Tool("git_force_push", "vcs", IRREVERSIBLE, 0.85, 3, 0.60),
    Tool("execute_sql", "db", IRREVERSIBLE, 0.70, 2, 0.35),
    Tool("shell_exec", "exec", IRREVERSIBLE, 0.70, 2, 0.30),
    Tool("send_email", "network", IRREVERSIBLE, 0.50, 3, 0.10),
    Tool("deploy", "deploy", IRREVERSIBLE, 0.90, 3, 0.20),
]
TOOL_INDEX: dict[str, int] = {t.name: i for i, t in enumerate(TOOLS)}
CATEGORIES: list[str] = sorted({t.category for t in TOOLS})

# Target resources, ordered by sensitivity tier in [0, 1].
TARGETS: dict[str, float] = {
    "sandbox_tmp": 0.00,
    "workspace_tests": 0.20,
    "workspace_src": 0.40,
    "build_config": 0.55,
    "ci_config": 0.65,
    "secrets_env": 0.90,
    "production_db": 0.95,
    "prod_infra": 1.00,
}
TARGET_NAMES: list[str] = list(TARGETS)

# Execution-context task categories.
TASKS: list[str] = [
    "feature_dev",
    "bugfix",
    "refactor",
    "ops_maintenance",
    "data_migration",
    "security_hardening",
    "exploration",
]

# Which targets are plausible for each tool category (keeps the joint
# distribution realistic: you do not git_push to sandbox_tmp).
_CATEGORY_TARGETS: dict[str, list[str]] = {
    "read": ["sandbox_tmp", "workspace_tests", "workspace_src", "build_config"],
    "search": ["workspace_tests", "workspace_src", "build_config", "ci_config"],
    "vcs": ["workspace_src", "build_config", "ci_config", "prod_infra"],
    "exec": ["sandbox_tmp", "workspace_tests", "workspace_src", "ci_config"],
    "write": ["sandbox_tmp", "workspace_tests", "workspace_src", "build_config",
              "ci_config"],
    "db": ["workspace_tests", "production_db"],
    "network": ["sandbox_tmp", "workspace_src", "prod_infra"],
    "deploy": ["ci_config", "prod_infra"],
}


@dataclass
class DecisionPoint:
    """One ``x_t = (a_t, c_t)`` arriving at global time index ``t``."""

    t: int
    tool: Tool
    target: str
    task: str
    arg_risk: int  # destructive argument pattern present

    target_sens: float = 0.0
    phi_tool: np.ndarray = field(default_factory=lambda: np.zeros(0))
    phi_ctx: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def featurize(self) -> "DecisionPoint":
        """Build the kernel feature blocks.

        ``phi_tool`` is a coarse semantic descriptor (reversibility, base
        sensitivity, blast radius, category one-hot). It deliberately omits a
        per-tool one-hot: a one-hot would put every distinct tool at the same
        fixed distance regardless of similarity, defeating the correlated
        generalization of Section 7. With the semantic descriptor, similar
        tools (same category, similar reversibility/blast) are genuinely close
        and share evidence; tool identity is still available to the
        independent baseline via ``Packed.tool_id``. ``phi_ctx`` holds the
        observable context: target sensitivity, the destructive-argument flag,
        and a task one-hot. The kernel never sees the oracle's time-varying
        veto conjunction (Section 6 drift x a three-way interaction), so the
        gateway must learn that region from labelled evidence through the
        kernel and ``k_time``.
        """
        cat = np.zeros(len(CATEGORIES))
        cat[CATEGORIES.index(self.tool.category)] = 1.0
        self.phi_tool = np.concatenate(
            [
                np.array(
                    [
                        self.tool.reversibility / 2.0,
                        self.tool.base_sensitivity,
                        self.tool.blast / 3.0,
                    ]
                ),
                cat,
            ]
        )
        task_oh = np.zeros(len(TASKS))
        task_oh[TASKS.index(self.task)] = 1.0
        self.phi_ctx = np.concatenate(
            [np.array([self.target_sens, float(self.arg_risk)]), task_oh]
        )
        return self


# Task -> tool-category propensity, so context correlates with action.
_TASK_CATEGORY_WEIGHTS: dict[str, dict[str, float]] = {
    "feature_dev": {"write": 3, "read": 3, "search": 2, "exec": 2, "vcs": 2},
    "bugfix": {"read": 3, "search": 3, "write": 2, "exec": 2, "vcs": 1},
    "refactor": {"write": 3, "read": 2, "search": 2, "vcs": 2},
    "ops_maintenance": {"exec": 3, "deploy": 2, "vcs": 2, "network": 2,
                         "read": 1},
    "data_migration": {"db": 4, "exec": 2, "write": 1, "read": 1},
    "security_hardening": {"search": 3, "read": 3, "write": 1, "vcs": 1},
    "exploration": {"read": 4, "search": 3, "vcs": 1},
}


def make_stream(n: int, seed: int) -> list[DecisionPoint]:
    """Sample ``n`` decision points with a realistic joint distribution.

    Time index ``t`` runs ``0..n-1`` and feeds both ``k_time`` and the
    oracle's drift. Unlike a fixed corpus, the generator produces a fresh,
    arbitrarily long, diverse stream, so the longitudinal non-stationary
    structure the manuscript's Section 6 needs is built in by construction.
    """
    rng = np.random.default_rng(seed)
    tools_by_cat: dict[str, list[Tool]] = {c: [] for c in CATEGORIES}
    for tl in TOOLS:
        tools_by_cat[tl.category].append(tl)

    stream: list[DecisionPoint] = []
    for t in range(n):
        task = TASKS[rng.integers(len(TASKS))]
        weights = _TASK_CATEGORY_WEIGHTS[task]
        cats = list(weights)
        probs = np.array([weights[c] for c in cats], dtype=float)
        probs /= probs.sum()
        cat = cats[rng.choice(len(cats), p=probs)]
        tool = tools_by_cat[cat][rng.integers(len(tools_by_cat[cat]))]
        cand_targets = _CATEGORY_TARGETS[cat]
        target = cand_targets[rng.integers(len(cand_targets))]
        arg_risk = int(rng.random() < tool.arg_risk_prob)
        dp = DecisionPoint(
            t=t,
            tool=tool,
            target=target,
            task=task,
            arg_risk=arg_risk,
            target_sens=TARGETS[target],
        )
        stream.append(dp.featurize())
    return stream
