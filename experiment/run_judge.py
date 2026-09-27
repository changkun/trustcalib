"""Extension study: calibrating an opaque judge (manuscript Section 12).

Pre-registered in ``experiment/judge_prereg.md``. Writes ``report_judge.md``,
``results_judge.json``, ``figures/judge_reliability.pdf`` and
``manuscript/generated/judge.tex``.

Usage:  uv run python -m experiment.run_judge
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .data import CATEGORIES, make_stream  # noqa: E402
from .eval import SCORED, phase_metrics, scored_queries  # noqa: E402
from .gateway import ALLOW, ASK, SAFETY, SYMMETRIC, run_gateway  # noqa: E402
from .gp import LaplaceGPC  # noqa: E402
from .judge import JudgePolicy, judge_stream  # noqa: E402
from .kernel import AdditiveKernel, ProductKernel  # noqa: E402
from .oracle import OracleConfig  # noqa: E402

HERE = os.path.dirname(__file__)
FIGDIR = os.path.join(HERE, "figures")
REPORT = os.path.join(HERE, "report_judge.md")
RESULTS = os.path.join(HERE, "results_judge.json")
GENDIR = os.path.join(HERE, "..", "manuscript", "generated")

N, T_WARM, T_LATE = 1500, 560, 1050
SEEDS = list(range(10))
EPS = 0.05
WINDOWS = ((560, 750), (750, 1050), (1050, 1500))
COSTS = {"symmetric": SYMMETRIC, "safety": SAFETY}


def m_judge_only():
    return JudgePolicy()


def m_judge_escalate():
    return JudgePolicy(escalate=True)


def m_always_ask():
    return JudgePolicy(always_ask=True)


def m_calibrated():
    return LaplaceGPC(AdditiveKernel(1.6, 1.0, 0.6, l_tool=0.7, lam=90.0))


def m_calibrated_product():
    return LaplaceGPC(ProductKernel(1.6, l_tool=0.7, l_ctx=1.2, lam=90.0))


# name -> (factory, audit rate, uses judge mode)
POLICIES = {
    "judge-only": (m_judge_only, 0.0, False),
    "judge-escalate": (m_judge_escalate, 0.0, False),
    "always-ask": (m_always_ask, 0.0, False),
    "calibrated": (m_calibrated, 0.0, True),
    "calibrated+audit": (m_calibrated, EPS, True),
    "calibrated-product+audit": (m_calibrated_product, EPS, True),
}
LABELS = {
    "judge-only": "Judge only (allow/block)",
    "judge-escalate": "Judge, escalate instead of block",
    "always-ask": "Always ask",
    "calibrated": "Calibrated gateway",
    "calibrated+audit": "Calibrated gateway + 5\\% audits",
    "calibrated-product+audit": "Calibrated (product kernel) + audits",
}


def reliability(res, jstream, lo, hi):
    """Judge reliability for this supervisor in window [lo, hi): the truth and
    the Horvitz-Thompson estimate from the labels the gateway collected, with
    known propensities (1 for ASK, eps for audited auto-decisions)."""
    out = {}
    for side, want_block, bad in (("allow", False, 0), ("block", True, 1)):
        idx = [i for i, d in enumerate(jstream) if lo <= d.t < hi and d.judge_block == want_block]
        if not idx:
            out[side] = dict(truth=float("nan"), est=float("nan"), n=0, labels=0)
            continue
        steps = [res.steps[i] for i in idx]
        truth = float(np.mean([(1 - s.p_true) if bad == 0 else s.p_true for s in steps]))
        est = 0.0
        ok = True
        for s in steps:
            if s.queried:
                prop = 1.0 if s.decision == ASK else res.audit_rate
                if prop <= 0:
                    ok = False
                est += float(s.y == bad) / prop
            elif s.decision != ASK and res.audit_rate <= 0:
                ok = False  # an auto-decision with zero audit propensity
        out[side] = dict(truth=truth, est=est / len(steps) if ok else float("nan"),
                         n=len(steps), labels=int(sum(s.queried for s in steps)))
    return out


def job(spec):
    policy, mode, costs, seed = spec
    fac, eps, _ = POLICIES[policy]
    stream = make_stream(N, seed=1000 + seed)
    js = judge_stream(stream, seed=5000 + seed, mode=mode)
    cfg = OracleConfig(changepoint=750)
    res = run_gateway(js, fac(), np.random.default_rng(seed), cfg, T_WARM, T_LATE,
                      band=COSTS[costs].band, audit_rate=eps)
    m = {k: v for k, v in phase_metrics(res, SCORED, COSTS[costs]).items() if not k.startswith("_")}
    q, n = scored_queries(res)
    m["labels_per_action"] = q / n
    blind = [(s, d) for s, d in zip(res.steps, js) if s.phase in SCORED and d.blind]
    m["blind_n"] = len(blind)
    m["blind_allow"] = sum(s.decision == ALLOW for s, _ in blind)
    m["blind_deny"] = sum(s.oracle_yes == 0 for s, _ in blind)
    m["blind_fa"] = sum(s.decision == ALLOW and s.oracle_yes == 0 for s, _ in blind)
    m["reliability"] = {f"{lo}-{hi}": reliability(res, js, lo, hi) for lo, hi in WINDOWS}
    # How likely is a judge ALLOW to be right, per category, as the gateway believes vs truth.
    percat = {}
    for c in CATEGORIES:
        sel = [s for s, d in zip(res.steps, js)
               if s.phase in SCORED and not d.judge_block and d.tool.category == c]
        if sel:
            percat[c] = (float(np.mean([s.p_hat for s in sel])), float(np.mean([s.p_true for s in sel])),
                         len(sel))
    m["percat"] = percat
    return spec, m


def main():
    specs = []
    for p, (_, _, uses_mode) in POLICIES.items():
        for mode in (("verdict", "score") if uses_mode else ("verdict",)):
            for c in COSTS:
                for s in SEEDS:
                    specs.append((p, mode, c, s))
    print(f"judge study: {len(specs)} runs ...", flush=True)
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 2)) as ex:
        res = dict(ex.map(job, specs, chunksize=1))
    write(res)
    with open(RESULTS, "w") as fh:
        json.dump({"|".join(map(str, k)): v for k, v in res.items()}, fh, indent=1, default=float)
    print(f"Done. See {REPORT}", flush=True)


def ms(v):
    v = np.asarray([x for x in v if np.isfinite(x)], dtype=float)
    return (float(v.mean()), float(v.std())) if v.size else (float("nan"), float("nan"))


def write(res):
    os.makedirs(GENDIR, exist_ok=True)
    L = []
    W = L.append
    mac = {}
    rows = []
    W("# Extension study: calibrating an opaque judge\n")
    W("Pre-registered in `experiment/judge_prereg.md`; no deviations. 10 seeds; the gateway sees "
      "only the judge's output, an 8-way command category and time.\n")
    for costs in COSTS:
        W(f"\n## {costs} costs (band {tuple(round(x, 2) for x in COSTS[costs].band)})\n")
        W("| Policy | Judge input | Regret | Labels/action | False-allow | Blind-spot ALLOW (pooled) "
          "| Blind-spot false-allow (pooled) |")
        W("|---|---|---|---|---|---|---|")
        for p, (_, _, uses_mode) in POLICIES.items():
            for mode in (("verdict", "score") if uses_mode else ("verdict",)):
                rs = [res[(p, mode, costs, s)] for s in SEEDS]
                reg, lab, fa = (ms([r[k] for r in rs]) for k in ("regret", "labels_per_action",
                                                                   "false_allow_rate"))
                bn = sum(r["blind_n"] for r in rs)
                ba = sum(r["blind_allow"] for r in rs)
                bd = sum(r["blind_deny"] for r in rs)
                bfa = sum(r["blind_fa"] for r in rs)
                W(f"| {p} | {mode if uses_mode else '—'} | {reg[0]:.3f} ± {reg[1]:.3f} | "
                  f"{lab[0]:.3f} | {fa[0]:.3f} | {ba}/{bn} | {bfa}/{bd} |")
                key = "".join(w.capitalize() for w in p.replace("+", "-").split("-")) + \
                    ("Score" if mode == "score" else "") + ("Safe" if costs == "safety" else "")
                mac[f"jr{key}Regret"] = f"{reg[0]:.3f}"
                mac[f"jr{key}Labels"] = f"{100 * lab[0]:.1f}"
                mac[f"jr{key}FA"] = "--" if not np.isfinite(fa[0]) else f"{100 * fa[0]:.1f}"
                mac[f"jr{key}Blind"] = f"{bfa}/{bd}"
                if costs == "symmetric" and mode == "verdict":
                    fa_txt = "--" if not np.isfinite(fa[0]) else f"{100 * fa[0]:.1f}"
                    rows.append(f"{LABELS[p]} & {reg[0]:.3f} & {100 * lab[0]:.1f} & {fa_txt} "
                                f"& {bfa}/{bd} & {SAFE_REG(res, p)} & {SAFE_LAB(res, p)} \\\\")

    W("\n## Judge reliability for this supervisor (calibrated+audit, verdict input, symmetric)\n")
    W("P(supervisor denies | judge ALLOW) and P(supervisor approves | judge BLOCK), truth vs "
      "Horvitz-Thompson estimate from the gateway's labels (known propensities), mean over seeds.\n")
    W("| Window | FA of judge ALLOW: truth | estimate (sd) | FB of judge BLOCK: truth | estimate (sd) |")
    W("|---|---|---|---|---|")
    for lo, hi in WINDOWS:
        rs = [res[("calibrated+audit", "verdict", "symmetric", s)]["reliability"][f"{lo}-{hi}"]
              for s in SEEDS]
        ta, ea = ms([r["allow"]["truth"] for r in rs]), ms([r["allow"]["est"] for r in rs])
        tb, eb = ms([r["block"]["truth"] for r in rs]), ms([r["block"]["est"] for r in rs])
        W(f"| [{lo},{hi}) | {ta[0]:.3f} | {ea[0]:.3f} ({ea[1]:.3f}) | {tb[0]:.3f} | {eb[0]:.3f} ({eb[1]:.3f}) |")
        tag = {560: "Pre", 750: "Post", 1050: "Late"}[lo]
        mac[f"jrRel{tag}FATrue"] = f"{100 * ta[0]:.1f}"
        mac[f"jrRel{tag}FAEst"] = f"{100 * ea[0]:.1f}"
        mac[f"jrRel{tag}FASd"] = f"{100 * ea[1]:.1f}"
        mac[f"jrRel{tag}FBTrue"] = f"{100 * tb[0]:.1f}"
        mac[f"jrRel{tag}FBEst"] = f"{100 * eb[0]:.1f}"
        mac[f"jrRel{tag}Labels"] = f"{np.mean([r['allow']['labels'] for r in rs]):.0f}"
    rs = [res[("calibrated", "verdict", "symmetric", s)]["reliability"]["560-750"] for s in SEEDS]
    W(f"\nWithout audits the estimate is undefined in every seed "
      f"({sum(np.isnan(r['allow']['est']) for r in rs)}/{len(rs)}): auto-allowed actions have "
      "zero labelling propensity.\n")

    W("\n## How likely is a judge ALLOW to be right, by command category (calibrated+audit, verdict)\n")
    W("| Category | Gateway's mean p̂ on judge-ALLOWs | True mean approval | Actions (pooled) |")
    W("|---|---|---|---|")
    for c in CATEGORIES:
        vals = [res[("calibrated+audit", "verdict", "symmetric", s)]["percat"].get(c) for s in SEEDS]
        vals = [v for v in vals if v]
        if vals:
            W(f"| {c} | {np.mean([v[0] for v in vals]):.3f} | {np.mean([v[1] for v in vals]):.3f} | "
              f"{sum(v[2] for v in vals)} |")

    with open(REPORT, "w") as fh:
        fh.write("\n".join(L) + "\n")
    with open(os.path.join(GENDIR, "judge.tex"), "w") as fh:
        fh.write("% Generated by experiment/run_judge.py; do not edit.\n")
        for k, v in mac.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
        fh.write("\\newcommand{\\TabJudge}{%\n" + "\n".join(rows) + "\n}\n")
    fig_reliability(res)


def SAFE_LAB(res, p):
    return f"{100 * np.mean([res[(p, 'verdict', 'safety', s)]['labels_per_action'] for s in SEEDS]):.1f}"


def SAFE_REG(res, p):
    return f"{np.mean([res[(p, 'verdict', 'safety', s)]['regret'] for s in SEEDS]):.3f}"


def fig_reliability(res):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    x = np.arange(len(WINDOWS))
    for side, col, lab in (("allow", "#DC2626", "P(deny | judge ALLOW)"),
                           ("block", "#0D9488", "P(approve | judge BLOCK)")):
        rs = [[res[("calibrated+audit", "verdict", "symmetric", s)]["reliability"][f"{lo}-{hi}"][side]
               for s in SEEDS] for lo, hi in WINDOWS]
        truth = [np.mean([r["truth"] for r in w]) for w in rs]
        est = [np.mean([r["est"] for r in w]) for w in rs]
        sd = [np.std([r["est"] for r in w]) for w in rs]
        ax.plot(x, truth, "o-", color=col, lw=2, label=f"{lab}, truth")
        ax.errorbar(x + 0.05, est, yerr=sd, fmt="s--", color=col, alpha=0.7, capsize=3,
                    label=f"{lab}, audit estimate")
    ax.set_xticks(x)
    ax.set_xticklabels(["before reset\n[560,750)", "after reset\n[750,1050)", "late\n[1050,1500)"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("judge error rate for this supervisor")
    fig.legend(fontsize=7, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.savefig(os.path.join(FIGDIR, "judge_reliability.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
