"""Run the progressive-autonomy simulation study (manuscript, Experiments).

WHAT THIS DOES AND DOES NOT SHOW
--------------------------------
A controlled simulation with a known ground-truth oracle
(:mod:`experiment.oracle`), the standard protocol for GP preference and
level-set methods: "the method recovers the oracle" means "the inference is
correct under the model", not "the real world behaves like this". No public
dataset carries a single supervisor's per-action approve/deny decisions as
their tolerance drifts.

Every number in ``report.md``, every figure in ``figures/`` and every macro in
``manuscript/generated/`` is produced by this script. There is no threshold
tuning: the three-tier band is derived from stated costs (Proposition 2).

Usage:  uv run python -m experiment.run
"""

from __future__ import annotations

import copy
import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402

from . import plotting  # noqa: E402
from .data import TOOLS, DecisionPoint, make_stream  # noqa: E402
from .eval import (  # noqa: E402
    SCORED,
    CellModel,
    IndependentModel,
    audit_estimate,
    boundary_accuracy,
    held_out_decisions,
    phase_metrics,
    policy_trajectory,
    query_placement,
    scored_queries,
    transfer_accuracy,
)
from .gateway import ALLOW, ASK, SAFETY, SYMMETRIC, Costs, run_gateway  # noqa: E402
from .gp import EvidenceSelectedGPC, LaplaceGPC  # noqa: E402
from .kernel import AdditiveKernel, LinearKernel, ProductKernel, pack  # noqa: E402
from .oracle import OracleConfig, _veto_active, approve_prob  # noqa: E402

HERE = os.path.dirname(__file__)
FIGDIR = os.path.join(HERE, "figures")
REPORT = os.path.join(HERE, "report.md")
RESULTS = os.path.join(HERE, "results.json")
GENDIR = os.path.join(HERE, "..", "manuscript", "generated")

N = 1500
T_WARM = 560       # learn [0, 560) is warm-up and not scored
T_LATE = 1050      # scored: early [560, 1050) (contains the changepoint), late [1050, 1500)
CHANGEPOINT = 750
SEEDS = list(range(10))
ACQ_SEEDS = list(range(20))
AUDIT_EPS = 0.05

LAMS = (60.0, 120.0, 240.0, 480.0)
ADD_SPLITS = ((1.6, 1.0, 0.6), (3.2, 1.0, 0.6), (1.6, 2.0, 0.6), (1.6, 1.0, 1.6))
COSTS = {"symmetric": SYMMETRIC, "safety": SAFETY}
ORACLES = {
    "changepoint": dict(changepoint=CHANGEPOINT),
    # trust saturates at once and never resets: f* is static, the veto is off
    "stationary": dict(kappa=1e-9, changepoint=None),
}
HOLDOUTS = {
    "dangerous": ("git_force_push", "prod_infra"),
    "benign": ("write_file", "workspace_tests"),
}


# --------------------------------------------------------------------------- #
# Models (module-level factories so worker processes can build them)
# --------------------------------------------------------------------------- #
def m_per_tool():
    return IndependentModel()


def m_per_cell():
    return CellModel()


def m_product():
    return LaplaceGPC(ProductKernel(sigma2=1.6, l_tool=1.1, l_ctx=1.2, lam=90.0))


def m_product_eb():
    return EvidenceSelectedGPC(
        [ProductKernel(s, 1.1, 1.2, lam) for s in (1.6, 3.2) for lam in (*LAMS, 960.0)])


def m_linear_eb():
    return EvidenceSelectedGPC([LinearKernel(lam=lam) for lam in LAMS])


def m_additive():
    # Fixed before any run: 1.6 / 1.0 / 0.6, lam = 90.
    return LaplaceGPC(AdditiveKernel(1.6, 1.0, 0.6, lam=90.0))


def m_additive_eb():
    return EvidenceSelectedGPC(
        [AdditiveKernel(a, b, c, lam=lam) for (a, b, c) in ADD_SPLITS for lam in LAMS])


MODELS = {
    "per-tool": m_per_tool,
    "per-cell": m_per_cell,
    "product": m_product,
    "product-EB": m_product_eb,
    "linear-EB": m_linear_eb,
    "additive": m_additive,
    "additive-EB": m_additive_eb,
}
LABELS = {
    "per-tool": "Per-tool Beta",
    "per-cell": "Per-cell Beta",
    "product": r"Product, $\lambda{=}90$",
    "product-EB": "Product, evidence-selected",
    "linear-EB": "Linear probit + drift, ev.-sel.",
    "additive": r"Additive, $\lambda{=}90$",
    "additive-EB": "Additive, evidence-selected",
}
GP_MODELS = ["product", "product-EB", "linear-EB", "additive", "additive-EB"]


def _cfg(name: str = "changepoint") -> OracleConfig:
    return OracleConfig(**ORACLES[name])


def _stream(seed: int):
    return make_stream(N, seed=1000 + seed)


def _probe_action() -> DecisionPoint:
    """A recurring moderate action whose acceptability drifts over time."""
    tool = next(t for t in TOOLS if t.name == "apply_patch")
    return DecisionPoint(t=0, tool=tool, target="build_config", task="feature_dev",
                         arg_risk=0, target_sens=0.55).featurize()


def _strip(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


def summarize(res, stream, cfg, costs: Costs) -> dict:
    out = {ph: _strip(phase_metrics(res, phases, costs))
           for ph, phases in (("early", ("early",)), ("late", ("late",)), ("scored", SCORED))}
    veto = [s for s, d in zip(res.steps, stream) if s.phase in SCORED and _veto_active(d, cfg)]
    out["veto_n"] = len(veto)
    out["veto_allow"] = float(np.mean([s.decision == ALLOW for s in veto])) if veto else float("nan")
    q, n = scored_queries(res)
    out["labels_per_action"] = q / n
    out["audit"] = audit_estimate(res)
    out["boundary"] = boundary_accuracy(res)
    hist = getattr(res.final_model, "history", None)
    out["selected"] = hist[-1][1] if hist else None
    return out


# --------------------------------------------------------------------------- #
# Jobs (run in worker processes)
# --------------------------------------------------------------------------- #
def job_main(spec):
    model, costs, eps, seed = spec
    stream, cfg = _stream(seed), _cfg()
    res = run_gateway(stream, MODELS[model](), np.random.default_rng(seed), cfg, T_WARM, T_LATE,
                      band=COSTS[costs].band, audit_rate=eps)
    out = summarize(res, stream, cfg, COSTS[costs])
    if costs == "symmetric" and eps == 0:
        out["_reliability"] = phase_metrics(res, SCORED)["_reliability"]
    if seed == 0 and costs == "symmetric" and eps == 0:
        out["_trajectory"] = {k: v.tolist() for k, v in policy_trajectory(res, 70).items()}
    if model == "product" and costs == "symmetric" and eps == 0:
        # Evidence ranking on the labels the product-kernel gateway actually collected.
        fm = res.final_model
        P, y = fm._P, (fm._y > 0).astype(int)
        cands = {
            "product λ=90": ProductKernel(1.6, 1.1, 1.2, 90.0),
            "product λ=480": ProductKernel(1.6, 1.1, 1.2, 480.0),
            "linear+drift λ=90": LinearKernel(lam=90.0),
            "additive λ=90": AdditiveKernel(1.6, 1.0, 0.6, lam=90.0),
            "additive λ=480": AdditiveKernel(1.6, 1.0, 0.6, lam=480.0),
        }
        out["_evidence"] = {k: LaplaceGPC(kk).fit(P, y).log_marginal for k, kk in cands.items()}
    return spec, out


def job_acq(spec):
    kernel, oracle, seed = spec
    stream, cfg = _stream(seed), _cfg(oracle)
    mk = {"product": m_product, "additive": m_additive}[kernel]
    ra = run_gateway(stream, mk(), np.random.default_rng(seed), cfg, T_WARM, T_LATE)
    rate = sum(s.queried for s in ra.steps) / N
    rr = run_gateway(stream, mk(), np.random.default_rng(100 + seed), cfg, T_WARM, T_LATE,
                     query_strategy="random", query_rate=rate)
    rb = run_gateway(stream, mk(), np.random.default_rng(200 + seed), cfg, T_WARM, T_LATE,
                     query_strategy="bald", query_rate=rate)
    return spec, {
        "budget": rate,
        "band": boundary_accuracy(ra), "random": boundary_accuracy(rr),
        "bald": boundary_accuracy(rb),
        "q_band": sum(s.queried for s in ra.steps), "q_random": sum(s.queried for s in rr.steps),
        "q_bald": sum(s.queried for s in rb.steps),
        "place_band": query_placement(ra), "place_random": query_placement(rr),
        "place_bald": query_placement(rb),
    }


def job_hold(spec):
    model, hold, seed = spec
    tool, target = HOLDOUTS[hold]

    def h(dp):
        return dp.tool.name == tool and dp.target == target

    stream, cfg = _stream(seed), _cfg()
    res = run_gateway(stream, MODELS[model](), np.random.default_rng(300 + seed), cfg,
                      T_WARM, T_LATE, hold_out=h)
    out = held_out_decisions(res, stream, h)
    out["final_acc"] = transfer_accuracy(res.final_model, [d for d in stream if h(d)], cfg)
    return spec, out


def job_probe(spec):
    """Track the probe action's p_hat through one run (figure only)."""
    kernel, seed = spec
    mk = {"product": m_product, "additive": m_additive}[kernel]
    res = run_gateway(_stream(seed), mk(), np.random.default_rng(seed), _cfg(), T_WARM, T_LATE,
                      probe=_probe_action())
    return spec, [(t, p) for t, p, _ in res.probe_trace]


def _pool_map(fn, specs):
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 2)) as ex:
        return dict(ex.map(fn, specs, chunksize=1))


# --------------------------------------------------------------------------- #
# Aggregation helpers
# --------------------------------------------------------------------------- #
def ms(vals) -> tuple[float, float]:
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], dtype=float)
    return (float(v.mean()), float(v.std())) if v.size else (float("nan"), float("nan"))


def paired(a, b) -> tuple[float, float, int, int]:
    d = np.asarray(a) - np.asarray(b)
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))), int(np.sum(d > 0)), len(d)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
COL = plotting.COLORS
VETO_END = CHANGEPOINT + int(np.ceil(OracleConfig().kappa * np.log(2.0)))  # trust < theta/2
DECISIONS = (("allow", "ALLOW (auto)"), ("ask", "ASK (escalate)"), ("block", "BLOCK (auto)"))


def _mark_time(ax, x, text, color="white"):
    ax.axvline(x, color=plotting.COLORS["marker"], ls=(0, (3, 2)), lw=0.7)
    ax.text(x - 12, 0.03, text, rotation=90, ha="right", va="bottom", fontsize=6.5, color=color)


def fig_policy_evolution(main, path):
    """Rolling decision mix over the stream for the two kernels (seed 0)."""
    fig, axes = plt.subplots(1, 2, figsize=(plotting.TEXT_WIDTH, 2.35), sharey=True)
    for ax, model, letter, ttl in ((axes[0], "product", "a", "Product kernel"),
                                   (axes[1], "additive", "b", "Additive kernel")):
        tr = main[(model, "symmetric", 0.0, 0)]["_trajectory"]
        ax.stackplot(tr["t"], tr["allow"], tr["ask"], tr["block"],
                     labels=[lab for _, lab in DECISIONS],
                     colors=[COL[k] for k, _ in DECISIONS], alpha=0.88, lw=0)
        _mark_time(ax, T_WARM, "warm-up ends")
        _mark_time(ax, CHANGEPOINT, "trust reset")
        # veto window: bracket above the plot
        tf = ax.get_xaxis_transform()
        ax.plot([CHANGEPOINT, VETO_END], [1.025, 1.025], color=COL["block"], lw=1.6,
                transform=tf, clip_on=False, solid_capstyle="butt")
        ax.text((CHANGEPOINT + VETO_END) / 2, 1.045, "veto", transform=tf, ha="center",
                va="bottom", fontsize=6.5, color=COL["block"])
        ax.set_xlim(0, N - 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(r"decision point $t$")
        plotting.panel(ax, letter, ttl)
    axes[0].set_ylabel("share of decisions\n(rolling 70 steps, seed 0)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.99))
    fig.tight_layout(w_pad=1.2)
    plotting.save(fig, path)


def probe_traces(probe_runs):
    """Mean and 25-75% band of the probe action's p_hat across seeds, on a
    common grid (each seed's trace is held constant between refits)."""
    grid = np.arange(0, N, 5)
    out = {}
    for kern in ("product", "additive"):
        rows = []
        for s in SEEDS:
            arr = np.asarray(probe_runs[(kern, s)])
            idx = np.searchsorted(arr[:, 0], grid, side="right") - 1
            v = np.where(idx >= 0, arr[np.clip(idx, 0, None), 1], np.nan)
            rows.append(v)
        rows = np.asarray(rows)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # before the first fit
            out[kern] = (np.nanmean(rows, 0), np.nanpercentile(rows, 25, 0),
                         np.nanpercentile(rows, 75, 0))
    probe = _probe_action()
    cfg = _cfg()
    truth = []
    for t in grid:
        pr = copy.copy(probe)
        pr.t = int(t)
        truth.append(approve_prob(pr, cfg))
    return grid, out, np.asarray(truth)


def forgetting_curves():
    """Proposition 4 illustration: fit on the labels collected up to t0, then
    predict fixed actions at t0 + Delta with no further labels."""
    stream, cfg = _stream(0), _cfg("stationary")
    t0 = 700
    res = run_gateway(stream[:t0], m_product(), np.random.default_rng(0), cfg, t0, t0)
    pts = [d for d, s in zip(stream[:t0], res.steps) if s.queried]
    ys = [s.y for s in res.steps if s.queried]
    acts = {
        "read_file → workspace_src": ("read_file", 0.40, "bugfix"),
        "delete_file → ci_config": ("delete_file", 0.65, "ops_maintenance"),
    }
    deltas = np.linspace(0, 800, 81)
    out = {}
    for kname, k in (("product", ProductKernel(1.6, 1.1, 1.2, 90.0)),
                     ("additive", AdditiveKernel(1.6, 1.0, 0.6, lam=90.0))):
        m = LaplaceGPC(k).fit(pack(pts), np.array(ys))
        for aname, (tool, sens, task) in acts.items():
            tl = next(t for t in TOOLS if t.name == tool)
            q = [DecisionPoint(t=int(t0 + dl), tool=tl, target="x", task=task, arg_risk=0,
                               target_sens=sens).featurize() for dl in deltas]
            out[(kname, aname)] = m.predict_prob(pack(q))
            out[("truth", aname)] = np.array([approve_prob(d, cfg) for d in q])
    return deltas, out, list(acts)


def _pooled_reliability(main, model):
    """Reliability curve pooled over seeds; every seed scores the same number
    of decisions, so bin fractions are exact pooling weights."""
    bins = [main[(model, "symmetric", 0.0, s)]["_reliability"] for s in SEEDS]
    xs, ys, ws = [], [], []
    for b in range(len(bins[0])):
        w = np.array([bs[b][2] for bs in bins])
        if w.sum() == 0:
            continue
        xs.append(sum(bs[b][0] * bs[b][2] for bs in bins if bs[b][2] > 0) / w.sum())
        ys.append(sum(bs[b][1] * bs[b][2] for bs in bins if bs[b][2] > 0) / w.sum())
        ws.append(w.mean())
    return np.array(xs), np.array(ys), np.array(ws)


def _tt(name):
    return r"$\mathtt{" + name.replace("_", r"\_") + "}$"


def fig_forgetting(main, probe_runs, path):
    """(a) Forgetting on a fixed training set, (b) tracking a probe action under
    drift, (c) the resulting calibration, pooled over seeds."""
    deltas, out, acts = forgetting_curves()
    fig = plt.figure(figsize=(plotting.TEXT_WIDTH, 2.25))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.3, 0.92], wspace=0.34,
                          left=0.07, right=0.995, bottom=0.17, top=0.9)
    ax_a, ax_b, ax_c = (fig.add_subplot(gs[0, i]) for i in range(3))

    # (a) forgetting
    ax_a.axhspan(SYMMETRIC.tau_low, SYMMETRIC.tau_high, color=COL["ask"], alpha=0.16, lw=0)
    ax_a.text(15, SYMMETRIC.tau_low + 0.015, "ASK band", fontsize=6.5, color="#92400E", va="bottom")
    for aname, ls in zip(acts, ("-", (0, (4, 2)))):
        for kern in ("product", "additive"):
            ax_a.plot(deltas, out[(kern, aname)], color=COL[kern], ls=ls)
    end = {k: out[(k, acts[0])][-1] for k in ("product", "additive")}
    ax_a.text(795, end["additive"] + 0.025, "additive", color=COL["additive"], ha="right",
              va="bottom", fontsize=7)
    ax_a.text(795, end["product"] + 0.025, "product", color=COL["product"], ha="right",
              va="bottom", fontsize=7)
    hs = [Line2D([], [], color="k", lw=1.0, ls=ls) for ls in ("-", (0, (4, 2)))]
    names = [_tt(a.split(" → ")[0]) + r" $\to$ " + _tt(a.split(" → ")[1]) for a in acts]
    ax_a.legend(hs, names, loc="lower left", fontsize=6.5, handlelength=2.2)
    ax_a.set_xlim(0, 800)
    ax_a.set_ylim(0, 1)
    ax_a.set_xlabel(r"steps since the last label, $\Delta$")
    ax_a.set_ylabel(r"predicted approval $\hat p$")
    plotting.panel(ax_a, "a", "No new labels")

    # (b) tracking under drift
    grid, tr, truth = probe_traces(probe_runs)
    ax_b.axvline(CHANGEPOINT, color=COL["marker"], ls=(0, (3, 2)), lw=0.7)
    ax_b.text(CHANGEPOINT - 25, 0.03, "trust reset", rotation=90, fontsize=6.5,
              color=COL["marker"], ha="right", va="bottom")
    for kern in ("product", "additive"):
        mean, lo, hi = tr[kern]
        ax_b.fill_between(grid, lo, hi, color=COL[kern], alpha=0.18, lw=0)
        ax_b.plot(grid, mean, color=COL[kern], lw=1.1, label=f"{kern} $\\hat p$")
    ax_b.plot(grid, truth, color=COL["oracle"], lw=1.4, label=r"oracle $\Phi(f^*)$")
    h, lab = ax_b.get_legend_handles_labels()
    ax_b.legend(h[::-1], lab[::-1], loc="lower right", fontsize=6.5, handlelength=1.2)
    ax_b.set_xlim(0, N)
    ax_b.set_xticks([0, 500, 1000, 1500])
    ax_b.set_ylim(0, 1)
    ax_b.set_xlabel(r"decision point $t$")
    ax_b.set_ylabel(r"approval probability")
    plotting.panel(ax_b, "b", "One action under drift")

    # (c) calibration
    ax_c.plot([0, 1], [0, 1], color="#9CA3AF", ls=(0, (3, 2)), lw=0.8)
    for model, key, lab in (("product", "product", "product"), ("additive", "additive", "additive"),
                            ("linear-EB", "linear", "linear probit")):
        xs, ys, _ = _pooled_reliability(main, model)
        ax_c.plot(xs, ys, "o-", color=COL[key], ms=2.8, lw=1.0, label=lab)
    ax_c.legend(loc="lower right", fontsize=6.5, handlelength=1.4)
    ax_c.set_xlim(0, 1)
    ax_c.set_ylim(0, 1)
    ax_c.set_xticks(np.linspace(0, 1, 6))
    ax_c.set_xlabel(r"predicted $\hat p$ (bin mean)")
    ax_c.set_ylabel(r"true $\Phi(f^*)$ (bin mean)")
    plotting.panel(ax_c, "c", "Calibration")
    plotting.save(fig, path)
    return deltas, out, acts


HOLD_MODELS = (("per-tool", "Per-tool Beta"), ("per-cell", "Per-cell Beta"),
               ("product", "Product GP"), ("linear-EB", "Linear probit"),
               ("additive", "Additive GP"))


def fig_transfer(hold, path):
    """Decisions on the two never-labelled combinations, pooled over seeds."""
    fig, axes = plt.subplots(1, 2, figsize=(plotting.TEXT_WIDTH, 1.95), sharey=True)
    y = np.arange(len(HOLD_MODELS))[::-1]
    for ax, hname, letter in zip(axes, ("benign", "dangerous"), ("a", "b")):
        for yi, (m, _) in zip(y, HOLD_MODELS):
            rows = [hold[(m, hname, s)] for s in SEEDS]
            n = sum(r["n"] for r in rows)
            shares = {k: sum(r[k] for r in rows) / n for k in ("allow", "ask", "block")}
            fa = sum(r["false_allow"] for r in rows)
            nd = sum(r["n_deny"] for r in rows)
            left = 0.0
            for k, _ in DECISIONS:
                ax.barh(yi, shares[k], left=left, height=0.62, color=COL[k], alpha=0.88)
                left += shares[k]
            if hname == "dangerous":
                ax.barh(yi, fa / n, left=0.0, height=0.62, facecolor="none",
                        edgecolor="white", hatch="//////", lw=0)
                ax.text(1.03, yi, f"{fa}/{nd} ({100 * fa / nd:.0f}%)", va="center", fontsize=6.5,
                        transform=ax.get_yaxis_transform(), clip_on=False)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.6, len(HOLD_MODELS) + 0.05)
        ax.set_yticks(y)
        ax.set_yticklabels([lab for _, lab in HOLD_MODELS])
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("share of occurrences (10 seeds pooled)")
        tool, target = HOLDOUTS[hname]
        plotting.panel(ax, letter, f"{hname.capitalize()}: {_tt(tool)}" + r" $\to$ " + _tt(target))
    axes[1].text(1.03, len(HOLD_MODELS) - 0.62, "denials allowed", fontsize=6.5,
                 va="bottom", transform=axes[1].get_yaxis_transform(), clip_on=False)
    handles = [Patch(color=COL[k], alpha=0.88, label=lab) for k, lab in DECISIONS]
    handles.append(Patch(facecolor=COL["allow"], alpha=0.88, edgecolor="white", hatch="//////",
                         label="ALLOW of an action the oracle denies"))
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.99))
    fig.tight_layout(w_pad=2.5)
    plotting.save(fig, path)


# --------------------------------------------------------------------------- #
# Report and LaTeX macros
# --------------------------------------------------------------------------- #
def f3(p):
    return "n/a" if not np.isfinite(p[0]) else f"{p[0]:.3f} ± {p[1]:.3f}"


def pct(p, nd=1):
    return "n/a" if not np.isfinite(p[0]) else f"{100 * p[0]:.{nd}f}"


def agg_main(main, model, costs, eps, key, phase="scored"):
    vals = []
    for s in SEEDS:
        r = main[(model, costs, eps, s)]
        v = r[phase].get(key) if phase else r.get(key)
        vals.append(v)
    return ms(vals)


def _write_table(path: str, name: str, rows: list[str]) -> None:
    """Write table rows as a macro: a bare \\input inside a tabular breaks the
    following \\bottomrule, a macro expands cleanly."""
    with open(path, "w") as fh:
        fh.write(f"% Generated by experiment/run.py; do not edit.\n\\newcommand{{\\{name}}}{{%\n")
        fh.write("\n".join(rows) + "\n}\n")


def pooled_veto(main, model, costs, eps) -> tuple[int, int]:
    """Veto-window ALLOWs pooled over seeds (the window holds only a few actions
    per seed, so per-seed rates are too noisy to average)."""
    k = n = 0
    for s in SEEDS:
        r = main[(model, costs, eps, s)]
        if r["veto_n"]:
            k += int(round(r["veto_allow"] * r["veto_n"]))
            n += r["veto_n"]
    return k, n


def pv(main, model, costs, eps) -> str:
    k, n = pooled_veto(main, model, costs, eps)
    return f"{k}/{n}"


def write_outputs(main, acq, hold, forget):
    os.makedirs(GENDIR, exist_ok=True)
    L = []
    W = L.append
    macros = {}

    def mac(name, value):
        macros[name] = value

    W("# Progressive Autonomy as Preference Learning: Experiment Report\n")
    W(f"Generated by `uv run python -m experiment.run`. Stream length N={N}; "
      f"{len(SEEDS)} seeds for the main tables, {len(ACQ_SEEDS)} paired seeds for the "
      "acquisition probe. Every number below is recomputed from the code; nothing is "
      "tuned on labels.\n")
    W("## Protocol\n")
    W(f"- Oracle: probit approval `Phi(f*)`, `f*` = static action acceptability + saturating "
      f"trust with an abrupt reset at t={CHANGEPOINT} + a three-way veto (irreversible AND "
      "sensitive AND low trust) + task offset (`experiment/oracle.py`).")
    W(f"- Phases: warm-up `[0,{T_WARM})` (not scored); scored early `[{T_WARM},{T_LATE})` "
      f"(contains the changepoint) and late `[{T_LATE},{N})`. Prequential: every decision is "
      "logged before any label at that step exists.")
    W(f"- Thresholds from Chow costs (Proposition 2). Symmetric: c_FA=c_FB={SYMMETRIC.c_fa:.3f}, "
      f"c_ask=1, band {tuple(round(x, 3) for x in SYMMETRIC.band)}. Safety-weighted: "
      f"c_FA={SAFETY.c_fa:g}, c_FB={SAFETY.c_fb:g}, c_ask=1, band "
      f"{tuple(round(x, 3) for x in SAFETY.band)}.")
    W("- Regret = mean Chow loss of the gateway's decisions minus that of the oracle policy "
      "that knows `Phi(f*)`, in units of one escalation.")
    W("- Hyperparameters: `l_tool=1.1`, `l_ctx=1.2` throughout. Product: σ²=1.6, λ=90. "
      "Additive (fixed before any run): σs²=1.6, σg²=1.0, "
      f"σd²=0.6, λ=90. Evidence-selected (EB) models re-select by Laplace marginal likelihood "
      f"every 64 labels over: product σ²∈{{1.6,3.2}} × λ∈{{60,120,240,480,960}}; additive "
      f"splits {ADD_SPLITS} × λ∈{{60,120,240,480}}; linear λ∈{{60,120,240,480}}.\n")

    # ---- main table (symmetric, no audits) -----------------------------------
    W("## Main comparison (symmetric costs, no audits; scored phases)\n")
    W("| Model | ASK | Auto acc. | False-allow | ECE | Regret | Labels/action | Veto-window ALLOW (pooled) |")
    W("|---|---|---|---|---|---|---|---|")
    rows_tex = []
    for m in MODELS:
        g = lambda k, ph="scored": agg_main(main, m, "symmetric", 0.0, k, ph)  # noqa: E731
        W(f"| {m} | {f3(g('ask_rate'))} | {f3(g('accuracy_auto'))} | {f3(g('false_allow_rate'))} "
          f"| {f3(g('ece'))} | {f3(g('regret'))} | {f3(g('labels_per_action', None))} "
          f"| {pv(main, m, 'symmetric', 0.0)} |")
        rows_tex.append(
            f"{LABELS[m]} & {g('regret')[0]:.3f} & {pct(g('ask_rate'))} & {pct(g('accuracy_auto'))} & "
            f"{pct(g('false_allow_rate'))} & {g('ece')[0]:.3f} & "
            f"{pv(main, m, 'symmetric', 0.0)} \\\\")
    floor_sym = agg_main(main, "additive", "symmetric", 0.0, "floor_ask")
    floor_safe = agg_main(main, "additive", "safety", 0.0, "floor_ask")
    W(f"\nPerfect-posterior escalation floor (Proposition 3): symmetric band "
      f"{pct(floor_sym)}%, safety band {pct(floor_safe)}%.\n")
    _write_table(os.path.join(GENDIR, "table_main.tex"), "TabMain", rows_tex)

    for m, key in (("product", "V"), ("additive", "A"), ("linear-EB", "L"),
                   ("product-EB", "PE"), ("additive-EB", "AE"), ("per-tool", "T"),
                   ("per-cell", "C")):
        for k, name in (("ask_rate", "Ask"), ("accuracy_auto", "Acc"), ("false_allow_rate", "FA"),
                        ("ece", "Ece"), ("regret", "Regret")):
            v = agg_main(main, m, "symmetric", 0.0, k)
            mac(f"res{key}{name}", f"{100 * v[0]:.1f}" if name not in ("Ece", "Regret")
                else f"{v[0]:.3f}")
        mac(f"res{key}Veto", pv(main, m, "symmetric", 0.0))
        mac(f"res{key}Labels",
            f"{100 * agg_main(main, m, 'symmetric', 0.0, 'labels_per_action', None)[0]:.1f}")
        mac(f"res{key}AskLate",
            f"{100 * agg_main(main, m, 'symmetric', 0.0, 'ask_rate', 'late')[0]:.1f}")
    mac("resFloorSym", f"{100 * floor_sym[0]:.1f}")
    rv = agg_main(main, "product", "symmetric", 0.0, "regret")[0]
    ra = agg_main(main, "additive", "symmetric", 0.0, "regret")[0]
    mac("resRegretCut", f"{100 * (1 - ra / rv):.0f}")
    mac("resFloorSafe", f"{100 * floor_safe[0]:.1f}")
    for m, key in (("product", "V"), ("additive", "A")):
        lpa = agg_main(main, m, "symmetric", 0.0, "labels_per_action", None)[0]
        mac(f"res{key}Burden", f"{1 / lpa:.1f}")
    sel = [main[("additive-EB", "symmetric", 0.0, s)]["selected"] for s in SEEDS]
    W("Final kernel chosen by evidence (additive-EB, per seed): " + "; ".join(map(str, sel)) + "\n")
    sel = [main[("product-EB", "symmetric", 0.0, s)]["selected"] for s in SEEDS]
    W("Final kernel chosen by evidence (product-EB, per seed): " + "; ".join(map(str, sel)) + "\n")

    # ---- phase split -------------------------------------------------------
    W("\n## Early (contains changepoint) vs late phase, symmetric costs\n")
    W("| Model | ASK early | ASK late | Acc early | Acc late | Regret early | Regret late |")
    W("|---|---|---|---|---|---|---|")
    for m in GP_MODELS:
        g = lambda k, ph: agg_main(main, m, "symmetric", 0.0, k, ph)  # noqa: E731
        W(f"| {m} | {f3(g('ask_rate', 'early'))} | {f3(g('ask_rate', 'late'))} | "
          f"{f3(g('accuracy_auto', 'early'))} | {f3(g('accuracy_auto', 'late'))} | "
          f"{f3(g('regret', 'early'))} | {f3(g('regret', 'late'))} |")

    # ---- safety-weighted ---------------------------------------------------
    W("\n## Safety-weighted costs (band (0.25, 0.90)), no audits\n")
    W("| Model | ASK | Auto acc. | False-allow | Regret | Veto-window ALLOW (pooled) |")
    W("|---|---|---|---|---|---|")
    rows_tex = []
    for m in GP_MODELS:
        g = lambda k, ph="scored": agg_main(main, m, "safety", 0.0, k, ph)  # noqa: E731
        W(f"| {m} | {f3(g('ask_rate'))} | {f3(g('accuracy_auto'))} | {f3(g('false_allow_rate'))} "
          f"| {f3(g('regret'))} | {pv(main, m, 'safety', 0.0)} |")
        rows_tex.append(f"{LABELS[m]} & {pct(g('ask_rate'))} & {pct(g('accuracy_auto'))} & "
                        f"{g('regret')[0]:.3f} \\\\")
    _write_table(os.path.join(GENDIR, "table_safety.tex"), "TabSafety", rows_tex)
    for m, key in (("product", "V"), ("additive", "A"), ("linear-EB", "L")):
        mac(f"resSafe{key}Ask", f"{100 * agg_main(main, m, 'safety', 0.0, 'ask_rate')[0]:.1f}")
        mac(f"resSafe{key}Regret", f"{agg_main(main, m, 'safety', 0.0, 'regret')[0]:.3f}")

    # ---- audits ------------------------------------------------------------
    W(f"\n## Random audits (ε={AUDIT_EPS} of auto-decided actions), symmetric costs\n")
    W("The Horvitz-Thompson estimate k/(ε N_allow) is the unbiased estimator of Proposition 5 "
      "(`lean/TrustCalib/Audit.lean`); the Jeffreys interval is on the audited fraction k/n "
      "(the Hájek ratio), which is consistent but not the object of the theorem.\n")
    W("Regret charges decisions only, so audits are free in it; the next column bills each audit as "
      "one escalation, and the break-even is the audit cost at which the two are equal.\n")
    W("| Model | Labels/action | Regret (audits free) | Regret (audit = 1 escalation) | Break-even audit cost "
      "| Veto-window ALLOW (pooled) | HT realized-FA estimate | Truth | Jeffreys 95% (on k/n) covers truth |")
    W("|---|---|---|---|---|---|---|---|---|")
    rows_tex = []
    for m in ("product", "linear-EB", "additive", "additive-EB"):
        g = lambda k, ph="scored": agg_main(main, m, "symmetric", AUDIT_EPS, k, ph)  # noqa: E731
        aud = [main[(m, "symmetric", AUDIT_EPS, s)]["audit"] for s in SEEDS]
        est = ms([a["ipw_estimate"] for a in aud])
        tru = ms([a["truth"] for a in aud])
        cov = np.mean([a["jeffreys_lo"] <= a["truth"] <= a["jeffreys_hi"] for a in aud])
        g0 = lambda k, ph="scored": agg_main(main, m, "symmetric", 0.0, k, ph)  # noqa: E731
        charged = g('regret_audit_charged')
        be = (g0('regret')[0] - g('regret')[0]) / max(g('audit_frac')[0], 1e-9)
        W(f"| {m} | {f3(g('labels_per_action', None))} | {f3(g('regret'))} | {f3(charged)} | {be:.2f} | "
          f"{pv(main, m, 'symmetric', AUDIT_EPS)} (no audit: {pv(main, m, 'symmetric', 0.0)}) | {f3(est)} | "
          f"{f3(tru)} | {100 * cov:.0f}% |")
        mac_key = {"product": "V", "linear-EB": "L", "additive": "A", "additive-EB": "AE"}[m]
        mac(f"resAud{mac_key}Charged", f"{charged[0]:.3f}")
        mac(f"resAud{mac_key}BreakEven", f"{be:.2f}")
        rows_tex.append(
            f"{LABELS[m]} & {pct(g0('labels_per_action', None))} $\\to$ {pct(g('labels_per_action', None))} & "
            f"{g0('regret')[0]:.3f} $\\to$ {g('regret')[0]:.3f} ({charged[0]:.3f}) & "
            f"{pv(main, m, 'symmetric', 0.0)} $\\to$ {pv(main, m, 'symmetric', AUDIT_EPS)} & "
            f"{100 * est[0]:.1f} / {100 * tru[0]:.1f} & {100 * cov:.0f}\\% \\\\")
    _write_table(os.path.join(GENDIR, "table_audit.tex"), "TabAudit", rows_tex)
    for m, key in (("product", "V"), ("additive", "A"), ("linear-EB", "L")):
        mac(f"resAud{key}Veto", pv(main, m, "symmetric", AUDIT_EPS))
        mac(f"resAud{key}Regret", f"{agg_main(main, m, 'symmetric', AUDIT_EPS, 'regret')[0]:.3f}")
        mac(f"resAud{key}Labels",
            f"{100 * agg_main(main, m, 'symmetric', AUDIT_EPS, 'labels_per_action', None)[0]:.1f}")
        aud = [main[(m, "symmetric", AUDIT_EPS, s)]["audit"] for s in SEEDS]
        mac(f"resAud{key}Est", f"{100 * np.mean([a['ipw_estimate'] for a in aud]):.1f}")
        mac(f"resAud{key}Truth", f"{100 * np.mean([a['truth'] for a in aud]):.1f}")
        mac(f"resAud{key}Cover",
            f"{100 * np.mean([a['jeffreys_lo'] <= a['truth'] <= a['jeffreys_hi'] for a in aud]):.0f}")

    # ---- acquisition -------------------------------------------------------
    W("\n## Acquisition probe (matched budget, prequential boundary accuracy, paired seeds)\n")
    W("| Kernel | Oracle | Band | Random | BALD | Band − random (SE, wins) | "
      "BALD − random (SE, wins) |")
    W("|---|---|---|---|---|---|---|")
    rows_tex = []
    for kern in ("product", "additive"):
        for orc in ("stationary", "changepoint"):
            rs = [acq[(kern, orc, s)] for s in ACQ_SEEDS]
            b = [r["band"] for r in rs]
            r_ = [r["random"] for r in rs]
            bd = [r["bald"] for r in rs]
            d1, d2 = paired(b, r_), paired(bd, r_)
            W(f"| {kern} | {orc} | {np.mean(b):.3f} | {np.mean(r_):.3f} | {np.mean(bd):.3f} | "
              f"{100 * d1[0]:+.1f} pp ({100 * d1[1]:.1f}, {d1[2]}/{d1[3]}) | "
              f"{100 * d2[0]:+.1f} pp ({100 * d2[1]:.1f}, {d2[2]}/{d2[3]}) |")
            rows_tex.append(
                f"{LABELS[kern]} & {orc} & {100 * np.mean(b):.1f} & {100 * np.mean(r_):.1f} & "
                f"{100 * np.mean(bd):.1f} & ${100 * d1[0]:+.1f} \\pm {100 * d1[1]:.1f}$ & "
                f"${100 * d2[0]:+.1f} \\pm {100 * d2[1]:.1f}$ \\\\")
            key = ("V" if kern == "product" else "A") + ("S" if orc == "stationary" else "C")
            mac(f"resAcq{key}Gap", f"{100 * d1[0]:+.1f}")
            mac(f"resAcq{key}Se", f"{100 * d1[1]:.1f}")
            mac(f"resAcq{key}BaldGap", f"{100 * d2[0]:+.1f}")
            mac(f"resAcq{key}BaldSe", f"{100 * d2[1]:.1f}")
    _write_table(os.path.join(GENDIR, "table_acq.tex"), "TabAcq", rows_tex)

    W("\nWhere labels go (changepoint oracle, scored phases, mean over seeds):\n")
    W("| Kernel | Strategy | Labels | Share on ambiguous actions (q∈[.35,.65]) | Mean latent var "
      "| Mean abs(mu) | Mean BALD (bits) |")
    W("|---|---|---|---|---|---|---|")
    for kern in ("product", "additive"):
        rs = [acq[(kern, "changepoint", s)] for s in ACQ_SEEDS]
        for strat in ("band", "random", "bald"):
            pl = [r[f"place_{strat}"] for r in rs if r[f"place_{strat}"]]
            W(f"| {kern} | {strat} | {np.mean([p['n'] for p in pl]):.0f} | "
              f"{np.mean([p['ambiguous_share'] for p in pl]):.3f} | "
              f"{np.mean([p['mean_var'] for p in pl]):.3f} | "
              f"{np.mean([p['mean_abs_mu'] for p in pl]):.3f} | "
              f"{np.mean([p['mean_bald'] for p in pl]):.3f} |")
            if kern == "product":
                mac(f"resPlace{strat.capitalize()}Amb",
                    f"{100 * np.mean([p['ambiguous_share'] for p in pl]):.1f}")
                mac(f"resPlace{strat.capitalize()}Var", f"{np.mean([p['mean_var'] for p in pl]):.2f}")
    stream_q = np.concatenate([[approve_prob(d, _cfg()) for d in _stream(s)][T_WARM:]
                               for s in ACQ_SEEDS])
    base = float(np.mean((stream_q >= 0.35) & (stream_q <= 0.65)))
    W(f"\nBase rate of ambiguous actions in the scored stream: {base:.3f}.\n")
    mac("resPlaceBaseAmb", f"{100 * base:.1f}")

    # ---- hold-outs ---------------------------------------------------------
    W("\n## Held-out combinations (never labelled; prequential decisions in scored phases)\n")
    W("| Model | Hold-out | Actions | Oracle denials | False-allow | ASK share | Correct auto |")
    W("|---|---|---|---|---|---|---|")
    for hname in ("benign", "dangerous"):
        for m in ("per-tool", "per-cell", "product", "linear-EB", "additive"):
            rs = [hold[(m, hname, s)] for s in SEEDS]
            n = sum(r["n"] for r in rs)
            nd = sum(r["n_deny"] for r in rs)
            fa = sum(r["false_allow"] for r in rs)
            W(f"| {m} | {hname} | {n} | {nd} | {fa} | {sum(r['ask'] for r in rs) / max(n, 1):.2f} "
              f"| {sum(r['correct_auto'] for r in rs) / max(n, 1):.2f} |")
            key = {"product": "V", "additive": "A", "linear-EB": "L", "per-tool": "T",
                   "per-cell": "C"}[m] + ("D" if hname == "dangerous" else "B")
            mac(f"resHold{key}FA", str(fa))
            mac(f"resHold{key}Deny", str(nd))
            mac(f"resHold{key}Correct", f"{100 * sum(r['correct_auto'] for r in rs) / max(n, 1):.0f}")

    # ---- evidence ranking --------------------------------------------------
    W("\n## Laplace log marginal likelihood on the labels the product-kernel gateway collected\n")
    W("The labelled set was selected by the product-kernel policy (escalated actions only), so this is "
      "evidence under a selected design, not on a random sample of the stream.\n")
    ev = [main[("product", "symmetric", 0.0, s)]["_evidence"] for s in SEEDS]
    names = list(ev[0])
    W("| Kernel | Mean log evidence | Best in # seeds |")
    W("|---|---|---|")
    for k in names:
        wins = sum(1 for e in ev if max(e, key=e.get) == k)
        W(f"| {k} | {np.mean([e[k] for e in ev]):.1f} | {wins}/{len(ev)} |")
    mac("resEvV", f"{np.mean([e['product λ=90'] for e in ev]):.1f}")
    mac("resEvA", f"{np.mean([e['additive λ=90'] for e in ev]):.1f}")
    mac("resEvAWins", str(sum(1 for e in ev if max(e, key=e.get) == 'additive λ=90')))
    mac("resEvLWins", str(sum(1 for e in ev if max(e, key=e.get) == 'linear+drift λ=90')))

    # ---- forgetting ---------------------------------------------------------
    deltas, out, acts = forget
    W("\n## Forgetting (Proposition 4): fixed training set, predictions Δ steps later\n")
    W("| Action | Kernel | p̂ at Δ=0 | Δ=200 | Δ=800 | truth |")
    W("|---|---|---|---|---|---|")
    i200 = int(np.argmin(np.abs(deltas - 200)))
    for a in acts:
        for kname in ("product", "additive"):
            v = out[(kname, a)]
            W(f"| {a} | {kname} | {v[0]:.3f} | {v[i200]:.3f} | {v[-1]:.3f} | {out[('truth', a)][0]:.3f} |")
    mac("resForgetProdEnd", f"{out[('product', acts[0])][-1]:.2f}")
    mac("resForgetAddEnd", f"{out[('additive', acts[0])][-1]:.2f}")
    mac("resForgetStart", f"{out[('product', acts[0])][0]:.2f}")

    W("\n## Claim → evidence map\n")
    W("| Claim | Status | Evidence |")
    W("|---|---|---|")
    W("| Three-tier rule is Bayes-optimal under Chow costs (Prop. 2) | proved | `lean/TrustCalib/Chow.lean` |")
    W("| Unary feedback identifies the boundary, pairwise does not (Prop. 1) | proved | `lean/TrustCalib/Identifiability.lean` |")
    W("| Separable time kernel forgets static risk; re-escalation (Prop. 4) | proved + measured | `lean/TrustCalib/Forgetting.lean`, `figures/forgetting.pdf`, test |")
    W("| Escalation floor is set by the supervisor (Prop. 3) | proved + measured | `lean/TrustCalib/Floor.lean`, floor above |")
    W("| Audits give unbiased false-allow estimates; certification size (Prop. 5) | proved + measured | `lean/TrustCalib/Audit.lean`, audit table |")
    W("| Additive kernel reduces burden at matched safety | supported (indicative: oracle is additive by construction) | main table |")
    W("| Band-as-acquisition fails because of forgetting, not stationarity | supported | acquisition table |")
    W("| Evidence selection optimizes average fit, not tail safety | observed | veto-window column, hold-outs |")

    W("\n## Limitations\n")
    W("- Simulation only; the oracle is synthetic and additive by construction, so the additive "
      "kernel's advantage is indicative, and a linear probit with the same time structure is "
      "competitive on average metrics because the oracle's static term is linear in the features.")
    W("- Veto-window numbers rest on few actions per seed (the veto is active for ~90 steps after "
      "the reset); read them as directional.")
    W("- Costs are stated, not elicited; the two operating points bracket plausible choices.")

    with open(REPORT, "w") as fh:
        fh.write("\n".join(L) + "\n")
    with open(os.path.join(GENDIR, "results.tex"), "w") as fh:
        fh.write("% Generated by experiment/run.py; do not edit.\n")
        for k, v in macros.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    specs = [(m, "symmetric", 0.0, s) for m in MODELS for s in SEEDS]
    specs += [(m, "safety", 0.0, s) for m in GP_MODELS for s in SEEDS]
    specs += [(m, "symmetric", AUDIT_EPS, s)
              for m in ("product", "linear-EB", "additive", "additive-EB") for s in SEEDS]
    print(f"main: {len(specs)} runs ...", flush=True)
    main_res = _pool_map(job_main, specs)
    acq_specs = [(k, o, s) for k in ("product", "additive") for o in ORACLES for s in ACQ_SEEDS]
    print(f"acquisition: {len(acq_specs)} paired jobs ...", flush=True)
    acq = _pool_map(job_acq, acq_specs)
    hold_specs = [(m, h, s) for m in ("per-tool", "per-cell", "product", "linear-EB", "additive")
                  for h in HOLDOUTS for s in SEEDS]
    print(f"hold-outs: {len(hold_specs)} runs ...", flush=True)
    hold = _pool_map(job_hold, hold_specs)

    probe_specs = [(k, s) for k in ("product", "additive") for s in SEEDS]
    print(f"probe traces: {len(probe_specs)} runs ...", flush=True)
    probes = _pool_map(job_probe, probe_specs)

    print("figures ...", flush=True)
    plotting.apply()
    fig_policy_evolution(main_res, os.path.join(FIGDIR, "policy_evolution.pdf"))
    forget = fig_forgetting(main_res, probes, os.path.join(FIGDIR, "forgetting.pdf"))
    fig_transfer(hold, os.path.join(FIGDIR, "transfer.pdf"))

    write_outputs(main_res, acq, hold, forget)
    with open(RESULTS, "w") as fh:
        json.dump({"main": {"|".join(map(str, k)): _strip(v) for k, v in main_res.items()},
                   "acq": {"|".join(map(str, k)): v for k, v in acq.items()},
                   "hold": {"|".join(map(str, k)): v for k, v in hold.items()}},
                  fh, indent=1, default=float)
    print(f"Done. See {REPORT}, {FIGDIR}/ and {GENDIR}/", flush=True)


if __name__ == "__main__":
    main()
