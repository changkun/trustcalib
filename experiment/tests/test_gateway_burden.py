"""Regression test for the headline operational claim (Experiments): with the
additive kernel and the cost-derived symmetric band, the gateway auto-decides
most actions accurately and safely while spending far fewer human labels than
the always-escalate status quo, and fewer than the separable product kernel."""

import numpy as np

from experiment.data import make_stream
from experiment.eval import phase_metrics, scored_queries
from experiment.gateway import SYMMETRIC, run_gateway
from experiment.gp import LaplaceGPC
from experiment.kernel import AdditiveKernel, ProductKernel
from experiment.oracle import OracleConfig


def _run(kernel, seeds=range(3)):
    n, t_warm, t_late = 900, 360, 650
    out = []
    for s in seeds:
        stream = make_stream(n, seed=1000 + s)
        cfg = OracleConfig(changepoint=470)
        res = run_gateway(stream, LaplaceGPC(kernel), np.random.default_rng(s), cfg,
                          t_warm, t_late, band=SYMMETRIC.band)
        m = phase_metrics(res)
        q, steps = scored_queries(res)
        out.append((m["auto_rate"], m["accuracy_auto"], m["false_allow_rate"], q / steps))
    return np.mean(np.array(out), axis=0)


def test_additive_gateway_reduces_human_burden_safely():
    auto, acc, fa, qfrac = _run(AdditiveKernel(1.6, 1.0, 0.6, lam=90.0))
    assert auto > 0.7, auto        # substantial automation ...
    assert acc > 0.90, acc         # ... that is accurate ...
    assert fa < 0.05, fa           # ... and safe (bounded false-allow) ...
    assert qfrac < 0.3, qfrac      # ... at far below one label per action.


def test_additive_needs_fewer_labels_than_product():
    *_, q_add = _run(AdditiveKernel(1.6, 1.0, 0.6, lam=90.0))
    *_, q_prod = _run(ProductKernel(1.6, 1.1, 1.2, lam=90.0))
    assert q_add < q_prod, (q_add, q_prod)
