"""Regression test for the manuscript's *supported* headline claim
(Section 1): the GP policy gateway auto-decides most actions safely and
accurately while spending far fewer human queries than the always-escalate
status quo.

(The manuscript's Section 5 claim that ASK-band acquisition is more
sample-efficient than random querying is, in this non-stationary setting,
not supported; that negative finding is reported in `report.md` and is not
asserted here because it is empirically false under the Section 6
changepoint.)"""

import numpy as np

from experiment.data import make_stream
from experiment.eval import phase_metrics, total_queries
from experiment.gateway import run_gateway
from experiment.gp import LaplaceGPC
from experiment.kernel import ProductKernel
from experiment.oracle import OracleConfig


def _k():
    return ProductKernel(sigma2=1.6, l_tool=1.1, l_ctx=1.2, lam=200.0)


def test_gateway_reduces_human_burden_safely():
    n, t1, t2 = 900, 360, 650
    auto_rate, acc, fa, q_frac = [], [], [], []
    for s in range(3):
        stream = make_stream(n, seed=1000 + s)
        cfg = OracleConfig(changepoint=470)
        res = run_gateway(
            stream, LaplaceGPC(_k()), np.random.default_rng(s), cfg, t1, t2,
            query_strategy="active",
        )
        m = phase_metrics(res, "val")
        scored = sum(1 for x in res.steps if x.phase in ("val", "test"))
        auto_rate.append(m["auto_rate"])
        acc.append(m["accuracy_auto"])
        fa.append(m["false_allow_rate"])
        q_frac.append(total_queries(res) / scored)

    # Substantial automation ...
    assert np.mean(auto_rate) > 0.5, np.mean(auto_rate)
    # ... that is accurate ...
    assert np.mean(acc) > 0.90, np.mean(acc)
    # ... and safe (bounded false-allow) ...
    assert np.mean(fa) < 0.05, np.mean(fa)
    # ... at far below one-query-per-action (the always-escalate status quo).
    assert np.mean(q_frac) < 0.6, np.mean(q_frac)
