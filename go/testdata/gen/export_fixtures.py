"""Export golden fixtures from the Python reference for the Go port.

Run from the repository root:

    PYTHONPATH=. uv run python go/testdata/gen/export_fixtures.py

It imports only experiment.kernel / experiment.gp (no oracle, no matplotlib,
no stream generator) and builds a Packed directly from a fixed, hardcoded
feature matrix, so the fixtures are independent of the RNG-driven simulation.
The generated arrays are written verbatim to JSON, so the Go tests never need
to reproduce NumPy's RNG -- they only re-run the deterministic kernel/GP math.
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy.special import log_ndtr
from scipy.stats import norm

from experiment.kernel import Packed, ProductKernel
from experiment.gp import LaplaceGPC
from experiment.data import make_stream
from experiment.oracle import OracleConfig, oracle_decision, sample_label

OUT = os.path.join(os.path.dirname(__file__), "..", "fixtures")

# Kernel hyperparameters pinned into every fixture (Go reads these rather than
# relying on its own defaults).
SIGMA2, L_TOOL, L_CTX, LAM = 1.6, 1.1, 1.2, 200.0
D_TOOL, D_CTX = 11, 9


def make_packed(rng: np.random.Generator, n: int, t0: int) -> Packed:
    phi_tool = rng.random((n, D_TOOL))
    phi_ctx = rng.random((n, D_CTX))
    t = np.arange(t0, t0 + n, dtype=float)
    tool_id = rng.integers(0, 18, size=n)
    return Packed(phi_tool=phi_tool, phi_ctx=phi_ctx, t=t, tool_id=tool_id)


def packed_dict(p: Packed) -> dict:
    return {
        "phi_tool": p.phi_tool.tolist(),
        "phi_ctx": p.phi_ctx.tolist(),
        "t": p.t.tolist(),
    }


def kernel_params() -> dict:
    return {"sigma2": SIGMA2, "l_tool": L_TOOL, "l_ctx": L_CTX, "lam": LAM}


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(20260526)
    k = ProductKernel(sigma2=SIGMA2, l_tool=L_TOOL, l_ctx=L_CTX, lam=LAM)

    n_train = 9
    train = make_packed(rng, n_train, t0=0)
    # Labels with structure (not all one class) so the GP is non-degenerate.
    score = train.phi_ctx[:, 0] - 0.5 + 0.3 * (train.phi_tool[:, 0] - 0.5)
    y01 = (score > 0).astype(int)
    if y01.sum() in (0, n_train):  # guarantee a mix
        y01[0], y01[-1] = 0, 1

    # 1) kernel matrix
    K = k.full(train)
    dump(
        "kernel_full.json",
        {"kernel": kernel_params(), "train": packed_dict(train), "K": K.tolist()},
    )

    # 2) Laplace fit
    m = LaplaceGPC(k)
    m.fit(train, y01)
    dump(
        "laplace_fit.json",
        {
            "kernel": kernel_params(),
            "train": packed_dict(train),
            "y01": y01.tolist(),
            "f_hat": m._f_hat.tolist(),
            "grad": m._grad.tolist(),
            "log_marginal": float(m.log_marginal),
        },
    )

    # 3) prediction
    query = make_packed(rng, 6, t0=3)
    f_bar, var, pi = m.predict(query)
    dump(
        "predict.json",
        {
            "kernel": kernel_params(),
            "train": packed_dict(train),
            "y01": y01.tolist(),
            "query": packed_dict(query),
            "f_bar": f_bar.tolist(),
            "var": var.tolist(),
            "pi": pi.tolist(),
        },
    )

    # 4) stable logCDF / logPDF over a wide grid, including the deep tail
    z = np.concatenate(
        [
            np.linspace(-40.0, -10.0, 31),
            np.linspace(-10.0, 10.0, 81),
        ]
    )
    dump(
        "logcdf.json",
        {
            "z": z.tolist(),
            "log_cdf": log_ndtr(z).tolist(),
            "log_pdf": norm.logpdf(z).tolist(),
        },
    )

    # 5) replay trajectory for the gateway burden test. A label is pre-sampled
    # for every stream point (used by Go only when it chooses to observe), along
    # with the oracle's Bayes-optimal decision for scoring. No oracle code is
    # ported to Go; only this recorded trajectory.
    export_trajectory()

    print("wrote fixtures to", os.path.normpath(OUT))


def export_trajectory() -> None:
    n = 1500
    t1, t2 = 560, 1050
    stream = make_stream(n, seed=1234)
    cfg = OracleConfig(changepoint=750)
    rng = np.random.default_rng(0)
    pts = []
    for dp in stream:
        pts.append(
            {
                "tool": dp.tool.name,
                "target": dp.target,
                "task": dp.task,
                "arg_risk": int(dp.arg_risk),
                "t": int(dp.t),
                "label": int(sample_label(dp, rng, cfg)),
                "oracle_yes": int(oracle_decision(dp, cfg)),
            }
        )
    dump("trajectory.json", {"t1": t1, "t2": t2, "points": pts})


def dump(name: str, obj: dict) -> None:
    with open(os.path.join(OUT, name), "w") as fh:
        json.dump(obj, fh, indent=2)


if __name__ == "__main__":
    main()
