"""The opaque-judge extension (Section 12): judge model and reliability
estimation. Propositions 6-8 are proved in lean/TrustCalib/Judge.lean."""

import numpy as np

from experiment.data import make_stream
from experiment.gateway import run_gateway
from experiment.judge import JudgePolicy, judge_stream
from experiment.oracle import OracleConfig
from experiment.run_judge import m_calibrated, reliability


def test_judge_stream_is_deterministic_and_blind_spot_allowed():
    s = make_stream(400, seed=1)
    a = judge_stream(s, seed=9)
    b = judge_stream(s, seed=9)
    assert [d.judge_score for d in a] == [d.judge_score for d in b]
    blind = [d for d in a if d.blind]
    assert blind and all(not d.judge_block for d in blind)
    # the gateway sees only [judge output] + category one-hot
    assert all(d.phi_tool.shape == a[0].phi_tool.shape for d in a)


def test_judge_only_policy_never_asks():
    js = judge_stream(make_stream(300, seed=2), seed=3)
    res = run_gateway(js, JudgePolicy(), np.random.default_rng(0), OracleConfig(), 100, 200)
    assert not any(s.queried for s in res.steps)


def test_reliability_needs_audits():
    js = judge_stream(make_stream(600, seed=4), seed=5)
    cfg = OracleConfig()
    no_audit = run_gateway(js, m_calibrated(), np.random.default_rng(0), cfg, 200, 400)
    audit = run_gateway(js, m_calibrated(), np.random.default_rng(0), cfg, 200, 400,
                        audit_rate=0.2)
    assert np.isnan(reliability(no_audit, js, 200, 600)["allow"]["est"])
    est = reliability(audit, js, 200, 600)["allow"]
    assert np.isfinite(est["est"]) and 0 <= est["truth"] <= 1
