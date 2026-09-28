package persist_test

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/changkun/trustcalib/config"
	"github.com/changkun/trustcalib/featurizer"
	"github.com/changkun/trustcalib/featurizer/trustcalib"
	"github.com/changkun/trustcalib/gateway"
	"github.com/changkun/trustcalib/persist"
)

// preAuditState is a state file as written before label provenance, audits and
// the kernel, cost and audit config keys existed.
const preAuditState = `{
  "version": 1,
  "config": {
    "Kernel": {"Sigma2": 1.6, "LTool": 1.1, "LCtx": 1.2, "Lambda": 200},
    "GP": {"Jitter": 1e-06, "MaxIter": 100, "Tol": 1e-06},
    "Gateway": {"TauLow": 0.35, "TauHigh": 0.65, "RefitEvery": 8, "SafetyEps": 0.02,
                "BlockEps": 0.05, "MaxTrain": 2000, "MaxHist": 2000, "MinTuneN": 30}
  },
  "tau_low": 0.3,
  "tau_high": 0.7,
  "tuned": true,
  "counter": 3,
  "points": [
    {"tool": "read_file", "target": "workspace_src", "task": "bugfix", "arg_risk": 0, "t": 0},
    {"tool": "git_force_push", "target": "prod_infra", "task": "ops_maintenance", "arg_risk": 1, "t": 1},
    {"tool": "read_file", "target": "workspace_src", "task": "bugfix", "arg_risk": 0, "t": 2}
  ],
  "labels": [1, 0, 1],
  "phat_hist": [0.5, 0.5, 0.6],
  "label_hist": [1, 0, 1]
}`

func TestLoadPreAuditStateBackwardCompatible(t *testing.T) {
	path := filepath.Join(t.TempDir(), "state.json")
	if err := os.WriteFile(path, []byte(preAuditState), 0o644); err != nil {
		t.Fatal(err)
	}
	st, ok, err := persist.Load(path, config.Default())
	if err != nil || !ok {
		t.Fatalf("load: ok=%v err=%v", ok, err)
	}
	c := st.Config
	if c.Kernel.Type != config.KernelProduct || c.Featurizer != config.FeaturizerTaxonomy {
		t.Errorf("selectors should default: %q %q", c.Kernel.Type, c.Featurizer)
	}
	if c.AdditiveKernel() != config.Default().AdditiveKernel() {
		t.Errorf("additive block should default: %+v", c.Kernel.Additive)
	}
	if a, d := c.Certify(); a != 0.01 || d != 0.05 {
		t.Errorf("certify params should default: (%v, %v)", a, d)
	}
	if c.Gateway.Costs != nil || c.Gateway.AuditRate != 0 {
		t.Errorf("costs/audits should be off: %+v", c.Gateway)
	}

	g, err := persist.Reconstruct(st, trustcalib.New())
	if err != nil {
		t.Fatal(err)
	}
	if !g.Fitted() || g.NumLabels() != 3 {
		t.Fatalf("fitted=%v labels=%d", g.Fitted(), g.NumLabels())
	}
	if lo, hi := g.Thresholds(); lo != 0.3 || hi != 0.7 {
		t.Errorf("persisted thresholds lost: (%v, %v)", lo, hi)
	}
	for _, li := range g.Labels() {
		if li.Source != gateway.FromAsk || li.Propensity != 1 {
			t.Fatalf("pre-audit labels should be escalations with propensity 1: %+v", li)
		}
	}
	if g.AuditStats() != (gateway.AuditStats{}) {
		t.Errorf("audit counters should start at zero: %+v", g.AuditStats())
	}
}

func TestAuditStateRoundTrip(t *testing.T) {
	cfg := config.Default()
	cfg.Gateway.RefitEvery = 1
	cfg.Gateway.AuditRate = 0.5
	g := cfg.NewGateway(trustcalib.New())
	safe := featurizer.Point{Tool: "read_file", Target: "workspace_src", Task: "bugfix"}
	risky := featurizer.Point{Tool: "git_force_push", Target: "prod_infra", Task: "ops_maintenance", ArgRisk: 1}
	for i := 0; i < 10; i++ {
		safe.T, risky.T = float64(i), float64(i)
		if err := g.Observe(safe, true); err != nil {
			t.Fatal(err)
		}
		if err := g.Observe(risky, false); err != nil {
			t.Fatal(err)
		}
	}
	if d, _, audit := g.DecideAudit(safe, 0.1); d != gateway.Allow || !audit {
		t.Fatalf("expected an audited allow, got %v audit=%v", d, audit)
	}
	if err := g.ObserveAudit(safe, false, gateway.Allow); err != nil {
		t.Fatal(err)
	}

	st := persist.NewState(cfg).Snapshot(g)
	st.SetPending("k1", "block")
	path := filepath.Join(t.TempDir(), "state.json")
	if err := persist.Save(path, st); err != nil {
		t.Fatal(err)
	}
	loaded, _, err := persist.Load(path, cfg)
	if err != nil {
		t.Fatal(err)
	}
	g2, err := persist.Reconstruct(loaded, trustcalib.New())
	if err != nil {
		t.Fatal(err)
	}
	if g2.AuditStats() != g.AuditStats() {
		t.Errorf("audit stats %+v, want %+v", g2.AuditStats(), g.AuditStats())
	}
	l1, l2 := g.Labels(), g2.Labels()
	if len(l1) != len(l2) {
		t.Fatalf("label info length %d, want %d", len(l2), len(l1))
	}
	for i := range l1 {
		if l1[i] != l2[i] {
			t.Fatalf("label info[%d] = %+v, want %+v", i, l2[i], l1[i])
		}
	}
	if l2[len(l2)-1].Source != gateway.FromAuditAllow || l2[len(l2)-1].Propensity != 0.5 {
		t.Errorf("audit provenance lost: %+v", l2[len(l2)-1])
	}
	if dec, ok := loaded.TakePending("k1"); !ok || dec != "block" {
		t.Errorf("pending audit lost: %q %v", dec, ok)
	}
}

func TestPendingAudits(t *testing.T) {
	var st persist.State
	if st.SetPending("a", "") {
		t.Error("clearing an absent key should not report a change")
	}
	if !st.SetPending("a", "allow") || !st.SetPending("b", "block") {
		t.Error("recording should report a change")
	}
	// A later decide of the same request supersedes its pending audit.
	if !st.SetPending("a", "") {
		t.Error("superseding should report a change")
	}
	if _, ok := st.TakePending("a"); ok {
		t.Error("superseded audit should be gone")
	}
	if dec, ok := st.TakePending("b"); !ok || dec != "block" {
		t.Errorf("TakePending(b) = %q, %v", dec, ok)
	}
	if _, ok := st.TakePending("b"); ok || st.Pending != nil {
		t.Errorf("TakePending should consume: %+v", st.Pending)
	}

	for i := 0; i < persist.MaxPending+5; i++ {
		st.SetPending(fmt.Sprint(i), "allow")
	}
	if len(st.Pending) != persist.MaxPending {
		t.Fatalf("pending not capped: %d", len(st.Pending))
	}
	if _, ok := st.TakePending("0"); ok {
		t.Error("oldest entries should be dropped first")
	}
	if _, ok := st.TakePending(fmt.Sprint(persist.MaxPending + 4)); !ok {
		t.Error("newest entry should be kept")
	}
}

func TestNewStateUsesCostBand(t *testing.T) {
	cfg := config.Default()
	cfg.Gateway.Costs = &config.CostsCfg{FalseAllow: 10, FalseBlock: 4, Ask: 1}
	st := persist.NewState(cfg)
	if st.TauLow != 0.25 || st.TauHigh != 0.9 {
		t.Fatalf("cold-start thresholds (%v, %v), want the cost band", st.TauLow, st.TauHigh)
	}
}

func TestPendingAuditPropensityRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "state.json")
	cfg := config.Default()
	st := persist.NewState(cfg)
	st.SetPendingWithPropensity("k", "allow", 0.05)
	if err := persist.Save(path, st); err != nil {
		t.Fatal(err)
	}
	loaded, _, err := persist.Load(path, cfg)
	if err != nil {
		t.Fatal(err)
	}
	p, ok := loaded.TakePendingAudit("k")
	if !ok || p.Decision != "allow" || p.Propensity != 0.05 {
		t.Fatalf("TakePendingAudit = %+v, %v", p, ok)
	}
}
