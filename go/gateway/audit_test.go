package gateway_test

import (
	"encoding/json"
	"math"
	"testing"

	"github.com/changkun/trustcalib/featurizer"
	"github.com/changkun/trustcalib/featurizer/trustcalib"
	"github.com/changkun/trustcalib/gateway"
	"github.com/changkun/trustcalib/gp"
	"github.com/changkun/trustcalib/kernel"
)

func riskyPoint(i int) featurizer.Point {
	return featurizer.Point{Tool: "git_force_push", Target: "prod_infra", Task: "ops_maintenance", ArgRisk: 1, T: float64(i)}
}

// trainedGateway returns a gateway that auto-allows safePoint and auto-blocks
// riskyPoint, refitting on every label.
func trainedGateway(t *testing.T, cfg gateway.Config) *gateway.Gateway {
	t.Helper()
	cfg.RefitEvery = 1
	g := gateway.New(trustcalib.New(), gp.NewLaplaceGPC(kernel.DefaultKernel()), cfg)
	for i := 0; i < 20; i++ {
		if err := g.Observe(safePoint(i), true); err != nil {
			t.Fatal(err)
		}
		if err := g.Observe(riskyPoint(i), false); err != nil {
			t.Fatal(err)
		}
	}
	if d, _ := g.Decide(safePoint(20)); d != gateway.Allow {
		t.Fatalf("safe point: %v, want allow", d)
	}
	if d, _ := g.Decide(riskyPoint(20)); d != gateway.Block {
		t.Fatalf("risky point: %v, want block", d)
	}
	return g
}

func TestDecideAuditSampling(t *testing.T) {
	cfg := gateway.DefaultConfig()
	cfg.AuditRate = 0.25
	g := trainedGateway(t, cfg)

	if d, _, audit := g.DecideAudit(safePoint(20), 0.1); d != gateway.Allow || !audit {
		t.Fatalf("u=0.1 < 0.25: got %v audit=%v, want allow audit", d, audit)
	}
	if d, _, audit := g.DecideAudit(safePoint(20), 0.3); d != gateway.Allow || audit {
		t.Fatalf("u=0.3 >= 0.25: got %v audit=%v, want allow without audit", d, audit)
	}
	if d, _, audit := g.DecideAudit(riskyPoint(20), 0.0); d != gateway.Block || !audit {
		t.Fatalf("block u=0: got %v audit=%v, want block audit", d, audit)
	}
	st := g.AuditStats()
	if st.NAllow != 2 || st.NBlock != 1 {
		t.Fatalf("counts: %+v", st)
	}
}

func TestDecideAuditNeverOnAskOrZeroRate(t *testing.T) {
	cfg := gateway.DefaultConfig()
	cfg.AuditRate = 1
	cold := gateway.New(trustcalib.New(), gp.NewLaplaceGPC(kernel.DefaultKernel()), cfg)
	if d, p, audit := cold.DecideAudit(safePoint(0), 0); d != gateway.Ask || p != 0.5 || audit {
		t.Fatalf("cold start: %v %v audit=%v", d, p, audit)
	}
	if st := cold.AuditStats(); st.NAllow != 0 || st.NBlock != 0 {
		t.Fatalf("ASK must not be counted: %+v", st)
	}

	g := trainedGateway(t, gateway.DefaultConfig()) // AuditRate 0
	if _, _, audit := g.DecideAudit(safePoint(20), 0); audit {
		t.Fatal("AuditRate 0 must never audit")
	}
	if g.AuditStats().NAllow != 1 {
		t.Fatal("auto-ALLOW must be counted even without audits")
	}
	if est, _, _ := g.FalseAllowEstimate(); !math.IsNaN(est) {
		t.Fatalf("estimate without audits should be NaN, got %v", est)
	}
	if err := g.ObserveAudit(safePoint(20), true, gateway.Allow); err == nil {
		t.Fatal("ObserveAudit with AuditRate 0 should fail")
	}
}

// TestFalseAllowEstimate: the Horvitz-Thompson estimate is
// (denied audits / AuditRate) / N_allow, and audit labels carry their
// propensity, train the model and stay out of the tuning history.
func TestFalseAllowEstimate(t *testing.T) {
	cfg := gateway.DefaultConfig()
	cfg.AuditRate = 0.25
	g := trainedGateway(t, cfg)
	if est, n, k := g.FalseAllowEstimate(); !math.IsNaN(est) || n != 0 || k != 0 {
		t.Fatalf("no allows yet: %v %d %d", est, n, k)
	}

	for i := 0; i < 40; i++ {
		g.DecideAudit(safePoint(20+i), 0.9)
	}
	labels := g.NumLabels()
	hist, _ := g.History()
	for i, approved := range []bool{false, true, false} {
		if err := g.ObserveAudit(safePoint(60+i), approved, gateway.Allow); err != nil {
			t.Fatal(err)
		}
	}
	if err := g.ObserveAudit(riskyPoint(63), true, gateway.Block); err != nil {
		t.Fatal(err)
	}
	if err := g.ObserveAudit(safePoint(64), true, gateway.Ask); err == nil {
		t.Fatal("ObserveAudit of an ASK should fail")
	}

	est, n, k := g.FalseAllowEstimate()
	if n != 3 || k != 2 {
		t.Fatalf("audits=%d denied=%d, want 3, 2", n, k)
	}
	if want := (2 / 0.25) / 40.0; math.Abs(est-want) > 1e-12 {
		t.Fatalf("estimate %v, want %v", est, want)
	}
	if st := g.AuditStats(); st.CleanStreak != 0 || st.AuditsBlock != 1 || st.ApprovedBlock != 1 {
		t.Fatalf("stats: %+v", st)
	}
	if est, _, _ := g.FalseBlockEstimate(); !math.IsNaN(est) {
		// No DecideAudit of the risky point was counted, so N_block = 0.
		t.Fatalf("false-block estimate with N_block = 0 should be NaN, got %v", est)
	}

	if g.NumLabels() != labels+4 {
		t.Fatalf("audit labels should train the model: %d, want %d", g.NumLabels(), labels+4)
	}
	if h, _ := g.History(); len(h) != len(hist) {
		t.Fatalf("audit labels must not enter the tuning history: %d, want %d", len(h), len(hist))
	}
	info := g.Labels()
	if len(info) != g.NumLabels() {
		t.Fatalf("labels not parallel: %d vs %d", len(info), g.NumLabels())
	}
	want := []gateway.LabelInfo{
		{Source: gateway.FromAsk, Propensity: 1},
		{Source: gateway.FromAuditAllow, Propensity: 0.25},
		{Source: gateway.FromAuditAllow, Propensity: 0.25},
		{Source: gateway.FromAuditAllow, Propensity: 0.25},
		{Source: gateway.FromAuditBlock, Propensity: 0.25},
	}
	if got := info[len(info)-5:]; !equalInfo(got, want) {
		t.Fatalf("label provenance %+v, want %+v", got, want)
	}
}

func equalInfo(a, b []gateway.LabelInfo) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestFalseBlockEstimate(t *testing.T) {
	cfg := gateway.DefaultConfig()
	cfg.AuditRate = 0.5
	g := trainedGateway(t, cfg)
	for i := 0; i < 10; i++ {
		g.DecideAudit(riskyPoint(20+i), 0.9)
	}
	if err := g.ObserveAudit(riskyPoint(30), true, gateway.Block); err != nil {
		t.Fatal(err)
	}
	if err := g.ObserveAudit(riskyPoint(31), false, gateway.Block); err != nil {
		t.Fatal(err)
	}
	est, n, k := g.FalseBlockEstimate()
	if n != 2 || k != 1 || math.Abs(est-(1/0.5)/10) > 1e-12 {
		t.Fatalf("false-block estimate %v (audits %d approved %d)", est, n, k)
	}
}

// TestCertificationSize checks Proposition 5(b): ceil(ln(1/delta)/alpha).
func TestCertificationSize(t *testing.T) {
	cases := []struct {
		alpha, delta float64
		n            int
	}{
		{0.01, 0.05, 300},
		{0.25, 0.05, 12},
		{0.05, 0.05, 60},
		{0, 0.05, 0},
		{0.01, 1, 0},
		{1.5, 0.05, 0},
	}
	for _, c := range cases {
		if got := gateway.CertificationSize(c.alpha, c.delta); got != c.n {
			t.Errorf("CertificationSize(%v, %v) = %d, want %d", c.alpha, c.delta, got, c.n)
		}
		if c.n > 0 && math.Pow(1-c.alpha, float64(c.n)) > c.delta {
			t.Errorf("(1-alpha)^n = %v > delta", math.Pow(1-c.alpha, float64(c.n)))
		}
	}
}

// TestCertifiedFalseAllowBelow: the certificate needs n >= ln(1/delta)/alpha
// consecutive clean audits of auto-ALLOWs, and a denied audit restarts it.
func TestCertifiedFalseAllowBelow(t *testing.T) {
	cfg := gateway.DefaultConfig()
	cfg.AuditRate = 0.25
	g := trainedGateway(t, cfg)

	g.RestoreAudit(gateway.AuditStats{CleanStreak: 299}, g.Labels())
	if g.CertifiedFalseAllowBelow(0.01, 0.05) {
		t.Fatal("299 clean audits must not certify alpha=0.01 at delta=0.05")
	}
	if err := g.ObserveAudit(safePoint(21), true, gateway.Allow); err != nil {
		t.Fatal(err)
	}
	if !g.CertifiedFalseAllowBelow(0.01, 0.05) {
		t.Fatal("300 clean audits should certify alpha=0.01 at delta=0.05")
	}
	if g.CertifiedFalseAllowBelow(0, 0.05) || g.CertifiedFalseAllowBelow(0.01, 0) {
		t.Fatal("invalid parameters must not certify")
	}
	if err := g.ObserveAudit(safePoint(22), false, gateway.Allow); err != nil {
		t.Fatal(err)
	}
	if g.CertifiedFalseAllowBelow(0.25, 0.05) || g.AuditStats().CleanStreak != 0 {
		t.Fatal("a denied audit must reset the clean streak")
	}
	// Audits of auto-BLOCKs do not touch the false-allow streak.
	for i := 0; i < 12; i++ {
		if err := g.ObserveAudit(safePoint(23+i), true, gateway.Allow); err != nil {
			t.Fatal(err)
		}
	}
	if err := g.ObserveAudit(riskyPoint(40), false, gateway.Block); err != nil {
		t.Fatal(err)
	}
	if !g.CertifiedFalseAllowBelow(0.25, 0.05) {
		t.Fatal("12 clean audits should certify alpha=0.25 at delta=0.05")
	}
}

// TestCostsGateway: with Costs set the gateway uses the cost band from the
// start, Tune returns it unchanged, and Restore ignores persisted thresholds.
func TestCostsGateway(t *testing.T) {
	cfg := gateway.DefaultConfig()
	cfg.Costs = &gateway.Costs{FalseAllow: 10, FalseBlock: 4, Ask: 1}
	cfg.MinTuneN = 1
	g := gateway.New(trustcalib.New(), gp.NewLaplaceGPC(kernel.DefaultKernel()), cfg)
	if lo, hi := g.Thresholds(); math.Abs(lo-0.25) > 1e-12 || math.Abs(hi-0.9) > 1e-12 {
		t.Fatalf("initial thresholds (%v, %v)", lo, hi)
	}
	for i := 0; i < 40; i++ {
		if err := g.Observe(safePoint(i), true); err != nil {
			t.Fatal(err)
		}
		if err := g.Observe(riskyPoint(i), false); err != nil {
			t.Fatal(err)
		}
	}
	if lo, hi := g.Tune(); math.Abs(lo-0.25) > 1e-12 || math.Abs(hi-0.9) > 1e-12 {
		t.Fatalf("Tune with costs = (%v, %v), want (0.25, 0.9)", lo, hi)
	}
	if g.Tuned() {
		t.Error("cost-derived thresholds are specified, not tuned")
	}

	pts, y := g.TrainingData()
	ph, lab := g.History()
	if err := g.Restore(pts, y, ph, lab, 0.1, 0.2, true); err != nil {
		t.Fatal(err)
	}
	if lo, hi := g.Thresholds(); math.Abs(lo-0.25) > 1e-12 || math.Abs(hi-0.9) > 1e-12 {
		t.Fatalf("Restore should keep the cost band, got (%v, %v)", lo, hi)
	}
}

func TestLabelsParallelUnderCaps(t *testing.T) {
	cfg := gateway.DefaultConfig()
	cfg.AuditRate = 0.5
	cfg.MaxTrain = 5
	g := trainedGateway(t, cfg)
	if err := g.ObserveAudit(safePoint(30), true, gateway.Allow); err != nil {
		t.Fatal(err)
	}
	info := g.Labels()
	if len(info) != 5 || g.NumLabels() != 5 {
		t.Fatalf("caps: %d labels, %d info", g.NumLabels(), len(info))
	}
	if info[4].Source != gateway.FromAuditAllow {
		t.Fatalf("last label should be the audit, got %+v", info[4])
	}

	// RestoreAudit with mismatched provenance falls back to ask/1.
	g.RestoreAudit(gateway.AuditStats{NAllow: 3}, nil)
	for _, li := range g.Labels() {
		if li.Source != gateway.FromAsk || li.Propensity != 1 {
			t.Fatalf("fallback provenance %+v", li)
		}
	}
	if g.AuditStats().NAllow != 3 {
		t.Fatal("RestoreAudit should reinstate counters")
	}
}

func TestLabelSourceText(t *testing.T) {
	for _, s := range []gateway.LabelSource{gateway.FromAsk, gateway.FromAuditAllow, gateway.FromAuditBlock} {
		b, err := json.Marshal(gateway.LabelInfo{Source: s, Propensity: 1})
		if err != nil {
			t.Fatal(err)
		}
		var back gateway.LabelInfo
		if err := json.Unmarshal(b, &back); err != nil || back.Source != s {
			t.Fatalf("round trip %v via %s: %v %v", s, b, back.Source, err)
		}
	}
	var s gateway.LabelSource
	if err := s.UnmarshalText([]byte("bogus")); err == nil {
		t.Fatal("expected error for unknown source")
	}
}

// An audit answered after AuditRate changed must keep the propensity with
// which it was sampled: the Horvitz-Thompson weight is 1/propensity.
func TestObserveAuditWithPropensity(t *testing.T) {
	cfg := gateway.DefaultConfig()
	cfg.AuditRate = 0.1
	g := trainedGateway(t, cfg)
	if d, _, audit := g.DecideAudit(safePoint(20), 0.0); d != gateway.Allow || !audit {
		t.Fatalf("got %v audit=%v", d, audit)
	}
	if err := g.ObserveAuditWithPropensity(safePoint(21), false, gateway.Allow, 0.5); err != nil {
		t.Fatal(err)
	}
	labels := g.Labels()
	last := labels[len(labels)-1]
	if last.Source != gateway.FromAuditAllow || last.Propensity != 0.5 {
		t.Fatalf("last label = %+v, want audit_allow with propensity 0.5", last)
	}
	// one auto-ALLOW counted, one denied audit with weight 1/0.5 = 2
	if est, audits, denied := g.FalseAllowEstimate(); est != 2 || audits != 1 || denied != 1 {
		t.Fatalf("estimate = %v (%d audits, %d denied), want 2 (1, 1)", est, audits, denied)
	}
	if err := g.ObserveAuditWithPropensity(safePoint(22), true, gateway.Allow, 0); err == nil {
		t.Fatal("propensity 0 should be rejected")
	}
}
