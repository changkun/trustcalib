package gateway

import (
	"math"
	"math/rand"
	"testing"
)

func approx(a, b float64) bool { return math.Abs(a-b) < 1e-12 }

// TestCostsClosedForm ports test_chow.py: tau_low = c_ask/c_FB and
// tau_high = 1 - c_ask/c_FA.
func TestCostsClosedForm(t *testing.T) {
	c := Costs{FalseAllow: 10, FalseBlock: 4, Ask: 1}
	if err := c.Validate(); err != nil {
		t.Fatal(err)
	}
	if !approx(c.TauLow(), 0.25) || !approx(c.TauHigh(), 0.9) {
		t.Fatalf("band = (%v, %v), want (0.25, 0.9)", c.TauLow(), c.TauHigh())
	}
	if !c.BandNonEmpty() {
		t.Fatal("band (0.25, 0.9) should be non-empty")
	}
	if lo, hi := c.Band(); !approx(lo, 0.25) || !approx(hi, 0.9) {
		t.Fatalf("Band() = (%v, %v)", lo, hi)
	}
}

// TestCostsSymmetricIsV1Band: the v1 band (0.35, 0.65) is the Chow rule with
// symmetric costs c_FA = c_FB = c_ask/0.35.
func TestCostsSymmetricIsV1Band(t *testing.T) {
	for _, c := range []Costs{SymmetricCosts(0.35), {FalseAllow: 1 / 0.35, FalseBlock: 1 / 0.35, Ask: 1}} {
		lo, hi := c.Band()
		if !approx(lo, 0.35) || !approx(hi, 0.65) {
			t.Fatalf("symmetric band = (%v, %v), want (0.35, 0.65)", lo, hi)
		}
	}
}

// TestRuleMinimizesExpectedLoss ports test_chow.py: on a fine grid of p
// (including values outside [0, 1]) the three-tier rule with the cost band
// picks a decision of minimal expected loss (Proposition 2).
func TestRuleMinimizesExpectedLoss(t *testing.T) {
	for _, c := range []Costs{SymmetricCosts(0.35), {10, 4, 1}, {3, 7, 0.5}} {
		lo, hi := c.Band()
		for i := 0; i <= 1400; i++ {
			p := -0.2 + 1.4*float64(i)/1400
			d := decision(p, lo, hi)
			best := math.Min(c.Loss(Allow, p), math.Min(c.Loss(Block, p), c.Loss(Ask, p)))
			if c.Loss(d, p) > best+1e-12 {
				t.Fatalf("costs %+v, p=%v: %v has loss %v > best %v", c, p, d, c.Loss(d, p), best)
			}
		}
	}
}

// TestEmptyBandCollapses: when c_ask (c_FA + c_FB) >= c_FA c_FB the closed
// forms cross; Band collapses to the allow/block indifference point, so the
// rule never asks and still minimizes expected loss away from that point.
func TestEmptyBandCollapses(t *testing.T) {
	c := Costs{FalseAllow: 1.5, FalseBlock: 2, Ask: 1}
	if c.BandNonEmpty() {
		t.Fatal("band should be empty")
	}
	if !(c.TauLow() > c.TauHigh()) {
		t.Fatalf("closed forms should cross: (%v, %v)", c.TauLow(), c.TauHigh())
	}
	lo, hi := c.Band()
	m := 1.5 / 3.5
	if !approx(lo, m) || !approx(hi, m) {
		t.Fatalf("Band() = (%v, %v), want (%v, %v)", lo, hi, m, m)
	}
	for i := 0; i <= 1000; i++ {
		p := float64(i) / 1000
		if math.Abs(p-m) < 1e-9 {
			continue
		}
		d := decision(p, lo, hi)
		if d == Ask {
			t.Fatalf("p=%v: empty band should never ask", p)
		}
		best := math.Min(c.Loss(Allow, p), math.Min(c.Loss(Block, p), c.Loss(Ask, p)))
		if c.Loss(d, p) > best+1e-12 {
			t.Fatalf("p=%v: %v not loss-minimal", p, d)
		}
	}
}

func TestCostsValidate(t *testing.T) {
	bad := []Costs{
		{},
		{FalseAllow: 1, FalseBlock: 1, Ask: 0},
		{FalseAllow: -1, FalseBlock: 1, Ask: 1},
		{FalseAllow: math.NaN(), FalseBlock: 1, Ask: 1},
		{FalseAllow: math.Inf(1), FalseBlock: 1, Ask: 1},
	}
	for _, c := range bad {
		if c.Validate() == nil {
			t.Errorf("Validate(%+v) = nil, want error", c)
		}
	}
}

// TestConfigThresholds: valid costs override TauLow/TauHigh; invalid costs are
// ignored.
func TestConfigThresholds(t *testing.T) {
	cfg := DefaultConfig()
	if lo, hi := cfg.Thresholds(); lo != 0.35 || hi != 0.65 {
		t.Fatalf("default thresholds (%v, %v)", lo, hi)
	}
	cfg.Costs = &Costs{FalseAllow: 10, FalseBlock: 4, Ask: 1}
	if lo, hi := cfg.Thresholds(); !approx(lo, 0.25) || !approx(hi, 0.9) {
		t.Fatalf("cost thresholds (%v, %v)", lo, hi)
	}
	cfg.Costs = &Costs{FalseAllow: 10, FalseBlock: 0, Ask: 1}
	if lo, hi := cfg.Thresholds(); lo != 0.35 || hi != 0.65 {
		t.Fatalf("invalid costs should fall back, got (%v, %v)", lo, hi)
	}
}

// TestTuneDegeneratesOnEscalatedHistory ports test_chow.py: fed only (p_hat,
// label) pairs recorded at ASK time, every p_hat is inside the band and the v1
// grid search returns the default band.
func TestTuneDegeneratesOnEscalatedHistory(t *testing.T) {
	rng := rand.New(rand.NewSource(0))
	pHat := make([]float64, 500)
	label := make([]int, 500)
	for i := range pHat {
		pHat[i] = 0.35 + 0.3*rng.Float64()
		if rng.Float64() < pHat[i] {
			label[i] = 1
		}
	}
	if lo, hi := TuneThresholds(pHat, label, 0.02, 0.05); lo != 0.35 || hi != 0.65 {
		t.Fatalf("expected degenerate default band, got (%v, %v)", lo, hi)
	}
}
