package gateway_test

import (
	"testing"

	"github.com/changkun/trustcalib/featurizer"
	"github.com/changkun/trustcalib/featurizer/trustcalib"
	"github.com/changkun/trustcalib/gateway"
	"github.com/changkun/trustcalib/gp"
	"github.com/changkun/trustcalib/kernel"
)

func TestDecisionString(t *testing.T) {
	cases := map[gateway.Decision]string{
		gateway.Allow:        "allow",
		gateway.Block:        "block",
		gateway.Ask:          "ask",
		gateway.Decision(99): "ask",
	}
	for d, want := range cases {
		if d.String() != want {
			t.Errorf("Decision(%d).String() = %q, want %q", d, d.String(), want)
		}
	}
}

func safePoint(i int) featurizer.Point {
	return featurizer.Point{Tool: "read_file", Target: "workspace_src", Task: "bugfix", T: float64(i)}
}

func TestNumLabelsAndCaps(t *testing.T) {
	cfg := gateway.DefaultConfig()
	cfg.RefitEvery = 1
	cfg.MaxTrain = 3
	cfg.MaxHist = 3
	g := gateway.New(trustcalib.New(), gp.NewLaplaceGPC(kernel.DefaultKernel()), cfg)

	for i := 0; i < 6; i++ {
		if err := g.Observe(safePoint(i), i%2 == 0); err != nil {
			t.Fatal(err)
		}
	}
	if g.NumLabels() != 3 {
		t.Fatalf("NumLabels = %d, want 3 (capped)", g.NumLabels())
	}
	ph, lab := g.History()
	if len(ph) != 3 || len(lab) != 3 {
		t.Fatalf("history not capped: %d %d", len(ph), len(lab))
	}
}

func TestTuneBelowMinReturnsDefault(t *testing.T) {
	cfg := gateway.DefaultConfig()
	cfg.RefitEvery = 1
	cfg.MinTuneN = 100
	g := gateway.New(trustcalib.New(), gp.NewLaplaceGPC(kernel.DefaultKernel()), cfg)
	for i := 0; i < 5; i++ {
		if err := g.Observe(safePoint(i), true); err != nil {
			t.Fatal(err)
		}
	}
	low, high := g.Tune()
	if low != cfg.TauLow || high != cfg.TauHigh {
		t.Fatalf("below MinTuneN should return defaults, got (%v,%v)", low, high)
	}
	if g.Tuned() {
		t.Error("Tuned() should be false below MinTuneN")
	}
}

func TestDecideUnknownPointFailsSafe(t *testing.T) {
	cfg := gateway.DefaultConfig()
	cfg.RefitEvery = 1
	g := gateway.New(trustcalib.New(), gp.NewLaplaceGPC(kernel.DefaultKernel()), cfg)
	// Fit on valid points.
	for i := 0; i < 4; i++ {
		if err := g.Observe(safePoint(i), i%2 == 0); err != nil {
			t.Fatal(err)
		}
	}
	if !g.Fitted() {
		t.Fatal("should be fitted")
	}
	d, ph := g.Decide(featurizer.Point{Tool: "unknown_tool", Target: "x", Task: "y", T: 99})
	if d != gateway.Ask || ph != 0.5 {
		t.Fatalf("unknown point should fail safe to ask/0.5, got %v/%v", d, ph)
	}
}

func TestObserveUnknownPoint(t *testing.T) {
	cfg := gateway.DefaultConfig()
	g := gateway.New(trustcalib.New(), gp.NewLaplaceGPC(kernel.DefaultKernel()), cfg)
	// Observing an unknown point still records it (predict fails safe to 0.5);
	// it just cannot be featurized for refit and is skipped there.
	if err := g.Observe(featurizer.Point{Tool: "nope", Target: "x", Task: "y"}, true); err != nil {
		t.Fatal(err)
	}
	if err := g.Observe(safePoint(1), true); err != nil {
		t.Fatal(err)
	}
}
