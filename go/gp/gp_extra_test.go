package gp

import (
	"math"
	"testing"

	"github.com/changkun/trustcalib/kernel"
)

func TestFitEmptyAndMismatch(t *testing.T) {
	m := NewLaplaceGPC(kernel.DefaultKernel())
	if err := m.Fit(kernel.Packed{}, nil); err == nil {
		t.Error("expected error on empty training set")
	}
	p := packed1D([]float64{0.1, 0.2, 0.3})
	if err := m.Fit(p, []int{1, 0}); err == nil {
		t.Error("expected error on label/point count mismatch")
	}
}

func TestPredictBeforeFit(t *testing.T) {
	m := NewLaplaceGPC(kernel.DefaultKernel())
	if _, _, _, err := m.Predict(packed1D([]float64{0.1})); err == nil {
		t.Error("expected error predicting before fit")
	}
	if _, err := m.PredictProb(packed1D([]float64{0.1})); err == nil {
		t.Error("expected error in PredictProb before fit")
	}
}

func TestLogMarginalFinite(t *testing.T) {
	m := NewLaplaceGPC(kernel.DefaultKernel())
	if err := m.Fit(packed1D([]float64{-1, -0.5, 0.5, 1}), []int{0, 0, 1, 1}); err != nil {
		t.Fatal(err)
	}
	if lm := m.LogMarginal(); math.IsNaN(lm) || math.IsInf(lm, 0) {
		t.Fatalf("log marginal not finite: %v", lm)
	}
}

func TestProbitDerivsExtremes(t *testing.T) {
	// Confident-correct (large positive z) and confident-wrong (large negative
	// z) both stay finite, exercising the Mills-ratio and W floor branches.
	_, grad, w := ProbitDerivs([]float64{50, -50}, []float64{1, 1})
	for i := range grad {
		if math.IsNaN(grad[i]) || math.IsInf(grad[i], 0) {
			t.Fatalf("grad[%d] not finite: %v", i, grad[i])
		}
		if w[i] < 1e-10 {
			t.Fatalf("W[%d] below floor: %v", i, w[i])
		}
	}
}
