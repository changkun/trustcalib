package gp

import (
	"math"
	"testing"

	"github.com/changkun/trustcalib/internal/testutil"
)

// TestLogCDFGolden checks the stable logCDF/logPDF against scipy across a wide
// grid including the deep negative tail (down to z = -40).
func TestLogCDFGolden(t *testing.T) {
	var fix struct {
		Z      []float64 `json:"z"`
		LogCDF []float64 `json:"log_cdf"`
		LogPDF []float64 `json:"log_pdf"`
	}
	testutil.Load(t, "logcdf.json", &fix)

	for i, z := range fix.Z {
		if got := logCDF(z); math.Abs(got-fix.LogCDF[i]) > 1e-10 {
			t.Errorf("logCDF(%g) = %.15g, want %.15g (diff %.3g)", z, got, fix.LogCDF[i], got-fix.LogCDF[i])
		}
		if got := logPDF(z); math.Abs(got-fix.LogPDF[i]) > 1e-12 {
			t.Errorf("logPDF(%g) = %.15g, want %.15g (diff %.3g)", z, got, fix.LogPDF[i], got-fix.LogPDF[i])
		}
	}
}

// TestProbitDerivativeMatchesFiniteDifference ports the finite-difference check
// from test_laplace.py: the analytic gradient of sum log Phi(y f) matches a
// central difference.
func TestProbitDerivativeMatchesFiniteDifference(t *testing.T) {
	f := []float64{0.3, -1.2, 2.1, -0.4, 0.05, -3.0}
	y := []float64{1, -1, 1, 1, -1, -1}
	_, grad, _ := ProbitDerivs(f, y)

	const eps = 1e-6
	for i := range f {
		fp := append([]float64(nil), f...)
		fm := append([]float64(nil), f...)
		fp[i] += eps
		fm[i] -= eps
		d := (sumLogPhi(fp, y) - sumLogPhi(fm, y)) / (2 * eps)
		if math.Abs(d-grad[i]) > 1e-4 {
			t.Errorf("grad[%d] = %.8g, finite-diff = %.8g", i, grad[i], d)
		}
	}
}

func sumLogPhi(f, y []float64) float64 {
	s := 0.0
	for i := range f {
		s += logCDF(y[i] * f[i])
	}
	return s
}
