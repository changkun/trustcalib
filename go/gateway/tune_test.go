package gateway

import "testing"

// TestTuneThresholdsSeparable: on cleanly separable data the tuner finds a band
// that auto-decides most traffic without violating the safety caps.
func TestTuneThresholdsSeparable(t *testing.T) {
	var pHat []float64
	var label []int
	for i := 0; i < 100; i++ {
		pHat = append(pHat, 0.9)
		label = append(label, 1)
		pHat = append(pHat, 0.1)
		label = append(label, 0)
	}
	low, high := TuneThresholds(pHat, label, 0.02, 0.05)
	if !(low < high) {
		t.Fatalf("expected low < high, got %v, %v", low, high)
	}
	if high > 0.9 {
		t.Errorf("tau_high %v too high to allow the 0.9 cluster", high)
	}
	if low < 0.1 {
		t.Errorf("tau_low %v too low to block the 0.1 cluster", low)
	}
}

// TestTuneThresholdsFallback: when nothing is feasible (pure noise), fall back
// to the default band.
func TestTuneThresholdsFallback(t *testing.T) {
	var pHat []float64
	var label []int
	for i := 0; i < 200; i++ {
		pHat = append(pHat, 0.5)
		label = append(label, i%2)
	}
	low, high := TuneThresholds(pHat, label, 0.02, 0.05)
	if low != 0.35 || high != 0.65 {
		t.Fatalf("expected default fallback (0.35, 0.65), got (%v, %v)", low, high)
	}
}
