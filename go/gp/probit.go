package gp

import "math"

const log2pi = 1.8378770664093454835606594728112352797227949472755668 // log(2*pi)

// logPDF is the log standard-normal density: -0.5 z^2 - 0.5 log(2*pi).
func logPDF(z float64) float64 {
	return -0.5*z*z - 0.5*log2pi
}

// logCDF is a numerically stable log standard-normal CDF, matching scipy's
// log_ndtr. log(CDF(z)) underflows to -Inf for z below about -37 if computed
// naively, so three regimes are used:
//
//   - z >= 0:    log1p(-0.5*erfc(z/sqrt2))   (avoids cancellation near 1)
//   - -37 <= z:  log(0.5*erfc(-z/sqrt2))     (erfc is accurate and finite here)
//   - z < -37:   asymptotic tail expansion of the upper tail in log space.
func logCDF(z float64) float64 {
	const sqrt2 = math.Sqrt2
	switch {
	case z >= 0:
		return math.Log1p(-0.5 * math.Erfc(z/sqrt2))
	case z >= -37.0:
		return math.Log(0.5 * math.Erfc(-z/sqrt2))
	default:
		// Phi(z) = phi(z)/(-z) * (1 - 1/z^2 + 3/z^4 - 15/z^6 + ...).
		// log Phi(z) = log phi(z) - log(-z) + log(series).
		z2 := z * z
		series := 1.0
		term := 1.0
		for k := 1; k <= 6; k++ {
			term *= -float64(2*k-1) / z2
			series += term
		}
		return logPDF(z) - math.Log(-z) + math.Log(series)
	}
}

// ProbitDerivs returns (sum log Phi(z_i), gradient, Hessian-diagonal W) for the
// probit likelihood at latent values f with labels y in {-1, +1}:
//
//	z = y*f
//	d/df  log p = y * phi(z)/Phi(z)              (Mills ratio, R&W 3.15)
//	W = (phi/Phi)^2 + z*(phi/Phi)                (R&W 3.16, >= 0)
//
// The Mills ratio is evaluated as exp(logPDF - logCDF) to stay finite when
// Phi(z) underflows. W is floored at 1e-10. Exported for the finite-difference
// gradient test.
func ProbitDerivs(f, y []float64) (sumLogPhi float64, grad, w []float64) {
	n := len(f)
	grad = make([]float64, n)
	w = make([]float64, n)
	for i := 0; i < n; i++ {
		z := y[i] * f[i]
		lP := logCDF(z)
		r := math.Exp(logPDF(z) - lP)
		grad[i] = y[i] * r
		wi := r*r + z*r
		if wi < 1e-10 {
			wi = 1e-10
		}
		w[i] = wi
		sumLogPhi += lP
	}
	return sumLogPhi, grad, w
}
