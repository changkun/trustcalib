// Package gp implements a Gaussian-process probit classifier via the Laplace
// approximation (Rasmussen & Williams 2006, Algorithm 3.1 for the posterior
// mode and 3.2 for predictions), specialized to the probit approve/deny
// likelihood. It is a direct port of experiment/gp.py.
package gp

import (
	"errors"
	"math"

	"gonum.org/v1/gonum/mat"

	"github.com/changkun/trustcalib/kernel"
)

// LaplaceGPC is a GP probit classifier. Construct with NewLaplaceGPC; Jitter,
// MaxIter and Tol may be overridden before the first Fit.
type LaplaceGPC struct {
	Kernel  kernel.ProductKernel
	Jitter  float64
	MaxIter int
	Tol     float64

	fitted      bool
	p           kernel.Packed
	grad        []float64
	sW          []float64
	chol        mat.Cholesky // factorization of B = I + sW K sW at the mode
	fHat        []float64
	logMarginal float64
}

// NewLaplaceGPC returns a classifier with the manuscript defaults (jitter 1e-6,
// 100 iterations, tolerance 1e-6).
func NewLaplaceGPC(k kernel.ProductKernel) *LaplaceGPC {
	return &LaplaceGPC{Kernel: k, Jitter: 1e-6, MaxIter: 100, Tol: 1e-6}
}

// Fit finds the posterior mode (R&W Algorithm 3.1). y01 are 0/1 approve labels,
// mapped internally to {-1, +1}.
func (m *LaplaceGPC) Fit(p kernel.Packed, y01 []int) error {
	n := len(y01)
	if n == 0 {
		return errors.New("gp: empty training set")
	}
	if p.Len() != n {
		return errors.New("gp: label count does not match point count")
	}
	y := make([]float64, n)
	for i, v := range y01 {
		if v > 0 {
			y[i] = 1
		} else {
			y[i] = -1
		}
	}

	K := m.Kernel.Full(p)
	for i := 0; i < n; i++ {
		K.SetSym(i, i, K.At(i, i)+m.Jitter)
	}

	f := make([]float64, n)
	var grad, w, a []float64
	var ll float64
	var chol mat.Cholesky
	lastObj := math.Inf(-1)

	for iter := 0; iter < m.MaxIter; iter++ {
		ll, grad, w = ProbitDerivs(f, y)
		sW := sqrtAll(w)
		if !factorizeB(&chol, K, sW, n) {
			return errors.New("gp: Cholesky factorization failed (B not positive definite)")
		}
		// b = W*f + grad
		b := make([]float64, n)
		for i := 0; i < n; i++ {
			b[i] = w[i]*f[i] + grad[i]
		}
		// a = b - sW * B^{-1} (sW * (K @ b)); note Lᵀ\(L\·) == B^{-1}·.
		a = newtonStep(&chol, K, b, sW, n)
		// f = K @ a
		f = matVec(K, a)
		obj := -0.5*dot(a, f) + ll
		if math.Abs(obj-lastObj) < m.Tol {
			lastObj = obj
			break
		}
		lastObj = obj
	}

	// Recompute likelihood/derivatives and the factorization at the final mode.
	ll, grad, w = ProbitDerivs(f, y)
	sW := sqrtAll(w)
	if !factorizeB(&chol, K, sW, n) {
		return errors.New("gp: final Cholesky factorization failed")
	}

	m.p = p
	m.grad = grad
	m.sW = sW
	m.chol = chol
	m.fHat = f
	// Laplace log marginal likelihood (R&W 3.32):
	//   log Z = -0.5 aᵀf + log p(y|f) - sum log diag(L),
	// and sum log diag(L) = 0.5 * log det(B).
	m.logMarginal = -0.5*dot(a, f) + ll - 0.5*chol.LogDet()
	m.fitted = true
	return nil
}

// Predict returns the posterior latent mean, variance (floored at 1e-12) and
// approval probability pi = Phi(fBar/sqrt(1+var)) for each query point
// (R&W Algorithm 3.2).
func (m *LaplaceGPC) Predict(q kernel.Packed) (fBar, variance, pi []float64, err error) {
	if !m.fitted {
		return nil, nil, nil, errors.New("gp: Predict called before Fit")
	}
	nTrain := m.p.Len()
	mq := q.Len()
	Ks := m.Kernel.Cross(m.p, q) // (nTrain x mq)
	kss := m.Kernel.Diag(q)

	// fBar = Ksᵀ @ grad
	var fb mat.VecDense
	fb.MulVec(Ks.T(), mat.NewVecDense(nTrain, m.grad))

	// M = sW[:,None] * Ks
	M := mat.NewDense(nTrain, mq, nil)
	for i := 0; i < nTrain; i++ {
		for j := 0; j < mq; j++ {
			M.Set(i, j, m.sW[i]*Ks.At(i, j))
		}
	}
	// The R&W variance reduction is sum_i v_ij^2 with v = L^{-1} M, i.e.
	// M_jᵀ B^{-1} M_j. Compute Z = B^{-1} M and reduce columnwise.
	var Z mat.Dense
	if err := m.chol.SolveTo(&Z, M); err != nil {
		return nil, nil, nil, err
	}

	fBar = make([]float64, mq)
	variance = make([]float64, mq)
	pi = make([]float64, mq)
	for j := 0; j < mq; j++ {
		reduction := 0.0
		for i := 0; i < nTrain; i++ {
			reduction += M.At(i, j) * Z.At(i, j)
		}
		v := kss[j] - reduction
		if v < 1e-12 {
			v = 1e-12
		}
		fBar[j] = fb.AtVec(j)
		variance[j] = v
		pi[j] = stdNormCDF(fBar[j] / math.Sqrt(1.0+v))
	}
	return fBar, variance, pi, nil
}

// PredictProb returns just the approval probability for each query point.
func (m *LaplaceGPC) PredictProb(q kernel.Packed) ([]float64, error) {
	_, _, pi, err := m.Predict(q)
	return pi, err
}

// LogMarginal returns the Laplace log marginal likelihood at the mode.
func (m *LaplaceGPC) LogMarginal() float64 { return m.logMarginal }

// Fitted reports whether Fit has succeeded at least once.
func (m *LaplaceGPC) Fitted() bool { return m.fitted }

// --- helpers ---------------------------------------------------------------

// factorizeB builds B = I + sW K sW (SPD by construction) and factorizes it.
func factorizeB(chol *mat.Cholesky, K *mat.SymDense, sW []float64, n int) bool {
	B := mat.NewSymDense(n, nil)
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			v := sW[i] * K.At(i, j) * sW[j]
			if i == j {
				v += 1.0
			}
			B.SetSym(i, j, v)
		}
	}
	return chol.Factorize(B)
}

// newtonStep returns a = b - sW ⊙ B^{-1}(sW ⊙ (K b)).
func newtonStep(chol *mat.Cholesky, K *mat.SymDense, b, sW []float64, n int) []float64 {
	Kb := matVec(K, b)
	rhs := mat.NewVecDense(n, nil)
	for i := 0; i < n; i++ {
		rhs.SetVec(i, sW[i]*Kb[i])
	}
	var x mat.VecDense
	if err := chol.SolveVecTo(&x, rhs); err != nil {
		// B is SPD; a solve failure should not happen after a successful
		// Factorize. Fall back to the un-corrected step rather than panicking.
		return append([]float64(nil), b...)
	}
	a := make([]float64, n)
	for i := 0; i < n; i++ {
		a[i] = b[i] - sW[i]*x.AtVec(i)
	}
	return a
}

func matVec(K *mat.SymDense, v []float64) []float64 {
	n := len(v)
	var out mat.VecDense
	out.MulVec(K, mat.NewVecDense(n, v))
	res := make([]float64, n)
	for i := 0; i < n; i++ {
		res[i] = out.AtVec(i)
	}
	return res
}

func sqrtAll(w []float64) []float64 {
	out := make([]float64, len(w))
	for i, v := range w {
		out[i] = math.Sqrt(v)
	}
	return out
}

func dot(a, b []float64) float64 {
	s := 0.0
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

// stdNormCDF is the standard-normal CDF via the complementary error function.
func stdNormCDF(x float64) float64 {
	return 0.5 * math.Erfc(-x/math.Sqrt2)
}
