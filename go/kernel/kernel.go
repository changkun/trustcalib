// Package kernel implements the structured kernels over (action, context,
// time) of the trustcalib manuscript (section "Trust Calibration as
// Classification with a Reject Option"). Two kernels share the same feature
// blocks and implement the Kernel interface.
//
// ProductKernel, the separable kernel,
//
//	k(x, x') = sigma2 * k_tool(a, a') * k_ctx(c, c') * k_time(t, t')
//
// multiplies every component by k_time, so all evidence, including the static
// risk structure of an action, is forgotten at rate 1/lambda (manuscript
// Proposition 4): once the labels near an action are older than a few lambda,
// its posterior returns to the prior p_hat = 1/2 and it is re-escalated.
//
// AdditiveKernel, the additive kernel,
//
//	k(x, x') = s_static * k_x(x, x')                  static action risk r(x)
//	         + s_global * k_time(t, t')               shared tolerance tau(t)
//	         + s_inter  * k_x(x, x') * k_time(t, t')  local drift
//
// with k_x = k_tool * k_ctx, mirrors the latent-tolerance decomposition
// f(x, t) = tau(t) - r(x) of the manuscript. Only the time-coupled components
// forget. The static component never decays, so what has been learned about
// how risky an action is persists; every label, whatever its action, updates
// the shared tolerance tau(t); and the interaction component lets individual
// actions drift locally.
//
// Block kernels: k_tool and k_ctx are squared-exponential (RBF) kernels
// exp(-d²/(2 l²)) over the tool and context feature blocks, and k_time =
// exp(-|t-t'|/lambda) is the Ornstein-Uhlenbeck covariance for
// non-stationarity. Every block is PSD with unit self-similarity, and sums and
// (Schur) products of PSD kernels are PSD, so both kernels are valid
// covariance functions whose prior variance (the diagonal) is the sum of
// their component scales: sigma2 for the product kernel and
// s_static + s_global + s_inter for the additive one.
//
// The kernels are stateless and operate on raw feature vectors (a Packed),
// independent of any domain taxonomy.
package kernel

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// Packed is a batch of decision points packed into dense arrays for vectorized
// kernel evaluation. PhiTool is (N x dTool), PhiCtx is (N x dCtx) and T has
// length N.
type Packed struct {
	PhiTool *mat.Dense
	PhiCtx  *mat.Dense
	T       []float64
}

// Len returns the number of points.
func (p Packed) Len() int { return len(p.T) }

// Kernel is a covariance function over packed decision points. ProductKernel
// and AdditiveKernel implement it; the GP classifier (package gp) accepts any
// implementation.
type Kernel interface {
	// Full returns the symmetric (N x N) train-train covariance K.
	Full(p Packed) *mat.SymDense
	// Cross returns the (len(p) x len(q)) cross covariance (rows p, cols q).
	Cross(p, q Packed) *mat.Dense
	// Diag returns the prior variances diag(K).
	Diag(p Packed) []float64
}

var (
	_ Kernel = ProductKernel{}
	_ Kernel = AdditiveKernel{}
)

// ProductKernel holds the kernel hyperparameters. The zero value is not valid;
// use DefaultKernel or set every field.
type ProductKernel struct {
	Sigma2 float64 // signal variance (prior diagonal)
	LTool  float64 // RBF lengthscale for the tool block
	LCtx   float64 // RBF lengthscale for the context block
	Lambda float64 // time lengthscale (OU decay, in steps)
}

// DefaultKernel returns the product-kernel defaults (kernel.py dataclass
// defaults; the paper's experiments use Lambda 90).
func DefaultKernel() ProductKernel {
	return ProductKernel{Sigma2: 1.6, LTool: 1.1, LCtx: 1.2, Lambda: 200.0}
}

// sqdist returns the (m x n) matrix of pairwise squared Euclidean distances
// between rows of a (m x d) and b (n x d), floored at 0 to clip numerical
// noise. Mirrors data.py / kernel.py _sqdist: a2 + b2 - 2 A Bᵀ.
func sqdist(a, b *mat.Dense) *mat.Dense {
	m, _ := a.Dims()
	n, _ := b.Dims()

	var cross mat.Dense
	cross.Mul(a, b.T())

	a2 := rowSqNorms(a, m)
	b2 := rowSqNorms(b, n)

	out := mat.NewDense(m, n, nil)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			v := a2[i] + b2[j] - 2.0*cross.At(i, j)
			if v < 0 {
				v = 0
			}
			out.Set(i, j, v)
		}
	}
	return out
}

func rowSqNorms(a *mat.Dense, m int) []float64 {
	out := make([]float64, m)
	for i := 0; i < m; i++ {
		s := 0.0
		for _, v := range a.RawRowView(i) {
			s += v * v
		}
		out[i] = s
	}
	return out
}

// staticSim returns the (m x n) matrix of static action similarities
// k_x = k_tool * k_ctx between p and q (kernel.py _k_x).
func staticSim(p, q Packed, lTool, lCtx float64) *mat.Dense {
	d2tool := sqdist(p.PhiTool, q.PhiTool)
	d2ctx := sqdist(p.PhiCtx, q.PhiCtx)
	m, n := d2tool.Dims()

	twoLTool2 := 2.0 * lTool * lTool
	twoLCtx2 := 2.0 * lCtx * lCtx

	out := mat.NewDense(m, n, nil)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			kTool := math.Exp(-d2tool.At(i, j) / twoLTool2)
			kCtx := math.Exp(-d2ctx.At(i, j) / twoLCtx2)
			out.Set(i, j, kTool*kCtx)
		}
	}
	return out
}

// timeSim is the Ornstein-Uhlenbeck time covariance exp(-|t-u|/lambda)
// (kernel.py _k_time).
func timeSim(t, u, lambda float64) float64 {
	return math.Exp(-math.Abs(t-u) / lambda)
}

// symmetric copies the upper triangle of a square matrix into a SymDense.
func symmetric(b *mat.Dense) *mat.SymDense {
	n, _ := b.Dims()
	s := mat.NewSymDense(n, nil)
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			s.SetSym(i, j, b.At(i, j))
		}
	}
	return s
}

// constant returns a slice of n copies of v.
func constant(n int, v float64) []float64 {
	out := make([]float64, n)
	for i := range out {
		out[i] = v
	}
	return out
}

// blocks computes the full (m x n) product-kernel matrix between p and q.
func (k ProductKernel) blocks(p, q Packed) *mat.Dense {
	out := staticSim(p, q, k.LTool, k.LCtx)
	m, n := out.Dims()
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			out.Set(i, j, k.Sigma2*out.At(i, j)*timeSim(p.T[i], q.T[j], k.Lambda))
		}
	}
	return out
}

// Full returns the symmetric (N x N) train-train covariance K.
func (k ProductKernel) Full(p Packed) *mat.SymDense {
	return symmetric(k.blocks(p, p))
}

// Cross returns the (len(p) x len(q)) train-test cross covariance (rows p,
// cols q).
func (k ProductKernel) Cross(p, q Packed) *mat.Dense {
	return k.blocks(p, q)
}

// Diag returns the prior variances diag(K); every entry equals Sigma2.
func (k ProductKernel) Diag(p Packed) []float64 {
	return constant(p.Len(), k.Sigma2)
}

// AdditiveKernel holds the hyperparameters of the additive kernel
//
//	s_static * k_x + s_global * k_time + s_inter * k_x * k_time
//
// with k_x = k_tool * k_ctx, a port of experiment/kernel.py AdditiveKernel.
// The static component does not forget; the global and interaction components
// decay with time lengthscale Lambda. The zero value is not valid; use
// DefaultAdditiveKernel or set every field.
type AdditiveKernel struct {
	SStatic float64 // scale of the static action-risk component k_x
	SGlobal float64 // scale of the shared-tolerance component k_time
	SInter  float64 // scale of the local-drift component k_x * k_time
	LTool   float64 // RBF lengthscale for the tool block
	LCtx    float64 // RBF lengthscale for the context block
	Lambda  float64 // time lengthscale (OU decay, in steps)
}

// DefaultAdditiveKernel returns the manuscript's additive-kernel defaults
// (kernel.py AdditiveKernel dataclass defaults).
func DefaultAdditiveKernel() AdditiveKernel {
	return AdditiveKernel{SStatic: 1.6, SGlobal: 1.0, SInter: 0.6, LTool: 1.1, LCtx: 1.2, Lambda: 90.0}
}

// PriorVar returns the prior variance SStatic + SGlobal + SInter.
func (k AdditiveKernel) PriorVar() float64 { return k.SStatic + k.SGlobal + k.SInter }

// blocks computes the full (m x n) additive-kernel matrix between p and q.
func (k AdditiveKernel) blocks(p, q Packed) *mat.Dense {
	out := staticSim(p, q, k.LTool, k.LCtx)
	m, n := out.Dims()
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			kx := out.At(i, j)
			kt := timeSim(p.T[i], q.T[j], k.Lambda)
			out.Set(i, j, k.SStatic*kx+k.SGlobal*kt+k.SInter*kx*kt)
		}
	}
	return out
}

// Full returns the symmetric (N x N) train-train covariance K.
func (k AdditiveKernel) Full(p Packed) *mat.SymDense {
	return symmetric(k.blocks(p, p))
}

// Cross returns the (len(p) x len(q)) train-test cross covariance (rows p,
// cols q).
func (k AdditiveKernel) Cross(p, q Packed) *mat.Dense {
	return k.blocks(p, q)
}

// Diag returns the prior variances diag(K); every entry equals PriorVar().
func (k AdditiveKernel) Diag(p Packed) []float64 {
	return constant(p.Len(), k.PriorVar())
}
