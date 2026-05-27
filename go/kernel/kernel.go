// Package kernel implements the structured product kernel from Section 4 of
// the trustcalib manuscript:
//
//	k(x, x') = sigma2 * k_tool(a, a') * k_ctx(c, c') * k_time(t, t')
//
// k_tool and k_ctx are squared-exponential (RBF) kernels over the tool and
// context feature blocks; k_time = exp(-|t-t'|/lambda) is the Ornstein-
// Uhlenbeck covariance for non-stationarity. The product of PSD kernels is PSD
// (Schur product). Each block's self-similarity is 1, so the prior variance on
// the diagonal is exactly sigma2.
//
// The kernel is stateless and operates on raw feature vectors (a Packed),
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

// ProductKernel holds the kernel hyperparameters. The zero value is not valid;
// use DefaultKernel or set every field.
type ProductKernel struct {
	Sigma2 float64 // signal variance (prior diagonal)
	LTool  float64 // RBF lengthscale for the tool block
	LCtx   float64 // RBF lengthscale for the context block
	Lambda float64 // time lengthscale (OU decay, in steps)
}

// DefaultKernel returns the manuscript defaults (kernel.py dataclass defaults).
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

// blocks computes the full (m x n) product-kernel matrix between p and q.
func (k ProductKernel) blocks(p, q Packed) *mat.Dense {
	d2tool := sqdist(p.PhiTool, q.PhiTool)
	d2ctx := sqdist(p.PhiCtx, q.PhiCtx)
	m, n := d2tool.Dims()

	twoLTool2 := 2.0 * k.LTool * k.LTool
	twoLCtx2 := 2.0 * k.LCtx * k.LCtx

	out := mat.NewDense(m, n, nil)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			kTool := math.Exp(-d2tool.At(i, j) / twoLTool2)
			kCtx := math.Exp(-d2ctx.At(i, j) / twoLCtx2)
			kTime := math.Exp(-math.Abs(p.T[i]-q.T[j]) / k.Lambda)
			out.Set(i, j, k.Sigma2*kTool*kCtx*kTime)
		}
	}
	return out
}

// Full returns the symmetric (N x N) train-train covariance K.
func (k ProductKernel) Full(p Packed) *mat.SymDense {
	n := p.Len()
	b := k.blocks(p, p)
	s := mat.NewSymDense(n, nil)
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			s.SetSym(i, j, b.At(i, j))
		}
	}
	return s
}

// Cross returns the (len(p) x len(q)) train-test cross covariance (rows p,
// cols q).
func (k ProductKernel) Cross(p, q Packed) *mat.Dense {
	return k.blocks(p, q)
}

// Diag returns the prior variances diag(K); every entry equals Sigma2.
func (k ProductKernel) Diag(p Packed) []float64 {
	out := make([]float64, p.Len())
	for i := range out {
		out[i] = k.Sigma2
	}
	return out
}
