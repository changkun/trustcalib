package kernel_test

import (
	"math"
	"math/rand"
	"testing"

	"gonum.org/v1/gonum/mat"

	"github.com/changkun/trustcalib/internal/testutil"
	"github.com/changkun/trustcalib/kernel"
)

func randPacked(rng *rand.Rand, n, dTool, dCtx int) kernel.Packed {
	pt := mat.NewDense(n, dTool, nil)
	pc := mat.NewDense(n, dCtx, nil)
	t := make([]float64, n)
	for i := 0; i < n; i++ {
		for j := 0; j < dTool; j++ {
			pt.Set(i, j, rng.Float64())
		}
		for j := 0; j < dCtx; j++ {
			pc.Set(i, j, rng.Float64())
		}
		t[i] = float64(i)
	}
	return kernel.Packed{PhiTool: pt, PhiCtx: pc, T: t}
}

// TestProductKernelPSD ports test_kernel_psd.py: K is symmetric, PSD, has
// diagonal sigma2, and admits a Cholesky after a small jitter.
func TestProductKernelPSD(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	k := kernel.DefaultKernel()
	p := randPacked(rng, 24, 11, 9)
	K := k.Full(p)
	n := K.SymmetricDim()

	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if math.Abs(K.At(i, j)-K.At(j, i)) > 1e-10 {
				t.Fatalf("kernel not symmetric at (%d,%d)", i, j)
			}
		}
		if math.Abs(K.At(i, i)-k.Sigma2) > 1e-9 {
			t.Fatalf("diag[%d] = %v, want %v", i, K.At(i, i), k.Sigma2)
		}
	}

	var eig mat.EigenSym
	if !eig.Factorize(K, false) {
		t.Fatal("eigendecomposition failed")
	}
	vals := eig.Values(nil)
	for _, v := range vals {
		if v < -1e-8 {
			t.Fatalf("kernel not PSD: min eigenvalue %v", v)
		}
	}

	jittered := mat.NewSymDense(n, nil)
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			v := K.At(i, j)
			if i == j {
				v += 1e-6
			}
			jittered.SetSym(i, j, v)
		}
	}
	var chol mat.Cholesky
	if !chol.Factorize(jittered) {
		t.Fatal("Cholesky of K + jitter failed")
	}
}

// TestKernelGolden checks the kernel matrix against the Python reference.
func TestKernelGolden(t *testing.T) {
	var fix struct {
		Kernel testutil.KernelParams `json:"kernel"`
		Train  testutil.PackedJSON   `json:"train"`
		K      [][]float64           `json:"K"`
	}
	testutil.Load(t, "kernel_full.json", &fix)

	k := fix.Kernel.Kernel()
	K := k.Full(fix.Train.Packed())
	n := K.SymmetricDim()
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if got, want := K.At(i, j), fix.K[i][j]; math.Abs(got-want) > 1e-9 {
				t.Fatalf("K[%d][%d] = %.12g, want %.12g (diff %.3g)", i, j, got, want, got-want)
			}
		}
	}
}

// TestAdditiveKernelGolden checks the additive kernel matrix against the
// Python reference (experiment.kernel.AdditiveKernel).
func TestAdditiveKernelGolden(t *testing.T) {
	var fix struct {
		Kernel testutil.AdditiveParams `json:"kernel"`
		Train  testutil.PackedJSON     `json:"train"`
		K      [][]float64             `json:"K"`
	}
	testutil.Load(t, "kernel_additive.json", &fix)

	k := fix.Kernel.Kernel()
	p := fix.Train.Packed()
	K := k.Full(p)
	n := K.SymmetricDim()
	if n != len(fix.K) {
		t.Fatalf("dim %d, want %d", n, len(fix.K))
	}
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if got, want := K.At(i, j), fix.K[i][j]; math.Abs(got-want) > 1e-9 {
				t.Fatalf("K[%d][%d] = %.12g, want %.12g (diff %.3g)", i, j, got, want, got-want)
			}
		}
	}
	// Cross on (p, p) must agree with Full.
	C := k.Cross(p, p)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if C.At(i, j) != K.At(i, j) {
				t.Fatalf("Cross[%d][%d] = %v, Full = %v", i, j, C.At(i, j), K.At(i, j))
			}
		}
	}
}

// TestAdditiveKernelPSD: the additive kernel is symmetric, PSD, and has
// diagonal s_static + s_global + s_inter (Diag agrees with Full).
func TestAdditiveKernelPSD(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	k := kernel.DefaultAdditiveKernel()
	p := randPacked(rng, 24, 11, 9)
	for i := range p.T {
		p.T[i] = float64(17 * i) // spread in time so k_time matters
	}
	K := k.Full(p)
	n := K.SymmetricDim()
	want := k.SStatic + k.SGlobal + k.SInter
	if k.PriorVar() != want {
		t.Fatalf("PriorVar = %v, want %v", k.PriorVar(), want)
	}
	diag := k.Diag(p)
	for i := 0; i < n; i++ {
		if math.Abs(K.At(i, i)-want) > 1e-12 || diag[i] != want {
			t.Fatalf("diag[%d] = %v (Diag %v), want %v", i, K.At(i, i), diag[i], want)
		}
	}
	var eig mat.EigenSym
	if !eig.Factorize(K, false) {
		t.Fatal("eigendecomposition failed")
	}
	for _, v := range eig.Values(nil) {
		if v < -1e-8 {
			t.Fatalf("kernel not PSD: min eigenvalue %v", v)
		}
	}
}

// TestAdditiveKernelDoesNotForgetStaticRisk: for the same action far apart in
// time, the product kernel's covariance decays to 0 while the additive
// kernel's keeps its static component s_static; for unrelated actions at the
// same time the additive kernel keeps the shared-tolerance component s_global.
func TestAdditiveKernelDoesNotForgetStaticRisk(t *testing.T) {
	x := []float64{0.3, 0.7, 0.1}
	far := kernel.Packed{
		PhiTool: mat.NewDense(2, 3, append(append([]float64(nil), x...), x...)),
		PhiCtx:  mat.NewDense(2, 1, []float64{0.5, 0.5}),
		T:       []float64{0, 1e5},
	}
	pk := kernel.DefaultKernel()
	ak := kernel.DefaultAdditiveKernel()
	if v := pk.Full(far).At(0, 1); v > 1e-12 {
		t.Errorf("product kernel should forget: k = %v", v)
	}
	if v := ak.Full(far).At(0, 1); math.Abs(v-ak.SStatic) > 1e-9 {
		t.Errorf("additive kernel should keep s_static = %v, got %v", ak.SStatic, v)
	}

	apart := kernel.Packed{
		PhiTool: mat.NewDense(2, 3, []float64{0, 0, 0, 100, 100, 100}),
		PhiCtx:  mat.NewDense(2, 1, []float64{0, 100}),
		T:       []float64{5, 5},
	}
	if v := pk.Full(apart).At(0, 1); v > 1e-12 {
		t.Errorf("product kernel: unrelated actions should be uncorrelated, got %v", v)
	}
	if v := ak.Full(apart).At(0, 1); math.Abs(v-ak.SGlobal) > 1e-9 {
		t.Errorf("additive kernel should share tolerance s_global = %v, got %v", ak.SGlobal, v)
	}
}
