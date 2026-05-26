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
