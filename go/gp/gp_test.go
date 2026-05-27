package gp

import (
	"math"
	"math/rand"
	"sort"
	"testing"

	"gonum.org/v1/gonum/mat"

	"github.com/changkun/trustcalib/internal/testutil"
	"github.com/changkun/trustcalib/kernel"
)

// TestModeStationarity ports test_laplace.py: at the Laplace mode,
// f_hat = K @ grad (R&W Alg 3.1) and the log marginal is finite.
func TestModeStationarity(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	p, y01 := randTrain(rng, 12)
	m := NewLaplaceGPC(kernel.DefaultKernel())
	if err := m.Fit(p, y01); err != nil {
		t.Fatal(err)
	}

	K := m.Kernel.Full(p)
	n := p.Len()
	for i := 0; i < n; i++ {
		K.SetSym(i, i, K.At(i, i)+m.Jitter)
	}
	Kg := matVec(K, m.grad)
	resid := 0.0
	for i := 0; i < n; i++ {
		d := m.fHat[i] - Kg[i]
		resid += d * d
	}
	if math.Sqrt(resid) > 1e-5 {
		t.Fatalf("stationarity residual %.3g", math.Sqrt(resid))
	}
	if math.IsInf(m.logMarginal, 0) || math.IsNaN(m.logMarginal) {
		t.Fatalf("log marginal not finite: %v", m.logMarginal)
	}
}

// TestN2HandExample ports test_laplace.py: two identical approved points yield a
// symmetric, positive latent mode and approval probabilities in (0.5, 1).
func TestN2HandExample(t *testing.T) {
	pt := mat.NewDense(2, 2, []float64{0.4, 0.7, 0.4, 0.7})
	pc := mat.NewDense(2, 1, []float64{0.2, 0.2})
	p := kernel.Packed{PhiTool: pt, PhiCtx: pc, T: []float64{0, 0}}
	m := NewLaplaceGPC(kernel.DefaultKernel())
	if err := m.Fit(p, []int{1, 1}); err != nil {
		t.Fatal(err)
	}
	if math.Abs(m.fHat[0]-m.fHat[1]) > 1e-6 {
		t.Fatalf("mode not symmetric: %v", m.fHat)
	}
	if m.fHat[0] <= 0 {
		t.Fatalf("approval should give positive latent, got %v", m.fHat[0])
	}
	_, _, pi, err := m.Predict(p)
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range pi {
		if v <= 0.5 || v >= 1.0 {
			t.Fatalf("pi = %v, want in (0.5, 1)", v)
		}
	}
}

// TestFitGolden checks the fitted mode, gradient and log marginal against the
// Python reference.
func TestFitGolden(t *testing.T) {
	var fix struct {
		Kernel      testutil.KernelParams `json:"kernel"`
		Train       testutil.PackedJSON   `json:"train"`
		Y01         []int                 `json:"y01"`
		FHat        []float64             `json:"f_hat"`
		Grad        []float64             `json:"grad"`
		LogMarginal float64               `json:"log_marginal"`
	}
	testutil.Load(t, "laplace_fit.json", &fix)

	m := NewLaplaceGPC(fix.Kernel.Kernel())
	if err := m.Fit(fix.Train.Packed(), fix.Y01); err != nil {
		t.Fatal(err)
	}
	approxSlice(t, "f_hat", m.fHat, fix.FHat, 1e-7)
	approxSlice(t, "grad", m.grad, fix.Grad, 1e-7)
	if math.Abs(m.logMarginal-fix.LogMarginal) > 1e-6 {
		t.Errorf("log_marginal = %.10g, want %.10g", m.logMarginal, fix.LogMarginal)
	}
}

// TestPredictGolden checks predictive mean/var/prob against the Python reference.
func TestPredictGolden(t *testing.T) {
	var fix struct {
		Kernel testutil.KernelParams `json:"kernel"`
		Train  testutil.PackedJSON   `json:"train"`
		Y01    []int                 `json:"y01"`
		Query  testutil.PackedJSON   `json:"query"`
		FBar   []float64             `json:"f_bar"`
		Var    []float64             `json:"var"`
		Pi     []float64             `json:"pi"`
	}
	testutil.Load(t, "predict.json", &fix)

	m := NewLaplaceGPC(fix.Kernel.Kernel())
	if err := m.Fit(fix.Train.Packed(), fix.Y01); err != nil {
		t.Fatal(err)
	}
	fBar, variance, pi, err := m.Predict(fix.Query.Packed())
	if err != nil {
		t.Fatal(err)
	}
	approxSlice(t, "f_bar", fBar, fix.FBar, 1e-7)
	approxSlice(t, "var", variance, fix.Var, 1e-7)
	approxSlice(t, "pi", pi, fix.Pi, 1e-7)
}

// TestRecovery ports test_recovery.py: with a 1-D probit ground truth
// f*(x) = x, the GP recovers a monotone, well-located approval boundary.
func TestRecovery(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	const nTrain = 300
	xs := make([]float64, nTrain)
	y := make([]int, nTrain)
	for i := range xs {
		x := -3.0 + 6.0*rng.Float64()
		xs[i] = x
		if rng.Float64() < stdNormCDF(x) {
			y[i] = 1
		}
	}
	m := NewLaplaceGPC(kernel.DefaultKernel())
	if err := m.Fit(packed1D(xs), y); err != nil {
		t.Fatal(err)
	}

	grid := make([]float64, 81)
	for i := range grid {
		grid[i] = -4.0 + 8.0*float64(i)/80.0
	}
	pi, err := m.PredictProb(packed1D(grid))
	if err != nil {
		t.Fatal(err)
	}

	var gridInner, piInner []float64
	for i, x := range grid {
		if x >= -2 && x <= 2 {
			gridInner = append(gridInner, x)
			piInner = append(piInner, pi[i])
		}
	}
	if rho := spearman(gridInner, piInner); rho < 0.98 {
		t.Fatalf("Spearman rank correlation %.4f, want > 0.98", rho)
	}

	cross, best := grid[0], math.Inf(1)
	for i, x := range grid {
		if d := math.Abs(pi[i] - 0.5); d < best {
			best, cross = d, x
		}
	}
	if math.Abs(cross) > 0.6 {
		t.Fatalf("boundary crossing at %.3f, want |x| < 0.6", cross)
	}

	xt := make([]float64, 400)
	for i := range xt {
		xt[i] = -3.0 + 6.0*rng.Float64()
	}
	pt, err := m.PredictProb(packed1D(xt))
	if err != nil {
		t.Fatal(err)
	}
	correct := 0
	for i, x := range xt {
		if (pt[i] >= 0.5) == (x > 0) {
			correct++
		}
	}
	if acc := float64(correct) / float64(len(xt)); acc < 0.85 {
		t.Fatalf("fresh accuracy %.3f, want > 0.85", acc)
	}
}

// --- helpers ---------------------------------------------------------------

func randTrain(rng *rand.Rand, n int) (kernel.Packed, []int) {
	pt := mat.NewDense(n, 11, nil)
	pc := mat.NewDense(n, 9, nil)
	t := make([]float64, n)
	y := make([]int, n)
	for i := 0; i < n; i++ {
		for j := 0; j < 11; j++ {
			pt.Set(i, j, rng.Float64())
		}
		for j := 0; j < 9; j++ {
			pc.Set(i, j, rng.Float64())
		}
		t[i] = float64(i)
		if pc.At(i, 0)+0.3*(pt.At(i, 0)-0.5) > 0.5 {
			y[i] = 1
		}
	}
	return kernel.Packed{PhiTool: pt, PhiCtx: pc, T: t}, y
}

func packed1D(xs []float64) kernel.Packed {
	n := len(xs)
	pt := mat.NewDense(n, 1, append([]float64(nil), xs...))
	pc := mat.NewDense(n, 1, make([]float64, n))
	return kernel.Packed{PhiTool: pt, PhiCtx: pc, T: make([]float64, n)}
}

func approxSlice(t *testing.T, name string, got, want []float64, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d, want %d", name, len(got), len(want))
	}
	for i := range got {
		if math.Abs(got[i]-want[i]) > tol {
			t.Errorf("%s[%d] = %.12g, want %.12g (diff %.3g)", name, i, got[i], want[i], got[i]-want[i])
		}
	}
}

// spearman returns the Spearman rank correlation between x and y.
func spearman(x, y []float64) float64 {
	rx := ranks(x)
	ry := ranks(y)
	return pearson(rx, ry)
}

func ranks(v []float64) []float64 {
	type iv struct {
		val float64
		idx int
	}
	pairs := make([]iv, len(v))
	for i, x := range v {
		pairs[i] = iv{x, i}
	}
	sort.Slice(pairs, func(a, b int) bool { return pairs[a].val < pairs[b].val })
	r := make([]float64, len(v))
	for rank, p := range pairs {
		r[p.idx] = float64(rank)
	}
	return r
}

func pearson(a, b []float64) float64 {
	n := float64(len(a))
	var sa, sb float64
	for i := range a {
		sa += a[i]
		sb += b[i]
	}
	ma, mb := sa/n, sb/n
	var cov, va, vb float64
	for i := range a {
		da, db := a[i]-ma, b[i]-mb
		cov += da * db
		va += da * da
		vb += db * db
	}
	return cov / math.Sqrt(va*vb)
}
