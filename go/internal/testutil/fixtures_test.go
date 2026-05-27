package testutil

import "testing"

func TestKernelParamsBuild(t *testing.T) {
	k := KernelParams{Sigma2: 1.6, LTool: 1.1, LCtx: 1.2, Lam: 200}.Kernel()
	if k.Sigma2 != 1.6 || k.LTool != 1.1 || k.LCtx != 1.2 || k.Lambda != 200 {
		t.Fatalf("kernel build: %+v", k)
	}
}

func TestPackedJSON(t *testing.T) {
	pj := PackedJSON{
		PhiTool: [][]float64{{1, 2}, {3, 4}},
		PhiCtx:  [][]float64{{5}, {6}},
		T:       []float64{0, 1},
	}
	p := pj.Packed()
	if p.Len() != 2 {
		t.Fatalf("len = %d", p.Len())
	}
	if p.PhiTool.At(1, 1) != 4 || p.PhiCtx.At(1, 0) != 6 {
		t.Fatal("packed values wrong")
	}
}

func TestDenseEmpty(t *testing.T) {
	r, c := dense(nil).Dims()
	if r != 0 || c != 0 {
		t.Fatalf("empty dense dims = (%d,%d)", r, c)
	}
}

func TestLoadFixture(t *testing.T) {
	var fix struct {
		Z []float64 `json:"z"`
	}
	Load(t, "logcdf.json", &fix)
	if len(fix.Z) == 0 {
		t.Fatal("fixture not loaded")
	}
}
