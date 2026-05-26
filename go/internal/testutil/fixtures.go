// Package testutil provides helpers shared by the golden-fixture tests.
package testutil

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"gonum.org/v1/gonum/mat"

	"github.com/changkun/trustcalib/kernel"
)

// KernelParams mirrors the "kernel" block in the fixtures.
type KernelParams struct {
	Sigma2 float64 `json:"sigma2"`
	LTool  float64 `json:"l_tool"`
	LCtx   float64 `json:"l_ctx"`
	Lam    float64 `json:"lam"`
}

// Kernel builds a kernel.ProductKernel from the fixture parameters.
func (p KernelParams) Kernel() kernel.ProductKernel {
	return kernel.ProductKernel{Sigma2: p.Sigma2, LTool: p.LTool, LCtx: p.LCtx, Lambda: p.Lam}
}

// PackedJSON mirrors a packed feature block in the fixtures.
type PackedJSON struct {
	PhiTool [][]float64 `json:"phi_tool"`
	PhiCtx  [][]float64 `json:"phi_ctx"`
	T       []float64   `json:"t"`
}

// Packed converts the JSON block into a kernel.Packed.
func (p PackedJSON) Packed() kernel.Packed {
	return kernel.Packed{
		PhiTool: dense(p.PhiTool),
		PhiCtx:  dense(p.PhiCtx),
		T:       p.T,
	}
}

func dense(rows [][]float64) *mat.Dense {
	n := len(rows)
	if n == 0 {
		return mat.NewDense(0, 0, nil)
	}
	d := len(rows[0])
	m := mat.NewDense(n, d, nil)
	for i, r := range rows {
		m.SetRow(i, r)
	}
	return m
}

// fixturesDir returns the absolute path to go/testdata/fixtures regardless of
// which package's tests are running.
func fixturesDir() string {
	_, file, _, _ := runtime.Caller(0)
	// file = .../go/internal/testutil/fixtures.go
	return filepath.Join(filepath.Dir(file), "..", "..", "testdata", "fixtures")
}

// Load reads and unmarshals a fixture file into v.
func Load(t *testing.T, name string, v any) {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(fixturesDir(), name))
	if err != nil {
		t.Fatalf("read fixture %s: %v", name, err)
	}
	if err := json.Unmarshal(data, v); err != nil {
		t.Fatalf("unmarshal fixture %s: %v", name, err)
	}
}
