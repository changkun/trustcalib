package featurizer_test

import (
	"testing"

	"github.com/changkun/trustcalib/featurizer"
)

func TestPack(t *testing.T) {
	fvs := []featurizer.FeatureVec{
		{PhiTool: []float64{1, 2}, PhiCtx: []float64{3}, T: 10},
		{PhiTool: []float64{4, 5}, PhiCtx: []float64{6}, T: 20},
	}
	p := featurizer.Pack(fvs)
	if p.Len() != 2 {
		t.Fatalf("len = %d, want 2", p.Len())
	}
	r, c := p.PhiTool.Dims()
	if r != 2 || c != 2 {
		t.Fatalf("phi_tool dims = (%d,%d), want (2,2)", r, c)
	}
	if p.PhiTool.At(1, 0) != 4 || p.PhiCtx.At(0, 0) != 3 || p.T[1] != 20 {
		t.Fatalf("packed values wrong: %v", p)
	}
}

func TestPackEmpty(t *testing.T) {
	p := featurizer.Pack(nil)
	if p.Len() != 0 {
		t.Fatalf("empty pack len = %d, want 0", p.Len())
	}
}
