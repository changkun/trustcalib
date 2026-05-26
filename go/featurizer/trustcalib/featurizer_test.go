package trustcalib_test

import (
	"testing"

	"github.com/changkun/trustcalib/featurizer"
	"github.com/changkun/trustcalib/featurizer/trustcalib"
)

func TestDims(t *testing.T) {
	f := trustcalib.New()
	if f.DimTool() != 3+len(trustcalib.Categories) || f.DimTool() != 11 {
		t.Errorf("DimTool = %d, want 11", f.DimTool())
	}
	if f.DimCtx() != 2+len(trustcalib.Tasks) || f.DimCtx() != 9 {
		t.Errorf("DimCtx = %d, want 9", f.DimCtx())
	}
}

func TestFeaturize(t *testing.T) {
	f := trustcalib.New()
	fv, err := f.Featurize(featurizer.Point{
		Tool: "git_force_push", Target: "prod_infra", Task: "ops_maintenance",
		ArgRisk: 1, T: 42,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(fv.PhiTool) != 11 || len(fv.PhiCtx) != 9 {
		t.Fatalf("dims: tool=%d ctx=%d", len(fv.PhiTool), len(fv.PhiCtx))
	}
	// git_force_push: reversibility 2 -> 1.0, base_sensitivity 0.85, blast 3 -> 1.0.
	if fv.PhiTool[0] != 1.0 || fv.PhiTool[1] != 0.85 || fv.PhiTool[2] != 1.0 {
		t.Errorf("phi_tool head = %v", fv.PhiTool[:3])
	}
	// prod_infra sensitivity 1.0, arg_risk 1.
	if fv.PhiCtx[0] != 1.0 || fv.PhiCtx[1] != 1.0 {
		t.Errorf("phi_ctx head = %v", fv.PhiCtx[:2])
	}
	if fv.T != 42 {
		t.Errorf("t = %v", fv.T)
	}
	// Exactly one category and one task one-hot set.
	if sum(fv.PhiTool[3:]) != 1.0 || sum(fv.PhiCtx[2:]) != 1.0 {
		t.Errorf("one-hots: %v %v", fv.PhiTool[3:], fv.PhiCtx[2:])
	}
}

func TestFeaturizeUnknown(t *testing.T) {
	f := trustcalib.New()
	cases := []featurizer.Point{
		{Tool: "nope", Target: "prod_infra", Task: "bugfix"},
		{Tool: "read_file", Target: "nope", Task: "bugfix"},
		{Tool: "read_file", Target: "prod_infra", Task: "nope"},
	}
	for _, p := range cases {
		if _, err := f.Featurize(p); err == nil {
			t.Errorf("expected error for %+v", p)
		}
	}
}

func TestCategoriesSorted(t *testing.T) {
	if len(trustcalib.Categories) != 8 {
		t.Fatalf("categories = %d, want 8", len(trustcalib.Categories))
	}
	for i := 1; i < len(trustcalib.Categories); i++ {
		if trustcalib.Categories[i-1] >= trustcalib.Categories[i] {
			t.Fatalf("categories not sorted: %v", trustcalib.Categories)
		}
	}
}

func sum(xs []float64) float64 {
	s := 0.0
	for _, x := range xs {
		s += x
	}
	return s
}
