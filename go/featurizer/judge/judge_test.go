package judge_test

import (
	"testing"

	"github.com/changkun/trustcalib/featurizer"
	"github.com/changkun/trustcalib/featurizer/bashmap"
	"github.com/changkun/trustcalib/featurizer/judge"
	"github.com/changkun/trustcalib/featurizer/trustcalib"
	"github.com/changkun/trustcalib/gateway"
	"github.com/changkun/trustcalib/gp"
	"github.com/changkun/trustcalib/kernel"
)

func score(v float64) *float64 { return &v }

func TestFeaturizeVerdictAndScore(t *testing.T) {
	f := judge.New()
	if f.DimTool() != 10 || f.DimCtx() != 1 {
		t.Fatalf("dims = (%d, %d), want (10, 1)", f.DimTool(), f.DimCtx())
	}
	cases := []struct {
		p    featurizer.Point
		feat float64
		cat  int
	}{
		{featurizer.Point{JudgeVerdict: "allow", Category: "read", T: 3}, 0, 0},
		{featurizer.Point{JudgeVerdict: "BLOCK", Category: "deploy"}, 1, 7},
		// A score takes precedence over the verdict.
		{featurizer.Point{JudgeVerdict: "block", JudgeScore: score(0.3), Category: "exec"}, 0.3, 3},
		{featurizer.Point{JudgeScore: score(0), Category: " Write "}, 0, 4},
		// Empty and unknown categories fall into "other".
		{featurizer.Point{JudgeVerdict: "allow"}, 0, 8},
		{featurizer.Point{JudgeVerdict: "allow", Category: "k8s"}, 0, 8},
	}
	for _, c := range cases {
		fv, err := f.Featurize(c.p)
		if err != nil {
			t.Fatalf("Featurize(%+v): %v", c.p, err)
		}
		if len(fv.PhiTool) != 10 || len(fv.PhiCtx) != 1 || fv.PhiCtx[0] != 0 || fv.T != c.p.T {
			t.Fatalf("shape/ctx/t wrong: %+v", fv)
		}
		if fv.PhiTool[0] != c.feat {
			t.Errorf("%+v: judge feature %v, want %v", c.p, fv.PhiTool[0], c.feat)
		}
		for i, v := range fv.PhiTool[1:] {
			want := 0.0
			if i == c.cat {
				want = 1
			}
			if v != want {
				t.Errorf("%+v: one-hot[%d] = %v, want %v", c.p, i, v, want)
			}
		}
	}
}

func TestFeaturizeErrors(t *testing.T) {
	f := judge.New()
	bad := []featurizer.Point{
		{Category: "read"},                        // neither verdict nor score
		{JudgeVerdict: "maybe", Category: "read"}, // unknown verdict
		{JudgeScore: score(1.5)},                  // out of range
		{JudgeScore: score(-0.1)},
	}
	for _, p := range bad {
		if _, err := f.Featurize(p); err == nil {
			t.Errorf("Featurize(%+v) should fail", p)
		}
	}
}

// TestCategoriesCoverTaxonomy: every bundled tool category (and so every
// bashmap.CategoryFromBash result) has its own one-hot slot.
func TestCategoriesCoverTaxonomy(t *testing.T) {
	for _, c := range trustcalib.Categories {
		if judge.NormalizeCategory(c) != c {
			t.Errorf("taxonomy category %q not in judge.Categories", c)
		}
	}
	for _, cmd := range []string{"ls", "rm -rf x", "terraform apply", "psql db", "git push", "make"} {
		if c := bashmap.CategoryFromBash(cmd); judge.NormalizeCategory(c) == "other" {
			t.Errorf("CategoryFromBash(%q) = %q maps to other", cmd, c)
		}
	}
}

// TestGatewayLearnsJudgeReliability: on top of a judge, the gateway learns
// from the supervisor's labels that the judge's ALLOWs are trustworthy for
// reads but not for exec commands (a blind spot this supervisor keeps
// denying), and escalates or blocks the latter instead of auto-allowing it.
func TestGatewayLearnsJudgeReliability(t *testing.T) {
	cfg := gateway.DefaultConfig()
	cfg.RefitEvery = 1
	g := gateway.New(judge.New(), gp.NewLaplaceGPC(kernel.DefaultAdditiveKernel()), cfg)
	for i := 0; i < 25; i++ {
		read := featurizer.Point{JudgeVerdict: "allow", Category: bashmap.CategoryFromBash("cat notes.txt"), T: float64(2 * i)}
		exec := featurizer.Point{JudgeVerdict: "allow", Category: bashmap.CategoryFromBash("./cleanup.sh --all"), T: float64(2*i + 1)}
		if err := g.Observe(read, true); err != nil {
			t.Fatal(err)
		}
		if err := g.Observe(exec, false); err != nil {
			t.Fatal(err)
		}
	}
	read := featurizer.Point{JudgeVerdict: "allow", Category: "read", T: 50}
	exec := featurizer.Point{JudgeVerdict: "allow", Category: "exec", T: 50}
	if d, p := g.Decide(read); d != gateway.Allow {
		t.Errorf("judge-allowed read: %v (p_hat %.3f), want allow", d, p)
	}
	if d, p := g.Decide(exec); d == gateway.Allow {
		t.Errorf("judge-allowed exec the supervisor denies: %v (p_hat %.3f), must not auto-allow", d, p)
	}
}
