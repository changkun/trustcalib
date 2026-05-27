package persist_test

import (
	"math"
	"path/filepath"
	"testing"

	"github.com/changkun/trustcalib/config"
	"github.com/changkun/trustcalib/featurizer"
	"github.com/changkun/trustcalib/featurizer/trustcalib"
	"github.com/changkun/trustcalib/persist"
)

func TestSaveLoadReconstruct(t *testing.T) {
	cfg := config.Default()
	// Refit on every observation so the live model reflects all data; this
	// makes the reload (which refits from all stored points) exactly
	// reproducible, isolating serialization fidelity from refit staleness.
	cfg.Gateway.RefitEvery = 1
	f := trustcalib.New()
	g := cfg.NewGateway(f)

	// Accumulate a mix of approve/deny labels.
	pts := []featurizer.Point{
		{Tool: "read_file", Target: "workspace_src", Task: "bugfix", ArgRisk: 0, T: 0},
		{Tool: "git_force_push", Target: "prod_infra", Task: "ops_maintenance", ArgRisk: 1, T: 1},
		{Tool: "write_file", Target: "workspace_src", Task: "feature_dev", ArgRisk: 0, T: 2},
		{Tool: "execute_sql", Target: "production_db", Task: "data_migration", ArgRisk: 1, T: 3},
		{Tool: "list_dir", Target: "sandbox_tmp", Task: "exploration", ArgRisk: 0, T: 4},
	}
	approve := []bool{true, false, true, false, true}
	for i, p := range pts {
		if err := g.Observe(p, approve[i]); err != nil {
			t.Fatal(err)
		}
	}
	if !g.Fitted() {
		t.Fatal("gateway should be fitted after observations")
	}

	// Decisions before persistence.
	probe := []featurizer.Point{
		{Tool: "delete_file", Target: "workspace_src", Task: "refactor", ArgRisk: 0, T: 10},
		{Tool: "grep_search", Target: "workspace_tests", Task: "bugfix", ArgRisk: 0, T: 11},
	}
	wantDec := make([]string, len(probe))
	wantP := make([]float64, len(probe))
	for i, p := range probe {
		d, ph := g.Decide(p)
		wantDec[i], wantP[i] = d.String(), ph
	}

	st := persist.NewState(cfg).Snapshot(g)
	st.Counter = 5
	path := filepath.Join(t.TempDir(), "state.json")
	if err := persist.Save(path, st); err != nil {
		t.Fatal(err)
	}

	loaded, ok, err := persist.Load(path, cfg)
	if err != nil || !ok {
		t.Fatalf("load: ok=%v err=%v", ok, err)
	}
	if loaded.Counter != 5 || len(loaded.Points) != len(pts) {
		t.Fatalf("state mismatch: counter=%d points=%d", loaded.Counter, len(loaded.Points))
	}

	g2, err := persist.Reconstruct(loaded, trustcalib.New())
	if err != nil {
		t.Fatal(err)
	}
	if !g2.Fitted() {
		t.Fatal("reconstructed gateway should be fitted")
	}
	for i, p := range probe {
		d, ph := g2.Decide(p)
		if d.String() != wantDec[i] {
			t.Errorf("probe %d decision %q, want %q", i, d.String(), wantDec[i])
		}
		if math.Abs(ph-wantP[i]) > 1e-9 {
			t.Errorf("probe %d p_hat %.12g, want %.12g", i, ph, wantP[i])
		}
	}
}
