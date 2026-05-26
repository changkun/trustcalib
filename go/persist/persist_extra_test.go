package persist_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/changkun/trustcalib/config"
	"github.com/changkun/trustcalib/featurizer/trustcalib"
	"github.com/changkun/trustcalib/persist"
)

func TestLoadMissingReturnsColdStart(t *testing.T) {
	cfg := config.Default()
	st, ok, err := persist.Load(filepath.Join(t.TempDir(), "none.json"), cfg)
	if err != nil {
		t.Fatal(err)
	}
	if ok {
		t.Error("ok should be false for a missing file")
	}
	if st.Version != persist.SchemaVersion || st.TauLow != cfg.Gateway.TauLow {
		t.Fatalf("cold-start state wrong: %+v", st)
	}
}

func TestLoadCorrupt(t *testing.T) {
	path := filepath.Join(t.TempDir(), "state.json")
	if err := os.WriteFile(path, []byte("{not json"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, _, err := persist.Load(path, config.Default()); err == nil {
		t.Fatal("expected unmarshal error on corrupt state")
	}
}

func TestReconstructEmpty(t *testing.T) {
	cfg := config.Default()
	st := persist.NewState(cfg)
	g, err := persist.Reconstruct(st, trustcalib.New())
	if err != nil {
		t.Fatal(err)
	}
	if g.Fitted() {
		t.Error("empty state should reconstruct an unfitted gateway")
	}
}

func TestSaveVersionDefaulted(t *testing.T) {
	path := filepath.Join(t.TempDir(), "state.json")
	st := persist.State{} // Version 0
	if err := persist.Save(path, st); err != nil {
		t.Fatal(err)
	}
	loaded, ok, err := persist.Load(path, config.Default())
	if err != nil || !ok {
		t.Fatalf("load: ok=%v err=%v", ok, err)
	}
	if loaded.Version != persist.SchemaVersion {
		t.Fatalf("version not defaulted on save: %d", loaded.Version)
	}
}
