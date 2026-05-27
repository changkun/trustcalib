package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// runWith redirects stdin/stdout around fn, feeding it stdin and capturing what
// it writes.
func runWith(t *testing.T, stdin string, fn func() error) (string, error) {
	t.Helper()
	dir := t.TempDir()
	inPath := filepath.Join(dir, "in")
	if err := os.WriteFile(inPath, []byte(stdin), 0o644); err != nil {
		t.Fatal(err)
	}
	in, err := os.Open(inPath)
	if err != nil {
		t.Fatal(err)
	}
	defer in.Close()
	out, err := os.Create(filepath.Join(dir, "out"))
	if err != nil {
		t.Fatal(err)
	}

	oldIn, oldOut := os.Stdin, os.Stdout
	os.Stdin, os.Stdout = in, out
	runErr := fn()
	os.Stdin, os.Stdout = oldIn, oldOut
	out.Close()

	data, err := os.ReadFile(filepath.Join(dir, "out"))
	if err != nil {
		t.Fatal(err)
	}
	return string(data), runErr
}

func decode(t *testing.T, s string) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal([]byte(s), &m); err != nil {
		t.Fatalf("decode %q: %v", s, err)
	}
	return m
}

func TestCLIFlow(t *testing.T) {
	dir := t.TempDir()
	state := filepath.Join(dir, "sub", "state.json") // exercises MkdirAll
	cfg := filepath.Join(dir, "missing.yaml")        // absent -> defaults

	// Cold start: ASK at 0.5.
	out, err := runWith(t, `{"tool":"read_file","target":"workspace_src","task":"bugfix","arg_risk":0}`,
		func() error { return runDecide(cfg, state) })
	if err != nil {
		t.Fatal(err)
	}
	if m := decode(t, out); m["decision"] != "ask" || m["p_hat"].(float64) != 0.5 {
		t.Fatalf("cold start: %v", m)
	}

	// Teach a clean separation.
	approve := `{"tool":"read_file","target":"workspace_src","task":"bugfix","arg_risk":0,"approved":true}`
	deny := `{"tool":"git_force_push","target":"prod_infra","task":"ops_maintenance","arg_risk":1,"approved":false}`
	for i := 0; i < 40; i++ {
		if _, err := runWith(t, approve, func() error { return runObserve(cfg, state) }); err != nil {
			t.Fatal(err)
		}
		if _, err := runWith(t, deny, func() error { return runObserve(cfg, state) }); err != nil {
			t.Fatal(err)
		}
	}

	// Stats reports a fitted model with labels.
	out, _ = runWith(t, "", func() error { return runStats(cfg, state) })
	if m := decode(t, out); m["fitted"] != true || m["num_labels"].(float64) < 2 {
		t.Fatalf("stats: %v", m)
	}

	// Tune produces a band.
	out, _ = runWith(t, "", func() error { return runTune(cfg, state) })
	tm := decode(t, out)
	if tm["tau_low"].(float64) >= tm["tau_high"].(float64) {
		t.Fatalf("tune band invalid: %v", tm)
	}

	// Learned decisions: allow the safe op, block the risky one.
	out, _ = runWith(t, `{"tool":"read_file","target":"workspace_src","task":"bugfix","arg_risk":0}`,
		func() error { return runDecide(cfg, state) })
	if m := decode(t, out); m["decision"] != "allow" {
		t.Fatalf("expected allow, got %v", m)
	}
	out, _ = runWith(t, `{"tool":"git_force_push","target":"prod_infra","task":"ops_maintenance","arg_risk":1}`,
		func() error { return runDecide(cfg, state) })
	if m := decode(t, out); m["decision"] != "block" {
		t.Fatalf("expected block, got %v", m)
	}
}

func TestObserveMissingApproved(t *testing.T) {
	dir := t.TempDir()
	state := filepath.Join(dir, "state.json")
	_, err := runWith(t, `{"tool":"read_file","target":"workspace_src","task":"bugfix","arg_risk":0}`,
		func() error { return runObserve("", state) })
	if err == nil || !strings.Contains(err.Error(), "approved") {
		t.Fatalf("expected missing-approved error, got %v", err)
	}
}

func TestDecideBadStdin(t *testing.T) {
	dir := t.TempDir()
	state := filepath.Join(dir, "state.json")
	_, err := runWith(t, `not json`, func() error { return runDecide("", state) })
	if err == nil {
		t.Fatal("expected decode error")
	}
}

func TestExplicitTimestampNotCounted(t *testing.T) {
	dir := t.TempDir()
	state := filepath.Join(dir, "state.json")
	in := `{"tool":"read_file","target":"workspace_src","task":"bugfix","arg_risk":0,"t":99,"approved":true}`
	if _, err := runWith(t, in, func() error { return runObserve("", state) }); err != nil {
		t.Fatal(err)
	}
	out, _ := runWith(t, "", func() error { return runStats("", state) })
	if m := decode(t, out); m["counter"].(float64) != 0 {
		t.Fatalf("explicit t should not advance counter, got %v", m["counter"])
	}
}

func TestMainDispatch(t *testing.T) {
	dir := t.TempDir()
	state := filepath.Join(dir, "state.json")
	oldArgs := os.Args
	os.Args = []string{"trustcalib-hook", "stats", "--state", state}
	defer func() { os.Args = oldArgs }()

	out, _ := runWith(t, "", func() error {
		main()
		return nil
	})
	if m := decode(t, out); m["fitted"] != false {
		t.Fatalf("stats on cold state: %v", m)
	}
}

func TestPointDefaults(t *testing.T) {
	p := point(request{Tool: "x"}, 7)
	if p.T != 7 {
		t.Fatalf("expected counter fallback t=7, got %v", p.T)
	}
	tval := 3.5
	p = point(request{Tool: "x", T: &tval}, 7)
	if p.T != 3.5 {
		t.Fatalf("expected explicit t=3.5, got %v", p.T)
	}
}

func TestDefaultPaths(t *testing.T) {
	t.Setenv("XDG_STATE_HOME", "/xdg/state")
	t.Setenv("XDG_CONFIG_HOME", "/xdg/config")
	if got := defaultStatePath(); got != "/xdg/state/trustcalib/state.json" {
		t.Errorf("state path: %s", got)
	}
	if got := defaultConfigPath(); got != "/xdg/config/trustcalib/config.yaml" {
		t.Errorf("config path: %s", got)
	}

	t.Setenv("XDG_STATE_HOME", "")
	t.Setenv("XDG_CONFIG_HOME", "")
	if got := defaultStatePath(); !strings.HasSuffix(got, "trustcalib/state.json") {
		t.Errorf("fallback state path: %s", got)
	}
	if got := defaultConfigPath(); !strings.HasSuffix(got, "trustcalib/config.yaml") {
		t.Errorf("fallback config path: %s", got)
	}
}
