package main

import (
	"encoding/json"
	"fmt"
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

func writeConfig(t *testing.T, dir, yaml string) string {
	t.Helper()
	path := filepath.Join(dir, "config.yaml")
	if err := os.WriteFile(path, []byte(yaml), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func setUniform(t *testing.T, u float64) {
	t.Helper()
	old := uniform
	uniform = func() float64 { return u }
	t.Cleanup(func() { uniform = old })
}

func mustRun(t *testing.T, stdin string, fn func() error) map[string]any {
	t.Helper()
	out, err := runWith(t, stdin, fn)
	if err != nil {
		t.Fatal(err)
	}
	return decode(t, out)
}

const (
	safeReq  = `{"tool":"read_file","target":"workspace_src","task":"bugfix","arg_risk":0}`
	riskyReq = `{"tool":"git_force_push","target":"prod_infra","task":"ops_maintenance","arg_risk":1}`
)

func withApproved(req string, approved bool) string {
	return strings.TrimSuffix(req, "}") + fmt.Sprintf(`,"approved":%v}`, approved)
}

// TestCLIAuditFlow: sampled audits are shown as ASK, remembered as pending,
// tagged with the audit propensity on observe, and reported by stats with the
// Horvitz-Thompson estimate and the certification test.
func TestCLIAuditFlow(t *testing.T) {
	dir := t.TempDir()
	state := filepath.Join(dir, "state.json")
	cfg := writeConfig(t, dir, "gateway:\n  refit_every: 1\n  audit_rate: 0.5\n  certify_alpha: 0.5\n  certify_delta: 0.5\n")
	decide := func(req string) map[string]any { return mustRun(t, req, func() error { return runDecide(cfg, state) }) }
	observe := func(req string, approved bool) map[string]any {
		return mustRun(t, withApproved(req, approved), func() error { return runObserve(cfg, state) })
	}
	stats := func() map[string]any { return mustRun(t, "", func() error { return runStats(cfg, state) }) }

	for i := 0; i < 20; i++ {
		observe(safeReq, true)
		observe(riskyReq, false)
	}
	setUniform(t, 0.99)
	if m := decide(safeReq); m["decision"] != "allow" || m["audit"] != nil {
		t.Fatalf("unsampled allow: %v", m)
	}

	setUniform(t, 0)
	m := decide(safeReq)
	if m["decision"] != "ask" || m["audit"] != true || m["audited_decision"] != "allow" {
		t.Fatalf("sampled audit should be shown as ask: %v", m)
	}
	if s := stats(); s["pending_audits"].(float64) != 1 || s["n_allow"].(float64) != 2 {
		t.Fatalf("pending/n_allow: %v", s)
	}
	if m := observe(safeReq, true); m["audit"] != true {
		t.Fatalf("observe should consume the pending audit: %v", m)
	}
	s := stats()
	if s["audits"].(float64) != 1 || s["audit_denied"].(float64) != 0 || s["clean_streak"].(float64) != 1 ||
		s["pending_audits"].(float64) != 0 || s["certified"] != false || s["certify_n"].(float64) != 2 {
		t.Fatalf("after one clean audit: %v", s)
	}
	if s["false_allow_estimate"].(float64) != 0 {
		t.Fatalf("estimate after a clean audit: %v", s["false_allow_estimate"])
	}

	decide(safeReq)
	observe(safeReq, true)
	if s := stats(); s["certified"] != true {
		t.Fatalf("two clean audits certify alpha=0.5 at delta=0.5: %v", s)
	}

	decide(safeReq)
	observe(safeReq, false)
	s = stats()
	if s["certified"] != false || s["clean_streak"].(float64) != 0 || s["audit_denied"].(float64) != 1 {
		t.Fatalf("a denied audit resets certification: %v", s)
	}
	// HT: (1 denied / 0.5) / 4 auto-ALLOWs.
	if got := s["false_allow_estimate"].(float64); got != 0.5 {
		t.Fatalf("false_allow_estimate = %v, want 0.5", got)
	}

	// A later decide of the same request supersedes a pending audit, so its
	// observe is an ordinary escalation.
	decide(safeReq)
	setUniform(t, 0.99)
	decide(safeReq)
	if m := observe(safeReq, true); m["audit"] != nil {
		t.Fatalf("superseded audit must not tag the label: %v", m)
	}

	// Audits of auto-BLOCKs are tracked separately.
	setUniform(t, 0)
	if m := decide(riskyReq); m["audited_decision"] != "block" {
		t.Fatalf("expected an audited block: %v", m)
	}
	observe(riskyReq, false)
	if s := stats(); s["audits_block"].(float64) != 1 || s["audits"].(float64) != 3 {
		t.Fatalf("block audit: %v", s)
	}
}

// TestCLIJudgeFeaturizer drives the hook with judge verdicts and shell
// commands, the additive kernel and cost-derived thresholds from the config.
func TestCLIJudgeFeaturizer(t *testing.T) {
	dir := t.TempDir()
	state := filepath.Join(dir, "state.json")
	cfg := writeConfig(t, dir, `featurizer: judge
kernel:
  type: additive
gateway:
  refit_every: 1
  costs: {false_allow: 10, false_block: 4, ask: 1}
`)
	setUniform(t, 0.99)
	read := `{"judge_verdict":"allow","command":"cat notes.txt"}`
	exec := `{"judge_verdict":"allow","command":"./cleanup.sh --all"}`
	for i := 0; i < 30; i++ {
		mustRun(t, withApproved(read, true), func() error { return runObserve(cfg, state) })
		mustRun(t, withApproved(exec, false), func() error { return runObserve(cfg, state) })
	}
	if m := mustRun(t, `{"judge_verdict":"allow","command":"head README.md"}`, func() error { return runDecide(cfg, state) }); m["decision"] != "allow" {
		t.Fatalf("judge-allowed read should be auto-allowed: %v", m)
	}
	if m := mustRun(t, `{"judge_verdict":"allow","category":"exec"}`, func() error { return runDecide(cfg, state) }); m["decision"] == "allow" {
		t.Fatalf("judge-allowed exec this supervisor denies must not be auto-allowed: %v", m)
	}
	// No verdict and no score: the featurizer errors and the hook fails safe.
	if m := mustRun(t, `{"command":"ls"}`, func() error { return runDecide(cfg, state) }); m["decision"] != "ask" || m["p_hat"].(float64) != 0.5 {
		t.Fatalf("missing verdict should fail safe to ask: %v", m)
	}
	s := mustRun(t, "", func() error { return runStats(cfg, state) })
	if s["kernel"] != "additive" || s["featurizer"] != "judge" || s["tau_low"].(float64) != 0.25 || s["tau_high"].(float64) != 0.9 {
		t.Fatalf("stats: %v", s)
	}
	if tm := mustRun(t, "", func() error { return runTune(cfg, state) }); tm["tau_low"].(float64) != 0.25 || tm["tau_high"].(float64) != 0.9 {
		t.Fatalf("tune with costs must return the cost band: %v", tm)
	}
}

// TestConfigFileOverridesState: an existing config file applies to a state
// created under different settings.
func TestConfigFileOverridesState(t *testing.T) {
	dir := t.TempDir()
	state := filepath.Join(dir, "state.json")
	missing := filepath.Join(dir, "missing.yaml")
	mustRun(t, withApproved(safeReq, true), func() error { return runObserve(missing, state) })
	if s := mustRun(t, "", func() error { return runStats(missing, state) }); s["tau_low"].(float64) != 0.35 {
		t.Fatalf("default band expected: %v", s)
	}
	cfg := writeConfig(t, dir, "gateway:\n  costs: {false_allow: 10, false_block: 4, ask: 1}\n")
	if s := mustRun(t, "", func() error { return runStats(cfg, state) }); s["tau_low"].(float64) != 0.25 || s["tau_high"].(float64) != 0.9 {
		t.Fatalf("config file costs should apply to the existing state: %v", s)
	}
}

func TestRequestKey(t *testing.T) {
	yes, no := true, false
	t1, t2 := 1.0, 2.0
	a := request{Tool: "x", Command: "ls", Approved: &yes}
	b := request{Tool: "x", Command: "ls", Approved: &no}
	if requestKey(a) != requestKey(b) {
		t.Error("approved must not affect the key")
	}
	if requestKey(request{Tool: "x", Command: "ls"}) == requestKey(request{Tool: "x", Command: "rm"}) {
		t.Error("the command must affect the key")
	}
	if requestKey(request{Tool: "x", T: &t1}) == requestKey(request{Tool: "x", T: &t2}) {
		t.Error("an explicit t must affect the key")
	}
}

func TestPointJudgeFields(t *testing.T) {
	s := 0.7
	p := point(request{JudgeVerdict: "block", JudgeScore: &s, Command: "terraform apply"}, 0)
	if p.JudgeVerdict != "block" || p.JudgeScore == nil || *p.JudgeScore != 0.7 || p.Category != "deploy" {
		t.Fatalf("judge fields: %+v", p)
	}
	if p := point(request{Category: "read", Command: "terraform apply"}, 0); p.Category != "read" {
		t.Fatalf("an explicit category wins over the command: %+v", p)
	}
}
