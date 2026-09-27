// Command trustcalib-hook is a thin CLI around the trustcalib gateway, intended
// to be wired as an agent-harness PreToolUse hook. It persists the learned
// model across invocations in a JSON state file.
//
// Subcommands:
//
//	decide   read a tool-call point on stdin, print {decision, p_hat}
//	observe  record a human approve/deny on stdin, refit, persist
//	tune     recompute the allow/ask/block thresholds from history
//	stats    print the current model status and audit estimates
//
// Flags: --config <path> (YAML hyperparameters), --state <path> (JSON state).
// When the config file exists it takes precedence over the config recorded in
// the state; otherwise the state's recorded config is used.
//
// Audits. With gateway.audit_rate > 0, decide samples auto-decisions for a
// random audit. A sampled decision is printed as
// {"decision": "ask", "audit": true, "audited_decision": "allow"|"block"}, so
// the harness shows it to the human like any other ASK, and the audit is
// remembered in the state as pending, keyed by the request's identity (every
// request field except "t" and "approved"; "t" too when it is given
// explicitly). The next observe of the same request consumes it and records
// the label with the audit propensity instead of 1, so the harness needs no
// extra bookkeeping. A later decide of the same request supersedes (drops) the
// pending audit, and at most persist.MaxPending audits are kept.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"

	"golang.org/x/sys/unix"

	"github.com/changkun/trustcalib/config"
	"github.com/changkun/trustcalib/featurizer"
	"github.com/changkun/trustcalib/featurizer/bashmap"
	"github.com/changkun/trustcalib/featurizer/judge"
	"github.com/changkun/trustcalib/featurizer/trustcalib"
	"github.com/changkun/trustcalib/gateway"
	"github.com/changkun/trustcalib/persist"
)

type request struct {
	Tool     string   `json:"tool"`
	Target   string   `json:"target"`
	Task     string   `json:"task"`
	ArgRisk  int      `json:"arg_risk"`
	T        *float64 `json:"t"`
	Approved *bool    `json:"approved"`

	// Judge featurizer inputs. Category is derived from Command (via
	// bashmap.CategoryFromBash) when it is empty.
	JudgeVerdict string   `json:"judge_verdict"`
	JudgeScore   *float64 `json:"judge_score"`
	Category     string   `json:"category"`
	Command      string   `json:"command"`
}

// uniform draws the audit coin; math/rand/v2's global source is randomly
// seeded. Tests replace it.
var uniform = rand.Float64

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: trustcalib-hook <decide|observe|tune|stats> [flags]")
		os.Exit(2)
	}
	sub := os.Args[1]

	fs := flag.NewFlagSet(sub, flag.ExitOnError)
	configPath := fs.String("config", defaultConfigPath(), "path to YAML config")
	statePath := fs.String("state", defaultStatePath(), "path to JSON state file")
	_ = fs.Parse(os.Args[2:])

	var err error
	switch sub {
	case "decide":
		err = runDecide(*configPath, *statePath)
	case "observe":
		err = runObserve(*configPath, *statePath)
	case "tune":
		err = runTune(*configPath, *statePath)
	case "stats":
		err = runStats(*configPath, *statePath)
	default:
		fmt.Fprintf(os.Stderr, "unknown subcommand %q\n", sub)
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "trustcalib-hook:", err)
		os.Exit(1)
	}
}

func runDecide(configPath, statePath string) error {
	// decide writes: it counts auto-decisions (the audit denominators) and
	// records pending audits.
	unlock, err := lockState(statePath, true)
	if err != nil {
		return err
	}
	defer unlock()

	cfg, g, st, err := loadGateway(configPath, statePath)
	if err != nil {
		return err
	}

	var req request
	if err := json.NewDecoder(os.Stdin).Decode(&req); err != nil {
		return fmt.Errorf("decode stdin: %w", err)
	}
	p := point(req, st.Counter)
	dec, pHat, audit := g.DecideAudit(p, uniform())

	audited := ""
	if audit {
		audited = dec.String()
	}
	// Record the audit rate in force now: the label's propensity is the
	// probability with which this action was audited, even if the config
	// changes before the human answers.
	changed := st.SetPendingWithPropensity(requestKey(req), audited, cfg.Gateway.AuditRate)
	if dec != gateway.Ask || changed {
		st = st.Snapshot(g)
		if err := persist.Save(statePath, st); err != nil {
			return err
		}
	}

	out := map[string]any{"decision": dec.String(), "p_hat": pHat}
	if audit {
		// Show the auto-decision to the human as an ordinary ASK.
		out["decision"] = gateway.Ask.String()
		out["audit"] = true
		out["audited_decision"] = audited
	}
	return emit(out)
}

func runObserve(configPath, statePath string) error {
	unlock, err := lockState(statePath, true)
	if err != nil {
		return err
	}
	defer unlock()

	cfg, g, st, err := loadGateway(configPath, statePath)
	if err != nil {
		return err
	}

	var req request
	if err := json.NewDecoder(os.Stdin).Decode(&req); err != nil {
		return fmt.Errorf("decode stdin: %w", err)
	}
	if req.Approved == nil {
		return fmt.Errorf("observe: missing \"approved\" field")
	}
	p := point(req, st.Counter)
	pending, isAudit := st.TakePendingAudit(requestKey(req))
	prop := pending.Propensity
	if !(prop > 0) {
		prop = cfg.Gateway.AuditRate // pending audit from an older state file
	}
	isAudit = isAudit && prop > 0
	if isAudit {
		d := gateway.Allow
		if pending.Decision == gateway.Block.String() {
			d = gateway.Block
		}
		err = g.ObserveAuditWithPropensity(p, *req.Approved, d, prop)
	} else {
		err = g.Observe(p, *req.Approved)
	}
	if err != nil {
		return err
	}

	st = st.Snapshot(g)
	if req.T == nil {
		st.Counter++ // advance the monotonic time index
	}
	if err := persist.Save(statePath, st); err != nil {
		return err
	}

	low, high := g.Thresholds()
	out := map[string]any{
		"ok":         true,
		"num_labels": g.NumLabels(),
		"fitted":     g.Fitted(),
		"tau_low":    low,
		"tau_high":   high,
	}
	if isAudit {
		out["audit"] = true
	}
	return emit(out)
}

func runTune(configPath, statePath string) error {
	unlock, err := lockState(statePath, true)
	if err != nil {
		return err
	}
	defer unlock()

	_, g, st, err := loadGateway(configPath, statePath)
	if err != nil {
		return err
	}
	low, high := g.Tune()
	st = st.Snapshot(g)
	if err := persist.Save(statePath, st); err != nil {
		return err
	}
	return emit(map[string]any{"tau_low": low, "tau_high": high, "num_labels": g.NumLabels()})
}

func runStats(configPath, statePath string) error {
	unlock, err := lockState(statePath, false)
	if err != nil {
		return err
	}
	defer unlock()

	_, g, st, err := loadGateway(configPath, statePath)
	if err != nil {
		return err
	}
	low, high := g.Thresholds()
	fa, audits, denied := g.FalseAllowEstimate()
	fb, auditsBlock, approvedBlock := g.FalseBlockEstimate()
	alpha, delta := st.Config.Certify()
	a := g.AuditStats()
	return emit(map[string]any{
		"fitted":     g.Fitted(),
		"tuned":      g.Tuned(),
		"num_labels": g.NumLabels(),
		"counter":    st.Counter,
		"tau_low":    low,
		"tau_high":   high,

		"kernel":     st.Config.KernelType(),
		"featurizer": st.Config.FeaturizerName(),
		"audit_rate": st.Config.Gateway.AuditRate,

		// Horvitz-Thompson estimates from random audits (null when undefined).
		"n_allow":              a.NAllow,
		"n_block":              a.NBlock,
		"audits":               audits,
		"audit_denied":         denied,
		"false_allow_estimate": nullIfNaN(fa),
		"audits_block":         auditsBlock,
		"audit_block_approved": approvedBlock,
		"false_block_estimate": nullIfNaN(fb),
		"pending_audits":       len(st.Pending),

		// Certification of the false-allow rate (Proposition 5(b)).
		"clean_streak":  a.CleanStreak,
		"certify_alpha": alpha,
		"certify_delta": delta,
		"certify_n":     gateway.CertificationSize(alpha, delta),
		"certified":     g.CertifiedFalseAllowBelow(alpha, delta),
	})
}

// nullIfNaN maps NaN (an undefined estimate) to JSON null.
func nullIfNaN(v float64) any {
	if math.IsNaN(v) {
		return nil
	}
	return v
}

// loadGateway loads config + state and reconstructs the gateway (refitting from
// stored points) with the configured kernel and featurizer. An existing config
// file takes precedence over the config recorded in the state, so edits to it
// (kernel, costs, audit rate, featurizer, ...) apply to an existing state; the
// model is refit from the raw labelled points on every load anyway.
func loadGateway(configPath, statePath string) (config.Config, *gateway.Gateway, persist.State, error) {
	cfg, err := config.Load(configPath)
	if err != nil {
		return cfg, nil, persist.State{}, fmt.Errorf("load config: %w", err)
	}
	st, _, err := persist.Load(statePath, cfg)
	if err != nil {
		return cfg, nil, persist.State{}, fmt.Errorf("load state: %w", err)
	}
	if _, err := os.Stat(configPath); err == nil {
		st.Config = cfg
	}
	g, err := persist.Reconstruct(st, newFeaturizer(st.Config))
	if err != nil {
		return cfg, nil, st, fmt.Errorf("reconstruct gateway: %w", err)
	}
	return cfg, g, st, nil
}

// newFeaturizer returns the featurizer named by the config.
func newFeaturizer(cfg config.Config) featurizer.Featurizer {
	if cfg.FeaturizerName() == config.FeaturizerJudge {
		return judge.New()
	}
	return trustcalib.New()
}

func point(req request, counter int) featurizer.Point {
	t := float64(counter)
	if req.T != nil {
		t = *req.T
	}
	category := req.Category
	if category == "" && req.Command != "" {
		category = bashmap.CategoryFromBash(req.Command)
	}
	return featurizer.Point{
		Tool:         req.Tool,
		Target:       req.Target,
		Task:         req.Task,
		ArgRisk:      req.ArgRisk,
		T:            t,
		JudgeVerdict: req.JudgeVerdict,
		JudgeScore:   req.JudgeScore,
		Category:     category,
	}
}

// requestKey is the identity under which a pending audit is remembered
// between decide and observe: every request field except "approved", and
// except "t" unless it was given explicitly (the implicit counter only
// advances on observe, so it is the same for a decide and its observe).
func requestKey(req request) string {
	req.Approved = nil
	b, _ := json.Marshal(req) // cannot fail: plain fields only
	return string(b)
}

func emit(v map[string]any) error {
	enc := json.NewEncoder(os.Stdout)
	return enc.Encode(v)
}

// lockState takes a shared (read) or exclusive (write) advisory lock on a lock
// file alongside the state file, so concurrent hook processes serialize their
// read-modify-write cycles.
func lockState(statePath string, exclusive bool) (func(), error) {
	if err := os.MkdirAll(filepath.Dir(statePath), 0o755); err != nil {
		return nil, err
	}
	lockPath := statePath + ".lock"
	fd, err := os.OpenFile(lockPath, os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		return nil, err
	}
	how := unix.LOCK_SH
	if exclusive {
		how = unix.LOCK_EX
	}
	if err := unix.Flock(int(fd.Fd()), how); err != nil {
		fd.Close()
		return nil, err
	}
	return func() {
		unix.Flock(int(fd.Fd()), unix.LOCK_UN)
		fd.Close()
	}, nil
}

func defaultStatePath() string {
	if dir := os.Getenv("XDG_STATE_HOME"); dir != "" {
		return filepath.Join(dir, "trustcalib", "state.json")
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".local", "state", "trustcalib", "state.json")
}

func defaultConfigPath() string {
	if dir := os.Getenv("XDG_CONFIG_HOME"); dir != "" {
		return filepath.Join(dir, "trustcalib", "config.yaml")
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".config", "trustcalib", "config.yaml")
}
