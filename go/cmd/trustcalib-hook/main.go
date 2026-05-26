// Command trustcalib-hook is a thin CLI around the trustcalib gateway, intended
// to be wired as an agent-harness PreToolUse hook. It persists the learned
// model across invocations in a JSON state file.
//
// Subcommands:
//
//	decide   read a tool-call point on stdin, print {decision, p_hat}
//	observe  record a human approve/deny on stdin, refit, persist
//	tune     recompute the allow/ask/block thresholds from history
//	stats    print the current model status
//
// Flags: --config <path> (YAML hyperparameters), --state <path> (JSON state).
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"golang.org/x/sys/unix"

	"github.com/changkun/trustcalib/config"
	"github.com/changkun/trustcalib/featurizer"
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
}

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
	unlock, err := lockState(statePath, false)
	if err != nil {
		return err
	}
	defer unlock()

	_, g, st, err := loadGateway(configPath, statePath)
	if err != nil {
		return err
	}

	var req request
	if err := json.NewDecoder(os.Stdin).Decode(&req); err != nil {
		return fmt.Errorf("decode stdin: %w", err)
	}
	p := point(req, st.Counter)
	dec, pHat := g.Decide(p)

	return emit(map[string]any{"decision": dec.String(), "p_hat": pHat})
}

func runObserve(configPath, statePath string) error {
	unlock, err := lockState(statePath, true)
	if err != nil {
		return err
	}
	defer unlock()

	_, g, st, err := loadGateway(configPath, statePath)
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
	if err := g.Observe(p, *req.Approved); err != nil {
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
	return emit(map[string]any{
		"ok":         true,
		"num_labels": g.NumLabels(),
		"fitted":     g.Fitted(),
		"tau_low":    low,
		"tau_high":   high,
	})
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
	return emit(map[string]any{
		"fitted":     g.Fitted(),
		"tuned":      g.Tuned(),
		"num_labels": g.NumLabels(),
		"counter":    st.Counter,
		"tau_low":    low,
		"tau_high":   high,
	})
}

// loadGateway loads config + state and reconstructs the gateway (refitting from
// stored points).
func loadGateway(configPath, statePath string) (config.Config, *gateway.Gateway, persist.State, error) {
	cfg, err := config.Load(configPath)
	if err != nil {
		return cfg, nil, persist.State{}, fmt.Errorf("load config: %w", err)
	}
	st, _, err := persist.Load(statePath, cfg)
	if err != nil {
		return cfg, nil, persist.State{}, fmt.Errorf("load state: %w", err)
	}
	g, err := persist.Reconstruct(st, trustcalib.New())
	if err != nil {
		return cfg, nil, st, fmt.Errorf("reconstruct gateway: %w", err)
	}
	return cfg, g, st, nil
}

func point(req request, counter int) featurizer.Point {
	t := float64(counter)
	if req.T != nil {
		t = *req.T
	}
	return featurizer.Point{
		Tool:    req.Tool,
		Target:  req.Target,
		Task:    req.Task,
		ArgRisk: req.ArgRisk,
		T:       t,
	}
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
