// Package persist serializes gateway state to disk so the stateless CLI hook
// can carry the learned model across invocations. The fitted Cholesky is NOT
// serialized; only the raw labelled points, tuning history, thresholds and
// hyperparameters are stored, and the model is refit on load. This keeps the
// state small and consistent across hyperparameter or featurizer changes.
//
// Fields added with audits (label provenance, audit counters, pending audits
// and the kernel, cost and audit config keys) are optional in the file: a
// state written before they existed loads with every label taken as an
// escalation (propensity 1), zero audit counters and the defaults for every
// config key it lacks.
package persist

import (
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"path/filepath"

	"github.com/changkun/trustcalib/config"
	"github.com/changkun/trustcalib/featurizer"
	"github.com/changkun/trustcalib/gateway"
)

// SchemaVersion is bumped on incompatible State changes.
const SchemaVersion = 1

// MaxPending bounds the pending-audit list; the oldest entries are dropped
// first (an audit whose observe never arrives is simply lost).
const MaxPending = 64

// State is the on-disk gateway snapshot.
type State struct {
	Version   int                `json:"version"`
	Config    config.Config      `json:"config"`
	TauLow    float64            `json:"tau_low"`
	TauHigh   float64            `json:"tau_high"`
	Tuned     bool               `json:"tuned"`
	Counter   int                `json:"counter"` // monotonic time index for the CLI
	Points    []featurizer.Point `json:"points"`
	Labels    []int              `json:"labels"`
	PHatHist  []float64          `json:"phat_hist"`
	LabelHist []int              `json:"label_hist"`

	// LabelInfo is the provenance (source, propensity) of each label,
	// parallel to Points and Labels.
	LabelInfo []gateway.LabelInfo `json:"label_info,omitempty"`
	// Audit holds the cumulative audit counters.
	Audit gateway.AuditStats `json:"audit"`
	// Pending lists auto-decisions sampled for audit whose human answer has
	// not been observed yet (see SetPending).
	Pending []PendingAudit `json:"pending_audits,omitempty"`
}

// PendingAudit is an auto-decision that was sampled for audit and shown to
// the human as an ASK; the next observe of the same request consumes it.
type PendingAudit struct {
	Key      string `json:"key"`      // caller-defined identity of the request
	Decision string `json:"decision"` // the audited auto-decision: "allow" or "block"
	// Propensity is the audit rate in force when the audit was sampled; the
	// label's Horvitz-Thompson weight is 1/Propensity. Zero means unknown
	// (state written before this field existed).
	Propensity float64 `json:"propensity,omitempty"`
}

// NewState returns an empty cold-start state for the given config.
func NewState(cfg config.Config) State {
	low, high := cfg.GatewayConfig().Thresholds()
	return State{
		Version: SchemaVersion,
		Config:  cfg,
		TauLow:  low,
		TauHigh: high,
	}
}

// SetPending records the outcome of a decide for the request identified by
// key: any earlier pending audit for the same key is superseded (removed),
// and if audited is "allow" or "block" a new pending audit is recorded. It
// reports whether the list changed. The list keeps at most MaxPending entries.
func (s *State) SetPending(key, audited string) bool {
	return s.SetPendingWithPropensity(key, audited, 0)
}

// SetPendingWithPropensity is SetPending that also records the audit rate in
// force when the audit was sampled.
func (s *State) SetPendingWithPropensity(key, audited string, propensity float64) bool {
	changed := false
	kept := s.Pending[:0]
	for _, p := range s.Pending {
		if p.Key == key {
			changed = true
			continue
		}
		kept = append(kept, p)
	}
	s.Pending = kept
	if audited != "" {
		s.Pending = append(s.Pending, PendingAudit{Key: key, Decision: audited, Propensity: propensity})
		if len(s.Pending) > MaxPending {
			s.Pending = append([]PendingAudit(nil), s.Pending[len(s.Pending)-MaxPending:]...)
		}
		changed = true
	}
	if len(s.Pending) == 0 {
		s.Pending = nil
	}
	return changed
}

// TakePending removes and returns the pending audit decision for key, if any.
func (s *State) TakePending(key string) (audited string, ok bool) {
	p, ok := s.TakePendingAudit(key)
	return p.Decision, ok
}

// TakePendingAudit removes and returns the pending audit for key, if any,
// including the propensity recorded when it was sampled.
func (s *State) TakePendingAudit(key string) (PendingAudit, bool) {
	for i, p := range s.Pending {
		if p.Key == key {
			s.Pending = append(s.Pending[:i:i], s.Pending[i+1:]...)
			if len(s.Pending) == 0 {
				s.Pending = nil
			}
			return p, true
		}
	}
	return PendingAudit{}, false
}

// Load reads a State from path. A missing file yields a cold-start state for
// the given fallback config (ok=false signals the file did not exist).
func Load(path string, fallback config.Config) (State, bool, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return NewState(fallback), false, nil
		}
		return State{}, false, err
	}
	// Unmarshal over the defaults so that config keys a state file predates
	// keep their default values rather than zero.
	s := State{Config: config.Default()}
	if err := json.Unmarshal(data, &s); err != nil {
		return State{}, false, err
	}
	return s, true, nil
}

// Save writes a State atomically (temp file + rename).
func Save(path string, s State) error {
	if s.Version == 0 {
		s.Version = SchemaVersion
	}
	data, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return err
	}
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(dir, ".trustcalib-state-*.tmp")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpName, path)
}

// Reconstruct builds a gateway from a State, refitting the model from the
// stored points and reinstating the audit counters and label provenance.
func Reconstruct(s State, f featurizer.Featurizer) (*gateway.Gateway, error) {
	g := s.Config.NewGateway(f)
	if err := g.Restore(s.Points, s.Labels, s.PHatHist, s.LabelHist, s.TauLow, s.TauHigh, s.Tuned); err != nil {
		return nil, err
	}
	g.RestoreAudit(s.Audit, s.LabelInfo)
	return g, nil
}

// Snapshot updates a State from a gateway's current state, preserving the
// config, counter and pending audits.
func (s State) Snapshot(g *gateway.Gateway) State {
	pts, y := g.TrainingData()
	ph, lab := g.History()
	low, high := g.Thresholds()
	s.Points = pts
	s.Labels = y
	s.PHatHist = ph
	s.LabelHist = lab
	s.TauLow = low
	s.TauHigh = high
	s.Tuned = g.Tuned()
	s.LabelInfo = g.Labels()
	s.Audit = g.AuditStats()
	if s.Version == 0 {
		s.Version = SchemaVersion
	}
	return s
}
