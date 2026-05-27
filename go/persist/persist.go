// Package persist serializes gateway state to disk so the stateless CLI hook
// can carry the learned model across invocations. The fitted Cholesky is NOT
// serialized; only the raw labelled points, tuning history, thresholds and
// hyperparameters are stored, and the model is refit on load. This keeps the
// state small and consistent across hyperparameter or featurizer changes.
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
}

// NewState returns an empty cold-start state for the given config.
func NewState(cfg config.Config) State {
	return State{
		Version: SchemaVersion,
		Config:  cfg,
		TauLow:  cfg.Gateway.TauLow,
		TauHigh: cfg.Gateway.TauHigh,
	}
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
	var s State
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
// stored points.
func Reconstruct(s State, f featurizer.Featurizer) (*gateway.Gateway, error) {
	g := s.Config.NewGateway(f)
	if err := g.Restore(s.Points, s.Labels, s.PHatHist, s.LabelHist, s.TauLow, s.TauHigh, s.Tuned); err != nil {
		return nil, err
	}
	return g, nil
}

// Snapshot updates a State from a gateway's current state, preserving the
// config and counter.
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
	if s.Version == 0 {
		s.Version = SchemaVersion
	}
	return s
}
