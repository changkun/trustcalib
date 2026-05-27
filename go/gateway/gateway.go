// Package gateway implements the three-tier allow/ask/block policy and the
// online learning loop from Section 5 of the manuscript, redesigned from the
// experiment's batch oracle loop (experiment/gateway.py run_gateway) into a
// stateful object driven by real human approve/deny feedback.
//
// Given the posterior-predictive approval probability p_hat the gateway emits
//
//	ALLOW  if p_hat > tau_high
//	BLOCK  if p_hat < tau_low
//	ASK    otherwise
//
// The ASK band is where the human is queried; only observed (queried) points
// enter the training set. The model refits online as labels accumulate; the
// thresholds are tuned from the collected (p_hat, human-label) history.
package gateway

import (
	"github.com/changkun/trustcalib/featurizer"
	"github.com/changkun/trustcalib/gp"
)

// Decision is a three-tier policy outcome.
type Decision int

const (
	Ask Decision = iota
	Allow
	Block
)

func (d Decision) String() string {
	switch d {
	case Allow:
		return "allow"
	case Block:
		return "block"
	default:
		return "ask"
	}
}

// Config holds the policy hyperparameters.
type Config struct {
	TauLow     float64 // initial/default block threshold
	TauHigh    float64 // initial/default allow threshold
	RefitEvery int     // refit after this many new labels
	SafetyEps  float64 // false-allow cap for tuning (tightened to /2)
	BlockEps   float64 // false-block cap for tuning
	MaxTrain   int     // sliding-window cap on training points (0 = unbounded)
	MaxHist    int     // sliding-window cap on tuning history (0 = unbounded)
	MinTuneN   int     // minimum labels before Tune leaves the default band
}

// DefaultConfig returns the manuscript defaults plus sane caps for a long-
// running deployment.
func DefaultConfig() Config {
	return Config{
		TauLow:     0.35,
		TauHigh:    0.65,
		RefitEvery: 8,
		SafetyEps:  0.02,
		BlockEps:   0.05,
		MaxTrain:   2000,
		MaxHist:    2000,
		MinTuneN:   30,
	}
}

// Gateway is an online policy that learns from approve/deny feedback.
type Gateway struct {
	f     featurizer.Featurizer
	model *gp.LaplaceGPC
	cfg   Config

	trainPts  []featurizer.Point
	trainY    []int
	pHatHist  []float64
	labelHist []int

	sinceRefit int
	tuned      bool
	tauLow     float64
	tauHigh    float64
}

// New creates a gateway over the given featurizer and (already configured)
// model. The model's kernel and GP hyperparameters are taken as-is.
func New(f featurizer.Featurizer, model *gp.LaplaceGPC, cfg Config) *Gateway {
	return &Gateway{
		f:       f,
		model:   model,
		cfg:     cfg,
		tauLow:  cfg.TauLow,
		tauHigh: cfg.TauHigh,
	}
}

// Decide returns the policy decision and p_hat for a point without recording a
// label. On cold start (model not yet fitted) or an unknown point it fails safe
// to ASK with p_hat = 0.5.
func (g *Gateway) Decide(p featurizer.Point) (Decision, float64) {
	if !g.model.Fitted() {
		return Ask, 0.5
	}
	ph, ok := g.predict(p)
	if !ok {
		return Ask, 0.5
	}
	return decision(ph, g.tauLow, g.tauHigh), ph
}

// Observe records a human approve/deny for a point, appends it to the training
// set and tuning history, and refits the model every RefitEvery new labels
// (and as soon as two labels exist). Returns an error only if a refit fails.
func (g *Gateway) Observe(p featurizer.Point, approved bool) error {
	ph := 0.5
	if g.model.Fitted() {
		if v, ok := g.predict(p); ok {
			ph = v
		}
	}
	label := 0
	if approved {
		label = 1
	}

	g.trainPts = append(g.trainPts, p)
	g.trainY = append(g.trainY, label)
	g.pHatHist = append(g.pHatHist, ph)
	g.labelHist = append(g.labelHist, label)
	g.applyCaps()
	g.sinceRefit++

	if len(g.trainY) >= 2 && (!g.model.Fitted() || g.sinceRefit >= g.cfg.RefitEvery) {
		if err := g.refit(); err != nil {
			return err
		}
		g.sinceRefit = 0
	}
	return nil
}

// Tune recomputes the thresholds from the collected (p_hat, label) history. If
// fewer than MinTuneN labels exist it returns the default band.
func (g *Gateway) Tune() (low, high float64) {
	if len(g.labelHist) < g.cfg.MinTuneN {
		g.tauLow, g.tauHigh = g.cfg.TauLow, g.cfg.TauHigh
		return g.tauLow, g.tauHigh
	}
	g.tauLow, g.tauHigh = TuneThresholds(g.pHatHist, g.labelHist, g.cfg.SafetyEps, g.cfg.BlockEps)
	g.tuned = true
	return g.tauLow, g.tauHigh
}

// Thresholds returns the current (tauLow, tauHigh).
func (g *Gateway) Thresholds() (low, high float64) { return g.tauLow, g.tauHigh }

// NumLabels returns the number of accumulated training labels.
func (g *Gateway) NumLabels() int { return len(g.trainY) }

// Fitted reports whether the underlying model has been fitted.
func (g *Gateway) Fitted() bool { return g.model.Fitted() }

// Tuned reports whether the thresholds have been tuned.
func (g *Gateway) Tuned() bool { return g.tuned }

// TrainingData returns copies of the accumulated training points and labels.
func (g *Gateway) TrainingData() ([]featurizer.Point, []int) {
	return append([]featurizer.Point(nil), g.trainPts...), append([]int(nil), g.trainY...)
}

// History returns copies of the (p_hat, label) tuning history.
func (g *Gateway) History() ([]float64, []int) {
	return append([]float64(nil), g.pHatHist...), append([]int(nil), g.labelHist...)
}

// Restore reinstates persisted state and refits the model from the training
// points (the Cholesky factorization is reconstructed rather than serialized).
func (g *Gateway) Restore(pts []featurizer.Point, y []int, pHat []float64, labels []int, low, high float64, tuned bool) error {
	g.trainPts = append([]featurizer.Point(nil), pts...)
	g.trainY = append([]int(nil), y...)
	g.pHatHist = append([]float64(nil), pHat...)
	g.labelHist = append([]int(nil), labels...)
	g.tauLow, g.tauHigh = low, high
	g.tuned = tuned
	g.sinceRefit = 0
	if len(g.trainY) >= 2 {
		return g.refit()
	}
	return nil
}

// --- internals -------------------------------------------------------------

func (g *Gateway) predict(p featurizer.Point) (float64, bool) {
	fv, err := g.f.Featurize(p)
	if err != nil {
		return 0, false
	}
	probs, err := g.model.PredictProb(featurizer.Pack([]featurizer.FeatureVec{fv}))
	if err != nil || len(probs) == 0 {
		return 0, false
	}
	return probs[0], true
}

func (g *Gateway) refit() error {
	fvs := make([]featurizer.FeatureVec, 0, len(g.trainPts))
	labels := make([]int, 0, len(g.trainY))
	for i, pt := range g.trainPts {
		fv, err := g.f.Featurize(pt)
		if err != nil {
			// Skip points that no longer featurize (e.g. taxonomy changed);
			// they simply do not contribute evidence.
			continue
		}
		fvs = append(fvs, fv)
		labels = append(labels, g.trainY[i])
	}
	if len(labels) < 2 {
		return nil
	}
	return g.model.Fit(featurizer.Pack(fvs), labels)
}

func (g *Gateway) applyCaps() {
	if g.cfg.MaxTrain > 0 && len(g.trainPts) > g.cfg.MaxTrain {
		drop := len(g.trainPts) - g.cfg.MaxTrain
		g.trainPts = append([]featurizer.Point(nil), g.trainPts[drop:]...)
		g.trainY = append([]int(nil), g.trainY[drop:]...)
	}
	if g.cfg.MaxHist > 0 && len(g.pHatHist) > g.cfg.MaxHist {
		drop := len(g.pHatHist) - g.cfg.MaxHist
		g.pHatHist = append([]float64(nil), g.pHatHist[drop:]...)
		g.labelHist = append([]int(nil), g.labelHist[drop:]...)
	}
}

func decision(p, low, high float64) Decision {
	if p > high {
		return Allow
	}
	if p < low {
		return Block
	}
	return Ask
}
