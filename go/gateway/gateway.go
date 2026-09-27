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
// enter the training set, and the model refits online as labels accumulate.
//
// Thresholds. With Config.Costs set, the band is derived in closed form from
// the costs of a false allow, a false block and one escalation (Chow's
// reject option, see Costs); this is the recommended configuration. Without
// costs the band is the configured (TauLow, TauHigh), which Tune can replace
// by a grid search over the collected history; that search is kept for
// backward compatibility but degenerates to the default band (see Tune).
//
// Audits. Auto-decided actions produce no label, so their errors are
// invisible (selective labels). With Config.AuditRate = eps > 0, DecideAudit
// also shows each auto-decided action to the human with probability eps;
// ObserveAudit records the answer with its propensity, which gives an unbiased
// Horvitz-Thompson estimate of the false-allow rate (FalseAllowEstimate) and a
// certification test (CertifiedFalseAllowBelow), manuscript Proposition 5.
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
	TauLow     float64 // initial/default block threshold (ignored when Costs is set)
	TauHigh    float64 // initial/default allow threshold (ignored when Costs is set)
	RefitEvery int     // refit after this many new labels
	SafetyEps  float64 // false-allow cap for tuning (tightened to /2)
	BlockEps   float64 // false-block cap for tuning
	MaxTrain   int     // sliding-window cap on training points (0 = unbounded)
	MaxHist    int     // sliding-window cap on tuning history (0 = unbounded)
	MinTuneN   int     // minimum labels before Tune leaves the default band

	// Costs, when set and valid, fixes the thresholds to the cost-derived
	// Chow band (Costs.Band); Tune then returns it unchanged.
	Costs *Costs
	// AuditRate is the probability with which DecideAudit samples an
	// auto-decided action for a human audit (0 = no audits).
	AuditRate float64
}

// Thresholds returns the configured band: the cost-derived band when Costs is
// set and valid, otherwise (TauLow, TauHigh).
func (c Config) Thresholds() (low, high float64) {
	if c.costsValid() {
		return c.Costs.Band()
	}
	return c.TauLow, c.TauHigh
}

func (c Config) costsValid() bool { return c.Costs != nil && c.Costs.Validate() == nil }

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
	trainInfo []LabelInfo // provenance of each training label (parallel)
	pHatHist  []float64
	labelHist []int
	audit     AuditStats

	sinceRefit int
	tuned      bool
	tauLow     float64
	tauHigh    float64
}

// New creates a gateway over the given featurizer and (already configured)
// model. The model's kernel and GP hyperparameters are taken as-is.
func New(f featurizer.Featurizer, model *gp.LaplaceGPC, cfg Config) *Gateway {
	low, high := cfg.Thresholds()
	return &Gateway{
		f:       f,
		model:   model,
		cfg:     cfg,
		tauLow:  low,
		tauHigh: high,
	}
}

// Decide returns the policy decision and p_hat for a point without recording
// anything (use DecideAudit for an action that is about to run, so that
// auto-decisions are counted and audits sampled). On cold start (model not yet
// fitted) or an unknown point it fails safe to ASK with p_hat = 0.5.
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

// Observe records a human approve/deny for an escalated (ASK) point, with
// propensity 1: it appends the point to the training set and the tuning
// history, and refits the model every RefitEvery new labels (and as soon as
// two labels exist). Report the answer to a random audit with ObserveAudit
// instead. Returns an error only if a refit fails.
func (g *Gateway) Observe(p featurizer.Point, approved bool) error {
	return g.observe(p, approved, askLabel, true)
}

// observe appends a label with the given provenance to the training set (and,
// if hist, to the tuning history) and refits on schedule.
func (g *Gateway) observe(p featurizer.Point, approved bool, info LabelInfo, hist bool) error {
	label := 0
	if approved {
		label = 1
	}
	if hist {
		ph := 0.5
		if g.model.Fitted() {
			if v, ok := g.predict(p); ok {
				ph = v
			}
		}
		g.pHatHist = append(g.pHatHist, ph)
		g.labelHist = append(g.labelHist, label)
	}

	g.trainPts = append(g.trainPts, p)
	g.trainY = append(g.trainY, label)
	g.trainInfo = append(g.trainInfo, info)
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

// Tune recomputes the thresholds. With Config.Costs set (and valid) it
// returns the cost-derived band unchanged and never grid-searches: the
// thresholds are specified, not learned. Otherwise it runs TuneThresholds on
// the collected (p_hat, label) history, and below MinTuneN labels returns the
// default band.
//
// The grid search is kept for backward compatibility only. The history is
// recorded at ASK time (Observe; audit labels are not added to it), so
// essentially every p_hat in it lies inside the current band, where labels are
// close to coin flips. Any pair that auto-decides enough of the history to meet
// the coverage cap then violates the false-allow or false-block cap, no pair
// is feasible, and the search degenerates to the default band (0.35, 0.65).
// Tuning would need labels for auto-decided actions, which a deployment does
// not have. Set Config.Costs instead.
func (g *Gateway) Tune() (low, high float64) {
	if g.cfg.costsValid() {
		g.tauLow, g.tauHigh = g.cfg.Thresholds()
		return g.tauLow, g.tauHigh
	}
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
// Every restored label is taken to be an escalation with propensity 1; call
// RestoreAudit afterwards to reinstate audit provenance and counters. With
// Config.Costs set, the persisted thresholds are ignored in favour of the
// cost-derived band (costs are specified, not learned).
func (g *Gateway) Restore(pts []featurizer.Point, y []int, pHat []float64, labels []int, low, high float64, tuned bool) error {
	g.trainPts = append([]featurizer.Point(nil), pts...)
	g.trainY = append([]int(nil), y...)
	g.trainInfo = askLabels(len(pts))
	g.pHatHist = append([]float64(nil), pHat...)
	g.labelHist = append([]int(nil), labels...)
	g.tauLow, g.tauHigh = low, high
	if g.cfg.costsValid() {
		g.tauLow, g.tauHigh = g.cfg.Thresholds()
	}
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
		g.trainInfo = append([]LabelInfo(nil), g.trainInfo[drop:]...)
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
