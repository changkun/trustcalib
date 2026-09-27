// Package judge is a featurizer for shell-centric harnesses with an LLM judge
// in front of a general-purpose shell tool, such as an auto-mode permission
// classifier (manuscript Section 12, experiment/judge.py).
//
// When an agent routes everything through one shell tool there is no fixed
// tool taxonomy; a judge decides allow/block from its own view of the action.
// A judge's error rate for a given supervisor is not identifiable from its
// verdicts alone and changes with the supervisor (Propositions 6 and 7), so
// the gateway treats the judge's output as a feature rather than a decision.
// It sees no taxonomy either, only
//
//	phi_tool = [judge feature] ++ category_onehot(9)   -> 10 dims
//	phi_ctx  = [0]                                     ->  1 dim
//	t        = Point.T
//
// where the judge feature is Point.JudgeScore (a risk score in [0, 1], higher
// = riskier) when present, and otherwise the verdict (1 for "block", 0 for
// "allow"), and the category is the coarse command category Point.Category.
// With the additive kernel the static component learns, per category, how
// this supervisor's approvals relate to the judge's output, and the time
// components track the supervisor's tolerance; the posterior p_hat is then
// the reliability of an individual verdict for this supervisor, the Chow rule
// escalates where the verdict is unreliable, and audits with logged
// propensities keep the judge's false-allow rate estimable.
//
// One limit is structural: the gateway cannot separate actions that look
// identical in these features, so it cannot repair a judge blind spot they do
// not expose. Argument-level features (for example bashmap's static command
// patterns) are the remedy, not more calibration.
//
// bashmap.CategoryFromBash derives the category from a shell command.
package judge

import (
	"errors"
	"fmt"
	"math"
	"strings"

	"github.com/changkun/trustcalib/featurizer"
)

// Categories is the coarse command-category vocabulary. The order fixes the
// one-hot index. The first eight are the bundled taxonomy's tool categories;
// "other" is the bucket for everything else.
var Categories = []string{"read", "search", "vcs", "exec", "write", "db", "network", "deploy", "other"}

// Verdicts accepted in Point.JudgeVerdict.
const (
	VerdictAllow = "allow"
	VerdictBlock = "block"
)

// Featurizer maps a judge-annotated Point to kernel features. The zero value
// is not usable; construct with New.
type Featurizer struct {
	catIndex map[string]int
}

var _ featurizer.Featurizer = (*Featurizer)(nil)

// New returns a judge featurizer.
func New() *Featurizer {
	f := &Featurizer{catIndex: make(map[string]int, len(Categories))}
	for i, c := range Categories {
		f.catIndex[c] = i
	}
	return f
}

// DimTool returns the tool-block dimension (1 + number of categories).
func (f *Featurizer) DimTool() int { return 1 + len(Categories) }

// DimCtx returns the context-block dimension (1, a constant 0).
func (f *Featurizer) DimCtx() int { return 1 }

// Featurize maps a Point to its kernel feature blocks. It is an error if the
// point carries neither a valid JudgeScore nor a valid JudgeVerdict (the
// gateway maps that to a fail-safe ASK). An empty or unrecognized category
// maps to "other", so an unfamiliar category never escalates by itself.
func (f *Featurizer) Featurize(p featurizer.Point) (featurizer.FeatureVec, error) {
	feat, err := Feature(p)
	if err != nil {
		return featurizer.FeatureVec{}, err
	}
	phiTool := make([]float64, f.DimTool())
	phiTool[0] = feat
	phiTool[1+f.catIndex[NormalizeCategory(p.Category)]] = 1
	return featurizer.FeatureVec{PhiTool: phiTool, PhiCtx: []float64{0}, T: p.T}, nil
}

// Feature returns the judge feature of a point: JudgeScore if present
// (it must lie in [0, 1]), else 1 for a "block" verdict and 0 for "allow"
// (case-insensitive).
func Feature(p featurizer.Point) (float64, error) {
	if p.JudgeScore != nil {
		s := *p.JudgeScore
		if math.IsNaN(s) || s < 0 || s > 1 {
			return 0, fmt.Errorf("judge: score %v outside [0, 1]", s)
		}
		return s, nil
	}
	switch strings.ToLower(strings.TrimSpace(p.JudgeVerdict)) {
	case VerdictAllow:
		return 0, nil
	case VerdictBlock:
		return 1, nil
	case "":
		return 0, errors.New("judge: point has neither judge_score nor judge_verdict")
	default:
		return 0, fmt.Errorf("judge: unknown verdict %q (want allow or block)", p.JudgeVerdict)
	}
}

// NormalizeCategory lower-cases and trims a category and maps anything
// outside Categories (including the empty string) to "other".
func NormalizeCategory(c string) string {
	c = strings.ToLower(strings.TrimSpace(c))
	for _, k := range Categories {
		if c == k {
			return c
		}
	}
	return "other"
}
