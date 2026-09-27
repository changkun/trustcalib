// Package featurizer defines the mapping from a domain-level decision point
// (a proposed tool call in some context, at some time) to the raw feature
// blocks the kernel consumes. The Featurizer interface lets an agent harness
// plug in its own action/context taxonomy; a bundled default that mirrors the
// trustcalib manuscript lives in the trustcalib subpackage.
package featurizer

import (
	"gonum.org/v1/gonum/mat"

	"github.com/changkun/trustcalib/kernel"
)

// FeatureVec is one decision point's kernel-ready features.
type FeatureVec struct {
	PhiTool []float64
	PhiCtx  []float64
	T       float64
}

// Point is the domain-level input a Featurizer maps into a FeatureVec. The
// bundled trustcalib featurizer interprets Tool, Target, Task and ArgRisk as a
// named tool acting on a named target resource within a task category. The
// judge featurizer (subpackage judge), for shell-centric harnesses with an
// LLM judge in front of a general-purpose shell, reads JudgeVerdict,
// JudgeScore and Category instead. A custom Featurizer may interpret the
// fields however it likes (or ignore them and define its own input).
//
// JudgeScore is a pointer so that "no score" is distinguishable from a score
// of 0; compare Points field by field rather than with ==.
type Point struct {
	Tool    string  `json:"tool"`
	Target  string  `json:"target"`
	Task    string  `json:"task"`
	ArgRisk int     `json:"arg_risk"`
	T       float64 `json:"t"`

	// JudgeVerdict is the judge's verdict on the action, "allow" or "block".
	JudgeVerdict string `json:"judge_verdict,omitempty"`
	// JudgeScore is the judge's risk score in [0, 1] (higher = riskier), if
	// the judge exposes one.
	JudgeScore *float64 `json:"judge_score,omitempty"`
	// Category is a coarse command category: read, search, vcs, exec, write,
	// db, network, deploy or other.
	Category string `json:"category,omitempty"`
}

// Featurizer maps a Point to a FeatureVec.
type Featurizer interface {
	Featurize(p Point) (FeatureVec, error)
	DimTool() int
	DimCtx() int
}

// Pack converts a slice of FeatureVec into a kernel.Packed. The slice must be
// non-empty and all vectors must share dimensions.
func Pack(fvs []FeatureVec) kernel.Packed {
	n := len(fvs)
	if n == 0 {
		return kernel.Packed{}
	}
	dTool := len(fvs[0].PhiTool)
	dCtx := len(fvs[0].PhiCtx)
	pt := mat.NewDense(n, dTool, nil)
	pc := mat.NewDense(n, dCtx, nil)
	t := make([]float64, n)
	for i, fv := range fvs {
		pt.SetRow(i, fv.PhiTool)
		pc.SetRow(i, fv.PhiCtx)
		t[i] = fv.T
	}
	return kernel.Packed{PhiTool: pt, PhiCtx: pc, T: t}
}
