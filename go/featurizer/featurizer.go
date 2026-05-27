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
// bundled trustcalib featurizer interprets the fields as a named tool acting on
// a named target resource within a task category; a custom Featurizer may
// interpret them however it likes (or ignore them and define its own input).
type Point struct {
	Tool    string  `json:"tool"`
	Target  string  `json:"target"`
	Task    string  `json:"task"`
	ArgRisk int     `json:"arg_risk"`
	T       float64 `json:"t"`
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
