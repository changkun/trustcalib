package trustcalib

import (
	"fmt"

	"github.com/changkun/trustcalib/featurizer"
)

// Featurizer builds the manuscript's feature blocks from a named tool/target/
// task point (data.py DecisionPoint.featurize):
//
//	phi_tool = [reversibility/2, base_sensitivity, blast/3] ++ category_onehot(8)   -> 11 dims
//	phi_ctx  = [target_sensitivity, arg_risk]               ++ task_onehot(7)       ->  9 dims
type Featurizer struct {
	toolByName map[string]Tool
	catIndex   map[string]int
	taskIndex  map[string]int
}

// New returns a featurizer over the bundled taxonomy.
func New() *Featurizer {
	f := &Featurizer{
		toolByName: make(map[string]Tool, len(Tools)),
		catIndex:   make(map[string]int, len(Categories)),
		taskIndex:  make(map[string]int, len(Tasks)),
	}
	for _, t := range Tools {
		f.toolByName[t.Name] = t
	}
	for i, c := range Categories {
		f.catIndex[c] = i
	}
	for i, t := range Tasks {
		f.taskIndex[t] = i
	}
	return f
}

// DimTool returns the tool-block dimension (3 + number of categories).
func (f *Featurizer) DimTool() int { return 3 + len(Categories) }

// DimCtx returns the context-block dimension (2 + number of tasks).
func (f *Featurizer) DimCtx() int { return 2 + len(Tasks) }

// Featurize maps a Point to its kernel feature blocks. An unknown tool, target
// or task is an error (the gateway maps that to a fail-safe ASK).
func (f *Featurizer) Featurize(p featurizer.Point) (featurizer.FeatureVec, error) {
	tool, ok := f.toolByName[p.Tool]
	if !ok {
		return featurizer.FeatureVec{}, fmt.Errorf("trustcalib: unknown tool %q", p.Tool)
	}
	targetSens, ok := Targets[p.Target]
	if !ok {
		return featurizer.FeatureVec{}, fmt.Errorf("trustcalib: unknown target %q", p.Target)
	}
	taskIdx, ok := f.taskIndex[p.Task]
	if !ok {
		return featurizer.FeatureVec{}, fmt.Errorf("trustcalib: unknown task %q", p.Task)
	}

	phiTool := make([]float64, f.DimTool())
	phiTool[0] = float64(tool.Reversibility) / 2.0
	phiTool[1] = tool.BaseSensitivity
	phiTool[2] = float64(tool.Blast) / 3.0
	phiTool[3+f.catIndex[tool.Category]] = 1.0

	phiCtx := make([]float64, f.DimCtx())
	phiCtx[0] = targetSens
	phiCtx[1] = float64(p.ArgRisk)
	phiCtx[2+taskIdx] = 1.0

	return featurizer.FeatureVec{PhiTool: phiTool, PhiCtx: phiCtx, T: p.T}, nil
}
