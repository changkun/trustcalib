// Package trustcalib is the bundled default featurizer that mirrors the
// manuscript's synthetic action/context taxonomy (experiment/data.py). It is
// the only place the concrete tool/target/task tables live; a harness that
// wants its own taxonomy can ignore this package and implement
// featurizer.Featurizer directly.
package trustcalib

import "sort"

// Reversibility classes.
const (
	Reversible   = 0
	Hard         = 1
	Irreversible = 2
)

// Tool carries the decision-time-knowable risk attributes of an agent tool.
// ArgRiskProb is retained for fidelity with the manuscript taxonomy but is not
// used by featurization (only the realized arg-risk flag enters the features).
type Tool struct {
	Name            string
	Category        string
	Reversibility   int
	BaseSensitivity float64
	Blast           int
	ArgRiskProb     float64
}

// Tools is the 18-tool taxonomy from data.py.
var Tools = []Tool{
	// read-only / low risk
	{"read_file", "read", Reversible, 0.10, 0, 0.00},
	{"list_dir", "read", Reversible, 0.08, 1, 0.00},
	{"grep_search", "search", Reversible, 0.10, 2, 0.00},
	{"git_status", "vcs", Reversible, 0.05, 2, 0.00},
	{"run_tests", "exec", Reversible, 0.20, 2, 0.02},
	{"web_fetch", "network", Reversible, 0.30, 3, 0.05},
	// moderate / hard-to-reverse
	{"write_file", "write", Hard, 0.30, 0, 0.05},
	{"apply_patch", "write", Hard, 0.40, 1, 0.08},
	{"format_code", "write", Hard, 0.25, 2, 0.03},
	{"install_package", "exec", Hard, 0.50, 2, 0.15},
	{"git_commit", "vcs", Hard, 0.30, 2, 0.04},
	// high / irreversible
	{"delete_file", "write", Irreversible, 0.60, 0, 0.30},
	{"git_push", "vcs", Irreversible, 0.60, 3, 0.10},
	{"git_force_push", "vcs", Irreversible, 0.85, 3, 0.60},
	{"execute_sql", "db", Irreversible, 0.70, 2, 0.35},
	{"shell_exec", "exec", Irreversible, 0.70, 2, 0.30},
	{"send_email", "network", Irreversible, 0.50, 3, 0.10},
	{"deploy", "deploy", Irreversible, 0.90, 3, 0.20},
}

// Targets maps target-resource names to their sensitivity tier in [0,1].
var Targets = map[string]float64{
	"sandbox_tmp":     0.00,
	"workspace_tests": 0.20,
	"workspace_src":   0.40,
	"build_config":    0.55,
	"ci_config":       0.65,
	"secrets_env":     0.90,
	"production_db":   0.95,
	"prod_infra":      1.00,
}

// Tasks is the execution-context task taxonomy (order matters: it fixes the
// one-hot index).
var Tasks = []string{
	"feature_dev",
	"bugfix",
	"refactor",
	"ops_maintenance",
	"data_migration",
	"security_hardening",
	"exploration",
}

// Categories is the sorted set of unique tool categories (fixes the one-hot
// index), matching sorted({t.category for t in TOOLS}) in data.py.
var Categories = sortedCategories()

func sortedCategories() []string {
	seen := map[string]struct{}{}
	for _, t := range Tools {
		seen[t.Category] = struct{}{}
	}
	out := make([]string, 0, len(seen))
	for c := range seen {
		out = append(out, c)
	}
	sort.Strings(out)
	return out
}
