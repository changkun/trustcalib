// Package bashmap is an illustrative example of mapping a raw shell command to
// a trustcalib featurizer.Point by static, worst-case analysis.
//
// It exists to answer a common question: a coarse tool like Bash can be
// harmless ("ls") or catastrophic ("rm -rf /"), so the tool identity alone is
// too vague to gate on. The pattern shown here drives the target resource and
// the destructive-argument flag from the command *content*, and reduces a
// command chain (commands joined by &&, ||, ;, or pipes) to its most dangerous
// component: the most severe tool match, the most sensitive target touched, and
// the logical OR of the destructive-argument flag.
//
// Everything here is computed statically, before execution, which is the whole
// point of gating. This is a heuristic demonstration, not a hardened parser: a
// production harness should analyze the properly parsed command and tune these
// rules to its own environment, or implement its own featurizer.Featurizer over
// a richer descriptor.
package bashmap

import (
	"regexp"

	"github.com/changkun/trustcalib/featurizer"
)

type rule struct {
	re   *regexp.Regexp
	name string
}

// toolRules map a command pattern to a trustcalib tool name, ordered most
// dangerous first; the first rule that matches anywhere in the command wins, so
// a chain is judged by its worst element.
var toolRules = []rule{
	{regexp.MustCompile(`(?i)\b(terraform|kubectl|helm)\s+(apply|destroy|delete)\b`), "deploy"},
	{regexp.MustCompile(`(?i)\bgit\s+push\b.*(--force|-f)\b`), "git_force_push"},
	{regexp.MustCompile(`(?i)\bgit\s+push\b`), "git_push"},
	{regexp.MustCompile(`(?i)\b(psql|mysql|sqlite3|mongosh?)\b|\b(drop\s+table|delete\s+from|truncate)\b`), "execute_sql"},
	{regexp.MustCompile(`(?i)\brm\s+-[a-z]*r`), "delete_file"},
	{regexp.MustCompile(`(?i)\b(apt|apt-get|yum|brew|pip3?|npm|go)\s+(install|get|add)\b`), "install_package"},
	{regexp.MustCompile(`(?i)\bgit\s+commit\b`), "git_commit"},
	{regexp.MustCompile(`(?i)\b(pytest|go\s+test|npm\s+test|jest)\b`), "run_tests"},
	{regexp.MustCompile(`(?i)\b(ls|cat|head|tail|grep|find|git\s+status)\b`), "read_file"},
}

const defaultTool = "shell_exec"

// targetRules map a command pattern to a trustcalib target name, ordered most
// sensitive first.
var targetRules = []rule{
	{regexp.MustCompile(`(?i)\b(terraform|kubectl|helm)\b|/etc/|\bprod(uction)?[-_./ ]`), "prod_infra"},
	{regexp.MustCompile(`(?i)\b(psql|mysql|mongosh?)\b|\bdatabase\b`), "production_db"},
	{regexp.MustCompile(`(?i)\.env\b|secret|credential|id_rsa|\.aws|\btoken\b|password`), "secrets_env"},
	{regexp.MustCompile(`(?i)\.github/workflows|\.gitlab-ci|jenkinsfile|\.circleci`), "ci_config"},
	{regexp.MustCompile(`(?i)dockerfile|makefile|package\.json|go\.mod|pyproject`), "build_config"},
	{regexp.MustCompile(`(?i)\bsrc/|\.go\b|\.py\b|\.ts\b|\.js\b`), "workspace_src"},
	{regexp.MustCompile(`(?i)test|spec`), "workspace_tests"},
}

const defaultTarget = "sandbox_tmp"

// destructive flags command-line patterns that commonly cause irreversible
// damage (the manuscript's "destructive argument pattern present").
var destructive = regexp.MustCompile(
	`(?i)\brm\s+-[a-z]*[rf]|--force\b|--hard\b|\bdrop\s+table\b|\btruncate\b|\bdd\s|\bmkfs|>\s*/dev/|chmod\s+-R\s+777|:\(\)\s*\{|\|\s*(sh|bash)\b|\bgit\s+push\b.*-f\b`,
)

// PointFromBash builds a trustcalib featurizer.Point from a shell command and
// the agent's current task (session intent, supplied by the harness). The time
// index t orders the decision stream for the kernel's recency weighting.
func PointFromBash(command, task string, t float64) featurizer.Point {
	tool := defaultTool
	for _, r := range toolRules {
		if r.re.MatchString(command) {
			tool = r.name
			break
		}
	}
	target := defaultTarget
	for _, r := range targetRules {
		if r.re.MatchString(command) {
			target = r.name
			break
		}
	}
	argRisk := 0
	if destructive.MatchString(command) {
		argRisk = 1
	}
	return featurizer.Point{Tool: tool, Target: target, Task: task, ArgRisk: argRisk, T: t}
}
