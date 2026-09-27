package gateway

import (
	"errors"
	"fmt"
	"math"

	"github.com/changkun/trustcalib/featurizer"
)

// LabelSource records why a human label was collected.
type LabelSource int

const (
	// FromAsk is a label from an escalation (propensity 1).
	FromAsk LabelSource = iota
	// FromAuditAllow is a label from a random audit of an auto-ALLOW
	// (propensity AuditRate).
	FromAuditAllow
	// FromAuditBlock is a label from a random audit of an auto-BLOCK
	// (propensity AuditRate).
	FromAuditBlock
)

func (s LabelSource) String() string {
	switch s {
	case FromAuditAllow:
		return "audit_allow"
	case FromAuditBlock:
		return "audit_block"
	default:
		return "ask"
	}
}

// MarshalText encodes the source as "ask", "audit_allow" or "audit_block".
func (s LabelSource) MarshalText() ([]byte, error) { return []byte(s.String()), nil }

// UnmarshalText decodes the output of MarshalText.
func (s *LabelSource) UnmarshalText(b []byte) error {
	switch string(b) {
	case "ask", "":
		*s = FromAsk
	case "audit_allow":
		*s = FromAuditAllow
	case "audit_block":
		*s = FromAuditBlock
	default:
		return fmt.Errorf("gateway: unknown label source %q", b)
	}
	return nil
}

// LabelInfo is the provenance of one training label: why it was collected and
// the probability with which it was going to be collected (its propensity).
// Logging propensities is what lets escalations and audits both count in a
// Horvitz-Thompson estimate (manuscript Proposition 8).
type LabelInfo struct {
	Source     LabelSource `json:"source"`
	Propensity float64     `json:"propensity"`
}

// askLabel is the provenance of an escalated label.
var askLabel = LabelInfo{Source: FromAsk, Propensity: 1}

// AuditStats are the cumulative counters behind the audit estimates. They are
// never truncated by the MaxTrain/MaxHist windows, so the numerators and the
// denominators cover the same period.
type AuditStats struct {
	NAllow int `json:"n_allow"` // auto-ALLOW decisions counted by DecideAudit (audited or not)
	NBlock int `json:"n_block"` // auto-BLOCK decisions counted by DecideAudit (audited or not)

	AuditsAllow   int     `json:"audits_allow"`    // audited auto-ALLOWs with a human label
	DeniedAllow   int     `json:"denied_allow"`    // ... of which the human denied
	HTDeniedAllow float64 `json:"ht_denied_allow"` // sum of 1/propensity over the denied ones

	AuditsBlock     int     `json:"audits_block"`      // audited auto-BLOCKs with a human label
	ApprovedBlock   int     `json:"approved_block"`    // ... of which the human approved
	HTApprovedBlock float64 `json:"ht_approved_block"` // sum of 1/propensity over the approved ones

	// CleanStreak is the number of consecutive most recent audits of
	// auto-ALLOWs the human approved; a denial resets it to 0.
	CleanStreak int `json:"clean_streak"`
}

// DecideAudit is Decide for an action that is actually about to run: it
// returns the policy decision and p_hat, counts auto-ALLOW and auto-BLOCK
// decisions (N_allow, N_block), and samples a random audit. When the decision
// is Allow or Block and u < AuditRate, audit is true and the caller should
// show the action to the human anyway, then report the answer with
// ObserveAudit(p, approved, decision). u must be a uniform draw from [0, 1)
// supplied by the caller, which keeps the gateway deterministic and testable.
//
// Call DecideAudit exactly once per real action (the counts are the
// denominators of the audit estimates); use Decide for side-effect-free
// queries. The returned decision is the auto-decision that was audited, not
// Ask.
func (g *Gateway) DecideAudit(p featurizer.Point, u float64) (d Decision, pHat float64, audit bool) {
	d, pHat = g.Decide(p)
	switch d {
	case Allow:
		g.audit.NAllow++
	case Block:
		g.audit.NBlock++
	default:
		return d, pHat, false
	}
	return d, pHat, g.cfg.AuditRate > 0 && u < g.cfg.AuditRate
}

// ObserveAudit records the human's answer to a random audit of an
// auto-decision (audited is the Allow or Block that DecideAudit returned). The
// label trains the model like an escalated one and is logged with propensity
// AuditRate; it updates the audit counters but, unlike Observe, is not added
// to the (p_hat, label) tuning history, whose points all come from the ASK
// band. It is an error to report an audit while AuditRate <= 0 (the label
// would have propensity 0) or for an audited decision other than Allow or
// Block.
func (g *Gateway) ObserveAudit(p featurizer.Point, approved bool, audited Decision) error {
	if !(g.cfg.AuditRate > 0) {
		return errors.New("gateway: ObserveAudit requires AuditRate > 0")
	}
	return g.ObserveAuditWithPropensity(p, approved, audited, g.cfg.AuditRate)
}

// ObserveAuditWithPropensity is ObserveAudit with the propensity recorded when
// the audit was sampled. Use it when the answer arrives in a later process
// (as in the CLI hook) and AuditRate may have changed in between: the
// Horvitz-Thompson weight must be the probability with which this action was
// actually audited (manuscript Proposition 8).
func (g *Gateway) ObserveAuditWithPropensity(p featurizer.Point, approved bool, audited Decision, propensity float64) error {
	rate := propensity
	if !(rate > 0) {
		return errors.New("gateway: an audit label needs a propensity > 0")
	}
	if rate > 1 {
		rate = 1
	}
	var info LabelInfo
	switch audited {
	case Allow:
		info = LabelInfo{Source: FromAuditAllow, Propensity: rate}
		g.audit.AuditsAllow++
		if approved {
			g.audit.CleanStreak++
		} else {
			g.audit.DeniedAllow++
			g.audit.HTDeniedAllow += 1 / rate
			g.audit.CleanStreak = 0
		}
	case Block:
		info = LabelInfo{Source: FromAuditBlock, Propensity: rate}
		g.audit.AuditsBlock++
		if approved {
			g.audit.ApprovedBlock++
			g.audit.HTApprovedBlock += 1 / rate
		}
	default:
		return fmt.Errorf("gateway: ObserveAudit of a %v decision (want allow or block)", audited)
	}
	return g.observe(p, approved, info, false)
}

// FalseAllowEstimate returns the Horvitz-Thompson estimate of the realized
// false-allow rate among auto-ALLOWs (manuscript Proposition 5(a)),
//
//	FA = (sum over audited auto-ALLOWs the human denied of 1/propensity) / N_allow,
//
// which equals (denied / AuditRate) / N_allow at a constant audit rate and is
// unbiased when every counted auto-ALLOW was audited with that rate. It also
// returns the number of audits of auto-ALLOWs and how many the human denied.
// The estimate is NaN when it is undefined: no auto-ALLOW has been counted,
// or auditing is disabled and no audit was ever recorded.
func (g *Gateway) FalseAllowEstimate() (estimate float64, audits int, denied int) {
	a := g.audit
	if a.NAllow == 0 || (!(g.cfg.AuditRate > 0) && a.AuditsAllow == 0) {
		return math.NaN(), a.AuditsAllow, a.DeniedAllow
	}
	return a.HTDeniedAllow / float64(a.NAllow), a.AuditsAllow, a.DeniedAllow
}

// FalseBlockEstimate is the false-block counterpart of FalseAllowEstimate:
// the Horvitz-Thompson estimate of the fraction of auto-BLOCKs the human
// would have approved, with the number of audits of auto-BLOCKs and how many
// the human approved.
func (g *Gateway) FalseBlockEstimate() (estimate float64, audits int, approved int) {
	a := g.audit
	if a.NBlock == 0 || (!(g.cfg.AuditRate > 0) && a.AuditsBlock == 0) {
		return math.NaN(), a.AuditsBlock, a.ApprovedBlock
	}
	return a.HTApprovedBlock / float64(a.NBlock), a.AuditsBlock, a.ApprovedBlock
}

// CertificationSize returns the number of consecutive clean audits,
// ceil(ln(1/delta)/alpha), that certify a false-allow rate below alpha with
// confidence 1 - delta (manuscript Proposition 5(b),
// lean/TrustCalib/Audit.lean certify_sample_size): if the true rate were at
// least alpha, n independent audits would all be clean with probability at
// most (1-alpha)^n <= exp(-alpha n) <= delta. For example alpha = 0.01,
// delta = 0.05 gives 300. It returns 0 unless 0 < alpha <= 1 and
// 0 < delta < 1.
func CertificationSize(alpha, delta float64) int {
	if !validCertParams(alpha, delta) {
		return 0
	}
	return int(math.Ceil(math.Log(1/delta) / alpha))
}

// CertifiedFalseAllowBelow reports whether the most recent run of clean
// audits of auto-ALLOWs is long enough, n >= ln(1/delta)/alpha, to certify
// that the false-allow rate is below alpha with confidence 1 - delta
// (manuscript Proposition 5(b)). The claim covers the period of those n
// audits for this supervisor; a single denied audit restarts the count. It is
// false for invalid parameters (see CertificationSize).
func (g *Gateway) CertifiedFalseAllowBelow(alpha, delta float64) bool {
	if !validCertParams(alpha, delta) {
		return false
	}
	return float64(g.audit.CleanStreak) >= math.Log(1/delta)/alpha
}

func validCertParams(alpha, delta float64) bool {
	return alpha > 0 && alpha <= 1 && delta > 0 && delta < 1
}

// AuditStats returns the cumulative audit counters.
func (g *Gateway) AuditStats() AuditStats { return g.audit }

// Labels returns a copy of the provenance of each training label, parallel to
// the points and labels returned by TrainingData.
func (g *Gateway) Labels() []LabelInfo {
	return append([]LabelInfo(nil), g.trainInfo...)
}

// RestoreAudit reinstates persisted audit counters and label provenance. Call
// it after Restore. If labels is not parallel to the restored training points
// (for example a state file written before audits existed), every label is
// taken to be an escalation with propensity 1.
func (g *Gateway) RestoreAudit(stats AuditStats, labels []LabelInfo) {
	g.audit = stats
	if len(labels) == len(g.trainPts) {
		g.trainInfo = append([]LabelInfo(nil), labels...)
		return
	}
	g.trainInfo = askLabels(len(g.trainPts))
}

func askLabels(n int) []LabelInfo {
	out := make([]LabelInfo, n)
	for i := range out {
		out[i] = askLabel
	}
	return out
}
