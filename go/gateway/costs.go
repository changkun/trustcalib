package gateway

import (
	"errors"
	"math"
)

// Costs are the Chow reject-option costs of the three-tier rule (manuscript
// Section 5, Proposition 2, machine-checked in lean/TrustCalib/Chow.lean).
// With approval probability q, the expected losses are
//
//	ALLOW: (1 - q) * FalseAllow    BLOCK: q * FalseBlock    ASK: Ask
//
// and the Bayes-optimal one-step rule is the three-tier rule with
//
//	tau_low = Ask / FalseBlock,    tau_high = 1 - Ask / FalseAllow.
//
// The ASK band is non-empty iff Ask * (FalseAllow + FalseBlock) <
// FalseAllow * FalseBlock. The version-1 default band (0.35, 0.65) is exactly
// this rule with symmetric costs FalseAllow = FalseBlock = Ask / 0.35 (see
// SymmetricCosts). Thresholds derived from costs are specified, not tuned: no
// labels are needed, and none of the selection bias of the collected history
// (see Gateway.Tune) can leak into them.
type Costs struct {
	FalseAllow float64 // c_FA: cost of auto-allowing an action the human would deny
	FalseBlock float64 // c_FB: cost of auto-blocking an action the human would approve
	Ask        float64 // c_ask: cost of one escalation to the human
}

// SymmetricCosts returns the symmetric costs (FalseAllow = FalseBlock =
// 1/tauLow, Ask = 1) whose band is (tauLow, 1 - tauLow).
func SymmetricCosts(tauLow float64) Costs {
	return Costs{FalseAllow: 1 / tauLow, FalseBlock: 1 / tauLow, Ask: 1}
}

// Validate reports whether every cost is finite and strictly positive.
func (c Costs) Validate() error {
	for _, v := range []float64{c.FalseAllow, c.FalseBlock, c.Ask} {
		if !(v > 0) || math.IsInf(v, 0) {
			return errors.New("gateway: costs must be finite and strictly positive")
		}
	}
	return nil
}

// TauLow returns the closed-form block threshold Ask / FalseBlock.
func (c Costs) TauLow() float64 { return c.Ask / c.FalseBlock }

// TauHigh returns the closed-form allow threshold 1 - Ask / FalseAllow.
func (c Costs) TauHigh() float64 { return 1 - c.Ask/c.FalseAllow }

// BandNonEmpty reports whether the ASK band is non-empty, i.e.
// Ask * (FalseAllow + FalseBlock) < FalseAllow * FalseBlock. When it is
// empty, escalating is never strictly better than the cheaper auto-decision.
func (c Costs) BandNonEmpty() bool {
	return c.Ask*(c.FalseAllow+c.FalseBlock) < c.FalseAllow*c.FalseBlock
}

// Band returns the thresholds the gateway applies. For a non-empty band it is
// (TauLow, TauHigh). For an empty band the closed forms cross (TauLow >=
// TauHigh) and applying them literally would auto-allow actions a binary
// Bayes rule blocks, so the band collapses to the allow/block indifference
// point FalseAllow / (FalseAllow + FalseBlock): the gateway then never asks
// (except at exactly that point, where every action costs the same or more).
func (c Costs) Band() (low, high float64) {
	if c.BandNonEmpty() {
		return c.TauLow(), c.TauHigh()
	}
	m := c.FalseAllow / (c.FalseAllow + c.FalseBlock)
	return m, m
}

// Loss returns the expected loss of decision d when the approval probability
// is q.
func (c Costs) Loss(d Decision, q float64) float64 {
	switch d {
	case Allow:
		return (1 - q) * c.FalseAllow
	case Block:
		return q * c.FalseBlock
	default:
		return c.Ask
	}
}
