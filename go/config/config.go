// Package config loads trustcalib hyperparameters from a YAML file. Every field
// has the manuscript default, so an absent file or an absent key falls back
// cleanly. The loaded Config builds the kernel, GP model and gateway, and
// names the featurizer the CLI should use.
package config

import (
	"errors"
	"fmt"
	"io/fs"
	"math"
	"os"

	"gopkg.in/yaml.v3"

	"github.com/changkun/trustcalib/featurizer"
	"github.com/changkun/trustcalib/gateway"
	"github.com/changkun/trustcalib/gp"
	"github.com/changkun/trustcalib/kernel"
)

// Kernel types accepted in KernelCfg.Type.
const (
	KernelProduct  = "product"  // v1 kernel.ProductKernel (default)
	KernelAdditive = "additive" // v2 kernel.AdditiveKernel
)

// Featurizer names accepted in Config.Featurizer.
const (
	FeaturizerTaxonomy = "taxonomy" // featurizer/trustcalib (default)
	FeaturizerJudge    = "judge"    // featurizer/judge
)

// Default certification parameters: certify a false-allow rate below 1% at
// 95% confidence (300 clean audits).
const (
	DefaultCertifyAlpha = 0.01
	DefaultCertifyDelta = 0.05
)

// KernelCfg selects the kernel. Type is "product" (the default, for backward
// compatibility) or "additive". The top-level Sigma2/LTool/LCtx/Lambda
// parameterize the product kernel; the additive kernel has its own block with
// its own defaults (notably lambda 90 rather than 200).
type KernelCfg struct {
	Type     string      `yaml:"type"`
	Sigma2   float64     `yaml:"sigma2"`
	LTool    float64     `yaml:"l_tool"`
	LCtx     float64     `yaml:"l_ctx"`
	Lambda   float64     `yaml:"lambda"`
	Additive AdditiveCfg `yaml:"additive"`
}

// AdditiveCfg mirrors kernel.AdditiveKernel.
type AdditiveCfg struct {
	SStatic float64 `yaml:"s_static"`
	SGlobal float64 `yaml:"s_global"`
	SInter  float64 `yaml:"s_inter"`
	LTool   float64 `yaml:"l_tool"`
	LCtx    float64 `yaml:"l_ctx"`
	Lambda  float64 `yaml:"lambda"`
}

// GPCfg holds the Laplace solver parameters.
type GPCfg struct {
	Jitter  float64 `yaml:"jitter"`
	MaxIter int     `yaml:"max_iter"`
	Tol     float64 `yaml:"tol"`
}

// CostsCfg mirrors gateway.Costs. All three costs are required when the block
// is present.
type CostsCfg struct {
	FalseAllow float64 `yaml:"false_allow"`
	FalseBlock float64 `yaml:"false_block"`
	Ask        float64 `yaml:"ask"`
}

// GatewayCfg mirrors gateway.Config, plus the certification parameters used
// by the CLI's stats.
type GatewayCfg struct {
	TauLow     float64 `yaml:"tau_low"`
	TauHigh    float64 `yaml:"tau_high"`
	RefitEvery int     `yaml:"refit_every"`
	SafetyEps  float64 `yaml:"safety_eps"`
	BlockEps   float64 `yaml:"block_eps"`
	MaxTrain   int     `yaml:"max_train"`
	MaxHist    int     `yaml:"max_hist"`
	MinTuneN   int     `yaml:"min_tune_n"`

	Costs        *CostsCfg `yaml:"costs"`         // cost-derived thresholds (overrides tau_low/tau_high)
	AuditRate    float64   `yaml:"audit_rate"`    // probability of auditing an auto-decision
	CertifyAlpha float64   `yaml:"certify_alpha"` // false-allow rate to certify below (0 = default)
	CertifyDelta float64   `yaml:"certify_delta"` // certification error probability (0 = default)
}

// Config is the full hyperparameter set.
type Config struct {
	Kernel     KernelCfg  `yaml:"kernel"`
	GP         GPCfg      `yaml:"gp"`
	Gateway    GatewayCfg `yaml:"gateway"`
	Featurizer string     `yaml:"featurizer"` // "taxonomy" (default) or "judge"
}

// Default returns the manuscript defaults.
func Default() Config {
	k := kernel.DefaultKernel()
	a := kernel.DefaultAdditiveKernel()
	g := gateway.DefaultConfig()
	return Config{
		Kernel: KernelCfg{
			Type:   KernelProduct,
			Sigma2: k.Sigma2, LTool: k.LTool, LCtx: k.LCtx, Lambda: k.Lambda,
			Additive: AdditiveCfg{
				SStatic: a.SStatic, SGlobal: a.SGlobal, SInter: a.SInter,
				LTool: a.LTool, LCtx: a.LCtx, Lambda: a.Lambda,
			},
		},
		GP: GPCfg{Jitter: 1e-6, MaxIter: 100, Tol: 1e-6},
		Gateway: GatewayCfg{
			TauLow: g.TauLow, TauHigh: g.TauHigh, RefitEvery: g.RefitEvery,
			SafetyEps: g.SafetyEps, BlockEps: g.BlockEps,
			MaxTrain: g.MaxTrain, MaxHist: g.MaxHist, MinTuneN: g.MinTuneN,
			AuditRate:    g.AuditRate,
			CertifyAlpha: DefaultCertifyAlpha,
			CertifyDelta: DefaultCertifyDelta,
		},
		Featurizer: FeaturizerTaxonomy,
	}
}

// Load reads a YAML config, overlaying present keys onto Default(), and
// validates it. A missing file is not an error: Default() is returned.
func Load(path string) (Config, error) {
	cfg := Default()
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return cfg, nil
		}
		return cfg, err
	}
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return Default(), err
	}
	if err := cfg.Validate(); err != nil {
		return Default(), err
	}
	return cfg, nil
}

// Validate checks the selectors and the new (version-2) keys: the kernel type
// and featurizer name, the additive kernel's parameters when selected, the
// costs when present, the audit rate and the certification parameters.
func (c Config) Validate() error {
	switch c.Kernel.Type {
	case "", KernelProduct:
	case KernelAdditive:
		a := c.Kernel.Additive
		if a.SStatic < 0 || a.SGlobal < 0 || a.SInter < 0 || !(a.SStatic+a.SGlobal+a.SInter > 0) {
			return errors.New("config: kernel.additive scales must be >= 0 with a positive sum")
		}
		if !(a.LTool > 0 && a.LCtx > 0 && a.Lambda > 0) {
			return errors.New("config: kernel.additive lengthscales must be > 0")
		}
	default:
		return fmt.Errorf("config: unknown kernel.type %q (want %q or %q)", c.Kernel.Type, KernelProduct, KernelAdditive)
	}
	switch c.Featurizer {
	case "", FeaturizerTaxonomy, FeaturizerJudge:
	default:
		return fmt.Errorf("config: unknown featurizer %q (want %q or %q)", c.Featurizer, FeaturizerTaxonomy, FeaturizerJudge)
	}
	if c.Gateway.Costs != nil {
		if err := c.Gateway.Costs.gateway().Validate(); err != nil {
			return fmt.Errorf("config: gateway.costs: %w", err)
		}
	}
	if r := c.Gateway.AuditRate; math.IsNaN(r) || r < 0 || r > 1 {
		return fmt.Errorf("config: gateway.audit_rate %v outside [0, 1]", r)
	}
	if a := c.Gateway.CertifyAlpha; math.IsNaN(a) || a < 0 || a > 1 {
		return fmt.Errorf("config: gateway.certify_alpha %v outside [0, 1] (0 = default)", a)
	}
	if d := c.Gateway.CertifyDelta; math.IsNaN(d) || d < 0 || d >= 1 {
		return fmt.Errorf("config: gateway.certify_delta %v outside [0, 1) (0 = default)", d)
	}
	return nil
}

// KernelType returns the selected kernel type, "product" when unset.
func (c Config) KernelType() string {
	if c.Kernel.Type == "" {
		return KernelProduct
	}
	return c.Kernel.Type
}

// FeaturizerName returns the selected featurizer, "taxonomy" when unset.
func (c Config) FeaturizerName() string {
	if c.Featurizer == "" {
		return FeaturizerTaxonomy
	}
	return c.Featurizer
}

// ProductKernel builds the product kernel.
func (c Config) ProductKernel() kernel.ProductKernel {
	return kernel.ProductKernel{
		Sigma2: c.Kernel.Sigma2,
		LTool:  c.Kernel.LTool,
		LCtx:   c.Kernel.LCtx,
		Lambda: c.Kernel.Lambda,
	}
}

// AdditiveKernel builds the additive kernel from the kernel.additive block.
func (c Config) AdditiveKernel() kernel.AdditiveKernel {
	a := c.Kernel.Additive
	return kernel.AdditiveKernel{
		SStatic: a.SStatic, SGlobal: a.SGlobal, SInter: a.SInter,
		LTool: a.LTool, LCtx: a.LCtx, Lambda: a.Lambda,
	}
}

// NewKernel builds the kernel selected by kernel.type: the additive kernel
// for "additive", otherwise the product kernel (Load rejects unknown types).
func (c Config) NewKernel() kernel.Kernel {
	if c.KernelType() == KernelAdditive {
		return c.AdditiveKernel()
	}
	return c.ProductKernel()
}

// NewModel builds a Laplace GP model with the configured kernel and solver
// parameters.
func (c Config) NewModel() *gp.LaplaceGPC {
	m := gp.NewLaplaceGPC(c.NewKernel())
	m.Jitter = c.GP.Jitter
	m.MaxIter = c.GP.MaxIter
	m.Tol = c.GP.Tol
	return m
}

func (cc CostsCfg) gateway() gateway.Costs {
	return gateway.Costs{FalseAllow: cc.FalseAllow, FalseBlock: cc.FalseBlock, Ask: cc.Ask}
}

// GatewayConfig builds the gateway.Config.
func (c Config) GatewayConfig() gateway.Config {
	gc := gateway.Config{
		TauLow:     c.Gateway.TauLow,
		TauHigh:    c.Gateway.TauHigh,
		RefitEvery: c.Gateway.RefitEvery,
		SafetyEps:  c.Gateway.SafetyEps,
		BlockEps:   c.Gateway.BlockEps,
		MaxTrain:   c.Gateway.MaxTrain,
		MaxHist:    c.Gateway.MaxHist,
		MinTuneN:   c.Gateway.MinTuneN,
		AuditRate:  c.Gateway.AuditRate,
	}
	if c.Gateway.Costs != nil {
		costs := c.Gateway.Costs.gateway()
		gc.Costs = &costs
	}
	return gc
}

// Certify returns the certification parameters (alpha, delta), substituting
// the defaults for unset (zero) values.
func (c Config) Certify() (alpha, delta float64) {
	alpha, delta = c.Gateway.CertifyAlpha, c.Gateway.CertifyDelta
	if alpha == 0 {
		alpha = DefaultCertifyAlpha
	}
	if delta == 0 {
		delta = DefaultCertifyDelta
	}
	return alpha, delta
}

// NewGateway builds a fresh gateway over the given featurizer.
func (c Config) NewGateway(f featurizer.Featurizer) *gateway.Gateway {
	return gateway.New(f, c.NewModel(), c.GatewayConfig())
}
