// Package config loads trustcalib hyperparameters from a YAML file. Every field
// has the manuscript default, so an absent file or an absent key falls back
// cleanly. The loaded Config builds the kernel, GP model and gateway.
package config

import (
	"errors"
	"io/fs"
	"os"

	"gopkg.in/yaml.v3"

	"github.com/changkun/trustcalib/featurizer"
	"github.com/changkun/trustcalib/gateway"
	"github.com/changkun/trustcalib/gp"
	"github.com/changkun/trustcalib/kernel"
)

// KernelCfg mirrors kernel.ProductKernel.
type KernelCfg struct {
	Sigma2 float64 `yaml:"sigma2"`
	LTool  float64 `yaml:"l_tool"`
	LCtx   float64 `yaml:"l_ctx"`
	Lambda float64 `yaml:"lambda"`
}

// GPCfg holds the Laplace solver parameters.
type GPCfg struct {
	Jitter  float64 `yaml:"jitter"`
	MaxIter int     `yaml:"max_iter"`
	Tol     float64 `yaml:"tol"`
}

// GatewayCfg mirrors gateway.Config.
type GatewayCfg struct {
	TauLow     float64 `yaml:"tau_low"`
	TauHigh    float64 `yaml:"tau_high"`
	RefitEvery int     `yaml:"refit_every"`
	SafetyEps  float64 `yaml:"safety_eps"`
	BlockEps   float64 `yaml:"block_eps"`
	MaxTrain   int     `yaml:"max_train"`
	MaxHist    int     `yaml:"max_hist"`
	MinTuneN   int     `yaml:"min_tune_n"`
}

// Config is the full hyperparameter set.
type Config struct {
	Kernel  KernelCfg  `yaml:"kernel"`
	GP      GPCfg      `yaml:"gp"`
	Gateway GatewayCfg `yaml:"gateway"`
}

// Default returns the manuscript defaults.
func Default() Config {
	k := kernel.DefaultKernel()
	g := gateway.DefaultConfig()
	return Config{
		Kernel: KernelCfg{Sigma2: k.Sigma2, LTool: k.LTool, LCtx: k.LCtx, Lambda: k.Lambda},
		GP:     GPCfg{Jitter: 1e-6, MaxIter: 100, Tol: 1e-6},
		Gateway: GatewayCfg{
			TauLow: g.TauLow, TauHigh: g.TauHigh, RefitEvery: g.RefitEvery,
			SafetyEps: g.SafetyEps, BlockEps: g.BlockEps,
			MaxTrain: g.MaxTrain, MaxHist: g.MaxHist, MinTuneN: g.MinTuneN,
		},
	}
}

// Load reads a YAML config, overlaying present keys onto Default(). A missing
// file is not an error: Default() is returned.
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
	return cfg, nil
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

// NewModel builds a Laplace GP model with the configured kernel and solver
// parameters.
func (c Config) NewModel() *gp.LaplaceGPC {
	m := gp.NewLaplaceGPC(c.ProductKernel())
	m.Jitter = c.GP.Jitter
	m.MaxIter = c.GP.MaxIter
	m.Tol = c.GP.Tol
	return m
}

// GatewayConfig builds the gateway.Config.
func (c Config) GatewayConfig() gateway.Config {
	return gateway.Config{
		TauLow:     c.Gateway.TauLow,
		TauHigh:    c.Gateway.TauHigh,
		RefitEvery: c.Gateway.RefitEvery,
		SafetyEps:  c.Gateway.SafetyEps,
		BlockEps:   c.Gateway.BlockEps,
		MaxTrain:   c.Gateway.MaxTrain,
		MaxHist:    c.Gateway.MaxHist,
		MinTuneN:   c.Gateway.MinTuneN,
	}
}

// NewGateway builds a fresh gateway over the given featurizer.
func (c Config) NewGateway(f featurizer.Featurizer) *gateway.Gateway {
	return gateway.New(f, c.NewModel(), c.GatewayConfig())
}
