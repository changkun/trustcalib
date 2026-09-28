package config

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/changkun/trustcalib/gateway"
	"github.com/changkun/trustcalib/kernel"
)

func TestDefault(t *testing.T) {
	c := Default()
	if c.Kernel.Sigma2 != 1.6 || c.Kernel.Lambda != 200.0 {
		t.Errorf("kernel defaults: %+v", c.Kernel)
	}
	if c.GP.MaxIter != 100 || c.GP.Jitter != 1e-6 {
		t.Errorf("gp defaults: %+v", c.GP)
	}
	if c.Gateway.TauLow != 0.35 || c.Gateway.TauHigh != 0.65 || c.Gateway.RefitEvery != 8 {
		t.Errorf("gateway defaults: %+v", c.Gateway)
	}
}

func TestLoadMissingFile(t *testing.T) {
	c, err := Load(filepath.Join(t.TempDir(), "nope.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	if c != Default() {
		t.Fatalf("missing file should yield defaults, got %+v", c)
	}
}

func TestLoadOverlay(t *testing.T) {
	path := filepath.Join(t.TempDir(), "c.yaml")
	yaml := "kernel:\n  sigma2: 2.5\ngateway:\n  refit_every: 3\n"
	if err := os.WriteFile(path, []byte(yaml), 0o644); err != nil {
		t.Fatal(err)
	}
	c, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if c.Kernel.Sigma2 != 2.5 {
		t.Errorf("overlay sigma2 = %v, want 2.5", c.Kernel.Sigma2)
	}
	if c.Gateway.RefitEvery != 3 {
		t.Errorf("overlay refit_every = %v, want 3", c.Gateway.RefitEvery)
	}
	// Untouched keys keep their defaults.
	if c.Kernel.LTool != 1.1 || c.GP.MaxIter != 100 {
		t.Errorf("defaults not preserved: %+v %+v", c.Kernel, c.GP)
	}
}

func TestLoadInvalid(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bad.yaml")
	if err := os.WriteFile(path, []byte("kernel: [this is not a map"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(path); err == nil {
		t.Fatal("expected unmarshal error")
	}
}

func TestBuilders(t *testing.T) {
	c := Default()
	k := c.ProductKernel()
	if k.Sigma2 != 1.6 || k.LCtx != 1.2 {
		t.Errorf("kernel build: %+v", k)
	}
	m := c.NewModel()
	if m.MaxIter != 100 || m.Tol != 1e-6 {
		t.Errorf("model build: maxiter=%d tol=%v", m.MaxIter, m.Tol)
	}
	gc := c.GatewayConfig()
	if gc.MinTuneN != 30 || gc.MaxTrain != 2000 {
		t.Errorf("gateway config: %+v", gc)
	}
}

func TestDefaultAuditAndCostKeys(t *testing.T) {
	c := Default()
	if c.KernelType() != KernelProduct || c.FeaturizerName() != FeaturizerTaxonomy {
		t.Errorf("selectors: kernel %q featurizer %q", c.KernelType(), c.FeaturizerName())
	}
	if _, ok := c.NewKernel().(kernel.ProductKernel); !ok {
		t.Errorf("default kernel should be the product kernel, got %T", c.NewKernel())
	}
	if c.AdditiveKernel() != kernel.DefaultAdditiveKernel() {
		t.Errorf("additive defaults: %+v", c.AdditiveKernel())
	}
	if c.Gateway.Costs != nil || c.Gateway.AuditRate != 0 {
		t.Errorf("costs/audits should be off by default: %+v", c.Gateway)
	}
	if a, d := c.Certify(); a != 0.01 || d != 0.05 {
		t.Errorf("certify defaults (%v, %v)", a, d)
	}
	if gc := c.GatewayConfig(); gc.Costs != nil || gc.AuditRate != 0 {
		t.Errorf("gateway config: %+v", gc)
	}
	if err := c.Validate(); err != nil {
		t.Errorf("defaults should validate: %v", err)
	}
	// Zero-valued selectors (a state file written before they existed) mean
	// the backward-compatible defaults.
	var old Config
	if old.KernelType() != KernelProduct || old.FeaturizerName() != FeaturizerTaxonomy {
		t.Errorf("zero selectors: %q %q", old.KernelType(), old.FeaturizerName())
	}
	if a, d := old.Certify(); a != 0.01 || d != 0.05 {
		t.Errorf("zero certify params should default, got (%v, %v)", a, d)
	}
}

func TestLoadAuditAndCostKeys(t *testing.T) {
	path := filepath.Join(t.TempDir(), "c.yaml")
	yaml := `kernel:
  type: additive
  additive:
    s_static: 2.0
    lambda: 50
gateway:
  costs:
    false_allow: 10
    false_block: 4
    ask: 1
  audit_rate: 0.05
  certify_alpha: 0.25
  certify_delta: 0.1
featurizer: judge
`
	if err := os.WriteFile(path, []byte(yaml), 0o644); err != nil {
		t.Fatal(err)
	}
	c, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	k, ok := c.NewKernel().(kernel.AdditiveKernel)
	if !ok {
		t.Fatalf("kernel.type additive should build an AdditiveKernel, got %T", c.NewKernel())
	}
	want := kernel.DefaultAdditiveKernel()
	want.SStatic, want.Lambda = 2.0, 50
	if k != want {
		t.Errorf("additive kernel %+v, want %+v", k, want)
	}
	if _, ok := c.NewModel().Kernel.(kernel.AdditiveKernel); !ok {
		t.Errorf("model kernel %T", c.NewModel().Kernel)
	}
	gc := c.GatewayConfig()
	if gc.Costs == nil || *gc.Costs != (gateway.Costs{FalseAllow: 10, FalseBlock: 4, Ask: 1}) {
		t.Fatalf("costs: %+v", gc.Costs)
	}
	if lo, hi := gc.Thresholds(); lo != 0.25 || hi != 0.9 {
		t.Errorf("cost band (%v, %v)", lo, hi)
	}
	if gc.AuditRate != 0.05 {
		t.Errorf("audit_rate %v", gc.AuditRate)
	}
	if a, d := c.Certify(); a != 0.25 || d != 0.1 {
		t.Errorf("certify (%v, %v)", a, d)
	}
	if c.FeaturizerName() != FeaturizerJudge {
		t.Errorf("featurizer %q", c.FeaturizerName())
	}
	// The product parameters keep their defaults.
	if c.Kernel.Lambda != 200 || c.Kernel.Sigma2 != 1.6 {
		t.Errorf("product params changed: %+v", c.Kernel)
	}
}

func TestLoadRejectsInvalidAuditAndCostKeys(t *testing.T) {
	cases := map[string]string{
		"kernel type":     "kernel:\n  type: spline\n",
		"featurizer":      "featurizer: magic\n",
		"costs missing":   "gateway:\n  costs:\n    false_allow: 10\n",
		"costs negative":  "gateway:\n  costs:\n    false_allow: 10\n    false_block: -4\n    ask: 1\n",
		"audit rate":      "gateway:\n  audit_rate: 1.5\n",
		"certify alpha":   "gateway:\n  certify_alpha: 2\n",
		"certify delta":   "gateway:\n  certify_delta: 1\n",
		"additive scales": "kernel:\n  type: additive\n  additive:\n    s_static: -1\n",
		"additive lambda": "kernel:\n  type: additive\n  additive:\n    lambda: 0\n",
	}
	for name, yaml := range cases {
		path := filepath.Join(t.TempDir(), "c.yaml")
		if err := os.WriteFile(path, []byte(yaml), 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := Load(path); err == nil {
			t.Errorf("%s: expected a validation error", name)
		}
	}
}

// TestExampleConfigLoads keeps trustcalib.example.yaml loadable and equal to
// the defaults (it documents them).
func TestExampleConfigLoads(t *testing.T) {
	c, err := Load(filepath.Join("..", "trustcalib.example.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	if c != Default() {
		t.Fatalf("example config differs from defaults:\n got %+v\nwant %+v", c, Default())
	}
}
