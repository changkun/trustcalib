package config

import (
	"os"
	"path/filepath"
	"testing"
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
