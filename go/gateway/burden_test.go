package gateway_test

import (
	"testing"

	"github.com/changkun/trustcalib/config"
	"github.com/changkun/trustcalib/featurizer"
	"github.com/changkun/trustcalib/featurizer/trustcalib"
	"github.com/changkun/trustcalib/gateway"
	"github.com/changkun/trustcalib/internal/testutil"
)

type trajPoint struct {
	Tool      string `json:"tool"`
	Target    string `json:"target"`
	Task      string `json:"task"`
	ArgRisk   int    `json:"arg_risk"`
	T         int    `json:"t"`
	Label     int    `json:"label"`
	OracleYes int    `json:"oracle_yes"`
}

// TestGatewayBurden ports test_gateway_burden.py: replaying a recorded stream
// through the online gateway (querying the human only on ASK) yields
// substantial, accurate, safe automation with far fewer queries than the
// always-escalate baseline. The Python oracle is not ported; only its recorded
// labels/decisions are replayed.
func TestGatewayBurden(t *testing.T) {
	var traj struct {
		T1     int         `json:"t1"`
		T2     int         `json:"t2"`
		Points []trajPoint `json:"points"`
	}
	testutil.Load(t, "trajectory.json", &traj)

	cfg := config.Default().GatewayConfig()
	g := gateway.New(trustcalib.New(), config.Default().NewModel(), cfg)

	var (
		testTotal, testAuto, autoCorrect int
		allowTotal, falseAllow           int
		queries                          int
		tuned                            bool
	)

	for _, p := range traj.Points {
		if !tuned && p.T >= traj.T2 {
			g.Tune()
			tuned = true
		}
		pt := featurizer.Point{
			Tool: p.Tool, Target: p.Target, Task: p.Task,
			ArgRisk: p.ArgRisk, T: float64(p.T),
		}
		dec, _ := g.Decide(pt)

		if dec == gateway.Ask {
			queries++
			if err := g.Observe(pt, p.Label == 1); err != nil {
				t.Fatal(err)
			}
		}

		if p.T < traj.T2 {
			continue
		}
		testTotal++
		switch dec {
		case gateway.Allow:
			testAuto++
			allowTotal++
			if p.OracleYes == 1 {
				autoCorrect++
			} else {
				falseAllow++
			}
		case gateway.Block:
			testAuto++
			if p.OracleYes == 0 {
				autoCorrect++
			}
		}
	}

	autoRate := float64(testAuto) / float64(testTotal)
	accuracy := 1.0
	if testAuto > 0 {
		accuracy = float64(autoCorrect) / float64(testAuto)
	}
	falseAllowRate := 0.0
	if allowTotal > 0 {
		falseAllowRate = float64(falseAllow) / float64(allowTotal)
	}
	queryFrac := float64(queries) / float64(len(traj.Points))

	t.Logf("auto-rate=%.3f accuracy=%.3f false-allow=%.3f query-frac=%.3f (test n=%d)",
		autoRate, accuracy, falseAllowRate, queryFrac, testTotal)

	if autoRate <= 0.5 {
		t.Errorf("auto-rate %.3f, want > 0.5", autoRate)
	}
	if accuracy <= 0.90 {
		t.Errorf("auto accuracy %.3f, want > 0.90", accuracy)
	}
	if falseAllowRate >= 0.05 {
		t.Errorf("false-allow rate %.3f, want < 0.05", falseAllowRate)
	}
	if queryFrac >= 0.6 {
		t.Errorf("query fraction %.3f, want < 0.6", queryFrac)
	}
}
