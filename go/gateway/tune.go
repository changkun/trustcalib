package gateway

// TuneThresholds picks (tauLow, tauHigh) by grid search over linspace(0.05,
// 0.95, 19), a direct port of experiment/gateway.py tune_thresholds. Among
// pairs (lo < hi) whose auto-decisions respect a safety cap (false-allow rate
// <= safetyEps/2), a usefulness cap (false-block rate <= blockEps) and a
// minimum auto-coverage (ask rate <= 0.6), it returns the one with the smallest
// ask rate. If no pair is feasible, or the best still escalates more than 70%
// of traffic, it falls back to the default (0.35, 0.65).
//
// The safety cap is tightened on the tuning set (safetyEps/2, floored at 1e-3)
// to leave headroom for distribution shift.
func TuneThresholds(pHat []float64, label []int, safetyEps, blockEps float64) (low, high float64) {
	const defaultLow, defaultHigh = 0.35, 0.65

	safetyEpsTuned := safetyEps / 2.0
	if safetyEpsTuned < 1e-3 {
		safetyEpsTuned = 1e-3
	}

	n := len(pHat)
	if n == 0 {
		return defaultLow, defaultHigh
	}

	grid := linspace(0.05, 0.95, 19)
	bestAsk := 2.0
	bestLow, bestHigh := defaultLow, defaultHigh
	found := false

	for _, lo := range grid {
		for _, hi := range grid {
			if hi <= lo {
				continue
			}
			var nAllow, nBlock, nAsk, faNum, fbNum int
			for i := 0; i < n; i++ {
				p := pHat[i]
				switch {
				case p > hi:
					nAllow++
					if label[i] == 0 {
						faNum++
					}
				case p < lo:
					nBlock++
					if label[i] == 1 {
						fbNum++
					}
				default:
					nAsk++
				}
			}
			askRate := float64(nAsk) / float64(n)
			if askRate > 0.6 {
				continue
			}
			da := nAllow
			if da < 1 {
				da = 1
			}
			db := nBlock
			if db < 1 {
				db = 1
			}
			falseAllow := float64(faNum) / float64(da)
			falseBlock := float64(fbNum) / float64(db)
			if falseAllow > safetyEpsTuned || falseBlock > blockEps {
				continue
			}
			if askRate < bestAsk {
				bestAsk = askRate
				bestLow, bestHigh = lo, hi
				found = true
			}
		}
	}
	if !found || bestAsk > 0.7 {
		return defaultLow, defaultHigh
	}
	return bestLow, bestHigh
}

// linspace returns num evenly spaced values from start to stop inclusive.
func linspace(start, stop float64, num int) []float64 {
	if num <= 1 {
		return []float64{start}
	}
	step := (stop - start) / float64(num-1)
	out := make([]float64, num)
	for i := 0; i < num; i++ {
		out[i] = start + step*float64(i)
	}
	return out
}
