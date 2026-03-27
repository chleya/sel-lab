# Phase 3 Digits Merge Sweep

- Best average-accuracy scale vs `adapt_only`: `0.02` (+5.3%)
- Best forgetting scale vs `adapt_only`: `0.20` (-3.2%)
- Closest scale to fixed: `0.02` (-11.5% vs fixed)

| Merge Scale | Avg Delta vs adapt_only | Forgetting Delta | Current Delta | Prior Delta | Gap vs Fixed | Unit Delta |
|---|---:|---:|---:|---:|---:|---:|
| 0.02 | +5.3% | -4.5% | +11.6% | +3.2% | -11.5% | +10.4 |
| 0.05 | +4.5% | -4.8% | +10.3% | +2.6% | -12.3% | +10.2 |
| 0.10 | +4.0% | -4.6% | +8.5% | +2.5% | -12.8% | +10.0 |
| 0.15 | +4.1% | -3.8% | +8.1% | +2.7% | -12.7% | +10.0 |
| 0.20 | +4.0% | -3.2% | +7.5% | +2.9% | -12.8% | +10.0 |
