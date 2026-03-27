# Phase 3 Specialist Merge Sweep

- Refined local sweep around the current promising merge-scale region.
- Best average-accuracy scale: `0.05` (+1.2% vs adapt_only)
- Best forgetting scale: `0.20` (+0.8% forgetting delta vs adapt_only)

| Merge Scale | Avg Delta vs adapt_only | Forgetting Delta | Current Delta | Prior Delta | Unit Delta |
|---|---:|---:|---:|---:|---:|
| 0.05 | +1.2% | +0.2% | +0.0% | +1.6% | +11.0 |
| 0.10 | +1.1% | +0.4% | -0.2% | +1.6% | +11.0 |
| 0.15 | +1.1% | +0.7% | -0.6% | +1.7% | +11.0 |
| 0.20 | +1.0% | +0.8% | -1.0% | +1.7% | +11.0 |
