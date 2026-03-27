# Phase 3 Suite Transfer

- Specialist merge scale: `0.05`
- Best suite by average accuracy delta: `rotated` (+2.4% vs adapt_only)
- Best suite by forgetting delta: `feature_shift` (+3.1% vs adapt_only)

| Task Suite | Avg Delta vs adapt_only | Forgetting Delta | Current Delta | Prior Delta | Unit Delta |
|---|---:|---:|---:|---:|---:|
| default | +1.2% | +0.2% | +0.0% | +1.6% | +11.0 |
| rotated | +2.4% | +1.5% | +1.2% | +2.8% | +11.0 |
| feature_shift | +2.2% | +3.1% | -1.0% | +3.3% | +11.0 |
| noisy | +2.1% | +0.7% | +0.4% | +2.7% | +11.0 |
