# Phase 3 Archived Specialist Quantization Benchmark

- Goal: stress-test how much precision archived specialists need under the current leading mechanism.
- Leading policy: `task_specialist_clone_current_path_boost_merge`
- Digits suites: `digits_pairs, digits_pairs_noisy, digits_pairs_permuted`
- Quantization bits tested: `0, 4, 2`
- Best bits by mean gain vs fixed in this band: `4`

## Aggregate Summary

- `0`-bit archived specialists: mean avg delta vs `adapt_only` +11.9%, mean current delta +31.4%, mean forgetting delta -10.1%, mean gain vs fixed -5.4%, mean delta vs unquantized +0.0%
- `4`-bit archived specialists: mean avg delta vs `adapt_only` +13.3%, mean current delta +34.1%, mean forgetting delta -12.8%, mean gain vs fixed -4.0%, mean delta vs unquantized +1.4%
- `2`-bit archived specialists: mean avg delta vs `adapt_only` +13.1%, mean current delta +33.9%, mean forgetting delta -12.9%, mean gain vs fixed -4.2%, mean delta vs unquantized +1.2%

## Suite Breakdown

### digits_pairs

- `0`-bit: avg delta vs `adapt_only` +11.0%, current delta +23.9%, forgetting delta -10.0%, gain vs fixed -5.8%
- `4`-bit: avg delta vs `adapt_only` +12.9%, current delta +28.4%, forgetting delta -13.1%, gain vs fixed -3.9%
- `2`-bit: avg delta vs `adapt_only` +12.5%, current delta +27.8%, forgetting delta -13.3%, gain vs fixed -4.3%

### digits_pairs_noisy

- `0`-bit: avg delta vs `adapt_only` +10.3%, current delta +20.1%, forgetting delta -7.0%, gain vs fixed -10.8%
- `4`-bit: avg delta vs `adapt_only` +11.9%, current delta +23.0%, forgetting delta -9.5%, gain vs fixed -9.2%
- `2`-bit: avg delta vs `adapt_only` +12.2%, current delta +23.1%, forgetting delta -9.2%, gain vs fixed -9.0%

### digits_pairs_permuted

- `0`-bit: avg delta vs `adapt_only` +14.5%, current delta +50.1%, forgetting delta -13.1%, gain vs fixed +0.5%
- `4`-bit: avg delta vs `adapt_only` +15.1%, current delta +51.0%, forgetting delta -15.7%, gain vs fixed +1.1%
- `2`-bit: avg delta vs `adapt_only` +14.7%, current delta +50.9%, forgetting delta -16.3%, gain vs fixed +0.7%
