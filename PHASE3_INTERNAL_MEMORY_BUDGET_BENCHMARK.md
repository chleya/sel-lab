# Phase 3 Internal Memory Budget Benchmark

- Digits suites: `digits_pairs, digits_pairs_noisy, digits_pairs_permuted`
- Archived specialist precision: `4`-bit
- Archived specialist budgets tested: `0, 2, 1` (`0` means full budget)
- Best budget by mean gain vs fixed: `0`

## Aggregate Summary

- budget `0`: mean avg delta vs `adapt_only` +13.3%, mean current delta +34.1%, mean forgetting delta -12.8%, mean gain vs fixed -4.0%, mean delta vs full budget +0.0%
- budget `2`: mean avg delta vs `adapt_only` +11.8%, mean current delta +32.0%, mean forgetting delta -13.5%, mean gain vs fixed -5.5%, mean delta vs full budget -1.5%
- budget `1`: mean avg delta vs `adapt_only` +11.3%, mean current delta +31.5%, mean forgetting delta -13.7%, mean gain vs fixed -6.1%, mean delta vs full budget -2.0%

## Suite Breakdown

### digits_pairs

- budget `0`: avg delta vs `adapt_only` +12.9%, current delta +28.4%, forgetting delta -13.1%, gain vs fixed -3.9%
- budget `2`: avg delta vs `adapt_only` +10.7%, current delta +25.5%, forgetting delta -14.6%, gain vs fixed -6.2%
- budget `1`: avg delta vs `adapt_only` +10.7%, current delta +25.9%, forgetting delta -14.7%, gain vs fixed -6.1%

### digits_pairs_noisy

- budget `0`: avg delta vs `adapt_only` +11.9%, current delta +23.0%, forgetting delta -9.5%, gain vs fixed -9.2%
- budget `2`: avg delta vs `adapt_only` +9.8%, current delta +20.9%, forgetting delta -11.1%, gain vs fixed -11.3%
- budget `1`: avg delta vs `adapt_only` +8.9%, current delta +19.9%, forgetting delta -11.9%, gain vs fixed -12.2%

### digits_pairs_permuted

- budget `0`: avg delta vs `adapt_only` +15.1%, current delta +51.0%, forgetting delta -15.7%, gain vs fixed +1.1%
- budget `2`: avg delta vs `adapt_only` +14.9%, current delta +49.8%, forgetting delta -14.9%, gain vs fixed +0.9%
- budget `1`: avg delta vs `adapt_only` +14.3%, current delta +48.6%, forgetting delta -14.5%, gain vs fixed +0.2%
