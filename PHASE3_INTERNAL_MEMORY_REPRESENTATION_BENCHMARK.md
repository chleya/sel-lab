# Phase 3 Internal Memory Representation Benchmark

- Digits suites: `digits_pairs, digits_pairs_noisy, digits_pairs_permuted`
- Archived specialist precision: `4`-bit
- Best case by mean gain vs fixed: `full`

## Aggregate Summary

- `full` (budget `0`, mode `recency`): mean avg delta vs `adapt_only` +13.3%, mean current delta +34.1%, mean forgetting delta -12.8%, mean gain vs fixed -4.0%, mean delta vs full budget +0.0%
- `recency_2` (budget `2`, mode `recency`): mean avg delta vs `adapt_only` +11.8%, mean current delta +32.0%, mean forgetting delta -13.5%, mean gain vs fixed -5.5%, mean delta vs full budget -1.5%
- `prototype_2` (budget `2`, mode `prototype`): mean avg delta vs `adapt_only` +11.8%, mean current delta +32.1%, mean forgetting delta -13.7%, mean gain vs fixed -5.6%, mean delta vs full budget -1.5%

## Suite Breakdown

### digits_pairs

- `full`: avg delta vs `adapt_only` +12.9%, current delta +28.4%, forgetting delta -13.1%, gain vs fixed -3.9%
- `recency_2`: avg delta vs `adapt_only` +10.7%, current delta +25.5%, forgetting delta -14.6%, gain vs fixed -6.2%
- `prototype_2`: avg delta vs `adapt_only` +10.6%, current delta +25.6%, forgetting delta -14.9%, gain vs fixed -6.2%

### digits_pairs_noisy

- `full`: avg delta vs `adapt_only` +11.9%, current delta +23.0%, forgetting delta -9.5%, gain vs fixed -9.2%
- `recency_2`: avg delta vs `adapt_only` +9.8%, current delta +20.9%, forgetting delta -11.1%, gain vs fixed -11.3%
- `prototype_2`: avg delta vs `adapt_only` +9.8%, current delta +20.8%, forgetting delta -11.1%, gain vs fixed -11.3%

### digits_pairs_permuted

- `full`: avg delta vs `adapt_only` +15.1%, current delta +51.0%, forgetting delta -15.7%, gain vs fixed +1.1%
- `recency_2`: avg delta vs `adapt_only` +14.9%, current delta +49.8%, forgetting delta -14.9%, gain vs fixed +0.9%
- `prototype_2`: avg delta vs `adapt_only` +14.9%, current delta +49.9%, forgetting delta -15.0%, gain vs fixed +0.8%
