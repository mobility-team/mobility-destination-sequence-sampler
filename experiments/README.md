# Experiments

Only bounded bidirectional top-K is active.

| Need | Location |
|---|---|
| Current hypothesis/decision | `active-bidirectional-top-k.md` |
| Current measurements | `../BENCHMARKS.md`, `benchmarks/bidirectional-top-k.md` |
| Quality harness | `analysis/compare_bidirectional_top_k_grand_geneve.py` |
| Runtime harness | `benchmarks/perf_bidirectional_grand_geneve.py` |
| Exact oracle | `../rust/ternary_reference.rs` |
| Archived directions | `historical.md`, `lessons-learned.md` |

The next hypothesis is continuation-aware support for early forward proposals.
Archived source is recoverable at `research-archive-2026-07-21`; do not revive
it without a new, testable hypothesis.
