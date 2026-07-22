# Experiments

Only bounded bidirectional top-K is active.

| Need | Location |
|---|---|
| Current hypothesis/decision | `active-bidirectional-top-k.md` |
| Current measurements | `../BENCHMARKS.md`, `benchmarks/bidirectional-top-k.md` |
| Quality harness | `analysis/compare_bidirectional_top_k_grand_geneve.py` |
| Runtime harness | `benchmarks/perf_bidirectional_grand_geneve.py` |
| Exact oracle | `../rust/oracle.rs` |
| Archived directions | `historical.md`, `lessons-learned.md` |

The next hypothesis is continuation-aware support for early forward proposals.
Archived source is recoverable from the `research-archive-2026-07-21` Git tag
(`git show research-archive-2026-07-21:<path>`); do not revive it without a
new, testable hypothesis. [`measurement-guide.md`](measurement-guide.md)
defines the active harness outputs and historical diagnostics.
