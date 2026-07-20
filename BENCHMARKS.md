# Current benchmarks

Release build, Windows, Grand Geneve iteration-5 cache, 2026-07-21. These are
kernel measurements, not end-to-end Mobility timings.

## Active bounded top-K

1,000 fixed-terminal contexts (4,767 steps, 1,110 zones), eight threads:
`frontier_width=32`, `proposal_limit_per_source=16`,
`continuation_state_limit=1`, `continuation_proposal_limit=1`,
`seam_refresh_per_prefix=1`, `top_k=10`.

| Wall time | Complete | Infeasible |
|---:|---:|---:|
| 0.135 s | 787 | 213 |

The refresh evaluated 357,150 proposals, added 494 boundary states, and used
19% of aggregate Rust search time. Seam lookahead gave no quality gain and was
removed (0.258 s versus 0.135 s).

## Exact-oracle quality

On 50 exact-proven short contexts (56 explicit oracle skips), retained
conditional exact top-10 mass was:

| F-to-B alternatives per retained prefix | Mass |
|---:|---:|
| 0 | 0.7599 |
| 1 (default) | 0.7676 |
| 2 | 0.7762 |
| 4 | 0.7996 |

The active limitation is proposal support, especially early forward layers;
larger refresh counts are an opt-in quality/runtime trade-off. Reproduce with
`experiments.benchmarks.perf_bidirectional_grand_geneve` and
`experiments.analysis.compare_seam_refresh`.

Historical measurements and decisions are under [`experiments/`](experiments/README.md).
