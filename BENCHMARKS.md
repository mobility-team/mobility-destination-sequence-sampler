# Current benchmarks

Release build, Windows, Grand Geneve iteration-5 cache, 2026-07-21. These are
kernel measurements, not end-to-end Mobility timings.

## Active bounded top-K

Full prepared Grand Geneve workload (81,844 contexts, 328,197 steps, 1,110
zones), eight threads; `frontier_width=40`, `proposal_limit_per_source=16`,
`factor_map_max_depth=5`, `stitch_bias=1`, `continuation_state_limit=1`,
`continuation_proposal_limit=1`, `seam_refresh_per_prefix=1`, `top_k=10`.

| Policy | Wall time | Complete | Infeasible |
|---|---:|---:|---:|
| Factor map, depth <=5 (default) | 23.011 s | 70,325 | 11,519 |
| Binned surface comparator | 8.785 s | 70,143 | 11,701 |
| Heuristic comparator | 6.591 s | 70,351 | 11,493 |

The default remains comfortably within the 30-second target. A five-per-stratum
audit (69 oracle-proven contexts; 99.4% population coverage) estimates
`Mass@10` at 0.784 (width 32: 0.772; surface 0.735; heuristic 0.707).
On the identical 69 contexts, the centred stitch scores 0.759.
An independent seed-43, three-per-stratum cohort estimates 0.774 over 36
oracle-proven contexts (91.5% coverage).
Older throughput and
quality measurements are experiment history, not the active baseline.

## Earlier bounded top-K reference

1,000 final-home contexts (4,767 steps, 1,110 zones), eight threads:
`frontier_width=32`, `proposal_limit_per_source=16`,
`continuation_state_limit=1`, `continuation_proposal_limit=1`,
`seam_refresh_per_prefix=1`, `top_k=10`.

| Wall time | Complete | Infeasible |
|---:|---:|---:|
| 0.135 s | 787 | 213 |

This is retained only to compare historic experiments. Seam lookahead gave no
quality gain and was removed.

## Exact-oracle K sweep

Five deterministic hash cohorts (seeds 42–46), 50 exact-proven contexts each.
Contexts have 3–4 layers and end at home; variable/repeated anchors and
additional fixed destinations are included. `frontier_width=128` fits every K;
the reference is a fixed exact top-500 support. Values are cohort means.

| K | Recall@K | Mass@K | Mass@500 | Efficiency | Bounded ms/context |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.632 | 0.652 | 0.084 | 0.880 | 1.48 |
| 20 | 0.565 | 0.592 | 0.126 | 0.828 | 1.29 |
| 50 | 0.464 | 0.505 | 0.196 | 0.727 | 1.36 |
| 100 | 0.385 | 0.437 | 0.254 | 0.618 | 1.49 |

`Mass@K` is conditional mass retained from the oracle top K; `Mass@500` is
returned-plan mass in one fixed reference support. More output increases total
captured mass but recovers a smaller share of the corresponding exact top K.
The dominant loss remains proposal support; beam loss is small but no longer
zero for the broader anchor cohort. Reproduce with `just compare-k-sweep` or
`just compare-k-sweep-seeds`.

Historical measurements and decisions are under [`experiments/`](experiments/README.md).
