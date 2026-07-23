# Current benchmarks

Release build, Windows, Grand Geneve iteration-5 cache, 2026-07-23. These are
kernel measurements, not end-to-end Mobility timings.

## Active bounded top-K

Full prepared workload (81,844 contexts, 328,197 steps, 1,110 zones), eight
threads; active defaults with `factor_map_max_depth=5` and
`deep_continuation_state_limit=2`:

| Policy | Wall time | Aggregate Rust | Factor-map CPU |
|---|---:|---:|---:|
| Active depth-5 / deep-width-2 | 34.642 s | 201.006 s | 169.204 s |
| Superseded depth-99 / width-1 | 38.303 s | 216.235 s | 186.209 s |

The interleaved full-workload comparison improves wall time by 9.6% and
aggregate Rust time by 7.0%. On the matched 41-stratum exact-top-10 audit, the
active policy raises the post-stratified certified `Mass@10` estimate from
0.832 to 0.839. The exact oracle solved 183 of 396 sampled contexts, covering
99.7% of the population by oracle-certifiable stratum.

The local-score cache uses a deterministic hasher for its trusted integer
tuple keys. Across five 20,000-context release runs this improved median wall
time from 9.246 to 8.898 seconds (-3.8%) and aggregate Rust search time from
54.240 to 51.594 seconds (-4.9%). Packing the key regressed runtime and is not
active.

## Prior depth-5 bounded top-K baseline

Full prepared Grand Geneve workload (81,844 contexts, 328,197 steps, 1,110
zones), eight threads; `frontier_width=40`, `proposal_limit_per_source=16`,
`factor_map_max_depth=5`, `stitch_bias=1`, `continuation_state_limit=1`,
`continuation_proposal_limit=1`, `symmetric_message_limit=4`,
`symmetric_state_limit=4`, `symmetric_forward_proposal_limit=20`,
`seam_refresh_per_prefix=1`, `top_k=10`.

| Policy | Wall time | Complete | Infeasible |
|---|---:|---:|---:|
| Symmetric factor map, depth <=5 | 22.35 s | 70,801 | 11,043 |
| Asymmetric factor map | 16.30 s | 70,335 | 11,509 |
| Binned surface comparator | 8.785 s | 70,143 | 11,701 |
| Heuristic comparator | 6.591 s | 70,351 | 11,493 |

This was the 2026-07-22 default and remains a useful comparison point. Across five deterministic
50-context exact cohorts, the default scores mean `Mass@10` 0.892 and
`Recall@10` 0.884. The previous compact partial-proposal budget of 8 scored
0.853/0.830; 20 is the measured quality/runtime knee (24 reaches 0.894).
Known-prefix factor scoring and compact repeated-anchor handoff account for
the gain. Exact-oracle results are cached by immutable-input/scorer fingerprint.
The later depth-99 setting used factor-map support on the primary reverse path.
In a full 81,844-context interleaved comparison, it ran in 39.644 s wall /
233.138 s aggregate Rust; the then-matched depth-5 setting took 47.978 s /
292.763 s. That result was confounded because the depth setting also activated
the 16-state deep continuation channel. The active policy above separates
those knobs. See
[`experiments/active-bidirectional-top-k.md`](experiments/active-bidirectional-top-k.md)
for the current decision record. Older throughput and quality measurements are
experiment history, not the active baseline.

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
