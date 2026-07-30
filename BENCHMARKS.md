# Current benchmarks

Release build, Windows, Grand Geneve iteration-5 cache, 2026-07-27. These are
kernel measurements, not end-to-end Mobility timings.

## Active bounded top-K

### Compiled exact factor scoring

The active kernel prepares one immutable scorer per context layer. Factor-map,
reverse-prefix, and pricing scans reuse direct activity-table references plus
the layer's first-choice, terminal, and adjacent-step state. Final plan scoring
and all `f64` factor arithmetic remain exact.

On a locked 20,000-context, two-block pure-performance validation, outputs and
all measured work counters are identical:

| Metric | Prior repeated setup | Compiled scorer | Paired-block delta |
|---|---:|---:|---:|
| Wall time, raw median | 12.772 s | 12.193 s | -5.8% |
| Aggregate Rust, raw median | 74.788 s | 64.195 s | -14.4% |
| Factor-map CPU, raw median | 55.404 s | 45.808 s | -17.3% |
| Pricing CPU, raw median | 8.779 s | 8.022 s | -8.9% |

The paired aggregate-Rust blocks span -15.9% to -12.8%, factor-map blocks
span -18.9% to -15.7%, and wall blocks span -9.0% to -2.7%.

### Adaptive structural factor-map router

The active `adaptive_factor_map` policy keeps partial symmetric guidance when
a variable run contains at least two adjacent unknowns or an anchor repeats.
When fixed destinations isolate every variable and no anchor repeats, it uses
ordinary exact factor maps and skips the second reverse pass. The rule routes
16,144 contexts, or 28.4% of the workload not handled by the exact two-step
path, to the cheaper channel.

On a locked ten-per-stratum validation cohort, the oracle certified 258 of 396
sampled contexts. The structural router and the prior always-symmetric policy
were identical at post-stratified `Mass@10=0.893736`, post-stratified
`Recall@10=0.885990`, and 24 certified zero-overlap cases.

The linked 20,000-context, two-block release validation reports:

| Metric | Always symmetric | Adaptive | Paired-block delta |
|---|---:|---:|---:|
| Wall time, raw median | 11.105 s | 10.491 s | -3.7% |
| Aggregate Rust, raw median | 66.217 s | 62.978 s | -3.9% |
| Factor-map CPU, raw median | 49.263 s | 46.550 s | -4.8% |

Raw median ratios are -5.5% wall, -4.9% aggregate Rust, and -5.5% factor-map
CPU. The paired wall blocks span -7.5% to +0.1%; both aggregate-Rust blocks
improve (-4.9% and -3.0%). The promotion is based on the predeclared >=3%
paired wall improvement target plus the independent, exactly unchanged
quality validation.

### Adaptive local interacting-pair pricing

The active router probes 4x4 exact interacting-pair neighborhoods and expands
to 8x8 only when the best probe candidate improves the current working Kth
score by more than 0.2. The former depth-routed 4/8 policy is the comparator.

On a fresh ten-per-stratum audit, the oracle certified 258 of 396 contexts.
Conditional `Mass@10` improves 0.861 to 0.864, post-stratified mass improves
0.891126 to 0.892315, and zero overlap is unchanged. On the 105
bounded-complete certified deep cases:

| Policy | Mass@10 | Pair evaluations/context | Wins vs no-pair | Zero |
|---|---:|---:|---:|---:|
| Uniform 4 | 0.817 | 982 | 15 | 11 |
| Prior routed 4/8 | 0.818 | 1,641 | 16 | 11 |
| Active local 4->8 | 0.825 | 1,370 | 18 | 11 |
| Uniform 8 | 0.826 | 3,539 | 19 | 11 |

The linked 20,000-context validation reports paired wall/Rust/factor-map/
pricing deltas of -2.5%/-2.7%/-2.9%/+1.0%. The final two-cycle full-workload
comparison (81,844 contexts, eight threads) reports paired medians of -0.7%
wall, -0.4% aggregate Rust, -0.7% factor-map CPU, and +3.0% pricing CPU.
Local pair evaluations are 9.57M versus 7.97M for the depth router; the
promotion is therefore based on the measured quality gain at the current
end-to-end runtime budget, not lower aggregate pair work.

### Prior routed interacting-pair pricing

Full prepared workload (81,844 contexts, 328,197 steps, 1,110 zones), eight
threads. The two-cycle interleaved comparison is against the otherwise
identical single-variable pricing policy:

| Policy | Wall time | Aggregate Rust | Factor-map CPU | Pricing CPU |
|---|---:|---:|---:|---:|
| Active routed pair pricing | 41.102 s | 241.539 s | 181.781 s | 25.553 s |
| Prior single-variable pricing | 39.627 s | 226.614 s | 173.944 s | 19.413 s |

The active policy adds 3.7% wall time and 6.6% aggregate Rust time. On 106
exact-certified deep contexts, conditional `Mass@10` rises from 0.771 to
0.821; 15 contexts improve, none regress, and zero-overlap cases fall from 7
to 5. The router crosses four exact conditional columns per interacting
variable at depths 6–8 and eight at depth 9 or greater.

### Initial single-variable pricing promotion

Full prepared workload (81,844 contexts, 328,197 steps, 1,110 zones), eight
threads. This is a one-cycle interleaved A/B comparison of the active adaptive
pricing policy against the same search with pricing disabled:

| Policy | Wall time | Aggregate Rust | Factor-map CPU | Pricing CPU |
|---|---:|---:|---:|---:|
| Active adaptive pricing | 48.349 s | 286.675 s | 221.147 s | 23.702 s |
| Same search, pricing disabled | 43.805 s | 240.532 s | 201.999 s | 0 s |

Pricing adds 10.4% wall time in this full-workload run. Two 5,000-context
repeats measured -1.4% and +5.4% wall time, with the longer repeat adding
12.5% aggregate Rust time. The active router runs its first pass only at depth
6 or greater and runs a second pass only when the first contributes at least
three new surviving plans.

In an all-depth audit with ten samples per stratum, the oracle proved 258 of
396 contexts; 246 also had bounded results. Adaptive pricing raises
conditional `Mass@10` from 0.793 to 0.850 and lowers zero-overlap cases from 15
to 9. The post-stratified certified estimate rises from 0.847 to 0.872.
Depth-6 through depth-10 conditional gains are respectively
+0.109/+0.252/+0.079/+0.129/+0.118.

The oracle itself now uses active bounded plans as exact-rescored incumbents.
On the two-per-stratum pilot, solved contexts rose from 40 to 56 and
state-limited contexts fell from 30 to 14, with identical exact outputs where
both modes completed.

## Pre-pricing active baseline

Full prepared workload, eight threads; active defaults at the time with
`factor_map_max_depth=5` and `deep_continuation_state_limit=2`:

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
