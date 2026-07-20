# Current benchmarks

This page contains only the latest measurements for the active bounded
bidirectional top-K search. It is not an end-to-end Mobility benchmark. Older
exact-sampling and exploratory results live beside the experiment that
produced them.

Last refreshed: 2026-07-21, release build on Windows, cached Grand Geneve
iteration-5 inputs.

## Active bounded top-K search

### Runtime

`experiments.benchmarks.perf_bidirectional_grand_geneve` was run on 1,000
fixed-terminal contexts (4,767 activity-chain steps, 1,110 zones), with eight
threads, `frontier_width=32`, `proposal_limit_per_source=16`,
`continuation_state_limit=1`, `continuation_proposal_limit=1`,
`seam_refresh_per_prefix=1`, and `top_k=10`.

| Method | Wall time | Complete contexts | Infeasible contexts |
|---|---:|---:|---:|
| Historical particle baseline, 32 particles | 0.090 s | 744 | 256 |
| Bounded top-K with one F-to-B alternative per retained prefix | 0.135 s | 787 | 213 |

The bounded search was 1.49 times the particle baseline wall time for this
workload. It evaluated 357,150 forward-to-backward refresh proposals, added
494 boundary states, and spent 19% of aggregate Rust search time in the
refresh. Seam lookahead was audited and removed: it preserved the same
50-context oracle mass but raised wall time from 0.135 s to 0.258 s.

### Quality against the exact oracle

`experiments.analysis.compare_bidirectional_top_k_grand_geneve` was run with
`top_k=10`, exact support depth 100, `frontier_width=32`,
`proposal_limit_per_source=16`, `continuation_state_limit=1`,
`continuation_proposal_limit=1`, and one thread.
It proved 10 short real contexts; eight others were skipped because the oracle
could not establish a valid result within its contract or state budget.

| Metric across the 10 proven contexts | Result |
|---|---:|
| Mean exact-plan recall at 10 | 0.660 |
| Mean retained conditional exact top-10 probability mass | 0.6842 |
| Median retained conditional exact top-10 probability mass | 0.7278 |
| Mean top-10 mass efficiency | 0.8545 |
| Missing top-10 mass: absent from base proposal pool | 0.2182 |
| Missing top-10 mass: proposal supported but beam-lost | 0.0975 |

The table above is the earlier one-way-exchange baseline. The current refresh
sweep used 50 exact-proven contexts (56 explicit oracle skips):

| F-to-B alternatives per retained prefix | Mean retained conditional exact top-10 mass |
|---:|---:|
| `0` | 0.7599 |
| `1` (current default) | 0.7676 |
| `2` | 0.7762 |
| `4` | 0.7996 |

The refresh preserves the reverse/home candidates and adds only bounded
forward-proposed boundary alternatives. It is therefore the active quality
improvement; the 2 and 4 settings remain opt-in while their runtime trade-off
is assessed on a larger oracle sample.

## Experiment records

- [Active bounded top-K](experiments/benchmarks/bidirectional-top-k.md)
- [Historical synthetic sampler](experiments/benchmarks/synthetic-sampler.md)
- [Historical exact references](experiments/benchmarks/exact-reference.md)
- [Hierarchical travel-kernel exploration](experiments/analysis/hierarchical-kernel.md)
- [Second-order recursion exploration](experiments/analysis/second-order-recursion.md)
- [Consolidated decisions](experiments/lessons-learned.md)

Reproduce the current results with:

```powershell
mamba run -n mobility-destination-sequence-sampler python -m experiments.benchmarks.perf_bidirectional_grand_geneve --contexts 1000 --threads 8 --profile
mamba run -n mobility-destination-sequence-sampler python -m experiments.analysis.compare_seam_refresh --contexts 50 --seam-refresh-per-prefix 0 --seam-refresh-per-prefix 1 --seam-refresh-per-prefix 2 --seam-refresh-per-prefix 4
```
