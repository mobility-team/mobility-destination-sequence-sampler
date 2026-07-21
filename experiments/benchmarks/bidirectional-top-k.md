# Bounded bidirectional top-K

## Current configuration

`top_k()` on Grand Geneve iteration-5 inputs: `frontier_width=32`,
`proposal_limit_per_source=16`, `continuation_state_limit=1`,
`continuation_proposal_limit=1`, `seam_refresh_per_prefix=1`, `top_k=10`.

| Workload | Result |
|---|---:|
| 1,000 contexts, 4,767 steps, 1,110 zones, 8 threads | 0.135 s; 787 complete, 213 infeasible |
| 50 final-home, no-variable-anchor contexts, refresh 0/1/2/4 | mass 0.7599 / 0.7676 / 0.7762 / 0.7996 |

One refresh alternative is the default: it preserves reverse/home candidates
and adds bounded activity-correct proposals from retained forward prefixes.
Its 357,150 proposals added 494 boundary states and used 19% of aggregate
search time. Higher settings remain opt-in.

## Output-size validation

Five seeded 50-context cohorts with 3–4 layers, a final home, and
variable/repeated anchors included used `frontier_width=128` and a fixed exact
top-500 support. Mean Mass@500 was 0.084 / 0.126 / 0.196 / 0.254 for
K=10 / 20 / 50 / 100; Mass@K fell from 0.652 to 0.437. The width must cover K.

Seam lookahead was removed: identical 50-context oracle mass, 0.258 s versus
0.135 s on the 1,000-context run.

## Decision

Keep the balanced stitch and one refresh alternative. Improve early proposal
support before widening beams or candidate lists.

## Reproduce

```powershell
just benchmark-throughput
just compare-quality
just compare-k-sweep
just compare-k-sweep-seeds
just compare-refresh
```
