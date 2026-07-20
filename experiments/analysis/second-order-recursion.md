# Second-order recursion exploration

## Objective

Evaluate an exact rigidity-aware second-order recursion on aggregated and raw
Grand Geneve destination zones. This remains research/reference work, not the
bounded top-K implementation.

## Aggregation results

For 100 sampled contexts on eight threads:

| Clusters | Aggregation | Rust core | Contexts/s |
|---:|---:|---:|---:|
| 64 | 0.09 s | 0.75 s | 125.0 |
| 128 | 0.28 s | 3.27 s | 29.6 |
| 256 | 0.95 s | 48.26 s | 2.1 |

From 128 to 256 clusters, first-destination total variation was 0.152 at the
median and 0.444 at p95. The median absolute log-partition change was 0.195.

### Lesson

The hierarchy validates reduced-resolution model behaviour, but its quality
changes materially across useful resolutions and it is not production speed at
raw resolution.

## Feasibility, endpoint rigidity, and the exact corridor

Treating negative adjusted activity time as local spatial infeasibility made
active continuation sets sparse. On 1,000 contexts, the 256-cluster core took
7.29 s and retained 734 feasible contexts; the 128-cluster version took 0.56 s
and retained 675.

Endpoint rigidity splits a travel-time change between the origin departure and
destination arrival. A fixed wrapped-home shadow price accounts for the first
order overnight-home effect without changing the cubic recursion. With that
rule, the 1,000-context eight-thread runs were:

| Clusters | Rust core | Feasible | Infeasible |
|---:|---:|---:|---:|
| 64 | 0.09 s | 610 | 390 |
| 128 | 0.69 s | 685 | 315 |
| 256 | 7.53 s | 762 | 238 |

The exact bidirectional feasibility corridor then reduced the reachable pair
space sharply and sped the same workload by about 4.3–4.4x:

| Clusters | Backward only | Corridor | Speed-up |
|---:|---:|---:|---:|
| 128 | 0.98 s | 0.23 s | 4.3x |
| 256 | 9.52 s | 2.14 s | 4.4x |

At raw 1,110-zone resolution, the corridor still took 177.46 s for 1,000
contexts (0.177 s per context). The estimated supported full iteration was
about 4.09 hours on eight threads, excluding 546 two-anchor contexts.

### Lesson

Local feasibility and an exact forward/backward corridor are correct and
valuable optimizations, but a narrow corridor still contains too many raw-zone
transitions. They do not change the decision to keep this path out of the
active bounded redesign.

## Context structure

The cached input had 81,844 contexts: 38,696 with no variable anchor type,
42,602 with one, and 546 with two. Chain length alone did not predict cost:
infeasible chains terminate cheaply while feasible long chains retain large
corridors. Two-anchor tours require a true multi-anchor treatment because the
intermediate home boundary depends on both neighbouring trips.

## Decision

Keep the solver for exact reduced-resolution research and validation. Do not
extend it as a raw-zone production replacement; reusable work across similar
contexts would need evidence before reopening the direction.

## Reproduce

```powershell
mamba run -n mobility-destination-sequence-sampler python -m experiments.analysis.explore_aggregated_backward_forward --clusters 64 128 256 --n-contexts 1000 --wrapped-home-shadow-price 2.0 --n-threads 8
mamba run -n mobility-destination-sequence-sampler python -m experiments.analysis.analyze_second_order_contexts --contexts-per-cell 5 --n-threads 8
```
