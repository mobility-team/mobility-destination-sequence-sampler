# Historical exact-reference benchmarks

## Objective

Retain the measurements and decisions for exact sampling, factor-tree, and
exact heap-search work used as validation support. None is the active bounded
destination-plan search.

## Complete-plan factor-tree baseline

The cached iteration-5 benchmark contained 81,844 contexts and 328,197 steps.
Three-draw sampling produced 921,444 output rows for 77,232 complete contexts.

| Phase | Result |
|---|---:|
| Polars preparation | 0.562 s |
| Rust index construction | 0.227 s |
| Rust sampling | 55.836 s |
| Peak total process memory | 1,317.3 MiB |

Of the factor-graph shapes, 57.1% had no variable destination, 41.8% were
tree-shaped, and 1.1% were cyclic. Cyclic repeated-anchor contexts were
reported and skipped, never sampled as independent anchors.

### Lesson and decision

Streaming pair factors reduced the original 136.214-second full pass to
55.836 seconds, but exact all-zone sampling is still historical reference work.
Keep it for validation, not production destination generation.

## Rejected pruning and sparse traversal

On 2,000 complete contexts, scalar branch-and-bound pruning took 5.035–5.211 s
against 2.197 s for the streaming scan. An incoming-OD continuation traversal
reduced estimated edge visits but still took 1.67–1.73 s against 1.36–1.39 s
for the outgoing-only scan.

### Lesson and decision

Contiguous CSR scans make infeasibility checks cheap. Sorting, random lookup,
and scattered writes cost more than the apparently avoided edges. Both variants
were removed; optimize finite contribution evaluation instead.

## Exact home splitting and heap top-K oracle

Exact home-to-home splitting reduced a dense two-tour synthetic example from
an unsplit 4.096-billion assignment lattice to 128,000 enumerated tour
configurations at 40 destinations per variable, with a 141.1 ms median.

The exact heap-search refinements produced:

| Grand Geneve case | Initial heap | Optimized heap |
|---|---:|---:|
| 10 contexts, 2–3 variable layers | 44.486 s | 0.349 s |
| One four-variable, 1.26-trillion-assignment context | >25 s | 0.061 s |
| Four independent home tours, context 38002 | 113.421 s | 0.039 s |
| Mixed 100-context sample | 120.674 s | 1.564 s |

The mixed 100-context heap benchmark scaled from 1.755 s on one thread to
0.415 s on eight threads. Its state limit remains essential for rare loose-bound
or repeated-anchor shapes.

### Lesson and decision

Exact splitting, relaxed bounds, and lazy sibling expansion make a useful
oracle. The oracle must either prove its result or fail explicitly; it is not
a fallback for the bounded active search.

## Exact profiling

On 2,000 representative contexts with three draws, backward messages consumed
28.691 of 30.729 aggregate CPU seconds (93.4%). The corresponding wall time was
1.526 s, versus 1.477 s without profiling.

### Lesson and decision

Profiling confirms that exact pair-factor backward messages are the old
kernel's bottleneck. Retain profiling as an explicit benchmark cost and do not
charge it to normal runs.

## Reproduce

```powershell
The historical enumeration and heap benchmark harnesses were archived at
`research-archive-2026-07-21`. The active exact oracle is exercised through
`DestinationPlanSearch.exact_top_k()` by the current quality analyses.
```
