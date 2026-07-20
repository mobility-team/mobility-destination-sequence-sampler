# Historical synthetic sampler benchmark

## Objective

Measure the retired exact backward-forward sampling baseline separately from
graph/index construction, and quantify the benefit of grouping identical
contexts before sampling multiple draws.

## Retained measurements

Release build on Windows, 2026-07-16.

| Zones | OD edges | Unique contexts | Layers | Draws | Index build | Sampling |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 24,576 | 200 | 6 | 3 | 0.004 s | 0.112 s |
| 1,110 | 499,500 | 1,000 | 6 | 3 | 0.085 s | 3.513 s |
| 1,110 | 499,500 | 5,000 | 6 | 3 | 0.078 s | 17.655 s |
| 1,110 | 1,071,150 | 1,000 | 6 | 3 | 0.206 s | 7.988 s |

For the largest graph, 1,000 grouped contexts with three draws took 7.988 s,
while 3,000 separate one-draw contexts took 29.265 s.

## Lesson

Reuse the prepared graph and destination index, deduplicate structurally
identical contexts, and request several draws together. This is a baseline
result for the historical sampler, not a performance claim for active top-K.
The implementation and harness were archived at `research-archive-2026-07-21`.

## Decision

Keep this harness as a regression/reference benchmark only. Do not use it to
choose active top-K parameters.

## Reproduce

```powershell
git show research-archive-2026-07-21:experiments/benchmarks/perf_synthetic_case.py
```
