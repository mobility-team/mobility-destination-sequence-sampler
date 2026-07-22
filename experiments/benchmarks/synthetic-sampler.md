# Archived synthetic sampler

The exact backward-forward sampling baseline was archived at
the `research-archive-2026-07-21` Git tag. Its useful lesson was to reuse prepared indexes
and group structurally identical contexts; it is not a basis for active top-K
parameter choices.

Recover its harness with:

```powershell
git show research-archive-2026-07-21:experiments/benchmarks/perf_synthetic_case.py
```
