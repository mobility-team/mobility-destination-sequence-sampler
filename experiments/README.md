# Experiments and benchmarks

This directory contains reproducible work that informs the destination-sampling
redesign but is not part of the stable package surface.

## Current direction

The active redesign is the bounded bidirectional top-K search:

- implementation: `rust/bidirectional.rs`;
- Python API: `DestinationPlanSearch.top_k()`;
- quality harness: `analysis/compare_bidirectional_top_k_grand_geneve.py`;
- runtime harness: `benchmarks/perf_bidirectional_grand_geneve.py`.

The latest active results are kept in [`../BENCHMARKS.md`](../BENCHMARKS.md).
The experiment record, including workload and decision rule, is
[`benchmarks/bidirectional-top-k.md`](benchmarks/bidirectional-top-k.md).

The next hypothesis is a small continuation-aware candidate source for early
forward layers. Candidate support, especially for the first destination, is
currently the main quality limitation. The search remains bounded; it is not
being replaced by an exact dynamic program.

## Supporting and reference paths

- `rust/ternary_reference.rs`: exact scorer and exact top-K oracle used for
  validation.
- `analysis/`: quality comparisons and exploratory analyses.
- `benchmarks/`: runtime and memory measurements.

The root Python package exposes only `DestinationPlanSearch`. Archived
experiments are recoverable from `research-archive-2026-07-21`.

## Historical experiment notes

The second-order recursion, hierarchical travel-kernel work, factor-tree
implementation, and heap-search benchmarks are retained as research history.
Their conclusions should be recorded in per-experiment notes before any code is
archived or removed.

See [`lessons-learned.md`](lessons-learned.md) for the consolidated decisions,
[`historical.md`](historical.md) for the experiment status table, and the
experiment-local records for the measurements behind each decision.
