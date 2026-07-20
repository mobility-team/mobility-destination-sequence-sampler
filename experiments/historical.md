# Experiment index

The active direction and retained reference directions are listed here. Only
bounded bidirectional top-K is the active destination-sampling redesign.

| Experiment | Status | Decision | Record |
|---|---|---|---|
| Bounded bidirectional top-K | Active | Continue with continuation-aware proposals. | [record](benchmarks/bidirectional-top-k.md) |
| Particle candidates | Baseline | Retain as a sequential bounded baseline/fallback. | [synthetic sampler](benchmarks/synthetic-sampler.md) |
| Exact ternary reference | Oracle | Keep for validation and small-domain checks. | [exact references](benchmarks/exact-reference.md) |
| Second-order aggregation | Archived | Not production-speed at raw resolution; source is tagged. | [record](analysis/second-order-recursion.md) |
| Factor-tree sampler | Historical | Retain for comparison; not the active path. | [exact references](benchmarks/exact-reference.md) |
| Hierarchical travel kernel | Archived | Useful research result, not a standalone runtime solution; source is tagged. | [record](analysis/hierarchical-kernel.md) |
| Exact heap top-K | Oracle support | Keep as an exact quality oracle, not as the bounded sampler. | [exact references](benchmarks/exact-reference.md) |

| Area | Implementation | Current interpretation |
|---|---|---|
| Particle candidates | `rust/particle.rs` | Useful sequential bounded baseline/fallback. |
| Exact ternary oracle | `rust/ternary_reference.rs` | Validation authority for small cases and top-K quality. |
| Second-order aggregation | `research-archive-2026-07-21` | Exact reduced-resolution research path; not production-speed at raw resolution. |
| Factor-tree sampler | `rust/factor_tree.rs` | Earlier exact structural approach retained for comparison. |
| Hierarchical kernel | `research-archive-2026-07-21` | Spatial compression hypothesis; not a standalone route to target runtime. |
| Heap top-K reference | `search_ternary_top_k()` and related benchmarks | Exact oracle/search reference, not the bounded production direction. |

Every record states its objective, latest retained measurements, lesson, and
decision. `../BENCHMARKS.md` intentionally contains only current active
bounded-search measurements.
