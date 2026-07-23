# Experiments

Only bounded bidirectional top-K is active.

| Need | Location |
|---|---|
| Current hypothesis/decision | `active-bidirectional-top-k.md` |
| Current measurements | `../BENCHMARKS.md`, `benchmarks/bidirectional-top-k.md` |
| Current global review and literature map | `quality-performance-review-2026-07-23.md` |
| Structural/data review and radical search boundary | `structural-search-review-2026-07-23.md` |
| Next experiment handoff | `admissible-block-search-experiment.md` |
| Reproducible structure/location diagnostic | `analysis/analyze_problem_structure.py` |
| Quality harness | `analysis/compare_bidirectional_top_k_grand_geneve.py` |
| Runtime harness | `benchmarks/perf_bidirectional_grand_geneve.py` |
| Compact agent diagnostic | `analysis/code_mode_probe.py` |
| Exact oracle | `../rust/oracle.rs` |
| Archived directions | `historical.md`, `lessons-learned.md` |

The next structural experiment is the diagnostic-first admissible block-bound
audit in `admissible-block-search-experiment.md`. Do not add its production
search path before the Phase A go/no-go gate.
Archived source is recoverable from the `research-archive-2026-07-21` Git tag
(`git show research-archive-2026-07-21:<path>`); do not revive it without a
new, testable hypothesis. [`measurement-guide.md`](measurement-guide.md)
defines the active harness outputs and historical diagnostics.

`code_mode_probe.py` is a read-only, local code-mode diagnostic: it performs
the bounded and exact calls for requested context IDs inside one Python process,
then writes only a compact JSON report to stdout. It is not an MCP server and
does not change production search behavior.
