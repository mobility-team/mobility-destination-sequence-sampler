# Experiments

Only bounded bidirectional top-K and the exact oracle are maintained.

| Need | Location |
|---|---|
| Current hypothesis/decision | `active-bidirectional-top-k.md` |
| Current measurements | `../BENCHMARKS.md`, `benchmarks/bidirectional-top-k.md` |
| Immutable experiment definitions | `manifests/` |
| Generated run artifacts | `runs/` |
| Manifest tooling | `experiment.py`, `harness.py` |
| Quality harness | `analysis/compare_bidirectional_top_k_grand_geneve.py` |
| Runtime harness | `benchmarks/perf_bidirectional_grand_geneve.py` |
| Plan-improvement router diagnostic | `analysis/evaluate_pricing_router.py` |
| Human-readable context explanation | `just explain-context <context_id>` |
| Compact agent diagnostic | `analysis/code_mode_probe.py` |
| Exact oracle | `../rust/oracle.rs` |
| Archived directions | `historical.md`, `retired-experiment-notes.md`, `lessons-learned.md` |

Archived source is recoverable from the `research-archive-2026-07-29` Git tag
(`git show research-archive-2026-07-29:<path>`); do not revive it without a
new, testable hypothesis. [`measurement-guide.md`](measurement-guide.md) defines
the active harness outputs and historical diagnostics.

Start from the table or `just --list`, not by browsing every analysis script.
An unlisted script is a specialised or historical diagnostic, not a supported
default workflow.

`code_mode_probe.py` is a read-only, local code-mode diagnostic: it performs
the bounded and exact calls for requested context IDs inside one Python process,
then writes only a compact JSON report to stdout. It is not an MCP server and
does not change production search behavior.
