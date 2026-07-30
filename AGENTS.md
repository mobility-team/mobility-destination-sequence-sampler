# Agent guide

## Scope

Working code is only bounded `DestinationPlanSearch.top_k()` and the exact
`exact_top_k()` oracle. Retired paths are preserved by the Git tag
`research-archive-2026-07-29`; inspect one with
`git show research-archive-2026-07-29:<path>`. Do not restore archived source
without a new, testable hypothesis.

## Commands

- `just install` — editable Python package.
- `just build-release` — benchmarkable PyO3 extension.
- `just build-fast` — quality-iteration build; never benchmark it.
- `just test` — Python tests (after a build).
- `just check` — format, Clippy, release build, then tests.
- Fast screens: `just explore-quality NAME=VALUE`, then
  `just explore-throughput NAME=VALUE`.
- Decision workflow and metrics: `experiments/measurement-guide.md`.

Run `just --list` for the current command catalog. Use `rg` for files/text and
`ast-grep` for syntax-aware Rust/Python search or mechanical refactors.

## Ownership and routing

- Python prepares Polars inputs and orchestrates Mobility. Rust owns compact
  indexes, feasibility, scoring, and search; no Polars in Rust hot loops.
- Contract/invariants: `DESIGN.md`.
- Active search: `rust/top_k/mod.rs`; proposals/cache:
  `rust/top_k/candidates.rs`.
- Shared scoring: `rust/scoring.rs`; exact oracle: `rust/oracle.rs`; Python
  boundary: `rust/api.rs`.
- Current decision/measurements: `experiments/active-bidirectional-top-k.md`,
  `BENCHMARKS.md`. Archive decisions: `experiments/historical.md`.
- Working-model and option guide: `ACTIVE_SEARCH.md`. Read it before opening
  `rust/top_k/mod.rs`; it maps tasks to the smallest relevant source files.

## Change rule

Preserve the factor-ownership invariant in `DESIGN.md`. For an exploratory
Rust change, release-build plus one focused check; at a decision point, run
the full suite and agreed oracle/runtime samples. Audit failed work before
removing it, then record the conclusion briefly.
