# Agent guide

## Scope

Working code is only bounded `DestinationPlanSearch.top_k()` and the exact
`exact_top_k()` oracle. Retired paths are at `research-archive-2026-07-21`.

## Commands

- `just install` — editable Python package.
- `just build-release` — release PyO3 extension.
- `just test` — Python tests (after a build).
- `just check` — format, Clippy, then tests.
- `just compare-quality` — 50-context bounded-versus-exact default comparison.
- `just compare-refresh` — 0/1/2/4 refresh quality trade-off.
- `just benchmark-throughput` — 1,000-context eight-thread runtime profile.

Use `rg` for files/text; use `ast-grep` for syntax-aware Rust/Python search or
mechanical refactors.

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

## Change rule

Preserve the factor-ownership invariant in `DESIGN.md`. For an exploratory
Rust change, release-build plus one focused check; at a decision point, run
the full suite and agreed oracle/runtime samples. Audit failed work before
removing it, then record the conclusion briefly.
