# Agent operating guide

## Commands

- Install: `mamba run -n mobility-destination-sequence-sampler python -m pip install -e .`
- Release extension: `mamba run -n mobility-destination-sequence-sampler python -m maturin develop --release`
- Tests: `mamba run -n mobility-destination-sequence-sampler python -m pytest`
- Experiments: run from the repository root as
  `mamba run -n mobility-destination-sequence-sampler python -m experiments.analysis.<name>`.

## Boundaries

- Python prepares columnar inputs and owns Mobility orchestration.
- Rust owns compact graph/index structures, feasibility, scoring, continuation
  values, and bounded search. Do not introduce Polars into Rust hot loops.
- Preserve the factor-ownership invariant in `DESIGN.md` when changing search.

## Routing: read only what the task needs

- Contract, schemas, kernel map: `DESIGN.md`.
- Current top-K hypothesis and decision: `experiments/active-bidirectional-top-k.md`.
- Retained measurements: `experiments/benchmarks/bidirectional-top-k.md`.
- Historical/rejected approaches: `experiments/historical.md` and
  `experiments/lessons-learned.md`.
- API conversion and Python reports: `rust/api.rs`.
- Active search orchestration: `rust/bidirectional.rs`.
- Candidate proposals/cache: `rust/bidirectional/candidates.rs`.
- Exact validation oracle: `rust/ternary_reference.rs`.

## Iteration policy

For an exploratory Rust change: release-build and run one focused test or
analysis command. At a decision point: run the full suite, agreed oracle and
  runtime samples, then update the active experiment record. Do not remove a
  failed experiment until it has been audited.
