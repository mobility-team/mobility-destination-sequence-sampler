# mobility-destination-sequence-sampler

Rust/Python bounded destination-plan top-K search with an exact validation
oracle. Mobility prepares inputs; this package owns destination choice.

## Public API

`DestinationPlanSearch` is the only exported class.

| Method | Purpose |
|---|---|
| `top_k()` | Bounded bidirectional search. Approximate, exact-score-ranked returned plans. |
| `exact_top_k()` | Small-context oracle. Proves the requested top-K or raises at `max_states`. |
| `exact_distribution()` | Guarded full exp(U) enumeration for a single small diagnostic context. |

`top_k()` uses unbinned exact factor-map proposals when each home-bounded tour
is at most depth 5, and bounded attractive, OD-near, and deterministic
exploration proposals for longer uninterrupted tours; it carries repeated
anchors and ranks stitched complete plans by
the full rigidity-aware utility. Deep contexts additionally use adaptive exact
single-variable and interacting-pair replacements on the best stitched plans,
with full reranking.
See [`DESIGN.md`](DESIGN.md) for schemas and invariants.

Only those paths are in the working tree. Particle, exhaustive-sampling, and
second-order research is preserved by the `research-archive-2026-07-21` Git
tag (for example, `git show research-archive-2026-07-21:<path>`). Start with
[`ACTIVE_SEARCH.md`](ACTIVE_SEARCH.md) for the active algorithm and tuning
contract.

## Development

```powershell
mamba env create -f environment.yml
mamba run -n mobility-destination-sequence-sampler python -m maturin develop --release
mamba run -n mobility-destination-sequence-sampler python -m pytest
```

Current quality/performance: [`BENCHMARKS.md`](BENCHMARKS.md). Experiment
routing and decisions: [`experiments/README.md`](experiments/README.md).
