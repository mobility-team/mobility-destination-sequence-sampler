# mobility-destination-sequence-sampler

Rust/Python bounded destination-plan top-K search with an exact validation
oracle. Mobility prepares inputs; this package owns destination choice.

## Public API

`DestinationPlanSearch` is the only exported class.

| Method | Purpose |
|---|---|
| `top_k()` | Bounded bidirectional search. Approximate, exact-score-ranked returned plans. |
| `exact_top_k()` | Small-context oracle. Proves the requested top-K or raises at `max_states`. |

`top_k()` uses unbinned exact factor-map proposals through depth 5 and bounded
attractive, OD-near, and deterministic exploration proposals for longer plans;
it carries repeated anchors and ranks stitched complete plans by
the full rigidity-aware utility. See [`DESIGN.md`](DESIGN.md) for schemas and
invariants.

Only those paths are in the working tree. Particle, exhaustive-sampling, and
second-order research is archived at `research-archive-2026-07-21`.

## Development

```powershell
mamba env create -f environment.yml
mamba run -n mobility-destination-sequence-sampler python -m maturin develop --release
mamba run -n mobility-destination-sequence-sampler python -m pytest
```

Current quality/performance: [`BENCHMARKS.md`](BENCHMARKS.md). Experiment
routing and decisions: [`experiments/README.md`](experiments/README.md).
