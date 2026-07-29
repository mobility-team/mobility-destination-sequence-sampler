# mobility-destination-sequence-sampler

Rust/Python bounded destination-plan top-K search with an exact validation
oracle. Mobility prepares inputs; this package owns destination choice.

Transport modeller? Start with
[`MODELLER_GUIDE.md`](MODELLER_GUIDE.md). It explains the algorithm with a
small activity-plan example, translates the search terminology, lists common
misreadings, and gives a one-context debugging workflow.

## Public API

`DestinationPlanSearch` is the only exported class.

| Method | Purpose |
|---|---|
| `top_k()` | Bounded bidirectional search. Approximate, exact-score-ranked returned plans. |
| `exact_top_k()` | Small-context oracle. Proves the requested top-K or raises at `max_states`. |

`top_k()` shortlists plausible destinations, grows partial plans from both
ends, joins them near the middle, improves the best complete plans, and ranks
them with the full rigidity-aware utility. Repeated anchors remain the same
destination throughout a plan.
See [`DESIGN.md`](DESIGN.md) for schemas and invariants.

Only those paths are in the working tree. Particle, exhaustive-sampling, and
second-order research is preserved by the `research-archive-2026-07-29` Git
tag (for example, `git show research-archive-2026-07-29:<path>`). Start with
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
