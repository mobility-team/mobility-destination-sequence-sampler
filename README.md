# mobility-destination-sequence-sampler

Rust/Python destination-sequence sampling kernels for Mobility.

## Scope

This package owns the destination-choice computation over compact, prepared
Mobility inputs:

- sparse ordered OD costs and travel times;
- activity and destination utility inputs;
- context-specific activity chains;
- exact reference scoring and bounded candidate generation;
- parallel processing of independent contexts.

Mobility remains responsible for preparing activity parameters, destination
saturation, mode-inclusive OD inputs, and demand-unit/context mappings.

The active redesign is the bounded bidirectional top-K search. Research status,
benchmarks, and historical decisions are documented in
[`experiments/`](experiments/README.md).

## Active API

### Bounded bidirectional top-K

`DestinationPlanSearch.top_k()` grows bounded forward and backward beams from
the initial and terminal homes. It proposes attractive, OD-near, and
deterministic exploration destinations, scores rigidity-aware factors, carries
repeated-anchor assignments, adds bounded forward-to-backward stitch
alternatives, and ranks stitched plans by exact complete-plan
utility.

It is intentionally bounded and approximate. The current next direction is a
small continuation-aware proposal for early forward layers, especially the
first destination.

### Historical experiments

Particle sampling, exhaustive sampling, and the second-order solver remain as
compiled reference code for the scripts in `experiments/`. They are not part
of the root package API.

### Exact oracle

`DestinationPlanSearch.exact_top_k()` is the validation oracle. It either
proves the requested result or raises when `max_states` is exceeded; it never
silently returns an approximate result.

## Python API

```python
from mobility_destination_sequence_sampler import DestinationPlanSearch

search = DestinationPlanSearch(
    od_costs=od_costs,
    destination_inputs=destination_inputs,
)

plans, report = search.top_k(
    steps=steps,
    initial_locations=initial_locations,
    logit_scale=1.0,
    update_plan_timings=True,
    use_shadow_prices=True,
    exploration_seed=42,
    frontier_width=32,
    proposal_limit_per_source=16,
    stitch_bias=0,
    continuation_state_limit=1,
    continuation_proposal_limit=1,
    seam_refresh_per_prefix=1,
    top_k=10,
)
```

The active report uses search terminology: proposals evaluated, complete-plan
candidates, stitched pairs, and optional phase timings. Historical reports and
schemas remain available only through the internal experimental class.

## Inputs

`steps` contains one row per context and activity-chain layer. It includes:

```text
context_id, layer, activity_id, anchor_id, fixed_destination,
departure_time, next_departure_time, duration_per_person,
value_of_time, mean_duration_per_person, min_activity_time
```

The rigidity-aware reference and bounded methods additionally use:

```text
arrival_time, arrival_time_rigidity, departure_time_rigidity
```

`initial_locations` contains `context_id` and `initial_zone`.

`od_costs` contains reachable ordered pairs:

```text
origin, destination, cost, time
```

`destination_inputs` contains one row per activity and available destination:

```text
activity_id, destination, opportunity_capacity,
country_value_coefficient, saturation_utility, shadow_price
```

All identifiers should be compact integer values before entering the kernel.
Repeated non-null `anchor_id` values share one sampled destination.

## Development

Create the environment:

```powershell
mamba env create -f environment.yml
```

Build the extension:

```powershell
mamba run -n mobility-destination-sequence-sampler python -m maturin develop --release
```

Run regression tests:

```powershell
mamba run -n mobility-destination-sequence-sampler python -m pytest
```

Run the active quality comparison:

```powershell
mamba run -n mobility-destination-sequence-sampler python -m experiments.analysis.compare_bidirectional_top_k_grand_geneve
```

Run the active runtime benchmark:

```powershell
mamba run -n mobility-destination-sequence-sampler python -m experiments.benchmarks.perf_bidirectional_grand_geneve --help
```

See [`experiments/README.md`](experiments/README.md) for experiment status and
reproduction paths, [`DESIGN.md`](DESIGN.md) for architecture, and
[`BENCHMARKS.md`](BENCHMARKS.md) for the latest active measurements.
