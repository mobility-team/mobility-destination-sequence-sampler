# Kernel contract

Read this only for API, scoring, or search changes. Current hypotheses and
measurements are under `experiments/`.

## Boundary

- Python supplies Polars tables and orchestrates Mobility.
- Rust owns indexed inputs, feasibility, scoring, continuation/search, and all
  hot loops. No Polars in Rust search loops.
- Reuse one `DestinationPlanSearch` per OD/destination iteration.

Required inputs:

```text
OD:           origin, destination, cost, time
destination:  activity_id, destination, opportunity_capacity,
              country_value_coefficient, saturation_utility, shadow_price
steps:        context_id, layer, activity_id, anchor_id, fixed_destination,
              departure_time, next_departure_time, duration_per_person,
              value_of_time, mean_duration_per_person, min_activity_time
initial:      context_id, initial_zone
```

The context key includes every recursion-affecting step/parameter. Repeated
non-null `anchor_id`s share one destination; capacity applies once, while each
visit still receives its activity/travel utility.

## Active top-K

```text
DestinationPlanSearch.top_k()
  -> rust/api.rs -> search_bidirectional_top_k_all() -> search_context()
```

- `rust/bidirectional.rs`: context state and backward/guidance/forward/refresh/stitch passes.
- `rust/bidirectional/candidates.rs`: bounded proposal construction and cache.
- New passes use `SearchInputs` + `SearchScratch`, not long parameter lists.

Correctness invariant:

```text
prefix[i] owns factors through i - 1
suffix[i] owns factors from i + 1 onward
stitch[i] owns factors i and i + 1 exactly once
factor i is destination[i - 1] -> destination[i] -> destination[i + 1]
```

F-to-B refresh may add states but must not evict the reverse/home-oriented
frontier. `exact_top_k()` is the bounded small-context oracle; repeated-anchor
cases may hit `max_states`.
