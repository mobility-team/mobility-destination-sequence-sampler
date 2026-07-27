# Kernel contract

## Boundary

Python supplies prepared Polars tables; Rust owns indexes, feasibility,
scoring, and hot loops. Reuse one `DestinationPlanSearch` per OD/destination
iteration. The public methods are `top_k()` (bounded) and `exact_top_k()`
(proof oracle). `top_k()` needs at least two steps and a fixed final
destination; two-step contexts use an exact direct scan, while longer contexts
use the bounded bidirectional search. That destination need not equal
`initial_zone`.

```text
OD:          origin, destination, cost, time
destination: activity_id, destination, opportunity_capacity,
             country_value_coefficient, saturation_utility, shadow_price
steps:       context_id, layer, activity_id, anchor_id, fixed_destination,
             departure_time, next_departure_time, duration_per_person,
             value_of_time, mean_duration_per_person, min_activity_time,
             arrival_time, arrival_time_rigidity, departure_time_rigidity
initial:     context_id, initial_zone
```

Repeated non-null `anchor_id`s share one destination. Capacity applies once;
each visit still receives activity/travel utility.

## Active search

```text
api.rs -> search_top_k_all() -> search_context()
```

- `top_k/mod.rs`: shared search state and per-context orchestration.
- `top_k/candidates.rs`: heuristic and binned-surface proposals/cache;
  `top_k/mod.rs` also owns factor-map proposal composition.
- `top_k/pricing.rs`: depth-routed exact replacements from stitched complete
  paths, including bounded interacting-pair neighborhoods, followed by full
  shared-scorer reranking.
- New passes take `SearchInputs` + `SearchScratch`, not long argument lists.
- `oracle.rs`: exact top-K oracle; it proves or fails at
  `max_states`, never approximates.

## Scoring invariant

```text
prefix[i] owns factors through i - 1
suffix[i] owns factors from i + 1 onward
stitch[i] owns factors i and i + 1 exactly once
factor i = destination[i - 1] -> destination[i] -> destination[i + 1]
```

Forward-to-backward refresh may add activity-correct states but must not evict
the reverse/home-oriented frontier. Proposal policies, including factor maps,
change support only; every retained factor still uses the shared exact scorer.
Post-stitch pricing obeys the same rule. A repeated anchor is replaced as one
group, interacting groups are changed atomically, and every retained complete
plan is rescored before final ranking.
