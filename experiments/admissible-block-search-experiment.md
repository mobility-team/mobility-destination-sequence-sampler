# Next experiment: admissible hierarchical zone-block search

## Handoff

Run this as a staged experiment. Phase A is diagnostic-only and is a hard
gate: do not add a production search path unless the block bounds are both
correct and selective on the cached exact cases.

Read, in order:

1. `AGENTS.md`, `ACTIVE_SEARCH.md`, and `DESIGN.md`;
2. `experiments/structural-search-review-2026-07-23.md`;
3. `rust/scoring.rs`, `rust/model.rs`, and the problem construction in
   `rust/oracle.rs`;
4. `rust/top_k/mod.rs` only after Phase A passes.

The hypothesis is:

> A deterministic hierarchy of OD-coherent zone blocks provides admissible
> and sufficiently selective upper bounds to explore the roughly 1,000
> values of each collapsed destination variable lazily. A best-first search
> over block boxes can recover substantially more exact top-10 mass than the
> current layer beams without paying for raw-domain exact search.

This is not the archived hierarchical kernel. The hierarchy may prune and
order work, but it must never approximate a returned plan's score. Every leaf
plan is evaluated by the shared exact scorer.

## Why this is next

The current evidence rules out the cheaper structural alternatives:

- exact width-zero decomposition made the full run 6.8% slower while fixing
  cases that were already exact;
- raw-zone exact best-first search was about 18x slower at induced width one
  and over 400x slower at width two;
- recombining the active search's complete output support raised cached
  Mass@10 only from 0.776 to 0.782;
- in the 73 zero-mass cases, the active proposal support contains only 0.079
  of exact mass.

The problem has few variables (mean 2.17, maximum 9) but large, overlapping
domains (759--1,110 zones). The next experiment must therefore change value
enumeration, not merely widen or recombine the existing beams.

## Exact bound to implement

Use the notation from `rust/scoring.rs`. At layer `i`, let `o -> z` be the
inbound edge and `z -> n` the outgoing edge. Define:

```text
r_i = clamp(arrival_i - departure_i, 0, 24)
b_i = departure_rigidity_i /
      (arrival_rigidity_i + departure_rigidity_i)

a_(i+1) = arrival_rigidity_(i+1) /
          (arrival_rigidity_(i+1) + departure_rigidity_(i+1))
```

Use `0.5` for either share when the corresponding rigidity sum is zero,
exactly as `adjusted_times()` does. When plan timings are updated,

```text
duration_i(o, z, n)
  = departure_(i+1) - arrival_i
    + a_(i+1) * r_(i+1) + b_i * r_i
    - b_i * time(o, z) - a_(i+1) * time(z, n).
```

For a destination `z`, define:

```text
g_i(d) = max(log(d / min_activity_time_i), 0)

coefficient_i(z) =
  country_value_coefficient(z) * value_of_time_i + shadow_price(z)
```

for the current `use_shadow_prices=true` workload. The general implementation
must use the existing saturation formula when shadow prices are disabled.
Let

```text
q_i(z) = coefficient_i(z) * mean_duration_per_person_i.
```

For origin, current, and next cells `O`, `C`, and `N`, compute separately for
each `z` in `C`:

```text
c_min(z, O) = min cost(o, z) over reachable o in O
t_in_min(z, O) = min time(o, z) over reachable o in O
t_out_min(z, N) = min time(z, n) over reachable n in N

d_max(z, O, N) =
  departure_(i+1) - arrival_i
  + a_(i+1) * r_(i+1) + b_i * r_i
  - b_i * t_in_min(z, O) - a_(i+1) * t_out_min(z, N)

U_i(z; O, N) =
  attraction_i(z)
  + logit_scale * (
      max(q_i(z), 0) * g_i(d_max(z, O, N))
      - c_min(z, O)
    )

U_i(O, C, N) = max U_i(z; O, N) over feasible z in C.
```

The independent cost and time extrema may come from different edges. That
makes the result looser but remains safe. If `d_max <= 0`, that `z` has no
feasible timed realization and is excluded. A missing inbound or outgoing
edge also excludes it. An empty maximum is negative infinity.

There are three scorer cases to preserve:

- With `update_plan_timings=false`, duration is the fixed
  `duration_per_person`; use the exact `q_i(z) * g_i(duration)` term rather
  than its positive-part relaxation.
- At the fixed terminal layer, there is no outgoing cell and duration is
  `MIN_ACTIVITY_DURATION_HOURS`; minimize only inbound cost.
- Attraction is the exact per-zone first-choice term. It is zero for fixed
  destinations and repeated visits to an anchor.

The positive part is applied to `q_i(z)`, not merely to the destination
coefficient, because the input validator currently permits a negative mean
duration. `logit_scale` is positive by API contract.

Repeated anchors require no special relaxation rule. Collapse them to one
variable, use the same cell wherever that variable occurs, and allow the
factor bound to relax correlations between its positions. This remains
admissible and its looseness must be measured separately.

For a complete block box `B`, containing one cell for every collapsed
variable:

```text
U(B) = sum_i U_i(cell(origin_i), cell(destination_i), cell(next_i)).
```

Fixed zones are singleton cells. Every exact zone assignment represented by
`B` has a total score at most `U(B)`.

## Phase A: prove and audit the bound

### A1. Build one deterministic hierarchy

Do not change the prepared input contract in this phase. Construct the
hierarchy from the existing OD matrix:

1. For a directed cost or time value, clamp negative values to zero, replace
   a missing edge with the observed p99, clip at p99, and divide by the
   positive median.
2. Define symmetric zone dissimilarity as the mean of normalized cost and
   time in both directions.
3. Recursively bisect a cell with deterministic two-medoid assignment. Start
   from the lowest external zone ID, choose the farthest zone, then choose the
   zone farthest from that seed as the other seed. Assign to the nearer seed;
   break ties by external zone ID.
4. Always split the cell with the largest diameter, breaking ties by its
   smallest zone ID. Stop at singleton leaves.

Audit cuts containing approximately 16, 32, 64, and 128 cells. Intersect
global cells with each activity domain and omit empty intersections.

Also construct a size-matched control hierarchy by sorting zones with
`splitmix64(zone_id ^ 0x9E3779B97F4A7C15)` and recursively halving the order.
The control distinguishes useful OD locality from mere domain subdivision.
Write the hierarchy fingerprint and parameters to the report; caches belong
under `experiments/.cache/block-search/` and are not source artifacts.

### A2. Test admissibility before measuring quality

Put the bound arithmetic beside the shared scorer in a private Rust module.
Do not reimplement scoring semantics only in Python.

Required tests:

1. Exhaustive synthetic cells of 3--6 zones, including missing edges,
   zero/nonzero rigidity sums, timing on/off, terminal factors, first and
   repeated anchor choices, positive and negative `q`, and shadow prices
   on/off. Enumerate all exact triples and assert
   `block_bound + 1e-12 >= exact_local_weight`.
2. A randomized property test with a fixed seed and at least 10,000 valid
   small factors. Print the complete fixture on the first violation.
3. A real-data audit of at least 100,000 exact factor assignments sampled
   from all depths and from both ordinary and repeated-anchor contexts.
4. For every cached exact top-10 plan, assert that the sum of its containing
   block-factor bounds is no smaller than its exact total score.

Any violation is an immediate stop. Fix the derivation or reject the
hypothesis; never add an epsilon large enough to hide a semantic mismatch.

### A3. Measure whether the safe bound is selective

Use the 573 current-fingerprint exact certificates. For every exact top-10
plan and variable factor, condition the neighboring positions on the cells
that contain their exact zones. When the candidate variable occupies more
than one position in the factor, replace all of those positions with the same
candidate cell; condition only the other variables. Rank every possible
candidate cell by the resulting factor upper bound.

Weight observations by the normalized exact top-10 plan probability and
report, for every cut and for OD/hash hierarchies:

- bound recall at ranks 1, 4, 8, and 16;
- the count of cells whose bound is at least the exact target factor score
  ("competitive cells"), p50/p90/p99;
- target-cell bound slack over its exact local score, p50/p90/p99;
- total zone count in the top 8 and top 16 cells, p50/p90/p99;
- the same metrics for current full-, partial-, and zero-mass classes;
- the same metrics by induced width, longest variable tour, and repeated
  anchor status.

Use exact-score ordering only to define the target. Do not feed the current
bounded search's proposals into this audit.

### Phase A go/no-go gate

Proceed only if one OD-coherent cut satisfies all of:

- zero synthetic, randomized, real-factor, and exact-plan bound violations;
- weighted bound recall@8 at least 0.90 and recall@16 at least 0.97;
- zero-mass-case recall@16 at least 0.90;
- p90 competitive cells no greater than 16;
- p90 zones contained in the top 16 cells no greater than 320;
- versus the size-matched hash control, either recall@8 improves by at least
  0.05 absolute or p90 competitive cells falls by at least 25%.

These are deliberately stronger than "some correlation." If every cut fails,
record the table in `experiments/historical.md`, remove temporary Rust
diagnostic exposure, and stop. Do not build the search.

## Phase B: bounded box best-first prototype

Only after Phase A passes, add a private experimental search module. Do not
change the default strategy or public API yet.

Represent a state as one hierarchy cell per collapsed destination variable.
The root state uses each variable's activity-domain root. Its priority is
`U(B)`.

Repeat:

1. Pop the state with greatest upper bound.
2. If every cell is a singleton, materialize the layer destinations, score
   the complete plan with `score_zones()`, deduplicate it, and add it to the
   exact leaf result heap.
3. Otherwise refine one variable cell into its two children and push the
   nonempty child boxes with finite bounds.
4. Once ten exact leaf plans exist, a proof is complete when the largest
   queued upper bound is no greater than the tenth exact score.

Initially choose the variable maximizing:

```text
log2(number of domain zones in its cell) * number of incident factors
```

with stable variable-ID tie breaking. Cache each local bound by factor ID and
the tuple of participating cell IDs. Preserve the factor-ownership invariant:
box bounds own no returned utility, and a leaf plan is scored once by the
shared exact scorer.

Use a configurable pop/refinement budget in the experiment harness. On budget
exhaustion, return the best exact leaf plans found and the remaining proof
gap. This is a bounded-search experiment, not a fallback to `exact_top_k()`.

Collect at least:

- boxes pushed, popped, and pruned;
- local-bound cache queries and hit rate;
- exact leaf plans scored and duplicate leaves;
- maximum agenda size;
- time in bound construction, agenda operations, and exact leaf scoring;
- proof completion and final `max_queued_bound - kth_exact_score`;
- result fingerprint and the existing Mass@10/top-1 metrics.

Do not implement an AND/OR cache in the first prototype. If the box search
meets the quality gate but repeated conditioned subproblems dominate its
profile, separator-keyed AND/OR caching is the next isolated optimization.
If the bounds themselves cause the explosion, AND/OR bookkeeping will not
rescue the hypothesis.

## Evaluation order and decision gates

Add reproducible commands equivalent to:

```text
just audit-block-bounds
just compare-block-search
just canary-quality
just compare-quality
just compare-k-sweep-seeds
just audit-global-quality
just compare-throughput
just benchmark-throughput
just check
```

Run them in that order and stop as soon as a hard gate fails.

On the 573 cached certificates, compare in one prepared process against the
current public defaults. The prototype is eligible for full validation only
if:

- mean Mass@10 is at least 0.90;
- exact top-1 hit rate is at least 0.93;
- zero-mass contexts fall from 73 to at most 20;
- no oracle-feasible context becomes infeasible;
- all returned plans and local weights reproduce `score_zones()`;
- median measured Rust time is below 1 ms/context and no more than 2x the
  interleaved baseline in aggregate.

For a promotion decision:

- the weighted global oracle audit must improve Mass@10 by at least 0.03
  absolute, with no depth stratum losing more than 0.02;
- the 81,844-context release wall time must be no more than 1.15x a freshly
  interleaved active-baseline run (the retained width-zero A/B measured
  28.749 seconds, making 33 seconds the reference threshold for that run);
- the A/B/A throughput fingerprint must be stable and the full `just check`
  suite must pass.

If cached Mass@10 reaches 0.90 but runtime exceeds 2x, keep the result only as
an oracle/diagnostic direction and do not change the default. If Mass@10 is
below 0.85 at the best reasonable budget, or if more than 50 cached contexts
remain at zero mass, reject without tuning secondary split policies.

Permit one budget sweep and one refinement-policy adjustment after the first
prototype. Beyond that, improvement smaller than 0.01 Mass@10 or 5% runtime
is diminishing return for this experiment.

## Deliverables

The next agent should leave:

- a report containing the Phase A tables, hierarchy fingerprint, and an
  explicit go/no-go decision;
- bound unit/property tests and a reproducible read-only audit harness;
- if Phase A passes, a private box-search prototype and its profile counters;
- the cached-certificate and release A/B results;
- one concise accepted/rejected entry in `experiments/historical.md`;
- no default change unless every promotion gate passes.

For an exploratory Rust change, run a release build and one focused check.
At each decision gate run the commands above that are proportional to the
decision. Audit rejected work before removing it; retain the result in the
report, not dormant production branches.

## Non-goals

- Do not restore the archived approximate hierarchical scorer.
- Do not substitute a block score for an exact leaf score.
- Do not add coordinates or change Python's prepared table contract in
  Phase A.
- Do not widen the existing beam and call it a block-search comparison.
- Do not add GPU, SIMD, batching, or separator caches before the bound and
  basic box search have been isolated.

The inference framing follows
[Peyrard et al.](https://arxiv.org/abs/1506.08544),
[Marinescu and Dechter](https://cdn.aaai.org/AAAI/2007/AAAI07-186.pdf),
[Dechter, Flerova, and Marinescu](https://doi.org/10.1609/aaai.v26i1.8405),
and the lazy extraction principle in
[Huang and Chiang](https://aclanthology.org/W05-1506/).
