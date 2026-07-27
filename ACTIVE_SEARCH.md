# Active search guide

Read this before changing bounded search. It is the small working model for
the active kernel; `DESIGN.md` remains the contract and `BENCHMARKS.md` remains
the measured decision record.

## What is active

`DestinationPlanSearch.top_k()` is a bounded search: it may miss a globally
best plan, but every plan it returns is ranked with the shared exact scorer.
`exact_top_k()` is the small-workload authority: it proves the requested top-K
or raises when its state budget is exhausted. It is never a production
fallback.

The default policy is `adaptive_factor_map`. It uses the symmetric factor-map
channel when a variable run has at least two adjacent unknowns or an anchor
repeats. When fixed destinations isolate every variable and no anchor repeats,
it skips the partial reverse channel and uses ordinary exact factor maps.
Factor maps apply when every home-bounded tour is at most
`factor_map_max_depth=5`; a longer uninterrupted tour uses the bounded
heuristic proposal pool with two exact continuation states. Fixed home returns
remain part of the full scorer and do not make tours independently solvable. A
two-step context is a direct exact scan. After stitching, contexts with at
least six layers price exact single-variable replacements from the best
complete plans. At most two rounds run; the second requires at least three new
surviving plans from the first. Each round also crosses exact conditional
columns for interacting variable pairs. Every pair probes four candidates per
variable, then expands to eight only when the best probe candidate improves
the current working Kth score by more than 0.2.

## One-context flow

```text
Python tables
  -> api.rs parses / validates
  -> build_scoring_problem
  -> backward exact frontier
  -> exact continuation guidance
  -> optional partial symmetric guidance
  -> forward beam and proposal support
  -> optional forward-to-backward seam refresh
  -> exact boundary stitch and deduplicate
  -> locally routed exact path pricing
  -> materialize
```

The implementation entry is `search_top_k_all()` and the per-context
orchestrator is `search_context()`. Search is parallel only across contexts;
all caches and mutable search state are per context.

`rust/top_k/mod.rs` owns shared private state; its explicit child modules own
one search phase each (`factor_maps`, `backward`, `forward`, `refresh`, and
`stitch`); `pricing` owns the post-stitch completed-path pass. Keep cross-phase
interfaces `pub(super)` and narrow: a phase must not reach into another phase's
cache implementation directly.

## State and ownership

`PrefixNode` stores a destination, its parent, known anchor assignments, and
the exact utility it owns. `SuffixNode` stores the symmetric right-hand form.
An anchor assignment is a compact vector keyed by the non-null `anchor_id`;
equal anchors must resolve to the same destination.

The ownership rule is non-negotiable:

```text
prefix[i] owns factors through i - 1
suffix[i] owns factors from i + 1 onward
stitch[i] scores factors i and i + 1 exactly once
```

Proposal scores, factor maps, continuation messages, and partial reverse
messages may decide which states survive. They never replace the exact local
scorer for a retained factor. The primary backward frontier must survive seam
refresh; refresh only adds boundary states.

Factor-map policies use destination-resolution maps in both the forward and
reverse proposal passes. A fixed home destination is therefore a map boundary,
not a legacy-candidate fallback; its crossing local factor remains owned and
scored exactly by the relevant beam.

Post-stitch pricing reuses ranked exact factor maps for unanchored layers. A
repeated anchor is replaced as one group and all affected factors are scored
with the shared scorer. Retained columns are then fully rescored, so the pass
changes support but not factor ownership or ranking semantics.

Pair pricing considers only groups whose affected factor windows overlap. It
scores the union of affected factors, keeps at most the working top-K from each
joint neighborhood, and fully rescores those survivors. This can cross a
two-variable utility valley without constructing an all-domain Cartesian
product. The 4x4 probe uses runtime scores only; exact certificates are
offline experiment labels and never router inputs.

## Read only what the task needs

| Task | First files | Usually avoid |
|---|---|---|
| Input/API/schema or report change | `rust/api.rs`, `rust/input.rs`, `rust/output.rs`, `_core.pyi` | search passes |
| Utility/feasibility issue | `DESIGN.md`, `rust/scoring.rs` | candidate policies |
| Proposal-support experiment | `rust/top_k/factor_maps.rs`, `rust/top_k/forward.rs`, `rust/top_k/candidates.rs` | oracle internals |
| Reverse/symmetric guidance | `rust/top_k/backward.rs` | forward ranking details |
| Stitch or anchor invariant | `rust/top_k/stitch.rs`, `DESIGN.md` | factor-map construction |
| Complete-path pricing/routing | `rust/top_k/pricing.rs`, `rust/top_k/factor_maps.rs`, `rust/top_k/stitch.rs` | oracle internals |
| Exactness/oracle issue | `rust/oracle.rs`, `experiments/benchmarks/exact-reference.md` | bounded passes |
| Quality or throughput measurement | `experiments/measurement-guide.md`, then the named harness | retired-tag source |

## Configuration contract

The public method deliberately exposes tuning knobs for experiments. Defaults
are the active production boundary, not a promise that every combination is
meaningful. Pass only the knobs relevant to the selected strategy.

| Setting | Default | Applies to | Effect / constraint |
|---|---:|---|---|
| `frontier_width` | 40 | all | retained states on the main beams |
| `proposal_limit_per_source` | 16 | all | proposal support per retained source |
| `candidate_strategy` | `adaptive_factor_map` | all | `heuristic`, `surface`, `factor_map`, `symmetric_factor_map`, or active structural router |
| `factor_map_max_depth` | 5 | factor-map policies | longest home-bounded tour allowed before falling back to heuristic support |
| `symmetric_message_limit` | 4 | symmetric only | partial reverse messages; zero disables that channel |
| `symmetric_state_limit` | 4 | symmetric only | retained partial reverse states away from the seam |
| `symmetric_forward_proposal_limit` | 20 | symmetric only | total compact partial-message proposals handed forward |
| `surface_bins` | 2 | `surface` only | binned comparator resolution (2 or 4) |
| `continuation_state_limit` | 1 | all | exact reverse guidance states consulted forward |
| `deep_continuation_state_limit` | 2 | contexts deeper than `factor_map_max_depth` | wider exact reverse guidance for heuristic-support depths |
| `continuation_log_gap` | 0.0 | all | additionally retain exact guidance states within this log-utility band; zero preserves fixed-width behavior |
| `continuation_proposal_limit` | 1 | all | reverse-projection proposals per guidance state |
| `seam_refresh_per_prefix` | 1 | all | extra suffix states from retained prefixes; never replaces reverse states |
| `stitch_bias` | 1 | contexts with 3+ steps | shifts the balanced stitch layer |
| `pricing_passes` | 2 | contexts with at least `pricing_min_layers` | maximum completed-path pricing rounds; zero disables |
| `pricing_seed_limit` | 10 | pricing | best complete plans used as pricing seeds |
| `pricing_column_limit` | 4 | pricing | exact replacement columns retained per variable/group and seed |
| `pricing_pair_candidate_limit` | 4 | pricing | exact conditional columns crossed by the local pair probe |
| `pricing_pair_deep_candidate_limit` | 8 | pricing | wider interacting-pair budget used after local escalation |
| `pricing_pair_deep_min_layers` | 0 | pricing | zero enables local probe-and-expand; values >=2 retain the depth-routed comparator |
| `pricing_next_pass_min_new` | 3 | pricing | minimum first-round surviving additions required for another round |
| `pricing_min_layers` | 6 | pricing | route pricing away from short contexts |
| `exploration_seed` | required | all | deterministic exploration tie/support choices |
| `top_k` | 10 | all | returned distinct plans; positive |

`surface`, `factor_map`, `symmetric_factor_map`, and `heuristic` are retained comparators. Do not
change active defaults based on a single context; use the quality harness and
record the decision in the active experiment note.

The Python experiment defaults live in `experiments/top_k_config.py`; keep
this table and the PyO3 defaults in `rust/api.rs` synchronized when changing
the active boundary.


The returned output has one row per `(context_id, draw_id, layer)`: `origin`,
`destination`, `local_log_weight`, and the plan-level `total_log_weight`.
Draws are descending by total utility. With `collect_profile=True`, the
bounded report also exposes proposal counts and per-pass nanosecond timings;
pair probe/expansion counts are included. Supplying an active trace also emits
per-pair probe signals for offline router analysis; their key names are
declared in `_core.pyi`.

`exact_top_k()` initializes branch-and-bound with the active bounded result by
default. This changes pruning only, never the proof result; pass
`use_bounded_incumbent=False` for a cold-oracle comparison.

## Safe change loop

1. State which invariant or measurable hypothesis the change targets.
2. Keep factor ownership in `DESIGN.md` true and add/adjust the mapped test.
3. For an exploratory Rust change, use `just build-release` and one focused
   check. At a decision point run `just check` plus the agreed quality/runtime
   samples.
4. Preserve a rejected result as a concise entry in `experiments/historical.md`;
   older source is available with `git show research-archive-2026-07-21:<path>`.
