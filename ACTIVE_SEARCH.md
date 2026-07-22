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

The default policy is `symmetric_factor_map`. It uses exact, unbinned local
factor maps through `factor_map_max_depth=5`; longer contexts use the bounded
heuristic proposal pool. A two-step context is a direct exact scan.

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
  -> exact boundary stitch, deduplicate, materialize
```

The implementation entry is `search_top_k_all()` and the per-context
orchestrator is `search_context()`. Search is parallel only across contexts;
all caches and mutable search state are per context.

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

## Read only what the task needs

| Task | First files | Usually avoid |
|---|---|---|
| Input/API/schema or report change | `rust/api.rs`, `rust/input.rs`, `rust/output.rs`, `_core.pyi` | search passes |
| Utility/feasibility issue | `DESIGN.md`, `rust/scoring.rs` | candidate policies |
| Proposal-support experiment | `rust/top_k/factor_maps.rs`, `rust/top_k/forward.rs`, `rust/top_k/candidates.rs` | oracle internals |
| Reverse/symmetric guidance | `rust/top_k/backward.rs` | forward ranking details |
| Stitch or anchor invariant | `rust/top_k/stitch.rs`, `DESIGN.md` | factor-map construction |
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
| `candidate_strategy` | `symmetric_factor_map` | all | `heuristic`, `surface`, `factor_map`, or active symmetric policy |
| `factor_map_max_depth` | 5 | factor-map policies | deeper contexts fall back to heuristic support |
| `symmetric_message_limit` | 4 | symmetric only | partial reverse messages; zero disables that channel |
| `symmetric_state_limit` | 4 | symmetric only | retained partial reverse states away from the seam |
| `symmetric_forward_proposal_limit` | 8 | symmetric only | total compact partial-message proposals handed forward |
| `surface_bins` | 2 | `surface` only | binned comparator resolution (2 or 4) |
| `continuation_state_limit` | 1 | all | exact reverse guidance states consulted forward |
| `continuation_proposal_limit` | 1 | all | reverse-projection proposals per guidance state |
| `seam_refresh_per_prefix` | 1 | all | extra suffix states from retained prefixes; never replaces reverse states |
| `stitch_bias` | 1 | contexts with 3+ steps | shifts the balanced stitch layer |
| `exploration_seed` | required | all | deterministic exploration tie/support choices |
| `top_k` | 10 | all | returned distinct plans; positive |

`surface`, `factor_map`, and `heuristic` are retained comparators. Do not
change active defaults based on a single context; use the quality harness and
record the decision in the active experiment note.


The returned output has one row per `(context_id, draw_id, layer)`: `origin`,
`destination`, `local_log_weight`, and the plan-level `total_log_weight`.
Draws are descending by total utility. With `collect_profile=True`, the
bounded report also exposes proposal counts and per-pass nanosecond timings;
their key names are declared in `_core.pyi`.

## Safe change loop

1. State which invariant or measurable hypothesis the change targets.
2. Keep factor ownership in `DESIGN.md` true and add/adjust the mapped test.
3. For an exploratory Rust change, use `just build-release` and one focused
   check. At a decision point run `just check` plus the agreed quality/runtime
   samples.
4. Preserve a rejected result as a concise entry in `experiments/historical.md`;
   older source is available with `git show research-archive-2026-07-21:<path>`.
