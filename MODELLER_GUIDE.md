# Guide for transport modellers

This package chooses destination sequences for already prepared activity
plans. Start here if you understand destination-choice models better than
search algorithms.

## The one-minute mental model

Imagine a plan that starts at home and contains three destination positions:

| `layer` | activity | destination |
|---:|---|---|
| 0 | work | unknown |
| 1 | shopping | unknown |
| 2 | return home | fixed |

For each unknown position, the model may have hundreds or thousands of
possible zones. Trying every complete sequence would multiply those choices
together. The production search therefore:

1. proposes a manageable shortlist of plausible zones;
2. grows partial plans from the start and the end;
3. keeps only the strongest partial plans;
4. joins the two sides near the middle;
5. improves the best complete plans with exact one- and two-choice
   replacements;
6. fully scores and ranks every returned complete plan.

Steps 1 and 3 are bounded: a good sequence can be discarded. Steps 5 and 6
cannot repair every possible discard. This is why `top_k()` is fast but not a
proof of the global top K.

`exact_top_k()` explores enough of the state space to prove the result, or
stops explicitly at its state budget. It is a validation tool for small
samples, not a production fallback.

## Words used in this repository

| Repository word | Plain-language meaning |
|---|---|
| context | one independent activity/destination plan being solved |
| layer | a position in that destination sequence; not a neural-network layer |
| zone / destination | one candidate location for a sequence position |
| anchor | an equality constraint: repeated `anchor_id` values must use the same zone |
| proposal / candidate | a zone admitted to the bounded shortlist |
| frontier / beam | the partial plans still under consideration |
| factor | one local utility contribution involving the previous, current, and next zones |
| factor map | exact local utility values for many possible current zones, used to form a shortlist |
| guidance / continuation | information from the unbuilt right side used to rank a left-side choice |
| stitch / seam | the position where forward and backward partial plans are joined |
| pricing | historical public name for complete-plan neighbourhood improvement; it is not fares or tolls |
| shadow price | the actual destination-saturation utility input; unrelated to the search's `pricing_*` options |
| support | the set of complete plans the bounded search was capable of constructing |
| exact scoring | the utility of a particular retained plan is computed with the shared full scorer |
| exact search | the oracle has proved that no omitted plan outranks the returned top K |

The distinction between the last two rows matters most. `top_k()` uses exact
scoring but bounded search. An exactly scored answer is not necessarily an
exactly searched answer.

## Where utility is counted

The utility at sequence position `i` can depend on:

```text
destination[i - 1] -> destination[i] -> destination[i + 1]
```

It includes the relevant destination and travel terms, including the timing
rigidity inputs. During a bidirectional search, incomplete edge factors are
deliberately deferred:

```text
left partial plan owns factors before the join
right partial plan owns factors after the join
the join scores the two crossing factors exactly once
```

This ownership rule prevents both missing and double-counting utility. A fixed
home return is a known destination boundary, but it does not automatically
make the tours on either side statistically independent: timing and repeated
anchors may still cross it.

## Normal use

Most callers should not pass search-tuning options. The active defaults are
validated together and change only after quality/runtime experiments.

```python
from mobility_destination_sequence_sampler import DestinationPlanSearch

search = DestinationPlanSearch(
    od_costs=od_costs,
    destination_inputs=destination_inputs,
)

plans, report = search.top_k(
    steps=steps,
    initial_locations=initial_locations,
    logit_scale=logit_scale,
    update_plan_timings=True,
    use_shadow_prices=True,
    exploration_seed=42,
    top_k=10,
    skip_infeasible=True,
)
```

Reuse the `DestinationPlanSearch` object while the OD and destination tables
stay unchanged. Its constructor prepares indexes that should not be rebuilt
for every batch.

The output has one row per returned plan and sequence position. `draw_id=1`
is the highest-utility returned plan. `total_log_weight` ranks complete plans;
`local_log_weight` is the position-level contribution used for diagnosis.

## What people commonly misread

- `infeasible_contexts` is not an accuracy measure. It means the bounded
  search produced no feasible complete plan under the supplied inputs.
- Increasing `top_k` does not make the search more exact. The frontier must
  also be wide enough to retain that many useful alternatives.
- A wider beam cannot recover a destination that was never proposed.
- A larger proposal list can be slower without helping if the continuation
  ranking is wrong.
- Repeated anchors are one shared destination choice, not independent visits.
- `Mass@K` in the quality harness is normalized over the exact reference
  support stated by that harness. It is not automatically mass over the full
  feasible distribution.
- An oracle state-limit failure is "not proved", not a bounded-search hit or
  miss.
- `pricing_*` controls plan improvement work. Monetary cost is read from the
  OD table, while destination shadow prices are read from destination inputs.

## Debug one context

Start with:

```powershell
just explain-context 26
```

Replace `26` with the failing `context_id`. The command runs the active
bounded search and exact oracle in one prepared process, then shows exact
target plans and their active trace.

Read the trace from broadest to most specific:

1. `proposed=false`: the target zone never entered that step's candidate set.
   This is a proposal-support problem.
2. `proposed=true`, `retained=false`: the zone was considered but all states
   containing it were pruned.
3. `prefix_proposed=false`: the correct zone may have appeared, but never
   after the correct preceding sequence. This is stronger evidence than the
   zone-only flag.
4. `prefix_retained=false`: the coherent exact prefix was evaluated and then
   lost at the beam.
5. `guidance_proposed` / `guidance_retained`: show whether the right-to-left
   continuation channel saw and kept the target.
6. `exact_guidance_rank` and `exact_guidance_log_gap`: show whether the target
   narrowly missed the guidance width or was scored far below it.

Then classify the problem before editing:

| Observation | First place to inspect |
|---|---|
| input is missing or malformed | `rust/input.rs`, then the caller's Polars preparation |
| exact and bounded utilities disagree for the same plan | `rust/scoring.rs` and the ownership invariant in `DESIGN.md` |
| target destination never proposed | `rust/top_k/factor_maps.rs` or `candidates.rs` |
| coherent prefix proposed but pruned | `rust/top_k/forward.rs` and beam widths |
| right-side target missing | `rust/top_k/backward.rs` |
| sides exist but do not join | `rust/top_k/stitch.rs`, anchor compatibility, feasibility |
| a nearby complete plan could be repaired | `rust/top_k/improvement.rs` |
| exact oracle cannot finish | `rust/oracle.rs` or a smaller proof cohort |

Do not tune from one context. Reproduce the mechanism on a discovery cohort,
then use the workflow in `experiments/measurement-guide.md`.

## Change the algorithm safely

Make one conceptual change at a time:

1. State whether it targets proposal support, pruning, feasibility/scoring, or
   complete-plan improvement.
2. Add a focused toy test that would fail without the change.
3. Run a release build and the smallest relevant quality check.
4. If the mechanism survives, run a stratified exact-quality comparison and a
   counterbalanced runtime comparison.
5. Keep it only if the gain is material; record rejected ideas in
   `experiments/historical.md`.

Useful entry points:

| Intended change | Read first |
|---|---|
| change model utility or feasibility | `DESIGN.md`, `rust/scoring.rs` |
| change destination shortlists | `rust/top_k/factor_maps.rs`, `candidates.rs` |
| change partial-plan retention | `rust/top_k/forward.rs`, `backward.rs` |
| change how the two sides meet | `rust/top_k/stitch.rs` |
| change complete-plan repair | `rust/top_k/improvement.rs` |
| change exact proof search | `rust/oracle.rs` |
| measure an idea | `experiments/measurement-guide.md` |

`ACTIVE_SEARCH.md` is the advanced implementation guide. `DESIGN.md` is the
small contract that must remain true. `BENCHMARKS.md` records measured
decisions rather than teaching the algorithm.
