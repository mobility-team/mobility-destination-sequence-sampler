# Active direction: bounded bidirectional top-K

`DestinationPlanSearch.top_k()` grows bounded forward and backward frontiers,
then exact-score stitches their plans. Proposals combine attractive, OD-near,
and deterministic exploration zones; repeated anchors and rigidity-aware
ternary factors are preserved.

Defaults: `frontier_width=32`, `proposal_limit_per_source=16`,
`continuation_state_limit=1`, `continuation_proposal_limit=1`,
`seam_refresh_per_prefix=1`, `top_k=10`.

The forward-to-backward refresh adds activity-correct forward proposals to the
reverse frontier without evicting home-oriented states. It is the current
quality improvement. The exact oracle measures retained conditional exact
top-K mass and separates missing proposal support from beam loss.

## Candidate experiments (2026-07-21)

`candidate_strategy="exact_local"` scans an activity domain against a known
backward successor and retains the best 34 exact local scores. On a two-per-
stratum Grand Geneve oracle audit it raised conditional `Mass@10` from 0.700
to 0.795, but made 1,000-context throughput 0.262 s versus 0.160 s and added
a bounded failure. It is the quality reference, not an active policy.

`projected_local` re-ranked the heuristic pool plus 256 cheap predecessor
zones of that successor. It failed the known five-step context 43094 and
reduced the short-cohort `Mass@10` to 0.550; rejected.

Next: a configurable binned surface. Precompute compact memberships for
activity potential, inbound cost/time pressure, and outbound time pressure;
rank bin upper bounds per rigidity query, expand several high-bound regions,
then exact-score their zones. Preserve region diversity and fall back to more
regions before widening a beam.
