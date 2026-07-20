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

Current limitation: good plans can be absent before beam ranking because the
first forward proposal pool lacks a destination that is jointly good for its
incoming and later legs. Next: test a small continuation-aware early proposal
source; retain it only if oracle mass rises at acceptable cost.
