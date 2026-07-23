# Active: symmetric unbinned factor-map top-K

`top_k()` defaults to `candidate_strategy="symmetric_factor_map"`,
`symmetric_message_limit=4`, `symmetric_state_limit=4`,
`symmetric_forward_proposal_limit=20`, and `stitch_bias=1`.
Depth 2 is a direct scan. Factor maps apply while every home-bounded tour is at
most depth 5; a longer uninterrupted tour uses the heuristic pool with two
exact continuation states. A fixed home return is a map boundary, but factors
crossing it remain exactly scored and cross-home anchors remain in the state.

Both primary backward support and reverse guidance use destination-resolution
factor maps under the factor-map policies. An independent four-state reverse
channel combines the exact known right factor with every locally complete known
prefix factor, falling back to endpoint/attraction terms. Repeated-anchor
proposals are handed forward as compact destination assignments. Forward search
preserves its primary beam, unions twenty partial candidates, and may retain
four extra partial-ranked states. Partial scores guide search only; completed
plans use the exact shared scorer.

## Evidence (Grand Geneve, 2026-07-22)

- Five deterministic 50-context cohorts: mean `Mass@10` 0.892 and
  `Recall@10` 0.884, versus 0.853/0.830 at the former compact-proposal width
  of 8. The wider support improves every cohort while keeping the primary
  backward ownership unchanged.
- Full prepared workload: 81,844 contexts, 328,197 steps, 1,110 zones, eight
  threads with profiling: 22.35 s, 70,801 complete; inside the 30-second target.

Proposal-width sweep: 8/12/16/20/24 compact partial proposals score mean
`Mass@10` 0.853/0.879/0.883/0.892/0.894. Twenty is the measured knee: 24 adds
only 0.002 mass while increasing proposal work. The full p20 workload remains
inside the target.

## Superseded factor-map expansion (2026-07-23)

The setting `factor_map_max_depth=99`, with factor-map support also used by the
primary reverse beam and exact reverse guidance, was promoted before a matched
all-depth quality comparison was available. The focused
six-layer internal-home regression, including an anchor that crosses the home
return, matches the exact top-8 oracle.

The full 81,844-context interleaved comparison favors the active setting:
`factor_map_max_depth=99` had median 39.644 s wall / 233.138 s aggregate Rust,
while the otherwise identical depth-5 setting had 47.978 s / 292.763 s
(+21.0% / +25.6% for depth 5). Outputs and work counters differ, as expected
for a quality/runtime policy change. This is not a pure map-depth attribution:
the setting also keeps `continuation_state_limit=1` instead of activating the
deep limit.

The all-depth exact audit sampled ten contexts in each of 41 depth/anchor
strata. The oracle certified 183 of 396 sampled contexts; 169 completed in
the bounded search and gave post-stratified certified `Mass@10` 0.832. It is
coverage evidence. The maintained 50-context short exact sample returned mean
conditional `Mass@10` 0.846; it does not exercise depth 99.

## Current deep fallback (2026-07-23)

A matched 41-stratum audit and full-workload interleaved comparison separated
the factor-map cutoff from the formerly coupled 16-state deep continuation.
The selected policy is `factor_map_max_depth=5` with
`deep_continuation_state_limit=2`.

On the same certified audit, the post-stratified `Mass@10` estimate improves
from 0.832 at depth 99 to 0.839. Deep widths 1/2/4/16 score
0.832/0.839/0.841/0.858; width 2 is the quality/runtime knee. On all 81,844
contexts, width 2 reduces median wall time from 38.303 to 34.642 seconds
(-9.6%), aggregate Rust search time from 216.235 to 201.006 seconds (-7.0%),
and factor-map time from 186.209 to 169.204 seconds (-9.1%).

An eight-candidate heuristic reserve on exact reverse guidance improved the
five short cohorts from mean `Mass@10` 0.886 to 0.895 and the global audit
from 0.832 to 0.845, but added roughly 15--18% runtime. Agreement gating and a
four-candidate reserve either retained the cost or lost the gain. It is not
active.

The local-score cache now uses a deterministic hasher for trusted integer
tuple keys. Five 20,000-context release runs improve median wall time by 3.8%
and aggregate Rust search time by 4.9%. Applying the hasher to factor-map
caches did not help, and packing the local key regressed runtime.
