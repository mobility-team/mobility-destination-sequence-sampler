# Active: symmetric unbinned factor-map top-K

`top_k()` defaults to `candidate_strategy="symmetric_factor_map"`,
`symmetric_message_limit=4`, `symmetric_state_limit=4`,
`symmetric_forward_proposal_limit=20`, and `stitch_bias=1`.
Depth 2 is a direct scan. `factor_map_max_depth=99` keeps ordinary
home-bounded tours on the factor-map path; a longer uninterrupted tour may use
the heuristic. A fixed home return is a map boundary, but factors crossing it
remain exactly scored and cross-home anchors remain in the state.

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

## Current factor-map expansion (2026-07-23)

The active setting is now `factor_map_max_depth=99`, with factor-map support
also used by the primary reverse beam and exact reverse guidance. The focused
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
coverage evidence, not a before/after quality claim, because the matched
depth-5 audit has not been run. The maintained 50-context short exact sample
returned mean conditional `Mass@10` 0.846; it does not exercise depth 99.
