# Active: unbinned factor-map top-K

`top_k()` defaults to `candidate_strategy="factor_map"` with
`stitch_bias=1`. It grows bounded
bidirectional frontiers, exact-scores retained plans, and stitches them
exactly. Depth 2 is a direct scan; factor maps apply through depth 5; longer
plans use the cheap heuristic.

For each forward candidate destination, the proposal path builds three exact
destination maps for the affected previous, current, and next activity
factors. Missing entries are infeasible. Their support is intersected and the
sum is partial-top-K selected without bins. Four propagated backward suffix
hypotheses each contribute part of the 32-candidate budget; maps with the same
fixed neighbours are cached per context.

## Evidence (Grand Geneve, 2026-07-21)

- Global five-per-stratum audit: weighted `Mass@10` 0.784 at width 40,
  versus 0.772 at width 32, centred factor-map 0.759, surface 0.735, and
  heuristic 0.707. It has 69 proven contexts (99.4% coverage); an independent
  seed-43 three-per-stratum cohort gives 0.774 over 36 (91.5%).
- Full prepared workload: 81,844 contexts, 328,197 steps, 1,110 zones,
  eight threads: factor-map depth 5 14.56 s, 70,320 complete; inside 30 s.

Rejected: one suffix collapses geographically (0.488 pilot); eight suffixes
split the fixed candidate budget (0.745); 32 proposals per source add beam
loss (0.698); depth 6 loses long-chain continuation quality (0.706). The
binned 2x2x2 surface remains the fast comparator (8.785 s, `Mass@10` 0.735).
