# Active: binned-surface top-K

`top_k()` defaults to `candidate_strategy="surface"`. It grows bounded
bidirectional frontiers, uses exact scoring for retained plans, and stitches
the two fronts exactly. Depth 2 is direct exact scan; depths 3--4 use the
surface on the forward front; longer plans retain the cheap heuristic.

The surface scores the full activity domain against the leading backward
successor, keeps winners from cells of three precomputed rank features:
activity attraction, inbound cost, and max inbound/outbound time pressure.
The active 2x2x2 surface retains up to 32 diverse candidates (two 16-wide
sources). It changes proposal support only; scoring-factor ownership is
unchanged.

## Evidence (Grand Geneve, 2026-07-21)

- Global stratified pilot, three contexts in each of 22 strata: weighted
  `Mass@10` 0.735 vs heuristic 0.707; oracle-support mass 0.296 vs 0.291;
  both certify 93.9% of sampled population (44 exact-proven contexts).
- Full prepared workload: 81,844 contexts, 328,197 steps, 1,110 zones,
  eight threads: surface 8.785 s, 70,143 complete; heuristic 6.591 s,
  70,351 complete. Both are inside the 30 s target.
- A 4x4x4 surface sharply degraded the short-cohort mass (0.315); keep the
  exposed resolution for experiments, but use `surface_bins=2`.

Rejected local-ranking, projection, widening, fallback, and seam-lookahead
variants are recorded in `historical.md` and Git history. Next work should
focus on a clearly measurable proposal-support improvement, not wider beams.
