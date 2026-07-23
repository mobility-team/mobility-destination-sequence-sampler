# Active: symmetric unbinned factor-map top-K

`top_k()` defaults to `candidate_strategy="symmetric_factor_map"`,
`symmetric_message_limit=4`, `symmetric_state_limit=4`,
`symmetric_forward_proposal_limit=20`, and `stitch_bias=1`.
Depth 2 is a direct scan; factor maps apply through depth 5; longer plans use
the heuristic.

The primary backward channel retains exact suffix utility. An independent
four-state reverse channel combines the exact known right factor with every
locally complete known prefix factor, falling back to endpoint/attraction
terms. Repeated-anchor proposals are handed forward as compact destination
assignments. Forward search preserves its primary beam, unions twenty partial
candidates, and may retain four extra partial-ranked states. Partial scores
guide search only; completed plans use the exact shared scorer.

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
