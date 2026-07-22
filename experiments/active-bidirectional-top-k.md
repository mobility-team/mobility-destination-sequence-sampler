# Active: symmetric unbinned factor-map top-K

`top_k()` defaults to `candidate_strategy="symmetric_factor_map"`,
`symmetric_message_limit=4`, `symmetric_state_limit=4`,
`symmetric_forward_proposal_limit=8`, and `stitch_bias=1`.
Depth 2 is a direct scan; factor maps apply through depth 5; longer plans use
the heuristic.

The primary backward channel retains exact suffix utility. An independent
four-state reverse channel combines the exact known right factor with every
locally complete known prefix factor, falling back to endpoint/attraction
terms. Repeated-anchor proposals are handed forward as compact destination
assignments. Forward search preserves its primary beam, unions eight partial
candidates, and may retain four extra partial-ranked states. Partial scores
guide search only; completed plans use the exact shared scorer.

## Evidence (Grand Geneve, 2026-07-22)

- Five deterministic 50-context cohorts: mean `Mass@10` 0.854 and
  `Recall@10` 0.832, versus 0.803/0.780 before known-factor and compact-anchor
  guidance, and 0.703/0.684 for the asymmetric channel.
- Full prepared workload: 81,844 contexts, 328,197 steps, 1,110 zones, eight
  threads with profiling: 28.93 s, 70,799 complete; inside the 30-second target.

Tuning: 12 proposals score 0.881 but take 30.60 s; 16 score 0.888 and take
31.70 s. Wider reverse messages reach 0.873/0.892 at widths 8/16 but cost
39.24/57.66 s. Defaults are the measured quality/runtime boundary.
