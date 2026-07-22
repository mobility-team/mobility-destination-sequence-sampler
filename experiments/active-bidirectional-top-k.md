# Active: symmetric unbinned factor-map top-K

`top_k()` defaults to `candidate_strategy="symmetric_factor_map"`,
`symmetric_state_limit=4`, `symmetric_forward_proposal_limit=16`, and
`stitch_bias=1`.
Depth 2 is a direct scan; factor maps apply through depth 5; longer plans use
the heuristic.

The primary backward channel retains exact suffix utility. An independent
four-state reverse channel combines the exact known right factor with known
prefix endpoint/attraction terms. Forward search preserves its primary beam,
unions 16 candidates from the partial channel, and may retain four extra
partial-ranked states. Partial scores guide search only; completed plans are
ranked by the exact shared scorer. Sparse factor maps omit infeasible cells and
are cached by fixed neighbours.

## Evidence (Grand Geneve, 2026-07-22)

- Five deterministic 50-context cohorts: mean `Mass@10` 0.803 and
  `Recall@10` 0.780, versus 0.703/0.684 for the asymmetric factor-map channel.
- Full prepared workload: 81,844 contexts, 328,197 steps, 1,110 zones, eight
  threads: 25.46 s, 70,733 complete; inside the 30-second target.

Tuning: zero auxiliary proposals loses the gain; eight proposals reduce the
full workload to 20.06 s but lower five-seed mean `Mass@10` to 0.787. Eight
states add little recall for ~48% more bounded-search time. Defaults are the
measured quality/runtime knee.
