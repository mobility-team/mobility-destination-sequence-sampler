# Active experiment: bounded bidirectional top-K

## Status

Active redesign direction.

## Objective

Generate high-quality destination-sequence candidates for a fixed terminal
home while keeping candidate work and beam state bounded.

## Implementation

The sampler grows bounded forward and backward beams, using attractive zones,
OD-cost-near zones, and deterministic exploration candidates. It scores
rigidity-aware ternary factors when both adjacent trips are known, sends a
narrow right-to-left continuation message to early forward layers, carries
repeated-anchor assignments, and ranks stitched plans with the exact
complete-plan scorer.

The current defaults are `frontier_width=32`,
`proposal_limit_per_source=16`, `stitch_bias=0`, `continuation_state_limit=1`,
`continuation_proposal_limit=1`, `seam_refresh_per_prefix=1`,
and `top_k=10`.
The public method is `DestinationPlanSearch.top_k()`.

The current runtime and oracle-quality measurements are in
[`benchmarks/bidirectional-top-k.md`](benchmarks/bidirectional-top-k.md).

## Current limitation

The base candidate pool can omit destinations that are not highly attractive
or individually close by OD cost but become competitive after considering both
outbound and return legs. The known next slice is a small continuation-aware
candidate source for early forward layers, especially the first destination.

## Validation

Use the exact ternary top-K oracle and compare retained conditional probability
mass, top-K efficiency, and the split between missing candidate support and
beam pruning. Use `--trace-context` for plan-level diagnostics.

## Decision rule

Keep a proposal change only if it improves oracle retained mass without an
unacceptable increase in continuation-scoring or total runtime. Do not solve
this by blindly widening every candidate list.
