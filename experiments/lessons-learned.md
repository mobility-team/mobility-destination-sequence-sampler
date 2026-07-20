# Lessons learned

This is the short decision record for the destination-sampling experiments.
Detailed measurements live in the linked experiment records; the root
[`../BENCHMARKS.md`](../BENCHMARKS.md) contains only current active results.

## Active bidirectional top-K

- Bounded beams are substantially cheaper than all-zone exact search.
- A narrow right-to-left continuation exchange improves mean oracle quality,
  but still has a zero-mass tail case under audit.
- The current attractive/OD-near candidate pool loses important support in
  some contexts before beam pruning occurs.
- Increasing returned `top_k` does not recover plans absent from proposal
  support.
- The next useful experiment is continuation-aware proposal support for early
  forward layers.

## Particle candidates

- Particles are a useful sequential bounded baseline.
- Retry behavior can recover dead frontiers, but retry tuning does not address
  bidirectional top-K proposal support.
- Completed particles should continue to be rescored by the exact reference
  scorer for small-case validation.

## Exact references

- The exhaustive ternary scorer is the correctness authority for small cases.
- The exact heap search is useful as a top-K oracle when its state limit can
  prove the result.
- These reference paths should not be treated as production replacements for
  the bounded sampler.

## Rejected optimization directions

- Scalar candidate-level utility-bound pruning was slower than sequential OD
  scans because of sorting and random lookups.
- Blindly widening candidate or continuation lists increases scoring cost
  without fixing the underlying proposal problem efficiently.
- An all-zone exact bidirectional dynamic program is outside the intended
  bounded/sample-like design.
- Seam lookahead retained identical oracle mass (0.767567 on 50 proven
  contexts) but nearly doubled 1,000-context wall time (0.135s to 0.258s).
  It was removed from the active API/kernel after this audit.

## Paused research paths

The second-order recursion, factor-tree approach, and hierarchical travel
kernel remain useful reference material, but their current runtime or quality
tradeoffs do not make them the active redesign.
