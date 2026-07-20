# Bounded bidirectional top-K benchmark

## Objective

Measure the active `DestinationPlanSearch.top_k()` path on raw Grand Geneve
zones and check whether its returned plans preserve probability mass from the
exact top-K oracle.

## Latest retained measurements

Release build on Windows, 2026-07-20, cached iteration-5 inputs.

### Runtime

`perf_bidirectional_grand_geneve.py --contexts 1000 --threads 8 --profile`
used 4,767 steps over 1,110 zones, `frontier_width=32`,
`proposal_limit_per_source=16`, `continuation_state_limit=1`,
`continuation_proposal_limit=1`, `seam_refresh_per_prefix=1`,
and `top_k=10`.

| Method | Wall time | Complete contexts | Infeasible contexts |
|---|---:|---:|---:|
| Particle baseline, 32 particles | 0.108 s | 744 | 256 |
| Bounded top-K with right-to-left exchange and one F-to-B alternative | 0.158 s | 787 | 213 |

The search evaluated 656,968 forward proposals, 143,761 backward proposals,
346,627 refresh proposals, 396,919 stitch pairs, and produced 278,134
complete-plan candidates. The refresh added 513 boundary states. In aggregate
Rust search time, the refresh consumed about 27%; ordinary forward search 37%;
continuation guidance 15%; stitching 9%; and backward construction/guidance
9%.

### Exact-oracle quality

`compare_bidirectional_top_k_grand_geneve.py --contexts 10
--candidate-contexts 50 --threads 1` proved 10 short contexts with a top-100
oracle support. Eight candidate contexts were skipped because the exact oracle
could not establish a valid result within its contract or state budget.

| Metric | Mean | Median | Minimum |
|---|---:|---:|---:|
| Exact-plan recall at 10 | 0.6600 | — | — |
| Retained conditional exact top-10 mass | 0.6842 | 0.7278 | 0.0000 |
| Top-10 mass efficiency | 0.8545 | 0.9507 | 0.2479 |

The missing exact top-10 mass split into 0.2182 absent from the original base
proposal pool and 0.0975 supported by that pool but lost by the bounded search.
That classification does not yet credit continuation-projected candidates, so
it is an audit baseline rather than the final attribution.

### Current forward-to-backward refresh sweep

`compare_seam_refresh.py --contexts 50 --seam-refresh-per-prefix 0
--seam-refresh-per-prefix 1 --seam-refresh-per-prefix 2
--seam-refresh-per-prefix 4` proved 50 short contexts; 56 candidate contexts
were explicitly skipped by the exact oracle contract.

| F-to-B alternatives per retained prefix | Mean retained conditional exact top-10 mass | Mean added boundary states |
|---:|---:|---:|
| `0` | 0.7558 | 0.00 |
| `1` (current default) | 0.7634 | 0.84 |
| `2` | 0.7720 | 1.82 |
| `4` | 0.7955 | 3.60 |

The refresh only adds activity-correct states to the reverse frontier; it does
not evict its home-oriented candidates. This improves support without
recreating the losses seen when the stitch boundary is moved later.

### Stitch-boundary bias audit

The balanced `stitch_bias=0` remains the default. A wider exact-oracle sweep
proved 50 short contexts (56 were explicitly skipped by the oracle contract):

| Stitch bias | Mean retained conditional exact top-10 mass | Mean recall@10 |
|---:|---:|---:|
| `-1` | 0.6190 | 0.592 |
| `0` | 0.6505 | 0.628 |
| `+1` | 0.6762 | 0.654 |

The later boundary improves this quality sample, but its 1,000-context,
eight-thread runtime is 0.125 s versus 0.111 s for balanced. It shifts work
from stitching into forward continuation scoring (993,565 forward proposals
versus 654,284). Keep the balanced default until a larger quality sweep and a
clear quality/runtime decision rule justify the extra 13% wall time.

The boundary remains a bounded reconstruction seam, not a restriction on
continuation guidance.

## Stitch-boundary mechanism audit

The asymmetric effect is confined to four-layer chains. For a chain
`A -> B -> C -> home`, balanced stitching owns `A, B` on the forward side and
proposes `C` from the reverse home-near source. Moving the seam later makes
the forward side propose `C` from `B` instead.

Context `17956` is the representative gain. Its exact high-mass plans include
`(173, 173, 173, 221)` and nearby alternatives. Zone `173` is cheap after the
preceding activity but is absent from the reverse home-near pool. Balanced
stitching retained only exact rank 7 (6.56% conditional mass); the later seam
returned ranks 1, 2, 4, 7, 9, and 10 (65.77%). These are repeated zones, not
repeated anchors: the context has no `anchor_id`.

Context `75543` is the counterexample. Exact rank 7 ends in zone `1088`:
it is reverse-home-near and is retained by balanced stitching, but it is not
forward-near from the preceding zone `1081`, so the later seam loses it.

The implemented optimization is therefore not a globally later seam. The
balanced boundary keeps its reverse frontier, then adds a bounded set of
activity-correct candidates proposed forward from every retained prefix. The
new candidates are scored against the retained downstream suffix and deduped
with the original reverse states. At four alternatives, context `17956`
recovers 93.7% of exact top-10 probability mass (versus 6.6% without the
refresh); the `75543` counterexample remains fully retained after the
forward-factor origin correction.

`analysis/compare_stitch_bias.py` reproduces the per-context comparison and
prints the exact top-K plans gained or lost by a chosen bias.

## Rejected seam lookahead

The 2026-07-21 audit found identical retained exact top-10 mass on 50 proven
contexts (0.767567 with or without lookahead). On the 1,000-context runtime
profile, it added 3,952,247 proposals, consumed 58.3% of aggregate Rust search
time, and raised wall time from 0.135 s to 0.258 s. The option and implementation
were removed from the active API/kernel; this section preserves the result.

## Decision

Keep the balanced stitch and 1x1 right-to-left exchange. The 1-per-prefix
forward-to-backward refresh is the conservative default. The 2 and 4
alternative settings are a clear quality/runtime trade-off and remain opt-in
until a larger sweep establishes the desired operating point.

## Reproduce

```powershell
mamba run -n mobility-destination-sequence-sampler python -m experiments.benchmarks.perf_bidirectional_grand_geneve --contexts 1000 --threads 8 --profile
mamba run -n mobility-destination-sequence-sampler python -m experiments.analysis.compare_seam_refresh --contexts 50 --seam-refresh-per-prefix 0 --seam-refresh-per-prefix 1 --seam-refresh-per-prefix 2 --seam-refresh-per-prefix 4
```
