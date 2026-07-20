# Hierarchical travel-kernel exploration

## Objective

Test whether a hierarchical low-rank representation of the Grand Geneve
exponentiated OD-cost kernel can reduce the historical exact backward-message
work without unacceptable destination-probability error.

## Retained measurements

At model scale 0.25, a single global approximation needed rank 353 for 99%
energy and rank 901 for 99.9% energy. It is not a useful global compression.

Spatially separated blocks were more compressible. With 16-zone leaves,
far-separation 0.25, and 99% retained energy:

| Measure | Result |
|---|---:|
| Matrix blocks | 2,030 |
| Far low-rank blocks | 1,172 |
| Far-block rank, median / p90 / max | 1 / 2 / 25 |
| Whole-matrix storage reduction | 6.12x |
| Conditional-distribution TV, median / p95 | 2.00% / 4.26% |
| Backward-message error, median / max | 0.67% / 1.61% |

The matched optimized matrix-product reference was 2.6–3.0 times faster than
the dense Rust loop. Factoring each transition profile is not practical: the
real input had 12,724 exact profiles and still 954 after one-hour rounding.

## Lesson

Far-field transport information is spatially compressible, but local travel
structure and missing OD pairs make a single global embedding too inaccurate.
The arithmetic gain alone cannot bring the retired exact sampler near the
target runtime, and per-profile factorization removes the reuse opportunity.

## Decision

Archived from the working tree at `research-archive-2026-07-21`. Revisit only
as a component of a fixed-budget proposal mechanism or a targeted reusable
approximation; recover the tagged source if that happens.
