# Top-K work audit, 2026-07-22

This audit separates quality outcomes from bounded-search work. Measurements
use the active symmetric factor-map defaults, release builds, Grand Geneve
iteration-5 inputs, and exact top-100 oracle support where quality is shown.

## Where time goes

On the full 81,844-context workload, factor maps account for about 74% of
aggregate Rust search time. Forward search is the dominant enclosing phase
(about 68%); backward symmetric guidance is about 22%. Stitching,
materialization, the primary backward beam, and scoring-problem construction
are individually around 1% or less.

The full search performs:

- 603,730,853 previous-factor destination scans;
- 1,194,368,028 current-factor destination scans;
- 325,240,675 next-factor destination scans;
- 136,547,357 reverse-prefix partial evaluations;
- 119,100,178 local-score cache hits and 63,726,521 builds.

Factor-map cache request hit rates are 57% previous, 55% current, and 89%
next. The retained feasible entries are 70%, 38%, and 67% of scanned entries,
respectively. Fully dense score maps would therefore waste substantial space,
especially for the current factor.

## Work versus quality

A deterministic 50-context depth-3/4 cohort gives:

| Strategy | Mean Mass@10 | Mean Rust ms/context | Mean map scans | Mean reverse partials |
|---|---:|---:|---:|---:|
| Heuristic | 0.554 | 0.67-0.81 | 0 | 0 |
| Asymmetric factor map | 0.652 | 1.72-1.86 | 25,341 | 0 |
| Symmetric factor map | 0.822 | 3.85-4.09 | 48,584 | 3,416 |

Symmetric versus asymmetric improved 21 contexts and tied 29; it did not
lose a context. Symmetric versus heuristic improved 38, lost 3, and tied 9.
Asymmetric versus heuristic improved 24, lost 17, and tied 9. The factor-map
policy is therefore stronger in aggregate but does discard useful legacy
support in real cases.

Symmetric work is negatively associated with success in this cohort:

| Symmetric outcome | Contexts | Mean Mass@10 | Mean map scans | Mean reverse partials |
|---|---:|---:|---:|---:|
| Mass at least 0.8 | 38 | 0.965 | 40,210 | 3,272 |
| Partial mass | 11 | 0.402 | 72,448 | 3,764 |
| Zero mass | 1 | 0.000 | 104,258 | 5,044 |

The expensive tail is not buying proportional quality. It indicates weak or
conflicting continuation hypotheses rather than insufficient raw width.
Increasing symmetric proposals from 8 to 12/16 repairs some cases but leaves
the zero cases unchanged and is non-monotonic on context 3647.

### Detailed cases

- Context 26 (five layers, one repeated anchor): asymmetric Mass@10 0.733;
  symmetric 0.904. Symmetry and compact anchor handoff are doing useful work.
- Context 45331 (four layers, repeated anchor): asymmetric 0.086; symmetric
  0.889. The reverse channel is essential.
- Context 2679 (four layers, no variable anchor): heuristic 0.667;
  asymmetric and symmetric both 0.000. Seven exact top-ten plans are supported
  by the legacy attractive/OD-near pool. Factor maps retain globally inferior
  recombinations despite scanning 183,292 entries in the symmetric search.
- Context 61440: heuristic 0.268; both factor-map policies 0.000. Symmetric
  work increases without recovering the missing support.
- Context 3506: heuristic 0.737; asymmetric 0.000; symmetric 0.082.

Exploration seeds 42-46 produce identical outputs on all eight fixed canary
cases. More random exploration is not a promising repair for these failures.

### All-depth pilot (2026-07-22)

An all-depth, two-context-per-stratum pilot at a 500k exact-oracle state cap
sampled 44 contexts across 22 depth/anchor strata. Only 27 were oracle-proven:
the cap or an oracle infeasibility outcome affected 17 samples, concentrated in
the `6+` depth band. This is an oracle-coverage limitation, not evidence that
the bounded search fails there. Conditional mean Mass@10 was 0.913 (weighted
pilot estimate 0.847 across the oracle-certifiable 91.6% of population).
Observed conditional means were 0.858 at depth 4, 0.797 at depth 5, and 0.950
for the one certified depth-6 case; the latter sample is far too small for a
depth conclusion.

Context 46889 (six layers, four anchors across two activity types) is the
useful certified deep trace: it returns exact ranks 1--9, retaining 0.950 of
top-ten mass. Primary forward and backward beams retain every target zone for
those nine plans. Its only miss, exact rank 10, requires backward-layer-4 zone
820; that zone is never proposed and lies outside the legacy heuristic pool.
The current deep failure hypothesis is therefore a missing reverse support
candidate, not factor-map ranking or seam pruning. The trace now records the
primary backward beam as well as the forward beam; it should be used before
testing any depth-dependent reserve.

### Removing the depth cutoff (2026-07-22)

Treating every context through the symmetric factor-map path (setting
`factor_map_max_depth=99`, while retaining the direct exact two-step path) did
not change any of the 27 oracle-proven results in the matched all-depth pilot:
conditional Mass@10 remained 0.913 and the post-stratified pilot estimate
remained 0.847. It is not a proof of equality for all deep contexts because
the oracle cannot certify many `6+` cases, but it shows no measured quality
gain.

On the full 81,844-context eight-thread workload it regressed wall time from
24.370s to 25.543s (+4.8%). Map destination scans rose from 2.123B to 3.015B
(+42%), reverse-prefix partials from 136.5M to 204.9M (+50%), and aggregate
factor-map CPU from 111.5s to 130.1s. Completed-plan candidates also fell
from 22.475M to 22.252M. Keep the depth cutoff: removing it is more work with
no established quality benefit.

### Top-K tail-normalizer calibration (2026-07-22)

The cached exact top-500 plans can calibrate an extrapolation only within that
finite support. Hiding ranks K+1--500 and extending the final ten visible
log-score gaps as a geometric tail is badly optimistic on 211 complete cached
top-500 contexts: mean actual top-K mass within top-500 versus mean estimated
mass is 0.135/0.477 at K=10, 0.208/0.444 at K=20, 0.349/0.495 at K=50, and
0.497/0.566 at K=100. The K=10 mean absolute error is 0.342. Score spacings
flatten in the tail, so fitting the visible head understates remaining mass.

Consequently top-10/20/50/100 scores alone cannot identify total path mass:
arbitrarily many lower-scoring feasible paths may remain. They provide an
exact lower-bound normalizer over the enumerated plans and can support a
learned, calibrated *top-500* coverage predictor, but estimating total mass
requires a separate partition-function estimator (for example sequential
importance sampling) or a defensible bound/count for all feasible paths.

Two fully enumerated small contexts demonstrate why a fixed top-K mass claim
is unsafe. Context 45331 (four layers, repeated anchor) has 111 feasible paths
from a 759-assignment lattice: its top 10 hold 0.522 of the full exp(U) mass,
top 50 hold 0.920, and entropy gives an effective support of 39.8 paths.
Context 37132 (three layers) has 570 feasible paths from 964 assignments: top
10 hold only 0.222, top 50 0.539, top 100 0.730, and its effective support is
188.9 paths. The guarded `exact_distribution` oracle and inspection script are
for these small diagnostic contexts only; they are not a production fallback.

On 49 fully enumerated cached small contexts, three head-only tail families
were evaluated on a deterministic holdout split while being given the true
feasible-path count (an optimistic condition unavailable on general contexts).
The locally geometric model, fit from the last ten visible score gaps,
overstates concentration: holdout MAE is 0.267/0.140/0.063/0.027 for K
10/20/50/100, with median signed errors -0.264/-0.148/-0.038/0.000 when error
is actual minus predicted mass. A log-rank fit has lower K=10 MAE (0.196) but
systematically understates concentration. Empirical residual intervals are not
yet stably calibrated across K. Treat this as a diagnostic bracket and a basis
for a future calibrated predictor, not an estimator for production total mass.

The full ranked score curves are usually closer to exponential ranked weights
than power-law tails: median R² for `U(rank)=a+b*rank` is 0.891, versus
0.851/0.799/0.694 for square-root stretched-exponential, quarter-power, and
log-rank (power-law-weight) curves; best-fit counts are 42/6/1/0. This is
descriptive only. The exponential decay rate changes between head and tail,
so the full-support family fit does not validate extrapolating that rate from
top K.

### Flat returned-support cases (2026-07-22)

The regular returned-top-100 diagnostic was followed by a matched comparison
of the 20 flattest supports in the fixed 1,000-context workload against the
most concentrated controls with the same depth and fixed/variable-step count.
Flat cases have mean returned Mass@10 0.145 and effective returned support
98.0, versus 0.557 and 42.9 for controls. They are not broad geographical
searches: inspected pairs commonly reuse only 5--7 distinct zones in their top
10, with many near-tied recombinations. They cost more nevertheless: 9.61ms
versus 6.75ms (+42%), factor-map CPU 6.49ms versus 4.56ms, and reverse-prefix
partials 4,901 versus 2,770 (+77%), despite 13% fewer map scans. The current
hypothesis is ambiguous local continuation messages, not an insufficiently
large destination domain.

### Sparse pair-state DP rejection (2026-07-22)

The isolated pair-state experiment retained top-K histories per
`(previous_zone, current_zone)` pair, using the cheap 16-attractive +
16-outgoing-cost-near + 2-exploration lattice. It recovered the exact winner
on context 2679 (`1035 -> 904 -> 824 -> 1036`), demonstrating that the unary
backward guidance can lose a predecessor-sensitive relation. That single-case
success did not generalize: a deterministic 50-context oracle-proven,
depth-3--5 cohort without repeated anchors gave mean Recall@10/Mass@10 of
0.649/0.695 for pair state (49/50 completed), versus 0.814/0.838 for active
symmetric factor maps (50/50). Pair state cost 6.75 ms/context versus 6.67
ms, and context 71190 hit its 100,000 retained-path cap. It never exceeded
the heuristic candidate lattice on that cohort. Remove the implementation;
retain the coherent trace as the diagnostic for this failure mechanism.

## Repeated and avoidable work

`reverse_prefix_partial_score` was called 136.5 million times and allocated a
fresh `Vec<bool>` on every call. Recomputing the small exact-factor predicate
without allocation preserved every fixed-case output. Matched release
measurements improved:

- 1,000 contexts, one thread: median wall 2.503 -> 2.415 seconds (-3.5%);
  factor-map CPU 1.825 -> 1.723 seconds (-5.6%).
- Full workload, eight threads: median wall 26.033 -> 25.192 seconds (-3.2%);
  factor-map CPU 137.823 -> 131.808 seconds (-4.4%).

Per-context factor-map caching is valuable, especially for the next map.
Cross-context exact-map caching is not attractive: among the 157,312 steps in
depth-3/5 contexts, adjacent timing/scoring profiles have only 1.18x average
reuse and 95.5% are singletons. Restricting to the 94,150 variable layers
raises the theoretical profile reuse to 1.35x, but 91.7% remain singletons and
dynamic origin/suffix keys reduce actual map overlap further. The already
rejected reverse-prefix query cache is consistent with this result.

`LocalScoreCache` has a 65% full-workload hit rate, so removing it would repeat
substantial exact scoring. Its tuple-keyed standard `HashMap` may still be
worth replacing with a packed integer key and cheaper trusted-input hasher.

Seam refresh evaluates 7,805,528 proposals to add 5,884 states (0.075%). It
uses about 1.8% of aggregate search time. Refresh widths 0/1/2/4 are
quality-identical on the current 50-context cohort and the eight canaries, but
disabling it loses 28 completed contexts on the full workload. Do not remove
it until those contexts and an all-depth oracle audit are examined.

## Current position

The active default remains symmetric factor-map search with its depth cutoff.
The dense graph views, compact sparse factor maps, and active coherent tracing
are retained. Its compact partial-message proposal budget is now 20: across
five deterministic 50-context cohorts it raises mean Mass@10/Recall@10 from
0.853/0.830 (budget 8) to 0.892/0.884, while the full 81,844-context release
workload finishes in 22.35 s. Budget 24 reaches only 0.894 Mass@10, so 20 is
the current knee. The small disagreement-gated heuristic reserve remains
disabled: it has no stable cohort gain. Exact one-layer seed repair is retained
only as a diagnostic: it recovers the rank-one 2679 path but costs about ten
seconds per pass. The next work should use the coherent trace on hard/deep
cases to distinguish missing candidate support from state pruning.

### Deep oracle coverage diagnostic (2026-07-23)

The former all-depth pilot requested exact top-100 plans even though its
primary comparison metric was Mass@10. A new stratified diagnostic records the
pre-search exact shape (independent variables, log10 assignment lattice,
home returns, and repeated anchors) for both solved and capped cases, plus
heap/pruning counters for solved cases.

On two deterministic samples per 22 strata at a 500k-state cap, exact top-10
proof completes 32/44 samples and gives 93.7% population coverage by
oracle-certifiable stratum (31 bounded completions). The remaining state caps
have median five independent variables and a `10^15` assignment lattice,
versus two variables and `10^6` for solved contexts. Repeated anchors do not
separate the two groups in this pilot. Requesting top 10 is exact for
Recall@10/Mass@10 and cuts median solved states from 1,110 under top-100 to
270. Keep top-100 audits for tail/distribution questions; use the new
`just audit-deep-top-k` gate for deep primary-quality coverage.

The audit then exposed an exact-oracle correctness issue: the home-range split
could concatenate independently ranked segments into a sequence rejected by
the full scorer (context 9619). The shortcut is disabled until its crossing
factor can be allocated exactly at merge. The corrected 5-per-stratum audit
has no internal oracle errors, proves 78/110 samples, and retains the same
99.4% stratum-population coverage. It certifies six depth-6 contexts at mean
Mass@10 0.593; deep fallback remains weak, but its measurement is now sound.

On that corrected cached support, removing the depth cutoff
(`factor_map_max_depth=99`) is not a deep repair: it completes one additional
case but lowers the certified depth-6 mean from 0.593 to 0.551 and leaves four
zero-mass cases. Contexts 944 and 50986 lose decisive first-layer destinations
under the heuristic fallback; full factor maps do not recover them. The next
deep experiment must add targeted first-layer/reverse support, not merely
extend factor-map depth.

### Depth-resolved deep audit (2026-07-23)

The `6+` band is now split into depths 6, 7, 8, 9, and 10+. The cached
10-per-stratum exact-top-10 audit has 183 oracle-proven cases and 99.7%
stratum-population coverage. Certified mean Mass@10 falls with depth:
depth 5/6/7/8 is 0.781/0.681/0.602/0.528 (depths 9 and 10 have only two and
four proven cases). State-capped contexts have median six independent
variables and a `10^18` assignment lattice.

Representative zeroes span depths 6--10. Full factor maps through depth 99
recover none of five such cases. Widening the heuristic pool from 16 to 64
per source recovers selected depth-7/8 cases (Mass@10 0.616 and 0.273), but
not the depth-6/7/10 zeroes. In context 944, that wider pool does retain the
true first-layer choices; the plan is then lost in later continuation/beam
selection. Pool 64 plus beam 128 recovers only one path (Mass@10 0.095) at
9.88 ms/context. Reject global deep widening: deep losses mix missing support
with continuation ranking, so a future repair must be selectively triggered.
