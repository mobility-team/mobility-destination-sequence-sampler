# Active: adaptive unbinned factor-map top-K

`top_k()` defaults to `candidate_strategy="adaptive_factor_map"`,
`symmetric_message_limit=4`, `symmetric_state_limit=4`,
`symmetric_forward_proposal_limit=20`, and `stitch_bias=1`.
Depth 2 is a direct scan. Factor maps apply while every home-bounded tour is at
most depth 5; a longer uninterrupted tour uses the heuristic pool with two
exact continuation states. A fixed home return is a map boundary, but factors
crossing it remain exactly scored and cross-home anchors remain in the state.

Both primary backward support and reverse guidance use destination-resolution
factor maps under the factor-map policies. An independent four-state reverse
channel combines the exact known right factor with every locally complete known
prefix factor, falling back to endpoint/attraction terms. Repeated-anchor
proposals are handed forward as compact destination assignments. Forward search
preserves its primary beam, unions twenty partial candidates, and may retain
four extra partial-ranked states. Partial scores guide search only; completed
plans use the exact shared scorer.

## Local symmetric-guidance router (2026-07-27)

The active strategy selects between the same two measured factor-map channels
using only local problem structure. It keeps symmetric partial messages when
an anchor repeats or at least two unresolved destinations are adjacent.
Otherwise every variable is bracketed by fixed destinations, and the router
uses ordinary exact factor maps without the partial reverse pass. This is
feature-based algorithm selection, not a scoring approximation: both branches
retain the same factor-ownership and exact final-ranking rules.

The rule routes 16,144 contexts, 28.4% of the non-two-step workload, to the
cheaper branch. It was selected after a 40-context discovery comparison:
symmetric guidance beat ordinary factor maps on 12 cases; nine had repeated
anchors, while all three remaining gains had a consecutive-variable run.

On a fresh locked ten-per-stratum validation cohort, both policies have
post-stratified `Mass@10=0.893736`, `Recall@10=0.885990`, and 24 certified
zero-overlap cases across 258 oracle-certified contexts. A linked
20,000-context, two-block release validation improves paired wall time by
3.7%, aggregate Rust time by 3.9%, and factor-map CPU by 4.8%. Raw median
deltas are -5.5%/-4.9%/-5.5%, respectively. The two paired wall blocks span
-7.5% to +0.1%, while aggregate Rust improves in both blocks.

## Compiled exact factor scoring (2026-07-27)

Factor-map construction previously repeated activity-table hash lookup,
first-choice and terminal checks, and adjacent-step resolution for every
destination in every map. The active implementation compiles those immutable
layer-local inputs once per context. Map construction, reverse-prefix
guidance, and exact pricing columns reuse the compiled evaluator; arithmetic
and feasibility remain the shared scorer's exact operations.

A locked 20,000-context, two-block pure-performance validation produced
identical output fingerprints and every measured work counter. Paired median
aggregate Rust time improves 14.4%, factor-map CPU 17.3%, pricing CPU 8.9%,
and wall time 5.8%. Both Rust blocks improve, spanning -15.9% to -12.8%;
paired wall blocks span -9.0% to -2.7%. A proposal-only logarithm
approximation was tested separately and removed because it did not accelerate
the Windows release build.

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

## Superseded factor-map expansion (2026-07-23)

The setting `factor_map_max_depth=99`, with factor-map support also used by the
primary reverse beam and exact reverse guidance, was promoted before a matched
all-depth quality comparison was available. The focused
six-layer internal-home regression, including an anchor that crosses the home
return, matches the exact top-8 oracle.

The full 81,844-context interleaved comparison favors the active setting:
`factor_map_max_depth=99` had median 39.644 s wall / 233.138 s aggregate Rust,
while the otherwise identical depth-5 setting had 47.978 s / 292.763 s
(+21.0% / +25.6% for depth 5). Outputs and work counters differ, as expected
for a quality/runtime policy change. This is not a pure map-depth attribution:
the setting also keeps `continuation_state_limit=1` instead of activating the
deep limit.

The all-depth exact audit sampled ten contexts in each of 41 depth/anchor
strata. The oracle certified 183 of 396 sampled contexts; 169 completed in
the bounded search and gave post-stratified certified `Mass@10` 0.832. It is
coverage evidence. The maintained 50-context short exact sample returned mean
conditional `Mass@10` 0.846; it does not exercise depth 99.

## Current deep fallback (2026-07-23)

A matched 41-stratum audit and full-workload interleaved comparison separated
the factor-map cutoff from the formerly coupled 16-state deep continuation.
The selected policy is `factor_map_max_depth=5` with
`deep_continuation_state_limit=2`.

On the same certified audit, the post-stratified `Mass@10` estimate improves
from 0.832 at depth 99 to 0.839. Deep widths 1/2/4/16 score
0.832/0.839/0.841/0.858; width 2 is the quality/runtime knee. On all 81,844
contexts, width 2 reduces median wall time from 38.303 to 34.642 seconds
(-9.6%), aggregate Rust search time from 216.235 to 201.006 seconds (-7.0%),
and factor-map time from 186.209 to 169.204 seconds (-9.1%).

An eight-candidate heuristic reserve on exact reverse guidance improved the
five short cohorts from mean `Mass@10` 0.886 to 0.895 and the global audit
from 0.832 to 0.845, but added roughly 15--18% runtime. Agreement gating and a
four-candidate reserve either retained the cost or lost the gain. It is not
active.

The local-score cache now uses a deterministic hasher for trusted integer
tuple keys. Five 20,000-context release runs improve median wall time by 3.8%
and aggregate Rust search time by 4.9%. Applying the hasher to factor-map
caches did not help, and packing the local key regressed runtime.

## Adaptive exact path pricing (2026-07-27)

The active policy adds a depth-routed completed-path pass:
`pricing_passes=2`, `pricing_seed_limit=10`, `pricing_column_limit=4`,
`pricing_next_pass_min_new=3`, and `pricing_min_layers=6`. For an unanchored
layer it reuses the exact destination-resolution factor map to rank
single-variable replacements. Repeated anchors are changed as one group and
all affected factors are scored exactly. Every retained column is fully
rescored before final ranking.

This is dynamic support generated around complete paths, not another static
candidate-pool expansion. It directly targets the high-quality columns that
the bounded forward/reverse proposal intersection missed. That distinction is
consistent with large-neighborhood search and column-generation views of
large-scale combinatorial inference: search a tractable incumbent neighborhood, price
promising missing structures, and rerank with the original objective
([Song et al., 2020](https://proceedings.neurips.cc/paper/2020/hash/e769e03a9d329b2e864b4bf4ff54ff39-Abstract.html);
[Desrosiers and Lübbecke, 2005](https://doi.org/10.1007/0-387-25486-2_1)).

On the ten-per-stratum all-depth audit, the seeded oracle proved 258 of 396
sampled contexts and 246 had bounded results. Relative to pricing disabled,
conditional `Mass@10` improves from 0.793 to 0.850, zero-overlap cases fall
from 15 to 9, and the post-stratified certified estimate improves from 0.847
to 0.872. Conditional depth means improve by +0.109 at depth 6, +0.252 at
depth 7, +0.079 at depth 8, +0.129 at depth 9, and +0.118 at depth 10.

The router study found that depth 6 or greater covers 20.5% of the workload.
Requiring at least three new surviving plans before a second pass routes 34.6%
of deep cases—about 7% of the full workload—while capturing 96.6% of the
observed second-pass quality gain.

On the full 81,844-context release workload, adaptive pricing measured
48.349 s wall / 286.675 s aggregate Rust versus 43.805 s / 240.532 s with
pricing disabled: +10.4% wall and +19.2% Rust. Exact factor-map reuse reduced
the pass from the earlier scalar prototype; two 5,000-context repeats measured
-1.4% and +5.4% wall. The policy is promoted because the large and consistent
deep-context gain clears the predeclared 15% full-workload wall-time ceiling.

The exact oracle now exact-rescores active bounded results as initial
branch-and-bound incumbents. In the two-per-stratum pilot this increased solved
contexts from 40 to 56 and reduced state-budget failures from 30 to 14.
Seeded and cold modes returned identical exact top-K results wherever both
completed. The cache now stores completed certificates under input and exact
semantics only, while capped attempts separately fingerprint the active
initializer and state budget. This follows the
well-established sensitivity of m-best branch-and-bound to heuristic strength
([Dechter, Flerova, and Marinescu, 2012](https://ojs.aaai.org/index.php/AAAI/article/view/8405)).

## Routed interacting-pair pricing (2026-07-27)

Single-variable pricing still misses plans where two individually weak
replacements are strong together. The promoted extension crosses the best
exact conditional columns only for variable groups whose affected factor
windows overlap. It prices the union of affected factors, keeps at most the
working top-K joint columns per neighborhood, then fully rescores every
survivor. Repeated anchors remain atomic groups.

The active router uses four conditional candidates per variable at depths 6–8
and eight from depth 9 onward. On 106 exact-certified deep contexts, the prior
active policy scores conditional `Mass@10` 0.771. Uniform pair limits 4 and 8
score 0.811 and 0.823; the routed 4/8 policy scores 0.821. It improves 15
contexts with no losses and reduces zero-overlap cases from 7 to 5. The largest
recoveries include 0.000→1.000, 0.000→0.909, 0.094→0.813, and 0.064→0.760.

Across all 250 bounded-complete certified cases, uniform limits 4 and 8 raise
conditional mass from 0.841 to 0.858/0.863 and the post-stratified estimate
from 0.871 to 0.876/0.878. The routed policy retains almost all of the
limit-8 deep gain while halving pair evaluations.

The final full-workload two-cycle comparison uses 81,844 contexts and eight
threads. The prior active policy has median 39.627 s wall / 226.614 s aggregate
Rust; routed pair pricing has 41.102 s / 241.539 s: +3.7% wall and +6.6% Rust.
Pricing CPU rises from 19.413 to 25.553 seconds. A 20,000-context calibrated
comparison measured +5.6% wall. The routed policy is promoted: its large,
one-sided deep-quality gain clears the output-changing quality gate at a small
full-workload wall-time increment.

## Adaptive local pair router (2026-07-27)

The active pair router now probes the best four exact conditional columns per
interacting variable and expands that pair to eight only when the best 4x4
candidate improves the current working Kth score by more than 0.2. The
second-pass requirement remains at least three newly surviving plans.
`pricing_pair_deep_min_layers=0` selects this local rule; positive values
retain the former depth-routed policy as an experiment comparator.

The simple short-context baseline was tried first. Lowering
`pricing_min_layers` from 6 to 3 improved a locked validation cohort from
conditional `Mass@10` 0.864 to 0.878 and post-stratified mass 0.884823 to
0.898616 without changing zero overlap. It was rejected because the
20,000-context paired runtime gate regressed wall time by 7.8%, aggregate Rust
by 10.4%, and pricing CPU by 113.6%.

The 4x4 probe study measured truncation gap, neighborhood saturation,
working-top-K entry, Kth-score improvement, pair non-additivity, and
feasible/evaluated ratio. Working-K entry was the strongest threshold-free
signal, but its full-workload pair count was 11.27M versus 7.97M for the
depth router. A 0.1 Kth-improvement margin preserved the exact repairs but
missed its locked 3% wall gate by a fraction. The frozen 0.2 margin cleared
fresh validation.

On the final untouched stratified cohort, the oracle certified 258 of 396
sampled contexts. Relative to the depth router, conditional `Mass@10`
improves 0.861 to 0.864, conditional recall 0.856 to 0.859, post-stratified
mass 0.891126 to 0.892315, and post-stratified recall 0.885857 to 0.887460;
zero-overlap cases are unchanged. The missing-certificate mass bounds are
0.749050/0.891126/0.921760 (lower/imputed/upper) for the depth router and
0.749825/0.892315/0.922536 for local. Among 105 certified deep
bounded-complete contexts, no-pair/uniform-4/depth-routed/local/uniform-8 mass is
0.775/0.817/0.818/0.825/0.826. Local improves 18 contexts with no losses and
uses 1,370 pair evaluations/context versus 1,641 depth-routed and 3,539
uniform-8.

The linked 20,000-context validation has paired wall/Rust/factor-map/pricing
deltas of -2.5%/-2.7%/-2.9%/+1.0%. On all 81,844 contexts, the two paired
blocks give median deltas of -0.7% wall, -0.4% aggregate Rust, -0.7%
factor-map CPU, and +3.0% pricing CPU. Local evaluates 9.57M pairs versus
7.97M for the depth router, so the promotion clears the alternative success
criterion: a repeatable exact-quality improvement within the current
end-to-end runtime budget, rather than pair-count parity.
