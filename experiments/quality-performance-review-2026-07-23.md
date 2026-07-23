# Quality and performance review, 2026-07-23

The follow-up full-input structural analysis and radical-search experiments
are in `structural-search-review-2026-07-23.md`.

## Outcome

Two changes survive the full review:

1. Use the heuristic proposal pool beyond home-bounded depth 5 with two exact
   continuation states. Against the superseded depth-99 policy, the matched
   global audit improves post-stratified `Mass@10` from 0.832 to 0.839. The
   full 81,844-context workload improves 9.6% in wall time and 7.0% in
   aggregate Rust search time.
2. Use a deterministic integer hasher for the trusted tuple keys in the
   per-context local-score cache. Five 20,000-context release runs improve
   median wall time from 9.246 to 8.898 seconds (-3.8%) and aggregate Rust
   time from 54.240 to 51.594 seconds (-4.9%) with unchanged search work.

The retained policy is not globally exact. It is a better measured point on
the bounded quality/runtime frontier.

## Baseline and failure anatomy

The exact-top-10 audit samples ten contexts in each of 41
depth/anchor/activity-type strata. The oracle certifies 183 of 396 samples,
representing 99.7% of the population by oracle-certifiable stratum. Under the
superseded depth-99 default, 169 certified cases complete in bounded search,
mean conditional `Mass@10` is 0.870, and the post-stratified estimate is
0.832. The seven complete zero-mass cases are not distributed uniformly:
quality falls from 1.000 at depth 2 to 0.936/0.880 at depths 3/4, 0.825 at
depth 5, 0.764 at depth 6, and 0.604 at depth 7.

The ordinary short-context result is stronger than the deep tail. Five
independent 50-context depth-3/4 cohorts at the actual default width 40 score
`Mass@10` 0.853/0.890/0.849/0.899/0.939 (mean 0.886). Their median is usually
exact, but three cohorts still contain at least one zero.

Detailed traces separate several mechanisms:

- Context 45331 is a success case for symmetric reverse guidance:
  `Mass@10=1.000`; the older asymmetric search scored 0.086.
- Context 2679 is a predecessor-sensitive shallow failure. Most target zones
  appear somewhere in both beams, but the coherent target lineage is never
  proposed after the first layer. The winner needs `1035 -> 904 -> 824`;
  unary continuation ranking prefers inferior recombinations.
- Context 944 is a deep support-and-ranking failure. Every exact top-10 plan
  loses its first destination; later target continuations are also pruned.
  Globally widening the heuristic pool can propose some targets but does not
  preserve the complete lineage.
- Flat returned supports are not broad geographic searches. They are
  near-tied recombinations over a few zones and cost more because ambiguous
  reverse messages generate more partial work.
- Oracle state caps are concentrated in contexts with a median of six
  independent variables and a `10^18` assignment lattice. The certified audit
  is population-weighted coverage evidence, not proof for every deep context.

## Literature map

The search is a top-K max-sum problem with ternary chain factors plus repeated
anchor equalities. Several established methods map cleanly to its observed
failure modes:

| Direction | Literature result | Relevance here |
|---|---|---|
| Lazy K-best deviations | [Eppstein's K-shortest-path algorithm](https://epubs.siam.org/doi/10.1137/S0097539795290477) enumerates alternatives after one shortest-path preprocessing pass. [Nilsson's M-most-probable configurations](https://vbn.aau.dk/en/publications/an-efficient-algorithm-for-finding-the-m-most-probable-configurat-2/) applies divide-and-conquer/message passing to graphical models. | Build one compact candidate lattice, then enumerate deviations from strong plans instead of making every top-K plan survive the same beam. Anchor equalities prevent a direct graph reduction, but low-treewidth contexts can use the graphical-model form. |
| Lazy product enumeration | Huang and Chiang's [forest rescoring/cube pruning](https://aclanthology.org/P07-1019/) avoids exhaustively expanding Cartesian products while integrating non-local scores. | Rank `(suffix hypothesis, destination)` combinations lazily rather than splitting a fixed proposal quota uniformly across speculative suffix maps. This targets both missing support and the dominant map-scan cost. |
| Bounded admissible heuristics | [Mini-bucket heuristics](https://arxiv.org/abs/1301.6708) provide a controlled preprocessing/search trade-off; [look-ahead with mini-bucket residuals](https://ics.uci.edu/~dechter/publications/r226.html) applies extra work where a heuristic is weak. | A small pair/anchor-aware mini-bucket message could guide A*/branch-and-bound or trigger selective repair only where the current unary continuation is inconsistent. The failed overlap gate shows that raw candidate disagreement is not a sufficient residual. |
| Best-first bounded expansion | [Best-First Beam Search](https://arxiv.org/abs/2007.03909) shows that expansion order and safe early pruning can reduce beam work substantially when scores admit a suitable monotone bound. | The current layer-synchronous beam fully expands many ambiguous states. A priority queue over partial plans with a valid relaxed suffix bound could spend the same state budget non-uniformly. Utility terms are not naively monotone, so the oracle-style upper bound is required. |
| Explicit diversity | [Diverse Beam Search](https://arxiv.org/abs/1610.02424) prevents a beam from collapsing to near-duplicate sequences. | Diversity should be defined over predecessor/current pairs, anchor assignments, or suffix roots—not Hamming distance between zones. The existing pair-alternative retention is a limited version; a principled group budget may prevent flat-support recombination collapse. |
| Sampling without replacement | [Ancestral Gumbel-Top-k](https://www.jmlr.org/beta/papers/v21/19-985.html) gives exact without-replacement sampling for discrete graphical structures under suitable ancestral access. | Useful if the product goal changes from deterministic top-K recovery to representative draws. It is not a repair for the present top-K oracle metric and should not replace score-ranked search. |

## Experiment ledger

| Experiment | Quality result | Runtime result | Decision |
|---|---|---|---|
| Ordinary continuation width 1 -> 4 | Eight-canary mean `Mass@10` 0.313 -> 0.189; zeroes 2 -> 4 | Not advanced | Reject |
| Depth 99 vs depth 5 / deep width 16 | Global post-stratified mass 0.832 -> 0.858 | Historical width-16 policy cost about 25% aggregate Rust | Reject width 16 |
| Depth 5 / deep width 1 | Global mass 0.832, no gain | Fast | Reject |
| Depth 5 / deep width 2 | Global mass 0.839 | Full wall -9.6%, Rust -7.0%, factor maps -9.1% | Keep |
| Depth 5 / deep width 4 | Global mass 0.841 | 5,000-context Rust +2.6% | Reject; only +0.002 mass over width 2 |
| Reverse heuristic reserve, 8 | Five-cohort mean 0.886 -> 0.895; global 0.832 -> 0.845 | About +15--18% | Reject |
| Reverse reserve disagreement gates | Thresholds <=12 lose the repair; 13+ retain it | Still broad and costly | Reject |
| Trusted local-score hasher | Output/work invariant by construction and tests | 20,000-context wall -3.8%, Rust -4.9% | Keep |
| Trusted factor-map cache hasher | No quality change | Less than 1% difference | Reject |
| Packed local-score key | No quality change | 20,000-context Rust +4.1% | Reject |
| Exact-guidance log gap 0.25 | Canary mean 0.313 -> 0.312; one case regresses | Not advanced | Reject |

## Hypotheses carried into the structural follow-up

These were the ranked directions at the end of the incremental review. The
subsequent data analysis and radical experiments in
`structural-search-review-2026-07-23.md` supersede this ordering.

1. **Lazy multi-suffix candidate enumeration.** Represent each suffix map as a
   sorted stream and use a cube-pruning heap to request the next best distinct
   `(suffix, destination)` proposal. Stop after a global budget or score-gap
   criterion. This is the clearest joint quality/performance hypothesis.
2. **Treewidth-aware exact route.** Build the factor graph after collapsing
   repeated anchors, estimate induced width with min-fill, and use exact
   max-product plus Nilsson-style K-best enumeration only when clique memory is
   below a hard cap. First use it to expand oracle coverage; consider it as a
   production fast path only after exact validation.
3. **Mini-bucket residual trigger.** Compile a cheap pair/anchor-aware relaxed
   message. Spend extra proposals only when its Bellman residual disagrees with
   the unary message. Candidate-set overlap is not a sufficient trigger.
4. **Best-first bounded search on the candidate lattice.** Reuse the oracle's
   admissible relaxed suffix bound, cap live states, and compare mass per
   expansion against the layer beam. This is a larger replacement experiment,
   not a new default path without a cohort win.
5. **Contiguous factor-map kernels.** Profile an SoA representation and
   vectorized feasibility/score passes before changing semantics. Map scans
   remain the dominant cost; hashing their cache keys did not matter.

## Stopping point

Scalar tuning has reached diminishing returns: width 2 is the measured deep
knee, width 4 buys only 0.002 mass, width 16 is too expensive, adaptive
score-band pruning loses mass, and reverse diversity buys quality at a
double-digit runtime cost. The retained cache optimization also reaches its
knee: a cheaper hasher helps, while packing the key and changing factor-map
hashing do not.

Further credible gains require an architectural hypothesis and its own bounded
implementation/audit cycle. It should not be mixed into the current kernel as
another reserve or width option.
