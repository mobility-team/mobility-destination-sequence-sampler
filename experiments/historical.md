# Archived directions

| Direction | Status | Decision |
|---|---|---|
| Particle candidates | Archived | Sequential baseline; does not fix top-K proposal support. |
| Factor-tree/exhaustive sampling | Archived | Useful historical reference, too costly for active search. |
| Second-order aggregation | Archived | Correct research path, not raw-zone speed. |
| Hierarchical travel kernel | Archived | Compression result, not a standalone runtime solution. |
| Exact heap top-K | Retained oracle | Keep for small-case validation only. |
| Single-suffix factor map | Rejected | Collapses proposals onto one speculative continuation. |
| Eight-suffix factor map | Rejected | Splits a fixed candidate budget too thinly. |
| Factor map depth 6 | Rejected | Long-chain guidance is too weak; weighted Mass@10 0.706. |
| Pure partial-map bootstrap | Rejected | One completed neighbouring factor collapses the beam: pilot Mass@10 0.491 vs 0.535 for factor-map. |
| Full/local map union | Rejected | More candidates do not provide a valid continuation rank: Mass@10 0.505 and 5.7x slower than factor-map. |
| Partial-screen map | Rejected | Small speed gain did not justify lower depth-five quality and fewer completed contexts. |
| Root-diverse ping-pong bridge | Rejected | Recovered context 26 (Mass@10 0.733→0.918), but a 15-context stratified audit was identical to baseline at 80% more bounded time (2.84→5.11 ms/context). |
| Pooled suffix factor maps | Rejected | Concentrated support on one suffix: a matched five-context cohort fell from Mass@10 0.899 to 0.800 and slowed 3.04→4.81 ms/context. |
| Independent map-guidance reservoir | Rejected | Recovered context 26 (0.733→0.904), but the 15-context stratified pilot was quality-identical at 3.1x bounded time (2.84→8.85 ms/context). |
| Reverse-prefix compilation/query-cache pilot | Rejected | Median 1,000-context release time improved only 1.3% (0.539 to 0.532 s), while the full workload regressed to 31.146 s and 70,798 complete contexts. |
| Sparse pair-state DP | Rejected | It recovered the rank-one 2679 failure by retaining `(previous, current)` identity, but a 50-context oracle-proven depth-3--5/no-repeated-anchor audit was lower quality than active factor maps (Mass@10 0.695 vs 0.838), slower (6.75 vs 6.67 ms/context), and hit its 100k retained-path cap once. |
| Exact one-layer seed repair | Diagnostic only | It recovered the rank-one 2679 path by freeing the middle layer of a bounded seed, but 27 exact neighbourhood calls took 10.3 s and returned only 1/10 oracle paths; a second 9.5 s pass reached 2/10. Shared-anchor contexts cannot use private activity domains. |
| Four-state ordinary continuation | Rejected | Raising `continuation_state_limit` from 1 to 4 cut the eight-canary mean Mass@10 from 0.313 to 0.189 and doubled zero cases from two to four. |
| Reverse heuristic reserve | Rejected | Eight extra exact-guidance candidates raised five-cohort mean Mass@10 from 0.886 to 0.895 and the global audit from 0.832 to 0.845, but cost roughly 15--18% runtime. Agreement thresholds retained the cost; four candidates lost the repair. |
| Trusted factor-map cache hasher | Rejected | Hashing was not the bottleneck in factor-map lookup; a 20,000-context release comparison changed aggregate Rust time by less than 1%. |
| Packed local-score key | Rejected | Packing the tuple/optional key regressed median aggregate Rust time by 4.1% on 20,000 contexts; keep the tuple with the cheaper trusted hasher. |
| Exact-guidance score band 0.25 | Rejected | The adaptive band did not repair a hard case and lowered context 57725 Mass@10 from 0.217 to 0.204. |
| Exact width-zero decomposition | Rejected | Collapsing anchors, exactly scanning independent unary variables, and lazily combining their top-K choices preserved exact quality, but a full release A/B regressed wall 28.749→30.703 s and aggregate measured phase time about 193.4→201.2 s. The 5,296 routed successful contexts added 5.3M domain evaluations. |
| Width-routed exact best-first search | Rejected | Recomputing 324 cached width-one contexts cost 9.71 ms/context and 147.5M child evaluations; 157 width-two contexts cost 229.74 ms/context and 1.15B children. Raw-zone domain size defeats the favorable treewidth. |
| Returned/internal candidate-lattice closure | Rejected | Exact recombination over returned per-variable zones could raise cached mean Mass@10 only 0.776→0.782. Internal proposed/retained support bounds were 0.800/0.787 overall and only 0.079/0.002 in zero-mass cases. |
| Admissible OD blocks and box search | Rejected | Bounds were admissible but not selective enough, and coarse aggregation changed path ranking. No hierarchy runtime path is retained. |
| Backtracking tickets | Rejected | A one-ticket canary repaired a few near-cutoff prefixes but improved the full oracle cohort by only about one percentage point in best-path recall. |
| Agenda-motif root proposals | Rejected | A warm bounded-result cache had high conditional value but retrieved only 3 of 28 active root misses; its whole-cohort mass potential was too small to justify a cache subsystem. |
| Pricing from depth 3 | Rejected | Improved locked conditional `Mass@10` 0.864->0.878 and post-stratified mass 0.884823->0.898616, but the 20,000-context paired gate regressed wall 7.8%, aggregate Rust 10.4%, and pricing CPU 113.6%. |
| Pair truncation/saturation/non-additivity routers | Rejected | Boundary-gap, saturation, feasible-ratio, and non-additivity rules routed too much work or captured less 8x8 pressure than the Kth-score signal; for example gap<=0.25 cost about 3,094 pair evaluations/context versus 1,690 for the depth router. |
| Zero-margin local pair expansion | Rejected | Working-top-K entry matched nearly all uniform-8 quality, but full-workload pair evaluations rose from 7.97M to 11.27M. |
| Kth-improvement margin 0.1 | Rejected | Preserved the exact repairs and reduced exact-cohort pair work, but missed its immutable 3% wall-regression gate by a fraction on the untouched 20,000-context runtime cohort. |
| Endpoint-aware home-subtour search | Rejected | Independent candidates were merged with the fixed-home boundary factor scored exactly and cross-home anchors routed away. Limit 8 raised certified conditional `Mass@10` only 0.87213->0.87349 and post-stratified mass 0.88742->0.88820 while bounded time rose 5.44->6.26 ms/context; zero overlap was unchanged, and limits 16/32 added no pilot quality. |

Archived code and retired benchmark harnesses are in the
`research-archive-2026-07-21` Git tag; use
`git show research-archive-2026-07-21:<path>` to inspect a file.
The retained oracle is `rust/oracle.rs`. Details and compact lessons
are in [`lessons-learned.md`](lessons-learned.md) and
[`retired-experiment-notes.md`](retired-experiment-notes.md).
