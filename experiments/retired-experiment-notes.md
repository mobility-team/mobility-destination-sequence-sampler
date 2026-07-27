# Retired experiment notes

This is the compact record for the 2026-07-27 exploration cleanup. The
maintained implementation remains bounded bidirectional top-K plus the exact
oracle. None of the mechanisms below is in the runtime path.

| Direction | What the evidence showed | Decision |
|---|---|---|
| OD blocks and box search | Local bounds were admissible on synthetic, factor, and cached-plan audits, but fixed 16/32/64/128-cell cuts had p90 top-16 volumes of 1,110/894/575/335 zones, above the 320-zone selectivity gate. Literal coarse aggregation also changed oracle block-path rankings. | Reject hierarchy/box runtime work. |
| Wider exact reverse guidance | It causally repaired context 26 only at a 512-state channel: about 243 ms and 22.2M factor-map evaluations versus 5.8 ms and 161k at default. | Reject brute-force guidance widening. |
| Destination-distinct partial anchors | It let context 26's root `1081` enter, but by itself was slightly worse than the active baseline in a 300-context audit. | Do not promote independently. |
| Parent-local forward tickets | One ticket repaired context 26 and changed a compact-anchor 300-context audit from 220 to 224 certified best paths and 0.7777 to 0.7873 mean top-10 mass. Against active default the gain was only 221 to 224 paths and 0.7803 to 0.7873; measured internal cost was roughly 0.3%. | Retire: real but too small for added policy complexity. |
| Aggregated factor-map messages | Normalized accumulation across retained messages created no new oracle root in the top 16 of 157 factor-map-traced roots. Context 26's root was not in that channel at all. | Reject message accumulation. |
| Truncated continuation choice model | A learned correction slightly reranked an oracle-selected head, but did not learn the omitted continuation residual and had no full-domain evidence. | Reject as a proposal source. |
| Agenda motifs, oracle-labelled cache | A warm leave-one-out cache retrieved 5 of 28 active root misses and raised target-hit trial mass from 0.056 to 0.340, but returned only two complete best paths. | Oracle-labelled result is an upper bound, not a deployable cache. |
| Agenda motifs, bounded-labelled cache | With other contexts labelled by ordinary bounded results, a fully warm 393-context cache retrieved 3 of 28 root misses, restored one best path, and raised trial mass from 0.171 to 0.226. Its estimated full-cohort mass potential was about 0.0044; a real online cache would start colder. | Reject cache lifecycle/routing work. |

The common lesson is that several mechanisms can repair a concrete trace, but
none produced a material whole-cohort improvement. Future work should begin
with a new hypothesis and a small passive recall/quality gate; do not revive a
retired mechanism merely because it fixes an individual deep context.
