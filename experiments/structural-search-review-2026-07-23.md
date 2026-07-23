# Structural search review, 2026-07-23

## Outcome

The prepared workload is not fundamentally a sparse-graph problem. It is a
small-factor-graph, large-domain problem:

- the OD graph has 1,110 zones and 1,070,826 edges (86.9% density);
- non-home activity domains contain 759--1,110 zones;
- a context has only 2.17 collapsed destination variables on average and at
  most 9;
- greedy induced width is 0/1/2/3 for
  30,475/37,797/13,269/303 contexts.

Low induced width does not by itself make exact inference cheap because a
width-one message can still require about one million zone pairs. Direct
structure-aware exact methods were slower than the bounded search, while the
hard cases need a wider *value* lattice than the current beams retain. Stop
direct width-specialization work here. The next credible radical direction is
coarse-to-fine AND/OR search with admissible bounds over blocks of zones, so
both variables and their roughly 1,000 values are explored lazily.

The reproducible diagnostic is:

```text
mamba run -n mobility-destination-sequence-sampler python \
  -m experiments.analysis.analyze_problem_structure \
  --output-contexts 10000 --top-k 10 --threads 8
```

Add `--benchmark-exact-width 1` or `2` to recompute the cached exact class
rather than merely reading its certificates.

## Input structure

### OD and destination lattice

| Measure | Result |
|---|---:|
| Zones | 1,110 |
| OD pairs | 1,070,826 |
| OD density | 86.9% |
| Median / p90 outgoing degree | 982 / 1,110 |
| Destination-domain median | 1,015 |
| Work / studies / shopping / leisure / other domains | 1,110 / 759 / 964 / 1,015 / 1,104 |

Activity domains overlap heavily: pairwise Jaccard overlap ranges from 68.1%
to 99.5%. Sparse graph traversal and activity-domain partitioning therefore
do not remove much raw work.

### Context factor graphs

Repeated anchors were collapsed to one variable. A local factor connects the
distinct variables appearing at the previous, current, and next layers; fixed
home layers are constants, not assumed separators.

| Measure | Result |
|---|---:|
| Prepared contexts / steps | 81,844 / 328,197 |
| Depth range | 2--10 |
| Collapsed variables, mean / p50 / p90 / max | 2.17 / 2 / 4 / 9 |
| Longest home-bounded variable tour, mean / p50 / p90 / max | 1.78 / 1 / 3 / 9 |
| Repeated-anchor contexts | 15,239 (18.6%) |
| Cross-home-anchor contexts | 8,692 (10.6%) |
| Width 0 / 1 / 2 / 3 | 30,475 / 37,797 / 13,269 / 303 |

Conditioning every variable anchor reduces the width distribution to
46,683/27,748/7,413 at widths 0/1/2, but the remaining clique tables still
have a p90 size around `10^6` assignments. The unconstrained p90 clique is
around `10^9`.

Structural patterns repeat (3,286 activity/anchor/home patterns, 24.9x reuse),
but exact home-independent timing/scoring shapes repeat only 4.0x. Earlier
factor-profile measurement is weaker still: 95.5% of adjacent profiles are
singletons. This argues against a large exact-factor memoization layer.

## Where exact outputs are located

There are 573 distinct current-fingerprint exact top-10 certificates in the
local cache. Their 17,610 variable rows show:

| Rank of chosen zone | p50 | p90 |
|---|---:|---:|
| Travel-time rank from home | 19 | 254 |
| Cost rank from home | 15 | 163 |
| Travel-time rank from preceding stop | 3 | 122 |
| Cost rank from preceding stop | 3 | 77 |
| Static `log(capacity) + shadow_price` rank | 61 | 514 |

Choices are local to the preceding stop much more often than to home, but the
tail is real. A simple union of inbound-near and static indices covers exact
variable rows well; coherent full-plan coverage is lower:

| Per-index N | Exact top-10 mass whose every transition is covered |
|---:|---:|
| 16 | 0.679 |
| 32 | 0.835 |
| 64 | 0.925 |
| 128 | 0.973 |

Using the production heuristic's actual two sources (inbound cost and
`log(capacity) + shadow_price`) gives 0.649/0.812/0.915/0.972 at the same
widths. A width-64 transition lattice has attractive quality potential, but
materializing all pair states was already measured as slower and worse than
the current search. It needs lazy block/value expansion, not another dense
dynamic program.

## Success and failure anatomy

On the 573 cached exact contexts, the current public defaults give mean
Mass@10 0.776 and find the exact winner in 80.6% of contexts.

| Outcome | Contexts | Mean layers | Variables | Width | Longest tour | Internal proposed-support mass |
|---|---:|---:|---:|---:|---:|---:|
| Full Mass@10 | 354 | 4.43 | 2.27 | 0.85 | 1.55 | 0.999 |
| Partial Mass@10 | 146 | 5.38 | 3.34 | 1.42 | 3.06 | 0.680 |
| Zero Mass@10 | 73 | 6.95 | 4.68 | 1.75 | 3.77 | 0.079 |

Quality by induced width is 1.000/0.850/0.490 for widths 0/1/2. By longest
home-bounded tour it falls from 0.977 at one variable stop to 0.820/0.567/
0.463 at two/three/four stops.

This is not merely an output-recombination defect. Recombining all zones
present in completed plans raises mean Mass@10 only 0.776 to 0.782. Even the
larger internal proposed support contains only 0.079 mass in zero cases. A
new inference kernel must introduce values that the active beam never
proposes under a coherent lineage.

## Radical experiments

### Exact width-zero decomposition: rejected

An experimental top-K path collapsed anchors, verified that every factor
scope contained at most one variable, scanned each independent variable
domain exactly, and lazily enumerated Cartesian combinations. It matched the
exact oracle on a focused multi-variable test and kept cached width-zero
Mass@10 at 1.000.

Full 81,844-context release A/B:

| Variant | Wall | Aggregate measured phase time | Result |
|---|---:|---:|---|
| Existing bounded search | 28.749 s | about 193.4 s | keep |
| Exact width-zero decomposition | 30.703 s | about 201.2 s | reject |

It routed 5,296 successful multi-step contexts but added about 5.3 million
full-domain evaluations. The easy width-zero cases were already exact and
cheaper through the bounded path. The implementation and its temporary test
were removed.

### Exact best-first search as a width router: rejected

The exact oracle is an admissible best-first proxy for m-A*/AND-OR inference.
Recomputing cached classes at `max_states=2,000,000` gave:

| Class | Contexts | Wall | ms/context | Children considered |
|---|---:|---:|---:|---:|
| Width 1 | 324 | 3.147 s | 9.71 | 147,506,816 |
| Width 2 | 157 | 36.070 s | 229.74 | 1,151,453,938 |

The bounded comparison of all 573 contexts takes about 0.31 s total. A direct
best-first/variable-elimination production route is therefore roughly 18x
slower at width one and over 400x slower at width two. Low treewidth without
lazy value-domain bounds is not a viable production replacement.

### Candidate-lattice closure: rejected as a standalone repair

The union of destinations in the returned top-10 produces a tiny assignment
lattice (median 18, p90 72 assignments), but its best possible Mass@10 is only
0.782. Internal retained/proposed supports reach 0.787/0.800 overall and
almost nothing in zero cases. Exact recombination after the current search
cannot repair the main failures.

## Literature-to-design map

- Variable elimination is controlled by induced width, but its complexity is
  exponential in width and domain size. The measured width is favorable; the
  measured 759--1,110-value domains are not. See Peyrard et al.,
  [Exact and approximate inference in graphical models](https://arxiv.org/abs/1506.08544).
- AND/OR search merges identical conditioned subproblems and scales with
  induced width. It maps naturally to the collapsed anchor graph, but only
  after value expansion becomes lazy. See Marinescu and Dechter,
  [Best-First AND/OR Search for Graphical Models](https://cdn.aaai.org/AAAI/2007/AAAI07-186.pdf).
- m-A* and m-best branch-and-bound provide the correct top-K search semantics;
  stronger admissible heuristics are the decisive lever. See Dechter,
  Flerova, and Marinescu,
  [Search Algorithms for m Best Solutions for Graphical Models](https://doi.org/10.1609/aaai.v26i1.8405).
- Lazy k-best extraction avoids materializing Cartesian products after a
  compact packed representation exists. It helps *after* block/domain search,
  not before. See Huang and Chiang,
  [Better k-best Parsing](https://aclanthology.org/W05-1506/).

## Next research program

The only remaining high-upside search redesign is **admissible coarse-to-fine
value search**:

1. Partition each activity domain into OD-coherent blocks. Use existing zone
   geography if the upstream boundary can supply it; otherwise derive a
   deterministic cost/time embedding from the OD matrix.
2. Precompute conservative block bounds: minimum inbound/outbound cost and
   time, maximum attraction terms, and interval-safe duration utility. The
   sign of the shadow-price-adjusted activity coefficient must be handled
   explicitly.
3. Run m-A*/AND-OR over the collapsed variable graph and block assignments.
   Refine only the best block cells to zones. Cache by graphical-model
   context, not by layer beam lineage.
4. Extract the top-K zone plans lazily from the refined packed forest and
   score every completed plan with the shared exact scorer.
5. Gate the prototype on the 573 cached certificates: target Mass@10 at least
   0.90 below 1 ms/context, then validate the global weighted audit and full
   throughput. Reject it if safe block bounds are too loose to beat raw scans.

The staged implementation brief, exact bound derivation, acceptance gates,
and stop conditions are in
[`admissible-block-search-experiment.md`](admissible-block-search-experiment.md).
Its diagnostic Phase A is a hard gate before production search work.

This differs from the archived hierarchical kernel: that kernel approximated
far-field scores and lost local accuracy. The proposed hierarchy is used only
for admissible pruning; final scores remain exact. It is also a larger
data-preparation project because robust spatial blocks are not present in the
current prepared tables. The first diagnostic can derive them from the OD
matrix without changing the input contract.

A secondary performance-only direction is batching factor-map scans across
the roughly 74 contexts per home zone (or on GPU/SIMD). It does not address
quality and exact factor profiles have little reuse, so it ranks below the
block-bound search.

## Stop decision

Direct full-domain decomposition, exact best-first routing, sparse pair-state
DP, output recombination, and global width increases have now all reached
diminishing returns or clear regressions. No additional production search
change is justified from the present input contract. Keep the existing
bounded defaults and the previously accepted cache/performance changes; use
the diagnostic gate in `admissible-block-search-experiment.md` to decide
whether a separate block-search prototype is justified.
