# Measurement guide

## Which harness to run

| Question | Command | Primary result |
|---|---|---|
| Default bounded quality against proof | `just compare-quality` | Recall@10 and Mass@10 |
| Deep primary-quality coverage | `just audit-deep-top-k` | all-depth stratified exact top-10 audit and oracle-cap diagnostics |
| Output-size behavior | `just compare-k-sweep-seeds` | Recall/Mass across K against top-500 support |
| Representative throughput smoke | `just benchmark-throughput` | 1,000 depth/anchor-calibrated contexts; wall time and profile counts |
| Fixed-only throughput smoke | `just benchmark-throughput-fixed-only` | anchor-independent regression signal |
| Fast candidate quality screen | `just explore-quality NAME=VALUE` | in-process A/B over a small stratified exact cohort |
| Fast candidate runtime screen | `just explore-throughput NAME=VALUE` | one counterbalanced block over 1,000 contexts |
| Exploratory performance hypothesis | `just compare-throughput` | counterbalanced paired deltas, observed block range, fingerprints, and counters |
| Decision-grade performance or quality/runtime gate | `just compare-throughput-manifest <manifest>` | typed verdict plus a self-contained run artifact |
| Decision-grade exact quality gate | `just compare-quality-manifest <manifest>` | locked-cohort A/B quality verdict and missing-oracle interval |
| Returned-support concentration | `just diagnose-returned-distribution` | conditional mass@10/20/50 within returned top-100 |
| Symmetric tuning | `just sweep-symmetric` | compact comparison of message/state/proposal widths |
| Plan-improvement router | `just evaluate-pricing-router` | first/second-pass marginal quality and router capture |
| Regression cases | `just canary-quality` | fixed difficult contexts with cached exact answers |

`BENCHMARKS.md` is the active baseline. A benchmark is comparable only when it
uses the listed workload, release build, and configuration.

## Quality terms

- **Recall@K**: fraction of exact top-K plans recovered.
- **Mass@K**: probability mass of exact top-K plans retained by the bounded
  result, normalized within those K reference plans (not the full feasible
  distribution).
- **Mass@500**: bounded result mass in one fixed exact top-500 reference;
  use it to compare differing requested K values.
- **Efficiency**: bounded result mass divided by the fixed reference mass as
  reported by the harness.
- **Complete / without plan**: contexts for which bounded search returned a
  complete sequence or returned none. "Without plan" is not proof of model
  infeasibility and is not an oracle accuracy measure.

The returned-support diagnostic normalizes only over the bounded plans it
received (normally top-100). It is useful for comparing head concentration and
detecting flat returned supports, but it is a lower-bound diagnostic—not an
estimate of the full exp(U) distribution.

The oracle reports `proof incomplete` when it reaches `max_states`. Coverage
and exclusions from that outcome must remain visible; do not silently treat an
unproven context as a bounded miss or hit.

The oracle cache has two identities. Completed exact certificates depend only
on inputs plus exact feasibility/scoring/oracle semantics and remain reusable
across bounded-search edits. Resource-limit failures live
under an attempt identity that also fingerprints the bounded initializer.

`audit-deep-top-k` deliberately proves only top 10: that is still exact for
Recall@10 and Mass@10, and materially expands deep-context certification. Use
`audit-global-quality` when a top-100 tail support is specifically required.

## One-context diagnosis

Use `just explain-context <context_id>` after reproducing a miss. It compares
the bounded result with a cached exact answer and traces each exact target
through the active proposal, coherent-prefix, reverse-guidance, and retention
stages. The old heuristic-pool diagnosis was removed because it could not
locate losses in the active factor-map search.

## Discovery, validation, and immutable configurations

Before implementing a search heuristic, pass this admission gate:

- show the mechanism in representative failures and estimate its quality/work
  upper bound;
- check `experiments/historical.md` for overlap with an existing channel;
- prefer a read-only diagnostic, and stop unless the estimated gain is
  material;
- treat another beam/candidate heuristic as marginal until measured otherwise;
- label unmeasured ideas as speculative. Fail-fast execution does not justify
  optimistic framing.

Keep discovery cheap: use `explore-quality` and `explore-throughput` with one
`NAME=VALUE` change. They prepare inputs once and compare A/B in the same
process; exact certificates are reused. Canaries and command-line overrides
are discovery tools, not promotion evidence. Only candidates that survive both
screens need a checked manifest under `experiments/manifests/`. The manifest
snapshots every A/B top-K option, declares the only permitted differences, and
records the hypothesis, mechanism, falsifier, important unknowns, cohort role,
and gates.

Use separate cohorts:

- `canary`: debugging and regression reproduction only;
- `discovery`: mechanism and parameter selection;
- `validation`: an untouched cohort with `expected_fingerprint` locked before
  the decision run.

Create a draft with `just experiment-new <name> <kind> [NAME=VALUE]`, fill its
TODO fields, and run `just experiment-validate <path>`. A manifest-driven run
writes `manifest.json`, resolved A/B configs, source and cohort fingerprints,
line-buffered progress, environment metadata, and `result.json` under
`experiments/runs/`.

Quality/runtime decisions use two cohorts. First run a `quality_only` manifest
on the stratified exact cohort. Then link that artifact's `result.json` from a
`quality_runtime` throughput manifest with identical resolved A/B configs.
The throughput harness refuses evidence produced by different configs and
reports `INCOMPLETE` when quality evidence is absent.

## Performance experiment gate

Use `just compare-throughput` for exploratory pure-performance checks. It uses
the same depth/variable-anchor stratification as the representative throughput
smoke and alternates counterbalanced `ABBA` / `BAAB` blocks. It reports paired
block deltas and their observed range. A pure-performance
candidate passes only when output fingerprints and work counters agree and its
aggregate Rust search time clears the declared improvement gate.

For an exploratory output-changing run, `--allow-output-change` reports only
the runtime component and therefore ends `INCOMPLETE`, not "do not promote".
A decision requires a `quality_runtime` manifest and linked passing quality
artifact. Pricing and factor-map counters may change because exact maps are
reused by the completed-path pass.

Make one low-level hypothesis per commit. For a code change, retain the
baseline commit and run the same harness from each revision; parameter A/B
comparisons can run together in one process via `--candidate-option`.

## Missing exact certificates

The global audit prints three values for each mass metric:

- lower: unresolved sampled contexts contribute zero;
- imputed: solved-context means are applied within each stratum;
- upper: unresolved sampled contexts contribute one.

It also ranks strata by their maximum contribution to the global interval.
Use that list to allocate the next exact-search budget. The imputed point alone
is not sufficient evidence when the interval remains decision-relevant.
