# Measurement guide

## Which harness to run

| Question | Command | Primary result |
|---|---|---|
| Default bounded quality against proof | `just compare-quality` | Recall@10 and Mass@10 |
| Deep primary-quality coverage | `just audit-deep-top-k` | all-depth stratified exact top-10 audit and oracle-cap diagnostics |
| Output-size behavior | `just compare-k-sweep-seeds` | Recall/Mass across K against top-500 support |
| Whole-workload budget | `just benchmark-throughput` | wall time, complete/infeasible contexts, profile counts |
| Pure performance hypothesis | `just compare-throughput` | interleaved A/B/A medians, fingerprint, and counter gate |
| Returned-support concentration | `just diagnose-returned-distribution` | conditional mass@10/20/50 within returned top-100 |
| Symmetric tuning | `just sweep-symmetric` | compact comparison of message/state/proposal widths |
| Regression cases | `just canary-quality` | fixed difficult contexts with cached exact answers |

`BENCHMARKS.md` is the active baseline. A benchmark is comparable only when it
uses the listed workload, release build, and configuration.

## Quality terms

- **Recall@K**: fraction of exact top-K plans recovered.
- **Mass@K**: exact top-K probability mass retained by the bounded result.
- **Mass@500**: bounded result mass in one fixed exact top-500 reference;
  use it to compare differing requested K values.
- **Efficiency**: bounded result mass divided by the fixed reference mass as
  reported by the harness.
- **Complete / infeasible**: contexts that yielded a complete sequence or
  none under the supplied feasibility/scoring inputs; this is not an oracle
  accuracy measure.

The returned-support diagnostic normalizes only over the bounded plans it
received (normally top-100). It is useful for comparing head concentration and
detecting flat returned supports, but it is a lower-bound diagnostic—not an
estimate of the full exp(U) distribution.

The oracle may reject a context at `max_states`. Coverage and exclusions from
that outcome must remain visible; do not silently treat an unproven context as
a bounded miss or hit.

`audit-deep-top-k` deliberately proves only top 10: that is still exact for
Recall@10 and Mass@10, and materially expands deep-context certification. Use
`audit-global-quality` when a top-100 tail support is specifically required.

## Diagnostics that are deliberately not active support analysis

`compare_bidirectional_top_k_grand_geneve.py` can print a **legacy heuristic
trace** and labels such as `inside-legacy-pool; active stage unknown`. These
diagnose the retired heuristic pool only. They do not locate loss within the
active factor-map/symmetric policy and must not be used as direct tuning
evidence.

## Presets and duplication

The Python quality and throughput harnesses both default to the active bounded
configuration. `just` recipes may intentionally override a value for a
historical comparison (for example, width 32 in the K sweep). Treat the
recipe, its arguments, and the benchmark heading as one named preset; record a
new preset in `BENCHMARKS.md` before calling it a new baseline.
## Performance experiment gate

Use `just compare-throughput` for a pure bounded-search performance hypothesis.
It prepares a fixed depth/variable-anchor-stratified cohort once and runs
`A B A B A` in one process. A candidate is promotable only when output
fingerprints and work counters agree and its median aggregate Rust search time
improves by at least 3%. Confirm a promoted candidate with
`just compare-throughput-full` before changing a production default.

Make one low-level hypothesis per commit. For a code change, retain the
baseline commit and run the same harness from each revision; parameter A/B
comparisons can run together in one process via `--candidate-option`.
