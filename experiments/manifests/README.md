# Experiment manifests

Checked-in manifests are immutable experimental definitions, not moving
presets. Each one snapshots every bounded-search option for both A and B,
declares the only allowed differences, identifies its cohort role, and states
the decision gates before a run starts.

Create a draft with:

```text
python -m experiments.experiment new experiments/manifests/my-test.toml \
  --id my-test --kind pure_perf --change frontier_width=48
```

Fill the hypothesis, mechanism, falsifier, and unknowns, then validate it with
`python -m experiments.experiment validate <path>`. Validation-stage manifests
also require the expected cohort fingerprint. Generated run artifacts live
under `experiments/runs/` and are deliberately ignored by Git.
