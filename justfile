[windows]
set shell := ["powershell.exe", "-NoLogo", "-Command"]

default:
    @just --list

install:
    mamba run -n mobility-destination-sequence-sampler python -m pip install -e .; exit $LASTEXITCODE

build-release:
    @cargo build --release --features pyo3/extension-module; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; Copy-Item -LiteralPath target\release\mobility_destination_sequence_sampler.dll -Destination src\mobility_destination_sequence_sampler\_core.pyd -Force

# Build an optimized extension quickly; timings are not production benchmarks.
build-fast:
    @cargo build --profile experiment --features pyo3/extension-module; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; Copy-Item -LiteralPath target\experiment\mobility_destination_sequence_sampler.dll -Destination src\mobility_destination_sequence_sampler\_core.pyd -Force

test:
    mamba run -n mobility-destination-sequence-sampler python -m pytest; exit $LASTEXITCODE

check:
    cargo fmt --check; exit $LASTEXITCODE
    cargo clippy --all-targets -- -D warnings; exit $LASTEXITCODE
    just build-release; exit $LASTEXITCODE
    mamba run -n mobility-destination-sequence-sampler python -m pytest; exit $LASTEXITCODE

compare-quality: build-release
    mamba run -n mobility-destination-sequence-sampler python -m experiments.analysis.compare_bidirectional_top_k_grand_geneve --contexts 50 --candidate-contexts 300 --top-k 10 --oracle-depth 100 --max-states 2000000 --frontier-width 32 --proposal-limit-per-source 16 --continuation-state-limit 1 --continuation-proposal-limit 1; exit $LASTEXITCODE

compare-k-sweep seed='42': build-release
    mamba run -n mobility-destination-sequence-sampler python -m experiments.analysis.compare_bidirectional_top_k_grand_geneve --contexts 50 --candidate-contexts 300 --top-k 10 --top-k 20 --top-k 50 --top-k 100 --oracle-depth 500 --max-states 2000000 --frontier-width 128 --proposal-limit-per-source 16 --continuation-state-limit 1 --continuation-proposal-limit 1 --exploration-seed {{seed}}; exit $LASTEXITCODE

compare-k-sweep-seeds: build-release
    mamba run -n mobility-destination-sequence-sampler python -m experiments.analysis.compare_bidirectional_top_k_grand_geneve --contexts 50 --candidate-contexts 300 --top-k 10 --top-k 20 --top-k 50 --top-k 100 --oracle-depth 500 --max-states 2000000 --frontier-width 128 --proposal-limit-per-source 16 --continuation-state-limit 1 --continuation-proposal-limit 1 --exploration-seed 42 --exploration-seed 43 --exploration-seed 44 --exploration-seed 45 --exploration-seed 46; exit $LASTEXITCODE

audit-global-quality per_stratum='1' max_states='200000': build-release
    mamba run -n mobility-destination-sequence-sampler python -m experiments.analysis.compare_bidirectional_top_k_grand_geneve --contexts-per-stratum {{per_stratum}} --top-k 10 --oracle-depth 100 --max-states {{max_states}} --frontier-width 32 --proposal-limit-per-source 16 --continuation-state-limit 1 --continuation-proposal-limit 1; exit $LASTEXITCODE

# Broader exact coverage for the primary Mass@10 metric; top-100 remains the tail audit.
audit-deep-top-k per_stratum='2' max_states='500000': build-release
    mamba run -n mobility-destination-sequence-sampler python -m experiments.analysis.compare_bidirectional_top_k_grand_geneve --contexts-per-stratum {{per_stratum}} --top-k 10 --oracle-depth 10 --max-states {{max_states}} --frontier-width 32 --proposal-limit-per-source 16 --continuation-state-limit 1 --continuation-proposal-limit 1; exit $LASTEXITCODE

# Depth-resolved exact top-10 audit. Use a larger cached sample for deep decisions.
audit-deep-by-depth per_stratum='10' max_states='500000': build-release
    mamba run -n mobility-destination-sequence-sampler python -m experiments.analysis.compare_bidirectional_top_k_grand_geneve --contexts-per-stratum {{per_stratum}} --top-k 10 --oracle-depth 10 --max-states {{max_states}} --frontier-width 32 --proposal-limit-per-source 16 --continuation-state-limit 1 --continuation-proposal-limit 1; exit $LASTEXITCODE

compare-refresh: build-release
    mamba run -n mobility-destination-sequence-sampler python -m experiments.analysis.compare_seam_refresh --contexts 50 --seam-refresh-per-prefix 0 --seam-refresh-per-prefix 1 --seam-refresh-per-prefix 2 --seam-refresh-per-prefix 4; exit $LASTEXITCODE

benchmark-throughput: build-release
    mamba run -n mobility-destination-sequence-sampler python -m experiments.benchmarks.perf_bidirectional_grand_geneve --contexts 1000 --calibrated --threads 8 --profile; exit $LASTEXITCODE

# Fixed-only smoke sample; use this to isolate anchor-independent regressions.
benchmark-throughput-fixed-only: build-release
    mamba run -n mobility-destination-sequence-sampler python -m experiments.benchmarks.perf_bidirectional_grand_geneve --contexts 1000 --threads 8 --profile; exit $LASTEXITCODE

# Counterbalanced timing for an exploratory parameterized bounded-search experiment.
compare-throughput: build-release
    mamba run -n mobility-destination-sequence-sampler python -m experiments.benchmarks.compare_bidirectional_throughput --contexts 5000 --calibrated --threads 8 --cycles 1; exit $LASTEXITCODE

compare-throughput-full: build-release
    mamba run -n mobility-destination-sequence-sampler python -m experiments.benchmarks.compare_bidirectional_throughput --all-supported --contexts 81844 --threads 8 --cycles 2; exit $LASTEXITCODE

# Decision-grade runs use immutable configs, typed gates, cohort locks, and artifacts.
compare-throughput-manifest manifest: build-release
    mamba run -n mobility-destination-sequence-sampler python -m experiments.benchmarks.compare_bidirectional_throughput --experiment-manifest {{manifest}} --require-promotion; exit $LASTEXITCODE

compare-quality-manifest manifest: build-release
    mamba run -n mobility-destination-sequence-sampler python -m experiments.analysis.compare_bidirectional_top_k_grand_geneve --experiment-manifest {{manifest}} --top-k 10 --oracle-depth 10 --max-states 500000; exit $LASTEXITCODE

experiment-new name kind change='':
    @if ("{{change}}" -eq "") { mamba run -n mobility-destination-sequence-sampler python -m experiments.experiment new experiments/manifests/{{name}}.toml --id {{name}} --kind {{kind}} } else { mamba run -n mobility-destination-sequence-sampler python -m experiments.experiment new experiments/manifests/{{name}}.toml --id {{name}} --kind {{kind}} --change "{{change}}" }; exit $LASTEXITCODE

experiment-validate manifest:
    mamba run -n mobility-destination-sequence-sampler python -m experiments.experiment validate {{manifest}}; exit $LASTEXITCODE

# Fast candidate screens: one prepared process, one A/B change, small cohorts.
explore-quality change per_stratum='2' max_states='500000': build-fast
    mamba run -n mobility-destination-sequence-sampler python -m experiments.analysis.compare_bidirectional_top_k_grand_geneve --contexts-per-stratum {{per_stratum}} --top-k 10 --oracle-depth 10 --max-states {{max_states}} --compact --candidate-option "{{change}}"; exit $LASTEXITCODE

explore-throughput change contexts='1000': build-release
    mamba run -n mobility-destination-sequence-sampler python -m experiments.benchmarks.compare_bidirectional_throughput --contexts {{contexts}} --calibrated --threads 8 --cycles 1 --allow-output-change --candidate-option "{{change}}"; exit $LASTEXITCODE

# Returned top-100 concentration; all mass is conditional on returned support.
diagnose-returned-distribution contexts='1000': build-release
    mamba run -n mobility-destination-sequence-sampler python -m experiments.analysis.diagnose_returned_distribution --contexts {{contexts}} --top-k 100 --frontier-width 128 --threads 8; exit $LASTEXITCODE

# Compare the flattest returned top-100 supports against concentrated controls.
analyze-flat-support contexts='1000' cases='5': build-release
    mamba run -n mobility-destination-sequence-sampler python -m experiments.analysis.analyze_flat_returned_support --contexts {{contexts}} --cases {{cases}} --top-k 100 --frontier-width 128 --threads 8; exit $LASTEXITCODE

# Full-input factor-graph structure and located top-K output analysis.
analyze-problem-structure output_contexts='10000': build-release
    mamba run -n mobility-destination-sequence-sampler python -m experiments.analysis.analyze_problem_structure --output-contexts {{output_contexts}} --top-k 10 --threads 8; exit $LASTEXITCODE

# Compare the main symmetric quality/runtime knobs in one prepared process.
sweep-symmetric: build-fast
    @mamba run -n mobility-destination-sequence-sampler python -m experiments.analysis.compare_bidirectional_top_k_grand_geneve --contexts 50 --candidate-contexts 300 --top-k 10 --oracle-depth 100 --max-states 2000000 --compact --symmetric-config p8:4:4:8 --symmetric-config p12:4:4:12 --symmetric-config p16:4:4:16 --symmetric-config m8:8:4:8; exit $LASTEXITCODE

# Run the fixed difficult/regression contexts with cached exact oracles.
canary-quality: build-fast
    @mamba run -n mobility-destination-sequence-sampler python -m experiments.analysis.compare_bidirectional_top_k_grand_geneve --top-k 10 --oracle-depth 100 --max-states 2000000 --compact --context-id 26 --context-id 45331 --context-id 3647 --context-id 2679 --context-id 61440 --context-id 3506 --context-id 57725 --context-id 61662; exit $LASTEXITCODE

# Populate the exact audit, then evaluate pricing-pass gains and observable routers.
evaluate-pricing-router: build-release
    @just audit-deep-by-depth 10 500000; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    @mamba run -n mobility-destination-sequence-sampler python -m experiments.analysis.evaluate_pricing_router --top-k 10 --max-states 500000 --pair-limit 4 --pair-limit 8; exit $LASTEXITCODE

# Run bounded and exact calls locally, returning only a compact JSON report.
code-mode-probe context_id='26': build-release
    @mamba run -n mobility-destination-sequence-sampler python -m experiments.analysis.code_mode_probe --context-id {{context_id}}; exit $LASTEXITCODE
