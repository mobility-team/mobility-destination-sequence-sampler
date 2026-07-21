[windows]
set shell := ["powershell.exe", "-NoLogo", "-Command"]

default:
    @just --list

install:
    mamba run -n mobility-destination-sequence-sampler python -m pip install -e .; exit $LASTEXITCODE

build-release:
    mamba run -n mobility-destination-sequence-sampler python -m maturin develop --release; exit $LASTEXITCODE

test:
    mamba run -n mobility-destination-sequence-sampler python -m pytest; exit $LASTEXITCODE

check:
    cargo fmt --check; exit $LASTEXITCODE
    cargo clippy --all-targets -- -D warnings; exit $LASTEXITCODE
    mamba run -n mobility-destination-sequence-sampler python -m pytest; exit $LASTEXITCODE

compare-quality: build-release
    mamba run -n mobility-destination-sequence-sampler python -m experiments.analysis.compare_bidirectional_top_k_grand_geneve --contexts 50 --candidate-contexts 300 --top-k 10 --oracle-depth 100 --max-states 2000000 --frontier-width 32 --proposal-limit-per-source 16 --continuation-state-limit 1 --continuation-proposal-limit 1; exit $LASTEXITCODE

compare-k-sweep seed='42': build-release
    mamba run -n mobility-destination-sequence-sampler python -m experiments.analysis.compare_bidirectional_top_k_grand_geneve --contexts 50 --candidate-contexts 300 --top-k 10 --top-k 20 --top-k 50 --top-k 100 --oracle-depth 500 --max-states 2000000 --frontier-width 128 --proposal-limit-per-source 16 --continuation-state-limit 1 --continuation-proposal-limit 1 --exploration-seed {{seed}}; exit $LASTEXITCODE

compare-k-sweep-seeds: build-release
    mamba run -n mobility-destination-sequence-sampler python -m experiments.analysis.compare_bidirectional_top_k_grand_geneve --contexts 50 --candidate-contexts 300 --top-k 10 --top-k 20 --top-k 50 --top-k 100 --oracle-depth 500 --max-states 2000000 --frontier-width 128 --proposal-limit-per-source 16 --continuation-state-limit 1 --continuation-proposal-limit 1 --exploration-seed 42 --exploration-seed 43 --exploration-seed 44 --exploration-seed 45 --exploration-seed 46; exit $LASTEXITCODE

audit-global-quality per_stratum='1' max_states='200000': build-release
    mamba run -n mobility-destination-sequence-sampler python -m experiments.analysis.compare_bidirectional_top_k_grand_geneve --contexts-per-stratum {{per_stratum}} --top-k 10 --oracle-depth 100 --max-states {{max_states}} --frontier-width 32 --proposal-limit-per-source 16 --continuation-state-limit 1 --continuation-proposal-limit 1; exit $LASTEXITCODE

compare-refresh: build-release
    mamba run -n mobility-destination-sequence-sampler python -m experiments.analysis.compare_seam_refresh --contexts 50 --seam-refresh-per-prefix 0 --seam-refresh-per-prefix 1 --seam-refresh-per-prefix 2 --seam-refresh-per-prefix 4; exit $LASTEXITCODE

benchmark-throughput: build-release
    mamba run -n mobility-destination-sequence-sampler python -m experiments.benchmarks.perf_bidirectional_grand_geneve --contexts 1000 --threads 8 --profile; exit $LASTEXITCODE
