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
