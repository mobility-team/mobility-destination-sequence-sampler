# Releasing

## Initial setup

These steps are required once for this repository:

1. Create a protected GitHub environment named `pypi`.
2. On PyPI, configure a pending Trusted Publisher with:
   - PyPI project name: `mobility-destination-sequence-sampler`
   - GitHub owner: `mobility-team`
   - GitHub repository: `mobility-destination-sequence-sampler`
   - Workflow: `wheels.yml`
   - Environment: `pypi`
3. Ensure the workflow in
   [`.github/workflows/wheels.yml`](.github/workflows/wheels.yml) is enabled.

The pending publisher creates the PyPI project during the first successful
release. It does not reserve the project name before then.

## Per release

1. Update the version in both `pyproject.toml` and `Cargo.toml`.
2. Commit and merge the version change to `main`.
3. Create a matching version tag, such as `v0.1.0`, on that `main` commit.
4. Push the tag.
5. Let GitHub Actions test the package, build wheels and the source
   distribution, and publish them to PyPI.
6. Verify the published files and metadata on PyPI.

Do not create the release tag before its commit is on `main`.
