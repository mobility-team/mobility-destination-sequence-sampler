"""Persist exact certificates separately from initializer-dependent attempts."""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import polars as pl


ORACLE_CERTIFICATE_VERSION = 3
ORACLE_ATTEMPT_VERSION = 1

_CERTIFICATE_SOURCES = (
    "experiments/benchmarks/perf_grand_geneve_cache.py",
    "rust/oracle.rs",
    "rust/scoring.rs",
    "rust/model.rs",
    "rust/input.rs",
    "rust/output.rs",
)
# `rust/api.rs` deliberately stays out of the certificate identity because it
# also contains bounded-search plumbing. Bump ORACLE_CERTIFICATE_VERSION when
# exact-oracle argument plumbing or result materialization changes there.

_BOUNDED_INITIALIZER_SOURCES = (
    "rust/api.rs",
    "rust/top_k/mod.rs",
    "rust/top_k/stitch.rs",
    "rust/top_k/backward.rs",
    "rust/top_k/forward.rs",
    "rust/top_k/factor_maps.rs",
    "rust/top_k/improvement.rs",
    "rust/top_k/refresh.rs",
    "rust/top_k/candidates.rs",
)


def _update_sources(digest: Any, relative_paths: tuple[str, ...]) -> None:
    root = Path(__file__).resolve().parents[1]
    for relative_path in relative_paths:
        digest.update(relative_path.encode())
        digest.update((root / relative_path).read_bytes())


def oracle_input_fingerprint(
    snapshot_files: dict[str, Path],
    *,
    logit_scale: float,
    update_plan_timings: bool,
    use_shadow_prices: bool,
) -> str:
    """Fingerprint only inputs that can change a completed exact result."""
    digest = hashlib.sha256()
    digest.update(f"oracle-certificate-v{ORACLE_CERTIFICATE_VERSION}".encode())
    digest.update(
        json.dumps(
            {
                "logit_scale": logit_scale,
                "update_plan_timings": update_plan_timings,
                "use_shadow_prices": use_shadow_prices,
            },
            sort_keys=True,
        ).encode()
    )
    for name, path in sorted(snapshot_files.items()):
        digest.update(name.encode())
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    _update_sources(digest, _CERTIFICATE_SOURCES)
    return digest.hexdigest()[:20]


def oracle_attempt_fingerprint(
    certificate_fingerprint: str,
    *,
    use_bounded_incumbent: bool,
) -> str:
    """Identify a resource-limited proof attempt and its bounded initializer."""
    digest = hashlib.sha256()
    digest.update(f"oracle-attempt-v{ORACLE_ATTEMPT_VERSION}".encode())
    digest.update(certificate_fingerprint.encode())
    digest.update(
        json.dumps(
            {"use_bounded_incumbent": use_bounded_incumbent},
            sort_keys=True,
        ).encode()
    )
    if use_bounded_incumbent:
        _update_sources(digest, _BOUNDED_INITIALIZER_SOURCES)
    return digest.hexdigest()[:20]


class OracleCache:
    """Reusable certificates plus attempt-specific reports and failures."""

    def __init__(
        self,
        fingerprint: str,
        oracle_depth: int,
        max_states: int,
        *,
        attempt_fingerprint: str | None = None,
    ) -> None:
        root = Path("experiments/.cache/oracle-top-k")
        self.path = root / "certificates" / fingerprint
        self.attempt_path = (
            root / "attempts" / attempt_fingerprint
            if attempt_fingerprint is not None
            else None
        )
        self.legacy_path = root / fingerprint
        self.fingerprint = fingerprint
        self.attempt_fingerprint = attempt_fingerprint
        self.oracle_depth = oracle_depth
        self.max_states = max_states

    def _certificate_paths(self, context_id: int) -> tuple[Path, Path]:
        stem = f"context-{context_id}-k{self.oracle_depth}"
        return self.path / f"{stem}.parquet", self.path / f"{stem}.certificate.json"

    def _attempt_error_path(self, context_id: int) -> Path | None:
        if self.attempt_path is None:
            return None
        stem = f"context-{context_id}-k{self.oracle_depth}-states-{self.max_states}"
        return self.attempt_path / f"{stem}.error.json"

    def _legacy_paths(self, context_id: int) -> tuple[Path, Path, Path]:
        stem = f"context-{context_id}-k{self.oracle_depth}-states-{self.max_states}"
        return (
            self.legacy_path / f"{stem}.parquet",
            self.legacy_path / f"{stem}.json",
            self.legacy_path / f"{stem}.error.json",
        )

    @staticmethod
    def _atomic_write(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
            temporary.write(payload)
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, path)

    def _load_certificate(
        self,
        context_id: int,
    ) -> tuple[pl.DataFrame, dict[str, int]] | None:
        table_path, metadata_path = self._certificate_paths(context_id)
        if table_path.exists() and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            return pl.read_parquet(table_path), metadata["certifying_report"]
        legacy_table, legacy_report, _ = self._legacy_paths(context_id)
        if legacy_table.exists() and legacy_report.exists():
            return pl.read_parquet(legacy_table), json.loads(legacy_report.read_text())
        return None

    def load_or_compute(
        self,
        context_id: int,
        compute: Callable[[], tuple[pl.DataFrame, dict[str, int]]],
    ) -> tuple[pl.DataFrame, dict[str, int], bool]:
        cached = self._load_certificate(context_id)
        if cached is not None:
            table, report = cached
            return table, report, True
        error_path = self._attempt_error_path(context_id)
        if error_path is not None:
            if error_path.exists():
                raise ValueError(json.loads(error_path.read_text())["error"])
        else:
            _, _, legacy_error = self._legacy_paths(context_id)
            if legacy_error.exists():
                raise ValueError(json.loads(legacy_error.read_text())["error"])

        try:
            oracle_table, oracle_report = compute()
        except ValueError as error:
            if error_path is not None:
                self._atomic_write(
                    error_path,
                    json.dumps(
                        {
                            "error": str(error),
                            "certificate_fingerprint": self.fingerprint,
                            "attempt_fingerprint": self.attempt_fingerprint,
                            "oracle_depth": self.oracle_depth,
                            "max_states": self.max_states,
                        }
                    ).encode(),
                )
            raise
        self.path.mkdir(parents=True, exist_ok=True)
        table_path, metadata_path = self._certificate_paths(context_id)
        with tempfile.NamedTemporaryFile(
            dir=self.path,
            suffix=".parquet",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            oracle_table.write_parquet(temporary_path)
            os.replace(temporary_path, table_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        self._atomic_write(
            metadata_path,
            json.dumps(
                {
                    "certificate_fingerprint": self.fingerprint,
                    "certifying_attempt_fingerprint": self.attempt_fingerprint,
                    "oracle_depth": self.oracle_depth,
                    "certifying_max_states": self.max_states,
                    "certifying_report": oracle_report,
                }
            ).encode(),
        )
        if error_path is not None:
            error_path.unlink(missing_ok=True)
        return oracle_table, oracle_report, False

    def load_cached(self, context_id: int) -> tuple[pl.DataFrame, dict[str, int]] | None:
        """Return an existing certificate without creating cache state."""
        cached = self._load_certificate(context_id)
        if cached is not None:
            return cached
        error_path = self._attempt_error_path(context_id)
        if error_path is not None:
            if error_path.exists():
                raise ValueError(json.loads(error_path.read_text())["error"])
        return None

    def cached_context_ids(self) -> list[int]:
        """List certificate IDs without relying on attempt-specific filenames."""
        current = {
            int(path.name.split("-", 2)[1])
            for path in self.path.glob(
                f"context-*-k{self.oracle_depth}.parquet"
            )
        }
        legacy = {
            int(path.name.split("-", 2)[1])
            for path in self.legacy_path.glob(
                f"context-*-k{self.oracle_depth}-states-{self.max_states}.parquet"
            )
        }
        return sorted(current | legacy)
