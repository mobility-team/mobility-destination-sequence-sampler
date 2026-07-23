"""Fingerprint and persist exact top-K certificates for experiment harnesses."""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
import os
from pathlib import Path
import tempfile

import polars as pl


ORACLE_CACHE_VERSION = 1


def oracle_input_fingerprint(
    snapshot_files: dict[str, Path],
    *,
    logit_scale: float,
    update_plan_timings: bool,
    use_shadow_prices: bool,
) -> str:
    """Fingerprint every input that can change an exact certificate."""
    digest = hashlib.sha256()
    digest.update(f"oracle-cache-v{ORACLE_CACHE_VERSION}".encode())
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
    root = Path(__file__).resolve().parents[1]
    for relative_path in (
        "experiments/benchmarks/perf_grand_geneve_cache.py",
        "rust/oracle.rs",
        "rust/scoring.rs",
        "rust/model.rs",
        "rust/input.rs",
    ):
        digest.update((root / relative_path).read_bytes())
    return digest.hexdigest()[:20]


class OracleCache:
    """Persistent exact results; fingerprints prevent stale proof reuse."""

    def __init__(self, fingerprint: str, oracle_depth: int, max_states: int) -> None:
        self.path = Path("experiments/.cache/oracle-top-k") / fingerprint
        self.oracle_depth = oracle_depth
        self.max_states = max_states

    def _paths(self, context_id: int) -> tuple[Path, Path, Path]:
        stem = f"context-{context_id}-k{self.oracle_depth}-states-{self.max_states}"
        return (
            self.path / f"{stem}.parquet",
            self.path / f"{stem}.json",
            self.path / f"{stem}.error.json",
        )

    def load_or_compute(
        self,
        context_id: int,
        compute: Callable[[], tuple[pl.DataFrame, dict[str, int]]],
    ) -> tuple[pl.DataFrame, dict[str, int], bool]:
        table_path, report_path, error_path = self._paths(context_id)
        if table_path.exists() and report_path.exists():
            return pl.read_parquet(table_path), json.loads(report_path.read_text()), True
        if error_path.exists():
            raise ValueError(json.loads(error_path.read_text())["error"])
        self.path.mkdir(parents=True, exist_ok=True)

        def atomic_write(path: Path, payload: bytes) -> None:
            with tempfile.NamedTemporaryFile(dir=self.path, delete=False) as temporary:
                temporary.write(payload)
                temporary_path = Path(temporary.name)
            os.replace(temporary_path, path)

        try:
            oracle_table, oracle_report = compute()
        except ValueError as error:
            atomic_write(error_path, json.dumps({"error": str(error)}).encode())
            raise
        with tempfile.NamedTemporaryFile(dir=self.path, suffix=".parquet", delete=False) as temporary:
            temporary_path = Path(temporary.name)
        try:
            oracle_table.write_parquet(temporary_path)
            os.replace(temporary_path, table_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        atomic_write(report_path, json.dumps(oracle_report).encode())
        return oracle_table, oracle_report, False

    def load_cached(self, context_id: int) -> tuple[pl.DataFrame, dict[str, int]] | None:
        """Return an existing certificate without creating cache state."""
        table_path, report_path, error_path = self._paths(context_id)
        if table_path.exists() and report_path.exists():
            return pl.read_parquet(table_path), json.loads(report_path.read_text())
        if error_path.exists():
            raise ValueError(json.loads(error_path.read_text())["error"])
        return None
