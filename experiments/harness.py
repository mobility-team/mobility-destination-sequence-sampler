"""Typed experiment manifests, cohort locks, and self-contained run artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
import tomllib
from typing import Any

from experiments.top_k_config import ACTIVE_TOP_K_DEFAULTS


MANIFEST_SCHEMA_VERSION = 1
RUN_ARTIFACT_VERSION = 1
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9-]*$")


class ExperimentKind(StrEnum):
    PURE_PERF = "pure_perf"
    QUALITY_RUNTIME = "quality_runtime"
    QUALITY_ONLY = "quality_only"


class ExperimentStage(StrEnum):
    CANARY = "canary"
    DISCOVERY = "discovery"
    VALIDATION = "validation"


QUALITY_GATE_METRICS = {
    "stratified_mass_delta": "stratified_mass",
    "stratified_recall_delta": "stratified_recall",
    "zero_mass_delta": "zero_mass",
    "oracle_interval_width_delta": "oracle_interval_width",
}


def canonical_fingerprint(payload: Any, *, length: int = 20) -> str:
    """Hash JSON-compatible data with stable ordering and representation."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()[:length]


def cohort_fingerprint(
    context_ids: list[int],
    *,
    selection: dict[str, Any] | None = None,
) -> str:
    """Identify a cohort independently of row order."""
    unique_ids = sorted(set(context_ids))
    if len(unique_ids) != len(context_ids):
        raise ValueError("cohort context IDs must be unique")
    return canonical_fingerprint(
        {
            "context_ids": unique_ids,
            "selection": selection or {},
        }
    )


def _require_text(payload: dict[str, Any], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"manifest field {name!r} must be non-empty text")
    if value.lstrip().lower().startswith("todo"):
        raise ValueError(f"manifest field {name!r} still contains a TODO")
    return value.strip()


def _validate_option_type(name: str, value: Any) -> None:
    expected = ACTIVE_TOP_K_DEFAULTS[name]
    if isinstance(expected, bool):
        valid = isinstance(value, bool)
    elif isinstance(expected, int):
        valid = isinstance(value, int) and not isinstance(value, bool)
    elif isinstance(expected, float):
        valid = isinstance(value, (int, float)) and not isinstance(value, bool)
    else:
        valid = isinstance(value, type(expected))
    if not valid:
        raise ValueError(
            f"top-K option {name!r} must have type {type(expected).__name__}, "
            f"not {type(value).__name__}"
        )


def _load_complete_options(payload: dict[str, Any], label: str) -> dict[str, Any]:
    section = payload.get(label)
    if not isinstance(section, dict) or not isinstance(section.get("options"), dict):
        raise ValueError(f"manifest requires [{label}.options]")
    options = dict(section["options"])
    expected = set(ACTIVE_TOP_K_DEFAULTS)
    missing = sorted(expected - set(options))
    unknown = sorted(set(options) - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unknown:
            details.append(f"unknown={unknown}")
        raise ValueError(
            f"{label} must snapshot every active top-K option ({'; '.join(details)})"
        )
    for name, value in options.items():
        _validate_option_type(name, value)
        if isinstance(ACTIVE_TOP_K_DEFAULTS[name], float):
            options[name] = float(value)
    return options


def _require_number(
    payload: dict[str, Any],
    name: str,
    *,
    minimum: float | None = None,
) -> float:
    value = payload.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"gate {name!r} must be numeric")
    value = float(value)
    if minimum is not None and value < minimum:
        raise ValueError(f"gate {name!r} must be at least {minimum}")
    return value


@dataclass(frozen=True)
class ExperimentManifest:
    """A validated, immutable experiment definition."""

    path: Path
    identifier: str
    kind: ExperimentKind
    stage: ExperimentStage
    hypothesis: str
    mechanism: str
    falsifier: str
    unknowns: tuple[str, ...]
    baseline: dict[str, Any]
    candidate: dict[str, Any]
    allowed_differences: tuple[str, ...]
    cohort: dict[str, Any]
    gates: dict[str, Any]
    evidence: dict[str, Any]
    raw: dict[str, Any]
    fingerprint: str

    @classmethod
    def load(cls, path: Path) -> ExperimentManifest:
        path = path.resolve()
        with path.open("rb") as source:
            payload = tomllib.load(source)
        if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"manifest schema_version must be {MANIFEST_SCHEMA_VERSION}"
            )
        identifier = _require_text(payload, "id")
        if not _IDENTIFIER.fullmatch(identifier):
            raise ValueError(
                "manifest id must contain only lowercase letters, digits, and hyphens"
            )
        try:
            kind = ExperimentKind(payload.get("kind"))
        except ValueError as error:
            raise ValueError(
                f"manifest kind must be one of {[kind.value for kind in ExperimentKind]}"
            ) from error
        try:
            stage = ExperimentStage(payload.get("stage"))
        except ValueError as error:
            raise ValueError(
                f"manifest stage must be one of {[stage.value for stage in ExperimentStage]}"
            ) from error

        hypothesis = _require_text(payload, "hypothesis")
        mechanism = _require_text(payload, "mechanism")
        falsifier = _require_text(payload, "falsifier")
        unknowns = payload.get("unknowns")
        if (
            not isinstance(unknowns, list)
            or not unknowns
            or not all(isinstance(value, str) and value.strip() for value in unknowns)
        ):
            raise ValueError("manifest unknowns must be a non-empty list of text")
        if any(value.lstrip().lower().startswith("todo") for value in unknowns):
            raise ValueError("manifest unknowns still contain a TODO")

        baseline = _load_complete_options(payload, "baseline")
        candidate = _load_complete_options(payload, "candidate")
        comparison = payload.get("comparison")
        if not isinstance(comparison, dict):
            raise ValueError("manifest requires [comparison]")
        allowed = comparison.get("allowed_differences")
        if not isinstance(allowed, list) or not all(
            isinstance(name, str) for name in allowed
        ):
            raise ValueError("comparison.allowed_differences must be a list of names")
        if len(set(allowed)) != len(allowed):
            raise ValueError("comparison.allowed_differences contains duplicates")
        unknown_allowed = sorted(set(allowed) - set(ACTIVE_TOP_K_DEFAULTS))
        if unknown_allowed:
            raise ValueError(f"unknown allowed differences: {unknown_allowed}")
        actual = sorted(
            name for name in baseline if baseline[name] != candidate[name]
        )
        if sorted(allowed) != actual:
            raise ValueError(
                "candidate differences do not match comparison.allowed_differences: "
                f"declared={sorted(allowed)}, actual={actual}"
            )

        cohort = payload.get("cohort")
        if not isinstance(cohort, dict):
            raise ValueError("manifest requires [cohort]")
        _require_text(cohort, "name")
        role = _require_text(cohort, "role")
        if role != stage.value:
            raise ValueError(
                f"cohort role {role!r} must match experiment stage {stage.value!r}"
            )
        if not isinstance(cohort.get("selection_seed"), int):
            raise ValueError("cohort.selection_seed must be an integer")
        selector = _require_text(cohort, "selector")
        if kind is ExperimentKind.QUALITY_ONLY:
            if selector != "stratified":
                raise ValueError(
                    "quality_only cohort.selector must be 'stratified'"
                )
            count_field = "contexts_per_stratum"
        else:
            if selector not in {"calibrated", "all_supported"}:
                raise ValueError(
                    f"{kind.value} cohort.selector must be 'calibrated' or "
                    "'all_supported'"
                )
            count_field = "contexts"
        context_count = cohort.get(count_field)
        if not isinstance(context_count, int) or context_count <= 0:
            raise ValueError(f"cohort.{count_field} must be a positive integer")
        expected_cohort = cohort.get("expected_fingerprint")
        if stage is ExperimentStage.VALIDATION:
            expected_cohort = _require_text(cohort, "expected_fingerprint")
        if expected_cohort is not None and not re.fullmatch(
            r"[0-9a-f]{20}", str(expected_cohort)
        ):
            raise ValueError(
                "cohort.expected_fingerprint must be a 20-character "
                "lowercase hexadecimal fingerprint"
            )

        gates = payload.get("gates")
        if not isinstance(gates, dict):
            raise ValueError("manifest requires [gates]")
        if kind is ExperimentKind.PURE_PERF:
            _require_number(gates, "min_rust_improvement", minimum=0.0)
            for name in ("require_same_output", "require_same_counters"):
                if gates.get(name) is not True:
                    raise ValueError(f"pure_perf requires {name}=true")
            allowed_gates = {
                "min_rust_improvement",
                "require_same_output",
                "require_same_counters",
            }
        elif kind is ExperimentKind.QUALITY_RUNTIME:
            _require_number(gates, "max_wall_regression", minimum=0.0)
            allowed_gates = {
                "max_wall_regression",
                *(
                    f"{metric}_{bound}"
                    for metric in QUALITY_GATE_METRICS
                    for bound in ("min", "max")
                ),
            }
            if not any(name in gates for name in allowed_gates - {"max_wall_regression"}):
                raise ValueError(
                    "quality_runtime requires at least one declared quality gate"
                )
        elif kind is ExperimentKind.QUALITY_ONLY:
            allowed_gates = {
                f"{metric}_{bound}"
                for metric in QUALITY_GATE_METRICS
                for bound in ("min", "max")
            }
            if not any(name in gates for name in allowed_gates):
                raise ValueError("quality_only requires at least one *_min or *_max gate")
        unknown_gates = sorted(set(gates) - allowed_gates)
        if unknown_gates:
            raise ValueError(f"unsupported gates for {kind.value}: {unknown_gates}")
        for name in set(gates) & {
            f"{metric}_{bound}"
            for metric in QUALITY_GATE_METRICS
            for bound in ("min", "max")
        }:
            _require_number(gates, name)

        evidence = payload.get("evidence", {})
        if not isinstance(evidence, dict):
            raise ValueError("manifest [evidence] must be a table")
        fingerprint = canonical_fingerprint(payload)
        return cls(
            path=path,
            identifier=identifier,
            kind=kind,
            stage=stage,
            hypothesis=hypothesis,
            mechanism=mechanism,
            falsifier=falsifier,
            unknowns=tuple(value.strip() for value in unknowns),
            baseline=baseline,
            candidate=candidate,
            allowed_differences=tuple(allowed),
            cohort=dict(cohort),
            gates=dict(gates),
            evidence=dict(evidence),
            raw=payload,
            fingerprint=fingerprint,
        )

    def verify_cohort(
        self,
        context_ids: list[int],
        *,
        selection: dict[str, Any] | None = None,
    ) -> str:
        if selection is None:
            selection = {
                name: value
                for name, value in self.cohort.items()
                if name not in {"name", "role", "expected_fingerprint"}
            }
        actual = cohort_fingerprint(context_ids, selection=selection)
        expected = self.cohort.get("expected_fingerprint")
        if expected and expected != actual:
            raise ValueError(
                f"cohort fingerprint mismatch: expected={expected}, actual={actual}"
            )
        return actual

    def evidence_path(self, name: str) -> Path | None:
        value = self.evidence.get(name)
        if not isinstance(value, str) or not value.strip():
            return None
        path = Path(value)
        return path if path.is_absolute() else (self.path.parent / path).resolve()


def _run_git(root: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return result.stdout


def source_state(root: Path) -> dict[str, Any]:
    """Fingerprint tracked changes and untracked source files without copying them."""
    try:
        commit = _run_git(root, "rev-parse", "HEAD").decode().strip()
        status_bytes = _run_git(root, "status", "--porcelain=v1", "-z")
        diff = _run_git(root, "diff", "--binary", "HEAD", "--")
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "fingerprint": None, "paths": []}

    records = [record for record in status_bytes.decode(errors="replace").split("\0") if record]
    paths = sorted(record[3:] for record in records if len(record) >= 4)
    digest = hashlib.sha256()
    digest.update(commit.encode())
    digest.update(diff)
    for record in records:
        if not record.startswith("?? "):
            continue
        relative = record[3:]
        path = root / relative
        digest.update(relative.encode())
        if path.is_file():
            digest.update(path.read_bytes())
    return {
        "commit": commit,
        "dirty": bool(records),
        "fingerprint": digest.hexdigest()[:20],
        "paths": paths,
    }


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary.write(encoded)
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


class RunRecorder:
    """Write one append-friendly directory that is sufficient to audit a run."""

    def __init__(
        self,
        manifest: ExperimentManifest,
        *,
        cohort_identity: str,
        root: Path,
        command: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        timestamp = datetime.now(UTC)
        run_id = (
            f"{timestamp.strftime('%Y%m%dT%H%M%S%fZ')}-"
            f"{manifest.identifier}-{manifest.fingerprint[:8]}"
        )
        self.path = root.resolve() / run_id
        self.path.mkdir(parents=True, exist_ok=False)
        repository = Path(__file__).resolve().parents[1]
        self._started = timestamp
        self._progress_path = self.path / "progress.jsonl"
        _atomic_json(self.path / "manifest.json", manifest.raw)
        _atomic_json(
            self.path / "resolved-configs.json",
            {
                "baseline": manifest.baseline,
                "candidate": manifest.candidate,
                "allowed_differences": list(manifest.allowed_differences),
            },
        )
        _atomic_json(
            self.path / "run.json",
            {
                "artifact_version": RUN_ARTIFACT_VERSION,
                "run_id": run_id,
                "manifest_path": str(manifest.path),
                "manifest_fingerprint": manifest.fingerprint,
                "cohort_fingerprint": cohort_identity,
                "started_at": timestamp.isoformat(),
                "command": command or sys.argv,
                "source": source_state(repository),
                "environment": {
                    "python": sys.version,
                    "platform": platform.platform(),
                    "processor": platform.processor(),
                },
                "metadata": metadata or {},
                "status": "running",
            },
        )

    def progress(self, event: str, **values: Any) -> None:
        record = {
            "at": datetime.now(UTC).isoformat(),
            "event": event,
            **values,
        }
        with self._progress_path.open("a", encoding="utf-8", buffering=1) as output:
            output.write(json.dumps(record, sort_keys=True) + "\n")

    def finalize(self, result: dict[str, Any], *, status: str) -> None:
        finished = datetime.now(UTC)
        _atomic_json(self.path / "result.json", result)
        run_path = self.path / "run.json"
        run = json.loads(run_path.read_text())
        run.update(
            {
                "finished_at": finished.isoformat(),
                "elapsed_seconds": (finished - self._started).total_seconds(),
                "status": status,
            }
        )
        _atomic_json(run_path, run)
        self.progress("finished", status=status)


def linked_verdict(
    path: Path | None,
    *,
    expected_configs: dict[str, dict[str, Any]] | None = None,
    expected_quality_gates: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Read a linked quality result without guessing at metric semantics."""
    if path is None:
        return "incomplete", "no quality artifact is linked"
    if not path.exists():
        return "incomplete", f"quality artifact does not exist: {path}"
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return "incomplete", f"quality artifact is unreadable: {error}"
    if expected_configs is not None:
        actual_configs = payload.get("resolved_configs")
        if actual_configs != expected_configs:
            return (
                "incomplete",
                "quality artifact was produced with different resolved A/B configs",
            )
    if expected_quality_gates is not None:
        summaries = payload.get("quality_summaries")
        if not isinstance(summaries, dict) or set(summaries) != {"A", "B"}:
            return "incomplete", "quality artifact has no comparable A/B summaries"
        reevaluated = quality_verdict(
            expected_quality_gates,
            summaries["A"],
            summaries["B"],
        )
        if reevaluated["status"] != "pass":
            reasons = reevaluated["failures"] + reevaluated["incomplete"]
            return reevaluated["status"], "; ".join(reasons)
    verdict = payload.get("verdict")
    status = verdict.get("status") if isinstance(verdict, dict) else None
    if status not in {"pass", "fail"}:
        return "incomplete", "quality artifact has no pass/fail verdict"
    return status, f"linked quality verdict={status}"


def quality_verdict(
    gates: dict[str, Any],
    baseline: dict[str, float | int],
    candidate: dict[str, float | int],
) -> dict[str, Any]:
    """Evaluate predeclared quality deltas without inventing a utility function."""
    results = []
    failures = []
    incomplete = []
    for gate, threshold_value in gates.items():
        bound = "min" if gate.endswith("_min") else "max" if gate.endswith("_max") else None
        if bound is None:
            continue
        delta_name = gate[: -(len(bound) + 1)]
        metric = QUALITY_GATE_METRICS.get(delta_name)
        if metric is None:
            continue
        if metric not in baseline or metric not in candidate:
            incomplete.append(f"metric {metric!r} is missing")
            continue
        threshold = float(threshold_value)
        delta = float(candidate[metric]) - float(baseline[metric])
        passed = delta >= threshold if bound == "min" else delta <= threshold
        result = {
            "gate": gate,
            "metric": metric,
            "baseline": baseline[metric],
            "candidate": candidate[metric],
            "delta": delta,
            "threshold": threshold,
            "passed": passed,
        }
        results.append(result)
        if not passed:
            operator = ">=" if bound == "min" else "<="
            failures.append(
                f"{metric} delta {delta:+.6f} does not satisfy "
                f"{operator} {threshold:+.6f}"
            )
    status = "fail" if failures else "incomplete" if incomplete else "pass"
    return {
        "status": status,
        "scope": "quality",
        "gate_results": results,
        "failures": failures,
        "incomplete": incomplete,
    }
