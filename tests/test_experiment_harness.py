from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
import pytest

from experiments.benchmarks.compare_bidirectional_throughput import (
    counterbalanced_sequence,
    throughput_verdict,
)
from experiments.analysis.compare_bidirectional_top_k_grand_geneve import (
    stratified_missingness_summary,
)
from experiments.experiment import write_draft
from experiments.harness import (
    ExperimentKind,
    ExperimentManifest,
    linked_verdict,
    quality_verdict,
)
from experiments.oracle_cache import OracleCache


def valid_manifest(
    tmp_path: Path,
    *,
    kind: str = "pure_perf",
    stage: str = "discovery",
) -> Path:
    path = tmp_path / "experiment.toml"
    write_draft(
        argparse.Namespace(
            path=path,
            identifier="test-experiment",
            kind=kind,
            stage=stage,
            change=["frontier_width=48"],
        )
    )
    text = path.read_text()
    text = text.replace(
        "TODO: state the expected measurable outcome",
        "A wider frontier reduces aggregate Rust time",
    )
    text = text.replace(
        "TODO: state why the change should produce that outcome",
        "It reduces repeated proposal reconstruction",
    )
    text = text.replace(
        "TODO: state what observation would reject the hypothesis",
        "The paired Rust improvement is below three percent",
    )
    text = text.replace(
        "TODO: name the most important unresolved assumption",
        "The calibrated cohort represents deep contexts",
    )
    if stage == "validation":
        text = text.replace(
            "TODO: lock after a dry run",
            "0123456789abcdef0123",
        )
    path.write_text(text)
    return path


def test_manifest_snapshots_configs_and_rejects_undeclared_drift(
    tmp_path: Path,
) -> None:
    path = valid_manifest(tmp_path)
    manifest = ExperimentManifest.load(path)

    assert manifest.allowed_differences == ("frontier_width",)
    assert manifest.baseline["frontier_width"] == 40
    assert manifest.candidate["frontier_width"] == 48

    path.write_text(
        path.read_text().replace(
            "pricing_passes = 2",
            "pricing_passes = 1",
            1,
        )
    )
    with pytest.raises(ValueError, match="allowed_differences"):
        ExperimentManifest.load(path)


def test_validation_manifest_locks_the_selected_cohort(tmp_path: Path) -> None:
    manifest = ExperimentManifest.load(valid_manifest(tmp_path, stage="validation"))

    with pytest.raises(ValueError, match="cohort fingerprint mismatch"):
        manifest.verify_cohort([1, 2, 3])


def test_validation_manifest_rejects_placeholder_cohort_lock(
    tmp_path: Path,
) -> None:
    path = valid_manifest(tmp_path, stage="validation")
    path.write_text(
        path.read_text().replace(
            "0123456789abcdef0123",
            "TODO: lock after a dry run",
        )
    )

    with pytest.raises(ValueError, match="still contains a TODO"):
        ExperimentManifest.load(path)


def test_manifest_rejects_selector_incompatible_with_kind(
    tmp_path: Path,
) -> None:
    path = valid_manifest(tmp_path, kind="pure_perf")
    path.write_text(
        path.read_text().replace(
            'selector = "calibrated"',
            'selector = "stratified"',
        )
    )

    with pytest.raises(ValueError, match="cohort.selector"):
        ExperimentManifest.load(path)


def test_counterbalanced_sequence_has_equal_labels_in_every_block() -> None:
    sequence = counterbalanced_sequence(3)

    assert [label for block, label in sequence if block == 1] == [
        "A",
        "B",
        "B",
        "A",
    ]
    assert [label for block, label in sequence if block == 2] == [
        "B",
        "A",
        "A",
        "B",
    ]
    for block in (1, 2, 3):
        labels = [label for value, label in sequence if value == block]
        assert labels.count("A") == labels.count("B") == 2


def test_quality_runtime_verdict_never_masquerades_as_pure_performance() -> None:
    common = {
        "kind": ExperimentKind.QUALITY_RUNTIME,
        "gates": {"max_wall_regression": 0.15},
        "relative_delta": {"wall_seconds": 0.04, "rust_seconds": 0.07},
        "output_same": False,
        "counter_drift": ["expected for a quality-changing policy"],
    }

    incomplete = throughput_verdict(
        **common,
        quality_status="incomplete",
        quality_reason="no quality artifact is linked",
    )
    passed = throughput_verdict(
        **common,
        quality_status="pass",
        quality_reason="linked quality verdict=pass",
    )

    assert incomplete["status"] == "incomplete"
    assert passed["status"] == "pass"


def test_pure_performance_verdict_enforces_output_work_and_speed() -> None:
    verdict = throughput_verdict(
        kind=ExperimentKind.PURE_PERF,
        gates={
            "min_rust_improvement": 0.03,
            "require_same_output": True,
            "require_same_counters": True,
        },
        relative_delta={"wall_seconds": -0.02, "rust_seconds": -0.04},
        output_same=True,
        counter_drift=[],
    )

    assert verdict["status"] == "pass"


def test_exact_certificate_is_reused_across_attempt_budgets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    table = pl.DataFrame({"draw_id": [1], "destination": [7]})
    first = OracleCache(
        "certificate",
        10,
        100,
        attempt_fingerprint="initializer-a",
    )
    computed, report, cached = first.load_or_compute(
        3,
        lambda: (table, {"states_pushed": 40}),
    )
    second = OracleCache(
        "certificate",
        10,
        1_000,
        attempt_fingerprint="initializer-b",
    )
    reused, reused_report, reused_from_cache = second.load_or_compute(
        3,
        lambda: pytest.fail("a completed exact certificate must be reused"),
    )

    assert not cached
    assert computed.equals(table)
    assert report["states_pushed"] == 40
    assert reused_from_cache
    assert reused.equals(table)
    assert reused_report["states_pushed"] == 40


def test_failed_oracle_attempt_does_not_poison_another_initializer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    failed = OracleCache(
        "certificate",
        10,
        100,
        attempt_fingerprint="initializer-a",
    )

    def exceed_budget() -> tuple[pl.DataFrame, dict[str, int]]:
        raise ValueError("exceeded max_states")

    with pytest.raises(ValueError, match="max_states"):
        failed.load_or_compute(9, exceed_budget)

    successful = OracleCache(
        "certificate",
        10,
        100,
        attempt_fingerprint="initializer-b",
    )
    table = pl.DataFrame({"draw_id": [1], "destination": [8]})
    result, _, cached = successful.load_or_compute(
        9,
        lambda: (table, {"states_pushed": 80}),
    )

    assert not cached
    assert result.equals(table)
    assert failed.load_cached(9)[0].equals(table)


def test_stratified_bounds_keep_uncertified_mass_explicit() -> None:
    summary = stratified_missingness_summary(
        population_by_stratum={"deep": 90, "short": 10},
        outcomes={
            "deep": {"sampled": 10},
            "short": {"sampled": 2},
        },
        metrics_by_stratum={
            "deep": {"retained_top_k_mass": [0.8] * 5},
            "short": {"retained_top_k_mass": [1.0, 1.0]},
        },
        metric="retained_top_k_mass",
    )

    assert summary["lower"] == pytest.approx(0.46)
    assert summary["imputed"] == pytest.approx(0.82)
    assert summary["upper"] == pytest.approx(0.91)
    assert summary["unknown_impacts"][0] == {
        "stratum": "deep",
        "sampled": 10,
        "unresolved": 5,
        "impact": pytest.approx(0.45),
    }


def test_quality_verdict_uses_only_predeclared_delta_gates() -> None:
    verdict = quality_verdict(
        {
            "stratified_mass_delta_min": 0.02,
            "zero_mass_delta_max": 0.0,
        },
        baseline={"stratified_mass": 0.80, "zero_mass": 5},
        candidate={"stratified_mass": 0.84, "zero_mass": 3},
    )

    assert verdict["status"] == "pass"
    assert [result["delta"] for result in verdict["gate_results"]] == pytest.approx(
        [0.04, -2.0]
    )


def test_linked_quality_evidence_must_match_resolved_configs(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "quality.json"
    artifact.write_text(
        '{"resolved_configs":{"A":{"width":40},"B":{"width":48}},'
        '"verdict":{"status":"pass"}}'
    )

    status, reason = linked_verdict(
        artifact,
        expected_configs={"A": {"width": 40}, "B": {"width": 64}},
    )

    assert status == "incomplete"
    assert "different resolved A/B configs" in reason


def test_linked_quality_evidence_is_rechecked_against_runtime_manifest_gates(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "quality.json"
    artifact.write_text(
        '{"resolved_configs":{"A":{"width":40},"B":{"width":48}},'
        '"quality_summaries":{'
        '"A":{"stratified_mass":0.80},'
        '"B":{"stratified_mass":0.81}},'
        '"verdict":{"status":"pass"}}'
    )

    status, reason = linked_verdict(
        artifact,
        expected_configs={"A": {"width": 40}, "B": {"width": 48}},
        expected_quality_gates={
            "max_wall_regression": 0.15,
            "stratified_mass_delta_min": 0.02,
        },
    )

    assert status == "fail"
    assert "does not satisfy" in reason
