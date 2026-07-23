from __future__ import annotations

from experiments.analysis.code_mode_probe import compare_plans, compact_report


def test_compare_plans_exposes_metrics_not_raw_destination_sequences() -> None:
    metrics = compare_plans(
        oracle=[((1, 2), 3.0), ((3, 4), 2.0)],
        bounded=[((1, 2), 3.0)],
        top_k=2,
    )

    assert metrics == {
        "exact_plans": 2,
        "bounded_plans": 1,
        "recovered": 1,
        "recall_at_k": 0.5,
        "mass_at_k": 0.7310585786300049,
    }


def test_compact_report_retains_failures_without_the_raw_tables() -> None:
    report = compact_report(
        [
            {
                "context_id": 1,
                "outcome": "compared",
                "recall_at_k": 1.0,
                "mass_at_k": 1.0,
            },
            {"context_id": 2, "outcome": "oracle_unproven", "reason": "state_limited"},
        ],
        top_k=10,
        oracle_depth=10,
        max_states=500_000,
    )

    assert report["summary"] == {
        "requested_contexts": 2,
        "compared_contexts": 1,
        "oracle_unproven_contexts": 1,
        "bounded_failed_contexts": 0,
        "mean_recall_at_k": 1.0,
        "mean_mass_at_k": 1.0,
    }
    assert "destination" not in repr(report)
