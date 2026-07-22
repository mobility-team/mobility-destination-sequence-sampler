"""Measure workload-level reuse available to bounded-search factor scoring."""

from __future__ import annotations

from experiments.benchmarks.perf_grand_geneve_cache import (
    DEFAULT_GROUP_DAY_TRIPS_FOLDER,
    prepare_complete_contexts,
    prepare_destination_inputs,
    resolve_snapshot_files,
)

import polars as pl


def main() -> None:
    files = resolve_snapshot_files(DEFAULT_GROUP_DAY_TRIPS_FOLDER)
    destination_inputs = prepare_destination_inputs(
        files["destination_saturation"], files["demand_groups"]
    )
    steps, _, _ = prepare_complete_contexts(
        activity_sequences_path=files["activity_sequences"],
        survey_plan_steps_path=files["survey_plan_steps"],
        demand_groups_path=files["demand_groups"],
        activity_dur_path=files["activity_dur"],
    )
    steps = steps.sort(["context_id", "layer"])
    depths = steps.group_by("context_id").agg(layers=pl.len())
    print("contexts by depth")
    print(depths.group_by("layers").len().sort("layers"))

    first_choice = (
        pl.col("fixed_destination").is_null()
        & (
            pl.col("anchor_id").is_null()
            | (
                pl.col("anchor_id")
                .cum_count()
                .over(["context_id", "anchor_id"])
                == 1
            )
        )
    )
    scoring_columns = [
        "activity_id",
        "fixed_destination",
        "departure_time",
        "arrival_time",
        "arrival_time_rigidity",
        "departure_time_rigidity",
        "duration_per_person",
        "value_of_time",
        "mean_duration_per_person",
        "min_activity_time",
    ]
    profiled = (
        steps.join(depths, on="context_id")
        .with_columns(first_choice=first_choice)
        .with_columns(
            step_profile=pl.struct(scoring_columns + ["first_choice"]).hash(seed=101)
        )
        .with_columns(
            next_step_profile=pl.col("step_profile").shift(-1).over("context_id")
        )
        .with_columns(
            factor_profile=pl.struct(["step_profile", "next_step_profile"]).hash(seed=103)
        )
    )
    active = profiled.filter(pl.col("layers").is_between(3, 5))

    def print_reuse(label: str, frame: pl.DataFrame, column: str) -> None:
        counts = frame.group_by(column).len()["len"]
        print(
            f"{label}: rows={frame.height} unique={counts.len()} "
            f"reuse={frame.height / counts.len():.2f}x "
            f"singletons={(counts == 1).sum() / counts.len():.1%} "
            f"max={counts.max()}"
        )

    print_reuse("active step profiles", active, "step_profile")
    print_reuse("active adjacent factor profiles", active, "factor_profile")

    domains = destination_inputs.group_by("activity_id").agg(domain=pl.len())
    active_variable = active.filter(pl.col("fixed_destination").is_null()).join(
        domains, on="activity_id", how="left"
    )
    print_reuse(
        "active variable adjacent factor profiles", active_variable, "factor_profile"
    )
    print(
        "active variable layers: "
        f"rows={active_variable.height} "
        f"mean-domain={active_variable['domain'].mean():.1f} "
        f"median-domain={active_variable['domain'].median():.0f} "
        f"max-domain={active_variable['domain'].max()}"
    )
    print("domain sizes by activity")
    print(domains.sort("domain", descending=True))


if __name__ == "__main__":
    main()
