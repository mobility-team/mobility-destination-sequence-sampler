from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
from mobility_destination_sequence_sampler._core import (
    ExperimentalDestinationSampler as DestinationSampler,
)

from experiments.benchmarks.perf_grand_geneve_cache import (
    DEFAULT_GROUP_DAY_TRIPS_FOLDER,
    LOGIT_SCALE,
    prepare_complete_contexts,
    prepare_destination_inputs,
    prepare_od_costs,
    resolve_snapshot_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Report raw-zone second-order workload by chain length and number "
            "of variable anchor types."
        )
    )
    parser.add_argument(
        "--group-day-trips-folder",
        type=Path,
        default=DEFAULT_GROUP_DAY_TRIPS_FOLDER,
    )
    parser.add_argument(
        "--contexts-per-cell",
        type=int,
        default=0,
        help="Run this many deterministic contexts in every supported cell.",
    )
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--n-threads", type=int, default=8)
    return parser.parse_args()


def context_summary(steps: pl.DataFrame, seed: int) -> pl.DataFrame:
    """Summarize the exact recursion state space without expanding destinations."""
    return (
        steps.group_by("context_id")
        .agg(
            layers=pl.len(),
            anchor_types=pl.col("anchor_id").drop_nulls().n_unique(),
        )
        .with_columns(
            sample_order=pl.struct(["context_id", "layers", "anchor_types"])
            .hash(seed=seed)
            .cast(pl.UInt64)
        )
    )


def print_distribution(summary: pl.DataFrame) -> None:
    distribution = (
        summary.group_by(["layers", "anchor_types"])
        .agg(contexts=pl.len())
        .sort(["layers", "anchor_types"])
    )
    print("Contexts by activity-chain length and variable anchor types")
    with pl.Config(tbl_rows=distribution.height):
        print(distribution)
    print()
    print(
        "Contexts with two or more anchor types are not passed to the current "
        "second-order solver, because it only conditions one repeated anchor "
        "type exactly."
    )
    print(
        summary.group_by("anchor_types")
        .agg(contexts=pl.len())
        .sort("anchor_types")
    )


def print_two_anchor_tours(steps: pl.DataFrame, summary: pl.DataFrame) -> None:
    """Show whether two anchor types coexist in one home-bounded tour."""
    two_anchor_contexts = summary.filter(pl.col("anchor_types") == 2).select(
        "context_id"
    )
    tour_anchor_counts = (
        steps.join(two_anchor_contexts, on="context_id")
        .sort(["context_id", "layer"])
        .with_columns(
            tour_id=(pl.col("activity_id") == 0)
            .cast(pl.UInt32)
            .cum_sum()
            .over("context_id")
        )
        .filter(pl.col("anchor_id").is_not_null())
        .group_by(["context_id", "tour_id"])
        .agg(anchor_types=pl.col("anchor_id").n_unique())
    )
    two_anchor_tours = (
        two_anchor_contexts.join(
            tour_anchor_counts.group_by("context_id").agg(
                anchor_types_in_one_tour=(pl.col("anchor_types") >= 2).any()
            ),
            on="context_id",
            how="left",
        )
        .with_columns(
            anchor_types_in_one_tour=pl.col("anchor_types_in_one_tour").fill_null(
                False
            )
        )
        .group_by("anchor_types_in_one_tour")
        .agg(contexts=pl.len())
        .sort("anchor_types_in_one_tour")
    )
    print()
    print("Two-anchor contexts by whether both anchors occur in one tour")
    print(two_anchor_tours)


def benchmark_cells(
    *,
    steps: pl.DataFrame,
    initial_locations: pl.DataFrame,
    od_costs: pl.DataFrame,
    destination_inputs: pl.DataFrame,
    summary: pl.DataFrame,
    contexts_per_cell: int,
    n_threads: int,
) -> None:
    sampler = DestinationSampler(
        od_costs=od_costs,
        destination_inputs=destination_inputs,
    )
    cells = (
        summary.filter(pl.col("anchor_types") <= 1)
        .group_by(["layers", "anchor_types"])
        .agg(
            context_ids=pl.col("context_id")
            .sort_by("sample_order")
            .head(contexts_per_cell)
        )
        .sort(["layers", "anchor_types"])
    )

    print()
    print("Raw-zone exact solver by supported context cell")
    print(
        "layers anchors sampled core_seconds seconds_per_context "
        "corridor_pair_share infeasible"
    )
    for row in cells.iter_rows(named=True):
        context_ids = row["context_ids"]
        selected_steps = steps.filter(pl.col("context_id").is_in(context_ids))
        selected_initial_locations = initial_locations.filter(
            pl.col("context_id").is_in(context_ids)
        )
        report = sampler.solve_second_order(
            steps=selected_steps,
            initial_locations=selected_initial_locations,
            logit_scale=LOGIT_SCALE,
            update_plan_timings=True,
            use_shadow_prices=True,
            wrapped_home_time_shadow_price=2.0,
            use_bidirectional_feasibility=True,
            n_threads=n_threads,
            skip_infeasible=True,
        )
        pair_states = int(report["pair_states"])
        corridor_pair_states = int(report["corridor_pair_states"])
        core_seconds = float(report["wall_seconds"])
        sampled = len(context_ids)
        print(
            f"{row['layers']:6d} {row['anchor_types']:7d} {sampled:7d} "
            f"{core_seconds:12.3f} {core_seconds / sampled:19.4f} "
            f"{corridor_pair_states / max(pair_states, 1):19.2%} "
            f"{int(report['infeasible_contexts']):10d}"
        )


def main() -> None:
    args = parse_args()
    if args.contexts_per_cell < 0:
        raise ValueError("contexts-per-cell must be non-negative")
    if args.n_threads <= 0:
        raise ValueError("n-threads must be positive")

    files = resolve_snapshot_files(args.group_day_trips_folder)
    od_costs = prepare_od_costs(
        files["transport_costs"], files["demand_groups"]
    )
    destination_inputs = prepare_destination_inputs(
        files["destination_saturation"],
        files["demand_groups"],
    )
    steps, initial_locations, _ = prepare_complete_contexts(
        activity_sequences_path=files["activity_sequences"],
        survey_plan_steps_path=files["survey_plan_steps"],
        demand_groups_path=files["demand_groups"],
        activity_dur_path=files["activity_dur"],
    )
    summary = context_summary(steps, args.seed)
    print_distribution(summary)
    print_two_anchor_tours(steps, summary)
    if args.contexts_per_cell:
        benchmark_cells(
            steps=steps,
            initial_locations=initial_locations,
            od_costs=od_costs,
            destination_inputs=destination_inputs,
            summary=summary,
            contexts_per_cell=args.contexts_per_cell,
            n_threads=args.n_threads,
        )


if __name__ == "__main__":
    main()
