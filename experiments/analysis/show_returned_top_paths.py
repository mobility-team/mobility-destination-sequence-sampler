"""Print exact-score-ranked bounded paths and returned-support mass for contexts."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import polars as pl

from mobility_destination_sequence_sampler import DestinationPlanSearch

from experiments.benchmarks.perf_grand_geneve_cache import (
    DEFAULT_GROUP_DAY_TRIPS_FOLDER,
    LOGIT_SCALE,
    prepare_complete_contexts,
    prepare_destination_inputs,
    prepare_od_costs,
    resolve_snapshot_files,
)
from experiments.top_k_config import add_top_k_tuning_arguments, top_k_tuning_options


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group-day-trips-folder", type=Path, default=DEFAULT_GROUP_DAY_TRIPS_FOLDER
    )
    parser.add_argument("--context-id", type=int, action="append", dest="context_ids")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--show", type=int, default=20)
    add_top_k_tuning_arguments(parser)
    parser.set_defaults(frontier_width=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context_ids = args.context_ids or [34961, 52014]
    if args.top_k < args.show or args.frontier_width < args.top_k:
        raise ValueError("top-k must cover --show and fit frontier-width")
    files = resolve_snapshot_files(args.group_day_trips_folder)
    print("Preparing cached Grand Geneve inputs (read-only)...")
    od_costs = prepare_od_costs(files["transport_costs"], files["demand_groups"])
    destination_inputs = prepare_destination_inputs(
        files["destination_saturation"], files["demand_groups"]
    )
    steps, initial_locations, _ = prepare_complete_contexts(
        activity_sequences_path=files["activity_sequences"],
        survey_plan_steps_path=files["survey_plan_steps"],
        demand_groups_path=files["demand_groups"],
        activity_dur_path=files["activity_dur"],
    )
    search = DestinationPlanSearch(od_costs=od_costs, destination_inputs=destination_inputs)
    for context_id in context_ids:
        context_steps = steps.filter(pl.col("context_id") == context_id)
        context_initial = initial_locations.filter(pl.col("context_id") == context_id)
        if context_steps.is_empty() or context_initial.is_empty():
            raise ValueError(f"unknown context {context_id}")
        output, _ = search.top_k(
            steps=context_steps,
            initial_locations=context_initial,
            logit_scale=LOGIT_SCALE,
            update_plan_timings=True,
            use_shadow_prices=True,
            exploration_seed=42,
            **top_k_tuning_options(args),
            top_k=args.top_k,
            n_threads=1,
            skip_infeasible=False,
        )
        paths = (
            output.group_by("draw_id")
            .agg(
                pl.col("destination").sort_by("layer").alias("zones"),
                pl.col("total_log_weight").first().alias("utility"),
            )
            .sort("utility", descending=True)
        )
        utilities = paths["utility"].to_list()
        maximum = max(utilities)
        weights = [math.exp(utility - maximum) for utility in utilities]
        normalizer = sum(weights)
        table = paths.with_columns(
            rank=pl.int_range(1, paths.height + 1, eager=True),
            delta_utility=pl.col("utility") - maximum,
            returned_mass=pl.Series([weight / normalizer for weight in weights]),
            cumulative_returned_mass=pl.Series(
                [sum(weights[:index]) / normalizer for index in range(1, len(weights) + 1)]
            ),
            path=pl.col("zones")
            .list.eval(pl.element().cast(pl.String))
            .list.join(" → "),
        ).select(
            "rank", "path", "utility", "delta_utility", "returned_mass", "cumulative_returned_mass"
        )
        print(
            f"\ncontext={context_id} layers={context_steps.height} returned={paths.height}; "
            "mass is conditional on this returned support"
        )
        # Keep every requested row visible; Polars otherwise abbreviates the
        # middle of a 20-row diagnostic table.
        with pl.Config(tbl_rows=args.show, tbl_cols=10, tbl_width_chars=180):
            print(table.head(args.show))


if __name__ == "__main__":
    main()
