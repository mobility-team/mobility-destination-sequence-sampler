"""Align one bounded-search failure against its exact top-K oracle paths."""

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
    parser.add_argument("--context-id", type=int, default=2679)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-states", type=int, default=2_000_000)
    add_top_k_tuning_arguments(parser)
    return parser.parse_args()


def ranked_paths(table: pl.DataFrame) -> list[tuple[tuple[int, ...], float]]:
    """Extract complete plans in descending complete-plan utility order."""
    paths = []
    for draw_id in table["draw_id"].unique().to_list():
        rows = table.filter(pl.col("draw_id") == draw_id).sort("layer")
        paths.append(
            (
                tuple(int(zone) for zone in rows["destination"].to_list()),
                float(rows.item(0, "total_log_weight")),
            )
        )
    return sorted(paths, key=lambda item: (-item[1], item[0]))


def main() -> None:
    args = parse_args()
    files = resolve_snapshot_files(args.group_day_trips_folder)
    print("Preparing cached Grand Geneve inputs (read-only)...")
    search = DestinationPlanSearch(
        od_costs=prepare_od_costs(files["transport_costs"], files["demand_groups"]),
        destination_inputs=prepare_destination_inputs(
            files["destination_saturation"], files["demand_groups"]
        ),
    )
    steps, initial_locations, _ = prepare_complete_contexts(
        activity_sequences_path=files["activity_sequences"],
        survey_plan_steps_path=files["survey_plan_steps"],
        demand_groups_path=files["demand_groups"],
        activity_dur_path=files["activity_dur"],
    )
    context_steps = steps.filter(pl.col("context_id") == args.context_id)
    context_initial = initial_locations.filter(pl.col("context_id") == args.context_id)
    if context_steps.is_empty() or context_initial.is_empty():
        raise ValueError(f"unknown context {args.context_id}")
    common = dict(
        steps=context_steps,
        initial_locations=context_initial,
        logit_scale=LOGIT_SCALE,
        update_plan_timings=True,
        use_shadow_prices=True,
        skip_infeasible=False,
    )
    oracle_output, _ = search.exact_top_k(
        **common, top_k=args.top_k, max_states=args.max_states, n_threads=1
    )
    bounded_output, _ = search.top_k(
        **common,
        exploration_seed=42,
        **top_k_tuning_options(args),
        top_k=args.top_k,
        n_threads=1,
    )
    oracle = ranked_paths(oracle_output)
    bounded = ranked_paths(bounded_output)
    bounded_rank = {zones: rank for rank, (zones, _) in enumerate(bounded, start=1)}
    best_utility = oracle[0][1]
    weights = [math.exp(utility - best_utility) for _, utility in oracle]
    normalizer = sum(weights)
    rows = [
        {
            "oracle_rank": rank,
            "path": " -> ".join(str(zone) for zone in zones),
            "oracle_utility": utility,
            "delta_utility": utility - best_utility,
            "oracle_top_k_mass": weight / normalizer,
            "bounded_result": "returned" if zones in bounded_rank else "missed",
            "bounded_rank": bounded_rank.get(zones),
        }
        for rank, ((zones, utility), weight) in enumerate(zip(oracle, weights, strict=True), start=1)
    ]
    initial_zone = context_initial.item(0, "initial_zone")
    print(
        f"context={args.context_id} initial_zone={initial_zone} layers={context_steps.height}; "
        f"oracle and bounded K={args.top_k}; mass is normalized within oracle top-K"
    )
    table = pl.DataFrame(rows)
    with pl.Config(tbl_rows=args.top_k, tbl_cols=10, tbl_width_chars=180):
        print(table)
    returned = sum(row["bounded_result"] == "returned" for row in rows)
    retained_mass = sum(
        row["oracle_top_k_mass"] for row in rows if row["bounded_result"] == "returned"
    )
    print(
        f"returned oracle paths={returned}/{args.top_k}; "
        f"retained oracle-top-K mass={retained_mass:.6f}"
    )


if __name__ == "__main__":
    main()
