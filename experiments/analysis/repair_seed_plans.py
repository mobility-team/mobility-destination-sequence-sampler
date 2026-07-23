"""Diagnose LNS-style exact single-layer repair of bounded seed plans.

Each repair fixes every layer except one to a seed plan, then uses the exact
oracle to reinsert the free layer. It is intentionally diagnostic-only: it
tests whether diverse bounded plans contain enough correct context for a cheap
local repair to recover missed oracle paths.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

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
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed-count", type=int, default=10)
    parser.add_argument("--repair-k", type=int, default=4)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--max-states", type=int, default=2_000_000)
    add_top_k_tuning_arguments(parser)
    return parser.parse_args()


def ranked_paths(table: pl.DataFrame) -> list[tuple[tuple[int, ...], float]]:
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


def make_seed_neighborhood(
    destination_inputs: pl.DataFrame,
    context_steps: pl.DataFrame,
    zones: tuple[int, ...],
    free_layer: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Keep the original scoring semantics while fixing non-free domains.

    Replacing a variable step with ``fixed_destination`` changes first-choice
    scoring. Instead, make a private activity type for each fixed layer whose
    domain contains only the seed's selected destination.  Its sole copied
    destination-input row has the same utility attributes as the original, so
    this is an exact domain restriction even when activity types repeat.
    """
    activity_type = context_steps.schema["activity_id"]
    next_activity = int(destination_inputs["activity_id"].max()) + 1
    restricted_inputs = [destination_inputs]
    repaired_steps = context_steps
    for layer, step in enumerate(context_steps.iter_rows(named=True)):
        if layer == free_layer or step["fixed_destination"] is not None:
            continue
        source_activity = int(step["activity_id"])
        replacement_activity = next_activity
        next_activity += 1
        selected = destination_inputs.filter(
            (pl.col("activity_id") == source_activity)
            & (pl.col("destination") == zones[layer])
        )
        if selected.height != 1:
            raise ValueError(
                f"seed destination {zones[layer]} is not unique for activity "
                f"{source_activity} at layer {layer}"
            )
        restricted_inputs.append(
            selected.with_columns(
                pl.lit(replacement_activity).cast(activity_type).alias("activity_id")
            )
        )
        repaired_steps = repaired_steps.with_columns(
            pl.when(pl.col("layer") == layer)
            .then(pl.lit(replacement_activity).cast(activity_type))
            .otherwise(pl.col("activity_id"))
            .alias("activity_id")
        )
    return repaired_steps, pl.concat(restricted_inputs)


def main() -> None:
    args = parse_args()
    files = resolve_snapshot_files(args.group_day_trips_folder)
    print("Preparing cached Grand Geneve inputs (read-only)...")
    od_costs = prepare_od_costs(files["transport_costs"], files["demand_groups"])
    destination_inputs = prepare_destination_inputs(
        files["destination_saturation"], files["demand_groups"]
    )
    search = DestinationPlanSearch(od_costs=od_costs, destination_inputs=destination_inputs)
    steps, initial_locations, _ = prepare_complete_contexts(
        activity_sequences_path=files["activity_sequences"],
        survey_plan_steps_path=files["survey_plan_steps"],
        demand_groups_path=files["demand_groups"],
        activity_dur_path=files["activity_dur"],
    )
    context_steps = steps.filter(pl.col("context_id") == args.context_id).sort("layer")
    context_initial = initial_locations.filter(pl.col("context_id") == args.context_id)
    if context_steps.is_empty() or context_initial.is_empty():
        raise ValueError(f"unknown context {args.context_id}")
    if context_steps.filter(pl.col("anchor_id").is_not_null()).height:
        raise ValueError(
            "diagnostic currently requires a context without anchors; private "
            "activity domains are incompatible with a shared anchor id"
        )
    common = dict(
        initial_locations=context_initial,
        logit_scale=LOGIT_SCALE,
        update_plan_timings=True,
        use_shadow_prices=True,
        skip_infeasible=False,
    )
    oracle_table, _ = search.exact_top_k(
        steps=context_steps,
        **common,
        top_k=args.top_k,
        max_states=args.max_states,
        n_threads=1,
    )
    oracle = ranked_paths(oracle_table)
    bounded_table, _ = search.top_k(
        steps=context_steps,
        **common,
        exploration_seed=42,
        **top_k_tuning_options(args),
        top_k=args.top_k,
        n_threads=1,
    )
    current = ranked_paths(bounded_table)[: args.seed_count]
    variable_layers = [
        layer
        for layer, fixed in enumerate(context_steps["fixed_destination"].to_list())
        if fixed is None
    ]
    print(
        f"context={args.context_id} layers={context_steps.height} "
        f"variable_layers={variable_layers}; bounded hits="
        f"{sum(zones in {path for path, _ in current} for zones, _ in oracle)}/{args.top_k}"
    )
    for repair_pass in range(1, args.passes + 1):
        repair_started = perf_counter()
        repaired = {zones: score for zones, score in current}
        calls = 0
        infeasible_repairs = 0
        for zones, _ in current:
            for layer in variable_layers:
                try:
                    repair_steps, repair_inputs = make_seed_neighborhood(
                        destination_inputs, context_steps, zones, layer
                    )
                    neighborhood_search = DestinationPlanSearch(
                        od_costs=od_costs,
                        destination_inputs=repair_inputs,
                    )
                    repaired_table, _ = neighborhood_search.exact_top_k(
                        steps=repair_steps,
                        **common,
                        top_k=args.repair_k,
                        max_states=args.max_states,
                        n_threads=1,
                    )
                except ValueError as error:
                    if infeasible_repairs == 0:
                        print(f"  first infeasible repair: {error}")
                    infeasible_repairs += 1
                    continue
                calls += 1
                for repaired_zones, score in ranked_paths(repaired_table):
                    repaired[repaired_zones] = score
        current = sorted(repaired.items(), key=lambda item: (-item[1], item[0]))[: args.seed_count]
        oracle_rank = {zones: rank for rank, (zones, _) in enumerate(oracle, start=1)}
        print(
            f"pass={repair_pass} exact-repair-calls={calls} "
            f"infeasible-neighborhoods={infeasible_repairs} "
            f"elapsed={perf_counter() - repair_started:.3f}s"
        )
        for rank, (zones, score) in enumerate(current, start=1):
            print(
                f"  repaired rank={rank:2d} exact-rank={oracle_rank.get(zones, f'>{args.top_k}')} "
                f"utility={score:.6f} zones={zones}"
            )
        hits = sum(zones in {path for path, _ in current} for zones, _ in oracle)
        print(f"  oracle hits={hits}/{args.top_k}")


if __name__ == "__main__":
    main()
