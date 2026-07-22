"""Compare structurally flat and concentrated bounded top-100 returned supports."""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

import polars as pl

from mobility_destination_sequence_sampler import DestinationPlanSearch

from experiments.analysis.diagnose_returned_distribution import summarize_context
from experiments.benchmarks.perf_bidirectional_grand_geneve import eligible_contexts
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
    parser.add_argument("--contexts", type=int, default=1_000)
    parser.add_argument("--cases", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--exploration-seed", type=int, default=42)
    parser.add_argument("--context-id", type=int, action="append", dest="context_ids")
    add_top_k_tuning_arguments(parser)
    parser.set_defaults(frontier_width=128)
    return parser.parse_args()


def returned_support_rows(returned: pl.DataFrame, top_k: int) -> list[dict[str, float | int]]:
    scores = returned.group_by(["context_id", "draw_id"]).agg(
        pl.col("total_log_weight").first().alias("score")
    )
    return [
        summarize_context(int(context_id), values, top_k)
        for context_id, values in (
            scores.group_by("context_id").agg(pl.col("score")).iter_rows()
        )
    ]


def context_shape(context_steps: pl.DataFrame, returned: pl.DataFrame, top_k: int) -> dict[str, int | str]:
    ordered = context_steps.sort("layer")
    anchors = ordered.filter(pl.col("anchor_id").is_not_null())
    counts = anchors.group_by("anchor_id").len()
    top10 = returned.filter(pl.col("draw_id") <= 10)
    return {
        "layers": ordered.height,
        "fixed": ordered.filter(pl.col("fixed_destination").is_not_null()).height,
        "variable": ordered.filter(pl.col("fixed_destination").is_null()).height,
        "anchors": counts.height,
        "repeated_anchors": counts.filter(pl.col("len") > 1).height,
        "activities": "/".join(str(value) for value in ordered["activity_id"].to_list()),
        "unique_destinations_top10": top10["destination"].n_unique(),
        "unique_destinations_topk": returned["destination"].n_unique(),
    }


def profile_context(
    search: DestinationPlanSearch,
    context_steps: pl.DataFrame,
    context_initial: pl.DataFrame,
    args: argparse.Namespace,
) -> tuple[pl.DataFrame, dict]:
    return search.top_k(
        steps=context_steps,
        initial_locations=context_initial,
        logit_scale=LOGIT_SCALE,
        update_plan_timings=True,
        use_shadow_prices=True,
        exploration_seed=args.exploration_seed,
        **top_k_tuning_options(args),
        top_k=args.top_k,
        n_threads=1,
        skip_infeasible=False,
        collect_profile=True,
    )


def print_group_summary(label: str, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    numeric = ("mass_at_10", "effective_paths", "rust_ms", "factor_map_ms", "map_scans", "reverse_partials")
    print(f"\n{label} mean")
    print(" | ".join(f"{key}={mean(float(row[key]) for row in rows):.3f}" for key in numeric))


def main() -> None:
    args = parse_args()
    if args.contexts <= 0 or args.cases <= 0 or args.top_k < 10 or args.frontier_width < args.top_k:
        raise ValueError("invalid contexts/cases/top-k/frontier-width")
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
    if args.context_ids:
        selected = pl.DataFrame({"context_id": list(dict.fromkeys(args.context_ids))})
    else:
        selected = eligible_contexts(steps, args.contexts, args.exploration_seed, False)
    selected_steps = steps.join(selected, on="context_id", how="semi")
    selected_initial = initial_locations.join(selected, on="context_id", how="semi")
    search = DestinationPlanSearch(od_costs=od_costs, destination_inputs=destination_inputs)
    returned, report = search.top_k(
        steps=selected_steps,
        initial_locations=selected_initial,
        logit_scale=LOGIT_SCALE,
        update_plan_timings=True,
        use_shadow_prices=True,
        exploration_seed=args.exploration_seed,
        **top_k_tuning_options(args),
        top_k=args.top_k,
        n_threads=args.threads,
        skip_infeasible=True,
    )
    support = pl.DataFrame(returned_support_rows(returned, args.top_k)).filter(
        pl.col("returned") >= args.top_k
    )
    profiles = selected_steps.group_by("context_id").agg(
        layers=pl.len(),
        fixed=pl.col("fixed_destination").is_not_null().sum(),
        variable=pl.col("fixed_destination").is_null().sum(),
    )
    support = support.join(profiles, on="context_id", how="inner")
    if args.context_ids:
        chosen = [("selected", int(context_id)) for context_id in support["context_id"].to_list()]
    else:
        flat_rows = support.sort("mass_at_10").head(args.cases).to_dicts()
        flat_ids = {int(row["context_id"]) for row in flat_rows}
        concentrated_ids = set()
        chosen = []
        for row in flat_rows:
            context_id = int(row["context_id"])
            chosen.append(("flat", context_id))
            controls = support.filter(
                (pl.col("layers") == row["layers"])
                & (pl.col("fixed") == row["fixed"])
                & (~pl.col("context_id").is_in(list(flat_ids | concentrated_ids)))
            ).sort("mass_at_10", descending=True)
            if controls.is_empty():
                controls = support.filter(
                    (pl.col("layers") == row["layers"])
                    & (~pl.col("context_id").is_in(list(flat_ids | concentrated_ids)))
                ).sort("mass_at_10", descending=True)
            if controls.is_empty():
                continue
            control_id = int(controls.item(0, "context_id"))
            concentrated_ids.add(control_id)
            chosen.append(("concentrated", control_id))
    print(
        f"candidate contexts={selected.height}; returned={support.height} complete top-{args.top_k}; "
        f"infeasible={report['infeasible_contexts']}"
    )
    rows = []
    for group, context_id in chosen:
        context_steps = selected_steps.filter(pl.col("context_id") == context_id)
        context_initial = selected_initial.filter(pl.col("context_id") == context_id)
        profiled, profile = profile_context(search, context_steps, context_initial, args)
        support_row = support.filter(pl.col("context_id") == context_id).to_dicts()[0]
        shape = context_shape(context_steps, profiled, args.top_k)
        rows.append(
            {
                "group": group,
                "context_id": context_id,
                "mass_at_10": support_row["mass_at_10"],
                "mass_at_50": support_row["mass_at_50"],
                "effective_paths": support_row["effective_paths"],
                **shape,
                "rust_ms": profile["total_search_ns"] / 1e6,
                "factor_map_ms": profile["factor_map_ns"] / 1e6,
                "map_scans": sum(
                    profile[f"factor_map_{name}_destination_scans"]
                    for name in ("previous", "current", "next")
                ),
                "reverse_partials": profile["reverse_prefix_partial_calls"],
                "forward_proposals": profile["forward_proposals_evaluated"],
                "stitch_pairs": profile["stitch_pairs"],
            }
        )
    table = pl.DataFrame(rows).sort(["group", "mass_at_10"])
    print("\nDetailed contexts")
    print(
        table.select(
            "group",
            "context_id",
            "layers",
            "fixed",
            "variable",
            "anchors",
            "repeated_anchors",
            "activities",
            "mass_at_10",
            "mass_at_50",
            "effective_paths",
            "unique_destinations_top10",
            "unique_destinations_topk",
        )
    )
    print("\nPer-context work")
    print(
        table.select(
            "group",
            "context_id",
            "rust_ms",
            "factor_map_ms",
            "map_scans",
            "reverse_partials",
            "forward_proposals",
            "stitch_pairs",
        )
    )
    print("\nArchetype detail")
    for row in table.iter_rows(named=True):
        print(
            f"{row['group']:12s} context={row['context_id']} "
            f"layers={row['layers']} fixed={row['fixed']} variable={row['variable']} "
            f"anchors={row['anchors']} repeated={row['repeated_anchors']} "
            f"activities={row['activities']} unique-zones(top10/top{args.top_k})="
            f"{row['unique_destinations_top10']}/{row['unique_destinations_topk']}"
        )
    for group in table["group"].unique().to_list():
        print_group_summary(str(group), table.filter(pl.col("group") == group).to_dicts())


if __name__ == "__main__":
    main()
