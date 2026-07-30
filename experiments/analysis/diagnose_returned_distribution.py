"""Regular concentration diagnostic for bounded, exact-score-ranked top-K output.

All reported mass is normalized over the returned support, normally top-100.
It is a lower-bound/concentration diagnostic, never an estimate of total path
probability mass outside the returned plans.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import polars as pl

from mobility_destination_sequence_sampler import DestinationPlanSearch

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
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--exploration-seed", type=int, default=42)
    parser.add_argument("--show-worst", type=int, default=10)
    add_top_k_tuning_arguments(parser)
    parser.set_defaults(frontier_width=128)
    return parser.parse_args()


def logsumexp(scores: list[float]) -> float:
    maximum = max(scores)
    return maximum + math.log(sum(math.exp(score - maximum) for score in scores))


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] if lower == upper else ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize_context(context_id: int, scores: list[float], top_k: int) -> dict[str, float | int]:
    scores = sorted(scores, reverse=True)
    normalizer = logsumexp(scores)
    weights = [math.exp(score - normalizer) for score in scores]
    row: dict[str, float | int] = {
        "context_id": context_id,
        "returned": len(scores),
        "effective_paths": math.exp(-sum(weight * math.log(weight) for weight in weights)),
        "logz_returned": normalizer,
    }
    for cutoff in (10, 20, 50, top_k):
        if cutoff <= len(scores):
            row[f"mass_at_{cutoff}"] = sum(weights[:cutoff])
            row[f"logz_at_{cutoff}"] = logsumexp(scores[:cutoff])
    if len(scores) >= top_k:
        row["logz_gain_10_to_top_k"] = row[f"logz_at_{top_k}"] - row["logz_at_10"]
    return row


def main() -> None:
    args = parse_args()
    if args.contexts <= 0 or args.top_k < 10 or args.frontier_width < args.top_k:
        raise ValueError("contexts must be positive; top-k must be at least 10 and fit frontier-width")
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
    selected = eligible_contexts(steps, args.contexts, args.exploration_seed, False)
    steps = steps.join(selected, on="context_id", how="semi")
    initial_locations = initial_locations.join(selected, on="context_id", how="semi")
    search = DestinationPlanSearch(od_costs=od_costs, destination_inputs=destination_inputs)
    started = time.perf_counter()
    returned, report = search.top_k(
        steps=steps,
        initial_locations=initial_locations,
        logit_scale=LOGIT_SCALE,
        update_plan_timings=True,
        use_shadow_prices=True,
        exploration_seed=args.exploration_seed,
        **top_k_tuning_options(args),
        top_k=args.top_k,
        n_threads=args.threads,
        skip_contexts_without_plan=True,
    )
    elapsed = time.perf_counter() - started
    grouped = returned.group_by(["context_id", "draw_id"]).agg(
        pl.col("total_log_weight").first().alias("score")
    )
    rows = [
        summarize_context(int(context_id), scores, args.top_k)
        for context_id, scores in (
            grouped.group_by("context_id").agg(pl.col("score")).iter_rows()
        )
    ]
    frame = pl.DataFrame(rows)
    complete = frame.filter(pl.col("returned") >= args.top_k)
    print(
        f"returned-support diagnostic: contexts={frame.height}/{args.contexts} "
        f"complete-top-{args.top_k}={complete.height} wall={elapsed:.3f}s "
        f"contexts-without-plan={report['contexts_without_plan']}"
    )
    print("All mass is conditional on the returned support, not full exp(U) mass.")
    print("metric | mean | p10 | p50 | p90")
    for column in ("mass_at_10", "mass_at_20", "mass_at_50", f"mass_at_{args.top_k}", "effective_paths", "logz_gain_10_to_top_k"):
        values = complete[column].drop_nulls().to_list()
        if values:
            print(
                f"{column} | {sum(values) / len(values):.3f} | {quantile(values, .1):.3f} | "
                f"{quantile(values, .5):.3f} | {quantile(values, .9):.3f}"
            )
    if args.show_worst > 0:
        print(f"\nFlattest returned supports by mass_at_10 (showing {args.show_worst})")
        print(
            complete.sort("mass_at_10").head(args.show_worst).select(
                "context_id", "returned", "mass_at_10", "mass_at_50", "effective_paths", "logz_gain_10_to_top_k"
            )
        )


if __name__ == "__main__":
    main()
