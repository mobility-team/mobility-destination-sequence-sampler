"""Print cumulative exp(U) mass for one fully enumerable reference context."""

from __future__ import annotations

import argparse
import math
import time
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


def geometric_head_fit(scores: list[float], top_k: int) -> float:
    """Estimate top-K exp(U) mass from the head and the known support size.

    The score tail is fit as locally linear in rank, hence geometric in exp(U).
    This is intentionally a diagnostic baseline, not a claimed tail model.
    """
    head = [math.exp(score - scores[0]) for score in scores[:top_k]]
    window = min(10, top_k - 1)
    log_ratio = sum(
        scores[index + 1] - scores[index]
        for index in range(top_k - window - 1, top_k - 1)
    ) / window
    ratio = math.exp(log_ratio)
    remaining = len(scores) - top_k
    estimated_tail = (
        head[-1] * ratio * (1.0 - ratio**remaining) / (1.0 - ratio)
        if remaining and ratio < 1.0
        else 0.0
    )
    return sum(head) / (sum(head) + estimated_tail)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("context_id", type=int)
    parser.add_argument("--max-assignments", type=int, default=100_000)
    parser.add_argument(
        "--group-day-trips-folder", type=Path, default=DEFAULT_GROUP_DAY_TRIPS_FOLDER
    )
    args = parser.parse_args()
    files = resolve_snapshot_files(args.group_day_trips_folder)
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
    context_steps = steps.filter(pl.col("context_id") == args.context_id)
    context_initial = initial_locations.filter(pl.col("context_id") == args.context_id)
    if context_steps.is_empty() or context_initial.is_empty():
        raise ValueError(f"unknown context {args.context_id}")
    search = DestinationPlanSearch(od_costs=od_costs, destination_inputs=destination_inputs)
    started = time.perf_counter()
    distribution = search.exact_distribution(
        steps=context_steps,
        initial_locations=context_initial,
        logit_scale=LOGIT_SCALE,
        update_plan_timings=True,
        use_shadow_prices=True,
        max_assignments=args.max_assignments,
    )
    enumeration_ms = (time.perf_counter() - started) * 1e3
    scores = distribution["scores"]
    weights = [math.exp(score - distribution["log_normalizer"]) for score in scores]
    entropy = -sum(weight * math.log(weight) for weight in weights)
    print(
        f"context={args.context_id} layers={context_steps.height} "
        f"assignment-lattice={distribution['assignment_lattice']} "
        f"feasible={distribution['feasible_plans']} logZ={distribution['log_normalizer']:.6f} "
        f"enumeration-ms={enumeration_ms:.3f}"
    )
    print(f"effective-number-of-paths={math.exp(entropy):.2f}")
    print("rank | cumulative exp(U) mass | score")
    cumulative = 0.0
    for rank, (score, weight) in enumerate(zip(scores, weights, strict=True), start=1):
        cumulative += weight
        if rank in {1, 5, 10, 20, 50, 100, 200, 500} or rank == len(scores):
            print(f"{rank:4d} | {cumulative:.6f} | {score:.6f}")
    print("\nhead-only locally-geometric fit (given true feasible-path count for this test)")
    print("K | actual mass | fitted mass | error")
    for top_k in (10, 20, 50, 100):
        if top_k >= len(scores):
            continue
        actual = sum(weights[:top_k])
        fitted = geometric_head_fit(scores, top_k)
        print(f"{top_k:3d} | {actual:.6f} | {fitted:.6f} | {fitted - actual:+.6f}")


if __name__ == "__main__":
    main()
