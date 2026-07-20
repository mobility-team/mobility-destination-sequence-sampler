"""Read-only raw-zone timing for the bidirectional top-K search.

This benchmark selects fixed-terminal contexts without variable anchors to
measure bounded top-K search cost using fixed candidate parameters.
"""

from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group-day-trips-folder",
        type=Path,
        default=DEFAULT_GROUP_DAY_TRIPS_FOLDER,
    )
    parser.add_argument("--contexts", type=int, default=1_000)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--frontier-width", type=int, default=32)
    parser.add_argument("--proposal-limit-per-source", type=int, default=16)
    parser.add_argument("--stitch-bias", type=int, default=0)
    parser.add_argument("--continuation-state-limit", type=int, default=1)
    parser.add_argument("--continuation-proposal-limit", type=int, default=1)
    parser.add_argument("--seam-refresh-per-prefix", type=int, default=1)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--exploration-seed", type=int, default=42)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="report Rust phase timings for the bidirectional search",
    )
    return parser.parse_args()


def eligible_contexts(steps: pl.DataFrame, count: int, seed: int) -> pl.DataFrame:
    """Choose fixed-terminal contexts the bounded stitch-layer search supports."""
    if count <= 0:
        raise ValueError("--contexts must be positive")
    eligible = (
        steps.group_by("context_id")
        .agg(
            layers=pl.len(),
            variable_anchors=(
                pl.col("fixed_destination").is_null()
                & pl.col("anchor_id").is_not_null()
            ).sum(),
            terminal_fixed=pl.col("fixed_destination")
            .sort_by("layer")
            .last()
            .is_not_null(),
        )
        .filter(
            (pl.col("layers") >= 3)
            & (pl.col("variable_anchors") == 0)
            & pl.col("terminal_fixed")
        )
        .with_columns(sample_order=pl.col("context_id").hash(seed=seed))
        .sort("sample_order")
        .head(count)
        .select("context_id")
    )
    if eligible.height < count:
        raise ValueError(
            f"only {eligible.height} contexts support the initial bidirectional top-K search"
        )
    return eligible


def main() -> None:
    args = parse_args()
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
    selected = eligible_contexts(steps, args.contexts, args.exploration_seed)
    steps = steps.join(selected, on="context_id", how="semi")
    initial_locations = initial_locations.join(selected, on="context_id", how="semi")
    search = DestinationPlanSearch(
        od_costs=od_costs,
        destination_inputs=destination_inputs,
    )
    started = time.perf_counter()
    bidirectional, bidirectional_report = search.top_k(
        steps=steps,
        initial_locations=initial_locations,
        logit_scale=LOGIT_SCALE,
        update_plan_timings=True,
        use_shadow_prices=True,
        exploration_seed=args.exploration_seed,
        frontier_width=args.frontier_width,
        proposal_limit_per_source=args.proposal_limit_per_source,
        stitch_bias=args.stitch_bias,
        continuation_state_limit=args.continuation_state_limit,
        continuation_proposal_limit=args.continuation_proposal_limit,
        seam_refresh_per_prefix=args.seam_refresh_per_prefix,
        top_k=args.top_k,
        n_threads=args.threads,
        skip_infeasible=True,
        collect_profile=args.profile,
    )
    bidirectional_seconds = time.perf_counter() - started

    print("\nbounded top-K")
    print(
        f"workload  contexts={initial_locations.height} steps={steps.height} "
        f"zones={od_costs['origin'].n_unique()} "
        f"top-k={args.top_k} "
        f"frontier-width={args.frontier_width} "
        f"stitch-bias={args.stitch_bias} "
        f"proposal-limit={args.proposal_limit_per_source} "
        f"continuation={args.continuation_state_limit}x{args.continuation_proposal_limit} "
        f"seam-refresh={args.seam_refresh_per_prefix} "
        f"threads={args.threads}"
    )
    print(
        f"bidir     wall={bidirectional_seconds:.3f}s "
        f"plans={bidirectional['context_id'].n_unique()} "
        f"complete-plan-candidates={bidirectional_report['complete_plan_candidates']} "
        f"forward-proposals={bidirectional_report['forward_proposals_evaluated']} "
        f"backward-proposals={bidirectional_report['backward_proposals_evaluated']} "
        f"refresh-proposals={bidirectional_report['seam_refresh_proposals']} "
        f"refresh-states={bidirectional_report['seam_refresh_states']} "
        f"stitch-pairs={bidirectional_report['stitch_pairs']} "
        f"infeasible={bidirectional_report['infeasible_contexts']}"
    )
    if args.profile:
        total_ns = bidirectional_report["total_search_ns"]
        if total_ns:
            print("bidir phase profile (aggregate Rust search time)")
            continuation_guidance_ns = bidirectional_report["continuation_guidance_ns"]
            for name, value in (
                ("build_problem", bidirectional_report["build_problem_ns"]),
                ("backward_search", bidirectional_report["backward_search_ns"]),
                ("backward_guidance", bidirectional_report["backward_guidance_ns"]),
                (
                    "forward_search (without continuation)",
                    bidirectional_report["forward_search_ns"]
                    - continuation_guidance_ns,
                ),
                ("continuation_guidance", continuation_guidance_ns),
                ("seam_refresh", bidirectional_report["seam_refresh_ns"]),
                ("stitch", bidirectional_report["stitch_ns"]),
                ("materialize", bidirectional_report["materialize_ns"]),
            ):
                print(f"  {name:<35} {value / 1e9:7.3f}s {value / total_ns:5.1%}")


if __name__ == "__main__":
    main()
