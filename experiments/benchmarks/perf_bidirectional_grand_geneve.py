"""Read-only raw-zone timing for the bidirectional top-K search.

This benchmark selects final-home contexts without variable anchors to
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
from experiments.top_k_config import add_top_k_tuning_arguments, top_k_tuning_options


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group-day-trips-folder",
        type=Path,
        default=DEFAULT_GROUP_DAY_TRIPS_FOLDER,
    )
    parser.add_argument("--contexts", type=int, default=1_000)
    parser.add_argument(
        "--all-supported",
        action="store_true",
        help="time every prepared depth>=2 context, including variable anchors",
    )
    parser.add_argument("--top-k", type=int, default=10)
    add_top_k_tuning_arguments(parser)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--exploration-seed", type=int, default=42)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="report Rust phase timings for the bidirectional search",
    )
    return parser.parse_args()


def eligible_contexts(
    steps: pl.DataFrame, count: int, seed: int, all_supported: bool
) -> pl.DataFrame:
    """Choose short final-home contexts for the bounded stitch-layer search."""
    if count <= 0:
        raise ValueError("--contexts must be positive")
    profiles = (
        steps.group_by("context_id")
        .agg(
            layers=pl.len(),
            variable_anchors=(
                pl.col("fixed_destination").is_null()
                & pl.col("anchor_id").is_not_null()
            ).sum(),
        )
    )
    if all_supported:
        return profiles.filter(pl.col("layers") >= 2).select("context_id")
    eligible = (
        profiles.filter((pl.col("layers") >= 3) & (pl.col("variable_anchors") == 0))
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
    selected = eligible_contexts(
        steps, args.contexts, args.exploration_seed, args.all_supported
    )
    steps = steps.join(selected, on="context_id", how="semi")
    initial_locations = initial_locations.join(selected, on="context_id", how="semi")
    search = DestinationPlanSearch(
        od_costs=od_costs,
        destination_inputs=destination_inputs,
    )
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    runs = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        bidirectional, bidirectional_report = search.top_k(
            steps=steps,
            initial_locations=initial_locations,
            logit_scale=LOGIT_SCALE,
            update_plan_timings=True,
            use_shadow_prices=True,
            exploration_seed=args.exploration_seed,
            **top_k_tuning_options(args),
            top_k=args.top_k,
            n_threads=args.threads,
            skip_infeasible=True,
            collect_profile=args.profile,
        )
        runs.append((time.perf_counter() - started, bidirectional, bidirectional_report))
    runs.sort(key=lambda run: run[0])
    bidirectional_seconds, bidirectional, bidirectional_report = runs[len(runs) // 2]

    print("\nbounded top-K")
    print(
        f"workload  contexts={initial_locations.height} steps={steps.height} "
        f"zones={od_costs['origin'].n_unique()} "
        f"top-k={args.top_k} "
        f"frontier-width={args.frontier_width} "
        f"stitch-bias={args.stitch_bias} "
        f"proposal-limit={args.proposal_limit_per_source} "
        f"symmetric-message-limit={args.symmetric_message_limit} "
        f"symmetric-state-limit={args.symmetric_state_limit} "
        f"symmetric-forward-proposal-limit={args.symmetric_forward_proposal_limit} "
        f"candidate-strategy={args.candidate_strategy} "
        f"surface-bins={args.surface_bins} "
        f"factor-map-max-depth={args.factor_map_max_depth} "
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
    print(
        "factor-map cache "
        f"previous={bidirectional_report['factor_map_previous_hits']}/"
        f"{bidirectional_report['factor_map_previous_builds']} hit/build "
        f"current={bidirectional_report['factor_map_current_hits']}/"
        f"{bidirectional_report['factor_map_current_builds']} hit/build "
        f"next={bidirectional_report['factor_map_next_hits']}/"
        f"{bidirectional_report['factor_map_next_builds']} hit/build"
    )
    print(
        "factor-map exact work "
        f"scans previous/current/next="
        f"{bidirectional_report['factor_map_previous_destination_scans']}/"
        f"{bidirectional_report['factor_map_current_destination_scans']}/"
        f"{bidirectional_report['factor_map_next_destination_scans']} "
        f"feasible="
        f"{bidirectional_report['factor_map_previous_feasible_entries']}/"
        f"{bidirectional_report['factor_map_current_feasible_entries']}/"
        f"{bidirectional_report['factor_map_next_feasible_entries']} "
        f"reverse-prefix-partials={bidirectional_report['reverse_prefix_partial_calls']} "
        f"local-score-cache hit/build={bidirectional_report['local_score_cache_hits']}/"
        f"{bidirectional_report['local_score_cache_builds']}"
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
                (
                    "surface_proposals",
                    bidirectional_report["surface_proposal_ns"],
                ),
                ("factor_map", bidirectional_report["factor_map_ns"]),
                ("seam_refresh", bidirectional_report["seam_refresh_ns"]),
                ("stitch", bidirectional_report["stitch_ns"]),
                ("materialize", bidirectional_report["materialize_ns"]),
            ):
                print(f"  {name:<35} {value / 1e9:7.3f}s {value / total_ns:5.1%}")


if __name__ == "__main__":
    main()
