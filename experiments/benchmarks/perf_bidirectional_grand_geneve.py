"""Read-only raw-zone timing for the bidirectional top-K search."""

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
    sample_mode = parser.add_mutually_exclusive_group()
    sample_mode.add_argument(
        "--all-supported",
        action="store_true",
        help="time every prepared depth>=2 context, including variable anchors",
    )
    sample_mode.add_argument(
        "--calibrated",
        action="store_true",
        help="sample depth and variable-anchor strata in workload proportions",
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


def calibrated_contexts(steps: pl.DataFrame, count: int, seed: int) -> pl.DataFrame:
    """Sample all depth/variable-anchor strata in workload proportions."""
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
        .filter(pl.col("layers") >= 2)
        .with_columns(
            depth_band=pl.when(pl.col("layers") >= 10)
            .then(pl.lit("10+"))
            .otherwise(pl.col("layers").cast(pl.String)),
            anchor_band=pl.when(pl.col("variable_anchors") > 0)
            .then(pl.lit("variable-anchor"))
            .otherwise(pl.lit("fixed-only")),
        )
        .with_columns(stratum=pl.concat_str(["depth_band", "anchor_band"], separator="|"))
    )
    populations = {
        stratum: int(population)
        for stratum, population in profiles.group_by("stratum").len().iter_rows()
    }
    if count > sum(populations.values()):
        raise ValueError("--contexts exceeds the supported calibration population")
    total_population = sum(populations.values())
    raw = {
        stratum: count * population / total_population
        for stratum, population in populations.items()
    }
    quotas = {
        stratum: min(populations[stratum], int(value))
        for stratum, value in raw.items()
    }
    for stratum in sorted(
        populations,
        key=lambda item: (raw[item] - quotas[item], item),
        reverse=True,
    ):
        if sum(quotas.values()) == count:
            break
        if quotas[stratum] < populations[stratum]:
            quotas[stratum] += 1
    selected = []
    for stratum, quota in quotas.items():
        if not quota:
            continue
        selected.extend(
            profiles.filter(pl.col("stratum") == stratum)
            .with_columns(sample_order=pl.col("context_id").hash(seed=seed))
            .sort("sample_order")
            .head(quota)["context_id"]
            .to_list()
        )
    return pl.DataFrame({"context_id": selected})


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
    if args.calibrated:
        selected = calibrated_contexts(steps, args.contexts, args.exploration_seed)
        sample_name = "calibrated"
    else:
        selected = eligible_contexts(
            steps, args.contexts, args.exploration_seed, args.all_supported
        )
        sample_name = "all-supported" if args.all_supported else "fixed-only"
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
    pair_route = (
        "local"
        if args.pricing_pair_deep_min_layers == 0
        else f"depth-{args.pricing_pair_deep_min_layers}+"
    )

    print("\nbounded top-K")
    print(
        f"workload  sample={sample_name} "
        f"contexts={initial_locations.height} steps={steps.height} "
        f"zones={od_costs['origin'].n_unique()} "
        f"top-k={args.top_k} "
        f"frontier-width={args.frontier_width} "
        f"stitch-bias={args.stitch_bias} "
        f"proposal-limit={args.proposal_limit_per_source} "
        f"symmetric-message-limit={args.symmetric_message_limit} "
        f"symmetric-state-limit={args.symmetric_state_limit} "
        f"symmetric-forward-proposal-limit={args.symmetric_forward_proposal_limit} "
        f"candidate-strategy={args.candidate_strategy} "
        f"factor-map-max-depth={args.factor_map_max_depth} "
        f"continuation={args.continuation_state_limit}x{args.continuation_proposal_limit} "
        f"deep-continuation={args.deep_continuation_state_limit} "
        f"seam-refresh={args.seam_refresh_per_prefix} "
        f"improvement={args.pricing_passes}x{args.pricing_column_limit} "
        f"pair-probe-limit={args.pricing_pair_candidate_limit} "
        f"pair-expansion-limit={args.pricing_pair_deep_candidate_limit}"
        f"({pair_route}) "
        f"next-pass-min-new={args.pricing_next_pass_min_new} "
        f"improvement-min-layers={args.pricing_min_layers} "
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
        f"improvement-rounds={bidirectional_report['pricing_rounds']} "
        f"improvement-evaluations={bidirectional_report['pricing_candidate_evaluations']} "
        f"pair-evaluations={bidirectional_report['pricing_pair_evaluations']} "
        f"pair-probes={bidirectional_report['pricing_pair_probes']} "
        f"pair-expansions={bidirectional_report['pricing_pair_expansions']} "
        f"improved-plans-added={bidirectional_report['pricing_plans_added']} "
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
                ("factor_map", bidirectional_report["factor_map_ns"]),
                ("seam_refresh", bidirectional_report["seam_refresh_ns"]),
                ("plan_improvement", bidirectional_report["pricing_ns"]),
                ("stitch", bidirectional_report["stitch_ns"]),
                ("materialize", bidirectional_report["materialize_ns"]),
            ):
                print(f"  {name:<35} {value / 1e9:7.3f}s {value / total_ns:5.1%}")


if __name__ == "__main__":
    main()
