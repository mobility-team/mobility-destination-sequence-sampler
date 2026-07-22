"""Compare bounded top-K work and quality on fixed oracle-proven contexts."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import polars as pl

from mobility_destination_sequence_sampler import DestinationPlanSearch

from experiments.analysis.compare_bidirectional_top_k_grand_geneve import (
    OracleCache,
    oracle_input_fingerprint,
    ranked_plans,
    retained_probability_mass,
)
from experiments.benchmarks.perf_grand_geneve_cache import (
    DEFAULT_GROUP_DAY_TRIPS_FOLDER,
    LOGIT_SCALE,
    prepare_complete_contexts,
    prepare_destination_inputs,
    prepare_od_costs,
    resolve_snapshot_files,
)


DEFAULT_CONTEXTS = (26, 45331, 3647, 2679, 61440, 3506, 57725, 61662)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group-day-trips-folder",
        type=Path,
        default=DEFAULT_GROUP_DAY_TRIPS_FOLDER,
    )
    parser.add_argument("--context-id", type=int, action="append", dest="context_ids")
    parser.add_argument("--cohort-size", type=int)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--oracle-depth", type=int, default=100)
    parser.add_argument("--max-states", type=int, default=2_000_000)
    parser.add_argument("--repeats", type=int, default=3)
    return parser.parse_args()


def recall_and_mass(
    oracle: list[tuple[tuple[int, ...], float]],
    bounded: list[tuple[tuple[int, ...], float]],
    top_k: int,
) -> tuple[float, float]:
    oracle_top_k = oracle[:top_k]
    bounded_zones = {zones for zones, _ in bounded}
    recall = sum(zones in bounded_zones for zones, _ in oracle_top_k) / len(
        oracle_top_k
    )
    return recall, retained_probability_mass(oracle_top_k, bounded_zones)


def cache_rate(report: dict, prefix: str) -> float:
    hits = report[f"factor_map_{prefix}_hits"]
    builds = report[f"factor_map_{prefix}_builds"]
    return hits / (hits + builds) if hits + builds else math.nan


def run_bounded(
    search: DestinationPlanSearch,
    context_steps: pl.DataFrame,
    context_initial: pl.DataFrame,
    strategy: str,
    repeats: int,
    top_k: int,
) -> tuple[pl.DataFrame, dict, float]:
    kwargs = dict(
        steps=context_steps,
        initial_locations=context_initial,
        logit_scale=LOGIT_SCALE,
        update_plan_timings=True,
        use_shadow_prices=True,
        exploration_seed=42,
        frontier_width=40,
        proposal_limit_per_source=16,
        symmetric_message_limit=4,
        symmetric_state_limit=4,
        symmetric_forward_proposal_limit=8,
        candidate_strategy=strategy,
        surface_bins=2,
        factor_map_max_depth=5,
        stitch_bias=1,
        continuation_state_limit=1,
        continuation_proposal_limit=1,
        seam_refresh_per_prefix=1,
        top_k=top_k,
        n_threads=1,
        skip_infeasible=False,
        collect_profile=True,
    )
    runs = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        table, report = search.top_k(**kwargs)
        runs.append((time.perf_counter_ns() - started, table, report))
    runs.sort(key=lambda run: run[0])
    wall_ns, table, report = runs[len(runs) // 2]
    return table, report, wall_ns / 1e6


def main() -> None:
    args = parse_args()
    if args.top_k <= 0 or args.oracle_depth < args.top_k or args.repeats <= 0:
        raise ValueError("top-k, oracle depth, and repeats are inconsistent")
    if args.context_ids and args.cohort_size:
        raise ValueError("pass either --context-id or --cohort-size")
    context_ids = list(dict.fromkeys(args.context_ids or DEFAULT_CONTEXTS))
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
    if args.cohort_size:
        if args.cohort_size <= 0:
            raise ValueError("--cohort-size must be positive")
        context_ids = (
            steps.group_by("context_id")
            .agg(layers=pl.len())
            .filter(pl.col("layers").is_between(3, 4))
            .with_columns(sample_order=pl.col("context_id").hash(seed=42))
            .sort("sample_order")
            .head(args.cohort_size * 6)["context_id"]
            .to_list()
        )
    missing = set(context_ids) - set(steps["context_id"].unique().to_list())
    if missing:
        raise ValueError(f"contexts do not exist: {sorted(missing)}")
    search = DestinationPlanSearch(od_costs=od_costs, destination_inputs=destination_inputs)
    fingerprint = oracle_input_fingerprint(
        files,
        logit_scale=LOGIT_SCALE,
        update_plan_timings=True,
        use_shadow_prices=True,
    )
    oracle_cache = OracleCache(fingerprint, args.oracle_depth, args.max_states)

    if not args.summary_only:
        print(
            "context | layers | variable/repeated anchors | strategy | recall | mass | "
            "wall ms | rust ms | map ms | map destinations | map builds | "
            "actual scans p/c/n | cache p/c/n | reverse partial | local cache | "
            "forward/backward proposals | complete/stitch"
        )
    results = []
    proven_contexts = 0
    for context_id in context_ids:
        context_steps = steps.filter(pl.col("context_id") == context_id).sort("layer")
        context_initial = initial_locations.filter(pl.col("context_id") == context_id)
        anchor_counts = (
            context_steps.filter(pl.col("anchor_id").is_not_null())
            .group_by("anchor_id")
            .len()
        )
        variable_anchors = anchor_counts.height
        repeated_anchors = anchor_counts.filter(pl.col("len") > 1).height

        def compute_oracle() -> tuple[pl.DataFrame, dict]:
            return search.exact_top_k(
                steps=context_steps,
                initial_locations=context_initial,
                logit_scale=LOGIT_SCALE,
                update_plan_timings=True,
                use_shadow_prices=True,
                top_k=args.oracle_depth,
                max_states=args.max_states,
                n_threads=1,
                skip_infeasible=False,
            )

        try:
            oracle_table, _, _ = oracle_cache.load_or_compute(context_id, compute_oracle)
        except ValueError:
            continue
        proven_contexts += 1
        oracle = ranked_plans(oracle_table)
        for strategy in ("heuristic", "factor_map", "symmetric_factor_map"):
            table, report, wall_ms = run_bounded(
                search,
                context_steps,
                context_initial,
                strategy,
                args.repeats,
                args.top_k,
            )
            recall, mass = recall_and_mass(oracle, ranked_plans(table), args.top_k)
            map_builds = sum(
                report[f"factor_map_{prefix}_builds"]
                for prefix in ("previous", "current", "next")
            )
            rates = "/".join(
                "-" if math.isnan(rate := cache_rate(report, prefix)) else f"{rate:.0%}"
                for prefix in ("previous", "current", "next")
            )
            scans = "/".join(
                str(report[f"factor_map_{prefix}_destination_scans"])
                for prefix in ("previous", "current", "next")
            )
            local_total = (
                report["local_score_cache_hits"] + report["local_score_cache_builds"]
            )
            local_rate = (
                report["local_score_cache_hits"] / local_total if local_total else math.nan
            )
            results.append(
                {
                    "context": context_id,
                    "strategy": strategy,
                    "mass": mass,
                    "rust_ms": report["total_search_ns"] / 1e6,
                    "map_ms": report["factor_map_ns"] / 1e6,
                    "scans": sum(
                        report[f"factor_map_{prefix}_destination_scans"]
                        for prefix in ("previous", "current", "next")
                    ),
                    "reverse_partial": report["reverse_prefix_partial_calls"],
                }
            )
            if not args.summary_only:
                print(
                    f"{context_id} | {context_steps.height} | "
                    f"{variable_anchors}/{repeated_anchors} | {strategy} | "
                    f"{recall:.3f} | {mass:.3f} | {wall_ms:.3f} | "
                    f"{report['total_search_ns'] / 1e6:.3f} | "
                    f"{report['factor_map_ns'] / 1e6:.3f} | "
                    f"{report['factor_map_destinations_evaluated']} | {map_builds} | "
                    f"{scans} | {rates} | {report['reverse_prefix_partial_calls']} | "
                    f"{local_rate:.0%} | {report['forward_proposals_evaluated']}/"
                    f"{report['backward_proposals_evaluated']} | "
                    f"{report['complete_plan_candidates']}/{report['stitch_pairs']}"
                )

        if args.cohort_size and proven_contexts == args.cohort_size:
            break

    result_frame = pl.DataFrame(results)
    print("\nstrategy aggregate")
    print(
        result_frame.group_by("strategy")
        .agg(
            contexts=pl.len(),
            mean_mass=pl.col("mass").mean(),
            mean_rust_ms=pl.col("rust_ms").mean(),
            mean_map_ms=pl.col("map_ms").mean(),
            mean_scans=pl.col("scans").mean(),
            mean_reverse_partial=pl.col("reverse_partial").mean(),
        )
        .sort("strategy")
    )
    symmetric = result_frame.filter(pl.col("strategy") == "symmetric_factor_map").with_columns(
        quality=pl.when(pl.col("mass") == 0)
        .then(pl.lit("zero"))
        .when(pl.col("mass") >= 0.8)
        .then(pl.lit("high"))
        .otherwise(pl.lit("partial"))
    )
    print("\nsymmetric work by outcome")
    print(
        symmetric.group_by("quality")
        .agg(
            contexts=pl.len(),
            mean_mass=pl.col("mass").mean(),
            mean_map_ms=pl.col("map_ms").mean(),
            mean_scans=pl.col("scans").mean(),
            mean_reverse_partial=pl.col("reverse_partial").mean(),
        )
        .sort("quality")
    )
    mass_wide = result_frame.pivot(on="strategy", index="context", values="mass")
    print("\nper-context quality comparisons")
    for left, right in (
        ("symmetric_factor_map", "factor_map"),
        ("symmetric_factor_map", "heuristic"),
        ("factor_map", "heuristic"),
    ):
        better = mass_wide.filter(pl.col(left) > pl.col(right))
        worse = mass_wide.filter(pl.col(left) < pl.col(right))
        print(
            f"{left} vs {right}: better={better.height} "
            f"worse={worse.height} equal={mass_wide.height - better.height - worse.height} "
            f"mean-delta={(mass_wide[left] - mass_wide[right]).mean():.3f}"
        )


if __name__ == "__main__":
    main()
