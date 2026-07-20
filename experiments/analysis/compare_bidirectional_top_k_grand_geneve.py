"""Read-only top-K error check against the exact raw-zone oracle.

This intentionally uses short real contexts but keeps the complete raw-zone
domain. A context is included only when ``DestinationPlanSearch.exact_top_k`` proves its
oracle result within ``--max-states``; a state-limit error is reported and the
context is skipped rather than treated as an approximate oracle result.
"""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group-day-trips-folder",
        type=Path,
        default=DEFAULT_GROUP_DAY_TRIPS_FOLDER,
    )
    parser.add_argument("--contexts", type=int, default=10)
    parser.add_argument(
        "--candidate-contexts",
        type=int,
        default=50,
        help="short contexts to try before stopping after --contexts proven cases",
    )
    parser.add_argument("--max-layers", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--oracle-depth",
        type=int,
        default=100,
        help="exact plans retained as the conditional probability-mass support",
    )
    parser.add_argument("--max-states", type=int, default=2_000_000)
    parser.add_argument("--frontier-width", type=int, default=32)
    parser.add_argument("--proposal-limit-per-source", type=int, default=16)
    parser.add_argument("--stitch-bias", type=int, default=0)
    parser.add_argument("--continuation-state-limit", type=int, default=1)
    parser.add_argument("--continuation-proposal-limit", type=int, default=1)
    parser.add_argument(
        "--trace-context",
        type=int,
        help="inspect candidate-pool coverage for one known context ID",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="show every returned plan and every missed exact top-K plan",
    )
    parser.add_argument("--exploration-seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=8)
    return parser.parse_args()


def eligible_context_ids(steps: pl.DataFrame, args: argparse.Namespace) -> list[int]:
    """Select short contexts accepted by the current bidirectional contract."""
    if args.contexts <= 0 or args.candidate_contexts <= 0:
        raise ValueError("--contexts and --candidate-contexts must be positive")
    if args.max_layers < 3:
        raise ValueError("--max-layers must be at least three")
    if args.trace_context is not None:
        if steps.filter(pl.col("context_id") == args.trace_context).is_empty():
            raise ValueError(f"context {args.trace_context} does not exist")
        return [args.trace_context]
    selected = (
        steps.group_by("context_id")
        .agg(
            layers=pl.len(),
            variable_anchors=(
                pl.col("fixed_destination").is_null()
                & pl.col("anchor_id").is_not_null()
            ).sum(),
            fixed_layers=pl.col("fixed_destination").is_not_null().sum(),
            terminal_fixed=pl.col("fixed_destination")
            .sort_by("layer")
            .last()
            .is_not_null(),
        )
        .filter(
            (pl.col("layers") >= 3)
            & (pl.col("layers") <= args.max_layers)
            & (pl.col("variable_anchors") == 0)
            & (pl.col("fixed_layers") == 1)
            & pl.col("terminal_fixed")
        )
        .with_columns(
            sample_order=pl.col("context_id").hash(seed=args.exploration_seed)
        )
        .sort("sample_order")
        .head(args.candidate_contexts)
    )
    if selected.is_empty():
        raise ValueError("no eligible short contexts")
    return [int(context_id) for context_id in selected["context_id"].to_list()]


def ranked_plans(table: pl.DataFrame) -> list[tuple[tuple[int, ...], float]]:
    """Return plans sorted by their exact complete-plan utilities."""
    plans = []
    for rank in sorted(table["draw_id"].unique().to_list()):
        rows = table.filter(pl.col("draw_id") == rank).sort("layer")
        plans.append(
            (
                tuple(int(zone) for zone in rows["destination"].to_list()),
                float(rows.item(0, "total_log_weight")),
            )
        )
    return sorted(plans, key=lambda plan: (-plan[1], plan[0]))


def quantile(values: list[float], probability: float) -> float:
    """Linearly interpolated quantile for a small, dependency-free report."""
    if not values:
        raise ValueError("cannot summarize an empty distribution")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def print_distribution(label: str, values: list[float]) -> None:
    """Print a compact distribution report, including its high-error tail."""
    if not values:
        return
    summary = {
        "min": min(values),
        "p25": quantile(values, 0.25),
        "p50": quantile(values, 0.50),
        "p75": quantile(values, 0.75),
        "p90": quantile(values, 0.90),
        "p95": quantile(values, 0.95),
        "p99": quantile(values, 0.99),
        "max": max(values),
    }
    zero = sum(math.isclose(value, 0.0, abs_tol=1e-9) for value in values)
    print(
        f"{label}: n={len(values)} mean={sum(values) / len(values):.6f} "
        f"min={summary['min']:.6f} p25={summary['p25']:.6f} "
        f"p50={summary['p50']:.6f} p75={summary['p75']:.6f} "
        f"p90={summary['p90']:.6f} p95={summary['p95']:.6f} "
        f"p99={summary['p99']:.6f} max={summary['max']:.6f} "
        f"zero={zero}/{len(values)}"
    )


def retained_probability_mass(
    oracle: list[tuple[tuple[int, ...], float]],
    returned_zones: set[tuple[int, ...]],
) -> float:
    """Probability mass retained after normalizing over a finite oracle support."""
    if not oracle:
        return 0.0
    maximum_utility = max(score for _, score in oracle)
    weights = [(zones, math.exp(score - maximum_utility)) for zones, score in oracle]
    normalizer = sum(weight for _, weight in weights)
    return sum(weight for zones, weight in weights if zones in returned_zones) / normalizer


def stitch_layer_index(step_count: int, stitch_bias: int) -> int:
    """Return the bounded stitch layer used by the Rust kernel."""
    return max(0, min(step_count - 2, (step_count - 1) // 2 + stitch_bias))


def base_candidate_sources(
    od_costs: pl.DataFrame,
    destination_inputs: pl.DataFrame,
    activity_id: int,
    reference_zone: int,
    reverse: bool,
    candidate_count: int,
) -> tuple[set[int], set[int]]:
    """Mirror the deterministic 16 attractive + 16 OD-cost candidate pools."""
    activity_values = (
        destination_inputs.filter(
            (pl.col("activity_id") == activity_id)
            & (pl.col("opportunity_capacity") > 0)
        )
        .with_columns(
            proposal_attraction=pl.col("opportunity_capacity").log()
            + pl.col("shadow_price")
        )
    )
    attractive = set(
        activity_values.sort(
            ["proposal_attraction", "destination"], descending=[True, False]
        )
        .head(candidate_count)["destination"]
        .to_list()
    )
    if reverse:
        nearby = (
            od_costs.filter(pl.col("destination") == reference_zone)
            .sort(["cost", "origin"])
            .head(candidate_count)
            .rename({"origin": "candidate"})
            .join(
                activity_values.select(pl.col("destination").alias("candidate")),
                on="candidate",
                how="inner",
            )
            .select("candidate")
        )
    else:
        nearby = (
            od_costs.filter(pl.col("origin") == reference_zone)
            .sort(["cost", "destination"])
            .head(candidate_count)
            .join(activity_values.select("destination"), on="destination", how="inner")
            .select("destination")
        )
    return attractive, set(nearby.to_series().to_list())


def trace_oracle_candidate_coverage(
    context_steps: pl.DataFrame,
    initial_zone: int,
    oracle: list[tuple[tuple[int, ...], float]],
    od_costs: pl.DataFrame,
    destination_inputs: pl.DataFrame,
    candidate_count: int,
    top_k: int,
    stitch_bias: int,
) -> None:
    """Identify whether each true top-K path is lost before beam selection."""
    steps = context_steps.sort("layer").to_dicts()
    stitch_layer = stitch_layer_index(len(steps), stitch_bias)
    print(
        f"candidate trace: stitch-layer={stitch_layer}, "
        f"base pool={candidate_count}+{candidate_count}"
    )
    for rank, (zones, _) in enumerate(oracle[:top_k], start=1):
        missing_base_pool = False
        print(f"  exact rank={rank} zones={zones}")
        for layer, (step, target) in enumerate(zip(steps, zones, strict=True)):
            if step["fixed_destination"] is not None:
                print(f"    layer={layer}: fixed destination {target}")
                continue
            reverse = stitch_layer < layer < len(steps) - 1
            reference = zones[layer + 1] if reverse else (
                initial_zone if layer == 0 else zones[layer - 1]
            )
            attractive, nearby = base_candidate_sources(
                od_costs,
                destination_inputs,
                step["activity_id"],
                reference,
                reverse,
                candidate_count,
            )
            sources = []
            if target in attractive:
                sources.append("attractive")
            if target in nearby:
                sources.append("cost-near")
            if not sources:
                missing_base_pool = True
                sources.append("not-in-base-pool")
            direction = "backward" if reverse else "forward"
            print(
                f"    layer={layer} {direction} reference={reference} target={target}: "
                f"{', '.join(sources)}"
            )
        if missing_base_pool:
            print("    diagnosis: candidate-truncated unless one of two exploration draws restores it")
        else:
            print("    diagnosis: base-supported; if absent from output, a front beam pruned it")


def trace_first_layer_and_plan_components(
    context_steps: pl.DataFrame,
    initial_zone: int,
    oracle_table: pl.DataFrame,
    oracle: list[tuple[tuple[int, ...], float]],
    od_costs: pl.DataFrame,
    destination_inputs: pl.DataFrame,
    top_k: int,
) -> None:
    """Show why the exact first destinations escape the one-step proposal."""
    steps = context_steps.sort("layer").to_dicts()
    first_activity = steps[0]["activity_id"]
    first_destinations = {zones[0] for zones, _ in oracle[:top_k]}
    activity_values = (
        destination_inputs.filter(
            (pl.col("activity_id") == first_activity)
            & (pl.col("opportunity_capacity") > 0)
        )
        .with_columns(
            proposal_attraction=pl.col("opportunity_capacity").log()
            + pl.col("shadow_price")
        )
        .sort(["proposal_attraction", "destination"], descending=[True, False])
        .with_row_index("attraction_rank", offset=1)
    )
    first_leg = (
        od_costs.filter(pl.col("origin") == initial_zone)
        .sort(["cost", "destination"])
        .with_row_index("cost_rank", offset=1)
    )
    print("first-layer exact destinations: global-attraction and home-leg cost ranks")
    print(
        activity_values.filter(pl.col("destination").is_in(first_destinations))
        .join(first_leg, on="destination", how="left")
        .select(
            "destination",
            "attraction_rank",
            "proposal_attraction",
            "cost_rank",
            "cost",
            "time",
        )
        .sort("destination")
        .to_dicts()
    )

    by_zones: dict[tuple[int, ...], list[float]] = {}
    for draw_id in oracle_table["draw_id"].unique().to_list():
        rows = oracle_table.filter(pl.col("draw_id") == draw_id).sort("layer")
        by_zones[tuple(int(zone) for zone in rows["destination"].to_list())] = [
            float(value) for value in rows["local_log_weight"].to_list()
        ]
    leg_origins = {initial_zone}
    for zones, _ in oracle[:top_k]:
        leg_origins.update(zones[:-1])
    cost_by_leg = {
        (int(row["origin"]), int(row["destination"])): float(row["cost"])
        for row in od_costs.filter(
            pl.col("origin").is_in(leg_origins)
        ).to_dicts()
    }
    print("exact top-K local utility factors and travel costs")
    for rank, (zones, _) in enumerate(oracle[:top_k], start=1):
        origins = [initial_zone, *zones[:-1]]
        costs = [
            cost_by_leg[(origin, destination)]
            for origin, destination in zip(origins, zones, strict=True)
        ]
        print(
            f"  exact rank={rank:2d} zones={zones} "
            f"local={tuple(round(value, 3) for value in by_zones[zones])} "
            f"leg-cost={tuple(round(value, 3) for value in costs)}"
        )


def is_base_supported(
    context_steps: pl.DataFrame,
    initial_zone: int,
    zones: tuple[int, ...],
    od_costs: pl.DataFrame,
    destination_inputs: pl.DataFrame,
    candidate_count: int,
    stitch_bias: int,
) -> bool:
    """Whether a complete plan is reachable without either exploration draw."""
    steps = context_steps.sort("layer").to_dicts()
    stitch_layer = stitch_layer_index(len(steps), stitch_bias)
    for layer, (step, target) in enumerate(zip(steps, zones, strict=True)):
        if step["fixed_destination"] is not None:
            continue
        reverse = stitch_layer < layer < len(steps) - 1
        reference = zones[layer + 1] if reverse else (
            initial_zone if layer == 0 else zones[layer - 1]
        )
        attractive, nearby = base_candidate_sources(
            od_costs,
            destination_inputs,
            step["activity_id"],
            reference,
            reverse,
            candidate_count,
        )
        if target not in attractive and target not in nearby:
            return False
    return True


def show_context(
    context_id: int,
    oracle: list[tuple[tuple[int, ...], float]],
    bounded: list[tuple[tuple[int, ...], float]],
    top_k: int,
    states_pushed: int,
    context_steps: pl.DataFrame,
    initial_zone: int,
    od_costs: pl.DataFrame,
    destination_inputs: pl.DataFrame,
    candidate_count: int,
    stitch_bias: int,
    verbose: bool,
) -> dict[str, float]:
    oracle_top_k = oracle[:top_k]
    oracle_rank = {zones: rank for rank, (zones, _) in enumerate(oracle, start=1)}
    bounded_zones = {zones for zones, _ in bounded}
    hits = sum(zones in bounded_zones for zones, _ in oracle_top_k)
    recall = hits / len(oracle_top_k) if oracle_top_k else 0.0
    retained_top_k_mass = retained_probability_mass(oracle_top_k, bounded_zones)
    retained_oracle_mass = retained_probability_mass(oracle, bounded_zones)
    top_k_oracle_mass = retained_probability_mass(oracle, {zones for zones, _ in oracle_top_k})
    missed_base_pool_mass = 0.0
    missed_beam_mass = 0.0
    missed_diagnoses: dict[tuple[int, ...], str] = {}
    maximum_utility = max(score for _, score in oracle_top_k)
    top_k_normalizer = sum(
        math.exp(score - maximum_utility) for _, score in oracle_top_k
    )
    for zones, score in oracle_top_k:
        if zones in bounded_zones:
            continue
        probability = math.exp(score - maximum_utility) / top_k_normalizer
        if is_base_supported(
            context_steps,
            initial_zone,
            zones,
            od_costs,
            destination_inputs,
            candidate_count,
            stitch_bias,
        ):
            missed_beam_mass += probability
            missed_diagnoses[zones] = "base-supported, beam-lost"
        else:
            missed_base_pool_mass += probability
            missed_diagnoses[zones] = "not-in-base-pool"
    print(
        f"context={context_id} oracle-top-k={len(oracle_top_k)} "
        f"bounded={len(bounded)} recall@{top_k}={hits}/{len(oracle_top_k)} "
        f"retained-top-{top_k}-mass={retained_top_k_mass:.4f} "
        f"retained-oracle-{len(oracle)}-mass={retained_oracle_mass:.4f} "
        f"exact-states={states_pushed}"
    )
    if verbose:
        for rank, (zones, score) in enumerate(bounded, start=1):
            oracle_position = oracle_rank.get(zones)
            position = (
                str(oracle_position)
                if oracle_position is not None
                else f">{len(oracle)}"
            )
            print(
                f"  bounded rank={rank:2d} exact-rank={position:>4} "
                f"utility={score:.6f} zones={zones}"
            )
        print("  exact top-K probability diagnosis")
        for rank, (zones, score) in enumerate(oracle_top_k, start=1):
            probability = math.exp(score - maximum_utility) / top_k_normalizer
            status = "returned" if zones in bounded_zones else missed_diagnoses[zones]
            print(
                f"    exact rank={rank:2d} probability={probability:.4f} "
                f"{status:>25} zones={zones}"
            )
    return {
        "recall": recall,
        "retained_top_k_mass": retained_top_k_mass,
        "retained_oracle_mass": retained_oracle_mass,
        "top_k_oracle_mass": top_k_oracle_mass,
        "top_k_mass_efficiency": retained_oracle_mass / top_k_oracle_mass,
        "missed_base_pool_mass": missed_base_pool_mass,
        "missed_beam_mass": missed_beam_mass,
    }


def main() -> None:
    args = parse_args()
    if args.top_k <= 0 or args.oracle_depth < args.top_k:
        raise ValueError("--top-k must be positive and --oracle-depth must be at least --top-k")
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

    proven = 0
    skipped = 0
    recalls = []
    retained_top_k_masses = []
    retained_oracle_masses = []
    top_k_oracle_masses = []
    top_k_mass_efficiencies = []
    missed_base_pool_masses = []
    missed_beam_masses = []
    exact_search_seconds = 0.0
    bounded_search_seconds = 0.0
    started = time.perf_counter()
    for context_id in eligible_context_ids(steps, args):
        context_steps = steps.filter(pl.col("context_id") == context_id)
        context_initial = initial_locations.filter(pl.col("context_id") == context_id)
        try:
            oracle_started = time.perf_counter()
            oracle_table, oracle_report = search.exact_top_k(
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
            exact_search_seconds += time.perf_counter() - oracle_started
        except ValueError as error:
            skipped += 1
            print(f"context={context_id} oracle skipped: {error}")
            continue
        try:
            bounded_started = time.perf_counter()
            bounded_table, _ = search.top_k(
                steps=context_steps,
                initial_locations=context_initial,
                logit_scale=LOGIT_SCALE,
                update_plan_timings=True,
                use_shadow_prices=True,
                exploration_seed=args.exploration_seed,
                frontier_width=args.frontier_width,
                proposal_limit_per_source=args.proposal_limit_per_source,
                stitch_bias=args.stitch_bias,
                continuation_state_limit=args.continuation_state_limit,
                continuation_proposal_limit=args.continuation_proposal_limit,
                top_k=args.top_k,
                n_threads=1,
                skip_infeasible=False,
            )
            bounded_search_seconds += time.perf_counter() - bounded_started
        except ValueError as error:
            skipped += 1
            print(f"context={context_id} bounded search failed: {error}")
            continue
        oracle = ranked_plans(oracle_table)
        bounded = ranked_plans(bounded_table)
        if args.trace_context == context_id:
            trace_oracle_candidate_coverage(
                context_steps,
                int(context_initial.item(0, "initial_zone")),
                oracle,
                od_costs,
                destination_inputs,
                args.proposal_limit_per_source,
                args.top_k,
                args.stitch_bias,
            )
            trace_first_layer_and_plan_components(
                context_steps,
                int(context_initial.item(0, "initial_zone")),
                oracle_table,
                oracle,
                od_costs,
                destination_inputs,
                args.top_k,
            )
        metrics = show_context(
            context_id,
            oracle,
            bounded,
            args.top_k,
            oracle_report["states_pushed"],
            context_steps,
            int(context_initial.item(0, "initial_zone")),
            od_costs,
            destination_inputs,
            args.proposal_limit_per_source,
            args.stitch_bias,
            args.verbose or args.trace_context is not None,
        )
        recalls.append(metrics["recall"])
        retained_top_k_masses.append(metrics["retained_top_k_mass"])
        retained_oracle_masses.append(metrics["retained_oracle_mass"])
        top_k_oracle_masses.append(metrics["top_k_oracle_mass"])
        top_k_mass_efficiencies.append(metrics["top_k_mass_efficiency"])
        missed_base_pool_masses.append(metrics["missed_base_pool_mass"])
        missed_beam_masses.append(metrics["missed_beam_mass"])
        proven += 1
        if proven == args.contexts:
            break
    if not proven:
        raise RuntimeError("no context completed the exact top-K oracle")
    print(
        f"\nproven-contexts={proven} skipped={skipped} "
        f"mean-recall@{args.top_k}={sum(recalls) / proven:.4f} "
        f"wall={time.perf_counter() - started:.3f}s"
    )
    print(
        f"search-only exact={exact_search_seconds:.3f}s "
        f"({exact_search_seconds / proven * 1e3:.2f}ms/context) "
        f"bounded={bounded_search_seconds:.3f}s "
        f"({bounded_search_seconds / proven * 1e3:.2f}ms/context) "
        f"speedup={exact_search_seconds / bounded_search_seconds:.1f}x"
    )
    print_distribution(
        f"retained exact top-{args.top_k} probability mass (conditional)",
        retained_top_k_masses,
    )
    print_distribution(
        f"retained exact top-{args.oracle_depth} probability mass (conditional)",
        retained_oracle_masses,
    )
    print_distribution(
        f"exact top-{args.top_k} share of top-{args.oracle_depth} probability mass",
        top_k_oracle_masses,
    )
    print_distribution(
        f"top-{args.top_k} mass efficiency (1 = exact top-{args.top_k} support)",
        top_k_mass_efficiencies,
    )
    print(
        f"missed exact top-{args.top_k} mass by cause (per-context mean): "
        f"not-in-base-pool={sum(missed_base_pool_masses) / proven:.4f} "
        f"base-supported-but-beam-lost={sum(missed_beam_masses) / proven:.4f}"
    )


if __name__ == "__main__":
    main()
