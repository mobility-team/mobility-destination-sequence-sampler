"""Read-only validation of bounded destination-sequence proposals.

The exact run deliberately uses a small raw-zone subset. It is an oracle for
proposal quality, not an attempt to make raw-zone enumeration scalable. The
full-zone run measures only particle feasibility, candidate work, ESS, and
wall time.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import polars as pl

from mobility_destination_sequence_sampler._core import (
    ExperimentalDestinationSampler as DestinationSampler,
)

from experiments.benchmarks.perf_grand_geneve_cache import (
    DEFAULT_GROUP_DAY_TRIPS_FOLDER,
    LOGIT_SCALE,
    prepare_complete_contexts,
    prepare_destination_inputs,
    prepare_od_costs,
    resolve_snapshot_files,
    select_contexts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group-day-trips-folder",
        type=Path,
        default=DEFAULT_GROUP_DAY_TRIPS_FOLDER,
    )
    parser.add_argument("--exact-contexts", type=int, default=10)
    parser.add_argument("--exact-zones", type=int, default=8)
    parser.add_argument("--exact-draws", type=int, default=256)
    parser.add_argument("--max-assignments", type=int, default=100_000)
    parser.add_argument("--particle-contexts", type=int, default=1_000)
    parser.add_argument("--particles", type=int, default=32)
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--radiation-plans",
        type=Path,
        help=(
            "Optional read-only parquet with context_id, draw_id, layer, and "
            "either final_utility or total_log_weight."
        ),
    )
    return parser.parse_args()


def shared_home_contexts(
    steps: pl.DataFrame,
    initial_locations: pl.DataFrame,
    count: int,
    seed: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Pick short contexts from one home zone so a tiny-zone oracle is valid."""
    eligible = (
        steps.group_by("context_id")
        .agg(
            variable_layers=pl.col("fixed_destination").is_null().sum(),
            variable_anchors=(
                pl.col("fixed_destination").is_null()
                & pl.col("anchor_id").is_not_null()
            ).sum(),
            fixed_layers=pl.col("fixed_destination").is_not_null().sum(),
            n_layers=pl.len(),
        )
        .join(initial_locations, on="context_id")
        .filter(
            (pl.col("variable_layers") <= 4)
            & (pl.col("variable_anchors") == 0)
            & (pl.col("fixed_layers") == 1)
            & (pl.col("n_layers") >= 3)
            & (pl.col("n_layers") <= 6)
        )
    )
    if eligible.is_empty():
        raise ValueError("no short contexts are available for exact comparison")
    home_zone = (
        eligible.group_by("initial_zone")
        .len()
        .sort(["len", "initial_zone"], descending=[True, False])
        .item(0, "initial_zone")
    )
    selected = (
        eligible.filter(pl.col("initial_zone") == home_zone)
        .with_columns(sample_order=pl.col("context_id").hash(seed=seed))
        .sort("sample_order")
        .head(count)
        .select("context_id")
    )
    if selected.height < count:
        raise ValueError(
            f"home zone {home_zone} has only {selected.height} short contexts; "
            "lower --exact-contexts"
        )
    return (
        steps.join(selected, on="context_id", how="semi"),
        initial_locations.join(selected, on="context_id", how="semi"),
    )


def restrict_to_oracle_zones(
    od_costs: pl.DataFrame,
    destination_inputs: pl.DataFrame,
    initial_locations: pl.DataFrame,
    zone_count: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    homes = initial_locations["initial_zone"].unique().to_list()
    if len(homes) != 1:
        raise ValueError("exact comparison must use one shared home zone")
    if zone_count < 2:
        raise ValueError("--exact-zones must be at least two")
    attractive = (
        destination_inputs.group_by("destination")
        .agg(attraction=pl.col("opportunity_capacity").sum())
        .filter(pl.col("destination") != homes[0])
        .sort(["attraction", "destination"], descending=[True, False])
        .head(zone_count - 1)["destination"]
        .to_list()
    )
    zones = homes + attractive
    return (
        od_costs.filter(
            pl.col("origin").is_in(zones) & pl.col("destination").is_in(zones)
        ),
        destination_inputs.filter(pl.col("destination").is_in(zones)),
    )


def run_exact(
    sampler: DestinationSampler,
    steps: pl.DataFrame,
    initial_locations: pl.DataFrame,
    args: argparse.Namespace,
) -> tuple[pl.DataFrame, float]:
    started = time.perf_counter()
    result = sampler.sample_ternary_reference(
        steps=steps,
        initial_locations=initial_locations,
        logit_scale=LOGIT_SCALE,
        update_plan_timings=True,
        use_shadow_prices=True,
        seed=args.seed,
        n_draws=args.exact_draws,
        max_assignments=args.max_assignments,
        skip_infeasible=True,
    )
    return result, time.perf_counter() - started


def run_particles(
    sampler: DestinationSampler,
    steps: pl.DataFrame,
    initial_locations: pl.DataFrame,
    args: argparse.Namespace,
) -> tuple[pl.DataFrame, dict[str, int | float], float]:
    started = time.perf_counter()
    result, report = sampler.sample_particles(
        steps=steps,
        initial_locations=initial_locations,
        logit_scale=LOGIT_SCALE,
        update_plan_timings=True,
        use_shadow_prices=True,
        seed=args.seed,
        n_particles=args.particles,
        candidate_count=args.candidates,
        max_retries=args.max_retries,
        # Retain every completed particle for fair marginal comparisons.
        n_draws=args.particles,
        n_threads=args.threads,
        skip_infeasible=True,
    )
    return result, report, time.perf_counter() - started


def run_bidirectional(
    sampler: DestinationSampler,
    steps: pl.DataFrame,
    initial_locations: pl.DataFrame,
    args: argparse.Namespace,
) -> tuple[pl.DataFrame, dict[str, int], float]:
    started = time.perf_counter()
    result, report = sampler.search_bidirectional_top_k(
        steps=steps,
        initial_locations=initial_locations,
        logit_scale=LOGIT_SCALE,
        update_plan_timings=True,
        use_shadow_prices=True,
        seed=args.seed,
        beam_width=args.particles,
        candidate_count=args.candidates,
        top_k=args.particles,
        n_threads=args.threads,
        skip_infeasible=True,
    )
    return result, report, time.perf_counter() - started


def plan_utilities(plans: pl.DataFrame) -> pl.DataFrame:
    return plans.filter(pl.col("layer") == 0).select(
        "context_id", "draw_id", final_utility=pl.col("total_log_weight")
    )


def utility_summary(plans: pl.DataFrame) -> str:
    utilities = plan_utilities(plans)
    if utilities.is_empty():
        return "no feasible plans"
    best = utilities.group_by("context_id").agg(pl.col("final_utility").max())
    return (
        f"contexts={utilities['context_id'].n_unique()} "
        f"mean-best={best['final_utility'].mean():.4f} "
        f"p50={utilities['final_utility'].median():.4f}"
    )


def weighted_first_destination_tv(exact: pl.DataFrame, particle: pl.DataFrame) -> float:
    """Mean context-level TV with self-normalized particle weights.

    Global normalisation would let high-utility contexts dominate the reported
    destination marginal, even though the exact reference gives every context
    the same number of draws.
    """
    if exact.is_empty() or particle.is_empty():
        return float("nan")
    shared_contexts = particle.select("context_id").unique()
    exact = exact.join(shared_contexts, on="context_id", how="semi")
    exact_probability = (
        exact.filter(pl.col("layer") == 0)
        .group_by(["context_id", "destination"])
        .len()
        .with_columns(probability=pl.col("len") / pl.col("len").sum().over("context_id"))
        .select("context_id", "destination", "probability")
    )
    weights = particle.filter(pl.col("layer") == 0).with_columns(
        weight=(
            pl.col("importance_log_weight")
            - pl.col("importance_log_weight").max().over("context_id")
        ).exp()
    )
    particle_probability = (
        weights.group_by(["context_id", "destination"])
        .agg(weight=pl.col("weight").sum())
        .with_columns(probability=pl.col("weight") / pl.col("weight").sum().over("context_id"))
        .select(
            "context_id",
            "destination",
            pl.col("probability").alias("particle_probability"),
        )
    )
    comparison = exact_probability.join(
        particle_probability,
        on=["context_id", "destination"],
        how="full",
        coalesce=True,
    ).fill_null(0.0)
    per_context = comparison.group_by("context_id").agg(
        tv=0.5
        * (pl.col("probability") - pl.col("particle_probability"))
        .abs()
        .sum()
    )
    return float(per_context["tv"].mean())


def first_destination_comparison(
    reference: pl.DataFrame, generated: pl.DataFrame
) -> pl.DataFrame:
    """Compare first-destination probabilities without proposal correction."""
    if reference.is_empty() or generated.is_empty():
        return pl.DataFrame(
            schema={
                "context_id": pl.UInt64,
                "destination": pl.UInt32,
                "reference_probability": pl.Float64,
                "generated_probability": pl.Float64,
            }
        )
    shared_contexts = generated.select("context_id").unique()
    reference = reference.join(shared_contexts, on="context_id", how="semi")

    def probabilities(frame: pl.DataFrame, name: str) -> pl.DataFrame:
        return (
            frame.filter(pl.col("layer") == 0)
            .group_by(["context_id", "destination"])
            .len()
            .with_columns(probability=pl.col("len") / pl.col("len").sum().over("context_id"))
            .select("context_id", "destination", pl.col("probability").alias(name))
        )

    return probabilities(reference, "reference_probability").join(
        probabilities(generated, "generated_probability"),
        on=["context_id", "destination"],
        how="full",
        coalesce=True,
    ).fill_null(0.0)


def empirical_first_destination_tv(reference: pl.DataFrame, generated: pl.DataFrame) -> float:
    """Mean context-level TV for plans without proposal-weight correction."""
    comparison = first_destination_comparison(reference, generated)
    if comparison.is_empty():
        return float("nan")
    per_context = comparison.group_by("context_id").agg(
        tv=0.5
        * (pl.col("reference_probability") - pl.col("generated_probability"))
        .abs()
        .sum()
    )
    return float(per_context["tv"].mean())


def score_weighted_first_destination_comparison(
    reference: pl.DataFrame, generated: pl.DataFrame
) -> pl.DataFrame:
    """Compare exact draws with the generated plans' normalized model scores."""
    if reference.is_empty() or generated.is_empty():
        return pl.DataFrame(
            schema={
                "context_id": pl.UInt64,
                "destination": pl.UInt32,
                "reference_probability": pl.Float64,
                "score_weighted_probability": pl.Float64,
            }
        )
    shared_contexts = generated.select("context_id").unique()
    reference = reference.join(shared_contexts, on="context_id", how="semi")
    reference_probability = (
        reference.filter(pl.col("layer") == 0)
        .group_by(["context_id", "destination"])
        .len()
        .with_columns(probability=pl.col("len") / pl.col("len").sum().over("context_id"))
        .select("context_id", "destination", "probability")
    )
    generated_probability = (
        generated.filter(pl.col("layer") == 0)
        .with_columns(
            weight=(
                pl.col("total_log_weight")
                - pl.col("total_log_weight").max().over("context_id")
            ).exp()
        )
        .group_by(["context_id", "destination"])
        .agg(weight=pl.col("weight").sum())
        .with_columns(probability=pl.col("weight") / pl.col("weight").sum().over("context_id"))
        .select(
            "context_id",
            "destination",
            pl.col("probability").alias("score_weighted_probability"),
        )
    )
    return reference_probability.join(
        generated_probability,
        on=["context_id", "destination"],
        how="full",
        coalesce=True,
    ).fill_null(0.0).rename({"probability": "reference_probability"})


def score_weighted_first_destination_tv(
    reference: pl.DataFrame, generated: pl.DataFrame
) -> float:
    comparison = score_weighted_first_destination_comparison(reference, generated)
    if comparison.is_empty():
        return float("nan")
    per_context = comparison.group_by("context_id").agg(
        tv=0.5
        * (pl.col("reference_probability") - pl.col("score_weighted_probability"))
        .abs()
        .sum()
    )
    return float(per_context["tv"].mean())


def print_worst_bidirectional_example(
    exact: pl.DataFrame,
    generated: pl.DataFrame,
    steps: pl.DataFrame,
    initial_locations: pl.DataFrame,
    od_costs: pl.DataFrame,
    destination_inputs: pl.DataFrame,
) -> None:
    comparison = first_destination_comparison(exact, generated)
    if comparison.is_empty():
        return
    score_weighted = score_weighted_first_destination_comparison(exact, generated)
    per_context = (
        score_weighted.group_by("context_id")
        .agg(
            tv=0.5
            * (
                pl.col("reference_probability")
                - pl.col("score_weighted_probability")
            )
            .abs()
            .sum()
        )
        .sort("tv", descending=True)
    )
    context_id = per_context.item(0, "context_id")
    print(
        "\nworst score-weighted bidirectional first-destination "
        f"context={context_id} tv={per_context.item(0, 'tv'):.4f}"
    )
    print(
        steps.filter(pl.col("context_id") == context_id)
        .select(
            "layer",
            "activity_id",
            "fixed_destination",
            "departure_time",
            "arrival_time",
            "next_departure_time",
            "arrival_time_rigidity",
            "departure_time_rigidity",
        )
    )
    first_step = (
        steps.filter((pl.col("context_id") == context_id) & (pl.col("layer") == 0))
        .select("activity_id")
        .item(0, "activity_id")
    )
    origin = initial_locations.filter(pl.col("context_id") == context_id).item(
        0, "initial_zone"
    )
    print("first-layer endpoint proposal scores")
    print(
        destination_inputs.filter(pl.col("activity_id") == first_step)
        .join(
            od_costs.filter(pl.col("origin") == origin).select(
                "destination", "cost", "time"
            ),
            on="destination",
            how="inner",
        )
        .with_columns(
            endpoint_score=pl.col("opportunity_capacity").log()
            - pl.lit(LOGIT_SCALE) * pl.col("cost")
        )
        .select(
            "destination",
            "opportunity_capacity",
            "shadow_price",
            "cost",
            "time",
            "endpoint_score",
        )
        .sort("endpoint_score", descending=True)
    )
    print(
        comparison.filter(pl.col("context_id") == context_id)
        .join(
            score_weighted.filter(pl.col("context_id") == context_id).select(
                "destination", "score_weighted_probability"
            ),
            on="destination",
            how="left",
        )
        .select(
            "destination",
            "reference_probability",
            "generated_probability",
            "score_weighted_probability",
        )
        .sort("reference_probability", descending=True)
    )
    print("bidirectional complete plans (first 10)")
    plans = generated.filter(pl.col("context_id") == context_id)
    for plan in plans.partition_by("draw_id", maintain_order=True)[:10]:
        ordered = plan.sort("layer")
        sequence = " -> ".join(str(zone) for zone in ordered["destination"].to_list())
        print(
            f"draw={ordered.item(0, 'draw_id')} "
            f"utility={ordered.item(0, 'total_log_weight'):.4f} "
            f"zones={sequence}"
        )


def mean_draw_utility(plans: pl.DataFrame) -> float:
    if plans.is_empty():
        return float("nan")
    return float(plans.filter(pl.col("layer") == 0)["total_log_weight"].mean())


def failure_summary(report: dict[str, object]) -> str:
    contexts = report["context_reports"]
    assert isinstance(contexts, list)
    failures = [context for context in contexts if context["failure_reason"]]
    if not failures:
        return "none"
    counts: dict[str, int] = {}
    coverage_gaps = 0
    for failure in failures:
        reason = str(failure["failure_reason"])
        counts[reason] = counts.get(reason, 0) + 1
        if (
            reason == "no_locally_feasible_candidate"
            and int(failure["domain_locally_feasible_candidates"] or 0) > 0
        ):
            coverage_gaps += 1
    result = ", ".join(f"{reason}={count}" for reason, count in sorted(counts.items()))
    return f"{result}; candidate-coverage-gaps={coverage_gaps}"


def retry_summary(report: dict[str, object]) -> str:
    return (
        f"attempts={report['retry_attempts']} "
        f"recovered-contexts={report['recovered_contexts']}"
    )


def radiation_summary(path: Path, exact_context_ids: pl.DataFrame) -> str:
    plans = pl.read_parquet(path).join(exact_context_ids, on="context_id", how="semi")
    required = {"context_id", "draw_id", "layer"}
    missing = required - set(plans.columns)
    if missing:
        raise ValueError(f"radiation parquet is missing columns: {sorted(missing)}")
    if "final_utility" in plans.columns:
        plans = plans.filter(pl.col("layer") == 0).rename(
            {"final_utility": "total_log_weight"}
        )
    elif "total_log_weight" not in plans.columns:
        raise ValueError("radiation parquet needs final_utility or total_log_weight")
    return utility_summary(plans)


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

    exact_steps, exact_initial = shared_home_contexts(
        steps, initial_locations, args.exact_contexts, args.seed
    )
    exact_od, exact_destinations = restrict_to_oracle_zones(
        od_costs, destination_inputs, exact_initial, args.exact_zones
    )
    exact_sampler = DestinationSampler(
        od_costs=exact_od, destination_inputs=exact_destinations
    )
    exact, exact_seconds = run_exact(exact_sampler, exact_steps, exact_initial, args)
    particle_oracle, oracle_report, particle_oracle_seconds = run_particles(
        exact_sampler, exact_steps, exact_initial, args
    )
    bidirectional_oracle, bidirectional_report, bidirectional_oracle_seconds = (
        run_bidirectional(exact_sampler, exact_steps, exact_initial, args)
    )

    if args.particle_contexts:
        full_steps, full_initial = select_contexts(
            steps,
            initial_locations,
            min(args.particle_contexts, initial_locations.height),
            seed=args.seed,
        )
        full_sampler = DestinationSampler(
            od_costs=od_costs, destination_inputs=destination_inputs
        )
        particle_full, full_report, full_seconds = run_particles(
            full_sampler, full_steps, full_initial, args
        )
    else:
        full_initial = pl.DataFrame({"context_id": [], "initial_zone": []})
        particle_full = pl.DataFrame()
        full_report = {"mean_effective_sample_size": 0.0, "candidate_evaluations": 0}
        full_seconds = 0.0

    print("\ncomparison")
    print("method             workload                 result")
    print(f"exact              {args.exact_contexts} contexts / {args.exact_zones} zones  {utility_summary(exact)} wall={exact_seconds:.3f}s")
    print(f"particle           {args.exact_contexts} contexts / {args.exact_zones} zones  {utility_summary(particle_oracle)} wall={particle_oracle_seconds:.3f}s ESS={oracle_report['mean_effective_sample_size']:.2f}")
    print(f"bidirectional      {args.exact_contexts} contexts / {args.exact_zones} zones  {utility_summary(bidirectional_oracle)} wall={bidirectional_oracle_seconds:.3f}s completed-plans={bidirectional_report['completed_plans']}")
    print(f"particle failures  oracle subset             {failure_summary(oracle_report)}")
    print(f"particle retries   oracle subset             {retry_summary(oracle_report)}")
    if args.particle_contexts:
        print(f"particle           {full_initial.height} contexts / raw zones  {utility_summary(particle_full)} wall={full_seconds:.3f}s ESS={full_report['mean_effective_sample_size']:.2f} candidate-evaluations={full_report['candidate_evaluations']}")
        print(f"particle failures  raw-zone sample            {failure_summary(full_report)}")
        print(f"particle retries   raw-zone sample            {retry_summary(full_report)}")
    print(
        "exact-particle first-destination total variation: "
        f"{weighted_first_destination_tv(exact, particle_oracle):.4f}"
    )
    print(
        "exact-bidirectional unweighted listed-plan first-destination TV: "
        f"{empirical_first_destination_tv(exact, bidirectional_oracle):.4f}"
    )
    print(
        "exact-bidirectional score-weighted first-destination TV: "
        f"{score_weighted_first_destination_tv(exact, bidirectional_oracle):.4f}"
    )
    print(
        "exact-bidirectional mean draw-utility shift: "
        f"{mean_draw_utility(bidirectional_oracle) - mean_draw_utility(exact):.4f}"
    )
    print_worst_bidirectional_example(
        exact,
        bidirectional_oracle,
        exact_steps,
        exact_initial,
        exact_od,
        exact_destinations,
    )
    if args.radiation_plans:
        print(f"radiation          {radiation_summary(args.radiation_plans, exact_initial.select('context_id'))}")
    else:
        print("radiation          not supplied; pass --radiation-plans <parquet> for the third row")


if __name__ == "__main__":
    main()
