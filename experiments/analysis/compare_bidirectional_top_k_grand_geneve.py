"""Read-only top-K error check against the exact raw-zone oracle.

This keeps the complete raw-zone domain. The default comparison targets short,
oracle-proven contexts; ``--contexts-per-stratum`` instead audits all supported
depths and retains oracle failures as coverage data.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from statistics import median
from pathlib import Path
from typing import Any, Callable

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
from experiments.harness import (
    ExperimentKind,
    ExperimentManifest,
    RunRecorder,
    quality_verdict,
)
from experiments.oracle_cache import (
    OracleCache,
    oracle_attempt_fingerprint,
    oracle_input_fingerprint,
)
from experiments.top_k_config import (
    add_top_k_tuning_arguments,
    apply_top_k_overrides,
    top_k_tuning_options,
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
    parser.add_argument(
        "--contexts-per-stratum",
        type=int,
        help=(
            "audit mode: sample this many contexts from every depth/anchor stratum; "
            "all layers >= 3 are included and oracle failures are retained"
        ),
    )
    parser.add_argument("--max-layers", type=int, default=4)
    parser.add_argument(
        "--top-k",
        type=int,
        action="append",
        dest="top_ks",
        help="bounded result count; repeat to compare several K values (default: 10)",
    )
    parser.add_argument(
        "--oracle-depth",
        type=int,
        default=100,
        help="fixed exact-plan support used to normalize all requested K values",
    )
    parser.add_argument("--max-states", type=int, default=2_000_000)
    add_top_k_tuning_arguments(parser)
    parser.add_argument(
        "--symmetric-config",
        action="append",
        metavar="LABEL:MESSAGES:STATES:PROPOSALS",
        help="repeat to compare symmetric configurations in one prepared process",
    )
    parser.add_argument(
        "--candidate-option",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="fast exploratory A/B override; repeat as needed",
    )
    parser.add_argument("--archetype-strata-limit", type=int, default=12)
    parser.add_argument(
        "--context-id",
        type=int,
        action="append",
        dest="context_ids",
        help="repeat to run a fixed diagnostic context set",
    )
    parser.add_argument(
        "--trace-context",
        type=int,
        help="show where exact target plans enter or leave the active bounded search",
    )
    parser.add_argument(
        "--design-manifest",
        type=Path,
        help=(
            "write a reproducible, depth-diverse set of oracle-proven success, "
            "partial, and zero-mass cases from audit mode"
        ),
    )
    parser.add_argument(
        "--audit-min-layers",
        type=int,
        default=2,
        help="audit mode: exclude shallow contexts so oracle budget targets deep strata",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="show every returned plan and every missed exact top-K plan",
    )
    parser.add_argument(
        "--exploration-seed",
        type=int,
        action="append",
        dest="exploration_seeds",
        help="candidate-context hash seed; repeat for independent validation cohorts",
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--no-oracle-cache",
        action="store_true",
        help="recompute exact cases instead of reusing fingerprinted oracle results",
    )
    parser.add_argument(
        "--no-bounded-incumbent",
        action="store_true",
        help="disable active bounded plans as exact-oracle lower bounds",
    )
    parser.add_argument(
        "--cached-oracles-only",
        action="store_true",
        help="evaluate only already-certified cached exact results; never start an oracle search",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="print one summary row per seed/configuration instead of distributions",
    )
    parser.add_argument(
        "--experiment-manifest",
        type=Path,
        help="immutable quality-only A/B config, cohort lock, and gates",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("experiments/runs"),
        help="generated artifact root for manifest-driven runs",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="write machine-readable per-configuration quality summaries",
    )
    return parser.parse_args()


def symmetric_configurations(args: argparse.Namespace) -> list[tuple[str, int, int, int]]:
    if not args.symmetric_config:
        return [
            (
                "default",
                args.symmetric_message_limit,
                args.symmetric_state_limit,
                args.symmetric_forward_proposal_limit,
            )
        ]
    configurations = []
    for value in args.symmetric_config:
        parts = value.split(":")
        if len(parts) != 4 or not parts[0]:
            raise ValueError(
                "--symmetric-config must be LABEL:MESSAGES:STATES:PROPOSALS"
            )
        try:
            limits = tuple(int(part) for part in parts[1:])
        except ValueError as error:
            raise ValueError("symmetric configuration limits must be integers") from error
        if min(limits) < 0:
            raise ValueError("symmetric configuration limits must be non-negative")
        configurations.append((parts[0], *limits))
    return configurations


def eligible_context_ids(
    profiles: pl.DataFrame, args: argparse.Namespace, exploration_seed: int
) -> list[int]:
    """Select either a bounded comparison cohort or one audit cohort per stratum."""
    if args.contexts_per_stratum is not None:
        if args.contexts_per_stratum <= 0:
            raise ValueError("--contexts-per-stratum must be positive")
    elif args.contexts <= 0 or args.candidate_contexts <= 0:
        raise ValueError("--contexts and --candidate-contexts must be positive")
    if args.max_layers < 2:
        raise ValueError("--max-layers must be at least two")
    if args.audit_min_layers < 2:
        raise ValueError("--audit-min-layers must be at least two")
    if args.cached_oracles_only and args.no_oracle_cache:
        raise ValueError("--cached-oracles-only requires the oracle cache")
    if args.context_ids:
        missing = [
            context_id
            for context_id in args.context_ids
            if profiles.filter(pl.col("context_id") == context_id).is_empty()
        ]
        if missing:
            raise ValueError(f"contexts do not exist: {missing}")
        return list(dict.fromkeys(args.context_ids))
    if args.trace_context is not None:
        if profiles.filter(pl.col("context_id") == args.trace_context).is_empty():
            raise ValueError(f"context {args.trace_context} does not exist")
        return [args.trace_context]
    minimum_layers = args.audit_min_layers if args.contexts_per_stratum is not None else 3
    eligible = profiles.filter(pl.col("layers") >= minimum_layers).with_columns(
        sample_order=pl.col("context_id").hash(seed=exploration_seed)
    )
    selected = (
        eligible.sort(["audit_stratum", "sample_order"])
        .group_by("audit_stratum", maintain_order=True)
        .head(args.contexts_per_stratum)
        if args.contexts_per_stratum is not None
        else eligible.filter(pl.col("layers") <= args.max_layers)
        .sort("sample_order")
        .head(args.candidate_contexts)
    )
    if selected.is_empty():
        raise ValueError("no eligible short contexts")
    return [int(context_id) for context_id in selected["context_id"].to_list()]


def context_profiles(steps: pl.DataFrame) -> pl.DataFrame:
    """Compact plan-type fields used for sampling and stratified reporting."""
    anchored = pl.col("fixed_destination").is_not_null() | pl.col("anchor_id").is_not_null()
    profiles = steps.group_by("context_id").agg(
        layers=pl.len(),
        anchor_count=anchored.sum(),
        anchor_activity_types=pl.col("activity_id")
        .filter(anchored)
        .n_unique(),
    )
    return (
        profiles.with_columns(
            depth_band=pl.when(pl.col("layers") >= 10)
            .then(pl.lit("10+"))
            .when(pl.col("layers") == 9)
            .then(pl.lit("9"))
            .when(pl.col("layers") == 8)
            .then(pl.lit("8"))
            .when(pl.col("layers") == 7)
            .then(pl.lit("7"))
            .otherwise(pl.col("layers").cast(pl.String)),
            anchor_count_band=pl.when(pl.col("anchor_count") >= 3)
            .then(pl.lit("3+"))
            .otherwise(pl.col("anchor_count").cast(pl.String)),
            anchor_type_band=pl.when(pl.col("anchor_activity_types") >= 2)
            .then(pl.lit("2+"))
            .otherwise(pl.col("anchor_activity_types").cast(pl.String)),
        )
        .with_columns(
            audit_stratum=pl.concat_str(
                ["depth_band", "anchor_count_band", "anchor_type_band"],
                separator="|",
            )
        )
    )


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


def print_distribution_table(label: str, values_by_top_k: dict[int, list[float]]) -> None:
    """Print full distribution detail with one concise row per requested K."""
    print(f"\n{label}")
    print("  K | mean  | min   | p25   | p50   | p75   | p90   | p95   | p99   | max   | zero")
    for top_k, values in values_by_top_k.items():
        if not values:
            continue
        zero = sum(math.isclose(value, 0.0, abs_tol=1e-9) for value in values)
        print(
            f"{top_k:3d} | {sum(values) / len(values):.3f} | {min(values):.3f} "
            f"| {quantile(values, 0.25):.3f} | {quantile(values, 0.50):.3f} "
            f"| {quantile(values, 0.75):.3f} | {quantile(values, 0.90):.3f} "
            f"| {quantile(values, 0.95):.3f} | {quantile(values, 0.99):.3f} "
            f"| {max(values):.3f} | {zero}/{len(values)}"
        )


def append_stratum_metrics(
    by_top_k: dict[int, dict[str, list[float]]],
    stratum: str,
    top_k: int,
    metrics: dict[str, float],
) -> None:
    by_top_k[top_k].setdefault(stratum, []).append(metrics["retained_top_k_mass"])


def print_stratum_table(
    label: str, by_top_k: dict[int, dict[str, list[float]]], top_ks: list[int], limit: int
) -> None:
    """Print mean Mass@K for common proven plan archetypes."""
    first_top_k = top_ks[0]
    counts = {stratum: len(values) for stratum, values in by_top_k[first_top_k].items()}
    ranked = sorted(counts, key=lambda stratum: (-counts[stratum], stratum))[:limit]
    print(f"\n{label} (mean Mass@K)")
    print("stratum | n | " + " | ".join(f"K={top_k}" for top_k in top_ks))
    for stratum in ranked:
        values = [
            sum(by_top_k[top_k][stratum]) / counts[stratum] for top_k in top_ks
        ]
        print(f"{stratum} | {counts[stratum]} | " + " | ".join(f"{value:.3f}" for value in values))
    if len(counts) > len(ranked):
        print(f"... {len(counts) - len(ranked)} rarer strata omitted")


def print_audit_coverage(
    population_by_stratum: dict[str, int], outcomes: dict[str, dict[str, int]]
) -> None:
    """Show whether the sampled strata support an unbiased global estimate."""
    print("\nGlobal audit coverage")
    print("stratum | population | sampled | oracle solved | bounded complete | state cap | infeasible | oracle error | bounded failed")
    for stratum in sorted(population_by_stratum):
        outcome = outcomes[stratum]
        print(
            f"{stratum} | {population_by_stratum[stratum]} | {outcome['sampled']} "
            f"| {outcome['oracle_completed']} | {outcome['proven']} | {outcome['state_limited']} "
            f"| {outcome['infeasible']} | {outcome['oracle_error']} | {outcome['bounded_failed']}"
        )


def print_global_audit_estimate(
    population_by_stratum: dict[str, int],
    outcomes: dict[str, dict[str, int]],
    metrics_by_top_k: dict[int, dict[str, list[float]]],
) -> None:
    """Report model-based estimates and bounds for missing exact certificates."""
    covered = [stratum for stratum in outcomes if stratum in metrics_by_top_k]
    covered_population = sum(population_by_stratum[stratum] for stratum in covered)
    population = sum(population_by_stratum.values())
    if not covered_population:
        return
    print(
        "\nPilot quality estimate (solved-context mean imputed within each stratum; "
        f"strata with evidence={covered_population / population:.1%} of population)"
    )
    print("K | Mass@K | Mass@oracle-support")
    for top_k, metrics_by_stratum in metrics_by_top_k.items():
        weighted = lambda metric: sum(
            population_by_stratum[stratum] * sum(metrics_by_stratum[stratum][metric])
            / len(metrics_by_stratum[stratum][metric])
            for stratum in covered
        ) / population
        print(
            f"{top_k} | {weighted('retained_top_k_mass'):.3f} "
            f"| {weighted('retained_oracle_mass'):.3f}"
        )
    print(
        "\nOracle-missingness bounds "
        "(sampled contexts without certificates may have mass anywhere in [0, 1])"
    )
    print("K | metric | lower | imputed | upper | observed/sampled")
    for top_k, metrics_by_stratum in metrics_by_top_k.items():
        for metric, label in (
            ("retained_top_k_mass", "Mass@K"),
            ("retained_oracle_mass", "Mass@oracle"),
        ):
            summary = stratified_missingness_summary(
                population_by_stratum,
                outcomes,
                metrics_by_stratum,
                metric,
            )
            print(
                f"{top_k} | {label} | {summary['lower']:.3f} | "
                f"{summary['imputed']:.3f} | {summary['upper']:.3f} | "
                f"{summary['observed']}/{summary['sampled']}"
            )
    first_top_k = min(metrics_by_top_k)
    unknowns = stratified_missingness_summary(
        population_by_stratum,
        outcomes,
        metrics_by_top_k[first_top_k],
        "retained_top_k_mass",
    )["unknown_impacts"]
    if unknowns:
        print("\nHighest-impact oracle unknowns")
        print("stratum | unresolved/sampled | maximum global interval reduction")
        for row in unknowns[:10]:
            print(
                f"{row['stratum']} | {row['unresolved']}/{row['sampled']} | "
                f"{row['impact']:.3f}"
            )


def stratified_missingness_summary(
    population_by_stratum: dict[str, int],
    outcomes: dict[str, dict[str, int]],
    metrics_by_stratum: dict[str, dict[str, list[float]]],
    metric: str,
) -> dict[str, Any]:
    """Bound a stratified estimate without treating oracle failures as random hits."""
    population = sum(population_by_stratum.values())
    if population <= 0:
        raise ValueError("audit population must be positive")
    lower = 0.0
    upper = 0.0
    imputed = 0.0
    sampled_total = 0
    observed_total = 0
    unknown_impacts = []
    for stratum, stratum_population in population_by_stratum.items():
        sampled = int(outcomes[stratum]["sampled"])
        values = metrics_by_stratum.get(stratum, {}).get(metric, [])
        observed = len(values)
        if observed > sampled:
            raise ValueError(
                f"stratum {stratum} has {observed} metrics for {sampled} samples"
            )
        population_weight = stratum_population / population
        sampled_total += sampled
        observed_total += observed
        if sampled == 0:
            upper += population_weight
            unknown_impacts.append(
                {
                    "stratum": stratum,
                    "sampled": 0,
                    "unresolved": stratum_population,
                    "impact": population_weight,
                }
            )
            continue
        observed_sum = sum(values)
        unresolved = sampled - observed
        lower += population_weight * observed_sum / sampled
        upper += population_weight * (observed_sum + unresolved) / sampled
        imputed += population_weight * (
            observed_sum / observed if observed else 0.0
        )
        if unresolved:
            unknown_impacts.append(
                {
                    "stratum": stratum,
                    "sampled": sampled,
                    "unresolved": unresolved,
                    "impact": population_weight * unresolved / sampled,
                }
            )
    unknown_impacts.sort(key=lambda row: (-float(row["impact"]), str(row["stratum"])))
    return {
        "lower": lower,
        "imputed": imputed,
        "upper": upper,
        "sampled": sampled_total,
        "observed": observed_total,
        "unknown_impacts": unknown_impacts,
    }


def print_audit_quality_by_stratum(
    population_by_stratum: dict[str, int],
    outcomes: dict[str, dict[str, int]],
    metrics_by_stratum: dict[str, dict[str, list[float]]],
) -> None:
    """Expose every sampled stratum, including bounded failures as zero mass."""
    print("\nAll-stratum certified Mass@K")
    print("stratum | population | exact n | bounded n | recall | mass")
    for stratum in sorted(population_by_stratum):
        outcome = outcomes[stratum]
        metrics = metrics_by_stratum.get(stratum)
        if not metrics:
            print(
                f"{stratum} | {population_by_stratum[stratum]} | 0 | 0 | - | -"
            )
            continue
        count = len(metrics["retained_top_k_mass"])
        print(
            f"{stratum} | {population_by_stratum[stratum]} | "
            f"{outcome['oracle_completed']} | {count} | "
            f"{sum(metrics['recall']) / count:.3f} | "
            f"{sum(metrics['retained_top_k_mass']) / count:.3f}"
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


def bounded_failure_metrics(
    oracle: list[tuple[tuple[int, ...], float]], top_k: int
) -> dict[str, float]:
    """Use zero retained mass when the bounded search returns no complete plan."""
    oracle_top_k = oracle[:top_k]
    return {
        "recall": 0.0,
        "retained_top_k_mass": 0.0,
        "retained_oracle_mass": 0.0,
        "top_k_oracle_mass": retained_probability_mass(
            oracle, {zones for zones, _ in oracle_top_k}
        ),
        "top_k_mass_efficiency": 0.0,
    }


def oracle_failure_kind(error: ValueError) -> str:
    """Keep oracle limits, infeasibility, and internal errors distinct in audits."""
    message = str(error)
    if "exceeded max_states" in message:
        return "state_limited"
    if "no feasible destination sequence" in message:
        return "infeasible"
    return "oracle_error"


def trace_active_stage_coverage(report: dict[str, object]) -> None:
    """Show bounded-search proposal and retention for oracle targets."""
    traces = report.get("active_trace_targets", [])
    print("active factor-map trace (proposed / retained / pruned):")
    for rank, trace in enumerate(traces, start=1):
        zones = trace["zones"]
        proposed = trace["proposed"]
        retained = trace["retained"]
        pruned = trace["pruned"]
        prefix_proposed = trace["prefix_proposed"]
        prefix_retained = trace["prefix_retained"]
        prefix_pruned = trace["prefix_pruned"]
        guidance_retained = trace["guidance_retained"]
        guidance_proposed = trace["guidance_proposed"]
        exact_guidance_rank = trace["exact_guidance_rank"]
        exact_guidance_log_gap = trace["exact_guidance_log_gap"]
        print(f"  exact rank={rank} zones={tuple(zones)}")
        for layer, (zone, was_proposed, was_retained, was_pruned) in enumerate(
            zip(zones, proposed, retained, pruned, strict=True)
        ):
            state = "retained" if was_retained else "pruned" if was_pruned else "not-proposed"
            print(
                f"    layer={layer} zone={zone}: {state} "
                f"(proposed={was_proposed}, retained={was_retained})"
            )
            if prefix_proposed[layer] is None:
                print("      coherent prefix: not evaluated in the forward pass")
                continue
            coherent_state = (
                "retained"
                if prefix_retained[layer]
                else "pruned"
                if prefix_pruned[layer]
                else "not-proposed"
            )
            print(
                f"      coherent prefix through layer {layer}: {coherent_state} "
                f"(proposed={prefix_proposed[layer]}, retained={prefix_retained[layer]})"
            )
            if guidance_proposed[layer] or guidance_retained[layer]:
                print(
                    f"      reverse guidance: target proposed={guidance_proposed[layer]}, "
                    f"retained={guidance_retained[layer]}, "
                    f"exact-rank={exact_guidance_rank[layer]}, "
                    f"log-gap={exact_guidance_log_gap[layer]}"
                )


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


def show_context(
    context_id: int,
    oracle: list[tuple[tuple[int, ...], float]],
    bounded: list[tuple[tuple[int, ...], float]],
    top_k: int,
    states_pushed: int,
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
    maximum_utility = max(score for _, score in oracle_top_k)
    top_k_normalizer = sum(
        math.exp(score - maximum_utility) for _, score in oracle_top_k
    )
    if verbose:
        print(
            f"context={context_id} oracle-top-k={len(oracle_top_k)} "
            f"bounded={len(bounded)} recall@{top_k}={hits}/{len(oracle_top_k)} "
            f"retained-top-{top_k}-mass={retained_top_k_mass:.4f} "
            f"retained-oracle-{len(oracle)}-mass={retained_oracle_mass:.4f} "
            f"exact-states={states_pushed}"
        )
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
            status = "returned" if zones in bounded_zones else "missed"
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
    }


def compare_seed(
    args: argparse.Namespace,
    configuration_label: str,
    top_ks: list[int],
    exploration_seed: int,
    search: DestinationPlanSearch,
    steps: pl.DataFrame,
    initial_locations: pl.DataFrame,
    od_costs: pl.DataFrame,
    destination_inputs: pl.DataFrame,
    destination_domain_sizes: dict[int, int],
    profiles: pl.DataFrame,
    profile_by_context: dict[int, tuple[int, int, int, str]],
    population_by_stratum: dict[str, int],
    oracle_cache: OracleCache | None,
    oracle_memory: dict[int, tuple[pl.DataFrame, dict]],
    progress_callback: Callable[..., None] | None = None,
) -> dict[str, Any]:
    proven = 0
    skipped = 0
    audit_mode = args.contexts_per_stratum is not None
    audit_outcomes = {
        stratum: {
            "sampled": 0,
            "oracle_completed": 0,
            "proven": 0,
            "state_limited": 0,
            "infeasible": 0,
            "oracle_error": 0,
            "bounded_failed": 0,
        }
        for stratum in population_by_stratum
    }
    metrics_by_top_k = {
        top_k: {
            "recall": [],
            "retained_top_k_mass": [],
            "retained_oracle_mass": [],
            "top_k_oracle_mass": [],
            "top_k_mass_efficiency": [],
        }
        for top_k in top_ks
    }
    exact_search_seconds = 0.0
    oracle_cache_hits = 0
    oracle_cache_misses = 0
    bounded_search_seconds = {top_k: 0.0 for top_k in top_ks}
    layer_metrics = {top_k: {} for top_k in top_ks}
    archetype_metrics = {top_k: {} for top_k in top_ks}
    audit_metrics = {top_k: {} for top_k in top_ks}
    oracle_diagnostics: list[dict[str, float | int | str]] = []
    audit_cases: list[dict[str, float | int | str]] = []
    started = time.perf_counter()
    selected_context_ids = eligible_context_ids(profiles, args, exploration_seed)
    for position, context_id in enumerate(selected_context_ids, start=1):
        layers, anchor_count, anchor_activity_types, audit_stratum = profile_by_context[context_id]
        if audit_mode:
            audit_outcomes[audit_stratum]["sampled"] += 1
        context_steps = steps.filter(pl.col("context_id") == context_id)
        context_initial = initial_locations.filter(pl.col("context_id") == context_id)
        oracle_shape = oracle_context_shape(
            context_steps,
            destination_domain_sizes,
            int(context_initial.item(0, "initial_zone")),
        )
        try:
            oracle_started = time.perf_counter()
            compute_oracle = lambda: search.exact_top_k(
                steps=context_steps,
                initial_locations=context_initial,
                logit_scale=LOGIT_SCALE,
                update_plan_timings=True,
                use_shadow_prices=True,
                top_k=args.oracle_depth,
                max_states=args.max_states,
                n_threads=1,
                skip_infeasible=False,
                use_bounded_incumbent=not args.no_bounded_incumbent,
            )
            if context_id in oracle_memory:
                oracle_table, oracle_report = oracle_memory[context_id]
                oracle_cache_hits += 1
            elif args.cached_oracles_only:
                cached_result = oracle_cache.load_cached(context_id)
                if cached_result is None:
                    skipped += 1
                    if args.verbose:
                        print(f"context={context_id} oracle not cached")
                    if progress_callback:
                        progress_callback(
                            "context_complete",
                            position=position,
                            total=len(selected_context_ids),
                            context_id=context_id,
                            outcome="not_cached",
                        )
                    continue
                oracle_table, oracle_report = cached_result
                oracle_cache_hits += 1
            elif oracle_cache is None:
                oracle_table, oracle_report = compute_oracle()
                oracle_cache_misses += 1
                exact_search_seconds += time.perf_counter() - oracle_started
            else:
                oracle_table, oracle_report, cached = oracle_cache.load_or_compute(
                    context_id, compute_oracle
                )
                oracle_cache_hits += int(cached)
                oracle_cache_misses += int(not cached)
                if not cached:
                    exact_search_seconds += time.perf_counter() - oracle_started
            oracle_memory[context_id] = (oracle_table, oracle_report)
        except ValueError as error:
            skipped += 1
            oracle_diagnostics.append(
                {
                    **oracle_shape,
                    "outcome": oracle_failure_kind(error),
                    "states_pushed": 0,
                    "maximum_heap_size": 0,
                    "children_pruned_by_incumbent": 0,
                }
            )
            if audit_mode:
                audit_outcomes[audit_stratum][oracle_failure_kind(error)] += 1
            if args.verbose:
                print(f"context={context_id} oracle skipped: {error}")
            if progress_callback:
                progress_callback(
                    "context_complete",
                    position=position,
                    total=len(selected_context_ids),
                    context_id=context_id,
                    outcome=oracle_failure_kind(error),
                )
            continue
        oracle_diagnostics.append(
            {
                **oracle_shape,
                "outcome": "solved",
                "states_pushed": int(oracle_report["states_pushed"]),
                "maximum_heap_size": int(oracle_report["maximum_heap_size"]),
                "children_pruned_by_incumbent": int(
                    oracle_report["children_pruned_by_incumbent"]
                ),
            }
        )
        oracle = ranked_plans(oracle_table)
        if audit_mode:
            audit_outcomes[audit_stratum]["oracle_completed"] += 1
        if args.trace_context == context_id:
            trace_first_layer_and_plan_components(
                context_steps,
                int(context_initial.item(0, "initial_zone")),
                oracle_table,
                oracle,
                od_costs,
                destination_inputs,
                max(top_ks),
            )
        context_metrics = {}
        for top_k in top_ks:
            try:
                bounded_started = time.perf_counter()
                active_trace = (
                    {
                        "active_trace_context_id": context_id,
                        "active_trace_target_plans": [list(zones) for zones, _ in oracle[:top_k]],
                    }
                    if args.trace_context == context_id
                    else {}
                )
                bounded_table, bounded_report = search.top_k(
                    steps=context_steps,
                    initial_locations=context_initial,
                    logit_scale=LOGIT_SCALE,
                    update_plan_timings=True,
                    use_shadow_prices=True,
                    exploration_seed=exploration_seed,
                    **top_k_tuning_options(args),
                    top_k=top_k,
                    n_threads=1,
                    skip_contexts_without_plan=False,
                    **active_trace,
                )
                bounded_search_seconds[top_k] += time.perf_counter() - bounded_started
                if args.trace_context == context_id:
                    trace_active_stage_coverage(bounded_report)
            except ValueError as error:
                skipped += 1
                if audit_mode:
                    audit_outcomes[audit_stratum]["bounded_failed"] += 1
                    metrics = bounded_failure_metrics(oracle, top_k)
                    audit_metrics[top_k].setdefault(audit_stratum, {})
                    for name, value in metrics.items():
                        audit_metrics[top_k][audit_stratum].setdefault(name, []).append(value)
                if args.verbose:
                    print(f"context={context_id} bounded search failed: {error}")
                break
            context_metrics[top_k] = show_context(
                context_id,
                oracle,
                ranked_plans(bounded_table),
                top_k,
                oracle_report["states_pushed"],
                args.verbose or args.trace_context is not None,
            )
        if len(context_metrics) != len(top_ks):
            if progress_callback:
                progress_callback(
                    "context_complete",
                    position=position,
                    total=len(selected_context_ids),
                    context_id=context_id,
                    outcome="bounded_failed",
                )
            continue
        for top_k, metrics in context_metrics.items():
            for name, values in metrics_by_top_k[top_k].items():
                values.append(metrics[name])
            append_stratum_metrics(layer_metrics, f"layers={layers}", top_k, metrics)
            append_stratum_metrics(
                archetype_metrics,
                f"depth={layers}, anchors={anchor_count}, anchor-types={anchor_activity_types}",
                top_k,
                metrics,
            )
            if audit_mode:
                audit_metrics[top_k].setdefault(audit_stratum, {})
                for name, value in metrics.items():
                    audit_metrics[top_k][audit_stratum].setdefault(name, []).append(value)
                if top_k == min(top_ks):
                    audit_cases.append(
                        {
                            "context": context_id,
                            "layers": layers,
                            "stratum": audit_stratum,
                            "recall": metrics["recall"],
                            "mass": metrics["retained_top_k_mass"],
                        }
                    )
            if args.compact and args.context_ids:
                print(
                    f"case | {context_id} | {configuration_label} | {exploration_seed} | "
                    f"{top_k} | {metrics['recall']:.3f} | "
                    f"{metrics['retained_top_k_mass']:.3f}"
                )
        proven += 1
        if audit_mode:
            audit_outcomes[audit_stratum]["proven"] += 1
        if progress_callback:
            progress_callback(
                "context_complete",
                position=position,
                total=len(selected_context_ids),
                context_id=context_id,
                outcome="compared",
            )
        if not args.compact and position % 10 == 0:
            print(
                f"progress | config={configuration_label} seed={exploration_seed} "
                f"contexts={position}/{len(selected_context_ids)} proven={proven} "
                f"skipped={skipped}",
                flush=True,
            )
        if not audit_mode and proven == args.contexts:
            break
    if audit_mode:
        print_audit_coverage(population_by_stratum, audit_outcomes)
        print_oracle_search_diagnostics(oracle_diagnostics)
        print_lowest_deep_cases(audit_cases)
        if args.design_manifest:
            write_design_manifest(args.design_manifest, audit_cases)
    if not proven:
        if audit_mode:
            print("No sampled context completed the exact top-K oracle.")
            return {
                "configuration": configuration_label,
                "seed": exploration_seed,
                "status": "no_proven_contexts",
                "proven": 0,
            }
        raise RuntimeError("no context completed the exact top-K oracle")
    wall_seconds = time.perf_counter() - started
    result: dict[str, Any] = {
        "configuration": configuration_label,
        "seed": exploration_seed,
        "status": "completed",
        "proven": proven,
        "skipped": skipped,
        "wall_seconds": wall_seconds,
        "oracle_cache": {
            "hits": oracle_cache_hits,
            "misses": oracle_cache_misses,
        },
        "metrics": {
            str(top_k): {
                "recall": sum(metrics_by_top_k[top_k]["recall"]) / proven,
                "mass": sum(metrics_by_top_k[top_k]["retained_top_k_mass"])
                / proven,
                "minimum_mass": min(
                    metrics_by_top_k[top_k]["retained_top_k_mass"]
                ),
                "zero_mass": sum(
                    mass <= 0.0
                    for mass in metrics_by_top_k[top_k][
                        "retained_top_k_mass"
                    ]
                ),
                "bounded_ms_per_context": bounded_search_seconds[top_k]
                / proven
                * 1e3,
            }
            for top_k in top_ks
        },
    }
    if audit_mode:
        result["audit"] = {
            "outcomes": audit_outcomes,
            "cases": audit_cases,
            "missingness": {
                str(top_k): {
                    metric: stratified_missingness_summary(
                        population_by_stratum,
                        audit_outcomes,
                        audit_metrics[top_k],
                        metric,
                    )
                    for metric in (
                        "recall",
                        "retained_top_k_mass",
                        "retained_oracle_mass",
                    )
                }
                for top_k in top_ks
            },
        }
        primary = result["audit"]["missingness"][str(min(top_ks))]
        mass_summary = primary["retained_top_k_mass"]
        recall_summary = primary["recall"]
        result["quality_summary"] = {
            "stratified_mass": mass_summary["imputed"],
            "stratified_recall": recall_summary["imputed"],
            "zero_mass": sum(
                value <= 0.0
                for metrics in audit_metrics[min(top_ks)].values()
                for value in metrics.get("retained_top_k_mass", [])
            ),
            "oracle_interval_width": mass_summary["upper"]
            - mass_summary["lower"],
            "oracle_observed": mass_summary["observed"],
            "oracle_sampled": mass_summary["sampled"],
        }
    if args.compact:
        if args.context_ids:
            print(
                "config | seed | K | n | recall | mass | min | zero | "
                "bounded-ms | wall-s | oracle-hit/miss"
            )
        for top_k in top_ks:
            metrics = metrics_by_top_k[top_k]
            masses = metrics["retained_top_k_mass"]
            print(
                f"{configuration_label} | {exploration_seed} | {top_k} | {proven} | "
                f"{sum(metrics['recall']) / proven:.3f} | "
                f"{sum(masses) / proven:.3f} | {min(masses):.3f} | "
                f"{sum(mass <= 0.0 for mass in masses)} | "
                f"{bounded_search_seconds[top_k] / proven * 1e3:.2f} | "
                f"{wall_seconds:.2f} | {oracle_cache_hits}/{oracle_cache_misses}"
            )
        return result
    print(
        f"\nseed={exploration_seed} proven-contexts={proven} skipped={skipped} "
        f"wall={wall_seconds:.3f}s "
        f"exact={exact_search_seconds / max(oracle_cache_misses, 1) * 1e3:.2f}ms/context "
        f"oracle-cache={oracle_cache_hits} hit/{oracle_cache_misses} miss"
    )
    print("Search performance")
    print("  K | recall | bounded ms | speedup")
    for top_k in top_ks:
        metrics = metrics_by_top_k[top_k]
        mean = lambda values: sum(values) / proven
        bounded_ms = bounded_search_seconds[top_k] / proven * 1e3
        speedup = (
            f"{exact_search_seconds / bounded_search_seconds[top_k]:.1f}x"
            if exact_search_seconds
            else "cached"
        )

        print(
            f"{top_k:3d} | {mean(metrics['recall']):.3f}  | {bounded_ms:.2f}       "
            f"| {speedup:<7}"
        )
    print_distribution_table(
        "Retained exact top-K mass (conditional)",
        {top_k: metrics_by_top_k[top_k]["retained_top_k_mass"] for top_k in top_ks},
    )
    print_distribution_table(
        f"Retained mass over fixed exact top-{args.oracle_depth} support",
        {top_k: metrics_by_top_k[top_k]["retained_oracle_mass"] for top_k in top_ks},
    )
    print_distribution_table(
        f"Exact top-K share of top-{args.oracle_depth} mass",
        {top_k: metrics_by_top_k[top_k]["top_k_oracle_mass"] for top_k in top_ks},
    )
    print_distribution_table(
        "Top-K mass efficiency",
        {top_k: metrics_by_top_k[top_k]["top_k_mass_efficiency"] for top_k in top_ks},
    )
    print_stratum_table("By layer count", layer_metrics, top_ks, limit=len(layer_metrics[top_ks[0]]))
    print_stratum_table("By plan archetype", archetype_metrics, top_ks, args.archetype_strata_limit)
    if audit_mode:
        print_audit_quality_by_stratum(
            population_by_stratum, audit_outcomes, audit_metrics[min(top_ks)]
        )
        print_global_audit_estimate(population_by_stratum, audit_outcomes, audit_metrics)
    return result


def oracle_context_shape(
    context_steps: pl.DataFrame,
    destination_domain_sizes: dict[int, int],
    initial_zone: int,
) -> dict[str, float | int]:
    """Exact-search shape known before the oracle is allowed to expand states."""
    variable_activities: dict[tuple[str, int], int] = {}
    home_returns = 0
    repeated_anchor_visits = 0
    seen_anchors: set[int] = set()
    for step in context_steps.sort("layer").iter_rows(named=True):
        if step["fixed_destination"] == initial_zone:
            home_returns += 1
        if step["fixed_destination"] is not None:
            continue
        anchor_id = step["anchor_id"]
        if anchor_id is None:
            key = ("layer", int(step["layer"]))
        else:
            key = ("anchor", int(anchor_id))
            if int(anchor_id) in seen_anchors:
                repeated_anchor_visits += 1
            seen_anchors.add(int(anchor_id))
        variable_activities.setdefault(key, int(step["activity_id"]))
    domain_sizes = [destination_domain_sizes[activity] for activity in variable_activities.values()]
    lattice_log10 = sum(math.log10(size) for size in domain_sizes) if domain_sizes else 0.0
    return {
        "variables": len(domain_sizes),
        "lattice_log10": lattice_log10,
        "home_returns": home_returns,
        "repeated_anchor_visits": repeated_anchor_visits,
    }


def print_oracle_search_diagnostics(records: list[dict[str, float | int | str]]) -> None:
    """Expose the shape of completed and capped exact searches in audit mode."""
    if not records:
        return
    print("\nExact-oracle coverage diagnostics")
    print(
        "outcome | n | variables median | log10 lattice median | home returns median | "
        "repeated anchors median | states median | heap median | child-prune median"
    )
    for outcome in sorted({str(record["outcome"]) for record in records}):
        rows = [record for record in records if record["outcome"] == outcome]
        value = lambda name: median(float(record[name]) for record in rows)
        states = value("states_pushed") if outcome == "solved" else math.nan
        heap = value("maximum_heap_size") if outcome == "solved" else math.nan
        pruned = value("children_pruned_by_incumbent") if outcome == "solved" else math.nan
        print(
            f"{outcome} | {len(rows)} | {value('variables'):.1f} | "
            f"{value('lattice_log10'):.2f} | {value('home_returns'):.1f} | "
            f"{value('repeated_anchor_visits'):.1f} | "
            f"{states:.0f} | {heap:.0f} | {pruned:.0f}"
        )


def print_lowest_deep_cases(cases: list[dict[str, float | int | str]]) -> None:
    """Make the worst certified deep cases directly traceable from an audit."""
    deep = [case for case in cases if int(case["layers"]) >= 6]
    if not deep:
        return
    print("\nLowest certified deep Mass@10 cases")
    print("context | layers | stratum | recall | mass")
    for case in sorted(deep, key=lambda case: (float(case["mass"]), int(case["context"])))[:10]:
        print(
            f"{case['context']} | {case['layers']} | {case['stratum']} | "
            f"{case['recall']:.3f} | {case['mass']:.3f}"
        )


def write_design_manifest(path: Path, cases: list[dict[str, float | int | str]]) -> None:
    """Persist representative certified cases without tuning to one failure."""
    by_depth_and_outcome: dict[tuple[int, str], list[dict[str, float | int | str]]] = {}
    for case in cases:
        mass = float(case["mass"])
        outcome = "success" if mass >= 0.999 else "zero" if mass <= 0.0 else "partial"
        by_depth_and_outcome.setdefault((int(case["layers"]), outcome), []).append(case)
    selected = []
    for (_, outcome), candidates in sorted(by_depth_and_outcome.items()):
        # Median avoids selecting only the easiest success or the single most
        # pathological miss; the case ID makes this set reproducible.
        candidates.sort(key=lambda case: (float(case["mass"]), int(case["context"])))
        selected.append(candidates[len(candidates) // 2] | {"outcome": outcome})
    payload = {
        "description": "Certified exact-oracle cases selected by depth and bounded top-10 outcome.",
        "selection": "one median case for each observed depth x {success, partial, zero} bucket",
        "cases": selected,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote depth-diverse design manifest | {path} | cases={len(selected)}")


def main() -> None:
    args = parse_args()
    manifest = (
        ExperimentManifest.load(args.experiment_manifest)
        if args.experiment_manifest
        else None
    )
    if manifest and manifest.kind is not ExperimentKind.QUALITY_ONLY:
        raise ValueError(
            "the quality harness accepts only quality_only manifests; "
            "link its result from a quality_runtime throughput manifest"
        )
    if manifest and args.symmetric_config:
        raise ValueError(
            "--symmetric-config cannot override an immutable experiment manifest"
        )
    if manifest and args.candidate_option:
        raise ValueError(
            "--candidate-option cannot override an immutable experiment manifest"
        )
    if args.candidate_option and args.symmetric_config:
        raise ValueError(
            "use either --candidate-option or --symmetric-config, not both"
        )
    if manifest:
        if manifest.cohort.get("selector") != "stratified":
            raise ValueError("quality manifests require cohort.selector='stratified'")
        args.contexts_per_stratum = int(manifest.cohort["contexts_per_stratum"])
        args.exploration_seeds = [int(manifest.cohort["selection_seed"])]
    top_ks = list(dict.fromkeys(args.top_ks or [10]))
    if min(top_ks) <= 0 or args.oracle_depth < max(top_ks):
        raise ValueError("--top-k must be positive and --oracle-depth must cover every K")
    if args.frontier_width < max(top_ks):
        raise ValueError("--frontier-width must cover every requested K")
    if args.archetype_strata_limit <= 0:
        raise ValueError("--archetype-strata-limit must be positive")
    if args.design_manifest and args.contexts_per_stratum is None:
        raise ValueError("--design-manifest requires --contexts-per-stratum audit mode")
    files = resolve_snapshot_files(args.group_day_trips_folder)
    print("Preparing cached Grand Geneve inputs (read-only)...")
    od_costs = prepare_od_costs(files["transport_costs"], files["demand_groups"])
    destination_inputs = prepare_destination_inputs(
        files["destination_saturation"], files["demand_groups"]
    )
    destination_domain_sizes = {
        int(activity_id): int(domain_size)
        for activity_id, domain_size in destination_inputs.group_by("activity_id")
        .len()
        .iter_rows()
    }
    steps, initial_locations, _ = prepare_complete_contexts(
        activity_sequences_path=files["activity_sequences"],
        survey_plan_steps_path=files["survey_plan_steps"],
        demand_groups_path=files["demand_groups"],
        activity_dur_path=files["activity_dur"],
    )
    search = DestinationPlanSearch(od_costs=od_costs, destination_inputs=destination_inputs)
    oracle_cache = None
    if not args.no_oracle_cache:
        fingerprint = oracle_input_fingerprint(
            files,
            logit_scale=LOGIT_SCALE,
            update_plan_timings=True,
            use_shadow_prices=True,
        )
        attempt_fingerprint = oracle_attempt_fingerprint(
            fingerprint,
            use_bounded_incumbent=not args.no_bounded_incumbent,
        )
        oracle_cache = OracleCache(
            fingerprint,
            args.oracle_depth,
            args.max_states,
            attempt_fingerprint=attempt_fingerprint,
        )
        print(f"Oracle cache: {oracle_cache.path}")
    profiles = context_profiles(steps)
    supported = profiles.filter(pl.col("layers") >= 2)
    audit_supported = supported.filter(pl.col("layers") >= args.audit_min_layers)
    short = supported.filter(pl.col("layers") <= args.max_layers)
    print("Coverage (Grand Geneve terminal home is an input invariant)")
    print("stage | contexts")
    print(f"all | {profiles.height}")
    print(f"top-k supported (depth>=2) | {supported.height}")
    if args.contexts_per_stratum is None:
        print(f"comparison depth=3..{args.max_layers} | {short.height}")
    else:
        print(f"audit strata | {audit_supported['audit_stratum'].n_unique()}")
    profile_by_context = {
        int(context_id): (
            int(layers),
            int(anchor_count),
            int(anchor_activity_types),
            str(audit_stratum),
        )
        for context_id, layers, anchor_count, anchor_activity_types, audit_stratum in profiles.select(
            "context_id", "layers", "anchor_count", "anchor_activity_types", "audit_stratum"
        ).iter_rows()
    }
    population_by_stratum = {
        stratum: int(population)
        for stratum, population in audit_supported.group_by("audit_stratum").len().iter_rows()
    }
    if manifest:
        configurations: list[tuple[str, dict[str, Any]]] = [
            ("A", manifest.baseline),
            ("B", manifest.candidate),
        ]
    elif args.candidate_option:
        baseline_options = top_k_tuning_options(args)
        configurations = [
            ("A", baseline_options),
            (
                "B",
                apply_top_k_overrides(
                    baseline_options,
                    args.candidate_option,
                ),
            ),
        ]
    else:
        configurations = [
            (
                label,
                {
                    "symmetric_message_limit": message_limit,
                    "symmetric_state_limit": state_limit,
                    "symmetric_forward_proposal_limit": proposal_limit,
                },
            )
            for label, message_limit, state_limit, proposal_limit in symmetric_configurations(
                args
            )
        ]
    recorder = None
    cohort_identity = None
    if manifest:
        seed = int(manifest.cohort["selection_seed"])
        selected_ids = eligible_context_ids(profiles, args, seed)
        cohort_identity = manifest.verify_cohort(selected_ids)
        recorder = RunRecorder(
            manifest,
            cohort_identity=cohort_identity,
            root=args.run_root,
            metadata={
                "harness": "compare_bidirectional_top_k_grand_geneve",
                "contexts": len(selected_ids),
                "top_ks": top_ks,
                "oracle_depth": args.oracle_depth,
                "max_states": args.max_states,
            },
        )
        print(f"Cohort fingerprint: {cohort_identity}")
        print(f"Run artifact: {recorder.path}")
        recorder.progress("cohort_ready", contexts=len(selected_ids))
    oracle_memory: dict[int, tuple[pl.DataFrame, dict]] = {}
    if args.compact:
        if args.context_ids:
            print("row | context | config | seed | K | recall | mass")
        else:
            print(
                "config | seed | K | n | recall | mass | min | zero | "
                "bounded-ms | wall-s | oracle-hit/miss"
            )
    results = []
    for label, options in configurations:
        config_args = argparse.Namespace(**vars(args))
        for name, value in options.items():
            setattr(config_args, name, value)
        if not args.compact and len(configurations) > 1:
            differences = (
                ", ".join(
                    f"{name}={options[name]!r}"
                    for name in manifest.allowed_differences
                )
                if manifest
                else ", ".join(
                    f"{name}={options[name]!r}"
                    for name in (
                        value.partition("=")[0] for value in args.candidate_option
                    )
                )
                if args.candidate_option
                else ", ".join(f"{name}={value!r}" for name, value in options.items())
            )
            print(f"\nConfiguration {label}: {differences}")
        for exploration_seed in args.exploration_seeds or [42]:
            progress_callback = (
                (
                    lambda event, **values: recorder.progress(
                        event,
                        configuration=label,
                        seed=exploration_seed,
                        **values,
                    )
                )
                if recorder
                else None
            )
            result = compare_seed(
                config_args,
                label,
                top_ks,
                exploration_seed,
                search,
                steps,
                initial_locations,
                od_costs,
                destination_inputs,
                destination_domain_sizes,
                profiles,
                profile_by_context,
                population_by_stratum,
                oracle_cache,
                oracle_memory,
                progress_callback,
            )
            results.append(result)
            if recorder:
                recorder.progress(
                    "configuration_complete",
                    configuration=label,
                    seed=exploration_seed,
                    proven=result["proven"],
                )
    payload: dict[str, Any] = {
        "kind": manifest.kind.value if manifest else "untyped_quality",
        "cohort_fingerprint": cohort_identity,
        "results": results,
    }
    if manifest:
        by_configuration = {
            result["configuration"]: result
            for result in results
            if result["status"] == "completed"
        }
        if set(by_configuration) == {"A", "B"}:
            quality_summaries = {
                label: by_configuration[label]["quality_summary"]
                for label in ("A", "B")
            }
            verdict = quality_verdict(
                manifest.gates,
                quality_summaries["A"],
                quality_summaries["B"],
            )
        else:
            quality_summaries = {}
            verdict = {
                "status": "incomplete",
                "scope": "quality",
                "failures": [],
                "incomplete": ["A and B did not both complete"],
                "gate_results": [],
            }
        payload.update(
            {
                "hypothesis": manifest.hypothesis,
                "resolved_configs": {
                    "A": manifest.baseline,
                    "B": manifest.candidate,
                },
                "quality_summaries": quality_summaries,
                "verdict": verdict,
            }
        )
        print(f"quality verdict | {verdict['status'].upper()}")
        for reason in verdict["failures"] + verdict["incomplete"]:
            print(f"  {reason}")
        if recorder:
            recorder.finalize(payload, status=verdict["status"])
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.json_output}")


if __name__ == "__main__":
    main()
