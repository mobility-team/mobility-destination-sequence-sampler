"""Read-only top-K error check against the exact raw-zone oracle.

This keeps the complete raw-zone domain. The default comparison targets short,
oracle-proven contexts; ``--contexts-per-stratum`` instead audits all supported
depths and retains oracle failures as coverage data.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import hashlib
import json
import math
import os
import tempfile
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
        help="inspect candidate-pool coverage for one known context ID",
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
        "--compact",
        action="store_true",
        help="print one summary row per seed/configuration instead of distributions",
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


ORACLE_CACHE_VERSION = 1


def oracle_input_fingerprint(
    snapshot_files: dict[str, Path],
    *,
    logit_scale: float,
    update_plan_timings: bool,
    use_shadow_prices: bool,
) -> str:
    """Fingerprint exact-score inputs and the Rust code that defines them."""
    digest = hashlib.sha256()
    digest.update(f"oracle-cache-v{ORACLE_CACHE_VERSION}".encode())
    digest.update(
        json.dumps(
            {
                "logit_scale": logit_scale,
                "update_plan_timings": update_plan_timings,
                "use_shadow_prices": use_shadow_prices,
            },
            sort_keys=True,
        ).encode()
    )
    for name, path in sorted(snapshot_files.items()):
        digest.update(name.encode())
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    root = Path(__file__).resolve().parents[2]
    for relative_path in (
        "experiments/benchmarks/perf_grand_geneve_cache.py",
        "rust/oracle.rs",
        "rust/scoring.rs",
        "rust/model.rs",
        "rust/input.rs",
    ):
        digest.update((root / relative_path).read_bytes())
    return digest.hexdigest()[:20]


class OracleCache:
    """Persistent exact results; the fingerprint prevents stale proof reuse."""

    def __init__(self, fingerprint: str, oracle_depth: int, max_states: int) -> None:
        self.path = Path("experiments/.cache/oracle-top-k") / fingerprint
        self.oracle_depth = oracle_depth
        self.max_states = max_states

    def load_or_compute(
        self,
        context_id: int,
        compute: Callable[[], tuple[pl.DataFrame, dict[str, int]]],
    ) -> tuple[pl.DataFrame, dict[str, int], bool]:
        stem = f"context-{context_id}-k{self.oracle_depth}-states-{self.max_states}"
        table_path = self.path / f"{stem}.parquet"
        report_path = self.path / f"{stem}.json"
        error_path = self.path / f"{stem}.error.json"
        if table_path.exists() and report_path.exists():
            return pl.read_parquet(table_path), json.loads(report_path.read_text()), True
        if error_path.exists():
            raise ValueError(json.loads(error_path.read_text())["error"])
        self.path.mkdir(parents=True, exist_ok=True)
        def atomic_write(path: Path, payload: bytes) -> None:
            with tempfile.NamedTemporaryFile(dir=self.path, delete=False) as temporary:
                temporary.write(payload)
                temporary_path = Path(temporary.name)
            os.replace(temporary_path, path)

        try:
            oracle_table, oracle_report = compute()
        except ValueError as error:
            atomic_write(error_path, json.dumps({"error": str(error)}).encode())
            raise
        with tempfile.NamedTemporaryFile(dir=self.path, suffix=".parquet", delete=False) as temporary:
            temporary_path = Path(temporary.name)
        try:
            oracle_table.write_parquet(temporary_path)
            os.replace(temporary_path, table_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        atomic_write(report_path, json.dumps(oracle_report).encode())
        return oracle_table, oracle_report, False


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
    minimum_layers = 2 if args.contexts_per_stratum is not None else 3
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
            depth_band=pl.when(pl.col("layers") >= 6)
            .then(pl.lit("6+"))
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
    """Post-stratify oracle-certifiable strata; bounded failures carry zero mass."""
    covered = [
        stratum for stratum, outcome in outcomes.items() if outcome["oracle_completed"]
    ]
    covered_population = sum(population_by_stratum[stratum] for stratum in covered)
    population = sum(population_by_stratum.values())
    if not covered_population:
        return
    print(
        "\nPilot quality estimate (post-stratified over oracle-certifiable strata; "
        f"population coverage={covered_population / population:.1%})"
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
        "missed_base_pool_mass": 0.0,
        "missed_beam_mass": 0.0,
    }


def oracle_failure_kind(error: ValueError) -> str:
    """Keep oracle limits, infeasibility, and internal errors distinct in audits."""
    message = str(error)
    if "exceeded max_states" in message:
        return "state_limited"
    if "no feasible destination sequence" in message:
        return "infeasible"
    return "oracle_error"


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
    """Show legacy heuristic coverage; this is not active factor-map support."""
    steps = context_steps.sort("layer").to_dicts()
    stitch_layer = stitch_layer_index(len(steps), stitch_bias)
    print(
        f"legacy heuristic trace (not active factor-map support): "
        f"stitch-layer={stitch_layer}, pool={candidate_count}+{candidate_count}"
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
                sources.append("outside-legacy-pool")
            direction = "backward" if reverse else "forward"
            print(
                f"    layer={layer} {direction} reference={reference} target={target}: "
                f"{', '.join(sources)}"
            )
        if missing_base_pool:
            print("    legacy diagnosis: outside heuristic pool; active loss stage unknown")
        else:
            print("    legacy diagnosis: heuristic-supported; active loss stage unknown")


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
            missed_diagnoses[zones] = "inside-legacy-pool; active stage unknown"
        else:
            missed_base_pool_mass += probability
            missed_diagnoses[zones] = "outside-legacy-pool; active stage unknown"
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
    profiles: pl.DataFrame,
    profile_by_context: dict[int, tuple[int, int, int, str]],
    population_by_stratum: dict[str, int],
    oracle_cache: OracleCache | None,
    oracle_memory: dict[int, tuple[pl.DataFrame, dict]],
) -> None:
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
            "missed_base_pool_mass": [],
            "missed_beam_mass": [],
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
    started = time.perf_counter()
    for context_id in eligible_context_ids(profiles, args, exploration_seed):
        layers, anchor_count, anchor_activity_types, audit_stratum = profile_by_context[context_id]
        if audit_mode:
            audit_outcomes[audit_stratum]["sampled"] += 1
        context_steps = steps.filter(pl.col("context_id") == context_id)
        context_initial = initial_locations.filter(pl.col("context_id") == context_id)
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
            )
            if context_id in oracle_memory:
                oracle_table, oracle_report = oracle_memory[context_id]
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
            if audit_mode:
                audit_outcomes[audit_stratum][oracle_failure_kind(error)] += 1
            if args.verbose:
                print(f"context={context_id} oracle skipped: {error}")
            continue
        oracle = ranked_plans(oracle_table)
        if audit_mode:
            audit_outcomes[audit_stratum]["oracle_completed"] += 1
        if args.trace_context == context_id:
            trace_oracle_candidate_coverage(
                context_steps,
                int(context_initial.item(0, "initial_zone")),
                oracle,
                od_costs,
                destination_inputs,
                args.proposal_limit_per_source,
                max(top_ks),
                args.stitch_bias,
            )
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
                bounded_table, _ = search.top_k(
                    steps=context_steps,
                    initial_locations=context_initial,
                    logit_scale=LOGIT_SCALE,
                    update_plan_timings=True,
                    use_shadow_prices=True,
                    exploration_seed=exploration_seed,
                    **top_k_tuning_options(args),
                    top_k=top_k,
                    n_threads=1,
                    skip_infeasible=False,
                )
                bounded_search_seconds[top_k] += time.perf_counter() - bounded_started
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
                context_steps,
                int(context_initial.item(0, "initial_zone")),
                od_costs,
                destination_inputs,
                args.proposal_limit_per_source,
                args.stitch_bias,
                args.verbose or args.trace_context is not None,
            )
        if len(context_metrics) != len(top_ks):
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
            if args.compact and args.context_ids:
                print(
                    f"case | {context_id} | {configuration_label} | {exploration_seed} | "
                    f"{top_k} | {metrics['recall']:.3f} | "
                    f"{metrics['retained_top_k_mass']:.3f}"
                )
        proven += 1
        if audit_mode:
            audit_outcomes[audit_stratum]["proven"] += 1
        if not audit_mode and proven == args.contexts:
            break
    if audit_mode:
        print_audit_coverage(population_by_stratum, audit_outcomes)
    if not proven:
        if audit_mode:
            print("No sampled context completed the exact top-K oracle.")
            return
        raise RuntimeError("no context completed the exact top-K oracle")
    wall_seconds = time.perf_counter() - started
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
        return
    print(
        f"\nseed={exploration_seed} proven-contexts={proven} skipped={skipped} "
        f"wall={wall_seconds:.3f}s "
        f"exact={exact_search_seconds / max(oracle_cache_misses, 1) * 1e3:.2f}ms/context "
        f"oracle-cache={oracle_cache_hits} hit/{oracle_cache_misses} miss"
    )
    print("Search performance and miss diagnosis")
    print("  K | recall | bounded ms | speedup | missed outside/inside legacy pool")
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
            f"| {speedup:<7} "
            f"| {mean(metrics['missed_base_pool_mass']):.3f}/{mean(metrics['missed_beam_mass']):.3f}"
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
        print_global_audit_estimate(population_by_stratum, audit_outcomes, audit_metrics)


def main() -> None:
    args = parse_args()
    top_ks = list(dict.fromkeys(args.top_ks or [10]))
    if min(top_ks) <= 0 or args.oracle_depth < max(top_ks):
        raise ValueError("--top-k must be positive and --oracle-depth must cover every K")
    if args.frontier_width < max(top_ks):
        raise ValueError("--frontier-width must cover every requested K")
    if args.archetype_strata_limit <= 0:
        raise ValueError("--archetype-strata-limit must be positive")
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
    oracle_cache = None
    if not args.no_oracle_cache:
        fingerprint = oracle_input_fingerprint(
            files,
            logit_scale=LOGIT_SCALE,
            update_plan_timings=True,
            use_shadow_prices=True,
        )
        oracle_cache = OracleCache(fingerprint, args.oracle_depth, args.max_states)
        print(f"Oracle cache: {oracle_cache.path}")
    profiles = context_profiles(steps)
    supported = profiles.filter(pl.col("layers") >= 2)
    short = supported.filter(pl.col("layers") <= args.max_layers)
    print("Coverage (Grand Geneve terminal home is an input invariant)")
    print("stage | contexts")
    print(f"all | {profiles.height}")
    print(f"top-k supported (depth>=2) | {supported.height}")
    if args.contexts_per_stratum is None:
        print(f"comparison depth=3..{args.max_layers} | {short.height}")
    else:
        print(f"audit strata | {supported['audit_stratum'].n_unique()}")
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
        for stratum, population in supported.group_by("audit_stratum").len().iter_rows()
    }
    configurations = symmetric_configurations(args)
    oracle_memory: dict[int, tuple[pl.DataFrame, dict]] = {}
    if args.compact:
        if args.context_ids:
            print("row | context | config | seed | K | recall | mass")
        else:
            print(
                "config | seed | K | n | recall | mass | min | zero | "
                "bounded-ms | wall-s | oracle-hit/miss"
            )
    for label, message_limit, state_limit, proposal_limit in configurations:
        config_args = argparse.Namespace(**vars(args))
        config_args.symmetric_message_limit = message_limit
        config_args.symmetric_state_limit = state_limit
        config_args.symmetric_forward_proposal_limit = proposal_limit
        if not args.compact and len(configurations) > 1:
            print(
                f"\nConfiguration {label}: messages={message_limit}, "
                f"states={state_limit}, proposals={proposal_limit}"
            )
        for exploration_seed in args.exploration_seeds or [42]:
            compare_seed(
                config_args,
                label,
                top_ks,
                exploration_seed,
                search,
                steps,
                initial_locations,
                od_costs,
                destination_inputs,
                profiles,
                profile_by_context,
                population_by_stratum,
                oracle_cache,
                oracle_memory,
            )


if __name__ == "__main__":
    main()
