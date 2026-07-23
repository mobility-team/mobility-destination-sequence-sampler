"""Measure the structural search problem and the locations used by bounded top-K.

This is a read-only diagnostic.  It prepares the same Grand Geneve snapshot as
the maintained quality and throughput harnesses, analyzes every unique context,
then runs a configurable calibrated output sample to measure where returned
plans actually live in the 1,110-zone lattice.
"""

from __future__ import annotations

import argparse
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median

import polars as pl

from mobility_destination_sequence_sampler import DestinationPlanSearch

from experiments.benchmarks.perf_bidirectional_grand_geneve import (
    calibrated_contexts,
)
from experiments.benchmarks.perf_grand_geneve_cache import (
    DEFAULT_GROUP_DAY_TRIPS_FOLDER,
    LOGIT_SCALE,
    prepare_complete_contexts,
    prepare_destination_inputs,
    prepare_od_costs,
    resolve_snapshot_files,
)
from experiments.oracle_cache import oracle_input_fingerprint
from experiments.top_k_config import ACTIVE_TOP_K_DEFAULTS, top_k_tuning_options


ACTIVITY_NAMES = {
    0: "home",
    1: "work",
    2: "studies",
    3: "shopping",
    4: "leisure",
    5: "other",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group-day-trips-folder",
        type=Path,
        default=DEFAULT_GROUP_DAY_TRIPS_FOLDER,
    )
    parser.add_argument(
        "--output-contexts",
        type=int,
        default=10_000,
        help="calibrated contexts on which to locate top-K output; zero skips search",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--exploration-seed", type=int, default=42)
    parser.add_argument(
        "--benchmark-exact-width",
        type=int,
        action="append",
        default=[],
        help="recompute cached contexts of this induced width with the exact oracle",
    )
    return parser.parse_args()


def quantile(values: list[float | int], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def describe(values: list[float | int]) -> str:
    return (
        f"mean={mean(values):.2f} p10={quantile(values, 0.1):.2f} "
        f"p50={quantile(values, 0.5):.2f} p90={quantile(values, 0.9):.2f} "
        f"max={max(values):.2f}"
    )


def print_counter(label: str, values: Counter[object]) -> None:
    print(f"\n{label}")
    for key, count in sorted(values.items(), key=lambda item: str(item[0])):
        print(f"  {key}: {count}")


@dataclass(frozen=True)
class Variable:
    kind: str
    identifier: int
    activity_id: int


@dataclass
class ContextShape:
    context_id: int
    layers: int
    fixed_layers: int
    variables: int
    anchor_variables: int
    repeated_anchor_variables: int
    cross_home_anchor_variables: int
    home_tours: int
    longest_tour: int
    min_fill_width: int
    max_clique_log10: float
    conditioned_width: int
    conditioned_max_clique_log10: float
    structure_key: tuple[object, ...]
    scoring_shape_key: tuple[object, ...]


def eliminate_min_fill(
    variables: list[Variable],
    scopes: list[set[int]],
    *,
    conditioned: set[int] | None = None,
    domain_sizes: dict[int, int],
) -> tuple[int, float]:
    """Return greedy min-fill width and largest materialized-clique log size."""
    conditioned = conditioned or set()
    active = set(range(len(variables))) - conditioned
    adjacency = {variable: set() for variable in active}
    for scope in scopes:
        live = list(scope - conditioned)
        for left in live:
            for right in live:
                if left != right:
                    adjacency[left].add(right)
    width = 0
    max_clique_log10 = 0.0
    while active:
        def priority(variable: int) -> tuple[int, float, int, int]:
            neighbors = adjacency[variable] & active
            missing = sum(
                1
                for left in neighbors
                for right in neighbors
                if left < right and right not in adjacency[left]
            )
            clique_log = math.log10(domain_sizes[variables[variable].activity_id])
            clique_log += sum(
                math.log10(domain_sizes[variables[item].activity_id])
                for item in neighbors
            )
            return missing, clique_log, len(neighbors), variable

        chosen = min(active, key=priority)
        neighbors = adjacency[chosen] & active
        width = max(width, len(neighbors))
        clique_log = math.log10(domain_sizes[variables[chosen].activity_id])
        clique_log += sum(
            math.log10(domain_sizes[variables[item].activity_id])
            for item in neighbors
        )
        max_clique_log10 = max(max_clique_log10, clique_log)
        for left in neighbors:
            adjacency[left].update(neighbors - {left})
        active.remove(chosen)
    return width, max_clique_log10


def context_shape(
    context_id: int,
    rows: list[dict[str, object]],
    domain_sizes: dict[int, int],
) -> ContextShape:
    rows.sort(key=lambda row: int(row["layer"]))
    variables: list[Variable] = []
    variable_by_key: dict[tuple[str, int], int] = {}
    variable_by_layer: list[int | None] = []
    anchor_layers: defaultdict[int, list[int]] = defaultdict(list)
    tour_by_layer: list[int] = []
    tour = 0
    longest_tour = 0
    current_tour_length = 0
    structure_parts: list[object] = []
    scoring_parts: list[object] = []

    for layer, row in enumerate(rows):
        fixed = row["fixed_destination"] is not None
        activity_id = int(row["activity_id"])
        anchor = None if row["anchor_id"] is None else int(row["anchor_id"])
        if fixed:
            variable_by_layer.append(None)
            tour_by_layer.append(tour)
            longest_tour = max(longest_tour, current_tour_length)
            current_tour_length = 0
            tour += 1
            structure_parts.append((activity_id, "H"))
        else:
            key = ("a", anchor) if anchor is not None else ("l", layer)
            if key not in variable_by_key:
                variable_by_key[key] = len(variables)
                variables.append(Variable(key[0], key[1], activity_id))
            variable_by_layer.append(variable_by_key[key])
            tour_by_layer.append(tour)
            current_tour_length += 1
            structure_parts.append(
                (activity_id, "A", anchor)
                if anchor is not None
                else (activity_id, "V")
            )
            if anchor is not None:
                anchor_layers[anchor].append(layer)
        scoring_parts.append(
            (
                activity_id,
                structure_parts[-1][1:],
                row["departure_time"],
                row["arrival_time"],
                row["arrival_time_rigidity"],
                row["departure_time_rigidity"],
                row["duration_per_person"],
                row["mean_duration_per_person"],
                row["min_activity_time"],
            )
        )
    longest_tour = max(longest_tour, current_tour_length)

    scopes: list[set[int]] = []
    for layer in range(len(rows)):
        scope = {
            variable
            for adjacent in (layer - 1, layer, layer + 1)
            if 0 <= adjacent < len(rows)
            if (variable := variable_by_layer[adjacent]) is not None
        }
        if scope:
            scopes.append(scope)
    anchor_variables = {
        index for index, variable in enumerate(variables) if variable.kind == "a"
    }
    repeated = {
        anchor
        for anchor, layers in anchor_layers.items()
        if len(layers) > 1
    }
    cross_home = {
        anchor
        for anchor, layers in anchor_layers.items()
        if len({tour_by_layer[layer] for layer in layers}) > 1
    }
    width, clique = eliminate_min_fill(
        variables, scopes, domain_sizes=domain_sizes
    )
    conditioned_width, conditioned_clique = eliminate_min_fill(
        variables,
        scopes,
        conditioned=anchor_variables,
        domain_sizes=domain_sizes,
    )
    return ContextShape(
        context_id=context_id,
        layers=len(rows),
        fixed_layers=sum(row["fixed_destination"] is not None for row in rows),
        variables=len(variables),
        anchor_variables=len(anchor_variables),
        repeated_anchor_variables=len(repeated),
        cross_home_anchor_variables=len(cross_home),
        home_tours=tour,
        longest_tour=longest_tour,
        min_fill_width=width,
        max_clique_log10=clique,
        conditioned_width=conditioned_width,
        conditioned_max_clique_log10=conditioned_clique,
        structure_key=tuple(structure_parts),
        scoring_shape_key=tuple(scoring_parts),
    )


def analyze_inputs(
    od_costs: pl.DataFrame,
    destination_inputs: pl.DataFrame,
    steps: pl.DataFrame,
    initial_locations: pl.DataFrame,
    raw_context_count: int,
) -> list[ContextShape]:
    zones = set(od_costs["origin"].to_list()) | set(
        od_costs["destination"].to_list()
    )
    zone_count = len(zones)
    outdegrees = od_costs.group_by("origin").len()["len"].to_list()
    print("INPUT LATTICE")
    print(
        f"zones={zone_count} od-pairs={od_costs.height} "
        f"density={od_costs.height / (zone_count * zone_count):.3%}"
    )
    print(f"OD outdegree: {describe(outdegrees)}")
    print(
        "OD edge cost: "
        + describe(od_costs["cost"].cast(pl.Float64).to_list())
    )
    print(
        "OD edge time: "
        + describe(od_costs["time"].cast(pl.Float64).to_list())
    )

    domains = {
        int(activity): set(values)
        for activity, values in destination_inputs.group_by("activity_id").agg(
            pl.col("destination")
        ).iter_rows()
    }
    domain_sizes = {
        activity: len(values) for activity, values in domains.items()
    }
    print("\nDestination domains")
    for activity, size in sorted(domain_sizes.items()):
        print(
            f"  {ACTIVITY_NAMES[activity]:8s} activity={activity} "
            f"zones={size} ({size / zone_count:.1%})"
        )
    print("\nDestination-domain Jaccard overlap")
    activities = sorted(domains)
    for left_index, left in enumerate(activities):
        for right in activities[left_index + 1 :]:
            intersection = len(domains[left] & domains[right])
            union = len(domains[left] | domains[right])
            print(
                f"  {ACTIVITY_NAMES[left]:8s}/{ACTIVITY_NAMES[right]:8s} "
                f"intersection={intersection:4d} jaccard={intersection / union:.1%}"
            )

    print("\nCONTEXT FACTOR GRAPHS")
    print(
        f"raw-contexts={raw_context_count} unique-prepared={initial_locations.height} "
        f"preparation-collapse={raw_context_count / initial_locations.height:.2f}x "
        f"steps={steps.height}"
    )
    shapes = [
        context_shape(int(context_id), rows, domain_sizes)
        for context_id, rows in (
            steps.group_by("context_id", maintain_order=True)
            .agg(pl.struct(pl.exclude("context_id")))
            .iter_rows()
        )
    ]
    print_counter("contexts by depth", Counter(shape.layers for shape in shapes))
    print_counter(
        "contexts by greedy min-fill width",
        Counter(shape.min_fill_width for shape in shapes),
    )
    print_counter(
        "contexts by width after conditioning all anchor variables",
        Counter(shape.conditioned_width for shape in shapes),
    )
    print_counter(
        "contexts by cross-home anchor count",
        Counter(shape.cross_home_anchor_variables for shape in shapes),
    )
    print(
        "\nCollapsed variable count: "
        + describe([shape.variables for shape in shapes])
    )
    print(
        "Anchor-variable count: "
        + describe([shape.anchor_variables for shape in shapes])
    )
    print(
        "Longest home-bounded variable tour: "
        + describe([shape.longest_tour for shape in shapes])
    )
    print(
        "Largest greedy-elimination table (log10 assignments): "
        + describe([shape.max_clique_log10 for shape in shapes])
    )
    print(
        "After anchor conditioning (log10 assignments): "
        + describe([shape.conditioned_max_clique_log10 for shape in shapes])
    )
    cross_home = sum(shape.cross_home_anchor_variables > 0 for shape in shapes)
    repeated = sum(shape.repeated_anchor_variables > 0 for shape in shapes)
    print(
        f"repeated-anchor contexts={repeated} ({repeated / len(shapes):.1%}); "
        f"cross-home-anchor contexts={cross_home} ({cross_home / len(shapes):.1%})"
    )

    structure_counts = Counter(shape.structure_key for shape in shapes)
    scoring_shape_counts = Counter(shape.scoring_shape_key for shape in shapes)
    home_counts = Counter(initial_locations["initial_zone"].to_list())
    print("\nWORKLOAD REUSE")
    for label, counts in (
        ("activity/anchor/home pattern", structure_counts),
        ("home-independent timing/scoring shape", scoring_shape_counts),
        ("home zone", home_counts),
    ):
        print(
            f"{label}: unique={len(counts)} reuse={len(shapes) / len(counts):.2f}x "
            f"singletons={sum(value == 1 for value in counts.values()) / len(counts):.1%} "
            f"max={max(counts.values())}"
        )
    return shapes


def share_in_top(counter: Counter[int], cutoffs: tuple[int, ...]) -> str:
    total = sum(counter.values())
    ordered = sorted(counter.values(), reverse=True)
    return " ".join(
        f"top-{cutoff}={sum(ordered[:cutoff]) / total:.1%}"
        for cutoff in cutoffs
    )


def ranked_od(od_costs: pl.DataFrame) -> pl.DataFrame:
    return (
        od_costs.with_columns(
            time_rank=pl.col("time").rank("ordinal").over("origin"),
            cost_rank=pl.col("cost").rank("ordinal").over("origin"),
            origin_degree=pl.len().over("origin"),
        )
        .select(
            "origin",
            "destination",
            "cost",
            "time",
            "time_rank",
            "cost_rank",
            "origin_degree",
        )
    )


def locate_rows(
    returned: pl.DataFrame,
    steps: pl.DataFrame,
    initial_locations: pl.DataFrame,
    od_ranks: pl.DataFrame,
    destination_inputs: pl.DataFrame,
) -> pl.DataFrame:
    static_ranks = destination_inputs.with_columns(
        capacity_rank=pl.col("opportunity_capacity")
        .rank("ordinal", descending=True)
        .over("activity_id"),
        shadow_rank=pl.col("shadow_price")
        .rank("ordinal", descending=True)
        .over("activity_id"),
        attractive_score=(
            pl.col("opportunity_capacity").log()
            + pl.col("shadow_price")
        ),
    ).with_columns(
        attractive_rank=pl.col("attractive_score")
        .rank("ordinal", descending=True)
        .over("activity_id"),
    ).select(
        "activity_id",
        "destination",
        "capacity_rank",
        "shadow_rank",
        "attractive_rank",
    )
    home_ranks = od_ranks.rename(
        {
            "origin": "initial_zone",
            "time_rank": "home_time_rank",
            "cost_rank": "home_cost_rank",
            "origin_degree": "home_degree",
            "cost": "home_cost",
            "time": "home_time",
        }
    )
    inbound_ranks = od_ranks.rename(
        {
            "time_rank": "inbound_time_rank",
            "cost_rank": "inbound_cost_rank",
            "origin_degree": "inbound_degree",
            "cost": "inbound_cost",
            "time": "inbound_time",
        }
    )
    return (
        returned.join(
            steps.select(
                "context_id",
                "layer",
                "activity_id",
                "fixed_destination",
            ),
            on=["context_id", "layer"],
        )
        .join(initial_locations, on="context_id")
        .with_columns(
            variable=pl.col("fixed_destination").is_null(),
            is_home=pl.col("destination") == pl.col("initial_zone"),
        )
        .join(
            home_ranks,
            on=["initial_zone", "destination"],
            how="left",
        )
        .join(
            inbound_ranks,
            on=["origin", "destination"],
            how="left",
        )
        .join(
            static_ranks,
            on=["activity_id", "destination"],
            how="left",
        )
        .with_columns(
            home_time_percentile=pl.col("home_time_rank")
            / pl.col("home_degree"),
            home_cost_percentile=pl.col("home_cost_rank")
            / pl.col("home_degree"),
            inbound_time_percentile=pl.col("inbound_time_rank")
            / pl.col("inbound_degree"),
            inbound_cost_percentile=pl.col("inbound_cost_rank")
            / pl.col("inbound_degree"),
        )
    )


def print_location_summary(
    label: str, located: pl.DataFrame, top_k: int
) -> None:
    variable = located.filter("variable")
    destination_counter = Counter(variable["destination"].to_list())
    print(f"\n{label}")
    print(
        f"contexts={located['context_id'].n_unique()} rows={located.height} "
        f"variable-rows={variable.height}; destination concentration "
        f"unique={len(destination_counter)} "
        f"{share_in_top(destination_counter, (1, 10, 32, 100))}"
    )
    print("Chosen destination rank among all zones from home")
    for metric in (
        "home_time_rank",
        "home_time_percentile",
        "home_cost_rank",
        "home_cost_percentile",
    ):
        print(
            f"  {metric}: {describe(variable[metric].drop_nulls().to_list())}"
        )
    print("Chosen destination rank among all zones from its preceding stop")
    for metric in (
        "inbound_time_rank",
        "inbound_time_percentile",
        "inbound_cost_rank",
        "inbound_cost_percentile",
    ):
        print(
            f"  {metric}: {describe(variable[metric].drop_nulls().to_list())}"
        )
    print("Chosen destination static rank within its activity domain")
    for metric in ("capacity_rank", "shadow_rank", "attractive_rank"):
        print(
            f"  {metric}: {describe(variable[metric].drop_nulls().to_list())}"
        )
    print(f"same-as-home variable choices={variable['is_home'].mean():.1%}")
    print("Simple input-index shortlist coverage of returned variable rows")
    for cutoff in (8, 16, 32, 64, 128, 256):
        near_home = (
            (pl.col("home_time_rank") <= cutoff)
            | (pl.col("home_cost_rank") <= cutoff)
        )
        near_inbound = (
            (pl.col("inbound_time_rank") <= cutoff)
            | (pl.col("inbound_cost_rank") <= cutoff)
        )
        attractive = (
            (pl.col("capacity_rank") <= cutoff)
            | (pl.col("shadow_rank") <= cutoff)
        )
        production_base = (
            (pl.col("inbound_cost_rank") <= cutoff)
            | (pl.col("attractive_rank") <= cutoff)
        )
        print(
            f"  N={cutoff:3d}: home={variable.select(near_home.mean()).item():.1%} "
            f"inbound={variable.select(near_inbound.mean()).item():.1%} "
            f"static={variable.select(attractive.mean()).item():.1%} "
            f"cost+attractive={variable.select(production_base.mean()).item():.1%} "
            f"home+static={variable.select((near_home | attractive).mean()).item():.1%} "
            f"inbound+static={variable.select((near_inbound | attractive).mean()).item():.1%}"
        )

    print("\nLocated support by activity")
    for activity, frame in variable.partition_by(
        "activity_id", as_dict=True
    ).items():
        activity_id = int(activity[0])
        counter = Counter(frame["destination"].to_list())
        print(
            f"  {ACTIVITY_NAMES[activity_id]:8s} rows={frame.height:7d} "
            f"zones={len(counter):4d} {share_in_top(counter, (10, 32, 100))} "
            f"home-time-rank-p50={frame['home_time_rank'].median():.0f} "
            f"p90={frame['home_time_rank'].quantile(.9):.0f}"
        )

    support = (
        variable.group_by(["context_id", "layer"])
        .agg(
            distinct_zones=pl.col("destination").n_unique(),
            best_zone=pl.col("destination")
            .sort_by("draw_id")
            .first(),
        )
    )
    per_context = support.group_by("context_id").agg(
        mean_layer_support=pl.col("distinct_zones").mean(),
        max_layer_support=pl.col("distinct_zones").max(),
        best_plan_union_zones=pl.col("best_zone").n_unique(),
    )
    print(f"\nWithin-context top-{top_k} support")
    for metric in (
        "mean_layer_support",
        "max_layer_support",
        "best_plan_union_zones",
    ):
        print(f"  {metric}: {describe(per_context[metric].to_list())}")


def analyze_cached_exact_outputs(
    snapshot_files: dict[str, Path],
    od_costs: pl.DataFrame,
    destination_inputs: pl.DataFrame,
    steps: pl.DataFrame,
    initial_locations: pl.DataFrame,
    shapes: list[ContextShape],
    args: argparse.Namespace,
) -> None:
    fingerprint = oracle_input_fingerprint(
        snapshot_files,
        logit_scale=LOGIT_SCALE,
        update_plan_timings=True,
        use_shadow_prices=True,
    )
    cache_path = (
        Path("experiments/.cache/oracle-top-k") / fingerprint
    )
    pattern = re.compile(
        r"context-(?P<context>\d+)-k(?P<k>\d+)-states-(?P<states>\d+)\.parquet"
    )
    best: dict[int, tuple[int, int, Path]] = {}
    for path in cache_path.glob("*.parquet"):
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        context_id = int(match.group("context"))
        priority = (int(match.group("k")), int(match.group("states")), path)
        if context_id not in best or priority[:2] > best[context_id][:2]:
            best[context_id] = priority
    if not best:
        print(f"\nNo cached exact outputs at {cache_path}")
        return
    tables = [
        pl.read_parquet(priority[2]).filter(pl.col("draw_id") <= 10)
        for priority in best.values()
    ]
    exact = pl.concat(tables)
    exact_steps = steps.join(
        exact.select("context_id").unique(), on="context_id", how="semi"
    )
    exact_initial = initial_locations.join(
        exact.select("context_id").unique(), on="context_id", how="semi"
    )
    located = locate_rows(
        exact,
        exact_steps,
        exact_initial,
        ranked_od(od_costs),
        destination_inputs,
    )
    print_location_summary(
        f"CACHED EXACT TOP-10 LOCATION ({cache_path}, one certificate/context)",
        located,
        10,
    )
    benchmark_exact_widths(
        exact,
        od_costs,
        destination_inputs,
        exact_steps,
        exact_initial,
        shapes,
        args,
    )
    compare_bounded_to_cached_exact(
        exact,
        located,
        od_costs,
        destination_inputs,
        exact_steps,
        exact_initial,
        shapes,
        args,
    )


def benchmark_exact_widths(
    cached_exact: pl.DataFrame,
    od_costs: pl.DataFrame,
    destination_inputs: pl.DataFrame,
    steps: pl.DataFrame,
    initial_locations: pl.DataFrame,
    shapes: list[ContextShape],
    args: argparse.Namespace,
) -> None:
    if not args.benchmark_exact_width:
        return
    shape_by_context = {shape.context_id: shape for shape in shapes}
    search = DestinationPlanSearch(
        od_costs=od_costs, destination_inputs=destination_inputs
    )
    for width in dict.fromkeys(args.benchmark_exact_width):
        selected_ids = [
            context_id
            for context_id in cached_exact["context_id"].unique().to_list()
            if shape_by_context[int(context_id)].min_fill_width == width
        ]
        selected = pl.DataFrame({"context_id": selected_ids})
        selected_steps = steps.join(
            selected, on="context_id", how="semi"
        )
        selected_initial = initial_locations.join(
            selected, on="context_id", how="semi"
        )
        started = time.perf_counter()
        recomputed, report = search.exact_top_k(
            steps=selected_steps,
            initial_locations=selected_initial,
            logit_scale=LOGIT_SCALE,
            update_plan_timings=True,
            use_shadow_prices=True,
            top_k=10,
            max_states=2_000_000,
            n_threads=args.threads,
            skip_infeasible=False,
        )
        elapsed = time.perf_counter() - started
        cached = cached_exact.join(
            selected, on="context_id", how="semi"
        )
        fingerprints_match = {
            context_id: [zones for zones, _ in plans]
            for context_id, plans in plan_scores(recomputed).items()
        } == {
            context_id: [zones for zones, _ in plans]
            for context_id, plans in plan_scores(cached).items()
        }
        print(
            f"\nEXACT WIDTH-{width} RECOMPUTATION: contexts={len(selected_ids)} "
            f"wall={elapsed:.3f}s ({elapsed / max(len(selected_ids), 1) * 1e3:.2f}ms/context) "
            f"states-popped={report['states_popped']} "
            f"children={report['children_considered']} "
            f"max-heap={report['maximum_heap_size']} "
            f"cached-top10-match={fingerprints_match}"
        )


def plan_scores(table: pl.DataFrame) -> dict[int, list[tuple[tuple[int, ...], float]]]:
    plans: defaultdict[int, list[tuple[tuple[int, ...], float]]] = defaultdict(list)
    grouped = table.group_by(["context_id", "draw_id"]).agg(
        pl.col("destination").sort_by("layer").alias("zones"),
        pl.col("total_log_weight").first().alias("score"),
    )
    for context_id, _, zones, score in grouped.iter_rows():
        plans[int(context_id)].append(
            (tuple(int(zone) for zone in zones), float(score))
        )
    for context_plans in plans.values():
        context_plans.sort(key=lambda item: item[1], reverse=True)
    return dict(plans)


def compare_bounded_to_cached_exact(
    exact: pl.DataFrame,
    exact_located: pl.DataFrame,
    od_costs: pl.DataFrame,
    destination_inputs: pl.DataFrame,
    steps: pl.DataFrame,
    initial_locations: pl.DataFrame,
    shapes: list[ContextShape],
    args: argparse.Namespace,
) -> None:
    search = DestinationPlanSearch(
        od_costs=od_costs, destination_inputs=destination_inputs
    )
    started = time.perf_counter()
    bounded, report = search.top_k(
        steps=steps,
        initial_locations=initial_locations,
        logit_scale=LOGIT_SCALE,
        update_plan_timings=True,
        use_shadow_prices=True,
        exploration_seed=args.exploration_seed,
        **ACTIVE_TOP_K_DEFAULTS,
        top_k=10,
        n_threads=args.threads,
        skip_infeasible=True,
    )
    elapsed = time.perf_counter() - started
    exact_plans = plan_scores(exact)
    bounded_plans = plan_scores(bounded)
    shape_by_context = {shape.context_id: shape for shape in shapes}
    traced_support_mass = trace_internal_support_mass(
        search,
        exact_plans,
        steps,
        initial_locations,
        args,
    )
    bounded_support = {
        (int(context_id), int(layer)): {
            int(destination) for destination in destinations
        }
        for context_id, layer, destinations in (
            bounded.group_by(["context_id", "layer"])
            .agg(pl.col("destination"))
            .iter_rows()
        )
    }
    transition_lattices: dict[int, dict[int, set[tuple[int, ...]]]] = {}
    production_lattices: dict[int, dict[int, set[tuple[int, ...]]]] = {}
    exact_variable = exact_located.filter("variable")
    for cutoff in (16, 32, 64, 128):
        eligible = (
            exact_variable.with_columns(
                eligible=(
                    (pl.col("inbound_time_rank") <= cutoff)
                    | (pl.col("inbound_cost_rank") <= cutoff)
                    | (pl.col("capacity_rank") <= cutoff)
                    | (pl.col("shadow_rank") <= cutoff)
                )
            )
            .group_by(["context_id", "draw_id"])
            .agg(pl.col("eligible").all())
            .filter("eligible")
            .select("context_id", "draw_id")
        )
        eligible_plans = plan_scores(
            exact.join(
                eligible,
                on=["context_id", "draw_id"],
                how="semi",
            )
        )
        transition_lattices[cutoff] = {
            context_id: {zones for zones, _ in plans}
            for context_id, plans in eligible_plans.items()
        }
        production_eligible = (
            exact_variable.with_columns(
                eligible=(
                    (pl.col("inbound_cost_rank") <= cutoff)
                    | (pl.col("attractive_rank") <= cutoff)
                )
            )
            .group_by(["context_id", "draw_id"])
            .agg(pl.col("eligible").all())
            .filter("eligible")
            .select("context_id", "draw_id")
        )
        production_plans = plan_scores(
            exact.join(
                production_eligible,
                on=["context_id", "draw_id"],
                how="semi",
            )
        )
        production_lattices[cutoff] = {
            context_id: {zones for zones, _ in plans}
            for context_id, plans in production_plans.items()
        }
    variable_keys: dict[int, list[tuple[str, int] | None]] = {}
    for context_id, context_rows in (
        steps.sort(["context_id", "layer"])
        .group_by("context_id", maintain_order=True)
        .agg(pl.struct(pl.exclude("context_id")))
        .iter_rows()
    ):
        keys = []
        for row in context_rows:
            layer = int(row["layer"])
            if row["fixed_destination"] is not None:
                keys.append(None)
            elif row["anchor_id"] is not None:
                keys.append(("a", int(row["anchor_id"])))
            else:
                keys.append(("l", layer))
        variable_keys[int(context_id)] = keys
    shortlist = (
        exact_located.filter("variable")
        .with_columns(
            inbound_static_32=(
                (pl.col("inbound_time_rank") <= 32)
                | (pl.col("inbound_cost_rank") <= 32)
                | (pl.col("capacity_rank") <= 32)
                | (pl.col("shadow_rank") <= 32)
            ),
            home_static_32=(
                (pl.col("home_time_rank") <= 32)
                | (pl.col("home_cost_rank") <= 32)
                | (pl.col("capacity_rank") <= 32)
                | (pl.col("shadow_rank") <= 32)
            ),
        )
        .group_by("context_id")
        .agg(
            inbound_static_32=pl.col("inbound_static_32").mean(),
            home_static_32=pl.col("home_static_32").mean(),
            max_inbound_time_rank=pl.col("inbound_time_rank").max(),
            max_home_time_rank=pl.col("home_time_rank").max(),
        )
    )
    shortlist_by_context = {
        int(row["context_id"]): row for row in shortlist.iter_rows(named=True)
    }
    rows: list[dict[str, float | int | str]] = []
    for context_id, reference in exact_plans.items():
        maximum = max(score for _, score in reference)
        weights = {
            zones: math.exp(score - maximum) for zones, score in reference
        }
        normalizer = sum(weights.values())
        found = {
            zones for zones, _ in bounded_plans.get(context_id, [])
        }
        domain_by_variable: defaultdict[tuple[str, int], set[int]] = defaultdict(set)
        for layer, variable_key in enumerate(variable_keys[context_id]):
            if variable_key is not None:
                domain_by_variable[variable_key].update(
                    bounded_support.get((context_id, layer), set())
                )
        lattice_assignments = math.prod(
            len(domain) for domain in domain_by_variable.values()
        )

        def in_candidate_lattice(zones: tuple[int, ...]) -> bool:
            return all(
                variable_key is None
                or zones[layer] in domain_by_variable[variable_key]
                for layer, variable_key in enumerate(variable_keys[context_id])
            )

        lattice_mass = sum(
            weight
            for zones, weight in weights.items()
            if in_candidate_lattice(zones)
        ) / normalizer
        transition_mass = {
            cutoff: sum(
                weight
                for zones, weight in weights.items()
                if zones
                in transition_lattices[cutoff].get(context_id, set())
            )
            / normalizer
            for cutoff in transition_lattices
        }
        production_mass = {
            cutoff: sum(
                weight
                for zones, weight in weights.items()
                if zones
                in production_lattices[cutoff].get(context_id, set())
            )
            / normalizer
            for cutoff in production_lattices
        }
        mass = sum(
            weight for zones, weight in weights.items() if zones in found
        ) / normalizer
        shape = shape_by_context[context_id]
        support = shortlist_by_context.get(context_id, {})
        rows.append(
            {
                "context_id": context_id,
                "mass_at_10": mass,
                "lattice_mass_at_10": lattice_mass,
                "lattice_top1_hit": int(
                    in_candidate_lattice(reference[0][0])
                ),
                "lattice_assignments": lattice_assignments,
                "proposed_support_mass": traced_support_mass.get(
                    context_id, (0.0, 0.0)
                )[0],
                "retained_support_mass": traced_support_mass.get(
                    context_id, (0.0, 0.0)
                )[1],
                **{
                    f"transition_mass_{cutoff}": mass
                    for cutoff, mass in transition_mass.items()
                },
                **{
                    f"production_mass_{cutoff}": mass
                    for cutoff, mass in production_mass.items()
                },
                "top1_hit": int(reference[0][0] in found),
                "layers": shape.layers,
                "variables": shape.variables,
                "width": shape.min_fill_width,
                "anchors": shape.anchor_variables,
                "cross_home": shape.cross_home_anchor_variables,
                "longest_tour": shape.longest_tour,
                "inbound_static_32": float(
                    support.get("inbound_static_32", 1.0)
                ),
                "home_static_32": float(
                    support.get("home_static_32", 1.0)
                ),
                "max_inbound_time_rank": int(
                    support.get("max_inbound_time_rank", 0)
                ),
                "max_home_time_rank": int(
                    support.get("max_home_time_rank", 0)
                ),
            }
        )
    quality = pl.DataFrame(rows).with_columns(
        outcome=(
            pl.when(pl.col("mass_at_10") >= 1.0 - 1e-12)
            .then(pl.lit("full"))
            .when(pl.col("mass_at_10") > 0.0)
            .then(pl.lit("partial"))
            .otherwise(pl.lit("zero"))
        )
    )
    print("\nBOUNDED VERSUS CACHED EXACT TOP-10")
    print(
        f"contexts={quality.height} wall={elapsed:.3f}s "
        f"mean-mass={quality['mass_at_10'].mean():.3f} "
        f"top1-hit={quality['top1_hit'].mean():.1%} "
        f"infeasible={report['infeasible_contexts']}"
    )
    print(
        "Candidate-lattice closure upper bound from returned per-variable zones: "
        f"mean-mass={quality['lattice_mass_at_10'].mean():.3f} "
        f"top1-hit={quality['lattice_top1_hit'].mean():.1%} "
        f"full-contexts={(quality['lattice_mass_at_10'] >= 1.0 - 1e-12).mean():.1%}; "
        f"assignment lattice {describe(quality['lattice_assignments'].to_list())}"
    )
    print(
        "Per-layer closure upper bound from internal search support: "
        f"proposed-mass={quality['proposed_support_mass'].mean():.3f} "
        f"retained-mass={quality['retained_support_mass'].mean():.3f}"
    )
    print(
        "Exact top-10 mass whose every variable transition is in the "
        "inbound-near-or-static input lattice: "
        + " ".join(
            f"N={cutoff}:{quality[f'transition_mass_{cutoff}'].mean():.3f}"
            for cutoff in transition_lattices
        )
    )
    print(
        "Exact top-10 mass covered by the production heuristic base lattice "
        "(inbound-cost or capacity+shadow rank): "
        + " ".join(
            f"N={cutoff}:{quality[f'production_mass_{cutoff}'].mean():.3f}"
            for cutoff in production_lattices
        )
    )
    outcomes = (
        quality.group_by("outcome")
        .agg(
            contexts=pl.len(),
            mass=pl.col("mass_at_10").mean(),
            layers=pl.col("layers").mean(),
            variables=pl.col("variables").mean(),
            width=pl.col("width").mean(),
            anchors=pl.col("anchors").mean(),
            cross_home=pl.col("cross_home").mean(),
            longest_tour=pl.col("longest_tour").mean(),
            inbound_static_32=pl.col("inbound_static_32").mean(),
            home_static_32=pl.col("home_static_32").mean(),
            lattice_mass=pl.col("lattice_mass_at_10").mean(),
            transition_mass_32=pl.col("transition_mass_32").mean(),
            transition_mass_64=pl.col("transition_mass_64").mean(),
            proposed_support_mass=pl.col("proposed_support_mass").mean(),
            retained_support_mass=pl.col("retained_support_mass").mean(),
        )
        .sort("mass")
    )
    for row in outcomes.iter_rows(named=True):
        print(
            f"  {row['outcome']:7s} n={row['contexts']:3d} "
            f"mass={row['mass']:.3f} layers={row['layers']:.2f} "
            f"variables={row['variables']:.2f} width={row['width']:.2f} "
            f"anchors={row['anchors']:.2f} cross-home={row['cross_home']:.2f} "
            f"longest-tour={row['longest_tour']:.2f} "
            f"lattice-mass={row['lattice_mass']:.3f} "
            f"transition-mass32/64={row['transition_mass_32']:.3f}/"
            f"{row['transition_mass_64']:.3f} "
            f"internal(proposed/retained)="
            f"{row['proposed_support_mass']:.3f}/"
            f"{row['retained_support_mass']:.3f} "
            f"shortlist32(inbound/home)="
            f"{row['inbound_static_32']:.1%}/{row['home_static_32']:.1%}"
        )
    print("\nQuality by structural width")
    print(
        quality.group_by("width")
        .agg(
            contexts=pl.len(),
            mass=pl.col("mass_at_10").mean(),
            lattice_mass=pl.col("lattice_mass_at_10").mean(),
            top1_hit=pl.col("top1_hit").mean(),
            zero=(pl.col("mass_at_10") == 0).mean(),
            layers=pl.col("layers").mean(),
            longest_tour=pl.col("longest_tour").mean(),
        )
        .sort("width")
    )
    print("\nQuality by longest home-bounded tour")
    print(
        quality.group_by("longest_tour")
        .agg(
            contexts=pl.len(),
            mass=pl.col("mass_at_10").mean(),
            lattice_mass=pl.col("lattice_mass_at_10").mean(),
            top1_hit=pl.col("top1_hit").mean(),
            zero=(pl.col("mass_at_10") == 0).mean(),
            width=pl.col("width").mean(),
        )
        .sort("longest_tour")
    )
    print("\nLowest-mass exact-certified contexts")
    print(
        quality.sort(
            ["mass_at_10", "layers"],
            descending=[False, True],
        )
        .head(15)
        .select(
            "context_id",
            "mass_at_10",
            "lattice_mass_at_10",
            "proposed_support_mass",
            "retained_support_mass",
            "top1_hit",
            "layers",
            "variables",
            "width",
            "anchors",
            "cross_home",
            "longest_tour",
            "inbound_static_32",
            "home_static_32",
            "max_inbound_time_rank",
            "max_home_time_rank",
        )
    )


def trace_internal_support_mass(
    search: DestinationPlanSearch,
    exact_plans: dict[int, list[tuple[tuple[int, ...], float]]],
    steps: pl.DataFrame,
    initial_locations: pl.DataFrame,
    args: argparse.Namespace,
) -> dict[int, tuple[float, float]]:
    step_by_context = {
        int(context_id): frame
        for (context_id,), frame in steps.partition_by(
            "context_id", as_dict=True
        ).items()
    }
    initial_by_context = {
        int(context_id): frame
        for (context_id,), frame in initial_locations.partition_by(
            "context_id", as_dict=True
        ).items()
    }
    result: dict[int, tuple[float, float]] = {}
    started = time.perf_counter()
    for context_id, reference in exact_plans.items():
        context_steps = step_by_context[context_id]
        variable_layers = {
            int(layer)
            for layer, fixed in context_steps.select(
                "layer", "fixed_destination"
            ).iter_rows()
            if fixed is None
        }
        maximum = max(score for _, score in reference)
        weights = [
            math.exp(score - maximum) for _, score in reference
        ]
        normalizer = sum(weights)
        if context_steps.height == 2:
            # The dedicated two-step path scans the complete activity domain
            # and does not emit beam trace events.
            result[context_id] = (1.0, 1.0)
            continue
        try:
            _, report = search.top_k(
                steps=context_steps,
                initial_locations=initial_by_context[context_id],
                logit_scale=LOGIT_SCALE,
                update_plan_timings=True,
                use_shadow_prices=True,
                exploration_seed=args.exploration_seed,
                **ACTIVE_TOP_K_DEFAULTS,
                top_k=10,
                n_threads=1,
                skip_infeasible=False,
                active_trace_context_id=context_id,
                active_trace_target_plans=[
                    list(zones) for zones, _ in reference
                ],
            )
        except ValueError:
            result[context_id] = (0.0, 0.0)
            continue
        proposed_mass = 0.0
        retained_mass = 0.0
        for weight, trace in zip(
            weights, report["active_trace_targets"], strict=True
        ):
            if all(
                trace["proposed"][layer] for layer in variable_layers
            ):
                proposed_mass += weight
            if all(
                trace["retained"][layer] for layer in variable_layers
            ):
                retained_mass += weight
        result[context_id] = (
            proposed_mass / normalizer,
            retained_mass / normalizer,
        )
    print(
        f"Internal support trace: contexts={len(result)} "
        f"wall={time.perf_counter() - started:.3f}s"
    )
    return result


def analyze_outputs(
    od_costs: pl.DataFrame,
    destination_inputs: pl.DataFrame,
    steps: pl.DataFrame,
    initial_locations: pl.DataFrame,
    args: argparse.Namespace,
) -> None:
    if args.output_contexts <= 0:
        return
    selected = calibrated_contexts(
        steps, args.output_contexts, args.exploration_seed
    )
    sample_steps = steps.join(selected, on="context_id", how="semi")
    sample_initial = initial_locations.join(
        selected, on="context_id", how="semi"
    )
    search = DestinationPlanSearch(
        od_costs=od_costs, destination_inputs=destination_inputs
    )
    options = argparse.Namespace(**ACTIVE_TOP_K_DEFAULTS)
    started = time.perf_counter()
    returned, report = search.top_k(
        steps=sample_steps,
        initial_locations=sample_initial,
        logit_scale=LOGIT_SCALE,
        update_plan_timings=True,
        use_shadow_prices=True,
        exploration_seed=args.exploration_seed,
        **top_k_tuning_options(options),
        top_k=args.top_k,
        n_threads=args.threads,
        skip_infeasible=True,
    )
    elapsed = time.perf_counter() - started
    print("\nLOCATED BOUNDED OUTPUT SEARCH")
    print(
        f"sampled-contexts={args.output_contexts} returned-contexts="
        f"{returned['context_id'].n_unique()} rows={returned.height} "
        f"top-k={args.top_k} wall={elapsed:.3f}s "
        f"infeasible={report['infeasible_contexts']}"
    )

    od_ranks = ranked_od(od_costs)
    located = locate_rows(
        returned,
        sample_steps,
        sample_initial,
        od_ranks,
        destination_inputs,
    )
    variable = located.filter("variable")
    print_location_summary("CALIBRATED BOUNDED TOP-K LOCATION", located, args.top_k)

    home_activity = (
        variable.group_by(["initial_zone", "activity_id"])
        .agg(
            contexts=pl.col("context_id").n_unique(),
            rows=pl.len(),
            zones=pl.col("destination").n_unique(),
        )
    )
    reusable = home_activity.filter(pl.col("contexts") >= 2)
    print(
        "\nHome/activity reuse in sample: "
        f"groups={home_activity.height} repeated-groups={reusable.height} "
        f"({reusable.height / home_activity.height:.1%}); "
        f"contexts/group {describe(home_activity['contexts'].to_list())}; "
        f"located-zones/group {describe(home_activity['zones'].to_list())}"
    )


def main() -> None:
    args = parse_args()
    if args.output_contexts < 0 or args.top_k <= 0 or args.threads <= 0:
        raise ValueError("output-contexts must be nonnegative; top-k/threads positive")
    files = resolve_snapshot_files(args.group_day_trips_folder)
    print("Preparing cached Grand Geneve inputs (read-only)...")
    od_costs = prepare_od_costs(
        files["transport_costs"], files["demand_groups"]
    )
    destination_inputs = prepare_destination_inputs(
        files["destination_saturation"], files["demand_groups"]
    )
    steps, initial_locations, raw_context_count = prepare_complete_contexts(
        activity_sequences_path=files["activity_sequences"],
        survey_plan_steps_path=files["survey_plan_steps"],
        demand_groups_path=files["demand_groups"],
        activity_dur_path=files["activity_dur"],
    )
    shapes = analyze_inputs(
        od_costs,
        destination_inputs,
        steps,
        initial_locations,
        raw_context_count,
    )
    analyze_cached_exact_outputs(
        files,
        od_costs,
        destination_inputs,
        steps,
        initial_locations,
        shapes,
        args,
    )
    analyze_outputs(
        od_costs, destination_inputs, steps, initial_locations, args
    )


if __name__ == "__main__":
    main()
